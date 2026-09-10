# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Keep raw ridge Gram/WY factors; solve only the final query vector per token.

Research only. Exact state folding and Cholesky arithmetic come from the
validated full-coordinate implementation. No inverse matrix is constructed.
"""

import functools
import types
from pathlib import Path

import torch
from gdn_unfolded_research import ROOT, replace
from gdn_unfolded_research import step_module as inverse_step_module
from torch.utils.cpp_extension import load_inline

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as flush


@functools.lru_cache(None)
def step_module():
    # Reuse the established raw-Z/WY coordinate transformation, replacing its
    # explicit inverse matvec with two warp-distributed triangular substitutions.
    inverse_step_module()
    source = (ROOT / "step_unfolded_generated.py").read_text()
    start = source.index("        if (mh < K) {\n            const float* im = inverse")
    end = source.index(
        "        #pragma unroll 1\n        for (int c = c_u; c < NC; ++c)", start
    )
    conversion = r"""        if (mh > 0 && mh < K) {
            mbar_wait(&factor_bar[warp], 0);
            const float* lm = sFactor;
            float x[NG];
            #pragma unroll
            for (int q = 0; q < NG; ++q) {
                const int row = lane + 32 * q;
                x[q] = row < mh ? sA[warp][row] : 0.f;
            }
            for (int k = 0; k < mh; ++k) {
                float value = x[0];
                #pragma unroll
                for (int q = 1; q < NG; ++q) if (q == k / 32) value = x[q];
                const float pivot = __shfl_sync(FULL, value, k & 31) * lm[k * G + k];
                #pragma unroll
                for (int q = 0; q < NG; ++q) {
                    const int row = lane + 32 * q;
                    if (row > k && row < mh) x[q] = fmaf(-lm[k * G + row], pivot, x[q]);
                    else if (row == k) x[q] = pivot;
                }
            }
            for (int k = mh - 1; k >= 0; --k) {
                float value = x[0];
                #pragma unroll
                for (int q = 1; q < NG; ++q) if (q == k / 32) value = x[q];
                const float pivot = __shfl_sync(FULL, value, k & 31) * lm[k * G + k];
                #pragma unroll
                for (int q = 0; q < NG; ++q) {
                    const int row = lane + 32 * q;
                    if (row < k) x[q] = fmaf(-lm[row * G + k], pivot, x[q]);
                    else if (row == k) x[q] = pivot;
                }
            }
            #pragma unroll
            for (int q = 0; q < NG; ++q) {
                const int row = lane + 32 * q;
                if (row < mh) sA[warp][row] = x[q];
            }
            __syncwarp();
        }
"""
    source = source[:start] + conversion + source[end:]
    source = replace(
        source,
        "__shared__ __align__(8) unsigned long long mbar_st[NW][2];",
        "__shared__ __align__(8) unsigned long long mbar_st[NW][2], factor_bar[NW];",
    )
    source = replace(
        source,
        "const int mh = mh_raw;",
        r"""const int mh = mh_raw;
    float* sFactor = (float*)(dsm + SM::BYTES) + warp * GT * GT;
    if (mh > 0 && mh < K) {
        if (lane == 0) {
            mbar_init(&factor_bar[warp], 1);
            mbar_expect_tx(&factor_bar[warp], G * G * 4);
            bulk_g2s(sFactor, inverse + cidx * s_inv_slot + (long)i_hv * G * G,
                     G * G * 4, &factor_bar[warp]);
        }
        __syncwarp();
    }""",
    )
    source = replace(
        source,
        "constexpr int SMEM=Smem<K,V,HPG>::BYTES;",
        "constexpr int SMEM=Smem<K,V,HPG>::BYTES + HPG * GT * GT * 4;",
    )
    source = source.replace("inverse", "cholesky")
    source = replace(
        source, 'name="gdn_unfolded_step_v1"', 'name="gdn_deferred_step_v2"'
    )
    source = source.replace("NS_GDN_UNFOLDED_BUILD_DIR", "NS_GDN_DEFERRED_BUILD_DIR")
    source = source.replace(
        "gdn_solver_research/unfolded_step", "gdn_solver_research/deferred_step_smem"
    )
    module = types.ModuleType("gdn_deferred_step")
    module.__file__ = str(ROOT / "step_deferred_generated.py")
    Path(module.__file__).write_text(source)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@functools.lru_cache(None)
def factor_extension():
    source = flush._SRC[: flush._SRC.index("#define NT 256")]
    source = replace(
        source,
        "long s_phi_slot, int HV, int G)",
        "long s_phi_slot, float* factor, long s_factor_slot, int HV, int G)",
    )
    source = replace(
        source,
        "if constexpr (M <= 64) solve_rhs_registers<M>(sM, sX, s_rd, m, ncol, GS, NCS);\n"
        "    else ls6_solve_cols<1, 4>(sM, sX, s_rd, m, ncol, GS, NCS, lane, warp);",
        "// No coefficient RHS solves: keep the Cholesky factor for the next window.",
    )
    source = replace(
        source,
        "if (g < m) v = (tt < m) ? ((tt == g) ? 1.f : 0.f) : sX[g * NCS + (tt - m)];",
        "if (g < m) v = sc[tt * G + g];",
    )
    marker = "            pphi[g * K + tt] = v;\n        }"
    source = replace(
        source,
        marker,
        marker
        + r"""
    float* lf = factor + cidx * s_factor_slot + (long)hv * G * G;
    for (int o = t; o < G * G; o += NTB) {
        const int i = o % G, j = o / G;
        lf[o] = i < m && j < m ? (i == j ? s_rd[i] : i > j ? sM[i * GS + j] : 0.f) : 0.f;
    }
""",
    )
    declaration = """void run_factor(torch::Tensor scratch, torch::Tensor phi, torch::Tensor factor,
    torch::Tensor rows, torch::Tensor mapping, torch::Tensor widths, int max_rows,
    torch::Tensor heads, std::vector<int64_t> offsets)"""
    source += (
        declaration
        + r""" {
    const int HV = phi.size(1), G = phi.size(2);
    if (max_rows == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool planned = !offsets.empty();
    static std::unordered_map<int, std::array<int, 6>> attributes;
    auto& attr = attributes[phi.get_device()];
    #define FACTOR(M, B) do { \
        const int count = planned ? offsets[B + 1] - offsets[B] : HV; \
        const int* indices = planned && count ? heads.data_ptr<int>() + offsets[B] : nullptr; \
        const int smem = (M == 128 ? G * (G + 1) : M * (M + 1)) * 4; \
        if (count) { \
            if (attr[B] < smem) { C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_ls6_solve_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr[B] = smem; } \
            gdn_ls6_solve_kernel<M><<<dim3(max_rows, count), 128, smem, stream>>>(rows.data_ptr<int>(), rows.data_ptr<int>() + max_rows, mapping.data_ptr<int>(), widths.data_ptr<int>(), indices, scratch.data_ptr<float>(), phi.data_ptr<float>(), phi.stride(0), factor.data_ptr<float>(), factor.stride(0), HV, G); \
        } \
    } while (0)
    FACTOR(8, 0);
    if (G > 8) { FACTOR(16, 1); }
    if (G > 16) { FACTOR(32, 2); }
    if (G > 32) { FACTOR(48, 3); }
    if (G > 48) { FACTOR(64, 4); }
    if (G > 64) { FACTOR(128, 5); }
    #undef FACTOR
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    )
    # The unused RHS load would access a removed shared-memory buffer.
    source = replace(
        source,
        "    for (int o = t; o < m * ncol; o += NTB) {\n"
        "        const int i = o / ncol, c = o % ncol;\n"
        "        sX[i * NCS + c] = sc[(m + c) * G + i];\n"
        "    }\n",
        "",
    )
    build = ROOT / "factor_only"
    build.mkdir(exist_ok=True)
    return load_inline(
        name="gdn_deferred_factor_v1",
        cpp_sources="#include <torch/extension.h>\n" + declaration + ";",
        cuda_sources=source,
        functions=["run_factor"],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
    )


class DeferredWorkspace(flush.FlushWorkspace):
    """Raw Z_eta and Z-WY factors, plus a Cholesky cache with reciprocal diagonal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = None
        self.factor_ext = factor_extension()
        self.step = step_module().step

    def _run(self, phase, *args):
        if self.factor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Call refresh before graph capture")
            phi = args[-1]
            self.factor = torch.full(
                (*phi.shape[:2], self.g, self.g),
                float("nan"),
                dtype=torch.float32,
                device=phi.device,
            )
        if phase == 0:
            super()._run(1, *args)
            super()._run(2, *args)
        elif phase == 4:
            super()._run(4, *args)
        elif phase != 3:
            return super()._run(phase, *args)
        self.factor_ext.run_factor(
            self.scratch,
            args[-1],
            self.factor,
            args[4],
            args[5],
            args[6],
            self.max_rows,
            self._solve_heads,
            self._solve_offsets,
        )
