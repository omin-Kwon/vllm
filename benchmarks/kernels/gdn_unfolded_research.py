# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Research-only equivalent representation: raw Z_eta, explicit inverse, Z factors.

Derives isolated variants from the trusted experimental CUDA sources; does not
alter their default dispatch. One inverse application follows projected WY.
"""

import functools
import types
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as flush
from vllm.third_party.flash_linear_attention.ops import gdn_step_full_cuda as step

ROOT = Path("/disk2/omin/.cache/gdn_solver_research")


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f"Source changed: expected {count} matches for {old[:70]!r}")
    return text.replace(old, new)


@functools.lru_cache(None)
def step_module():
    source = Path(step.__file__).read_text()
    source = replace(
        source,
        "float* __restrict__ beta_ring,\n",
        "float* __restrict__ beta_ring, const float* inverse,\n",
    )
    source = replace(
        source,
        "long s_u_slot, long s_phi_slot, long s_fs_slot,",
        "long s_u_slot, long s_phi_slot, long s_fs_slot, long s_inv_slot,",
    )
    source = replace(
        source,
        "c10::optional<torch::Tensor> beta_ring, double scale",
        "c10::optional<torch::Tensor> beta_ring, torch::Tensor inverse, double scale",
        count=3,
    )
    source = replace(
        source,
        "beta_ring.has_value() ? beta_ring->data_ptr<float>() : nullptr,",
        "beta_ring.has_value() ? beta_ring->data_ptr<float>() : nullptr, inverse.data_ptr<float>(),",
    )
    source = replace(
        source,
        "phi.stride(0),factors.stride(0),H,HV,W,G);",
        "phi.stride(0),factors.stride(0),inverse.stride(0),H,HV,W,G);",
    )
    source = replace(
        source, "mapping,beta_ring,scale)", "mapping,beta_ring,inverse,scale)", count=2
    )
    marker = "        #pragma unroll 1\n        for (int c = c_u; c < NC; ++c) {"
    conversion = """        if (mh < K) {
            const float* im = inverse + cidx * s_inv_slot + (long)i_hv * G * G;
            #pragma unroll
            for (int i = 0; i < NG; ++i) {
                const int row = lane + 32 * i;
                float value = 0.f;
                if (row < mh) {
                    for (int j = 0; j < mh; ++j)
                        value = fmaf(im[j * G + row], sA[warp][j], value);
                    sA[warp][GT + row] = value;
                }
            }
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < NG; ++i) {
                const int row = lane + 32 * i;
                if (row < mh) sA[warp][row] = sA[warp][GT + row];
            }
            __syncwarp();
        }
"""
    source = replace(source, marker, conversion + marker)
    source = replace(
        source, "    beta_ring=None,\n", "    beta_ring=None,\n    inverse=None,\n"
    )
    source = replace(
        source,
        "            beta_ring,\n            float(scale),",
        "            beta_ring,\n            inverse,\n            float(scale),",
    )
    source = replace(source, 'name="gdn_full_cuda_v1"', 'name="gdn_unfolded_step_v1"')
    source = replace(
        source,
        '"NS_GDN_FULL_BUILD_DIR", "/disk2/omin/.cache/gdn_full_cuda"',
        '"NS_GDN_UNFOLDED_BUILD_DIR", "/disk2/omin/.cache/gdn_solver_research/unfolded_step"',
    )
    module = types.ModuleType("gdn_unfolded_step")
    module.__file__ = str(ROOT / "step_unfolded_generated.py")
    Path(module.__file__).write_text(source)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@functools.lru_cache(None)
def inverse_extension():
    source = flush._SRC[: flush._SRC.index("#define NT 256")]
    source = replace(
        source,
        "long s_phi_slot, int HV, int G)",
        "long s_phi_slot, float* inverse, long s_inv_slot, int HV, int G)",
    )
    source = replace(source, "const int tail = K - m;", "const int tail = m;")
    source = replace(
        source,
        "sX[i * NCS + c] = sc[(m + c) * G + i];",
        "sX[i * NCS + c] = float(i == c);",
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
        + """
    float* inv = inverse + cidx * s_inv_slot + (long)hv * G * G;
    for (int o = t; o < G * G; o += NTB) {
        const int i = o % G, j = o / G;
        inv[o] = i < m && j < m ? sX[i * NCS + j] : 0.f;
    }
""",
    )
    declaration = """void run_inverse(torch::Tensor scratch, torch::Tensor phi, torch::Tensor inverse,
    torch::Tensor rows, torch::Tensor mapping, torch::Tensor widths, int max_rows)"""
    source += (
        declaration
        + """ {
    const int HV = phi.size(1), G = phi.size(2);
    if (max_rows == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    #define INV(M) do { \\
        constexpr int smem = (M * (M + 1) + M * (M | 1)) * 4; \\
        static bool init = false; \\
        if (!init) { C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_ls6_solve_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); init = true; } \\
        gdn_ls6_solve_kernel<M><<<dim3(max_rows, HV), 128, smem, stream>>>(rows.data_ptr<int>(), rows.data_ptr<int>() + max_rows, mapping.data_ptr<int>(), widths.data_ptr<int>(), nullptr, scratch.data_ptr<float>(), phi.data_ptr<float>(), phi.stride(0), inverse.data_ptr<float>(), inverse.stride(0), HV, G); \\
    } while (0)
    INV(8);
    if (G > 8) { INV(16); }
    if (G > 16) { INV(32); }
    if (G > 32) { INV(48); }
    if (G > 48) { INV(64); }
    if (G > 64) { INV(128); }
    #undef INV
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    )
    build = ROOT / "inverse"
    build.mkdir(exist_ok=True)
    return load_inline(
        name="gdn_explicit_inverse_v1",
        cpp_sources="#include <torch/extension.h>\n" + declaration + ";",
        cuda_sources=source,
        functions=["run_inverse"],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
    )


class UnfoldedWorkspace(flush.FlushWorkspace):
    """Extra m-by-m inverse cache; raw-WY factors live in Z coordinates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = None
        self.inverse_ext = inverse_extension()
        self.step = step_module().step

    def _run(self, phase, *args):
        if self.inverse is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Call refresh before graph capture")
            phi = args[-1]
            self.inverse = torch.full(
                (*phi.shape[:2], self.g, self.g), float("nan"), device=phi.device
            )
        if phase == 0:
            super()._run(1, *args)
            super()._run(2, *args)
        elif phase == 4:
            # Initial setup: reuse the existing Gram refresh; its coefficient
            # solve is overwritten below and excluded from steady-state timing.
            super()._run(4, *args)
        elif phase != 3:
            return super()._run(phase, *args)
        self.inverse_ext.run_inverse(
            self.scratch,
            args[-1],
            self.inverse,
            args[4],
            args[5],
            args[6],
            self.max_rows,
        )
