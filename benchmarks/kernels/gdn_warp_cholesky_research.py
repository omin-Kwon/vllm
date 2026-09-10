# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Research: factor small Gram matrices in one warp, keeping the current format."""

import functools

from gdn_unfolded_research import ROOT, replace
from torch.utils.cpp_extension import load_inline

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as flush


def warp_factor_source(source):
    start = source.index("    // Cholesky, right-looking, width-four panels;")
    end = source.index("    // Small widths solve all RHS together;", start)
    original = source[start:end]
    replacement = (
        r"""
    if constexpr (M <= 32) {
        if (warp == 0) {
            // Lane owns a row. Static column indices keep the matrix in registers.
            float a[M];
            #pragma unroll
            for (int j = 0; j < M; ++j)
                a[j] = lane < m && j < m ? sM[lane * GS + j] : float(lane == j);
            #pragma unroll
            for (int p = 0; p < M; ++p) {
                if (p < m) {
                    const float diagonal = __shfl_sync(FULL, a[p], p);
                    const float rd = rsqrtf(fmaxf(diagonal, 1e-30f));
                    if (lane == p) s_rd[p] = rd;
                    const float l = lane > p && lane < m ? a[p] * rd : 0.f;
                    if (lane > p) a[p] = l;
                    #pragma unroll
                    for (int j = p + 1; j < M; ++j) {
                        const float lj = __shfl_sync(FULL, l, j);
                        if (lane >= j && lane < m) a[j] = fmaf(-l, lj, a[j]);
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < M; ++j)
                if (lane < m && j < lane) sM[lane * GS + j] = a[j];
        }
        __syncthreads();
    } else {
"""
        + original
        + "    }\n"
    )
    return source[:start] + replacement + source[end:]


@functools.lru_cache(None)
def extension(coalesced=False):
    source = flush._SRC[: flush._SRC.index("#define NT 256")]
    if coalesced:
        source = replace(
            source,
            "const int i = o / ncol, c = o % ncol;",
            "const int i = o % m, c = o / m;",
        )
    else:
        source = warp_factor_source(source)
    declaration = """void run_solve(torch::Tensor scratch, torch::Tensor phi,
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
    #define SOLVE(M, B) do { \
        const int count = planned ? offsets[B + 1] - offsets[B] : HV; \
        const int* indices = planned && count ? heads.data_ptr<int>() + offsets[B] : nullptr; \
        constexpr int LOWER = M == 8 ? 0 : M == 48 ? 32 : M == 64 ? 48 : M / 2; \
        const int stride = M == 128 ? G : M; \
        const int smem = (stride * (stride + 1) + stride * ((128 - LOWER) | 1)) * 4; \
        if (count) { \
            if (attr[B] < smem) { C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_ls6_solve_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr[B] = smem; } \
            gdn_ls6_solve_kernel<M><<<dim3(max_rows, count), 128, smem, stream>>>(rows.data_ptr<int>(), rows.data_ptr<int>() + max_rows, mapping.data_ptr<int>(), widths.data_ptr<int>(), indices, scratch.data_ptr<float>(), phi.data_ptr<float>(), phi.stride(0), HV, G); \
        } \
    } while (0)
    SOLVE(8, 0);
    if (G > 8) { SOLVE(16, 1); }
    if (G > 16) { SOLVE(32, 2); }
    if (G > 32) { SOLVE(48, 3); }
    if (G > 48) { SOLVE(64, 4); }
    if (G > 64) { SOLVE(128, 5); }
    #undef SOLVE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    )
    build = ROOT / ("coalesced_solve" if coalesced else "warp_cholesky")
    build.mkdir(exist_ok=True)
    (build / "candidate.cu").write_text(source)
    return load_inline(
        name="gdn_coalesced_solve_v1" if coalesced else "gdn_warp_cholesky_v1",
        cpp_sources="#include <torch/extension.h>\n" + declaration + ";",
        cuda_sources=source,
        functions=["run_solve"],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
    )


class WarpCholeskyWorkspace(flush.FlushWorkspace):
    coalesced = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solve_ext = extension(self.coalesced)

    def _run(self, phase, *args):
        if phase == 0:
            super()._run(1, *args)
            super()._run(2, *args)
        elif phase == 4:
            super()._run(4, *args)
        elif phase != 3:
            return super()._run(phase, *args)
        self.solve_ext.run_solve(
            self.scratch,
            args[-1],
            args[4],
            args[5],
            args[6],
            self.max_rows,
            self._solve_heads,
            self._solve_offsets,
        )


class CoalescedSolveWorkspace(WarpCholeskyWorkspace):
    """Unchanged Cholesky and RHS arithmetic; adjacent lanes load adjacent rows."""

    coalesced = True
