# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research fixed-metric reads: exact raw-WY fold, U extraction, no online solve.

This changes the read approximation. The full state and its updates retain the
current GDN contract. Coefficients are constructed offline and shared by slots.
"""

import functools

import torch
from gdn_unfolded_research import ROOT, replace
from torch.utils.cpp_extension import load_inline

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as flush


@functools.lru_cache(None)
def fold_extension():
    source = flush._SRC
    start = source.index("#define NTB 128")
    end = source.index("#include <cuda.h>", start)
    source = (
        source[:start]
        + r"""
#define NT 256
template <int K, int V>
__global__ void gdn_fixed_u_kernel(
    const float* state, const int* rows, const int* count, const int* mapping,
    const int* widths, float* u, long ss, long sh, long su, int HV, int G)
{
    const int row = blockIdx.x, head = blockIdx.y;
    if (row >= *count) return;
    const int slot = rows[row], m = widths[head];
    if (slot <= 0 || m <= 0) return;
    const int compact = mapping[slot];
    const float* src = state + (long)slot * ss + head * sh;
    float* dst = u + (long)compact * su + (long)head * G * V;
    for (int o = threadIdx.x; o < G * V; o += NT) {
        const int g = o / V, v = o % V;
        dst[o] = g < m ? src[v * K + g] : 0.f;
    }
}

"""
        + source[end:]
    )
    # Keep the verified hi/lo Tensor Core state fold. Compile out all Gram work.
    source = replace(
        source, "const bool do_phi = exact && m < SK;", "const bool do_phi = false;"
    )
    # Gram helpers are dead, but reserve no Gram staging memory in the stream.
    source = replace(source, "#define SPHI_MAX 2048", "#define SPHI_MAX 0")
    source = replace(
        source,
        "    TORCH_CHECK(scratch.numel() >= (long)max_rows * HV * SK * G, "
        '"full flush: scratch size");\n',
        "",
    )
    # Remove solve-only attribute setup and all solve launches.
    start = source.index("    if (phase == 4) {", source.index("void run("))
    end = source.index("    C10_CUDA_KERNEL_LAUNCH_CHECK();", start)
    source = (
        source[:start]
        + r"""
    if (phase == 4) {
        gdn_fixed_u_kernel<128,128><<<dim3(max_rows, HV), NT, 0, st>>>(
            h0.data_ptr<float>(), fl, fl + n_off, lm, widths.data_ptr<int>(),
            u.data_ptr<float>(), h0.stride(0), h0.stride(1), u.stride(0), HV, G);
    }
"""
        + source[end:]
    )
    # The host used to set a dynamic-memory attribute on the large solve kernel.
    start = source.index("    if (G > 64 && attr[1] < smemB128)")
    end = source.index("    const int* fl = rows.data_ptr<int>();", start)
    source = source[:start] + source[end:]
    build = ROOT / "fixed_metric_fold"
    build.mkdir(exist_ok=True)
    (build / "candidate.cu").write_text(source)
    return load_inline(
        name="gdn_fixed_metric_fold_v1",
        cpp_sources=flush._CPP,
        cuda_sources=source,
        functions=["run"],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
    )


def coefficients_from_metric(metric, widths, g):
    """Offline P0=[I, solve(E0[:m,:m], E0[:m,m:])] in embedded coordinates.

    This setup operation may synchronize and must run before CUDA graph capture.
    The caller supplies the calibration metric in the runtime's rotated basis.
    """
    if metric.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Construct fixed coefficients before graph capture")
    if metric.ndim != 3 or metric.shape[1:] != (128, 128):
        raise ValueError("Metric must have shape (HV,128,128)")
    values = widths.tolist()
    if len(values) != metric.shape[0] or any(m < 0 or m > g for m in values):
        raise ValueError("Invalid fixed widths")
    if not 4 <= g <= 128 or g % 4:
        raise ValueError("G must be a multiple of four in [4,128]")
    coeff = torch.full(
        (len(values), g, 128), float("nan"), device=metric.device, dtype=torch.float32
    )
    for head, m in enumerate(values):
        if 0 < m < 128:
            coeff[head].zero_()
            coeff[head, :m, :m] = torch.eye(m, device=metric.device)
            coeff[head, :m, m:] = torch.linalg.solve(
                metric[head, :m, :m], metric[head, :m, m:]
            ).float()
    return coeff


class FixedMetricWorkspace(flush.FlushWorkspace):
    """Exact state fold with only U refreshed; fixed P0 is never written here."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ext = fold_extension()
        self.scratch = torch.empty(0, device=self.scratch.device, dtype=torch.float32)

    def _run(self, phase, *args):
        if phase == 3:
            return
        # The fixed fold has no Phi argument accesses. Use U as a shape/layout
        # placeholder so the trusted flush validator can also accept shared P0.
        super()._run(phase, *args[:-1], args[-2])
