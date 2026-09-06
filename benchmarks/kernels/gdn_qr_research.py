# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ridge-preserving augmented QR representation, using the unchanged GDN step."""

import ctypes
import functools

import torch
from gdn_unfolded_research import ROOT, replace
from torch.utils.cpp_extension import load_inline

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as flush
from vllm.triton_utils import tl, triton


@triton.jit
def qr_product(
    H,
    U,
    P,
    ROWS,
    MAP,
    WIDTHS,
    NR: tl.constexpr,
    HV: tl.constexpr,
    G: tl.constexpr,
    SH: tl.constexpr,
    BG: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    head, ri = b % HV, b // HV
    if ri >= tl.load(ROWS + NR):
        return
    slot = tl.load(ROWS + ri)
    m = tl.load(WIDTHS + head)
    if slot <= 0 or m <= 0:
        return
    compact = tl.load(MAP + slot)
    gs = tl.arange(0, BG)
    ks = tile * 32 + tl.arange(0, 32)
    cache = (compact * HV + head) * G * 128
    value = tl.full((BG, 32), 0.0, tl.float32)
    for start in range(0, 128, 32):
        vs = start + tl.arange(0, 32)
        qv = tl.load(U + cache + gs[:, None] * 128 + vs[None, :], gs[:, None] < m, 0)
        h = tl.load(H + slot * SH + head * 16384 + vs[:, None] * 128 + ks[None, :])
        value = tl.dot(qv, h, value, input_precision="tf32x3")
    dest = P + cache + gs[:, None] * 128 + ks[None, :]
    ridge = tl.load(dest, gs[:, None] < G, 0)
    tl.store(dest, value + ridge, gs[:, None] < G)


@functools.lru_cache(None)
def fold_extension():
    source = replace(
        flush._SRC, "const bool do_phi = exact && m < SK;", "const bool do_phi = false;"
    )
    build = ROOT / "qr_fold"
    build.mkdir(exist_ok=True)
    return load_inline(
        name="gdn_qr_fold_v1",
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


class QRWorkspace(flush.FlushWorkspace):
    """Store Q_v and Q_v^T H + sqrt(eta) Q_b^T J in the existing U/P buffers.

    The research implementation handles every width in [0,G] for G <= 32.
    A positive ridge and nonzero state ensure a full-rank augmented sketch.
    """

    def __init__(self, *args, split=False, mgs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split or mgs
        if self.g > 32:
            raise ValueError("QR research specialization currently requires G <= 32")
        self._ext = fold_extension()
        self.library = ctypes.CDLL(
            str(ROOT / ("libgdn_qr_mgs.so" if mgs else "libgdn_qr_candidates.so"))
        )
        self.refresh_fn = self.library.mgs_refresh if mgs else self.library.qr_refresh
        p, i = ctypes.c_void_p, ctypes.c_int
        self.refresh_fn.argtypes = (
            [i] * 3 + [ctypes.c_long] + [p] * 6 + [ctypes.c_float, i, p]
        )
        self.refresh_fn.restype = i

    def _run(self, phase, *args):
        if phase == 0:
            super()._run(1, *args)
            super()._run(2, *args)
        elif phase not in (3, 4):
            return super()._run(phase, *args)
        state, _, _, _, rows, mapping, widths, _, u, phi = args
        status = self.refresh_fn(
            self.g,
            self.max_rows,
            self.hv,
            state.stride(0),
            *[x.data_ptr() for x in (state, u, phi, rows, mapping, widths)],
            self.ridge,
            int(self.split),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"QR metadata returned CUDA status {status}")
        if self.split and self.max_rows:
            qr_product[(self.max_rows * self.hv, 4)](
                state,
                u,
                phi,
                rows,
                mapping,
                widths,
                self.max_rows,
                self.hv,
                self.g,
                state.stride(0),
                max(16, triton.next_power_of_2(self.g)),
                num_warps=4,
                num_stages=1,
            )
