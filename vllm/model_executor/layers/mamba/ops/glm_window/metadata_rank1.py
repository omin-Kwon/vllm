# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused native-coordinate pivot metadata: no full frame rotation or Gram.

Uses ||S||_F^2 = ||S R||_F^2 for the offline orthogonal frame R.
The pivot span, diagonal residual, ridge and all 128 query coordinates are
unchanged. Persistent state is never rotated. k/v rings are untouched.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def rank1_metadata(
    state,
    slots,
    pos,
    frame,
    widths,
    u,
    phi,
    kr,
    vr,
    gr,
    br,
    fresh,
    WorkRows,
    WorkCounts,
    Heads,
    NH: tl.constexpr,
    Capacity: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
    FLUSH: tl.constexpr,
    Q=None,
    Out=None,
    EXACT_OUTPUT: tl.constexpr = False,
):
    total = tl.load(WorkCounts + (1 if FLUSH else 0)) * NH
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + (Capacity if FLUSH else 0) + item // NH)
        h = tl.load(Heads + item % NH)
        slot = tl.load(slots + row).to(tl.int64)
        m = tl.load(widths + h)
        if m > 0:
            k = tl.arange(0, 128)
            v = tl.arange(0, 128)
            raw = tl.load(
                state + (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :]
            )
            if FLUSH:
                ring = (slot * H + h) * 16
                for t in range(16):
                    key = tl.load(kr + (ring + t) * 128 + k).to(tl.float32)
                    key *= tl.rsqrt(tl.sum(key * key) + 1e-6)
                    value = tl.load(vr + (ring + t) * 128 + v).to(tl.float32)
                    decay = tl.load(gr + (ring + t) * 128 + k)
                    beta = tl.load(br + ring + t)
                    raw *= tl.exp(decay[None, :])
                    delta = beta * (value - tl.sum(raw * key[None, :], axis=1))
                    raw += delta[:, None] * key[None, :]
                tl.store(
                    state + (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :], raw
                )
                if EXACT_OUTPUT:
                    query = tl.load(Q + (row.to(tl.int64) * H + h) * 128 + k).to(
                        tl.float32
                    )
                    query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
                    tl.store(
                        Out + (row.to(tl.int64) * H + h) * 128 + v,
                        tl.sum(raw * query[None, :], axis=1),
                    )
            omega = tl.load(frame + h.to(tl.int64) * 16384 + k * 128)
            uu = tl.sum(raw * omega[None, :], axis=1)
            energy = tl.sum(uu * uu)
            mu = tl.sum(tl.sum(raw * raw, axis=0), axis=0) / 128.0
            denominator = energy + 0.1 * mu
            numerator = tl.sum(raw * uu[:, None], axis=0)
            result = numerator / tl.where(denominator > 0, denominator, 1.0)
            tl.store(u + (slot * H + h) * 128 * G + v * G, uu)
            tl.store(phi + (slot * H + h) * 128 * G + k * G, result)
