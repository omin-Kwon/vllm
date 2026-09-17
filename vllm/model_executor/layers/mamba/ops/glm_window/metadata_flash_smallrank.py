# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse tiled Flash-style flush with exact small-rank ridge sufficient statistics."""

from vllm.triton_utils import tl, triton

from .flash_wy import flash_wy_update


@triton.jit
def flash_small_metadata(
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
    FLASH_WY: tl.constexpr = False,
    PreparedA=None,
    Restored=None,
    Inverse=None,
    Decay=None,
):
    tl.static_assert(FLUSH and FLASH_WY)
    tl.static_assert(WG == 1 or WG == 4)
    total = tl.load(WorkCounts + 1) * NH
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + Capacity + item // NH)
        h = tl.load(Heads + item % NH)
        slot = tl.load(slots + row).to(tl.int64)
        m = tl.load(widths + h)
        k = tl.arange(0, 128)
        norm = tl.full((), 0.0, tl.float32)
        if WG > 0:
            omega0 = tl.load(frame + h.to(tl.int64) * 16384 + k + 0 * 128, m > 0, 0.0)
            rhs0 = tl.full((128,), 0.0, tl.float32)
            gram0_0 = tl.full((), 0.0, tl.float32)
        if WG > 1:
            omega1 = tl.load(frame + h.to(tl.int64) * 16384 + k + 1 * 128, m > 1, 0.0)
            rhs1 = tl.full((128,), 0.0, tl.float32)
            gram1_0 = tl.full((), 0.0, tl.float32)
            gram1_1 = tl.full((), 0.0, tl.float32)
        if WG > 2:
            omega2 = tl.load(frame + h.to(tl.int64) * 16384 + k + 2 * 128, m > 2, 0.0)
            rhs2 = tl.full((128,), 0.0, tl.float32)
            gram2_0 = tl.full((), 0.0, tl.float32)
            gram2_1 = tl.full((), 0.0, tl.float32)
            gram2_2 = tl.full((), 0.0, tl.float32)
        if WG > 3:
            omega3 = tl.load(frame + h.to(tl.int64) * 16384 + k + 3 * 128, m > 3, 0.0)
            rhs3 = tl.full((128,), 0.0, tl.float32)
            gram3_0 = tl.full((), 0.0, tl.float32)
            gram3_1 = tl.full((), 0.0, tl.float32)
            gram3_2 = tl.full((), 0.0, tl.float32)
            gram3_3 = tl.full((), 0.0, tl.float32)
        if EXACT_OUTPUT:
            query = tl.load(Q + (row.to(tl.int64) * H + h) * 128 + k).to(tl.float32)
            query *= tl.rsqrt(tl.sum(query * query) + 1e-6) * 128**-0.5
        for tile in range(4):
            v = tile * 32 + tl.arange(0, 32)
            ptr = (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :]
            raw = tl.load(state + ptr)
            raw = flash_wy_update(
                raw,
                PreparedA,
                Restored,
                Inverse,
                Decay,
                vr,
                br,
                slot,
                h,
                H,
                tile * 32,
                32,
            )
            tl.store(state + ptr, raw)
            if EXACT_OUTPUT:
                tl.store(
                    Out + (row.to(tl.int64) * H + h) * 128 + v,
                    tl.sum(raw * query[None, :], axis=1),
                )
            norm += tl.sum(raw * raw)
            if WG > 0:
                u0 = tl.sum(raw * omega0[None, :], axis=1)
                tl.store(u + (slot * H + h) * 128 * G + v + 0 * 128, u0, m > 0)
                rhs0 += tl.sum(raw * u0[:, None], axis=0)
                gram0_0 += tl.sum(u0 * u0)
            if WG > 1:
                u1 = tl.sum(raw * omega1[None, :], axis=1)
                tl.store(u + (slot * H + h) * 128 * G + v + 1 * 128, u1, m > 1)
                rhs1 += tl.sum(raw * u1[:, None], axis=0)
                gram1_0 += tl.sum(u1 * u0)
                gram1_1 += tl.sum(u1 * u1)
            if WG > 2:
                u2 = tl.sum(raw * omega2[None, :], axis=1)
                tl.store(u + (slot * H + h) * 128 * G + v + 2 * 128, u2, m > 2)
                rhs2 += tl.sum(raw * u2[:, None], axis=0)
                gram2_0 += tl.sum(u2 * u0)
                gram2_1 += tl.sum(u2 * u1)
                gram2_2 += tl.sum(u2 * u2)
            if WG > 3:
                u3 = tl.sum(raw * omega3[None, :], axis=1)
                tl.store(u + (slot * H + h) * 128 * G + v + 3 * 128, u3, m > 3)
                rhs3 += tl.sum(raw * u3[:, None], axis=0)
                gram3_0 += tl.sum(u3 * u0)
                gram3_1 += tl.sum(u3 * u1)
                gram3_2 += tl.sum(u3 * u2)
                gram3_3 += tl.sum(u3 * u3)
        ridge = 0.1 * tl.where(norm > 0, norm / 128.0, 1.0)
        if WG == 1:
            tl.store(phi + (slot * H + h) * 128 * G + k, rhs0 / (gram0_0 + ridge))
        else:
            gram0_0 += ridge
            gram1_1 += ridge
            gram2_2 += ridge
            gram3_3 += ridge
            l0_0 = tl.sqrt(gram0_0)
            l1_0 = (gram1_0) / l0_0
            l1_1 = tl.sqrt(gram1_1 - l1_0 * l1_0)
            l2_0 = (gram2_0) / l0_0
            l2_1 = (gram2_1 - l2_0 * l1_0) / l1_1
            l2_2 = tl.sqrt(gram2_2 - l2_0 * l2_0 - l2_1 * l2_1)
            l3_0 = (gram3_0) / l0_0
            l3_1 = (gram3_1 - l3_0 * l1_0) / l1_1
            l3_2 = (gram3_2 - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
            l3_3 = tl.sqrt(gram3_3 - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2)
            y0 = (rhs0) / l0_0
            y1 = (rhs1 - l1_0 * y0) / l1_1
            y2 = (rhs2 - l2_0 * y0 - l2_1 * y1) / l2_2
            y3 = (rhs3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
            x3 = (y3) / l3_3
            x2 = (y2 - l3_2 * x3) / l2_2
            x1 = (y1 - l2_1 * x2 - l3_1 * x3) / l1_1
            x0 = (y0 - l1_0 * x1 - l2_0 * x2 - l3_0 * x3) / l0_0
            tl.store(phi + (slot * H + h) * 128 * G + k + 0 * 128, x0, m > 0)
            tl.store(phi + (slot * H + h) * 128 * G + k + 1 * 128, x1, m > 1)
            tl.store(phi + (slot * H + h) * 128 * G + k + 2 * 128, x2, m > 2)
            tl.store(phi + (slot * H + h) * 128 * G + k + 3 * 128, x3, m > 3)
