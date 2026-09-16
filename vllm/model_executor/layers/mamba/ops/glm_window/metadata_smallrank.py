# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""m<=4: P4/P6 span all sketch columns, so solve the exact small ridge system.

Phi = S.T U (U.T U + 0.1*mu I)^-1, U=S Omega; mu=||S||F^2/128.
This is algebraically the same pivot surrogate when m<=P, not a new
allocation/Full-Gram inference option. All K128 coordinates remain present.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def small_metadata(
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
        k = tl.arange(0, 128)
        v = tl.arange(0, 128)
        raw = tl.load(state + (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :])
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
                query = tl.load(Q + (row.to(tl.int64) * H + h) * 128 + k).to(tl.float32)
                query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
                tl.store(
                    Out + (row.to(tl.int64) * H + h) * 128 + v,
                    tl.sum(raw * query[None, :], axis=1),
                )
        mu = tl.sum(tl.sum(raw * raw, axis=0), axis=0) / 128.0
        ridge = 0.1 * tl.where(mu > 0, mu, 1.0)
        omega0 = tl.load(frame + h.to(tl.int64) * 16384 + k * 128 + 0, m > 0, 0.0)
        u0 = tl.sum(raw * omega0[None, :], axis=1)
        tl.store(u + (slot * H + h) * 128 * G + v * G + 0, u0, m > 0)
        omega1 = tl.load(frame + h.to(tl.int64) * 16384 + k * 128 + 1, m > 1, 0.0)
        u1 = tl.sum(raw * omega1[None, :], axis=1)
        tl.store(u + (slot * H + h) * 128 * G + v * G + 1, u1, m > 1)
        omega2 = tl.load(frame + h.to(tl.int64) * 16384 + k * 128 + 2, m > 2, 0.0)
        u2 = tl.sum(raw * omega2[None, :], axis=1)
        tl.store(u + (slot * H + h) * 128 * G + v * G + 2, u2, m > 2)
        omega3 = tl.load(frame + h.to(tl.int64) * 16384 + k * 128 + 3, m > 3, 0.0)
        u3 = tl.sum(raw * omega3[None, :], axis=1)
        tl.store(u + (slot * H + h) * 128 * G + v * G + 3, u3, m > 3)
        gram0_0 = tl.sum(u0 * u0) + ridge
        gram1_0 = tl.sum(u1 * u0)
        gram1_1 = tl.sum(u1 * u1) + ridge
        gram2_0 = tl.sum(u2 * u0)
        gram2_1 = tl.sum(u2 * u1)
        gram2_2 = tl.sum(u2 * u2) + ridge
        gram3_0 = tl.sum(u3 * u0)
        gram3_1 = tl.sum(u3 * u1)
        gram3_2 = tl.sum(u3 * u2)
        gram3_3 = tl.sum(u3 * u3) + ridge
        rhs0 = tl.sum(raw * u0[:, None], axis=0)
        rhs1 = tl.sum(raw * u1[:, None], axis=0)
        rhs2 = tl.sum(raw * u2[:, None], axis=0)
        rhs3 = tl.sum(raw * u3[:, None], axis=0)
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
        tl.store(phi + (slot * H + h) * 128 * G + k * G + 0, x0, m > 0)
        tl.store(phi + (slot * H + h) * 128 * G + k * G + 1, x1, m > 1)
        tl.store(phi + (slot * H + h) * 128 * G + k * G + 2, x2, m > 2)
        tl.store(phi + (slot * H + h) * 128 * G + k * G + 3, x3, m > 3)
