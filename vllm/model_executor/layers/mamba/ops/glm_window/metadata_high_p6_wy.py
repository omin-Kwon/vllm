# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused native-coordinate pivot metadata: no full frame rotation or Gram.

Uses ||S||_F^2 = ||S R||_F^2 for the offline orthogonal frame R.
The pivot span, diagonal residual, ridge and all 128 query coordinates are
unchanged. Persistent state is never rotated. k/v rings are untouched.
"""

from vllm.triton_utils import tl, triton

from .wy_update import wy_update


@triton.jit
def build_persistent(
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
            g = tl.arange(0, WG)
            raw = tl.load(
                state + (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :]
            )
            if FLUSH:
                raw = wy_update(raw, kr, vr, gr, br, slot, h, H)
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
            omega = tl.load(
                frame + h.to(tl.int64) * 16384 + k[:, None] * 128 + g[None, :],
                g[None, :] < m,
                0.0,
            )
            s = tl.dot(raw, omega, input_precision="tf32x3")
            tl.store(
                u + (slot * H + h) * 128 * G + v[:, None] * G + g[None, :],
                s,
                g[None, :] < G,
            )
            energy = tl.sum(s * s, axis=0)
            mean = tl.sum(raw * raw) / 128.0
            safe_mean = tl.where(mean > 0.0, mean, 1.0)
            u0 = tl.sum(tl.where(g[None, :] == 0, s, 0.0), axis=1)
            v0 = tl.full((128,), 0.0, tl.float32)
            if m > 0:
                w0 = u0
                e0 = tl.sum(w0 * w0)
                keep0 = e0 > 0.0
                v0 = tl.where(keep0, w0 / tl.sqrt(tl.where(keep0, e0, 1.0)), 0.0)
            zr0 = tl.sum(v0[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z0 = tl.sum(zr0[:, None] * omega, axis=0)
            u1 = tl.sum(tl.where(g[None, :] == 1, s, 0.0), axis=1)
            v1 = tl.full((128,), 0.0, tl.float32)
            if m > 1:
                w1 = u1
                w1 = w1 - v0 * tl.sum(v0 * w1)
                w1 = w1 - v0 * tl.sum(v0 * w1)
                e1 = tl.sum(w1 * w1)
                keep1 = e1 > 1.0e-12 * tl.sum(u1 * u1)
                v1 = tl.where(keep1, w1 / tl.sqrt(tl.where(keep1, e1, 1.0)), 0.0)
            zr1 = tl.sum(v1[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z1 = tl.sum(zr1[:, None] * omega, axis=0)
            u2 = tl.sum(tl.where(g[None, :] == 2, s, 0.0), axis=1)
            v2 = tl.full((128,), 0.0, tl.float32)
            if m > 2:
                w2 = u2
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                e2 = tl.sum(w2 * w2)
                keep2 = e2 > 1.0e-12 * tl.sum(u2 * u2)
                v2 = tl.where(keep2, w2 / tl.sqrt(tl.where(keep2, e2, 1.0)), 0.0)
            zr2 = tl.sum(v2[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z2 = tl.sum(zr2[:, None] * omega, axis=0)
            u3 = tl.sum(tl.where(g[None, :] == 3, s, 0.0), axis=1)
            v3 = tl.full((128,), 0.0, tl.float32)
            if m > 3:
                w3 = u3
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                e3 = tl.sum(w3 * w3)
                keep3 = e3 > 1.0e-12 * tl.sum(u3 * u3)
                v3 = tl.where(keep3, w3 / tl.sqrt(tl.where(keep3, e3, 1.0)), 0.0)
            zr3 = tl.sum(v3[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z3 = tl.sum(zr3[:, None] * omega, axis=0)
            u4 = tl.sum(tl.where(g[None, :] == 4, s, 0.0), axis=1)
            v4 = tl.full((128,), 0.0, tl.float32)
            if m > 4:
                w4 = u4
                w4 = w4 - v0 * tl.sum(v0 * w4)
                w4 = w4 - v1 * tl.sum(v1 * w4)
                w4 = w4 - v2 * tl.sum(v2 * w4)
                w4 = w4 - v3 * tl.sum(v3 * w4)
                w4 = w4 - v0 * tl.sum(v0 * w4)
                w4 = w4 - v1 * tl.sum(v1 * w4)
                w4 = w4 - v2 * tl.sum(v2 * w4)
                w4 = w4 - v3 * tl.sum(v3 * w4)
                e4 = tl.sum(w4 * w4)
                keep4 = e4 > 1.0e-12 * tl.sum(u4 * u4)
                v4 = tl.where(keep4, w4 / tl.sqrt(tl.where(keep4, e4, 1.0)), 0.0)
            zr4 = tl.sum(v4[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z4 = tl.sum(zr4[:, None] * omega, axis=0)
            u5 = tl.sum(tl.where(g[None, :] == 5, s, 0.0), axis=1)
            v5 = tl.full((128,), 0.0, tl.float32)
            if m > 5:
                w5 = u5
                w5 = w5 - v0 * tl.sum(v0 * w5)
                w5 = w5 - v1 * tl.sum(v1 * w5)
                w5 = w5 - v2 * tl.sum(v2 * w5)
                w5 = w5 - v3 * tl.sum(v3 * w5)
                w5 = w5 - v4 * tl.sum(v4 * w5)
                w5 = w5 - v0 * tl.sum(v0 * w5)
                w5 = w5 - v1 * tl.sum(v1 * w5)
                w5 = w5 - v2 * tl.sum(v2 * w5)
                w5 = w5 - v3 * tl.sum(v3 * w5)
                w5 = w5 - v4 * tl.sum(v4 * w5)
                e5 = tl.sum(w5 * w5)
                keep5 = e5 > 1.0e-12 * tl.sum(u5 * u5)
                v5 = tl.where(keep5, w5 / tl.sqrt(tl.where(keep5, e5, 1.0)), 0.0)
            zr5 = tl.sum(v5[:, None] * raw, axis=0) / tl.sqrt(safe_mean)
            z5 = tl.sum(zr5[:, None] * omega, axis=0)
            residual = tl.maximum(
                energy / safe_mean
                - z0 * z0
                - z1 * z1
                - z2 * z2
                - z3 * z3
                - z4 * z4
                - z5 * z5,
                0.0,
            )
            residual = tl.where(g < tl.minimum(m, 6), 0.0, residual)
            denominator = residual + 0.1
            a = residual / denominator
            b0 = tl.where(g < m, z0 / denominator, 0.0)
            b1 = tl.where(g < m, z1 / denominator, 0.0)
            b2 = tl.where(g < m, z2 / denominator, 0.0)
            b3 = tl.where(g < m, z3 / denominator, 0.0)
            b4 = tl.where(g < m, z4 / denominator, 0.0)
            b5 = tl.where(g < m, z5 / denominator, 0.0)
            l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
            l1_0 = (tl.sum(z1 * b0)) / l0_0
            l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
            l2_0 = (tl.sum(z2 * b0)) / l0_0
            l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
            l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
            l3_0 = (tl.sum(z3 * b0)) / l0_0
            l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
            l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
            l3_3 = tl.sqrt(
                1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
            )
            l4_0 = (tl.sum(z4 * b0)) / l0_0
            l4_1 = (tl.sum(z4 * b1) - l4_0 * l1_0) / l1_1
            l4_2 = (tl.sum(z4 * b2) - l4_0 * l2_0 - l4_1 * l2_1) / l2_2
            l4_3 = (tl.sum(z4 * b3) - l4_0 * l3_0 - l4_1 * l3_1 - l4_2 * l3_2) / l3_3
            l4_4 = tl.sqrt(
                1.0
                + tl.sum(z4 * b4)
                - l4_0 * l4_0
                - l4_1 * l4_1
                - l4_2 * l4_2
                - l4_3 * l4_3
            )
            l5_0 = (tl.sum(z5 * b0)) / l0_0
            l5_1 = (tl.sum(z5 * b1) - l5_0 * l1_0) / l1_1
            l5_2 = (tl.sum(z5 * b2) - l5_0 * l2_0 - l5_1 * l2_1) / l2_2
            l5_3 = (tl.sum(z5 * b3) - l5_0 * l3_0 - l5_1 * l3_1 - l5_2 * l3_2) / l3_3
            l5_4 = (
                tl.sum(z5 * b4) - l5_0 * l4_0 - l5_1 * l4_1 - l5_2 * l4_2 - l5_3 * l4_3
            ) / l4_4
            l5_5 = tl.sqrt(
                1.0
                + tl.sum(z5 * b5)
                - l5_0 * l5_0
                - l5_1 * l5_1
                - l5_2 * l5_2
                - l5_3 * l5_3
                - l5_4 * l5_4
            )
            y0 = (b0) / l0_0
            y1 = (b1 - l1_0 * y0) / l1_1
            y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
            y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
            y4 = (b4 - l4_0 * y0 - l4_1 * y1 - l4_2 * y2 - l4_3 * y3) / l4_4
            y5 = (b5 - l5_0 * y0 - l5_1 * y1 - l5_2 * y2 - l5_3 * y3 - l5_4 * y4) / l5_5
            g5 = (y5) / l5_5
            g4 = (y4 - l5_4 * g5) / l4_4
            g3 = (y3 - l4_3 * g4 - l5_3 * g5) / l3_3
            g2 = (y2 - l3_2 * g3 - l4_2 * g4 - l5_2 * g5) / l2_2
            g1 = (y1 - l2_1 * g2 - l3_1 * g3 - l4_1 * g4 - l5_1 * g5) / l1_1
            g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3 - l4_0 * g4 - l5_0 * g5) / l0_0
            factor = tl.where(g < m, 0.1 / denominator, 1.0)
            native0 = zr0 + tl.sum(omega * (z0 * (factor - 1.0))[None, :], axis=1)
            native1 = zr1 + tl.sum(omega * (z1 * (factor - 1.0))[None, :], axis=1)
            native2 = zr2 + tl.sum(omega * (z2 * (factor - 1.0))[None, :], axis=1)
            native3 = zr3 + tl.sum(omega * (z3 * (factor - 1.0))[None, :], axis=1)
            native4 = zr4 + tl.sum(omega * (z4 * (factor - 1.0))[None, :], axis=1)
            native5 = zr5 + tl.sum(omega * (z5 * (factor - 1.0))[None, :], axis=1)
            result = omega * tl.where(g < m, a, 0.0)[None, :]
            result += native0[:, None] * g0[None, :]
            result += native1[:, None] * g1[None, :]
            result += native2[:, None] * g2[None, :]
            result += native3[:, None] * g3[None, :]
            result += native4[:, None] * g4[None, :]
            result += native5[:, None] * g5[None, :]
            tl.store(
                phi + (slot * H + h) * 128 * G + k[:, None] * G + g[None, :],
                result,
                g[None, :] < G,
            )
