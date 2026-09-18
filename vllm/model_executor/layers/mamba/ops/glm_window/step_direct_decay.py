# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-bucketed exact-Z read with contiguous U/Phi columns and parallel WY."""

from vllm.triton_utils import tl, triton


@triton.jit
def _step_direct_decay(
    Q,
    KIn,
    VIn,
    Gate,
    Beta,
    A,
    Bias,
    Slots,
    Pos,
    State,
    U,
    Phi,
    LatchHeads,
    KR,
    VR,
    GR,
    BR,
    PrefixR,
    FR,
    UR,
    DR,
    Out,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
    W: tl.constexpr,
    SAFE: tl.constexpr,
    LOWER: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    RAW_K_RING: tl.constexpr = False,
    Ranks=None,
    ALL_SKETCH: tl.constexpr = False,
    EXACT_FLUSH_OUTPUT: tl.constexpr = False,
    Heads=None,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if Heads is not None:
        head = tl.load(Heads + head)
    slot = tl.load(Slots + row)
    if slot >= 0:
        pos = tl.load(Pos + slot)
        kh = tl.arange(0, K)
        vv = tl.arange(0, V)
        gg = tl.arange(0, WG)
        tt = tl.arange(0, W)
        q = tl.load(Q + (row * H + head) * K + kh).to(tl.float32)
        k = tl.load(KIn + (row * H + head) * K + kh).to(tl.float32)
        v = tl.load(VIn + (row * H + head) * V + vv).to(tl.float32)
        if RAW_K_RING:
            # Match current-step arithmetic to the stored raw input precision.
            k = k.to(KR.dtype.element_ty).to(tl.float32)
            v = v.to(VR.dtype.element_ty).to(tl.float32)
        raw_k = k
        if NORMALIZE:
            q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        q = q * SCALE
        raw_g = tl.load(Gate + (row * H + head) * K + kh).to(tl.float32)
        raw_g += tl.load(Bias + head * K + kh)
        amplitude = tl.exp(tl.load(A + head))
        if SAFE:
            log_a = LOWER / (1.0 + tl.exp(-amplitude * raw_g))
        else:
            log_a = -amplitude * tl.where(
                raw_g > 20.0, raw_g, tl.log(1.0 + tl.exp(raw_g))
            )
        beta = tl.sigmoid(tl.load(Beta + row * H + head).to(tl.float32))
        base = (slot * H + head) * W
        prev = tl.load(PrefixR + (base + pos - 1) * K + kh, mask=pos > 0, other=0.0)
        prefix = prev + log_a
        current_decay = tl.exp(prefix)
        # A derived FP32 replay factor, not a change to raw BF16 k/v rings.
        # GLM's LOWER=-5 and W16 bound -prefix by 80, within FP32 range.
        tl.store(DR + (base + pos) * K + kh, k * tl.exp(-prefix))
        if RAW_K_RING:
            tl.store(KR + (base + pos) * K + kh, raw_k)
        else:
            tl.store(KR + (base + pos) * K + kh, k)
        tl.store(VR + (base + pos) * V + vv, v)
        tl.store(GR + (base + pos) * K + kh, log_a)
        tl.store(BR + base + pos, beta)
        tl.store(PrefixR + (base + pos) * K + kh, prefix)
        is_latch = tl.load(LatchHeads + head)
        head_rank = tl.load(Ranks + head)
        # The exact flush consumes raw rings and overwrites output; the final
        # approximate read and replay f/u entry are never used after this boundary.
        if EXACT_FLUSH_OUTPUT and (ALL_SKETCH or is_latch):  # noqa: SIM102
            # Keep the constexpr branch separate for Triton specialization.
            if pos == W - 1:
                return
        if ALL_SKETCH or is_latch:
            past_d = tl.load(
                DR + (base + tt[:, None]) * K + kh[None, :],
                mask=tt[:, None] < pos,
                other=0.0,
            )
            ell = past_d * current_decay[None, :]
            kk = tl.sum(ell * k[None, :], axis=1)
            kq = tl.sum(ell * q[None, :], axis=1)
            phi = tl.load(
                Phi + (slot * H + head) * K * G + kh[:, None] + gg[None, :] * K,
                mask=gg[None, :] < head_rank,
                other=0.0,
            ).to(tl.float32)
            dk = current_decay * k
            dq = current_decay * q
            fs = tl.load(
                FR + (base + tt[:, None]) * G + gg[None, :],
                mask=(tt[:, None] < pos) & (gg[None, :] < head_rank),
                other=0.0,
            ).to(tl.float32)
            f = beta * (
                tl.sum(phi * dk[:, None], axis=0) - tl.sum(fs * kk[:, None], axis=0)
            )
            cur_kq = tl.sum(k * q)
            c = (
                tl.sum(phi * dq[:, None], axis=0)
                - tl.sum(fs * kq[:, None], axis=0)
                - f * cur_kq
            )
            us = tl.load(
                UR + (base + tt[:, None]) * V + vv[None, :],
                mask=tt[:, None] < pos,
                other=0.0,
            )
            u = beta * (v - tl.sum(us * kk[:, None], axis=0))
            latch = tl.load(
                U + (slot * H + head) * V * G + vv[:, None] + gg[None, :] * V,
                mask=gg[None, :] < head_rank,
                other=0.0,
            ).to(tl.float32)
            out = (
                tl.sum(latch * c[None, :], axis=1)
                + tl.sum(us * kq[:, None], axis=0)
                + u * cur_kq
            )
            tl.store(FR + (base + pos) * G + gg, f, mask=gg < head_rank)
            tl.store(UR + (base + pos) * V + vv, u)
        else:
            # Existing dense recurrence; this head updates state on every step.
            sp = State + (slot * H + head) * V * K + vv[:, None] * K + kh[None, :]
            state = tl.load(sp) * tl.exp(log_a[None, :])
            delta = beta * (v - tl.sum(state * k[None, :], axis=1))
            state += delta[:, None] * k[None, :]
            out = tl.sum(state * q[None, :], axis=1)
            tl.store(sp, state)
        tl.store(Out + (row * H + head) * V + vv, out)
