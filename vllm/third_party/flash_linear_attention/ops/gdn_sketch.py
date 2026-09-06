# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full-dimension GDN step with an implicit identity prefix.

Persistent-state refresh is owned by the caller. The step preserves the existing
raw-WY write and normalized projected-factor ring contracts. It has no input
truncation, anchor, or freeze mode. Packed metadata is per value head.
"""

import torch

from vllm.triton_utils import tl, triton


def pack_metadata(phi, widths):
    """Pack the nonidentity rows of full coefficient maps (NS, HV, G, K).

    This setup/reference helper accepts host allocation widths. Runtime refresh
    will write the same packed layout directly; do not call it during decode.
    """
    ns, hv, g, k = phi.shape
    if len(widths) != hv or any(m < 0 or m > min(g, k) for m in widths):
        raise ValueError("Invalid per-value-head sketch widths")
    offsets = [0]
    parts = []
    for h, m in enumerate(widths):
        parts.append(phi[:, h, :m, m:].reshape(ns, m * (k - m)))
        offsets.append(offsets[-1] + m * (k - m))
    packed = torch.cat(parts, dim=1).contiguous()
    return packed, torch.tensor(offsets, dtype=torch.int32, device=phi.device)


class StepPlan:
    """Fixed host allocation and reusable per-key-head workspace.

    Build before graph capture. Do not share a plan between concurrent streams.
    Widths and head geometry must match the metadata for its entire lifetime.
    """

    def __init__(self, widths, batch, key_heads, device, share_keys=True):
        if (
            batch < 1
            or key_heads < 1
            or not widths
            or len(widths) % key_heads
            or any(m < 0 or m > 128 for m in widths)
        ):
            raise ValueError("Invalid allocation")
        self.max_width = max(widths)
        self.batch = batch
        self.key_heads = key_heads
        self.value_heads = len(widths)
        self.share_keys = share_keys
        self.device = torch.device(device)
        buckets = {}
        for head, m in enumerate(widths):
            shape = (
                m > 0,
                triton.next_power_of_2(max(1, m)),
                (triton.next_power_of_2(128 - m) if m < 128 else 0) if m else 1,
            )
            buckets.setdefault(shape, []).append(head)
        self.groups = [
            (shape, torch.tensor(heads, device=device, dtype=torch.int32))
            for shape, heads in sorted(buckets.items())
        ]
        self.shared = torch.empty(
            batch, key_heads, 2 * 128 + 2 * 16 + 1, device=device, dtype=torch.float32
        )


@triton.jit
def _prepare_keys(
    MIX,
    KEYS,
    INDEX,
    POS,
    SHARED,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
    SM: tl.constexpr,
    SK: tl.constexpr,
    SCALE: tl.constexpr,
):
    row, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(INDEX + row)
    if slot > 0:
        pos = tl.load(POS + row)
        k, s = tl.arange(0, K), tl.arange(0, W)
        q = tl.load(MIX + row * SM + h * K + k).to(tl.float32)
        key = tl.load(MIX + row * SM + (H + h) * K + k).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
        key *= tl.rsqrt(tl.sum(key * key, 0) + 1.0e-6)
        ring = tl.load(
            KEYS + slot * SK + (h * W + s[:, None]) * K + k[None, :],
            s[:, None] < pos,
            0.0,
        )
        kq = tl.sum(ring * q[None, :], 1)
        kk = tl.sum(ring * key[None, :], 1)
        base = SHARED + (row * H + h) * (2 * K + 2 * W + 1)
        tl.store(base + k, q)
        tl.store(base + K + k, key)
        tl.store(base + 2 * K + s, kq)
        tl.store(base + 2 * K + W + s, kk)
        tl.store(base + 2 * K + 2 * W, tl.sum(key * q, 0))
        tl.store(KEYS + slot * SK + (h * W + pos) * K + k, key)


@triton.jit
def _step(
    MIX,
    A,
    BETA,
    ALOG,
    BIAS,
    OUT,
    STATE,
    DRING,
    KRING,
    GRING,
    INDEX,
    POS,
    MAP,
    U,
    PHI,
    OFFSET,
    WIDTH,
    FACTOR,
    HEADS,
    SHARED,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BT: tl.constexpr,
    PLANNED: tl.constexpr,
    SHARE: tl.constexpr,
    SCALE: tl.constexpr,
    SM: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    SH0: tl.constexpr,
    SH1: tl.constexpr,
    SH2: tl.constexpr,
    SH3: tl.constexpr,
    SD: tl.constexpr,
    SK: tl.constexpr,
    SG: tl.constexpr,
    SU: tl.constexpr,
    SP: tl.constexpr,
    SF: tl.constexpr,
    SKETCH: tl.constexpr,
):
    row = tl.program_id(0)
    hv = tl.load(HEADS + tl.program_id(1)) if PLANNED else tl.program_id(1)
    h = hv // (HV // H)
    slot = tl.load(INDEX + row)
    v = tl.arange(0, V)
    m = tl.load(WIDTH + hv)
    if slot <= 0:
        tl.store(OUT + (row * HV + hv) * V + v, 0.0)
    elif (m > 0) == SKETCH:
        compact = tl.load(MAP + slot)
        pos = tl.load(POS + row)
        k = tl.arange(0, K)
        s = tl.arange(0, W)
        g = tl.arange(0, BG)
        shared_base = SHARED + (row * H + h) * (2 * K + 2 * W + 1)
        if SHARE:
            q = tl.load(shared_base + k)
            key = tl.load(shared_base + K + k)
        else:
            q = tl.load(MIX + row * SM + h * K + k).to(tl.float32)
            key = tl.load(MIX + row * SM + (H + h) * K + k).to(tl.float32)
            q_scale = tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
            k_scale = tl.rsqrt(tl.sum(key * key, 0) + 1.0e-6)
            q *= q_scale
            key *= k_scale
        av = tl.load(A + row * SA + hv).to(tl.float32)
        x = av + tl.load(BIAS + hv).to(tl.float32)
        soft = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
        loga = -tl.exp(tl.load(ALOG + hv).to(tl.float32)) * soft
        alpha = tl.exp(loga)
        bv = tl.load(BETA + row * SB + hv).to(tl.float32)
        beta = (1.0 / (1.0 + tl.exp(-bv))).to(BETA.dtype.element_ty).to(tl.float32)
        if SHARE:
            kq = tl.load(shared_base + 2 * K + s)
            kk = tl.load(shared_base + 2 * K + W + s)
            cur = tl.load(shared_base + 2 * K + 2 * W)
        else:
            keys = tl.load(
                KRING + slot * SK + (h * W + s[:, None]) * K + k[None, :],
                s[:, None] < pos,
                0.0,
            )
            kq = tl.sum(keys * q[None, :], 1)
            kk = tl.sum(keys * key[None, :], 1)
            cur = tl.sum(key * q, 0)
        logs = tl.load(GRING + slot * SG + hv * W + s, s < pos, 0.0)
        total = tl.sum(logs, 0)
        decay = tl.where(s < pos, tl.exp(total - tl.cumsum(logs, 0)), 0.0)
        writes = tl.load(
            DRING + slot * SD + (hv * W + s[:, None]) * V + v[None, :],
            s[:, None] < pos,
            0.0,
        )
        # Exact write-ring contributions, unrelated to removed tail anchors.
        replay_q = tl.sum(writes * (decay * kq)[:, None], 0)
        replay_k = tl.sum(writes * (decay * kk)[:, None], 0)
        if SKETCH:
            # Each head owns m*(K-m) values; the m-by-m identity is implicit.
            off = tl.load(OFFSET + hv)
            t = tl.arange(0, max(1, BT))
            if BT == 0:
                tailq = tl.full((BG,), 0.0, tl.float32)
                tailk = tl.full((BG,), 0.0, tl.float32)
            elif BT > 1:
                tail = tl.load(
                    PHI + compact * SP + off + g[:, None] * (K - m) + t[None, :],
                    (g[:, None] < m) & (t[None, :] < K - m),
                    0.0,
                )
                if SHARE:
                    qt = tl.load(shared_base + m + t, t < K - m, 0.0)
                    kt = tl.load(shared_base + K + m + t, t < K - m, 0.0)
                else:
                    qt = (
                        tl.load(MIX + row * SM + h * K + m + t, t < K - m, 0.0).to(
                            tl.float32
                        )
                        * q_scale
                    )
                    kt = (
                        tl.load(
                            MIX + row * SM + (H + h) * K + m + t, t < K - m, 0.0
                        ).to(tl.float32)
                        * k_scale
                    )
                tailq = tl.sum(tail * qt[None, :], 1)
                tailk = tl.sum(tail * kt[None, :], 1)
            else:
                # BT=1 includes width 127 (one real tail value), not just 128.
                tail = tl.load(
                    PHI + compact * SP + off + g * (K - m), (g < m) & (m < K), 0.0
                )
                if SHARE:
                    qt = tl.load(shared_base + m, m < K, 0.0)
                    kt = tl.load(shared_base + K + m, m < K, 0.0)
                else:
                    qt = (
                        tl.load(MIX + row * SM + h * K + m, m < K, 0.0).to(tl.float32)
                        * q_scale
                    )
                    kt = (
                        tl.load(MIX + row * SM + (H + h) * K + m, m < K, 0.0).to(
                            tl.float32
                        )
                        * k_scale
                    )
                tailq, tailk = tail * qt, tail * kt
            if SHARE:
                qg = tl.load(shared_base + g, g < m, 0.0)
                kg = tl.load(shared_base + K + g, g < m, 0.0)
            else:
                qg = (
                    tl.load(MIX + row * SM + h * K + g, g < m, 0.0).to(tl.float32)
                    * q_scale
                )
                kg = (
                    tl.load(MIX + row * SM + (H + h) * K + g, g < m, 0.0).to(tl.float32)
                    * k_scale
                )
            xq = qg + tailq
            xk = kg + tailk
            fs = tl.load(
                FACTOR + compact * SF + (hv * W + s[:, None]) * G + g[None, :],
                (s[:, None] < pos) & (g[None, :] < m),
                0.0,
            )
            f = beta * (xk - tl.sum(fs * kk[:, None], 0))
            coeff = xq - tl.sum(fs * kq[:, None], 0) - f * cur
            u = tl.load(
                U + compact * SU + (hv * G + g[:, None]) * V + v[None, :],
                g[:, None] < m,
                0.0,
            )
            hq = tl.sum(u * coeff[:, None], 0)
            hk = tl.full((V,), 0.0, tl.float32)
            tl.store(FACTOR + compact * SF + (hv * W + pos) * G + g, f, g < m)
        else:
            state = tl.load(
                STATE + slot * SH0 + hv * SH1 + v[:, None] * SH2 + k[None, :] * SH3
            )
            hq = tl.sum(state * q[None, :], 1)
            hk = tl.sum(state * key[None, :], 1)
        value = tl.load(MIX + row * SM + 2 * H * K + hv * V + v).to(tl.float32)
        dc = beta * (value - alpha * (replay_k + tl.exp(total) * hk))
        output = alpha * (tl.exp(total) * hq + replay_q) + dc * cur
        tl.store(OUT + (row * HV + hv) * V + v, output)
        tl.store(DRING + slot * SD + (hv * W + pos) * V + v, dc)
        tl.store(GRING + slot * SG + hv * W + pos, loga)
        # Only one value head writes the shared key slot. Other heads only read
        # prior slots, so the current write cannot race with their reductions.
        if not SHARE and hv % (HV // H) == 0:
            tl.store(KRING + slot * SK + (h * W + pos) * K + k, key)


def step(
    mixed,
    a,
    beta,
    a_log,
    bias,
    out,
    state,
    writes,
    keys,
    gates,
    indices,
    positions,
    slot_map,
    u,
    phi,
    offsets,
    widths,
    factors,
    scale,
    plan=None,
):
    """Launch an anchor-free step; no allocation or host tensor reads.

    Inputs use the existing GDN mixed-QKV layout. Slots and positions must be
    valid, active slots unique, and slot_map must map them to initialized
    compact metadata. Zero/negative slots are padding. Dense widths are zero.
    Rings and metadata have contiguous inner dimensions; state strides may be
    padded. Beta rounding follows the existing step's input-dtype convention.
    """
    _, hv, v, k = state.shape
    h, w = keys.shape[1:3]
    g = u.shape[2]
    if (k, v, w) != (128, 128, 16) or hv % h:
        raise ValueError("Initial specialization requires K=V=128, W=16")
    if not 1 <= g <= k:
        raise ValueError("Invalid sketch width")
    batch, ns = mixed.shape[0], u.shape[0]
    expected = (
        (mixed, (batch, 2 * h * k + hv * v)),
        (out, (batch, hv, v)),
        (a, (batch, hv)),
        (beta, (batch, hv)),
        (a_log, (hv,)),
        (bias, (hv,)),
        (writes, (state.shape[0], hv, w, v)),
        (keys, (state.shape[0], h, w, k)),
        (gates, (state.shape[0], hv, w)),
        (u, (ns, hv, g, v)),
        (factors, (ns, hv, w, g)),
        (widths, (hv,)),
        (offsets, (hv + 1,)),
        (indices, (batch,)),
        (positions, (batch,)),
        (slot_map, (state.shape[0],)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.device != mixed.device:
            raise ValueError("Invalid shape or mixed devices in step inputs")
    if phi.ndim != 2 or phi.shape[0] != ns or phi.device != mixed.device:
        raise ValueError("Invalid packed metadata")
    if mixed.dtype not in (torch.float32, torch.bfloat16) or out.dtype != mixed.dtype:
        raise ValueError("Matching FP32/BF16 input and output required")
    if any(t.stride(-1) != 1 for t in (mixed, a, beta, a_log, bias)):
        raise ValueError("Input rows must be contiguous")
    for tensor in (writes, keys, gates, u, factors):
        if not tensor[0].is_contiguous() or tensor.dtype != torch.float32:
            raise ValueError("FP32 contiguous ring/metadata tails required")
    if not phi.is_contiguous() or phi.dtype != torch.float32:
        raise ValueError("Packed metadata must be contiguous FP32")
    if state.dtype != torch.float32 or not out.is_contiguous():
        raise ValueError("FP32 state and contiguous output required")
    for tensor in (indices, positions, slot_map, offsets, widths):
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError("Contiguous int32 indices required")
    if plan is not None and (
        plan.max_width > g
        or plan.batch < batch
        or plan.key_heads != h
        or plan.value_heads != hv
        or plan.shared.device != mixed.device
    ):
        raise ValueError("Plan does not match inputs")
    with torch.accelerator.device_index(mixed.device.index):
        if plan is not None and plan.share_keys:
            _prepare_keys[(batch, h)](
                mixed,
                keys,
                indices,
                positions,
                plan.shared,
                h,
                hv,
                k,
                w,
                mixed.stride(0),
                keys.stride(0),
                scale,
                num_warps=4,
            )
        groups = (
            plan.groups
            if plan is not None
            else [
                ((sketch, triton.next_power_of_2(g), k), widths)
                for sketch in (True, False)
            ]
        )
        for (sketch, bg, bt), heads in groups:
            _step[(batch, heads.numel() if plan is not None else hv)](
                mixed,
                a,
                beta,
                a_log,
                bias,
                out,
                state,
                writes,
                keys,
                gates,
                indices,
                positions,
                slot_map,
                u,
                phi,
                offsets,
                widths,
                factors,
                heads,
                plan.shared if plan is not None else mixed,
                h,
                hv,
                k,
                v,
                w,
                g,
                bg,
                bt,
                plan is not None,
                plan is not None and plan.share_keys,
                scale,
                mixed.stride(0),
                a.stride(0),
                beta.stride(0),
                *state.stride(),
                writes.stride(0),
                keys.stride(0),
                gates.stride(0),
                u.stride(0),
                phi.stride(0),
                factors.stride(0),
                SKETCH=sketch,
                num_warps=4 if not sketch or bg > 16 else 1,
            )
