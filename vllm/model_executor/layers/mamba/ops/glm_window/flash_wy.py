# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W16 preparation and BF16-operand recurrence inspired by pinned FlashKDA.

Raw rings already contain activated FP32 log-decay/beta: never activate twice.
This follows FlashKDA's storage/accumulation boundaries, not its CUDA instruction
schedule or bitwise gate/inverse implementation. P4/P6 metadata stays FP32.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def prepare_flash_wy(
    KR,
    GR,
    BR,
    Slots,
    Rows,
    Counts,
    Ranks,
    A,
    Restored,
    Inverse,
    Decay,
    H: tl.constexpr,
    Capacity: tl.constexpr,
):
    t = tl.arange(0, 16)
    k = tl.arange(0, 128)
    total = tl.load(Counts + 1) * H
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(Rows + Capacity + item // H)
        h = item % H
        if tl.load(Ranks + h) > 0:
            slot = tl.load(Slots + row).to(tl.int64)
            sh = slot * H + h
            ptr = (sh * 16 + t[:, None]) * 128 + k[None, :]
            key = tl.load(KR + ptr).to(tl.float32)
            key = (key * tl.rsqrt(tl.sum(key * key, 1) + 1e-6)[:, None]).to(tl.bfloat16)
            logs = tl.load(GR + ptr)
            prefix = tl.cumsum(logs * 1.4426950408889634, 0)
            decay = tl.exp2(tl.sum(tl.where(t[:, None] == 15, prefix, 0.0), 0))
            a = (
                key.to(tl.float32) * tl.exp2(prefix).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            c = (
                key.to(tl.float32) * tl.exp2(-prefix).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            restored = (
                c.to(tl.float32) * decay.to(tl.bfloat16).to(tl.float32)[None, :]
            ).to(tl.bfloat16)
            beta = tl.load(BR + sh * 16 + t)
            lower = tl.dot(a, tl.trans(c)) * beta[:, None]
            lower = tl.where(t[:, None] > t[None, :], lower, 0.0)
            diagonal = tl.where(t[:, None] // 8 == t[None, :] // 8, lower, 0.0)
            inv = (t[:, None] == t[None, :]).to(tl.float32)
            # Independent FP32 forward substitution for the two diagonal blocks.
            for i in tl.static_range(16):
                solved = tl.sum(tl.where(t[:, None] == i, inv, 0.0), 0)
                col = tl.sum(tl.where(t[None, :] == i, diagonal, 0.0), 1)
                inv -= col[:, None] * solved[None, :]
            off = tl.where((t[:, None] >= 8) & (t[None, :] < 8), lower, 0.0)
            inv_b = inv.to(tl.bfloat16)
            dc = tl.dot(inv_b, off.to(tl.bfloat16)).to(tl.bfloat16)
            merged = tl.dot(-dc, inv_b)
            inv_b = tl.where((t[:, None] >= 8) & (t[None, :] < 8), merged, inv).to(
                tl.bfloat16
            )
            tl.store(A + ptr, a)
            tl.store(Restored + ptr, restored)
            tl.store(Inverse + sh * 256 + t[:, None] * 16 + t[None, :], inv_b)
            tl.store(Decay + sh * 128 + k, decay)


@triton.jit
def flash_wy_update(
    raw,
    A,
    Restored,
    Inverse,
    Decay,
    VR,
    BR,
    slot,
    h,
    H: tl.constexpr,
    v_start=0,
    BV: tl.constexpr = 128,
):
    t = tl.arange(0, 16)
    k = tl.arange(0, 128)
    sh = slot * H + h
    ptr = (sh * 16 + t[:, None]) * 128 + k[None, :]
    a = tl.load(A + ptr)
    restored = tl.load(Restored + ptr)
    inv = tl.load(Inverse + sh * 256 + t[:, None] * 16 + t[None, :])
    v = v_start + tl.arange(0, BV)
    values = tl.load(VR + sh * 2048 + t[:, None] * 128 + v[None, :])
    beta = tl.load(BR + sh * 16 + t).to(tl.bfloat16).to(tl.float32)
    state = raw.to(tl.bfloat16)
    projection = tl.dot(state, tl.trans(a)).to(tl.bfloat16)
    residual = (tl.trans(values).to(tl.float32) - projection.to(tl.float32)).to(
        tl.bfloat16
    )
    residual = (residual.to(tl.float32) * beta[None, :]).to(tl.bfloat16)
    delta = tl.dot(residual, tl.trans(inv)).to(tl.bfloat16)
    update = tl.dot(delta, restored)
    decay = tl.load(Decay + sh * 128 + k)
    return (
        (state.to(tl.float32) * decay[None, :] + update).to(tl.bfloat16).to(tl.float32)
    )
