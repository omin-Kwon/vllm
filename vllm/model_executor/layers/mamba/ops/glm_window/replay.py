# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact raw-write replay; no sketch or coefficient approximation."""

from vllm.triton_utils import tl, triton


@triton.jit
def replay_step(
    Q,
    K,
    V,
    Gate,
    Beta,
    A,
    Bias,
    Slots,
    Pos,
    State,
    KR,
    VR,
    GR,
    BR,
    Out,
    H: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row)
    if slot < 0:
        return
    pos = tl.load(Pos + slot)
    kk = tl.arange(0, 128)
    vv = block * BV + tl.arange(0, BV)
    raw_k = tl.load(K + (row * H + head) * 128 + kk).to(tl.float32)
    value = tl.load(V + (row * H + head) * 128 + vv).to(tl.float32)
    query = tl.load(Q + (row * H + head) * 128 + kk).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
    key = raw_k * tl.rsqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
    raw_g = tl.load(Gate + (row * H + head) * 128 + kk).to(tl.float32)
    raw_g += tl.load(Bias + head * 128 + kk)
    log_a = -5.0 / (1.0 + tl.exp(-tl.exp(tl.load(A + head)) * raw_g))
    beta = tl.sigmoid(tl.load(Beta + row * H + head).to(tl.float32))
    base = (slot * H + head) * 16
    if block == 0:
        tl.store(KR + (base + pos) * 128 + kk, raw_k)
        tl.store(GR + (base + pos) * 128 + kk, log_a)
        tl.store(BR + base + pos, beta)
    tl.store(VR + (base + pos) * 128 + vv, value)
    sp = State + (slot * H + head) * 16384 + vv[:, None] * 128 + kk[None, :]
    state = tl.load(sp)
    for t in range(pos):
        past_k = tl.load(KR + (base + t) * 128 + kk).to(tl.float32)
        past_k *= tl.rsqrt(tl.sum(past_k * past_k) + 1.0e-6)
        past_v = tl.load(VR + (base + t) * 128 + vv).to(tl.float32)
        past_g = tl.load(GR + (base + t) * 128 + kk)
        past_b = tl.load(BR + base + t)
        state *= tl.exp(past_g[None, :])
        delta = past_b * (past_v - tl.sum(state * past_k[None, :], axis=1))
        state += delta[:, None] * past_k[None, :]
    state *= tl.exp(log_a[None, :])
    delta = beta * (value - tl.sum(state * key[None, :], axis=1))
    state += delta[:, None] * key[None, :]
    tl.store(Out + (row * H + head) * 128 + vv, tl.sum(state * query[None, :], axis=1))
    if pos == 15:
        tl.store(sp, state)
