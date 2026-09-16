# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact parallel WY read: full-dimensional counterpart of projected SketchSSM.

Maintain pi_s in K dimensions and raw-write u_s in V dimensions. Each decode
computes the new factors and effective query by parallel ring reductions;
only flush materializes the state. Raw BF16 k/v and FP32 gate/beta are retained.
"""

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
    PrefixR,
    DR,
    FR,
    UR,
    Effective,
    ReplayOut,
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
    previous = tl.load(PrefixR + (base + pos - 1) * 128 + kk, mask=pos > 0, other=0.0)
    prefix = previous + log_a
    decay = tl.exp(prefix)
    if block == 0:
        tl.store(PrefixR + (base + pos) * 128 + kk, prefix)
        tl.store(DR + (base + pos) * 128 + kk, key * tl.exp(-prefix))
    t = tl.arange(0, 16)
    past_d = tl.load(
        DR + (base + t[:, None]) * 128 + kk[None, :],
        mask=t[:, None] < pos,
        other=0.0,
    )
    ell = past_d * decay[None, :]
    kk_inner = tl.sum(ell * key[None, :], axis=1)
    kq_inner = tl.sum(ell * query[None, :], axis=1)
    factors = tl.load(
        FR + (base + t[:, None]) * 128 + kk[None, :],
        mask=t[:, None] < pos,
        other=0.0,
    )
    current_factor = beta * (decay * key - tl.sum(factors * kk_inner[:, None], axis=0))
    current_kq = tl.sum(key * query)
    effective = (
        decay * query
        - tl.sum(factors * kq_inner[:, None], axis=0)
        - current_factor * current_kq
    )
    if block == 0:
        tl.store(FR + (base + pos) * 128 + kk, current_factor)
    writes = tl.load(
        UR + (base + t[:, None]) * 128 + vv[None, :],
        mask=t[:, None] < pos,
        other=0.0,
    )
    current_write = beta * (value - tl.sum(writes * kk_inner[:, None], axis=0))
    replay_out = tl.sum(writes * kq_inner[:, None], axis=0) + current_write * current_kq
    tl.store(UR + (base + pos) * 128 + vv, current_write)
    if pos == 15:
        return
    tl.store(Effective + (row * H + head) * 128 + kk, effective)
    tl.store(ReplayOut + (row * H + head) * 128 + vv, replay_out)


@triton.jit
def replay_read(
    Slots,
    Pos,
    State,
    FR,
    UR,
    DeltaR,
    Effective,
    ReplayOut,
    Out,
    H: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row)
    if slot < 0:
        return
    pos = tl.load(Pos + slot)
    if pos == 15:
        return
    kk = tl.arange(0, 128)
    vv = block * BV + tl.arange(0, BV)
    effective = tl.load(Effective + (row * H + head) * 128 + kk)
    replay_out = tl.load(ReplayOut + (row * H + head) * 128 + vv)
    state = tl.load(State + (slot * H + head) * 16384 + vv[:, None] * 128 + kk[None, :])
    base = (slot * H + head) * 16 + pos
    factor = tl.load(FR + base * 128 + kk)
    write = tl.load(UR + base * 128 + vv)
    delta = write - tl.sum(state * factor[None, :], axis=1)
    tl.store(DeltaR + base * 128 + vv, delta)
    output = tl.sum(state * effective[None, :], axis=1) + replay_out
    tl.store(Out + (row * H + head) * 128 + vv, output)


@triton.jit
def replay_flush(
    Q,
    Slots,
    State,
    PrefixR,
    DR,
    FR,
    UR,
    DeltaR,
    Out,
    WorkRows,
    WorkCounts,
    Capacity: tl.constexpr,
    H: tl.constexpr,
    BV: tl.constexpr,
):
    """Fuse exact WY window update and full-state output for flush rows only."""
    tiles: tl.constexpr = 128 // BV
    total = tl.load(WorkCounts + 1) * H * tiles
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + Capacity + item // (H * tiles))
        head = item // tiles % H
        v_start = item % tiles * BV
        slot = tl.load(Slots + row).to(tl.int64)
        k = tl.arange(0, 128)
        v = v_start + tl.arange(0, BV)
        sp = State + (slot * H + head) * 16384 + v[:, None] * 128 + k[None, :]
        state = tl.load(sp)
        t = tl.arange(0, 16)
        base = (slot * H + head) * 16
        prefix = tl.load(PrefixR + (base + 15) * 128 + k)
        decay = tl.exp(prefix)
        last_factor = tl.load(FR + (base + 15) * 128 + k)
        left = tl.load(DR + (base + t[:, None]) * 128 + k[None, :]) * decay[None, :]
        last_write = tl.load(UR + (base + 15) * 128 + v)
        last_delta = last_write - tl.sum(state * last_factor[None, :], axis=1)
        residual = tl.load(
            DeltaR + (base + t[None, :]) * 128 + v[:, None],
            mask=t[None, :] < 15,
            other=0.0,
        )
        residual = tl.where(t[None, :] == 15, last_delta[:, None], residual)
        state = state * decay[None, :] + tl.dot(
            residual, left, input_precision="tf32x3"
        )
        tl.store(sp, state)
        query = tl.load(Q + (row.to(tl.int64) * H + head) * 128 + k).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
        tl.store(
            Out + (row.to(tl.int64) * H + head) * 128 + v,
            tl.sum(state * query[None, :], axis=1),
        )
