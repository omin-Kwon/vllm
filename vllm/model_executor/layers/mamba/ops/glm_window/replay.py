# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact parallel WY read: full-dimensional counterpart of projected SketchSSM.

Contract the exact WY factors as delta_s = u_s - S0 @ pi_s. This yields
S_t = S0 * d_t + sum_s delta_s ell_s(t)^T. Each non-flush reads S0 once
for the current query and key; ring reductions remain parallel. Only flush
materializes state. Raw BF16 k/v and FP32 gate/beta are retained for handoff.
This contraction uses the full state, never a sketched/approximate state read.
"""

from vllm.triton_utils import gl, gluon, tl, triton


@gluon.jit
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
    KR,
    VR,
    GR,
    BR,
    PrefixR,
    DR,
    DeltaR,
    QueryScaled,
    KeyScaled,
    Rhs,
    ReplayOut,
    Scalars,
    H: gl.constexpr,
):
    head, row = gl.program_id(0), gl.program_id(1)
    slot = gl.load(Slots + row)
    if slot < 0:
        return
    pos = gl.load(Pos + slot)
    # One warp keeps the ring reductions in registers; vectorize along K/V.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    t = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    offset = (row * H + head) * 128
    raw_k = gl.load(K + offset + k).to(gl.float32)
    value = gl.load(V + offset + k).to(gl.float32)
    query = gl.load(Q + offset + k).to(gl.float32)
    query *= gl.rsqrt(gl.sum(query * query) + 1e-6) * 128**-0.5
    key = raw_k * gl.rsqrt(gl.sum(raw_k * raw_k) + 1e-6)
    raw_g = gl.load(Gate + offset + k).to(gl.float32)
    raw_g += gl.load(Bias + head * 128 + k)
    log_a = -5.0 / (1.0 + gl.exp(-gl.exp(gl.load(A + head)) * raw_g))
    beta = 1.0 / (1.0 + gl.exp(-gl.load(Beta + row * H + head).to(gl.float32)))
    base = (slot * H + head) * 16
    gl.store(KR + (base + pos) * 128 + k, raw_k)
    gl.store(VR + (base + pos) * 128 + k, value)
    gl.store(GR + (base + pos) * 128 + k, log_a)
    gl.store(BR + base + pos, beta)
    previous = gl.load(PrefixR + (slot * H + head) * 128 + k, mask=pos > 0, other=0.0)
    prefix = previous + log_a
    decay = gl.exp(prefix)
    gl.store(PrefixR + (slot * H + head) * 128 + k, prefix)
    gl.store(DR + (base + pos) * 128 + k, key * gl.exp(-prefix))
    past_d = gl.load(
        DR + (base + t[:, None]) * 128 + k[None, :], mask=t[:, None] < pos, other=0.0
    )
    ell = past_d * decay[None, :]
    kk_inner = gl.sum(ell * key[None, :], axis=1)
    kq_inner = gl.sum(ell * query[None, :], axis=1)
    delta = gl.load(
        DeltaR + (base + t[:, None]) * 128 + k[None, :],
        mask=t[:, None] < pos,
        other=0.0,
    )
    rhs = beta * (value - gl.sum(delta * kk_inner[:, None], axis=0))
    ring_out = gl.sum(delta * kq_inner[:, None], axis=0)
    gl.store(QueryScaled + offset + k, decay * query)
    gl.store(KeyScaled + offset + k, decay * key)
    gl.store(Rhs + offset + k, rhs)
    gl.store(ReplayOut + offset + k, ring_out)
    gl.store(Scalars + (row * H + head) * 2, beta)
    gl.store(Scalars + (row * H + head) * 2 + 1, gl.sum(key * query))


@gluon.jit
def replay_read(
    Slots,
    Pos,
    State,
    DeltaR,
    QueryScaled,
    KeyScaled,
    Rhs,
    ReplayOut,
    Scalars,
    Out,
    H: gl.constexpr,
    BV: gl.constexpr,
):
    block, head, row = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    packed: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    kp = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    vp = block * BV + gl.arange(0, BV, layout=packed)
    vv = block * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    slot = gl.load(Slots + row)
    offset = (row * H + head) * 128
    if slot < 0:
        gl.store(Out + offset + vp, 0.0)
        return
    pos = gl.load(Pos + slot)
    if pos == 15:
        return
    q = gl.load(QueryScaled + offset + kp)
    k = gl.load(KeyScaled + offset + kp)
    rhs = gl.load(Rhs + offset + vp)
    replay_out = gl.load(ReplayOut + offset + vp)
    beta = gl.load(Scalars + (row * H + head) * 2)
    kq = gl.load(Scalars + (row * H + head) * 2 + 1)
    state = gl.load(State + (slot * H + head) * 16384 + vv[:, None] * 128 + kp[None, :])
    projection_k = gl.convert_layout(gl.sum(state * k[None, :], axis=1), packed)
    projection_q = gl.convert_layout(gl.sum(state * q[None, :], axis=1), packed)
    delta = rhs - beta * projection_k
    base = (slot * H + head) * 16 + pos
    gl.store(DeltaR + base * 128 + vp, delta)
    gl.store(Out + offset + vp, projection_q + replay_out + delta * kq)


@triton.jit
def replay_flush(
    Q,
    Slots,
    State,
    PrefixR,
    DR,
    KeyScaled,
    Rhs,
    Scalars,
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
        base = (slot * H + head) * 16
        prefix = tl.load(PrefixR + (slot * H + head) * 128 + k)
        decay = tl.exp(prefix)
        last_key = tl.load(KeyScaled + (row * H + head) * 128 + k)
        last_rhs = tl.load(Rhs + (row * H + head) * 128 + v)
        beta = tl.load(Scalars + (row * H + head) * 2)
        last_delta = last_rhs - beta * tl.sum(state * last_key[None, :], axis=1)
        state *= decay[None, :]
        for i in tl.static_range(16):
            left = tl.load(DR + (base + i) * 128 + k) * decay
            if i == 15:  # noqa: SIM108 -- static branch avoids slot-15 loads
                delta = last_delta
            else:
                delta = tl.load(DeltaR + (base + i) * 128 + v)
            state += delta[:, None] * left[None, :]
        tl.store(sp, state)
        query = tl.load(Q + (row.to(tl.int64) * H + head) * 128 + k).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
        tl.store(
            Out + (row.to(tl.int64) * H + head) * 128 + v,
            tl.sum(state * query[None, :], axis=1),
        )
