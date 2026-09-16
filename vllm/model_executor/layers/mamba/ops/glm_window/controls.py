# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


@triton.jit
def _resolve_work(
    IDs,
    Owners,
    Slots,
    Old,
    Fresh,
    B: tl.constexpr,
    P: tl.constexpr,
    WB: tl.constexpr,
    WP: tl.constexpr,
    Pos,
    OldPos,
    Counts,
    WorkRows,
    WorkCounts,
):
    rows = tl.arange(0, WB)
    pools = tl.arange(0, WP)
    ids = tl.load(IDs + rows, rows < B, other=-1)
    owners = tl.load(Owners + pools, pools < P, other=-2)
    match = (ids[:, None] == owners[None, :]) & (ids[:, None] > 0)
    found = tl.sum(match.to(tl.int32), 1) > 0
    available = (tl.sum(match.to(tl.int32), 0) == 0) & (pools < P)
    fresh = ~found & (ids > 0) & (rows < B)
    ordinal = tl.cumsum(fresh.to(tl.int32))
    empty = available & (owners < 0)
    occupied = available & (owners >= 0)
    free_order = tl.where(
        empty,
        tl.cumsum(empty.to(tl.int32)),
        tl.sum(empty.to(tl.int32)) + tl.cumsum(occupied.to(tl.int32)),
    )
    choose = (ordinal[:, None] == free_order[None, :]) & available[None, :]
    chosen = tl.sum(tl.where(choose, pools[None, :], 0), 1)
    existing = tl.sum(tl.where(match, pools[None, :], 0), 1)
    slots = tl.where(fresh, chosen, existing)
    slots = tl.where(ids > 0, slots, -1)
    old = tl.load(Owners + slots, (slots >= 0) & fresh, other=-1)
    tl.store(Slots + rows, slots, rows < B)
    tl.store(Old + rows, old, rows < B)
    tl.store(Fresh + rows, fresh, rows < B)

    position = tl.load(Pos + slots, (slots >= 0) & fresh & (old > 0), other=0)
    tl.store(OldPos + rows, position, rows < B)
    count = tl.sum(((slots >= 0) & (rows < B)).to(tl.int64), 0)
    tl.store(Counts, tl.load(Counts) + count)

    fresh_order = tl.cumsum(fresh.to(tl.int32))
    tl.store(WorkRows + fresh_order - 1, rows, fresh)
    current_pos = tl.load(Pos + slots, slots >= 0, other=0)
    flush = (slots >= 0) & (~fresh) & (current_pos == 15) & (rows < B)
    flush_order = tl.cumsum(flush.to(tl.int32))
    tl.store(WorkRows + P + flush_order - 1, rows, flush)
    tl.store(WorkCounts, tl.sum(fresh.to(tl.int32)))
    tl.store(WorkCounts + 1, tl.sum(flush.to(tl.int32)))


@triton.jit
def _acquire_row(
    row,
    h,
    IDs,
    Slots,
    Old,
    OldPos,
    Fresh,
    Owners,
    Pos,
    State,
    Pool,
    KR,
    VR,
    GR,
    BR,
    Ranks,
    H: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    DIRECT_STATE: tl.constexpr = False,
):
    slot = tl.load(Slots + row)
    if slot < 0:
        return
    if not tl.load(Fresh + row):
        return
    physical = tl.load(IDs + row).to(tl.int64)
    old = tl.load(Old + row).to(tl.int64)
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    offset = h * S1 + v[:, None] * S2 + k[None, :] * S3
    pool = Pool + (slot * H + h) * 16384 + v[:, None] * 128 + k[None, :]
    if old > 0:
        if DIRECT_STATE:  # noqa: SIM108 -- constexpr excludes the unused pointer
            raw = tl.load(State + old * S0 + offset)
        else:
            raw = tl.load(pool)
        if tl.load(Ranks + h) > 0:
            count = tl.load(OldPos + row)
            ring = (slot * H + h) * 16
            for t in range(count):
                key = tl.load(KR + (ring + t) * 128 + k).to(tl.float32)
                key *= tl.rsqrt(tl.sum(key * key) + 1e-6)
                value = tl.load(VR + (ring + t) * 128 + v).to(tl.float32)
                decay = tl.load(GR + (ring + t) * 128 + k)
                beta = tl.load(BR + ring + t)
                raw *= tl.exp(decay[None, :])
                delta = beta * (value - tl.sum(raw * key[None, :], axis=1))
                raw += delta[:, None] * key[None, :]
        tl.store(State + old * S0 + offset, raw)
    if not DIRECT_STATE:
        raw = tl.load(State + physical * S0 + offset)
        tl.store(pool, raw)
    if h == 0:
        tl.store(Owners + slot, physical)
        tl.store(Pos + slot, 0)


@triton.jit
def _acquire(
    IDs,
    Slots,
    Old,
    OldPos,
    Fresh,
    Owners,
    Pos,
    State,
    Pool,
    KR,
    VR,
    GR,
    BR,
    Ranks,
    H: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    _acquire_row(
        tl.program_id(0),
        tl.program_id(1),
        IDs,
        Slots,
        Old,
        OldPos,
        Fresh,
        Owners,
        Pos,
        State,
        Pool,
        KR,
        VR,
        GR,
        BR,
        Ranks,
        H,
        S0,
        S1,
        S2,
        S3,
    )


@triton.jit
def _acquire_work(
    IDs,
    Slots,
    Old,
    OldPos,
    Fresh,
    Owners,
    Pos,
    State,
    Pool,
    KR,
    VR,
    GR,
    BR,
    Ranks,
    H: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    WorkRows,
    WorkCounts,
    Capacity: tl.constexpr,
    DIRECT_STATE: tl.constexpr = False,
):
    total = tl.load(WorkCounts) * H
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + item // H)
        h = item % H
        _acquire_row(
            row,
            h,
            IDs,
            Slots,
            Old,
            OldPos,
            Fresh,
            Owners,
            Pos,
            State,
            Pool,
            KR,
            VR,
            GR,
            BR,
            Ranks,
            H,
            S0,
            S1,
            S2,
            S3,
            DIRECT_STATE,
        )


@triton.jit
def _advance_row(
    row,
    block,
    IDs,
    Slots,
    Pos,
    State,
    Pool,
    H: tl.constexpr,
    SIZE: tl.constexpr,
    W: tl.constexpr,
    X: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    slot = tl.load(Slots + row)
    if slot >= 0:
        pos = tl.load(Pos + slot)
        if pos == W - 1:
            physical = tl.load(IDs + row).to(tl.int64)
            x = block * X + tl.arange(0, X)
            offset = (x // 16384) * S1 + ((x // 128) % 128) * S2 + (x % 128) * S3
            value = tl.load(Pool + slot * SIZE + x, x < SIZE, other=0.0)
            tl.store(State + physical * S0 + offset, value, x < SIZE)


@triton.jit
def _advance(
    IDs,
    Slots,
    Pos,
    State,
    Pool,
    H: tl.constexpr,
    SIZE: tl.constexpr,
    W: tl.constexpr,
    X: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    _advance_row(
        tl.program_id(0),
        tl.program_id(1),
        IDs,
        Slots,
        Pos,
        State,
        Pool,
        H,
        SIZE,
        W,
        X,
        S0,
        S1,
        S2,
        S3,
    )


@triton.jit
def _advance_work(
    IDs,
    Slots,
    Pos,
    State,
    Pool,
    H: tl.constexpr,
    SIZE: tl.constexpr,
    W: tl.constexpr,
    X: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    WorkRows,
    WorkCounts,
    Capacity: tl.constexpr,
):
    blocks: tl.constexpr = triton.cdiv(SIZE, X)
    total = tl.load(WorkCounts + 1) * blocks
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + Capacity + item // blocks)
        block = item % blocks
        _advance_row(
            row, block, IDs, Slots, Pos, State, Pool, H, SIZE, W, X, S0, S1, S2, S3
        )


@triton.jit
def _bump(Slots, Pos, B: tl.constexpr, W: tl.constexpr, X: tl.constexpr):
    r = tl.arange(0, X)
    slot = tl.load(Slots + r, r < B, other=-1)
    pos = tl.load(Pos + slot, slot >= 0, other=0)
    tl.store(Pos + slot, (pos + 1) % W, slot >= 0)


@triton.jit
def _prefill_resolve(
    IDs,
    Initial,
    Owners,
    Slots,
    FlushSlots,
    B: tl.constexpr,
    P: tl.constexpr,
    WB: tl.constexpr,
    WP: tl.constexpr,
):
    rows, pools = tl.arange(0, WB), tl.arange(0, WP)
    ids = tl.load(IDs + rows, rows < B, other=-1)
    flags = tl.load(Initial + rows, rows < B, other=False)
    owner = tl.load(Owners + pools, pools < P, other=-2)
    match = (ids[:, None] == owner[None, :]) & (ids[:, None] > 0)
    exists = tl.sum(match.to(tl.int32), 1) > 0
    slot = tl.sum(tl.where(match, pools[None, :], 0), 1)
    tl.store(Slots + rows, tl.where(exists, slot, -1), rows < B)
    tl.store(FlushSlots + rows, tl.where(exists & flags, slot, -1), rows < B)


@triton.jit
def _prefill_handoff(
    IDs,
    Initial,
    Slots,
    Owners,
    State,
    Pool,
    SIZE: tl.constexpr,
    X: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    DIRECT_STATE: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    if slot >= 0:
        if not DIRECT_STATE and tl.load(Initial + row):
            physical = tl.load(IDs + row).to(tl.int64)
            x = block * X + tl.arange(0, X)
            offset = (x // 16384) * S1 + ((x // 128) % 128) * S2 + (x % 128) * S3
            value = tl.load(Pool + slot * SIZE + x, x < SIZE, other=0.0)
            tl.store(State + physical * S0 + offset, value, x < SIZE)
        if block == 0:
            tl.store(Owners + slot, -1)


@triton.jit
def _flush(
    Slots,
    Pos,
    State,
    KR,
    VR,
    GR,
    BR,
    LatchHeads,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    BV: tl.constexpr,
    PARTIAL: tl.constexpr = False,
    RAW_K_RING: tl.constexpr = False,
    DIRECT_STATE: tl.constexpr = False,
    Owners=None,
    S0: tl.constexpr = 0,
    S1: tl.constexpr = 0,
    S2: tl.constexpr = 0,
    S3: tl.constexpr = 0,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    slot = tl.load(Slots + row)
    if slot >= 0:
        pos = tl.load(Pos + slot)
        count = pos if PARTIAL else W
        ready = pos > 0 if PARTIAL else pos == W - 1
        if ready & tl.load(LatchHeads + head):
            kk = tl.arange(0, K)
            vv = block * BV + tl.arange(0, BV)
            if DIRECT_STATE:
                physical = tl.load(Owners + slot).to(tl.int64)
                sp = (
                    State
                    + physical * S0
                    + head * S1
                    + vv[:, None] * S2
                    + kk[None, :] * S3
                )
            else:
                sp = State + (slot * H + head) * V * K + vv[:, None] * K + kk[None, :]
            state = tl.load(sp, mask=vv[:, None] < V, other=0.0)
            base = (slot * H + head) * W
            for t in range(count):
                k = tl.load(KR + (base + t) * K + kk).to(tl.float32)
                if RAW_K_RING:
                    k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
                v = tl.load(VR + (base + t) * V + vv, mask=vv < V, other=0.0).to(
                    tl.float32
                )
                log_a = tl.load(GR + (base + t) * K + kk)
                beta = tl.load(BR + base + t)
                state *= tl.exp(log_a[None, :])
                delta = beta * (v - tl.sum(state * k[None, :], axis=1))
                state += delta[:, None] * k[None, :]
            tl.store(sp, state, mask=vv[:, None] < V)
