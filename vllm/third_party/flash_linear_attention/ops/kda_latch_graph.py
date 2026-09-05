# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident latch cache and conditional refresh for CUDA graph decode."""

from types import SimpleNamespace

import torch

from vllm.third_party.flash_linear_attention.ops.kda_latch import (
    _flush,
    _step,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _resolve(
    IDs,
    Owners,
    Slots,
    Old,
    Fresh,
    B: tl.constexpr,
    P: tl.constexpr,
    WB: tl.constexpr,
    WP: tl.constexpr,
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
    free_order = tl.cumsum(available.to(tl.int32))
    choose = (ordinal[:, None] == free_order[None, :]) & available[None, :]
    chosen = tl.sum(tl.where(choose, pools[None, :], 0), 1)
    existing = tl.sum(tl.where(match, pools[None, :], 0), 1)
    slots = tl.where(fresh, chosen, existing)
    slots = tl.where(ids > 0, slots, -1)
    old = tl.load(Owners + slots, (slots >= 0) & fresh, other=-1)
    tl.store(Slots + rows, slots, rows < B)
    tl.store(Old + rows, old, rows < B)
    tl.store(Fresh + rows, fresh, rows < B)


@triton.jit
def _eviction_slots(Slots, Old, Evict, B: tl.constexpr, X: tl.constexpr):
    r = tl.arange(0, X)
    slot = tl.load(Slots + r, r < B, other=-1)
    old = tl.load(Old + r, r < B, other=-1)
    tl.store(Evict + r, tl.where(old > 0, slot, -1), r < B)


@triton.jit
def _copy_slots(
    IDs,
    Slots,
    Old,
    Fresh,
    Owners,
    Pos,
    State,
    Pool,
    H: tl.constexpr,
    SIZE: tl.constexpr,
    X: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    fresh = tl.load(Fresh + row)
    if (slot >= 0) & fresh:
        physical, old = tl.load(IDs + row), tl.load(Old + row)
        x = block * X + tl.arange(0, X)
        offset = (x // 16384) * S1 + ((x // 128) % 128) * S2 + (x % 128) * S3
        value = tl.load(Pool + slot * SIZE + x, x < SIZE, other=0.0)
        tl.store(State + old * S0 + offset, value, (old > 0) & (x < SIZE))
        value = tl.load(State + physical * S0 + offset, x < SIZE, other=0.0)
        tl.store(Pool + slot * SIZE + x, value, x < SIZE)
        if block == 0:
            tl.store(Owners + slot, physical)
            tl.store(Pos + slot, 0)


@triton.jit
def _refresh_u(
    Slots,
    Pos,
    State,
    Omega,
    Ranks,
    U,
    H: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row)
    rank = tl.load(Ranks + head)
    if slot >= 0:
        if (tl.load(Pos + slot) == 0) & (rank > 0):
            v = block * 32 + tl.arange(0, 32)
            k = tl.arange(0, 128)
            g = tl.arange(0, WG)
            h = tl.load(
                State + (slot * H + head) * 16384 + v[:, None] * 128 + k[None, :]
            )
            om = tl.load(
                Omega + head * 128 * G + k[:, None] * G + g[None, :],
                g[None, :] < rank,
                other=0.0,
            )
            u = tl.dot(h, om, input_precision="ieee")
            tl.store(
                U + (slot * H + head) * 128 * G + v[:, None] * G + g[None, :],
                u,
                g[None, :] < G,
            )


@triton.jit
def _refresh_gram(
    Slots,
    Pos,
    State,
    Omega,
    Ranks,
    U,
    Gram,
    Eta,
    H: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
    RIDGE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    rank = tl.load(Ranks + head)
    if slot >= 0:
        if (tl.load(Pos + slot) == 0) & (rank > 0):
            x = tl.arange(0, 128)
            g = tl.arange(0, WG)
            h = tl.load(
                State + (slot * H + head) * 16384 + x[:, None] * 128 + x[None, :]
            )
            eta = tl.maximum(RIDGE * tl.sum(tl.sum(h * h, 0), 0) / 128.0, 1.0e-12)
            u = tl.load(
                U + (slot * H + head) * 128 * G + x[:, None] * G + g[None, :],
                g[None, :] < G,
                other=0.0,
            )
            om = tl.load(
                Omega + head * 128 * G + x[:, None] * G + g[None, :],
                g[None, :] < rank,
                other=0.0,
            )
            gram = tl.dot(tl.trans(u), u, input_precision="ieee")
            gram += eta * tl.dot(tl.trans(om), om, input_precision="ieee")
            gram += tl.where(
                (g[:, None] == g[None, :]) & (g[:, None] >= rank), 1.0, 0.0
            )
            tl.store(
                Gram + (row * H + head) * WG * WG + g[:, None] * WG + g[None, :], gram
            )
            tl.store(Eta + row * H + head, eta)


@triton.jit
def _refresh_rhs(
    Slots,
    Pos,
    State,
    Omega,
    Ranks,
    U,
    RHS,
    Eta,
    H: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row)
    rank = tl.load(Ranks + head)
    if slot >= 0:
        if (tl.load(Pos + slot) == 0) & (rank > 0):
            k = block * 32 + tl.arange(0, 32)
            v = tl.arange(0, 128)
            g = tl.arange(0, WG)
            h = tl.load(
                State + (slot * H + head) * 16384 + v[:, None] * 128 + k[None, :]
            )
            u = tl.load(
                U + (slot * H + head) * 128 * G + v[:, None] * G + g[None, :],
                g[None, :] < G,
                other=0.0,
            )
            om = tl.load(
                Omega + head * 128 * G + k[:, None] * G + g[None, :],
                g[None, :] < rank,
                other=0.0,
            )
            eta = tl.load(Eta + row * H + head)
            rhs = tl.dot(tl.trans(h), u, input_precision="ieee") + eta * om
            tl.store(
                RHS + (row * H + head) * 128 * WG + k[:, None] * WG + g[None, :], rhs
            )


@triton.jit
def _solve_refresh(
    Slots,
    Pos,
    Ranks,
    Gram,
    RHS,
    Phi,
    H: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    rank = tl.load(Ranks + head)
    if slot >= 0:
        if (tl.load(Pos + slot) == 0) & (rank > 0):
            g = tl.arange(0, WG)
            x = tl.arange(0, 128)
            a = tl.load(
                Gram + (row * H + head) * WG * WG + g[:, None] * WG + g[None, :]
            )
            b = tl.load(
                RHS + (row * H + head) * 128 * WG + x[None, :] * WG + g[:, None]
            )
            for k in range(rank):
                ar = tl.sum(tl.where(g[:, None] == k, a, 0.0), 0)
                br = tl.sum(tl.where(g[:, None] == k, b, 0.0), 0)
                pivot = tl.sum(tl.where(g == k, ar, 0.0), 0)
                column = tl.sum(tl.where(g[None, :] == k, a, 0.0), 1)
                factor = tl.where(g == k, 0.0, column / pivot)
                a = tl.where(
                    g[:, None] == k,
                    ar[None, :] / pivot,
                    a - factor[:, None] * ar[None, :],
                )
                b = tl.where(
                    g[:, None] == k,
                    br[None, :] / pivot,
                    b - factor[:, None] * br[None, :],
                )
            tl.store(
                Phi + (slot * H + head) * 128 * G + x[None, :] * G + g[:, None],
                b,
                g[:, None] < G,
            )


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
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    if slot >= 0:
        pos = tl.load(Pos + slot)
        if pos == W - 1:
            physical = tl.load(IDs + row)
            x = block * X + tl.arange(0, X)
            offset = (x // 16384) * S1 + ((x // 128) % 128) * S2 + (x % 128) * S3
            value = tl.load(Pool + slot * SIZE + x, x < SIZE, other=0.0)
            tl.store(State + physical * S0 + offset, value, x < SIZE)


@triton.jit
def _bump(Slots, Pos, B: tl.constexpr, W: tl.constexpr, X: tl.constexpr):
    r = tl.arange(0, X)
    slot = tl.load(Slots + r, r < B, other=-1)
    pos = tl.load(Pos + slot, slot >= 0, other=0)
    tl.store(Pos + slot, (pos + 1) % W, slot >= 0)


class KDALatchGraphCache:
    """Fixed-capacity GPU slot ownership; no host reads in decode."""

    def __init__(self, omega, ranks, capacity=128):
        device = omega.device
        self.capacity, self.heads, self.rank = capacity, len(ranks), omega.shape[-1]
        if omega.shape != (self.heads, 128, self.rank) or not 1 <= self.rank <= 128:
            raise ValueError("Expected a per-head 128-by-G basis")
        self.ranks = ranks.to(device=device, dtype=torch.int32)
        columns = torch.arange(self.rank, device=device)
        mask = columns[None, :] < self.ranks[:, None]
        self.basis = omega.float().contiguous()
        self.omega = (self.basis * mask[:, None, :]).contiguous()
        options = dict(device=device, dtype=torch.float32)
        ring = (capacity, self.heads, 16, 128)
        self.pool = SimpleNamespace(
            state=torch.zeros(capacity, self.heads, 128, 128, **options),
            pos=torch.zeros(capacity, device=device, dtype=torch.int32),
            latch_heads=self.ranks > 0,
            k=torch.zeros(ring, **options),
            v=torch.zeros(ring, **options),
            log_a=torch.zeros(ring, **options),
            beta=torch.zeros(capacity, self.heads, 16, **options),
            prefix=torch.zeros(ring, **options),
            f=torch.zeros(capacity, self.heads, 16, self.rank, **options),
            u=torch.zeros(ring, **options),
            latch=torch.zeros(capacity, self.heads, 128, self.rank, **options),
            phi=torch.zeros(capacity, self.heads, 128, self.rank, **options),
        )
        self.counts = torch.zeros(2, device=device, dtype=torch.int64)
        self.owners = torch.full((capacity,), -1, device=device, dtype=torch.int32)
        self.slots = torch.empty(capacity, device=device, dtype=torch.int32)
        self.old = torch.empty_like(self.slots)
        self.evict = torch.empty_like(self.slots)
        self.fresh = torch.empty(capacity, device=device, dtype=torch.bool)
        wg = max(16, triton.next_power_of_2(self.rank))
        self.gram = torch.empty(
            capacity, self.heads, wg, wg, device=device, dtype=torch.float32
        )
        self.rhs = torch.empty(
            capacity, self.heads, 128, wg, device=device, dtype=torch.float32
        )
        self.eta = torch.empty(capacity, self.heads, device=device, dtype=torch.float32)

    @property
    def decode_rows(self):
        return int(self.counts[0].item())

    @property
    def dense_handoffs(self):
        return int(self.counts[1].item())

    def before_prefill(self, state, indices, has_initial_state):
        batch = indices.numel()
        # Prefill batches can contain more requests than the decode pool.
        slots = torch.empty_like(indices, dtype=torch.int32)
        flush = torch.empty_like(slots)
        _prefill_resolve[(1,)](
            indices,
            has_initial_state,
            self.owners,
            slots,
            flush,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
            num_warps=4,
        )
        _track[(1,)](flush, self.counts, batch, 1, triton.next_power_of_2(batch))
        self._flush(flush, partial=True)
        _prefill_handoff[(batch, triton.cdiv(self.heads * 16384, 1024))](
            indices,
            has_initial_state,
            slots,
            self.owners,
            state,
            self.pool.state,
            self.heads * 16384,
            1024,
            *state.stride(),
        )

    def _flush(self, slots, partial=False):
        p = self.pool
        _flush[(slots.numel(), self.heads, 16)](
            slots,
            p.pos,
            p.state,
            p.k,
            p.v,
            p.log_a,
            p.beta,
            p.latch_heads,
            self.heads,
            128,
            128,
            16,
            8,
            PARTIAL=partial,
            num_warps=1,
        )

    def refresh(self, slots):
        wg = max(16, triton.next_power_of_2(self.rank))
        p = self.pool
        _refresh_u[(slots.numel(), self.heads, 4)](
            slots,
            p.pos,
            p.state,
            self.omega,
            self.ranks,
            p.latch,
            self.heads,
            self.rank,
            wg,
            num_warps=4,
            num_stages=1,
        )
        _refresh_gram[(slots.numel(), self.heads)](
            slots,
            p.pos,
            p.state,
            self.omega,
            self.ranks,
            p.latch,
            self.gram,
            self.eta,
            self.heads,
            self.rank,
            wg,
            1e-4,
            num_warps=4,
            num_stages=1,
        )
        _refresh_rhs[(slots.numel(), self.heads, 4)](
            slots,
            p.pos,
            p.state,
            self.omega,
            self.ranks,
            p.latch,
            self.rhs,
            self.eta,
            self.heads,
            self.rank,
            wg,
            num_warps=4,
            num_stages=1,
        )
        _solve_refresh[(slots.numel(), self.heads)](
            slots,
            self.pool.pos,
            self.ranks,
            self.gram,
            self.rhs,
            self.pool.phi,
            self.heads,
            self.rank,
            wg,
            num_warps=4,
            num_stages=1,
        )

    def step(self, state, indices, q, k, v, gate, beta, a_log, bias, lower_bound=-5.0):
        q, k, v, gate, beta = (tensor.contiguous() for tensor in (q, k, v, gate, beta))
        a_log, bias = a_log.float().contiguous(), bias.float().contiguous()
        batch, heads = q.shape[:2]
        if batch > self.capacity:
            raise ValueError("Decode batch exceeds latch capacity")
        slots = self.slots[:batch]
        _resolve[(1,)](
            indices,
            self.owners,
            slots,
            self.old,
            self.fresh,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
            num_warps=4,
        )
        _eviction_slots[(1,)](
            slots, self.old, self.evict, batch, triton.next_power_of_2(batch)
        )
        self._flush(self.evict[:batch], partial=True)
        _copy_slots[(batch, triton.cdiv(heads * 16384, 1024))](
            indices,
            slots,
            self.old,
            self.fresh,
            self.owners,
            self.pool.pos,
            state,
            self.pool.state,
            heads,
            heads * 16384,
            1024,
            *state.stride(),
        )
        self.refresh(slots)
        p = self.pool
        out = torch.zeros_like(v)
        _step[(batch, heads)](
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            bias,
            slots,
            p.pos,
            p.state,
            p.latch,
            p.phi,
            p.latch_heads,
            p.k,
            p.v,
            p.log_a,
            p.beta,
            p.prefix,
            p.f,
            p.u,
            out,
            heads,
            128,
            128,
            self.rank,
            triton.next_power_of_2(self.rank),
            16,
            True,
            lower_bound,
            128**-0.5,
            True,
            num_warps=4,
            num_stages=1,
        )
        self._flush(slots)
        _advance[(batch, triton.cdiv(heads * 16384, 1024))](
            indices,
            slots,
            p.pos,
            state,
            p.state,
            heads,
            heads * 16384,
            16,
            1024,
            *state.stride(),
        )
        _bump[(1,)](slots, p.pos, batch, 16, triton.next_power_of_2(batch))
        _track[(1,)](slots, self.counts, batch, 0, triton.next_power_of_2(batch))
        return out


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
):
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row)
    if slot >= 0:
        if tl.load(Initial + row):
            physical = tl.load(IDs + row)
            x = block * X + tl.arange(0, X)
            offset = (x // 16384) * S1 + ((x // 128) % 128) * S2 + (x % 128) * S3
            value = tl.load(Pool + slot * SIZE + x, x < SIZE, other=0.0)
            tl.store(State + physical * S0 + offset, value, x < SIZE)
        if block == 0:
            tl.store(Owners + slot, -1)


@triton.jit
def _track(Slots, Counts, B: tl.constexpr, INDEX: tl.constexpr, X: tl.constexpr):
    row = tl.arange(0, X)
    slot = tl.load(Slots + row, row < B, other=-1)
    count = tl.sum((slot >= 0).to(tl.int64), 0)
    tl.store(Counts + INDEX, tl.load(Counts + INDEX) + count)
