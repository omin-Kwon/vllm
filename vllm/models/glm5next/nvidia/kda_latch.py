# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental eager KDA cache adapter; raw state handoff to dense prefill."""

import json
from functools import lru_cache

import torch

from vllm.third_party.flash_linear_attention.ops.kda_latch import KDALatchState


@lru_cache(maxsize=4)
def _load_bases(path: str):
    return torch.load(path, map_location="cpu", weights_only=True)


class KDALatchCache:
    """Own one lazy latch window per physical scheduler state slot.

    Pure decode uses the latch. Batches containing prefill use the existing
    dense chunk kernel after exact materialization of any pending raw writes.
    Prefix caching, speculation and graphs must be disabled by the caller.
    """

    def __init__(self, omega: torch.Tensor):
        self.omega = omega
        self.slots: dict[int, KDALatchState] = {}
        self.decode_rows = 0
        self.dense_handoffs = 0

    @classmethod
    def from_config(cls, options, layer_idx, num_heads, tp_rank, tp_size):
        if options.get("allocation_path"):
            pack = _load_bases(options["basis_path"])
            with open(options["allocation_path"]) as stream:
                table = json.load(stream)
            omega = pack[layer_idx]
            ranks = torch.tensor(table["ranks"][str(layer_idx)], dtype=torch.long)
            if omega.shape != (num_heads, 128, 128) or ranks.shape != (num_heads,):
                raise ValueError("Invalid GLM per-head calibration pack")
            local = slice(
                tp_rank * (num_heads // tp_size), (tp_rank + 1) * (num_heads // tp_size)
            )
            ranks = ranks[local]
            width = max(1, int(ranks.max()))
            if options.get("graph"):
                from vllm.third_party.flash_linear_attention.ops import (
                    kda_latch_graph,
                )

                return kda_latch_graph.KDALatchGraphCache(
                    omega[local, :, :width].cuda(),
                    ranks.cuda(),
                    options.get("max_slots", 128),
                )
            return BatchedKDALatchCache(
                omega[local, :, :width], ranks, options.get("max_slots", 128)
            )
        rank = int(options.get("rank", 128))
        if not 1 <= rank <= 128:
            raise ValueError("KDA latch rank must be in [1, 128]")
        path = options.get("basis_path")
        if path:
            omega = _load_bases(path)[layer_idx]
            if omega.shape != (num_heads, 128, rank):
                raise ValueError("KDA basis must have shape (global heads, 128, rank)")
        elif rank == 128:
            omega = torch.eye(128).expand(num_heads, -1, -1)
        else:
            raise ValueError("Low-rank GLM evaluation requires a calibrated basis_path")
        local_heads = num_heads // tp_size
        return cls(omega[tp_rank * local_heads : (tp_rank + 1) * local_heads])

    @torch.no_grad()
    def before_prefill(self, state, indices, has_initial_state):
        ids = indices.reshape(-1).tolist()
        initial = has_initial_state.reshape(-1).tolist()
        if len(ids) != len(initial) or len(set(ids)) != len(ids):
            raise ValueError(
                "KDA prefill requires distinct slots and initial-state flags"
            )
        if any(slot <= 0 or slot >= len(state) for slot in ids):
            raise ValueError("KDA latch cannot use null or out-of-range state slots")
        for slot, use_initial in zip(ids, initial):
            latch = self.slots.pop(slot, None)
            if latch is not None and use_initial:
                latch.flush_pending(
                    torch.zeros(1, device=state.device, dtype=torch.int32)
                )
                state[slot].copy_(latch.state[0])
                self.dense_handoffs += 1

    @torch.no_grad()
    def step(self, state, indices, q, k, v, gate, beta, a_log, bias, lower_bound):
        ids = indices.reshape(-1).tolist()
        if len(ids) != len(q) or len(set(ids)) != len(ids):
            raise ValueError("KDA decode requires one token per distinct state slot")
        if any(slot <= 0 or slot >= len(state) for slot in ids):
            raise ValueError("KDA latch cannot use null or out-of-range state slots")
        out = torch.empty_like(v)
        for row, slot in enumerate(ids):
            if slot not in self.slots:
                self.slots[slot] = KDALatchState(
                    state[slot : slot + 1], self.omega.to(state.device)
                )
            latch = self.slots[slot]
            out[row : row + 1] = latch.step(
                q[row : row + 1],
                k[row : row + 1],
                v[row : row + 1],
                gate[row : row + 1],
                beta[row : row + 1],
                a_log,
                bias,
                lower_bound=lower_bound,
            )
            # The checkpoint is exact at window boundaries; pending writes
            # remain in the sidecar until a boundary or dense handoff.
            if latch.pos[0].item() == 0:
                state[slot].copy_(latch.state[0])
        self.decode_rows += len(ids)
        return out


class BatchedKDALatchCache:
    """Prefix-rank allocation with one batched pool indexed by physical slot.

    Padded columns are zero and dense heads retain the exact recurrence.
    Padding/FP64 refresh costs are not a realized traffic or speedup claim.
    """

    def __init__(self, omega, head_ranks, capacity=128):
        self.omega, self.head_ranks = omega, head_ranks
        self.pool = None
        self.live = set()
        self.capacity = capacity
        self.mapping = {}
        self.free = list(range(capacity - 1, -1, -1))
        self.decode_rows = self.dense_handoffs = 0

    def _ids(self, state, indices):
        ids = indices.reshape(-1).tolist()
        if len(set(ids)) != len(ids) or any(s <= 0 or s >= len(state) for s in ids):
            raise ValueError("KDA requires distinct non-null physical slots")
        return ids

    def _materialize(self, state, ids):
        assert self.pool is not None
        physical = torch.tensor(ids, device=state.device)
        slots = torch.tensor([self.mapping[s] for s in ids], device=state.device)
        self.pool.flush_pending(slots)
        state[physical] = self.pool.state[slots]
        self.dense_handoffs += len(ids)

    def _release(self, ids):
        for slot in ids:
            if slot in self.mapping:
                self.free.append(self.mapping.pop(slot))
                self.live.remove(slot)

    @torch.no_grad()
    def before_prefill(self, state, indices, has_initial_state):
        ids = self._ids(state, indices)
        initial = has_initial_state.reshape(-1).tolist()
        if len(ids) != len(initial):
            raise ValueError("Initial-state flags do not match slots")
        continuing = [s for s, flag in zip(ids, initial) if flag and s in self.live]
        if continuing:
            self._materialize(state, continuing)
        self._release(ids)

    @torch.no_grad()
    def step(self, state, indices, q, k, v, gate, beta, a_log, bias, lower_bound):
        ids = self._ids(state, indices)
        if len(ids) != len(q):
            raise ValueError("One decode token per slot is required")
        if len(ids) > self.capacity:
            raise ValueError("Decode batch exceeds configured latch pool capacity")
        if self.pool is None:
            self.pool = KDALatchState(
                torch.zeros(
                    (self.capacity, *state.shape[1:]),
                    device=state.device,
                    dtype=state.dtype,
                ),
                self.omega.to(state.device),
                head_ranks=self.head_ranks.to(state.device),
            )
        fresh = [s for s in ids if s not in self.live]
        if len(fresh) > len(self.free):
            victims = sorted(self.live - set(ids))[: len(fresh) - len(self.free)]
            self._materialize(state, victims)
            self._release(victims)
        if fresh:
            for physical in fresh:
                self.mapping[physical] = self.free.pop()
            slots = torch.tensor([self.mapping[s] for s in fresh], device=state.device)
            self.pool.reset(slots, state[fresh])
            self.live.update(fresh)
        slots = torch.tensor([self.mapping[s] for s in ids], device=state.device)
        out = self.pool.step(
            q, k, v, gate, beta, a_log, bias, slots=slots, lower_bound=lower_bound
        )
        ready = self.pool.pos[slots] == 0
        state[indices.reshape(-1)[ready]] = self.pool.state[slots[ready]]
        self.decode_rows += len(ids)
        return out
