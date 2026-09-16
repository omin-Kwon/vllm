# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared GPU ownership and native-coordinate state for Replay and Sketch."""

from types import SimpleNamespace

import torch

from vllm.triton_utils import triton

from .controls import (
    _acquire,
    _acquire_work,
    _advance,
    _bump,
    _flush,
    _prefill_handoff,
    _prefill_resolve,
    _resolve_work,
)
from .replay import replay_flush, replay_read, replay_step


class ReplayCache:
    """Keep raw updates until W16; materialize before native prefill handoff.

    Exact Replay reads and updates the native page directly. Sketch retains a
    private checkpoint pool. Both retain the same slot ownership and raw rings.

    Physical page zero is padding. Live page IDs in a batch must be unique.
    The runner must invalidate finished/preempted owners before page reuse.
    """

    def __init__(
        self,
        heads: int,
        capacity: int,
        device: torch.device,
        *,
        replay_factors: bool = True,
    ):
        if heads < 1 or capacity < 1:
            raise ValueError("Positive head count and slot capacity required")
        self.heads, self.capacity = heads, capacity
        self.replay_factors = replay_factors
        fp = dict(device=device, dtype=torch.float32)
        ip = dict(device=device, dtype=torch.int32)
        ring = (capacity, heads, 16, 128)
        self.pool = SimpleNamespace(
            state=(
                torch.empty(0, **fp)
                if replay_factors
                else torch.zeros(capacity, heads, 128, 128, **fp)
            ),
            pos=torch.zeros(capacity, **ip),
            k=torch.zeros(ring, device=device, dtype=torch.bfloat16),
            v=torch.zeros(ring, device=device, dtype=torch.bfloat16),
            log_a=torch.zeros(ring, **fp),
            beta=torch.zeros(capacity, heads, 16, **fp),
            latch_heads=torch.ones(heads, device=device, dtype=torch.bool),
        )
        if replay_factors:
            self.pool.prefix = torch.zeros(capacity, heads, 128, **fp)
            self.pool.direct_decay = torch.zeros(ring, **fp)
            self.pool.delta = torch.zeros(ring, **fp)
            self.query_scaled = torch.empty(capacity, heads, 128, **fp)
            self.key_scaled = torch.empty_like(self.query_scaled)
            self.rhs = torch.empty_like(self.query_scaled)
            self.replay_out = torch.empty_like(self.query_scaled)
            self.scalars = torch.empty(capacity, heads, 2, **fp)
        self.ranks = torch.ones(heads, **ip)
        self.owners = torch.full((capacity,), -1, **ip)
        self.slots = torch.empty(capacity, **ip)
        self.old = torch.empty_like(self.slots)
        self.old_pos = torch.empty_like(self.slots)
        self.fresh = torch.empty(capacity, device=device, dtype=torch.bool)
        self.counts = torch.zeros(1, device=device, dtype=torch.int64)
        self.work_rows = torch.empty(2 * capacity, **ip)
        self.work_counts = torch.zeros(2, **ip)
        self.mixed_decode_tokens = 0
        self.flush_programs = min(
            (8 if replay_factors else 4)
            * torch.cuda.get_device_properties(device).multi_processor_count,
            capacity * heads,
        )

    def reset(self):
        """Discard capture/warmup ownership without writing physical pages."""
        self.owners.fill_(-1)
        self.pool.pos.zero_()
        self.counts.zero_()
        self.mixed_decode_tokens = 0

    def release_finished(self, physical_ids: torch.Tensor):
        if physical_ids.numel():
            released = (self.owners[:, None] == physical_ids[None, :]).any(1)
            self.owners.masked_fill_(released, -1)
            self.pool.pos.masked_fill_(released, 0)

    def before_prefill(self, state, indices, has_initial_state):
        """Commit continuing rows, discard stale ownership for new prompts."""
        batch = indices.numel()
        if not batch:
            return
        slots = torch.empty_like(indices, dtype=torch.int32)
        flush_slots = torch.empty_like(slots)
        _prefill_resolve[(1,)](
            indices,
            has_initial_state,
            self.owners,
            slots,
            flush_slots,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
        )
        p = self.pool
        _flush[(batch, self.heads, 4)](
            flush_slots,
            p.pos,
            state if self.replay_factors else p.state,
            p.k,
            p.v,
            p.log_a,
            p.beta,
            p.latch_heads,
            self.heads,
            128,
            128,
            16,
            32,
            PARTIAL=True,
            RAW_K_RING=True,
            DIRECT_STATE=self.replay_factors,
            Owners=self.owners,
            S0=state.stride(0),
            S1=state.stride(1),
            S2=state.stride(2),
            S3=state.stride(3),
            num_warps=4,
        )
        _prefill_handoff[(batch, self.heads)](
            indices,
            has_initial_state,
            slots,
            self.owners,
            state,
            state if self.replay_factors else p.state,
            self.heads * 16384,
            16384,
            *state.stride(),
            DIRECT_STATE=self.replay_factors,
            num_warps=4,
        )

    def _decode(self, state, indices, slots, q, k, v, gate, beta, a_log, bias):
        p = self.pool
        out = torch.empty_like(v)
        replay_step[(self.heads, q.shape[0])](
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            bias,
            slots,
            p.pos,
            p.k,
            p.v,
            p.log_a,
            p.beta,
            p.prefix,
            p.direct_decay,
            p.delta,
            self.query_scaled,
            self.key_scaled,
            self.rhs,
            self.replay_out,
            self.scalars,
            self.heads,
            num_warps=1,
        )
        replay_read[(4, self.heads, q.shape[0])](
            indices,
            slots,
            p.pos,
            state,
            p.delta,
            self.query_scaled,
            self.key_scaled,
            self.rhs,
            self.replay_out,
            self.scalars,
            out,
            self.heads,
            32,
            *state.stride(),
            num_warps=4,
        )
        replay_flush[(self.flush_programs,)](
            indices,
            q,
            slots,
            state,
            p.prefix,
            p.direct_decay,
            self.key_scaled,
            self.rhs,
            self.scalars,
            p.delta,
            out,
            self.work_rows,
            self.work_counts,
            self.capacity,
            self.heads,
            32,
            num_warps=1,
            num_stages=1,
            S0=state.stride(0),
            S1=state.stride(1),
            S2=state.stride(2),
            S3=state.stride(3),
        )
        return out

    def step(self, state, indices, q, k, v, gate, beta, a_log, bias, lower_bound=-5.0):
        if lower_bound != -5.0:
            raise ValueError("GLM window cache requires bounded gate -5 and W16")
        if state.dtype != torch.float32 or state.shape[1:] != (self.heads, 128, 128):
            raise ValueError("Native FP32 [page, head, V128, K128] state required")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("Raw post-convolution k/v must be BF16")
        batch = q.shape[0]
        if batch > self.capacity:
            raise ValueError("Decode batch exceeds configured slot capacity")
        if not batch:
            return torch.empty_like(v)
        q, k, v, gate, beta = (x.contiguous() for x in (q, k, v, gate, beta))
        a_log, bias = a_log.float().contiguous(), bias.float().contiguous()
        slots, p = self.slots[:batch], self.pool
        _resolve_work[(1,)](
            indices,
            self.owners,
            slots,
            self.old,
            self.fresh,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
            p.pos,
            self.old_pos,
            self.counts,
            self.work_rows,
            self.work_counts,
            num_warps=4,
        )
        acquire_args = (
            indices,
            slots,
            self.old,
            self.old_pos,
            self.fresh,
            self.owners,
            p.pos,
            state,
            state if self.replay_factors else p.state,
            p.k,
            p.v,
            p.log_a,
            p.beta,
            self.ranks,
            self.heads,
            *state.stride(),
        )
        if self.replay_factors:
            _acquire_work[(self.flush_programs,)](
                *acquire_args,
                self.work_rows,
                self.work_counts,
                self.capacity,
                DIRECT_STATE=True,
                num_warps=4,
            )
        else:
            _acquire[(batch, self.heads)](*acquire_args, num_warps=4)
        out = self._decode(state, indices, slots, q, k, v, gate, beta, a_log, bias)
        if not self.replay_factors:
            _advance[(batch, self.heads)](
                indices,
                slots,
                p.pos,
                state,
                p.state,
                self.heads,
                self.heads * 16384,
                16,
                16384,
                *state.stride(),
                num_warps=4,
            )
        _bump[(1,)](slots, p.pos, batch, 16, triton.next_power_of_2(batch))
        return out
