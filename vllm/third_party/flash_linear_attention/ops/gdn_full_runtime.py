# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-compatible integration of full-coordinate GDN step and exact flush."""

import torch

from vllm.triton_utils import tl, triton

from .gdn_flush_full_cuda import FlushWorkspace
from .gdn_step_full_cuda import step


@triton.jit
def _flush_rows(
    POSITIONS, INDICES, ROWS, B: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    slot = tl.load(INDICES + i, i < B, 0)
    pos = tl.load(POSITIONS + i, i < B, 0)
    active = (i < B) & (slot > 0) & (pos == 15)
    offsets = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(ROWS + offsets, slot, active)
    tl.store(ROWS + CAP, tl.sum(active.to(tl.int32)))


class FullCoordinateRuntime:
    """One reusable workspace per layer; rows/count remain on the device."""

    def __init__(self, max_rows, h, hv, g, device):
        self.flush = FlushWorkspace(max_rows, h, hv, g, device)
        self.rows = torch.empty(max_rows + 1, dtype=torch.int32, device=device)
        self.max_rows = max_rows

    def refresh(self, state, indices, mapping, widths, u, phi):
        n = indices.numel()
        if n > self.max_rows:
            raise ValueError("prefill row count exceeds the allocated workspace")
        self.rows[:n].copy_(indices)
        self.rows[self.max_rows] = n
        self.flush.refresh_state(state, self.rows, mapping, widths, u, phi)

    def decode(
        self,
        mixed,
        a,
        b,
        a_log,
        bias,
        out,
        state,
        writes,
        keys,
        gates,
        indices,
        positions,
        u,
        phi,
        widths,
        factors,
        mapping,
        beta,
        scale,
    ):
        batch = mixed.shape[0]
        if batch > self.max_rows:
            raise ValueError("decode row count exceeds the allocated workspace")
        indices, positions = indices[:batch], positions[:batch]
        if batch == 0:
            return out, state
        # Decode uses (B,1,HV,V), mixed prefill/decode uses (1,B,HV,V).
        # Both are contiguous views of the same per-token output layout.
        step_out = out.view(batch, state.shape[1], 128)
        step(
            mixed,
            a,
            b,
            a_log,
            bias,
            step_out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions,
            u,
            phi,
            widths,
            factors,
            mapping,
            scale,
            beta_ring=beta,
        )
        _flush_rows[(1,)](
            positions,
            indices,
            self.rows,
            batch,
            self.max_rows,
            triton.next_power_of_2(batch),
            num_warps=4,
        )
        self.flush.flush(
            state, writes, keys, gates, self.rows, mapping, widths, beta, u, phi
        )
        return out, state
