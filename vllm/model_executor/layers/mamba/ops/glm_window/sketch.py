# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-Z P4/P6 reads over Full-Gram allocated, native-coordinate KDA state."""

from functools import lru_cache

import torch

from vllm.triton_utils import triton

from .cache import ReplayCache
from .metadata_high_p4_wy import build_persistent as build4
from .metadata_high_p6_wy import build_persistent as build6
from .metadata_rank1 import rank1_metadata
from .metadata_smallrank import small_metadata
from .step_direct_decay import _step_direct_decay


@lru_cache(maxsize=4)
def load_checkpoint(path):
    pack = torch.load(path, map_location="cpu", weights_only=True)
    contract = dict(
        schema="glm_kda_fullgram_allocation_v1",
        K=128,
        V=128,
        W=16,
        group_size=1,
        exact_Z=128,
        query_anchors=False,
        embedded_state=False,
        allocation_pivot_dependent=False,
        allocation_objective="joint_dot_sq_sum_fullgram",
    )
    for key, expected in contract.items():
        if pack["meta"].get(key) != expected:
            raise ValueError(f"Invalid calibration/runtime contract: {key}")
    if pack["frames"].keys() != pack["ranks"].keys():
        raise ValueError("Frame and rank layer IDs differ")
    eye = torch.eye(128, dtype=torch.float64, device="cpu")
    for layer, frame in pack["frames"].items():
        if frame.shape != (64, 128, 128) or not torch.isfinite(frame).all():
            raise ValueError(f"Invalid frame at layer {layer}")
        gram = frame.double().transpose(-1, -2) @ frame.double()
        if not torch.allclose(gram, eye.expand_as(gram), rtol=1e-5, atol=1e-5):
            raise ValueError("Frame must be orthogonal; native state is unrotated")
    return pack


class SketchCache(ReplayCache):
    """Approximate only non-flush reads; raw-write state and flush output are exact."""

    def __init__(self, frame, ranks, capacity=64, pivots=4):
        if pivots not in (4, 6):
            raise ValueError("Supported inference pivot counts are P4 and P6")
        if frame.ndim != 3 or frame.shape[1:] != (128, 128):
            raise ValueError("Expected orthogonal [head, K128, K128] frame")
        cpu = ranks.detach().cpu()
        if (
            cpu.shape != (len(frame),)
            or (cpu != cpu.long()).any()
            or ((cpu < 0) | (cpu > 128)).any()
        ):
            raise ValueError("One integer rank in [0, 128] is required per head")
        cpu = torch.where(cpu == 128, 0, cpu).long()
        super().__init__(len(frame), capacity, frame.device, replay_factors=False)
        self.frame = frame.float().contiguous()
        self.ranks = cpu.to(device=frame.device, dtype=torch.int32)
        self.rank = max(8, (int(cpu.max()) + 7) // 8 * 8)
        self.pivots = pivots
        self.all_sketch = bool((cpu > 0).all())
        p = self.pool
        p.latch_heads = self.ranks > 0
        fp = dict(device=frame.device, dtype=torch.float32)
        p.latch = torch.zeros(capacity, self.heads, 128, self.rank, **fp)
        p.phi = torch.zeros_like(p.latch)
        p.prefix = torch.zeros_like(p.log_a)
        p.direct_decay = torch.zeros_like(p.log_a)
        p.u = torch.zeros_like(p.log_a)
        p.f = torch.zeros(capacity, self.heads, 16, self.rank, **fp)
        sm = torch.cuda.get_device_properties(frame.device).multi_processor_count
        self.metadata_groups = []
        for mask, builder, warps, programs in (
            (cpu == 1, rank1_metadata, 4, 2 * sm),
            ((cpu > 1) & (cpu <= 4), small_metadata, 4, 2 * sm),
            (cpu > 4, build4 if pivots == 4 else build6, 8, sm),
        ):
            heads = torch.where(mask)[0].to(device=frame.device, dtype=torch.int32)
            if heads.numel():
                self.metadata_groups.append(
                    (
                        heads,
                        builder,
                        warps,
                        min(programs, capacity * heads.numel()),
                    )
                )

    def _metadata(self, slots, flush, q=None, out=None):
        p = self.pool
        for heads, builder, warps, programs in self.metadata_groups:
            builder[(programs,)](
                p.state,
                slots,
                p.pos,
                self.frame,
                self.ranks,
                p.latch,
                p.phi,
                p.k,
                p.v,
                p.log_a,
                p.beta,
                self.fresh,
                self.work_rows,
                self.work_counts,
                heads,
                heads.numel(),
                self.capacity,
                self.heads,
                self.rank,
                max(16, triton.next_power_of_2(self.rank)),
                flush,
                Q=q,
                Out=out,
                EXACT_OUTPUT=q is not None,
                num_warps=warps,
                num_stages=1,
            )

    def _decode(self, state, indices, slots, q, k, v, gate, beta, a_log, bias):
        self._metadata(slots, False)
        p = self.pool
        out = torch.zeros_like(v)
        _step_direct_decay[(q.shape[0], self.heads)](
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
            p.direct_decay,
            out,
            self.heads,
            128,
            128,
            self.rank,
            triton.next_power_of_2(self.rank),
            16,
            True,
            -5.0,
            128**-0.5,
            True,
            RAW_K_RING=True,
            Ranks=self.ranks,
            ALL_SKETCH=self.all_sketch,
            EXACT_FLUSH_OUTPUT=True,
            num_warps=4,
            num_stages=1,
        )
        self._metadata(slots, True, q, out)
        return out
