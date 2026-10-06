# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM of Mamba-2 layers (``MambaMixer2``)."""

import math
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2 import (
    SKETCH_DTYPES,
    SketchArgs,
    SketchTables,
    sketch_build,
    sketch_rotate,
    sketch_shapes,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_triton import (
    sketch_triton_decode,
)
from vllm.model_executor.layers.mamba.sketchssm import (
    SketchSSMCalibration,
    load_sketchssm_calibration,
)

if TYPE_CHECKING:
    from vllm.config import CacheConfig

logger = init_logger(__name__)


def mamba2_sketchssm_state_shapes(
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """State shapes of a SketchSSM Mamba-2 layer (key-major temporal state)."""
    heads, head_dim, state_size = shapes[1]
    return (shapes[0], (heads, state_size, head_dim), *shapes[2:])


class Mamba2SketchSSM(torch.nn.Module):
    """SketchSSM frames, sketch and decode kernels of one Mamba-2 layer."""

    def __init__(
        self,
        calibration: SketchSSMCalibration,
        layer: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
    ):
        super().__init__()
        # The kernels take state_size as a power-of-two block and use it and
        # head_dim as tl.dot dimensions, which must be at least 16.
        if state_size < 16 or state_size & (state_size - 1):
            raise ValueError(
                "SketchSSM requires a Mamba-2 state size that is a power of two "
                f"and at least 16, got {state_size}"
            )
        if head_dim < 16:
            raise ValueError(
                f"SketchSSM requires a Mamba-2 head dim of at least 16, got {head_dim}"
            )
        frames = calibration.frames[layer]
        ranks = calibration.ranks[layer]
        if frames.shape != (n_groups, state_size, state_size):
            raise ValueError(
                f"SketchSSM frames {tuple(frames.shape)} do not match "
                f"{n_groups} groups of state size {state_size}"
            )
        if ranks.shape != (num_heads,):
            raise ValueError(
                f"SketchSSM ranks cover {ranks.numel()} of {num_heads} heads"
            )
        self.register_buffer(
            "frames_t",
            frames.transpose(-1, -2)
            .contiguous()
            .to(torch.get_default_device(), torch.float32),
            persistent=False,
        )
        self.tables = SketchTables(ranks, state_size)
        # One sketch per concurrent request, indexed by its persistent index.
        shapes = sketch_shapes(ranks, head_dim, state_size)
        for name, shape, dtype in zip(("u", "w", "ag"), shapes, SKETCH_DTYPES):
            self.register_buffer(
                name, torch.zeros(max_num_reqs, *shape, dtype=dtype), persistent=False
            )

    def rotate(
        self, B: torch.Tensor, C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 rotated copies of B and C ``(tokens, groups * state_size)``."""
        return sketch_rotate(B, C, self.frames_t)

    def decode(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        x_cache: torch.Tensor,
        dt_cache: torch.Tensor,
        B_cache: torch.Tensor,
        attn_metadata,
        state_indices: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """One decode step with unrotated B and C; arguments as for the
        ReplaySSM decode."""
        sketch = SketchArgs(self.u, self.w, self.ag, self.tables, self.frames_t)
        if state_indices.dim() == 2:
            state_indices = state_indices[:, 0]
        sketch_triton_decode(
            state, x, dt, A, B, C, D, dt_bias, x_cache, dt_cache, B_cache,
            attn_metadata.bc_pre_scratch, attn_metadata.write_pos_d,
            attn_metadata.is_flush_d, attn_metadata.sketch_flush_rows_d,
            state_indices, attn_metadata.sketch_meta_d, out, sketch,
        )  # fmt: skip

    def prefilled(
        self, state: torch.Tensor, attn_metadata, state_indices: torch.Tensor
    ) -> None:
        """Build the sketch of each prefill row that completes its prompt."""
        sketch = SketchArgs(self.u, self.w, self.ag, self.tables, self.frames_t)
        sketch_build(
            state, attn_metadata.sketch_build_p, state_indices,
            attn_metadata.sketch_meta_p, sketch,
        )  # fmt: skip

    @classmethod
    def maybe_create(
        cls,
        cache_config: "CacheConfig | None",
        recurrent_layer_idx: int | None,
        model_layer_idx: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
    ) -> "Mamba2SketchSSM | None":
        """The layer's SketchSSM when ``--sketchssm`` is set."""
        if cache_config is None or cache_config.sketchssm is None:
            return None
        if recurrent_layer_idx is None:
            raise ValueError("SketchSSM needs the recurrent layer index")
        calibration = load_sketchssm_calibration(
            cache_config.sketchssm,
            cache_config.sketchssm_mean_rank,
            cache_config.replayssm_buffer_len,
        )
        # Per request over the whole calibration: independent of which layers
        # this rank holds (pipeline parallelism splits them across ranks).
        request_bytes = sum(
            math.prod(shape) * dtype.itemsize
            for ranks in calibration.ranks
            for shape, dtype in zip(
                sketch_shapes(ranks, head_dim, state_size), SKETCH_DTYPES
            )
        )
        logger.info_once(
            "SketchSSM sketch buffers: %.1f KiB per request over all %d "
            "calibrated layers; each layer holds them for %d requests",
            request_bytes / 2**10,
            calibration.num_layers,
            max_num_reqs,
        )
        return cls(
            calibration,
            calibration.layer_index(recurrent_layer_idx, model_layer_idx),
            num_heads,
            head_dim,
            n_groups,
            state_size,
            max_num_reqs,
        )
