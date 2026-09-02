# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Q-Mamba dynamic state quantization for GDN and KDA state caches."""

import os

import torch

from vllm.triton_utils import tl, triton

_F16_MIN = 5.96e-8
_SUPPORTED_BITS = frozenset((4, 6, 8, 10))


def bits_from_env() -> int:
    value = os.environ.get("NS_GDN_QBITS", "0")
    try:
        bits = int(value)
    except ValueError as exc:
        raise ValueError(f"NS_GDN_QBITS must be an integer, got {value!r}") from exc
    if bits == 0:
        return 0
    if bits not in _SUPPORTED_BITS:
        supported = ", ".join(str(bit) for bit in sorted(_SUPPORTED_BITS))
        raise ValueError(
            f"NS_GDN_QBITS={bits} is unsupported; expected one of {supported}"
        )
    granularity = os.environ.get("NS_GDN_QGRAN", "dsq_qm")
    if granularity != "dsq_qm":
        raise ValueError(
            "Q-Mamba supports DSQ only: "
            f"NS_GDN_QGRAN must be 'dsq_qm', got {granularity!r}"
        )
    if os.environ.get("NS_GDN_QSR", "0") != "0":
        raise ValueError(
            "Q-Mamba DSQ uses deterministic round-to-nearest; NS_GDN_QSR is unsupported"
        )
    return bits


def fake_quant_dsq(state: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply the Q-Mamba DSQ equation and immediately dequantize the state."""
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"unsupported Q-Mamba bit width: {bits}")
    state_fp32 = state.float()
    magnitude = state_fp32.abs()
    channel_scale = magnitude.mean(dim=-1, keepdim=True).sqrt()
    channel_scale = channel_scale.clamp(min=_F16_MIN).half().float()
    state_scale = (magnitude / channel_scale).amax(dim=-2, keepdim=True)
    state_scale = state_scale.clamp(min=_F16_MIN).half().float()
    qmax = 2 ** (bits - 1) - 1
    scale = (channel_scale * state_scale).clamp(min=1e-12) / qmax
    quantized = (state_fp32 / scale).round().clamp(-qmax - 1, qmax)
    return (quantized * scale).to(state.dtype)


@triton.jit
def _dsq_kernel(
    state,
    slots,
    row_mask,
    stride_slot,
    stride_head,
    stride_channel,
    stride_state,
    num_slots,
    qmax_f,
    num_state_f,
    num_heads: tl.constexpr,
    num_channels: tl.constexpr,
    num_state: tl.constexpr,
    qmax: tl.constexpr,
    block_channels: tl.constexpr,
    block_state: tl.constexpr,
):
    program_id = tl.program_id(0)
    row = program_id // num_heads
    head = program_id % num_heads
    if tl.load(row_mask + row) == 0:
        return

    slot = tl.load(slots + row).to(tl.int64)
    if (slot < 0) | (slot >= num_slots):
        return

    channel_offsets = tl.arange(0, block_channels)
    state_offsets = tl.arange(0, block_state)
    valid = (channel_offsets[:, None] < num_channels) & (
        state_offsets[None, :] < num_state
    )
    pointers = (
        state
        + slot * stride_slot
        + head * stride_head
        + channel_offsets[:, None] * stride_channel
        + state_offsets[None, :] * stride_state
    )
    values = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
    magnitude = tl.abs(values)

    channel_scale = tl.sqrt(
        tl.fdiv(
            tl.sum(tl.where(valid, magnitude, 0.0), axis=1)[:, None],
            num_state_f,
            ieee_rounding=True,
        )
    )
    channel_scale = tl.maximum(channel_scale, 5.96e-8)
    channel_scale = channel_scale.to(tl.float16).to(tl.float32)
    state_scale = tl.max(
        tl.where(
            valid,
            tl.fdiv(magnitude, channel_scale, ieee_rounding=True),
            -float("inf"),
        ),
        axis=0,
    )[None, :]
    state_scale = tl.maximum(state_scale, 5.96e-8)
    state_scale = state_scale.to(tl.float16).to(tl.float32)
    scale = tl.fdiv(
        tl.maximum(channel_scale * state_scale, 1e-12),
        qmax_f,
        ieee_rounding=True,
    )
    normalized = tl.fdiv(values, scale, ieee_rounding=True)
    rounded = tl.extra.cuda.libdevice.nearbyint(normalized)
    rounded = tl.minimum(tl.maximum(rounded, -qmax - 1.0), qmax)
    tl.store(pointers, (rounded * scale).to(state.dtype.element_ty), mask=valid)


def quantize_slots_(
    state: torch.Tensor,
    slots: torch.Tensor,
    bits: int,
    row_mask: torch.Tensor | None = None,
) -> None:
    """Fake-quantize selected ``(value, key)`` states in place."""
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"unsupported Q-Mamba bit width: {bits}")
    if not state.is_cuda:
        raise RuntimeError("Q-Mamba DSQ requires a CUDA state cache")
    if state.dtype != torch.float32:
        raise RuntimeError(
            "Q-Mamba DSQ requires a float32 recurrent state cache; "
            f"received {state.dtype}"
        )
    if state.ndim != 4:
        raise RuntimeError(
            "Q-Mamba DSQ expects state layout (slots, heads, value, key); "
            f"received shape {tuple(state.shape)}"
        )

    slots = slots.reshape(-1).contiguous()
    if row_mask is None:
        row_mask = torch.ones_like(slots, dtype=torch.bool)
    else:
        row_mask = row_mask.reshape(-1).contiguous()
        if row_mask.dtype != torch.bool:
            raise RuntimeError(
                f"Q-Mamba row mask must be bool, received {row_mask.dtype}"
            )
        if row_mask.numel() != slots.numel():
            raise RuntimeError(
                "Q-Mamba row-mask length does not match state-index length: "
                f"{row_mask.numel()} != {slots.numel()}"
            )

    num_rows = slots.numel()
    num_heads, num_channels, num_state = state.shape[1:]
    block_channels = triton.next_power_of_2(num_channels)
    block_state = triton.next_power_of_2(num_state)
    qmax = 2 ** (bits - 1) - 1
    _dsq_kernel[(num_rows * num_heads,)](
        state,
        slots,
        row_mask,
        *state.stride(),
        state.shape[0],
        float(qmax),
        float(num_state),
        num_heads,
        num_channels,
        num_state,
        qmax,
        block_channels,
        block_state,
        num_warps=8,
    )
