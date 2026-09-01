"""Apply LinearAttentionPruning DRRQR Q/K masks to Qwen GDN layers.

Set ``NS_GDN_PRUNE`` to a ratio-specific ``indices.pt`` produced by
``extract_qwen_gdn_rrqr_indices.py``.  The mask is applied after the depthwise
convolution and SiLU, before Q/K L2 normalization, matching the calibration
point exactly.  This is an accuracy baseline: it zeros pruned state columns but
does not physically shrink the recurrent-state allocation or kernel geometry.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_PATH = os.environ.get("NS_GDN_PRUNE", "").strip()
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_CPU_MASKS: dict[int, Any] | None = None
_METADATA: dict[str, Any] | None = None
_LOGGED = False


def armed() -> bool:
    return bool(_PATH)


def _load() -> tuple[dict[int, Any], dict[str, Any]]:
    global _CPU_MASKS, _METADATA
    if _CPU_MASKS is not None and _METADATA is not None:
        return _CPU_MASKS, _METADATA
    if not _PATH:
        raise RuntimeError("NS_GDN_PRUNE is not set")

    import torch

    path = Path(_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NS_GDN_PRUNE does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    layers = payload.get("layers")
    if not isinstance(metadata, dict) or not isinstance(layers, dict):
        raise RuntimeError(f"Invalid DRRQR payload: {path}")

    expected_layers = [int(value) for value in metadata.get("gdn_layer_indices", [])]
    num_heads = int(metadata.get("num_key_heads", 0))
    head_dim = int(metadata.get("head_dim", 0))
    if len(expected_layers) != 36 or num_heads != 16 or head_dim != 128:
        raise RuntimeError(
            "Flash-Next DRRQR geometry mismatch: expected 36 layers, H=16, K=128; "
            f"got layers={len(expected_layers)}, H={num_heads}, K={head_dim}"
        )
    layer_keys = {int(key) for key in layers}
    if layer_keys != set(expected_layers):
        raise RuntimeError(
            f"DRRQR layer set mismatch: expected={expected_layers}, "
            f"actual={sorted(layer_keys)}"
        )

    masks: dict[int, Any] = {}
    kept_width: int | None = None
    for layer_index in expected_layers:
        entry = (
            layers[layer_index] if layer_index in layers else layers[str(layer_index)]
        )
        keep = entry["keep_local_rrqr_order"].to(dtype=torch.int64)
        if keep.ndim != 2 or keep.shape[0] != num_heads:
            raise RuntimeError(
                f"Layer {layer_index}: expected keep indices [16, kept], "
                f"got {tuple(keep.shape)}"
            )
        if kept_width is None:
            kept_width = int(keep.shape[1])
        if int(keep.shape[1]) != kept_width:
            raise RuntimeError(
                f"Layer {layer_index}: inconsistent kept width {keep.shape[1]}"
            )
        if kept_width <= 0 or kept_width > head_dim:
            raise RuntimeError(f"Layer {layer_index}: invalid kept width {kept_width}")
        if bool(((keep < 0) | (keep >= head_dim)).any()):
            raise RuntimeError(f"Layer {layer_index}: keep indices are out of range")
        if any(torch.unique(row).numel() != kept_width for row in keep):
            raise RuntimeError(f"Layer {layer_index}: duplicate keep indices")
        mask = torch.zeros(
            num_heads, head_dim, dtype=torch.uint8, device="cpu"
        )
        mask.scatter_(1, keep, 1)
        masks[layer_index] = mask

    _CPU_MASKS = masks
    _METADATA = metadata
    return masks, metadata


def configure_layer(layer: Any, prefix: str) -> None:
    """Bind and validate a vLLM GDN layer during model construction."""
    if not armed():
        layer._gdn_prune_active = False
        return
    if int(layer.tp_size) != 1:
        raise RuntimeError(
            "NS_GDN_PRUNE currently requires tensor_parallel_size=1; global 16-head "
            "indices have not been sharded for TP execution"
        )
    match = _LAYER_RE.search(prefix)
    if match is None:
        raise RuntimeError(
            f"Cannot derive model layer index from GDN prefix: {prefix!r}"
        )
    layer_index = int(match.group(1))
    masks, metadata = _load()
    if layer_index not in masks:
        raise RuntimeError(f"GDN layer {layer_index} is absent from NS_GDN_PRUNE")
    layer._gdn_prune_active = True
    layer._gdn_prune_layer_index = layer_index
    layer._gdn_prune_mask = None
    layer._gdn_prune_mask_contract = None

    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        kept = int(masks[layer_index][0].sum())
        ratio = float(metadata.get("pruning_ratio", 1.0 - kept / 128))
        print(
            f"[gdn-prune] armed sparsity={ratio:.3f} kept={kept}/128 "
            f"layers={len(masks)} source={Path(_PATH).resolve()}",
            flush=True,
        )


def prepare_device_mask(layer: Any, device: Any, dtype: Any) -> None:
    """Move a tiny per-layer mask before CUDA graph capture starts."""
    if not getattr(layer, "_gdn_prune_active", False):
        return
    contract = (str(device), str(dtype))
    if getattr(layer, "_gdn_prune_mask_contract", None) == contract:
        return
    masks, _ = _load()
    layer_index = int(layer._gdn_prune_layer_index)
    layer._gdn_prune_mask = masks[layer_index].to(device=device, dtype=dtype)
    layer._gdn_prune_mask_contract = contract


def prune_mixed(layer: Any, mixed_qkv: Any) -> Any:
    """In-place mask packed post-convolution Q/K; CUDA-graph safe when prepared."""
    if not getattr(layer, "_gdn_prune_active", False) or mixed_qkv is None:
        return mixed_qkv

    import torch

    contract = (str(mixed_qkv.device), str(mixed_qkv.dtype))
    if getattr(layer, "_gdn_prune_mask_contract", None) != contract:
        if mixed_qkv.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GDN pruning mask was not prepared before CUDA graph capture"
            )
        prepare_device_mask(layer, mixed_qkv.device, mixed_qkv.dtype)

    num_heads = int(layer.num_k_heads)
    head_dim = int(layer.head_k_dim)
    expected_width = 2 * num_heads * head_dim + int(layer.value_dim)
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != expected_width:
        raise RuntimeError(
            f"Unexpected packed GDN shape {tuple(mixed_qkv.shape)}; "
            f"expected [tokens, {expected_width}]"
        )
    qk = mixed_qkv[:, : 2 * num_heads * head_dim].view(
        mixed_qkv.shape[0], 2, num_heads, head_dim
    )
    qk.mul_(layer._gdn_prune_mask[None, None])
    return mixed_qkv
