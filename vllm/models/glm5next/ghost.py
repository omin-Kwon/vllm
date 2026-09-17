# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def _ratio_key(sparsity: float) -> str:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"GHOST sparsity must be in [0, 1), got {sparsity}")
    return f"{sparsity:.3f}"


def resolve_glm5_ghost_indices(source: str, sparsity: float) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        path = path / "indices-manifest.json"
    if path.suffix == ".pt":
        if not path.is_file():
            raise FileNotFoundError(f"GHOST indices do not exist: {path}")
        return path
    if path.suffix != ".json" or not path.is_file():
        raise FileNotFoundError(
            "VLLM_GLM5_GHOST_INDICES must point to indices.pt, "
            f"indices-manifest.json, or its directory: {path}"
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    ratio = _ratio_key(sparsity)
    ratio_entry = manifest.get("ratios", {}).get(ratio)
    if not isinstance(ratio_entry, dict):
        available = sorted(manifest.get("ratios", {}))
        raise ValueError(
            f"GHOST sparsity {ratio} is unavailable in {path}; available={available}"
        )
    relative = ratio_entry.get("indices_pt")
    if not isinstance(relative, str):
        raise ValueError(f"GHOST manifest entry {ratio} has no indices_pt")
    indices_path = (path.parent / relative).resolve()
    if not indices_path.is_relative_to(path.parent):
        raise ValueError(f"GHOST indices escape the manifest directory: {relative}")
    if not indices_path.is_file():
        raise FileNotFoundError(f"GHOST indices do not exist: {indices_path}")
    return indices_path


@lru_cache(maxsize=8)
def _load_indices(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid GHOST payload at {path}")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"GHOST payload has no metadata: {path}")
    if not isinstance(payload.get("layers"), dict):
        raise ValueError(f"GHOST payload has no layers: {path}")
    return payload


def apply_glm5_ghost_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor,
    local_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query.view(-1, local_num_heads, head_dim).mul_(mask)
    key.view(-1, local_num_heads, head_dim).mul_(mask)
    return query, key


def build_glm5_ghost_mask(
    source: str,
    sparsity: float,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, int, Path]:
    indices_path = resolve_glm5_ghost_indices(source, sparsity)
    payload = _load_indices(str(indices_path))
    if payload["metadata"].get("method") != "ghost2_kda_horizon1_dense_per_head":
        raise ValueError("Expected the preserved GLM GHOST horizon-one indices")
    metadata = payload["metadata"]

    if metadata.get("linear_attention_family") != "glm5_kda":
        raise ValueError(f"GHOST indices are not for GLM-5 KDA: {indices_path}")
    recorded_sparsity = float(metadata.get("pruning_ratio", -1.0))
    if not math.isclose(recorded_sparsity, sparsity, abs_tol=1e-9):
        raise ValueError(
            f"GHOST sparsity mismatch: requested={sparsity}, "
            f"payload={recorded_sparsity}"
        )
    if int(metadata.get("num_key_heads", -1)) != num_heads:
        raise ValueError(
            f"GHOST head count mismatch: model={num_heads}, "
            f"payload={metadata.get('num_key_heads')}"
        )
    if int(metadata.get("head_dim", -1)) != head_dim:
        raise ValueError(
            f"GHOST head dim mismatch: model={head_dim}, "
            f"payload={metadata.get('head_dim')}"
        )
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP rank/size: {tp_rank}/{tp_size}")
    if num_heads % tp_size:
        raise ValueError(f"{num_heads} heads are not divisible by TP={tp_size}")

    layers = payload["layers"]
    layer = layers.get(layer_idx, layers.get(str(layer_idx)))
    if not isinstance(layer, dict):
        raise ValueError(f"GHOST indices are missing GLM layer {layer_idx}")
    prune = layer.get("prune_local_sorted")
    if not isinstance(prune, torch.Tensor) or prune.ndim != 2:
        raise ValueError(f"Invalid prune_local_sorted for layer {layer_idx}")
    prune = prune.to(dtype=torch.long, device="cpu")

    kept_head_dim = int(metadata.get("kept_head_dim", -1))
    expected_pruned = head_dim - kept_head_dim
    if not 0 < kept_head_dim <= head_dim:
        raise ValueError(f"Invalid kept_head_dim={kept_head_dim}")
    if tuple(prune.shape) != (num_heads, expected_pruned):
        raise ValueError(
            f"Layer {layer_idx} prune shape is {tuple(prune.shape)}, "
            f"expected {(num_heads, expected_pruned)}"
        )
    if prune.numel():
        if int(prune.min()) < 0 or int(prune.max()) >= head_dim:
            raise ValueError(f"Layer {layer_idx} contains out-of-range indices")
        sorted_prune = prune.sort(dim=1).values
        if bool((sorted_prune[:, 1:] == sorted_prune[:, :-1]).any()):
            raise ValueError(f"Layer {layer_idx} contains duplicate indices")

    local_heads = num_heads // tp_size
    head_start = tp_rank * local_heads
    local_prune = prune[head_start : head_start + local_heads]
    mask = torch.ones((local_heads, head_dim), dtype=torch.bool, device="cpu")
    if local_prune.numel():
        mask.scatter_(1, local_prune, False)
    return mask, kept_head_dim, indices_path


def configure_ghost(layer, vllm_config):
    options = vllm_config.additional_config.get("kda_ghost")
    if options is None:
        layer.register_buffer("_ghost_qk_mask", None, persistent=False)
        return
    if not isinstance(options, dict) or set(options) != {"indices", "sparsity"}:
        raise ValueError("kda_ghost requires exactly indices and sparsity")
    if (
        layer._kda_qmamba_bits
        or vllm_config.speculative_config is not None
        or vllm_config.cache_config.use_replayssm
        or vllm_config.cache_config.use_kda_recoverssm
        or "kda_window" in vllm_config.additional_config
    ):
        raise ValueError(
            "GHOST requires plain KDA without DSQ/speculation/replay/sketch"
        )
    import hashlib

    manifest_path = Path(options["indices"]) / "indices-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "glm_kda_ghost.indices.v1":
        raise ValueError("Expected GLM GHOST manifest")
    entry = manifest["ratios"][_ratio_key(float(options["sparsity"]))]
    path = resolve_glm5_ghost_indices(options["indices"], float(options["sparsity"]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("GHOST indices checksum mismatch")
    mask, kept, _ = build_glm5_ghost_mask(
        options["indices"],
        float(options["sparsity"]),
        layer.layer_idx,
        layer.num_heads,
        layer.head_dim,
        layer.tp_rank,
        layer.tp_size,
    )
    layer.register_buffer(
        "_ghost_qk_mask", mask.to(layer.A_log.device), persistent=False
    )
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "GLM GHOST enabled: layer=%d sparsity=%s kept_dim=%d",
        layer.layer_idx,
        options["sparsity"],
        kept,
    )
