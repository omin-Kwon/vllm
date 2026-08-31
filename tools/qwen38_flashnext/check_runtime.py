#!/usr/bin/env python3
"""Fail-closed runtime audit for the Flash-Next NVFP4 accuracy queue."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("Q38NEXT_VLLM_REPO"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--indices-root", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo or Path(__file__).resolve().parents[2]).resolve()
    model = Path(args.model).resolve()
    indices_root = Path(args.indices_root).resolve()

    import torch
    import vllm
    from vllm.model_executor.layers.mamba.gdn import gdn_prune, gdn_quant
    from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer

    imported = Path(vllm.__file__).resolve()
    if repo not in imported.parents:
        raise SystemExit(f"vLLM imported from {imported}, expected {repo}")
    for source in (Path(gdn_prune.__file__), Path(gdn_quant.__file__)):
        if repo not in source.resolve().parents:
            raise SystemExit(f"GDN extension imported outside source tree: {source}")
    if not PleOffloadLayer.__module__.startswith("vllm."):
        raise SystemExit("PLE offload layer is not available")
    flash_attn_extension = repo / "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
    if not flash_attn_extension.is_file():
        raise SystemExit(f"FlashAttention native extension is missing: {flash_attn_extension}")

    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config") or config
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != "qwen4_exp":
        raise SystemExit(f"unexpected model type: {config.get('model_type')}")
    if str(quant.get("quant_algo", "")).upper() != "NVFP4":
        raise SystemExit(f"expected NVFP4 checkpoint, got {quant}")
    if text_config.get("ple_embedding_dtype") != "float8_e4m3fn":
        raise SystemExit("NVFP4 checkpoint does not declare its FP8 PLE table")
    layer_types = text_config.get("layer_types", [])
    gdn_layers = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    if len(layer_types) != 48 or len(gdn_layers) != 36:
        raise SystemExit(
            f"unexpected geometry: layers={len(layer_types)} gdn={len(gdn_layers)}"
        )

    ratios: dict[str, dict[str, object]] = {}
    for ratio, kept in (("0.375", 80), ("0.500", 64), ("0.625", 48), ("0.750", 32)):
        root = indices_root / "ratios" / f"sparsity-{ratio}"
        validation = json.loads((root / "validation.json").read_text())
        index_path = root / "indices.pt"
        expected = {
            "status": "passed",
            "num_gdn_layers": 36,
            "num_heads_per_layer": 16,
            "original_head_dim": 128,
            "kept_head_dim": kept,
        }
        if not index_path.is_file() or any(
            validation.get(key) != value for key, value in expected.items()
        ):
            raise SystemExit(f"invalid DRRQR contract for {ratio}: {validation}")
        ratios[ratio] = {"kept_head_dim": kept, "indices": str(index_path)}

    ptxas = Path(os.environ.get("TRITON_PTXAS_PATH", "/nonexistent"))
    if not ptxas.is_file():
        raise SystemExit(f"TRITON_PTXAS_PATH is missing: {ptxas}")
    ptxas_help = subprocess.run(
        [str(ptxas), "--help"], check=True, text=True, capture_output=True
    ).stdout
    if "sm_103a" not in ptxas_help:
        raise SystemExit(f"ptxas does not support B300 sm_103a: {ptxas}")

    if os.environ.get("VLLM_PLE_CPU_OFFLOAD") != "1":
        raise SystemExit("VLLM_PLE_CPU_OFFLOAD must be 1")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise SystemExit("PLE accuracy queue requires the validated MRV1 path")

    gpu = None
    if args.require_gpu:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        capability = torch.cuda.get_device_capability(0)
        if capability != (10, 3):
            raise SystemExit(f"expected B300 compute capability (10, 3), got {capability}")
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    print(json.dumps({
        "status": "passed",
        "repo": str(repo),
        "branch": subprocess.run(
            ["git", "-C", str(repo), "branch", "--show-current"],
            check=True, text=True, capture_output=True
        ).stdout.strip(),
        "commit": commit,
        "vllm_import": str(imported),
        "model": str(model),
        "checkpoint_quant_algo": "NVFP4",
        "ple": {
            "embedding_dtype": text_config["ple_embedding_dtype"],
            "layer_ids": text_config["ple_layer_ids"],
            "backend": "dedicated_cpu_worker",
            "model_runner": "v1",
        },
        "gdn_layers": gdn_layers,
        "indices": ratios,
        "ptxas": str(ptxas),
        "flash_attn_extension": str(flash_attn_extension),
        "gpu": gpu,
        "versions": {
            name: version(name)
            for name in ("vllm", "torch", "triton", "flashinfer-python", "transformers", "ninja")
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
