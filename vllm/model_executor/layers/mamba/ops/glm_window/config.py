# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit opt-in; unsupported state/lifecycle contracts fail at model load."""

import torch

from .cache import ReplayCache


def create_cache(vllm_config, layer_idx, heads, head_dim, lower_bound):
    options = vllm_config.additional_config.get("kda_window")
    if options is None:
        return None
    from vllm.model_executor.layers.mamba.gdn.gdn_quant import bits_from_env

    if bits_from_env():
        raise ValueError("Q-Mamba DSQ cannot be combined with GLM window replay/sketch")
    allowed = {"mode", "window", "checkpoint", "pivots", "sketch_dtype"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Invalid kda_window configuration keys")
    mode = options.get("mode")
    if mode not in ("replay", "sketch") or options.get("window", 16) != 16:
        raise ValueError("Select kda_window mode replay/sketch and W16")
    parallel = vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
    ):
        raise ValueError("GLM window runtime currently requires TP1/PP1/DP1")
    if vllm_config.speculative_config is not None:
        raise ValueError("GLM window runtime does not support speculative decoding")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("GLM window ownership requires model runner v2")
    if vllm_config.scheduler_config.async_scheduling:
        raise ValueError("GLM window runtime requires synchronous scheduling")
    cc = vllm_config.cache_config
    if cc.enable_prefix_caching or cc.use_replayssm:
        raise ValueError("Disable prefix caching and Mamba2 use_replayssm")
    if cc.mamba_cache_mode != "none":
        raise ValueError("GLM window runtime requires mamba_cache_mode=none")
    if cc.mamba_ssm_cache_dtype != "float32":
        raise ValueError("Set mamba_ssm_cache_dtype=float32 explicitly")
    if head_dim != 128 or heads != 64 or lower_bound != -5.0:
        raise ValueError("Only GLM KDA H64/K128/V128/lower_bound=-5 is qualified")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("GLM window runtime requires BF16 activations")
    capacity = max(
        vllm_config.scheduler_config.max_num_seqs,
        vllm_config.compilation_config.max_cudagraph_capture_size or 0,
    )
    if mode == "replay":
        if any(k in options for k in ("checkpoint", "pivots", "sketch_dtype")):
            raise ValueError("Exact Replay has no checkpoint or pivot approximation")
        return ReplayCache(heads, capacity, torch.device("cuda"))
    from .sketch import SketchCache, load_checkpoint

    dtype = options.get("sketch_dtype", "bfloat16")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("sketch_dtype must be float32 or bfloat16")
    pack = load_checkpoint(options["checkpoint"])
    return SketchCache(
        pack["frames"][layer_idx].cuda(),
        pack["ranks"][layer_idx],
        capacity=capacity,
        pivots=options.get("pivots", 4),
        sketch_dtype=getattr(torch, dtype),
    )
