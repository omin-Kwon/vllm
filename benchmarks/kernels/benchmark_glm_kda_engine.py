# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-work engine timing for reference and graph KDA evaluation paths."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--rank-budget", type=int, default=28)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit and not args.graph:
        raise ValueError("Live reference audit requires the optimized cache")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    options = dict(
        model="/disk2/models/GLM-5.3-Flash-NVFP4",
        tensor_parallel_size=8,
        mamba_ssm_cache_dtype="float32",
        enforce_eager=not args.graph or args.audit,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        enable_flashinfer_autotune=False,
        additional_config={
            "gdn_prefill_backend": "triton",
            "kda_latch": {
                "basis_path": str(args.root / "basis.pt"),
                "allocation_path": str(args.root / f"fisher_g{args.rank_budget}.json"),
                "max_slots": 128,
                "graph": args.graph,
            },
        },
        worker_extension_cls=(
            "benchmarks.kernels.evaluate_glm_kda_latch.LatchStatsWorkerExtension"
        ),
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    if args.graph and not args.audit:
        options["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
        }
    llm = LLM(**options)
    batch = 2 if args.audit else 128
    length = 128 if args.audit else 256
    prompts = [
        f"Explain how to solve x squared plus {i % 13 + 1} x minus 100 equals zero."
        for i in range(batch)
    ]
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
    if args.audit:
        llm.collective_rpc("install_latch_reference_audit")
    times = []
    for _ in range(1 if args.audit else 2):
        started = time.perf_counter()
        output = llm.generate(
            prompts, SamplingParams(temperature=0, max_tokens=length, ignore_eos=True)
        )
        elapsed = time.perf_counter() - started
        tokens = sum(len(row.outputs[0].token_ids) for row in output)
        assert tokens == batch * length
        times.append(
            dict(seconds=elapsed, tokens=tokens, tokens_per_second=tokens / elapsed)
        )
    stats = llm.collective_rpc("get_kda_latch_stats")
    assert len(stats) == 8 and all(len(worker) == 34 for worker in stats)
    result = dict(
        graph=args.graph, rank_budget=args.rank_budget, runs=times, stats=stats
    )
    name = "live_audit" if args.audit else "graph" if args.graph else "reference"
    path = args.root / f"engine_benchmark_{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(path=str(path), runs=times)), flush=True)
    if args.audit:
        assert all(
            row["dense_audit"]["outside_tolerance"] == 0
            and row["dense_audit"]["nonfinite"] == 0
            for worker in stats
            for row in worker.values()
        )


if __name__ == "__main__":
    main()
