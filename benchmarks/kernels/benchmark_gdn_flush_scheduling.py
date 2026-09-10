# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interleave a saved full-coordinate kernel and the fixed-head solve plan."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import benchmark_gdn_sketch as bench
import torch

from vllm.third_party.flash_linear_attention.ops import gdn_flush_full_cuda as current


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            (
                "--query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.mem,"
                "power.draw,temperature.gpu"
            ),
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    spec = importlib.util.spec_from_file_location(
        "gdn_flush_before", args.baseline_source
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    candidate_build = os.environ["NS_GDN_FULL_FLUSH_BUILD_DIR"]
    os.environ["NS_GDN_FULL_FLUSH_BUILD_DIR"] = str(args.baseline_build)
    baseline._extension()
    os.environ["NS_GDN_FULL_FLUSH_BUILD_DIR"] = candidate_build
    current._extension()
    new_workspace = current.FlushWorkspace

    def before_workspace(*a, **kw):
        kw.pop("widths", None)
        workspace = baseline.FlushWorkspace(*a, **kw)
        # The common benchmark accounts for the newly added index buffer.
        workspace._solve_heads = torch.empty(0, dtype=torch.int32, device="cuda")
        return workspace

    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_status(),
        "torch": torch.__version__,
        "baseline_source": str(args.baseline_source),
        "baseline_sha256": hashlib.sha256(
            args.baseline_source.read_bytes()
        ).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            Path(current.__file__).read_bytes()
        ).hexdigest(),
        "protocol": "3 warmups, 32 calls/graph, median of 5 CUDA event samples",
        "rounds": args.rounds,
        "scope": "GDN K=V=128, W=16, FP32 state/rings, BF16 mixed QKV; kernel timings",
        "nonflush_reference": "legacy r-capable CUDA, r=128; not dense ReplaySSM",
        "stream_phase": "exact state fold + U refresh + Gram, combined",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples = []

    def save(kind, case, variant, round_index, result):
        item = dict(
            kind=kind,
            case=case,
            variant=variant,
            round=round_index,
            result=result,
            gpu=gpu_status(),
        )
        samples.append(item)
        (args.output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
        print(json.dumps(item), flush=True)
        gc.collect()
        torch.accelerator.empty_cache()

    try:
        # Exclude initial clock ramp and extension initialization from comparisons.
        for fixed in (False, True):
            current.FlushWorkspace = new_workspace if fixed else before_workspace
            bench.analyze_flush_width(32, fixed_solve_plan=fixed)
        for m, g in [
            (4, None),
            (8, None),
            (16, None),
            (32, None),
            (64, None),
            (128, None),
            (8, 128),
            (32, 128),
        ]:
            for round_index in range(args.rounds):
                order = (False, True) if round_index % 2 == 0 else (True, False)
                for fixed in order:
                    current.FlushWorkspace = (
                        new_workspace if fixed else before_workspace
                    )
                    result = bench.analyze_flush_width(m, g, fixed_solve_plan=fixed)
                    save(
                        "width",
                        dict(m=m, g=result["g"]),
                        "planned" if fixed else "before",
                        round_index,
                        result,
                    )
        cases = [
            (1, 8, False, False),
            (32, 8, False, False),
            (128, 8, False, False),
            (128, 32, True, False),
            (128, 128, True, False),
            (128, 128, True, True),
        ]
        for batch, g, dense_mix, heterogeneous in cases:
            case = dict(
                batch=batch, g=g, dense_mix=dense_mix, heterogeneous=heterogeneous
            )
            for round_index in range(args.rounds):
                order = (False, True) if round_index % 2 == 0 else (True, False)
                for fixed in order:
                    current.FlushWorkspace = (
                        new_workspace if fixed else before_workspace
                    )
                    result = bench.run_flush(
                        batch, g, dense_mix, heterogeneous, fixed_solve_plan=fixed
                    )
                    save(
                        "cycle",
                        case,
                        "planned" if fixed else "before",
                        round_index,
                        result,
                    )
            # Step source is unchanged by this optimization. Measure all 15
            # non-flush positions rather than selecting a favorable position.
            for round_index in range(args.rounds):
                result = bench.run(
                    batch,
                    g,
                    dense_mix,
                    heterogeneous,
                    cuda_only=True,
                    position=list(range(15)),
                )
                save("nonflush", case, "unchanged", round_index, result)
    finally:
        current.FlushWorkspace = new_workspace
    metadata.update(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        final_gpu=gpu_status(),
        sample_count=len(samples),
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
