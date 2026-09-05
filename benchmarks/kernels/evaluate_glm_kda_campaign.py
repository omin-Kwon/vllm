# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the existing shared-eval-v1 harness with an explicit KDA allocation."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank-budget", type=int, required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=128)
    args = parser.parse_args()
    allocation = args.root / f"fisher_g{args.rank_budget}.json"
    basis = args.root / "basis.pt"
    table = json.loads(allocation.read_text())
    if hashlib.sha256(basis.read_bytes()).hexdigest() != table["basis_sha256"]:
        raise ValueError("Allocation/basis hash mismatch")
    signature = hashlib.sha256(allocation.read_bytes()).hexdigest()[:12]
    tag = f"glm53_latch_fisher_g{args.rank_budget}_{signature}"
    if args.graph:
        tag += "_graph_v3"
    if args.smoke:
        tag += "_smoke"
    scale = Path("/disk2/omin/nested_ssm/scale")
    os.environ["NESTED_SSM_MODEL"] = "/disk2/models/GLM-5.3-Flash-NVFP4"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["NS_REASONING_EFFORT"] = "max"
    os.environ["NEMO_SKILLS_LCB_PACKAGE_PATH"] = str(scale / ".sandbox_env")
    sys.path.insert(0, str(scale))
    if args.bench == "livecodebench":
        sys.path.insert(1, str(scale / ".sandbox_env"))
        sys.path.insert(2, str(scale / ".lcb_env"))
        sys.path.append(str(scale / ".grader_env"))
    else:
        sys.path.insert(1, str(scale / ".grader_env"))
    import ns_run

    contract = dict(
        allocation=str(allocation),
        allocation_sha256=signature,
        basis_sha256=table["basis_sha256"],
        tensor_parallel_size=8,
        kv_cache="SM120 packed FP8 MLA",
        kda_state="FP32",
        graph=args.graph,
        implementation="graph_v3" if args.graph else "reference_eager",
        implementation_sha256=hashlib.sha256(
            (
                Path(__file__).resolve().parents[2] / "vllm/third_party/"
                "flash_linear_attention/ops/kda_latch_graph.py"
            ).read_bytes()
        ).hexdigest()
        if args.graph
        else None,
        baseline_note="existing B300 DP2/EP2 results; hardware/MLA arithmetic differ",
    )
    original_dials = ns_run._env_dials
    ns_run._env_dials = lambda: dict(original_dials(), kda_latch=contract)
    out = args.root / "eval"
    out.mkdir(exist_ok=True)
    (out / f"{tag}.contract.json").write_text(json.dumps(contract, indent=2))
    if not args.score_only:
        import vllm

        original_llm = vllm.LLM

        class AllocatedLLM(original_llm):
            def __init__(self, *pos, **kwargs):
                kwargs["tensor_parallel_size"] = 8
                if args.graph:
                    kwargs["compilation_config"] = {
                        "mode": 0,
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                    }
                kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
                kwargs["worker_extension_cls"] = (
                    "benchmarks.kernels.evaluate_glm_kda_latch.LatchStatsWorkerExtension"
                )
                kwargs["additional_config"]["kda_latch"] = {
                    "basis_path": str(basis),
                    "allocation_path": str(allocation),
                    "max_slots": args.max_seqs,
                    "graph": args.graph,
                }
                super().__init__(*pos, **kwargs)

            def generate(self, *pos, **kwargs):
                result = super().generate(*pos, **kwargs)
                stats = self.collective_rpc("get_kda_latch_stats")
                if len(stats) != 8 or any(len(worker) != 34 for worker in stats):
                    raise RuntimeError("Incomplete latch layer coverage")
                if any(
                    row["decode_rows"] <= 0
                    for worker in stats
                    for row in worker.values()
                ):
                    raise RuntimeError("A KDA layer did not execute latch decode")
                (out / f"{tag}.{args.bench}.latch_stats.json").write_text(
                    json.dumps(stats)
                )
                return result

        vllm.LLM = AllocatedLLM
    elif args.bench == "livecodebench":
        from score_glm53_flash_dp import ensure_lcb_sandbox

        ensure_lcb_sandbox()
    sys.argv = [
        str(scale / "ns_run.py"),
        "--bench",
        args.bench,
        "--tag",
        tag,
        "--out",
        str(out),
        "--seeds",
        "8" if args.bench == "aime25" else "1",
        "--max_new",
        "128" if args.smoke else "65536",
        "--max_model_len",
        "262144",
        "--util",
        "0.80",
        "--chunk",
        "500",
        "--cache_dt",
        "float32",
        "--prefix_caching",
        "0",
        "--max_num_batched_tokens",
        "8192",
        "--max_num_seqs",
        str(args.max_seqs),
        "--flashinfer_autotune",
        "0",
        "--gdn_prefill",
        "triton",
        "--async_scheduling",
        "0",
        "--score_only" if args.score_only else "--gen_only",
    ]
    if not args.graph:
        sys.argv += ["--eager"]
    if args.smoke:
        sys.argv += ["--n", "2"]
    ns_run.main()
    if args.score_only:
        metric_path = out / args.bench / f"{tag}.metrics.json"
        payload = json.loads(metric_path.read_text())
        payload["run"].update(
            tensor_parallel_size=8,
            data_parallel_size=1,
            kda_latch=contract,
            kv_cache="SM120 packed FP8 MLA",
        )
        metric_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
