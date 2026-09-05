# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restartable calibration/allocation/evaluation queue for the GLM campaign."""

import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/disk2/omin/kda-latch-results/glm_calibration"),
    )
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    scripts = Path(__file__).resolve().parent

    def status(stage, **kwargs):
        value = dict(
            stage=stage,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **kwargs,
        )
        (root / "campaign_status.json").write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    def run(script, parameters, label):
        status("running", job=label)
        with (root / f"{label}.log").open("a") as output:
            result = subprocess.run(
                [sys.executable, str(scripts / script), *parameters],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            status("failed", job=label, exit_code=result.returncode)
            raise SystemExit(result.returncode)

    if not (root / "validation.json").exists():
        run(
            "calibrate_glm_kda_latch.py",
            ["--output", str(root / "probe.json"), "--resume"],
            "calibration",
        )
    if not all((root / f"fisher_g{g}.json").exists() for g in (28, 12, 7, 4)):
        run("allocate_glm_kda_latch.py", ["--root", str(root)], "allocation")
    for rank in (28, 12, 7, 4):
        for bench in ("math-500", "aime25", "gpqa", "livecodebench"):
            common = ["--root", str(root), "--rank-budget", str(rank), "--bench", bench]
            if args.graph:
                common.append("--graph")
            variant = "graph_v3_" if args.graph else ""
            marker = root / f"complete_{variant}g{rank}_{bench}.json"
            if marker.exists():
                continue
            run("evaluate_glm_kda_campaign.py", common, f"generation_g{rank}_{bench}")
            run(
                "evaluate_glm_kda_campaign.py",
                [*common, "--score-only"],
                f"scoring_g{rank}_{bench}",
            )
            marker.write_text(json.dumps(dict(complete=True)))
            run("summarize_glm_kda_campaign.py", ["--root", str(root)], "summary")
    status("complete", evaluation_cells=16)


if __name__ == "__main__":
    main()
