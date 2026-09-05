# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tabulate completed metrics; missing cells remain explicitly pending."""

import argparse
import csv
import json
import sys
from pathlib import Path

SCALE = Path("/disk2/omin/nested_ssm/scale")
sys.path.insert(0, str(SCALE))
from read_metric import read as read_metric  # noqa: E402


def collect(root):
    baseline = Path("/disk2/omin/nested_ssm/scale/ns_official_results")
    records = []
    for bench in ("math-500", "aime25", "gpqa", "livecodebench"):
        for arm in (
            "dense",
            "qmamba_b10_dsqonly",
            "qmamba_b8_dsqonly",
            "qmamba_b6_dsqonly",
            "qmamba_b4_dsqonly",
            "ghost_37_5",
            "ghost_50",
            "ghost_62_5",
            "ghost_75",
            "ghost_87_5",
            "fisher_g28",
            "fisher_g12",
            "fisher_g7",
            "fisher_g4",
        ):
            if arm.startswith("fisher"):
                files = [
                    p
                    for p in (root / "eval" / bench).glob(
                        f"glm53_latch_{arm}_*.metrics.json"
                    )
                    if "_smoke" not in p.name
                ]
            elif arm.startswith("ghost"):
                files = []
            else:
                files = [
                    baseline
                    / bench
                    / f"glm53_nvfp4_{arm}_dp2ep2_replayoff_fp32_cap64k.metrics.json"
                ]
            files = [p for p in files if p.exists()]
            if len(files) > 1:
                raise ValueError(f"Ambiguous results: {files}")
            metric = {}
            key = ""
            if files:
                _, _, _, key = read_metric(files[0])
                payload = json.loads(files[0].read_text())["metrics"]
                metric = (payload.get("_") or payload)["_all_"][key]
            records.append(
                dict(
                    bench=bench,
                    arm=arm,
                    state="complete" if files else "pending",
                    score=metric.get("symbolic_correct", metric.get("accuracy", "")),
                    mean_tokens=metric.get("avg_tokens", ""),
                    no_answer=metric.get("no_answer", ""),
                    source=str(files[0]) if files else "",
                    metric=key,
                    runtime="RTX TP8 / FP8 MLA"
                    if arm.startswith("fisher")
                    else "not run"
                    if arm.startswith("ghost")
                    else "historical B300 DP2 EP2",
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    records = collect(root)
    with (root / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# GLM KDA campaign",
        "",
        (
            "Historical dense/Q-Mamba results use B300 DP2/EP2. "
            "New latch results use RTX TP8 and packed FP8 MLA. "
            "Differences are not an isolated KDA ablation."
        ),
        "",
        "| Benchmark | Arm | State | Score | Mean tokens |",
        "|---|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['bench']} | {r['arm']} | {r['state']} | "
        f"{r['score']} | {r['mean_tokens']} |"
        for r in records
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
