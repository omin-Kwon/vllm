# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM comparison board, using the existing Qwen/Super display conventions."""

import json
from pathlib import Path

import regex as re
from summarize_glm_kda_campaign import collect
from watch_glm53_flash import BENCHES, table

ROOT = Path("/disk2/omin/kda-latch-results/glm_calibration")
ARMS = [
    ("dense", "Dense (replay off)", "baseline"),
    *[(f"qmamba_b{b}_dsqonly", f"Q-Mamba {b}b", f"sim {b}b") for b in (10, 8, 6, 4)],
    *[
        (f"ghost_{key}", f"Ghost {sp}%", f"{ratio:.2f}×")
        for key, sp, ratio in (
            ("37_5", 37.5, 1.6),
            ("50", 50, 2),
            ("62_5", 62.5, 8 / 3),
            ("75", 75, 4),
            ("87_5", 87.5, 8),
        )
    ],
    *[(f"fisher_g{g}", f"Ours G={g}", "") for g in (28, 12, 7, 4)],
]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def progress(job):
    path = ROOT / f"{job}.log"
    if not path.exists():
        return "시작 중"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 20000))
        tail = stream.read().decode(errors="replace")
    matches = re.findall(r"Processed prompts:.*?(\d+)/(\d+) \[", tail)
    if not matches:
        return "모델 준비 중"
    done, total = map(int, matches[-1])
    return f"{done}/{total} ({100 * done / total:.1f}%)"


def main():
    campaign = read_json(ROOT / "campaign_status.json")
    records = {(r["arm"], r["bench"]): r for r in collect(ROOT)}
    rows = []
    for arm, label, storage in ARMS:
        if arm.startswith("fisher"):
            allocation = read_json(ROOT / f"{arm}.json")
            ratio = allocation.get("modeled_reduction")
            storage = f"{ratio:.3f}×" if ratio is not None else "배분 대기"
        cells = []
        for bench, _, _, _ in BENCHES:
            record = records[arm, bench]
            if record["state"] == "complete":
                token = record["mean_tokens"]
                cells.append(
                    (f"{float(record['score']):.1f} /{float(token):.0f}", "done")
                )
            elif arm.startswith("ghost"):
                cells.append(("—", "wait"))
            elif arm.startswith("fisher") and campaign.get("job", "").endswith(
                f"{arm[7:]}_{bench}"
            ):
                state = campaign.get("stage")
                job = campaign["job"]
                if state == "paused":
                    cells.append((f"중지 {progress(job)}", "wait"))
                elif state == "failed":
                    cells.append(("실패: 로그 확인", "fail"))
                elif job.startswith("scoring"):
                    cells.append(("채점 중", "score"))
                else:
                    cells.append((progress(job), "run"))
            else:
                cells.append(("대기", "wait"))
        rows.append((label, storage, cells))
    print("GLM 5.3 Flash NVFP4 · 정확도(%) / 평균 생성 토큰 · cap 65,536")
    print("\n".join(table(rows, ["방법", "모델상 절감/bit"] + [b[1] for b in BENCHES])))
    print("\nGhost: 미실험 빈칸. Q-Mamba: fake quant 명목 bit, 물리 cache FP32.")
    print("Ours: paired-Fisher, r=128, W=16. 절감률은 traffic 모델 기준.")
    print("기존 Dense/Q-Mamba: B300 DP2/EP2 | Ours: RTX TP8, packed FP8 MLA.")
    print("AIME25: 30문항 × 8 seeds의 평균 pass@1.")
    job = campaign.get("job", "")
    log = ROOT / f"{job}.log"
    if log.exists() and campaign.get("stage") == "running":
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 30000))
            tail = stream.read().decode(errors="replace")
        speed = re.findall(
            r"Avg generation throughput: ([\d.]+) tokens/s, "
            r"Running: (\d+) reqs, Waiting: (\d+) reqs",
            tail,
        )
        if speed:
            rate, active, waiting = speed[-1]
            print(f"생성 {rate} tokens/s · 실행 {active}문항 · 대기 {waiting}문항")
            print("진행률은 생성 완료 문항 기준; 실행 중 토큰은 완료 개수에 미포함.")

    rates = {}
    for variant in ("reference", "graph"):
        timing = read_json(ROOT / f"engine_benchmark_{variant}.json")
        if timing.get("runs"):
            runs = timing["runs"]
            rates[variant] = sum(r["tokens"] for r in runs) / sum(
                r["seconds"] for r in runs
            )
    if len(rates) == 2:
        print(
            f"고정 작업량 엔진 측정: {rates['reference']:.0f} → "
            f"{rates['graph']:.0f} tokens/s "
            f"({rates['graph'] / rates['reference']:.2f}×, batch 128)"
        )
    completed = sum(
        r["state"] == "complete" and r["arm"].startswith("fisher")
        for r in records.values()
    )
    print(
        f"\nPaired-Fisher {len(list(ROOT.glob('paired_*.pt')))}/48 · "
        f"평가 {completed}/16 · {campaign.get('stage', '대기')}: "
        f"{campaign.get('job', '')}"
    )


if __name__ == "__main__":
    main()
