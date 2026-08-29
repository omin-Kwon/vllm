# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep GDN ReplaySSM launch configs on the Flash-Next decode shape."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent_replayssm import (
    fused_recurrent_gated_delta_rule_replayssm,
)

NUM_K_HEADS = 16
NUM_V_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
CACHE_LEN = 16
DEFAULT_CONFIG = (64, 1, 2, 4)


@dataclass(frozen=True)
class Config:
    """A GDN ReplaySSM Triton launch config."""

    block_v: int
    num_warps: int
    num_stages: int
    nk: int


@dataclass
class Timing:
    """Timing result for one config and batch size."""

    batch: int
    block_v: int
    num_warps: int
    num_stages: int
    nk: int
    nonflush_us: float | None
    flush_us: float | None
    weighted_us: float | None
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-batch", type=int, default=256)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--block-v", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--nk", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_inputs(batch: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qkv_dim = 2 * NUM_K_HEADS * KEY_DIM + NUM_V_HEADS * VALUE_DIM
    generator = torch.Generator(device=device).manual_seed(20260829 + batch)
    randn = lambda *shape: torch.randn(
        *shape, device=device, generator=generator, dtype=dtype
    )
    return {
        "mixed_qkv": 0.01 * randn(batch, qkv_dim),
        "a": 0.01 * randn(batch, NUM_V_HEADS),
        "b": 0.01 * randn(batch, NUM_V_HEADS),
        "A_log": torch.zeros(NUM_V_HEADS, device=device, dtype=dtype),
        "dt_bias": torch.zeros(NUM_V_HEADS, device=device, dtype=dtype),
        "initial_state": torch.zeros(
            batch + 1,
            NUM_V_HEADS,
            VALUE_DIM,
            KEY_DIM,
            device=device,
            dtype=torch.float32,
        ),
        "d_cache": torch.zeros(
            batch + 1,
            NUM_V_HEADS,
            CACHE_LEN,
            VALUE_DIM,
            device=device,
            dtype=dtype,
        ),
        "k_cache": torch.zeros(
            batch + 1,
            NUM_K_HEADS,
            CACHE_LEN,
            KEY_DIM,
            device=device,
            dtype=dtype,
        ),
        "g_cache": torch.zeros(
            batch + 1,
            NUM_V_HEADS,
            CACHE_LEN,
            device=device,
            dtype=torch.float32,
        ),
        "out": torch.empty(
            batch, 1, NUM_V_HEADS, VALUE_DIM, device=device, dtype=dtype
        ),
        "ssm_state_indices": torch.arange(
            1, batch + 1, device=device, dtype=torch.int32
        ),
    }


def run_kernel(
    inputs: dict[str, torch.Tensor], config: Config, write_pos: torch.Tensor
) -> None:
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=KEY_DIM**-0.5,
        initial_state=inputs["initial_state"],
        d_cache=inputs["d_cache"],
        k_cache=inputs["k_cache"],
        g_cache=inputs["g_cache"],
        out=inputs["out"],
        ssm_state_indices=inputs["ssm_state_indices"],
        write_pos=write_pos,
        use_qk_l2norm_in_kernel=True,
        block_v=config.block_v,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        nk=config.nk,
    )


def measure(
    inputs: dict[str, torch.Tensor],
    config: Config,
    position: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> float:
    batch = inputs["mixed_qkv"].shape[0]
    write_pos = torch.full(
        (batch,), position, device="cuda", dtype=torch.int32
    )
    for _ in range(warmup):
        run_kernel(inputs, config, write_pos)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run_kernel(inputs, config, write_pos)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def benchmark_config(
    inputs: dict[str, torch.Tensor],
    config: Config,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Timing:
    batch = inputs["mixed_qkv"].shape[0]
    try:
        nonflush_us = measure(
            inputs, config, 3, warmup, iterations, repeats
        )
        flush_us = measure(
            inputs, config, CACHE_LEN - 1, warmup, iterations, repeats
        )
        weighted_us = (
            (CACHE_LEN - 1) * nonflush_us + flush_us
        ) / CACHE_LEN
        return Timing(
            batch=batch,
            **asdict(config),
            nonflush_us=nonflush_us,
            flush_us=flush_us,
            weighted_us=weighted_us,
            status="ok",
        )
    except Exception as error:
        torch.cuda.synchronize()
        return Timing(
            batch=batch,
            **asdict(config),
            nonflush_us=None,
            flush_us=None,
            weighted_us=None,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )


def write_csv(path: Path, rows: list[Timing]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(Timing.__annotations__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary.replace(path)


def correctness(config: Config) -> dict[str, object]:
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_steps = 2 * CACHE_LEN + 1
    qkv_dim = 2 * NUM_K_HEADS * KEY_DIM + NUM_V_HEADS * VALUE_DIM
    mixed_qkv = (0.1 * torch.randn(num_steps, qkv_dim, device=device)).to(dtype)
    a = (0.1 * torch.randn(num_steps, NUM_V_HEADS, device=device)).to(dtype)
    b = (0.1 * torch.randn_like(a)).to(dtype)
    A_log = torch.zeros(NUM_V_HEADS, device=device, dtype=dtype)
    dt_bias = torch.zeros_like(A_log)
    indices = torch.tensor([1], device=device, dtype=torch.int32)
    state_baseline = 0.1 * torch.randn(
        2,
        NUM_V_HEADS,
        VALUE_DIM,
        KEY_DIM,
        device=device,
        dtype=torch.float32,
    )
    state_replay = state_baseline.clone()
    d_cache = torch.empty(
        2, NUM_V_HEADS, CACHE_LEN, VALUE_DIM, device=device, dtype=dtype
    )
    k_cache = torch.empty(
        2, NUM_K_HEADS, CACHE_LEN, KEY_DIM, device=device, dtype=dtype
    )
    g_cache = torch.empty(
        2, NUM_V_HEADS, CACHE_LEN, device=device, dtype=torch.float32
    )
    output_error = 0.0
    flush_state_error = 0.0
    passed = True

    for step in range(num_steps):
        out_baseline = torch.empty(
            1, 1, NUM_V_HEADS, VALUE_DIM, device=device, dtype=dtype
        )
        out_replay = torch.empty_like(out_baseline)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=KEY_DIM**-0.5,
            initial_state=state_baseline,
            out=out_baseline,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=KEY_DIM**-0.5,
            initial_state=state_replay,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out_replay,
            ssm_state_indices=indices,
            write_pos=torch.tensor(
                [step % CACHE_LEN], device=device, dtype=torch.int32
            ),
            use_qk_l2norm_in_kernel=True,
            block_v=config.block_v,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            nk=config.nk,
        )
        output_error = max(
            output_error,
            float((out_replay.float() - out_baseline.float()).abs().max()),
        )
        passed &= torch.allclose(
            out_replay.float(), out_baseline.float(), rtol=5e-3, atol=5e-4
        )
        if step % CACHE_LEN == CACHE_LEN - 1:
            state_error = float(
                (state_replay[1] - state_baseline[1]).abs().max()
            )
            flush_state_error = max(flush_state_error, state_error)
            passed &= torch.allclose(
                state_replay[1], state_baseline[1], rtol=5e-3, atol=5e-4
            )

    return {
        **asdict(config),
        "passed": bool(passed),
        "max_output_abs_error": output_error,
        "max_flush_state_abs_error": flush_state_error,
    }


def configs_from_args(args: argparse.Namespace) -> list[Config]:
    if args.smoke:
        return [Config(*DEFAULT_CONFIG)]
    return [
        Config(*values)
        for values in itertools.product(
            args.block_v, args.num_warps, args.num_stages, args.nk
        )
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    configs = configs_from_args(args)
    metadata = {
        "started_at_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "geometry": {
            "num_k_heads": NUM_K_HEADS,
            "num_v_heads": NUM_V_HEADS,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "cache_len": CACHE_LEN,
        },
        "default_config": DEFAULT_CONFIG,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    primary_inputs = make_inputs(args.primary_batch)
    primary_rows: list[Timing] = []
    for index, config in enumerate(configs, start=1):
        row = benchmark_config(
            primary_inputs,
            config,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        primary_rows.append(row)
        write_csv(args.output_dir / "primary.csv", primary_rows)
        print(
            f"[{index:03d}/{len(configs):03d}] {config} "
            f"weighted_us={row.weighted_us} status={row.status}",
            flush=True,
        )

    successful = [row for row in primary_rows if row.status == "ok"]
    successful.sort(key=lambda row: row.weighted_us or float("inf"))
    selected = {
        Config(row.block_v, row.num_warps, row.num_stages, row.nk)
        for row in successful[: args.top_k]
    }
    selected.add(Config(*DEFAULT_CONFIG))
    del primary_inputs
    torch.cuda.empty_cache()

    followup_rows: list[Timing] = []
    if not args.smoke:
        for batch in args.batches:
            inputs = make_inputs(batch)
            for config in sorted(
                selected,
                key=lambda item: (
                    item.block_v,
                    item.num_warps,
                    item.num_stages,
                    item.nk,
                ),
            ):
                row = benchmark_config(
                    inputs,
                    config,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                followup_rows.append(row)
                write_csv(args.output_dir / "followup.csv", followup_rows)
                print(
                    f"[followup B={batch}] {config} "
                    f"weighted_us={row.weighted_us} status={row.status}",
                    flush=True,
                )
            del inputs
            torch.cuda.empty_cache()

    validation = [correctness(config) for config in sorted(selected, key=repr)]
    (args.output_dir / "correctness.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps({"top": [asdict(row) for row in successful[:10]]}, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
