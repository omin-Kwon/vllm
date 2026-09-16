# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered W16 CUDA-graph latency, native vs parallel Replay vs optional old Replay.

One synthetic KDA layer, H64/K128/V128, FP32 state and raw BF16 inputs.
Includes ownership and state handoff kernels in replay modes. The optional old
core retains the pooled-state lifecycle used by Sketch. These are kernel
measurements, not model throughput. No artificial cursor resets are timed.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.ops.glm_window.cache import ReplayCache
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda


def legacy_class(path):
    spec = importlib.util.spec_from_file_location("legacy_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class LegacyCache(ReplayCache):
        def __init__(self, heads, capacity, device):
            super().__init__(heads, capacity, device, replay_factors=False)

        def _decode(self, state, indices, slots, q, k, v, gate, beta, a_log, bias):
            p = self.pool
            out = torch.zeros_like(v)
            module.replay_step[(q.shape[0], self.heads, 4)](
                q,
                k,
                v,
                gate,
                beta,
                a_log,
                bias,
                slots,
                p.pos,
                p.state,
                p.k,
                p.v,
                p.log_a,
                p.beta,
                out,
                self.heads,
                32,
                num_warps=4,
            )
            return out

    return LegacyCache


def measure(batch, mode, cache_cls):
    torch.manual_seed(91)
    h = 64
    state = torch.randn(batch + 1, h, 128, 128, device="cuda") * 0.1
    ids = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    data = [
        torch.randn(batch, h, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    data[3].sub_(4)
    beta = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    cache = cache_cls(h, batch, state.device) if mode != "native" else None

    def step():
        if cache is not None:
            return cache.step(state, ids, *data, beta, a, bias)
        return fused_recurrent_kda(
            q=data[0][None],
            k=data[1][None],
            v=data[2][None],
            g=data[3][None],
            beta=beta[None],
            initial_state=state,
            ssm_state_indices=ids,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
            compute_gate=True,
            a_log=a,
            g_bias=bias,
            lower_bound=-5.0,
        )[0]

    for _ in range(32):
        step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    # Capture does not execute the recorded updates.
    for _ in range(32):
        graph.replay()
    torch.accelerator.synchronize()
    if cache is not None:
        assert (cache.pool.pos == 0).all()
    assert torch.isfinite(out).all()
    samples = [[] for _ in range(16)]
    events = []
    for _ in range(20):
        for phase in range(16):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            events.append((phase, start, end))
    torch.accelerator.synchronize()
    for phase, start, end in events:
        samples[phase].append(start.elapsed_time(end) * 1000)
    medians = [statistics.median(x) for x in samples]
    return dict(
        batch=batch,
        mode=mode,
        phase_us=medians,
        nonflush_us=statistics.mean(medians[:15]),
        flush_us=medians[15],
        window_mean_us=statistics.mean(medians),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 64, 128])
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--pivots", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    classes = {"native": ReplayCache, "parallel_replay": ReplayCache}
    if args.legacy_source:
        classes["sequential_replay"] = legacy_class(args.legacy_source)
    pack = None
    if args.checkpoint:
        from vllm.model_executor.layers.mamba.ops.glm_window.sketch import (
            SketchCache,
            load_checkpoint,
        )

        pack = load_checkpoint(str(args.checkpoint))
        classes = {"native": ReplayCache}
        for layer in args.layers or sorted(pack["frames"]):

            def factory(heads, capacity, device, layer=layer):
                return SketchCache(
                    pack["frames"][layer].to(device),
                    pack["ranks"][layer].to(device),
                    capacity=capacity,
                    pivots=args.pivots,
                )

            classes[f"sketch_p{args.pivots}_layer{layer}"] = factory
    rows = []
    for batch in args.batches:
        for mode, cls in classes.items():
            row = measure(batch, mode, cls)
            rows.append(row)
            print(json.dumps(row), flush=True)
            torch.accelerator.empty_cache()
    args.out.write_text(
        json.dumps(
            dict(
                scope="synthetic_one_layer_including_lifecycle",
                checkpoint=str(args.checkpoint) if args.checkpoint else None,
                checkpoint_meta=pack["meta"] if pack else None,
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
