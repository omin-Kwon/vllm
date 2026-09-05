# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare wall time of eager reference and CUDA-graph latch decode."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

from vllm.models.glm5next.nvidia.kda_latch import BatchedKDALatchCache
from vllm.third_party.flash_linear_attention.ops import kda_latch_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(394)
    results = []
    for batch in (1, 32, 128):
        heads, rank = 8, 32
        omega = torch.linalg.qr(torch.randn(heads, 128, rank, device="cuda")).Q
        ranks = torch.tensor([2, 8, 16, 32, 0, 4, 24, 12], device="cuda")
        state = torch.randn(batch + 1, heads, 128, 128, device="cuda") * 0.01
        state[0].zero_()
        ids = torch.arange(1, batch + 1, device="cuda")
        packed = torch.randn(batch, 4, heads, 128, device="cuda").bfloat16()
        packed[:, 3] -= 3
        raw_beta = torch.randn(batch, 2 * heads, device="cuda").bfloat16()[:, :heads]
        inputs = [packed[:, i] for i in range(4)] + [raw_beta]
        a = torch.zeros(heads, device="cuda")
        bias = torch.zeros(heads, 128, device="cuda")
        old = BatchedKDALatchCache(omega, ranks, capacity=batch)
        new = kda_latch_graph.KDALatchGraphCache(omega, ranks, capacity=batch)
        old_state, new_state = state.clone(), state.clone()

        reference = partial(old.step, old_state, ids, *inputs, a, bias, -5.0)
        optimized = partial(new.step, new_state, ids, *inputs, a, bias, -5.0)

        for _ in range(17):
            reference()
            optimized()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimized()
        row = dict(batch=batch, heads=heads, rank_width=rank)
        for name, function in (
            ("reference_eager", reference),
            ("optimized_eager", optimized),
            ("optimized_graph", graph.replay),
        ):
            torch.accelerator.synchronize()
            start = time.perf_counter()
            for _ in range(64):
                function()
            torch.accelerator.synchronize()
            row[name + "_ms"] = (time.perf_counter() - start) / 64 * 1000
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
