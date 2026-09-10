# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed-metric GDN reads separately from the current dynamic method."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from benchmark_gdn_solver_windows import World
from benchmark_gdn_solvers import measure
from gdn_fixed_metric_research import FixedMetricWorkspace, coefficients_from_metric


class FixedWorld(World):
    def __init__(self, initial, h, widths, g, metric, shared, small):
        super().__init__("current", initial, h, widths, g, g, None, small)
        self.kind = "fixed_shared" if shared else "fixed_private"
        self.metric = metric
        self.workspace = FixedMetricWorkspace(
            len(self.indices),
            h,
            initial.shape[1],
            g,
            initial.device,
            widths=self.widths,
            grid=2 if small else None,
        )
        self.coefficients = coefficients_from_metric(metric, self.widths, g)
        self.phi = self.coefficients.unsqueeze(0).expand(initial.shape[0], -1, -1, -1)
        if not shared:
            self.phi = self.phi.clone()
        self.args = (*self.args[:-1], self.phi)
        self.workspace.refresh(*self.args)


def make_worlds(g, batch, small=False, gate_dtype=torch.float32, widths=None):
    torch.manual_seed(923)
    h = 3 if small else 16
    hv = h * 3
    if widths is None:
        widths = [0, 1, g, max(1, g - 3), g, g - 1, 0, g, g] if small else [g] * hv
    initial = torch.randn(batch + 1, hv, 128, 128, device="cuda") * 0.03
    # A non-diagonal synthetic calibration metric; timings do not measure accuracy.
    sample = torch.randn(hv, 128, 128, device="cuda") * 0.1
    metric = sample.transpose(-1, -2) @ sample + torch.eye(128, device="cuda") * 0.1
    inputs = (
        torch.randn(
            16, batch, 2 * h * 128 + hv * 128, device="cuda", dtype=torch.bfloat16
        ),
        torch.randn(16, batch, hv, device="cuda", dtype=gate_dtype) * 0.2 - 3.0,
        torch.randn(16, batch, hv, device="cuda", dtype=gate_dtype),
        torch.zeros(hv, device="cuda", dtype=gate_dtype),
        torch.zeros(hv, device="cuda", dtype=gate_dtype),
    )
    worlds = [
        World("coalesced", initial, h, widths, g, g, None, small),
        FixedWorld(initial, h, widths, g, metric, False, small),
        FixedWorld(initial, h, widths, g, metric, True, small),
    ]
    return worlds, inputs


def check_windows(worlds, inputs, windows=3):
    base, private, shared = worlds
    out_error = 0.0
    for _ in range(windows):
        for t in range(16):
            for world in worlds:
                world.token(t, inputs)
            # Shared P0 and replicated P0 must execute the same new method.
            torch.testing.assert_close(shared.out, private.out, atol=0, rtol=0)
            for world in worlds[1:]:
                torch.testing.assert_close(world.writes, base.writes, atol=0, rtol=0)
                out_error = max(
                    out_error, (world.out.float() - base.out.float()).abs().max().item()
                )
        for world in worlds:
            world.flush()
        for world in worlds[1:]:
            # Removing Gram must not change exact state folding or embedded U.
            torch.testing.assert_close(world.state, base.state, atol=0, rtol=0)
            torch.testing.assert_close(world.u, base.u, atol=0, rtol=0, equal_nan=True)
    return {
        "state_max_abs": 0.0,
        "shared_vs_private_output_max_abs": 0.0,
        "changed_method_vs_dynamic_output_max_abs": out_error,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--batches", type=int, nargs="+", default=[128])
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    if args.rounds < 1:
        p.error("--rounds must be positive")
    records = []
    for batch in args.batches:
        for g in args.widths:
            worlds, inputs = make_worlds(g, batch)
            errors = check_windows(worlds, inputs)
            timings = {}
            for world in worlds:
                # Three windows initialize all extensions before graph timing.
                nonflush = (
                    sum(
                        measure(lambda t=t, w=world, x=inputs: w.token(t, x))
                        for t in range(15)
                    )
                    / 15
                )
                timings[world.kind] = {
                    "nonflush_us": nonflush,
                    "rounds": [],
                    "phi_storage_bytes": world.phi.untyped_storage().nbytes(),
                    "scratch_bytes": world.workspace.scratch.untyped_storage().nbytes(),
                }
            for round_index in range(args.rounds):
                order = worlds if round_index % 2 == 0 else list(reversed(worlds))
                for world in order:
                    timings[world.kind]["rounds"].append(
                        {
                            "flush_us": measure(world.flush),
                            "cycle_us": measure(lambda w=world, x=inputs: w.cycle(x)),
                        }
                    )
            for timing in timings.values():
                for key in ("flush_us", "cycle_us"):
                    timing[key] = statistics.median(r[key] for r in timing["rounds"])
            record = {
                "batch": batch,
                "h": 16,
                "hv": 48,
                "m": g,
                "g": g,
                "windows_checked": 3,
                "errors": errors,
                "timings": timings,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
