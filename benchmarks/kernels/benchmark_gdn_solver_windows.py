# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare solver/representation choices through real GDN windows."""

import argparse
import json
from pathlib import Path

import torch
from benchmark_gdn_solvers import Candidate, measure
from gdn_qr_research import QRWorkspace
from gdn_unfolded_research import UnfoldedWorkspace

from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
    FlushWorkspace,
)
from vllm.third_party.flash_linear_attention.ops.gdn_step_full_cuda import step


class World:
    def __init__(self, kind, initial, h, widths, m, g, library, small):
        self.kind = kind
        ns, hv, k, _ = initial.shape
        batch = ns - 1
        self.state = initial.clone()
        self.writes = torch.zeros(ns, hv, 16, k, device="cuda")
        self.keys = torch.zeros(ns, h, 16, k, device="cuda")
        self.gates = torch.zeros(ns, hv, 16, device="cuda")
        self.factors = torch.zeros(ns, hv, 16, g, device="cuda")
        self.u = torch.full((ns, hv, g, k), float("nan"), device="cuda")
        self.phi = torch.full_like(self.u, float("nan"))
        self.beta = torch.zeros(ns, hv, 16, device="cuda")
        self.widths = torch.tensor(widths, device="cuda", dtype=torch.int32)
        self.mapping = torch.arange(ns, device="cuda", dtype=torch.int32)
        self.indices = torch.arange(1, ns, device="cuda", dtype=torch.int32)
        self.rows = torch.cat([self.indices, self.indices.new_tensor([batch])])
        self.slots = list(range(1, ns))
        if small:
            self.mapping = self.mapping.new_tensor([0, 3, 1, 2, 4])
            self.indices = self.indices.new_tensor([3, 0, 2, 1])
            self.rows = self.rows.new_tensor([3, 1, 2, 0, 3])
            self.slots = [3, 1, 2]
        self.positions = [torch.full_like(self.indices, p) for p in range(16)]
        self.out = torch.zeros(batch, hv, k, device="cuda", dtype=torch.bfloat16)
        cls = {
            "unfolded": UnfoldedWorkspace,
            "qr": QRWorkspace,
            "qr_tensor": QRWorkspace,
            "qr_mgs": QRWorkspace,
        }.get(kind, FlushWorkspace)
        self.workspace = cls(
            batch,
            h,
            hv,
            g,
            "cuda",
            ridge=0.1,
            grid=2 if small else None,
            **(
                {"split": kind == "qr_tensor", "mgs": kind == "qr_mgs"}
                if kind.startswith("qr")
                else {}
            ),
        )
        self.args = (
            self.state,
            self.writes,
            self.keys,
            self.gates,
            self.rows,
            self.mapping,
            self.widths,
            self.beta,
            self.u,
            self.phi,
        )
        self.candidate = Candidate(library, m, g, batch, hv) if kind == "dx" else None
        self.dx_method = (
            "cusolverdx_padded"
            if m == 127
            else "cusolverdx"
            if m == 64
            else "cusolverdx_auto"
        )
        self.fallback = any(w not in (0, m, 128) for w in widths)
        self.workspace.refresh(*self.args)
        if self.candidate:
            self.solve()
        self.width_list = widths

    def solve(self):
        if self.candidate:
            if self.fallback:
                self.workspace._run(3, *self.args)
            self.candidate.run(
                self.dx_method,
                self.workspace.scratch,
                self.phi,
                self.rows,
                self.mapping,
                self.widths,
            )
        else:
            self.workspace._run(3, *self.args)

    def flush(self):
        if self.candidate:
            self.workspace._run(1, *self.args)
            self.workspace._run(2, *self.args)
            self.solve()
        else:
            self.workspace.flush(*self.args)

    def token(self, t, inputs):
        mixed, a, b, a_log, bias = inputs
        fn = self.workspace.step if self.kind == "unfolded" else step
        kwargs = {"beta_ring": self.beta}
        if self.kind == "unfolded":
            kwargs["inverse"] = self.workspace.inverse
        fn(
            mixed[t],
            a[t],
            b[t],
            a_log,
            bias,
            self.out,
            self.state,
            self.writes,
            self.keys,
            self.gates,
            self.indices,
            self.positions[t],
            self.u,
            self.phi,
            self.widths,
            self.factors,
            self.mapping,
            128**-0.5,
            **kwargs,
        )

    def cycle(self, inputs):
        for t in range(16):
            self.token(t, inputs)
        self.flush()


def run(
    m,
    g,
    library,
    small=True,
    gate_dtype=torch.float32,
    kinds=("current", "unfolded", "dx"),
):
    torch.manual_seed(319)
    batch, h = (4, 3) if small else (128, 16)
    hv, ns = 3 * h, batch + 1
    widths = (
        [0, 1, m, max(1, m - 3), m, m - 1, 0, 128 if g == 128 else m, m]
        if small
        else [m] * hv
    )
    initial = torch.randn(ns, hv, 128, 128, device="cuda") * 0.03
    inputs = (
        torch.randn(
            16, batch, 2 * h * 128 + hv * 128, device="cuda", dtype=torch.bfloat16
        ),
        torch.full((16, batch, hv), -3.0, device="cuda", dtype=gate_dtype),
        torch.randn(16, batch, hv, device="cuda", dtype=gate_dtype),
        torch.zeros(hv, device="cuda", dtype=gate_dtype),
        torch.zeros(hv, device="cuda", dtype=gate_dtype),
    )
    worlds = [World(kind, initial, h, widths, m, g, library, small) for kind in kinds]
    base = worlds[0]
    errors = {
        w.kind: dict(output=0.0, state=0.0, effective_phi=0.0) for w in worlds[1:]
    }
    for window in range(3):
        for t in range(16):
            for w in worlds:
                w.token(t, inputs)
            for w in worlds[1:]:
                torch.testing.assert_close(w.out, base.out, atol=4e-3, rtol=4e-3)
                errors[w.kind]["output"] = max(
                    errors[w.kind]["output"],
                    (w.out.float() - base.out.float()).abs().max().item(),
                )
                torch.testing.assert_close(w.writes, base.writes, atol=5e-5, rtol=4e-3)
        if small and "unfolded" in kinds:
            unfolded = next(w for w in worlds if w.kind == "unfolded")
            for s in unfolded.slots:
                c = int(unfolded.mapping[s])
                for head, width in enumerate(widths):
                    if 0 < width < 128:
                        im = unfolded.workspace.inverse[c, head, :width, :width].T
                        torch.testing.assert_close(
                            (im @ unfolded.factors[c, head, :, :width].T).T,
                            base.factors[c, head, :, :width],
                            atol=5e-5,
                            rtol=4e-3,
                        )
        for qr in [w for w in worlds if small and w.kind.startswith("qr")]:
            for s in qr.slots:
                c = int(qr.mapping[s])
                for head, width in enumerate(widths):
                    if 0 < width < 128:
                        torch.testing.assert_close(
                            qr.u[c, head, :width].T @ qr.factors[c, head, :, :width].T,
                            base.u[c, head, :width].T
                            @ base.factors[c, head, :, :width].T,
                            atol=5e-5,
                            rtol=4e-3,
                        )
        for w in worlds:
            w.flush()
        for w in worlds[1:]:
            torch.testing.assert_close(w.state, base.state, atol=5e-5, rtol=4e-3)
            errors[w.kind]["state"] = max(
                errors[w.kind]["state"], (w.state - base.state).abs().max().item()
            )
            if small:
                for s in w.slots:
                    c = int(w.mapping[s])
                    for head, width in enumerate(widths):
                        if 0 < width < 128:
                            effective = w.phi[c, head, :width]
                            if w.kind == "unfolded":
                                im = w.workspace.inverse[c, head, :width, :width].T
                                effective = im @ effective
                            expected = base.phi[c, head, :width]
                            if w.kind.startswith("qr"):
                                effective = w.u[c, head, :width].T @ effective
                                expected = base.u[c, head, :width].T @ expected
                            torch.testing.assert_close(
                                effective, expected, atol=5e-5, rtol=4e-3
                            )
                            errors[w.kind]["effective_phi"] = max(
                                errors[w.kind]["effective_phi"],
                                (effective - expected).abs().max().item(),
                            )
                        elif width == 128:
                            assert w.phi[c, head].isnan().all()
    result = dict(
        m=m,
        g=g,
        batch=batch,
        gate_dtype=str(gate_dtype),
        windows_checked=3,
        errors=errors,
    )
    if not small:
        result["timings"] = {
            w.kind: dict(
                solve_us=measure(w.solve),
                flush_us=measure(w.flush),
                cycle_us=measure(lambda w=w: w.cycle(inputs)),
            )
            for w in worlds
        }
    for w in worlds:
        if w.candidate:
            w.candidate.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--widths", nargs="+", type=int, default=[8, 16, 32, 80])
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=["current", "unfolded", "dx", "qr", "qr_tensor", "qr_mgs"],
        default=["current", "unfolded", "dx"],
    )
    args = p.parse_args()
    if args.kinds[0] != "current":
        p.error("--kinds must start with current as the reference")
    results = []
    for m in args.widths:
        for dtype in [torch.float32] if args.bench else [torch.float32, torch.bfloat16]:
            result = run(
                m, (m + 3) // 4 * 4, args.library, not args.bench, dtype, args.kinds
            )
            print(json.dumps(result), flush=True)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
