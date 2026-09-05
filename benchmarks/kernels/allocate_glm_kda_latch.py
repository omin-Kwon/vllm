# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the audited Qwen/Super allocator for paired GLM head curves."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root / "protocol.json").read_text())
    basis = torch.load(root / "basis.pt", weights_only=True)
    layers = sorted(basis)
    basis_pack = root / "allocator_basis.pt"
    torch.save(
        {"omega": torch.stack([basis[k].transpose(-1, -2) for k in layers])}, basis_pack
    )
    sums = {split: {} for split in ("allocation", "heldout")}
    counts = dict(allocation=0, heldout=0)
    nseq = protocol["allocation_sequences"] + protocol["heldout_sequences"]
    for seq in range(nseq):
        p = torch.load(root / f"paired_{seq:03d}.pt", weights_only=True)
        split = p["split"]
        counts[split] += p["measured_tokens"]
        for key in (
            "output_error_sum",
            "joint_dot_sq_sum",
            "scalar_grad_output_error_sum",
        ):
            curve = torch.stack([p["curves"][layer][key] for layer in layers])
            if not torch.isfinite(curve).all() or (curve < 0).any():
                raise ValueError(f"Invalid curve: {seq}/{key}")
            sums[split][key] = sums[split].get(key, 0) + curve
    direct = root / "direct_curves.pt"
    torch.save(
        dict(
            sums["allocation"],
            joint_nstep=counts["allocation"],
            meta={"basis": str(basis_pack)},
        ),
        direct,
    )
    allocator = Path("/disk2/omin/nested_ssm/scale/make_direct_head_allocation.py")
    records = []
    for objective, curve_key in (
        ("fisher", "joint_dot_sq_sum"),
        ("output", "output_error_sum"),
    ):
        for target, mean_rank in ((2, 28), (4, 12), (6, 7), (8, 4)):
            budget = 1024 + 272 * mean_rank
            out = root / f"{objective}_g{mean_rank}.pt"
            command = [
                sys.executable,
                str(allocator),
                "--direct",
                str(direct),
                "--basis",
                str(basis_pack),
                "--curve-key",
                curve_key,
                "--budget",
                str(budget),
                "--V",
                "128",
                "--W",
                "16",
                "--r",
                "128",
                "--anchor",
                "0",
                "--erase",
                "--out",
                str(out),
            ]
            with (root / f"{objective}_g{mean_rank}.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            pack = torch.load(out, weights_only=False)
            ranks, report = pack["m_table"].long(), pack["meta"]
            if ranks.shape != (34, 64) or (ranks < 0).any() or (ranks > 60).any():
                raise ValueError("Invalid recovered rank table")
            variable = torch.where(ranks == 0, 16384, ranks * 272).sum().item()
            if variable > (budget - 1024) * ranks.numel():
                raise ValueError("Recovered table exceeds its budget")
            heldout = {}
            for key in ("joint_dot_sq_sum", "output_error_sum"):
                error = sums["heldout"][key] / counts["heldout"]
                selected = error.gather(-1, ranks[..., None])[..., 0]
                selected[ranks == 0] = 0
                heldout[key] = dict(
                    allocated=float(selected.sum()),
                    uniform=float(error[..., mean_rank].sum()),
                )
            table = dict(
                ranks={str(layer): row for layer, row in zip(layers, ranks.tolist())},
                objective=objective,
                mean_rank_budget=mean_rank,
                target_reduction=target,
                modeled_reduction=17408 / (1024 + variable / ranks.numel()),
                budget_per_head=budget,
                actual_variable_traffic=variable,
                basis_sha256=hashlib.sha256(
                    (root / "basis.pt").read_bytes()
                ).hexdigest(),
                solver_gap=report["certified_or_mip_relative_gap"],
                heldout=heldout,
                allocator_sha256=hashlib.sha256(allocator.read_bytes()).hexdigest(),
                traffic_note=(
                    "paper state-read proxy, full r=128, no anchors; "
                    "padding and unfused refresh are not measured speedup"
                ),
            )
            (root / f"{objective}_g{mean_rank}.json").write_text(
                json.dumps(table, indent=2)
            )
            records.append(
                dict(
                    objective=objective,
                    target=target,
                    budget_g=mean_rank,
                    reduction=table["modeled_reduction"],
                    dense_heads=int((ranks == 0).sum()),
                    min_g=int(ranks[ranks > 0].min()),
                    max_g=int(ranks.max()),
                    solver_gap=table["solver_gap"],
                )
            )
    with (root / "allocations.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (root / "status.json").write_text(json.dumps(dict(stage="allocations_complete")))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
