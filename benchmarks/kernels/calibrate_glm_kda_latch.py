# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired Fisher calibration of GLM KDA; disjoint basis and allocation text."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from benchmarks.kernels.glm_kda_calibration_model import forward, hf, load
from benchmarks.kernels.glm_kda_calibration_stats import (
    features,
    paired_curves,
    prefix_qr,
)


class Collector:
    def __init__(self, warmup=256, mmax=60):
        self.warmup, self.mmax = warmup, mmax
        self.current = None
        self.stage = "basis"
        self.cov, self.metric, self.basis, self.curves = {}, {}, {}, {}
        self.sequence_curves = {}
        self.max_transition_error = 0.0
        self.max_fla_relative_error = 0.0
        self.rejected = 0

    def observe(self, q, k, v, g, beta, output):
        layer = self.current
        with torch.no_grad():
            try:
                starts, x, boundary, dense, error = features(q, k, v, g, beta)
            except ValueError:
                torch.save(
                    dict(
                        layer=layer,
                        q=q.cpu(),
                        k=k.cpu(),
                        v=v.cpu(),
                        g=g.cpu(),
                        beta=beta.cpu(),
                    ),
                    "/disk2/omin/kda-latch-results/glm_calibration/failed_features.pt",
                )
                raise
            self.max_transition_error = max(self.max_transition_error, error)
            rel = (dense - output[0].float()).norm() / dense.norm().clamp_min(1e-30)
            if not torch.isfinite(rel) or rel > 0.05:
                raise RuntimeError(f"FLA/reference mismatch at {layer}: {rel.item()}")
            self.max_fla_relative_error = max(self.max_fla_relative_error, float(rel))
            starts, x, boundary = [
                t[self.warmup // 16 :] for t in (starts, x, boundary)
            ]
            if self.stage == "basis":
                c = torch.einsum("uwhk,uwhj->hkj", x.double(), x.double())
                e = torch.einsum("uhvk,uhvj->hkj", starts.double(), starts.double())
                if layer not in self.cov:
                    self.cov[layer], self.metric[layer] = c, e
                else:
                    self.cov[layer] += c
                    self.metric[layer] += e
                return
            omega = self.basis[layer].to(q.device)
            u = starts.double() @ omega[None, ..., : self.mmax].double()
            basis_q, rejected = prefix_qr(u, tol=1e-5)
            self.rejected += int(rejected.sum())
            boundary = boundary.double()

        def backward(grad):
            grad = grad[0, self.warmup :].reshape(boundary.shape).double()
            stats = paired_curves(grad, boundary, basis_q)
            if layer in self.sequence_curves:
                raise RuntimeError(f"Duplicate gradient callback for {layer}")
            self.sequence_curves[layer] = stats

        output.register_hook(backward)

    def fit_basis(self):
        for layer in sorted(self.cov):
            e, c = self.metric[layer], self.cov[layer]
            ridge = (e.diagonal(dim1=-2, dim2=-1).sum(-1) / 128 * 1e-4).clamp_min(1e-12)
            e = e + ridge[:, None, None] * torch.eye(128, device=e.device)
            values, vectors = torch.linalg.eigh(e)
            root = (vectors * values.sqrt()[:, None, :]) @ vectors.transpose(-1, -2)
            inv = (vectors * values.rsqrt()[:, None, :]) @ vectors.transpose(-1, -2)
            product = root @ c @ root
            _, eig = torch.linalg.eigh((product + product.transpose(-1, -2)) / 2)
            omega = inv @ eig.flip(-1)
            # QR changes the coordinates, but preserves each ordered prefix span.
            self.basis[layer] = torch.linalg.qr(omega).Q.float().cpu()
        self.cov.clear()
        self.metric.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/disk2/models/GLM-5.3-Flash-NVFP4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--text",
        type=Path,
        default=Path("/disk2/omin/nested_ssm/scale/results/wiki2_test.txt"),
    )
    parser.add_argument("--basis-sequences", type=int, default=128)
    parser.add_argument("--allocation-start", type=int, default=160)
    parser.add_argument("--allocation-sequences", type=int, default=32)
    parser.add_argument("--heldout-sequences", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    from fla.ops.kda import chunk_kda

    collector = Collector(args.warmup)
    active = False

    def differentiable_kda(query, key, value, g, beta, **kwargs):
        out, state = chunk_kda(
            query, key, value, g=g, beta=beta.float(), safe_gate=True, **kwargs
        )
        if active:
            collector.observe(query, key, value, g, beta, out)
        return out, state

    hf.chunk_kimi_delta_attention = differentiable_kda
    model, head, devices, audit = load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(
        "The capital of France is Paris. Compute 12 + 30. The answer is 42.",
        return_tensors="pt",
    ).input_ids
    gradients = {}
    handles = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer.self_attn, "o_norm"):

            def hook(module, inputs, i=i):
                inputs[0].register_hook(
                    lambda g: gradients.__setitem__(i, float(g.float().square().sum()))
                )

            handles.append(layer.self_attn.o_norm.register_forward_pre_hook(hook))
    logits = forward(model, head, devices, ids[:, :-1])
    loss = F.cross_entropy(
        logits.flatten(0, 1).float(), ids[:, 1:].to(logits.device).flatten()
    )
    loss.backward()
    if len(gradients) != 34 or not all(v > 0 for v in gradients.values()):
        raise RuntimeError(f"Incomplete gradients: {gradients}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(loader=audit, loss=loss.item(), gradients=gradients), indent=2)
    )
    print(
        f"PROBE PASS: loss={loss.item()}, {len(gradients)} KDA gradient hooks",
        flush=True,
    )
    for handle in handles:
        handle.remove()
    del logits, loss
    if args.probe_only:
        return
    if (
        args.allocation_start < args.basis_sequences
        or args.seq_len % 16
        or args.warmup % 16
    ):
        raise ValueError("Require disjoint splits and complete windows")
    if not 0 <= args.warmup < args.seq_len:
        raise ValueError("Invalid warmup")
    root = args.output.parent
    text = args.text.read_text()
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    end = args.allocation_start + args.allocation_sequences + args.heldout_sequences
    if len(tokens) < end * args.seq_len + 1:
        raise ValueError("Insufficient calibration corpus")
    meta = dict(
        model=args.model,
        text=str(args.text),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        seq_len=args.seq_len,
        warmup=args.warmup,
        window=16,
        basis_sequences=args.basis_sequences,
        allocation_start=args.allocation_start,
        allocation_sequences=args.allocation_sequences,
        heldout_sequences=args.heldout_sequences,
        arithmetic="same-checkpoint dequantized BF16; no activation quantization",
        objective="same-token ideal-LS paired Fisher and output-error ablation",
        coefficient_rank=128,
        metric_ridge_in_runtime=1e-4,
        runtime_ridge_in_objective=False,
    )
    protocol_path = root / "protocol.json"
    if (
        args.resume
        and protocol_path.exists()
        and json.loads(protocol_path.read_text()) != meta
    ):
        raise ValueError("Cannot resume calibration with a different protocol")
    protocol_path.write_text(json.dumps(meta, indent=2))
    torch.save(torch.tensor(tokens[: end * args.seq_len + 1]), root / "tokens.pt")
    for i, layer in enumerate(model.layers):
        if hasattr(layer.self_attn, "o_norm"):
            layer.self_attn.register_forward_pre_hook(
                lambda m, inputs, i=i: setattr(collector, "current", i)
            )
    active = True
    basis_path = root / "basis.pt"
    if args.resume and basis_path.exists():
        collector.basis = torch.load(basis_path, weights_only=True)
    else:
        with torch.no_grad():
            for sequence in range(args.basis_sequences):
                chunk = torch.tensor(
                    tokens[sequence * args.seq_len : (sequence + 1) * args.seq_len]
                )[None]
                forward(model, head, devices, chunk, grad=False)
                print(f"BASIS {sequence + 1}/{args.basis_sequences}", flush=True)
                (root / "status.json").write_text(
                    json.dumps(
                        dict(
                            stage="basis", done=sequence + 1, total=args.basis_sequences
                        )
                    )
                )
        collector.fit_basis()
        torch.save(collector.basis, basis_path)
    collector.stage = "paired"
    for sequence in range(args.allocation_sequences + args.heldout_sequences):
        save = root / f"paired_{sequence:03d}.pt"
        if args.resume and save.exists():
            continue
        start = (args.allocation_start + sequence) * args.seq_len
        chunk = torch.tensor(tokens[start : start + args.seq_len + 1])[None]
        collector.sequence_curves = {}
        logits = forward(model, head, devices, chunk[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, 1).float(),
            chunk[:, 1:].to(logits.device).flatten(),
            reduction="sum",
        )
        loss.backward()
        if len(collector.sequence_curves) != 34:
            raise RuntimeError("Not all KDA layers contributed paired gradients")
        torch.save(
            dict(
                curves=collector.sequence_curves,
                nll=float(loss) / args.seq_len,
                measured_tokens=args.seq_len - args.warmup,
                block=args.allocation_start + sequence,
                split="allocation"
                if sequence < args.allocation_sequences
                else "heldout",
            ),
            save,
        )
        del logits, loss
        print(
            f"PAIRED {sequence + 1}/"
            f"{args.allocation_sequences + args.heldout_sequences}",
            flush=True,
        )
        (root / "status.json").write_text(
            json.dumps(
                dict(
                    stage="paired",
                    done=sequence + 1,
                    total=args.allocation_sequences + args.heldout_sequences,
                )
            )
        )
    (root / "validation.json").write_text(
        json.dumps(
            dict(
                max_transition_error=collector.max_transition_error,
                max_fla_relative_error=collector.max_fla_relative_error,
                rejected_qr_columns=collector.rejected,
            ),
            indent=2,
        )
    )
    (root / "status.json").write_text(json.dumps(dict(stage="calibration_complete")))


if __name__ == "__main__":
    main()
