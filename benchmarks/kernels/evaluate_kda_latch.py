# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AFM KDA dense versus r=128 LatchSSM, on disjoint WikiText continuations.

This is a teacher-forced small-model smoke evaluation, not a task benchmark
or a GLM accuracy claim. The pretrained softplus gate is kept unchanged.
Calibration uses the full-transition effective query, including the erase.
The dense arm calls the model's original FLA kernels with ReplaySSM off.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops.kda_latch import KDALatchState


class ModelHook:
    def __init__(self, model):
        import fla.layers.kda as layer_module

        self.module = layer_module
        self.original_chunk = layer_module.chunk_kda
        self.original_recurrent = layer_module.fused_recurrent_kda
        self.layers = {
            layer.self_attn.A_log.data_ptr(): i
            for i, layer in enumerate(model.model.layers)
            if isinstance(layer.self_attn, layer_module.KimiDeltaAttention)
        }
        self.mode = "dense"
        self.rank = 128
        self.stats = {}
        self.bases = {}
        self.reset()
        layer_module.chunk_kda = self.chunk
        layer_module.fused_recurrent_kda = self.recurrent

    def close(self):
        self.module.chunk_kda = self.original_chunk
        self.module.fused_recurrent_kda = self.original_recurrent

    def reset(self):
        self.runtime = {}
        self.window = {}

    def chunk(self, **kwargs):
        return self.original_chunk(**kwargs)

    @torch.no_grad()
    def recurrent(self, **kw):
        q, k, v = kw["q"], kw["k"], kw["v"]
        if q.shape[1] != 1:
            return self.original_recurrent(**kw)
        layer = self.layers[kw["A_log"].data_ptr()]
        if self.mode == "calibrate":
            self.capture(layer, kw)
        if self.mode != "ours":
            return self.original_recurrent(**kw)
        assert kw["state_v_first"] and kw["use_qk_l2norm_in_kernel"]
        assert kw["use_gate_in_kernel"] and kw["use_beta_sigmoid_in_kernel"]
        assert not kw["allow_neg_eigval"] and kw.get("cu_seqlens") is None
        if layer not in self.runtime:
            initial = kw["initial_state"]
            assert initial is not None and initial.dtype == torch.float32
            self.runtime[layer] = KDALatchState(
                initial, self.bases[layer][..., : self.rank].contiguous()
            )
        state = self.runtime[layer]
        lower = kw.get("lower_bound")
        out = state.step(
            q[:, 0],
            k[:, 0],
            v[:, 0],
            kw["g"][:, 0],
            kw["beta"][:, 0],
            kw["A_log"],
            kw["dt_bias"],
            safe_gate=lower is not None,
            lower_bound=-5 if lower is None else lower,
        )
        # FLA's cache carries the boundary state; only this hook consumes it
        # on subsequent decode calls. Multi-token prefill is never interleaved.
        return out[:, None], state.state

    def capture(self, layer, kw):
        if layer not in self.stats:
            heads = kw["q"].shape[2]
            zeros = torch.zeros(heads, 128, 128, dtype=torch.float64, device="cuda")
            self.stats[layer] = dict(
                cx=zeros.clone(), e=zeros.clone(), windows=0, tokens=0
            )
        stat = self.stats[layer]
        if layer not in self.window or self.window[layer]["pos"] == 16:
            state = kw["initial_state"][0].double()
            stat["e"] += state.transpose(-1, -2) @ state
            stat["windows"] += 1
            self.window[layer] = dict(
                pos=0,
                product=torch.eye(128, dtype=torch.float64, device="cuda")
                .expand(state.shape[0], 128, 128)
                .clone(),
            )
        win = self.window[layer]
        q, k = kw["q"][0, 0].double(), kw["k"][0, 0].double()
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**0.5
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        beta = kw["beta"][0, 0].double().sigmoid()
        raw = kw["g"][0, 0].double() + kw["dt_bias"].double().reshape_as(k)
        amp = kw["A_log"].double().exp().reshape(-1, 1)
        lower = kw.get("lower_bound")
        log_a = (
            -amp * F.softplus(raw) if lower is None else lower * (amp * raw).sigmoid()
        )
        # M = (I-beta*k*k.T) D, preserving the order of decay then erase.
        transition = torch.eye(128, device="cuda", dtype=torch.float64) - (
            beta[:, None, None] * k[:, :, None] * k[:, None, :]
        )
        transition *= log_a.exp()[:, None, :]
        win["product"] = transition @ win["product"]
        effective = (win["product"].transpose(-1, -2) @ q[:, :, None])[..., 0]
        stat["cx"] += effective[:, :, None] * effective[:, None, :]
        stat["tokens"] += 1
        win["pos"] += 1

    def bake(self):
        for layer, stat in self.stats.items():
            e = stat["e"] / stat["windows"]
            eta = 1e-4 * e.diagonal(dim1=-2, dim2=-1).sum(-1) / 128
            e += eta[:, None, None] * torch.eye(128, device=e.device)
            val, vec = torch.linalg.eigh(e)
            root = (vec * val.clamp_min(1e-12).sqrt()[:, None, :]) @ vec.transpose(
                -1, -2
            )
            inv = (vec * val.clamp_min(1e-12).rsqrt()[:, None, :]) @ vec.transpose(
                -1, -2
            )
            _, p = torch.linalg.eigh(root @ stat["cx"] @ root)
            omega = inv @ p.flip(-1)
            self.bases[layer] = torch.linalg.qr(omega)[0].contiguous()


@torch.inference_mode()
def run_continuation(model, hook, ids, prefix, length):
    hook.reset()
    out = model(ids[:, :prefix], use_cache=True)
    cache = out.past_key_values
    scores = [out.logits[0, -1].float().cpu()]
    for t in range(length - 1):
        out = model(
            ids[:, prefix + t : prefix + t + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        scores.append(out.logits[0, -1].float().cpu())
    return torch.stack(scores)


def metrics(logits, labels, reference=None):
    loss = F.cross_entropy(logits, labels).item()
    result = dict(
        tokens=labels.numel(),
        nll=loss,
        perplexity=math.exp(loss),
        next_token_accuracy=(logits.argmax(-1) == labels).float().mean().item(),
    )
    if reference is not None:
        result.update(
            dense_top1_agreement=(logits.argmax(-1) == reference.argmax(-1))
            .float()
            .mean()
            .item(),
            mean_abs_logit_error=(logits - reference).abs().mean().item(),
            max_abs_logit_error=(logits - reference).abs().max().item(),
            mean_kl_from_dense=F.kl_div(
                logits.log_softmax(-1),
                reference.log_softmax(-1),
                reduction="batchmean",
                log_target=True,
            ).item(),
        )
    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/disk2/omin/models/AFM-4.5B-Base-KDA-NoPE")
    parser.add_argument(
        "--text", default="/disk2/omin/nested_ssm/scale/results/wiki2_test.txt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", type=int, default=128)
    parser.add_argument("--decode", type=int, default=65)
    parser.add_argument("--calibration-segments", type=int, default=2)
    parser.add_argument("--evaluation-segments", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[128, 32, 8])
    args = parser.parse_args()
    torch.manual_seed(71)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    text = Path(args.text).read_text()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    hook = ModelHook(model)
    span = args.prefix + args.decode
    cal_offsets = [i * span for i in range(args.calibration_segments)]
    eval_offsets = [4096 + i * span for i in range(args.evaluation_segments)]
    assert max(cal_offsets) + span <= min(eval_offsets)
    assert max(eval_offsets) + span <= ids.shape[1]
    report = dict(
        model=args.model,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        dependencies={
            package: importlib.metadata.version(package)
            for package in ("triton", "transformers", "fla-core")
        },
        model_config_sha256=hashlib.sha256(
            Path(args.model, "config.json").read_bytes()
        ).hexdigest(),
        method="r=128; uniform head rank; metric-ridge=1e-4; W=16",
        gate="original AFM softplus; no gate replacement",
        state_dtype="float32",
        scope=(
            "teacher-forced WikiText continuation smoke; "
            "not task accuracy or GLM evaluation"
        ),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        calibration_offsets=cal_offsets,
        evaluation_offsets=eval_offsets,
        prefix=args.prefix,
        decode=args.decode,
        kda_layers=len(hook.layers),
        arms={},
    )
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        hook.mode = "calibrate"
        for offset in cal_offsets:
            run_continuation(
                model,
                hook,
                ids[:, offset : offset + span].cuda(),
                args.prefix,
                args.decode,
            )
            print(f"calibration offset={offset} done", flush=True)
        hook.bake()
        torch.save(
            {k: v.cpu() for k, v in hook.bases.items()}, dest.with_suffix(".basis.pt")
        )
        baseline = None
        labels = torch.cat([ids[0, o + args.prefix : o + span] for o in eval_offsets])
        for rank in [0] + args.ranks:
            hook.mode, hook.rank = ("dense" if rank == 0 else "ours"), rank
            scores = []
            for offset in eval_offsets:
                scores.append(
                    run_continuation(
                        model,
                        hook,
                        ids[:, offset : offset + span].cuda(),
                        args.prefix,
                        args.decode,
                    )
                )
                print(f"rank={rank} offset={offset} done", flush=True)
            logits = torch.cat(scores)
            if rank == 0:
                baseline = logits
            name = "dense" if rank == 0 else f"latch_g{rank}"
            report["arms"][name] = metrics(logits, labels, baseline)
            report["elapsed_seconds"] = time.monotonic() - start
            dest.write_text(json.dumps(report, indent=2) + "\n")
            print(name, report["arms"][name], flush=True)
    finally:
        hook.close()


if __name__ == "__main__":
    main()
