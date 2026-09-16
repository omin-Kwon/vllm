# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed GLM model gate: staggered arrivals force real mixed decode/prefill.

Run each mode in a separate process so CUDA graphs and state never cross modes.
This is an integration gate, not a benchmark accuracy measurement.
"""

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path


class WindowAudit:
    def install_replay_probe(self):
        """Arithmetic shadow for eager diagnostics; returns our original output."""
        import torch

        from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

        def wrap(cache):
            original = cache._decode
            shadow = torch.zeros(
                cache.capacity + 1,
                cache.heads,
                128,
                128,
                device=cache.pool.state.device,
                dtype=torch.float32,
            )
            stats = torch.zeros(3, device=shadow.device)
            cache.probe_stats = stats

            def checked(slots, q, k, v, gate, beta, a_log, bias):
                ids = (slots + 1).clamp_min(0).long()
                pool_ids = slots.clamp_min(0).long()
                initial = torch.where(
                    cache.fresh[: len(slots), None, None, None],
                    cache.pool.state[pool_ids],
                    shadow[ids],
                )
                shadow.index_copy_(0, ids, initial)
                out = original(slots, q, k, v, gate, beta, a_log, bias)
                native_out = torch.zeros_like(out).unsqueeze(0)
                fused_recurrent_kda(
                    q=q.unsqueeze(0),
                    k=k.unsqueeze(0),
                    v=v.unsqueeze(0),
                    g=gate.unsqueeze(0),
                    beta=beta.unsqueeze(0),
                    a_log=a_log,
                    g_bias=bias,
                    initial_state=shadow,
                    ssm_state_indices=ids.to(torch.int32),
                    cu_seqlens=torch.arange(
                        len(slots) + 1, device=slots.device, dtype=torch.int32
                    ),
                    out=native_out,
                    use_qk_l2norm_in_kernel=True,
                    sigmoid_beta=True,
                    compute_gate=True,
                    lower_bound=-5.0,
                )
                delta = out.float() - native_out[0].float()
                relative = delta.norm() / native_out.float().norm().clamp_min(1e-12)
                state_delta = cache.pool.state[pool_ids] - shadow[ids]
                state_relative = state_delta.flatten(1).norm(dim=1) / shadow[
                    ids
                ].flatten(1).norm(dim=1).clamp_min(1e-12)
                flush = (slots >= 0) & (cache.pool.pos[pool_ids] == 15)
                state_relative = torch.where(flush, state_relative, 0.0).max()
                stats.copy_(
                    torch.maximum(
                        stats,
                        torch.stack((delta.abs().max(), relative, state_relative)),
                    )
                )
                return out

            cache._decode = checked

        count = 0
        for module in self.get_model().modules():
            cache = getattr(module, "_window_cache", None)
            if cache is not None:
                assert type(cache).__name__ == "ReplayCache"
                wrap(cache)
                count += 1
        return count

    def window_snapshot(self):
        import torch

        from vllm.models.glm5next.common import kda
        from vllm.v1.worker.gpu import model_runner

        layers, packed = [], 0
        for module in self.get_model().modules():
            packed += sum(
                p.numel()
                for p in module.parameters(recurse=False)
                if p.dtype == torch.uint8
            )
            if type(module).__name__ != "Glm5NextLinearAttention":
                continue
            cache = module._window_cache
            row = dict(
                layer=module.layer_idx,
                state_dtype=str(module.kv_cache[1].dtype),
                prefill=module.kda_prefill_backend,
                window=None if cache is None else type(cache).__name__,
            )
            if cache is not None:
                row.update(
                    decoded_tokens=int(cache.counts[0]),
                    mixed_decode_tokens=cache.mixed_decode_tokens,
                    key_ring=str(cache.pool.k.dtype),
                    value_ring=str(cache.pool.v.dtype),
                    log_decay_ring=str(cache.pool.log_a.dtype),
                    beta_ring=str(cache.pool.beta.dtype),
                    capacity=cache.capacity,
                    pivot_count=getattr(cache, "pivots", None),
                )
                if hasattr(cache, "probe_stats"):
                    row["native_probe"] = cache.probe_stats.tolist()
            layers.append(row)
        sources = {}
        for module in (kda, model_runner):
            path = Path(inspect.getfile(module)).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(layers=layers, packed_uint8_bytes=packed, sources=sources)


def main():
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["dense", "replay", "sketch"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--probe-replay", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.probe_replay and args.mode != "replay":
        parser.error("--probe-replay requires --mode replay")
    extra = {"kda_prefill_backend": "auto"}
    if args.mode != "dense":
        extra["kda_window"] = {"mode": args.mode, "window": 16}
        if args.mode == "sketch":
            extra["kda_window"].update(checkpoint=args.checkpoint, pivots=4)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        mamba_ssm_cache_dtype="float32",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=args.probe_replay,
        enable_flashinfer_autotune=False,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config=extra,
        worker_extension_cls="glm_window_smoke.WindowAudit",
        compilation_config=dict(
            mode=0,
            cudagraph_mode="NONE" if args.probe_replay else "FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        ),
    )
    if args.probe_replay:
        assert llm.collective_rpc("install_replay_probe")[0] == 34
    tokenizer = llm.get_tokenizer()
    questions = [
        "What is 17 times 23? Explain the calculation.",
        "Explain why the sum of the first n odd integers is n squared.",
        "Solve x squared minus 5x plus 6 equals zero and check both roots.",
        "Give Python code that computes the greatest common divisor of two integers.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    prompts = [p["input_ids"] if isinstance(p, Mapping) else p for p in prompts]
    assert all(isinstance(token, int) for p in prompts for token in p)
    engine = llm.llm_engine
    results, step = {}, 0
    try:
        while step <= 18 or engine.has_unfinished_requests():
            if step in (0, 6, 12, 18):
                i = step // 6
                engine.add_request(
                    str(i),
                    {"prompt_token_ids": prompts[i]},
                    SamplingParams(
                        temperature=0,
                        max_tokens=96,
                        min_tokens=96,
                        ignore_eos=True,
                        logprobs=1,
                        seed=0,
                    ),
                )
            for output in engine.step():
                if output.finished:
                    item = output.outputs[0]
                    results[output.request_id] = dict(
                        text=item.text,
                        token_ids=list(item.token_ids),
                        finish_reason=item.finish_reason,
                        cumulative_logprob=item.cumulative_logprob,
                    )
            step += 1
            if step > 512:
                raise RuntimeError("Smoke requests failed to complete")
        audit = llm.collective_rpc("window_snapshot")[0]
        assert len(results) == 4 and all(
            len(x["token_ids"]) == 96 for x in results.values()
        )
        assert (
            len(audit["layers"]) == 34 and audit["packed_uint8_bytes"] > 100_000_000_000
        )
        for row in audit["layers"]:
            assert (
                row["prefill"] == "flashkda" and row["state_dtype"] == "torch.float32"
            )
            if args.mode != "dense":
                assert row["mixed_decode_tokens"] > 0 and row["decoded_tokens"] > 200
                assert row["key_ring"] == row["value_ring"] == "torch.bfloat16"
            if args.probe_replay:
                import math

                errors = row["native_probe"]
                assert all(math.isfinite(x) for x in errors)
                assert errors[1] < 0.007 and errors[2] < 3e-5, errors
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                dict(
                    mode=args.mode,
                    status="PASS",
                    steps=step,
                    audit=audit,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            f"PASS {args.mode}: four requests, 384 generated tokens, "
            f"{step} engine steps"
        )
    finally:
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
