# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM engine smoke comparison, not a task-accuracy or speed benchmark."""

import argparse
import json
import os
from pathlib import Path


def latch_stats(model):
    return {
        name: {
            "decode_rows": layer._latch_cache.decode_rows,
            "dense_handoffs": layer._latch_cache.dense_handoffs,
            "dense_audit": getattr(layer._latch_cache, "dense_audit", None),
        }
        for name, layer in model.named_modules()
        if getattr(layer, "_latch_cache", None) is not None
    }


class LatchStatsWorkerExtension:
    def get_kda_latch_stats(self):
        return latch_stats(self.get_model())

    def install_latch_reference_audit(self):
        """Compare optimized low-rank reads with FP64 refresh on live inputs."""
        import types

        import torch

        from vllm.third_party.flash_linear_attention.ops.kda_latch import KDALatchState

        def step(cache, state, indices, q, k, v, gate, beta, a, bias, lower):
            expected = []
            for row, slot in enumerate(indices.reshape(-1).tolist()):
                if slot not in cache.reference_slots:
                    cache.reference_slots[slot] = KDALatchState(
                        state[slot : slot + 1],
                        cache.basis,
                        head_ranks=cache.ranks,
                    )
                expected.append(
                    cache.reference_slots[slot].step(
                        q[row : row + 1],
                        k[row : row + 1],
                        v[row : row + 1],
                        gate[row : row + 1],
                        beta[row : row + 1],
                        a,
                        bias,
                        lower_bound=lower,
                    )
                )
            actual = cache.original_step(
                state, indices, q, k, v, gate, beta, a, bias, lower
            )
            reference = torch.cat(expected).float()
            delta = (actual.float() - reference).abs()
            stats = cache.dense_audit
            stats["max_abs"] = max(stats["max_abs"], delta.max().item())
            stats["error_sq"] += delta.square().sum().item()
            stats["reference_sq"] += reference.square().sum().item()
            stats["elements"] += delta.numel()
            stats["nonfinite"] += int(
                (~torch.isfinite(actual) | ~torch.isfinite(reference)).sum().item()
            )
            stats["outside_tolerance"] += int(
                (delta > 1e-4 + 0.01 * reference.abs()).sum().item()
            )
            if stats["nonfinite"] or stats["outside_tolerance"]:
                from pathlib import Path

                target = Path("/disk2/omin/kda-latch-results/glm_calibration")
                slots = cache.slots[: len(q)].long()
                diagnostic = dict(
                    inputs=[x.detach().cpu() for x in (q, k, v, gate, beta, a, bias)],
                    state_stride=state.stride(),
                    expected=reference.cpu(),
                    actual=actual.cpu(),
                    ranks=cache.ranks.cpu(),
                    basis=cache.basis.cpu(),
                    pool={
                        key: getattr(cache.pool, key)[slots].cpu()
                        for key in (
                            "state",
                            "phi",
                            "latch",
                            "pos",
                            "k",
                            "v",
                            "log_a",
                            "beta",
                            "f",
                            "u",
                            "prefix",
                        )
                    },
                )
                torch.save(diagnostic, target / f"live_failure_{cache.audit_name}.pt")
                raise RuntimeError(f"Live latch comparison failed: {cache.audit_name}")
            return actual

        def prefill(cache, state, indices, initial):
            for slot in indices.reshape(-1).tolist():
                cache.reference_slots.pop(slot, None)
            return cache.original_prefill(state, indices, initial)

        for layer in self.get_model().modules():
            cache = getattr(layer, "_latch_cache", None)
            if cache is None:
                continue
            cache.audit_name = f"tp{layer.tp_rank}_layer{layer.layer_idx}"
            cache.reference_slots = {}
            cache.dense_audit = dict(
                reference="FP64 latch",
                max_abs=0.0,
                error_sq=0.0,
                reference_sq=0.0,
                elements=0,
                outside_tolerance=0,
                nonfinite=0,
            )
            cache.original_step = cache.step
            cache.original_prefill = cache.before_prefill
            cache.step = types.MethodType(step, cache)
            cache.before_prefill = types.MethodType(prefill, cache)

    def install_dense_audit(self):
        """Compare identical live inputs against a separate dense recurrence."""
        import types

        import torch

        from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

        def audited_step(cache, state, indices, q, k, v, gate, beta, a, bias, lower):
            ids = indices.reshape(-1).tolist()
            expected = []
            for row, slot in enumerate(ids):
                if slot not in cache.audit_states:
                    cache.audit_states[slot] = torch.cat(
                        [torch.zeros_like(state[:1]), state[slot : slot + 1]]
                    )
                output, _ = fused_recurrent_kda(
                    q=q[row : row + 1].unsqueeze(0),
                    k=k[row : row + 1].unsqueeze(0),
                    v=v[row : row + 1].unsqueeze(0),
                    g=gate[row : row + 1].unsqueeze(0),
                    beta=beta[row : row + 1].unsqueeze(0),
                    initial_state=cache.audit_states[slot],
                    ssm_state_indices=torch.ones(1, device=q.device, dtype=torch.int32),
                    use_qk_l2norm_in_kernel=True,
                    sigmoid_beta=True,
                    compute_gate=True,
                    a_log=a,
                    g_bias=bias,
                    lower_bound=lower,
                )
                expected.append(output[0])
            result = cache.original_step(
                state, indices, q, k, v, gate, beta, a, bias, lower
            )
            reference = torch.cat(expected).float()
            error = (result.float() - reference).abs()
            stats = cache.dense_audit
            stats["max_abs"] = max(stats["max_abs"], error.max().item())
            stats["error_sq"] += error.square().sum().item()
            stats["reference_sq"] += reference.square().sum().item()
            stats["elements"] += error.numel()
            return result

        def audited_prefill(cache, state, indices, initial):
            for slot in indices.reshape(-1).tolist():
                cache.audit_states.pop(slot, None)
            return cache.original_prefill(state, indices, initial)

        for layer in self.get_model().modules():
            cache = getattr(layer, "_latch_cache", None)
            if cache is None:
                continue
            cache.audit_states = {}
            cache.dense_audit = dict(
                max_abs=0.0, error_sq=0.0, reference_sq=0.0, elements=0
            )
            cache.original_step = cache.step
            cache.original_prefill = cache.before_prefill
            cache.step = types.MethodType(audited_step, cache)
            cache.before_prefill = types.MethodType(audited_prefill, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0, help="0 selects original dense")
    parser.add_argument("--basis-path")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--audit-dense", action="store_true")
    args = parser.parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    additional = {"gdn_prefill_backend": "triton"}
    if args.rank:
        additional["kda_latch"] = {"rank": args.rank}
        if args.basis_path:
            additional["kda_latch"]["basis_path"] = args.basis_path
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        mamba_ssm_cache_dtype="float32",
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        enable_flashinfer_autotune=False,
        additional_config=additional,
        worker_extension_cls=(
            "benchmarks.kernels.evaluate_glm_kda_latch.LatchStatsWorkerExtension"
        ),
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    prompts = ["The capital of France is", "Compute 12 + 30. The answer is"]
    if args.audit_dense:
        if not args.rank:
            raise ValueError("--audit-dense requires a latch rank")
        llm.collective_rpc("install_dense_audit")
    results = []
    # Repeated calls exercise physical slot reuse after completed requests.
    for _ in range(2):
        outputs = llm.generate(
            prompts, SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)
        )
        results.append(
            [
                {
                    "prompt": x.prompt,
                    "text": x.outputs[0].text,
                    "token_ids": x.outputs[0].token_ids,
                }
                for x in outputs
            ]
        )
    stats = llm.collective_rpc("get_kda_latch_stats")
    if args.rank and (
        not all(stats)
        or any(row["decode_rows"] == 0 for worker in stats for row in worker.values())
    ):
        raise RuntimeError("At least one worker/layer did not execute latch decode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "rank": args.rank,
                "tp": args.tp,
                "results": results,
                "latch_stats": stats,
                "task_accuracy": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
