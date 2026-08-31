#!/usr/bin/env python3
"""One-request GPU smoke with the dedicated PLE CPU-offload worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=("replay16", "drrqr", "qmamba"))
    parser.add_argument("--indices")
    parser.add_argument("--bits", type=int, choices=(4, 6, 8, 10))
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.environ["VLLM_PLE_CPU_OFFLOAD"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    for key in ("NS_GDN_PRUNE", "NS_GDN_QBITS", "NS_GDN_QGRAN", "NS_GDN_QSR"):
        os.environ.pop(key, None)
    use_replayssm = args.arm in ("replay16", "drrqr")
    if args.arm == "drrqr":
        if not args.indices or not Path(args.indices).is_file():
            raise SystemExit("--indices is required for the DRRQR smoke")
        os.environ["NS_GDN_PRUNE"] = str(Path(args.indices).resolve())
        os.environ["VLLM_GDN_DECODE_KERNEL"] = "triton"
    elif args.arm == "qmamba":
        if args.bits is None:
            raise SystemExit("--bits is required for the Q-Mamba smoke")
        os.environ["NS_GDN_QBITS"] = str(args.bits)
        os.environ["NS_GDN_QGRAN"] = "dsq_qm"

    model_config = json.loads((Path(args.model) / "config.json").read_text())
    quant = model_config.get("quantization_config") or {}
    if str(quant.get("quant_algo", "")).upper() != "NVFP4":
        raise SystemExit("smoke requires Qwen3.8-Flash-Next-NVFP4")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly one word: Paris"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    kwargs = {}
    if use_replayssm:
        kwargs.update(use_replayssm=True, replayssm_buffer_len=16)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        kv_cache_dtype="auto",
        mamba_ssm_cache_dtype="float32",
        mamba_cache_mode="none",
        enable_prefix_caching=False,
        enable_flashinfer_autotune=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        max_cudagraph_capture_size=1,
        gpu_memory_utilization=0.90,
        enforce_eager=args.eager,
        disable_log_stats=True,
        additional_config={"gdn_prefill_backend": "triton"},
        **kwargs,
    )
    output = llm.generate(
        [prompt], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False
    )[0].outputs[0]
    print(json.dumps({
        "status": "passed",
        "checkpoint_quant_algo": "NVFP4",
        "arm": args.arm,
        "eager": args.eager,
        "use_replayssm": use_replayssm,
        "ple_offload": {"backend": "dedicated_cpu_worker", "model_runner": "v1"},
        "indices": os.environ.get("NS_GDN_PRUNE"),
        "qbits": os.environ.get("NS_GDN_QBITS"),
        "token_ids": output.token_ids,
        "text": output.text,
        "finish_reason": output.finish_reason,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
