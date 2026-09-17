# Nemotron Super compact hybrid KV cache

This opt-in experiment increases the number of 2K requests that fit in the
hybrid attention/Mamba cache by reducing padding in shared physical blocks.
The default cache path is unchanged.

Enable it with:

```bash
export VLLM_NEMOTRON_COMPACT_KV_CACHE_BLOCK_SIZE=2112
```

The flag is an alignment target. The resolved attention manager block is
2240 tokens with ReplaySSM and 2176 without it. Kernel pages remain 16 and
128 tokens respectively, matching the original kernel selection.

## Supported configuration

The implementation rejects configurations outside this measured scope:

- NemotronHForCausalLM with 40 Mamba layers and 8 attention layers.
- TP/PP/DP = 1, FP8 KV cache, FlashInfer attention, initial block size 16.
- max_model_len = 2304, prefix caching disabled, mamba_cache_mode = none.
- Hybrid KV cache manager enabled; no speculative decoding or KV transfer.
- No skipped KV quantization layers; BLHNC attention layout.

The historical measurement used one B300, FP32 recurrent state, ReplaySSM
history length 16, utilization 0.95, max_num_batched_tokens 8192 and
synchronous scheduling. Each request had 2048 input and 49 output tokens.

## Layout and measured capacity

There are twenty recurrent groups of two layers and one attention group of
eight layers. Attention occupies a complete physical pool block. Its virtual
kernel pages interleave complete layer sets:
`[manager block, kernel page, layer, head, token, KV]`.
Both block and layer strides divide by the page split ratio, so attention
writes cannot overlap another layer or a neighboring recurrent state block.
Padded, mixed, and partial attention tiles retain the original rejection.

| Setting | ReplaySSM ON | ReplaySSM OFF |
| --- | ---: | ---: |
| Attention manager block | 2240 | 2176 |
| Attention kernel page | 16 | 128 |
| Physical pool block bytes | 9,175,040 | 8,912,896 |
| Pool blocks per request | 21 | 21 |
| Confirmed sizing batch | 1002 | 1031 |

The paired measured batch increased from 881 to 1002 (+13.73%).
These are successful full-cohort measurements under a sizing ceiling of
1088, not an exhaustive OOM boundary search. FlashInfer workspace remained
570,425,344 bytes.

## Validation and numerical limits

The September 8 experiment passed six targeted ownership, CUDA FP8 KV
write/read, FlashInfer/TRTLLM decode, and recurrent-state-preserving copy
checks. Those regression cases are retained in
`tests/v1/worker/test_attn_utils.py` with names starting `test_compact_`.
CPU cases can run while GPUs are occupied:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/v1/worker/test_attn_utils.py -k 'compact and not flashinfer' -q
```

On an available GPU, run all compact cases by selecting `-k compact`.
The original experiment also ran 36 contiguous-packing/cache-copy
regressions before the kernel-page-preservation refinement, followed by
reruns of the six focused cases for kernel pages 16 and 128.

Twenty-four model comparisons over three ordinary-text-derived 2K prompts
matched all 49 generated token IDs. A pathological repeated-token prompt
still diverged between layouts, and the original layout itself varied
with batch composition. Comprehensive model-quality or bitwise equivalence
is therefore not established.

The measured source was recorded on base
`f17c5fad3e25f24ea3c3fa9486ba36bc9b12d4b3`.
This publication applies only the compact-cache changes to its published
parent `1ae189af692603b2684d2ccdb170cee544214c48`; the unrelated PLE
optimization commit is not required. The four cache runtime files match
the measured implementation apart from formatting; the environment file
adds only the compact-cache flag.

The later fixed-batch 128/256/512 Nsight captures are available in
[SSM_results commit 45b54b95](https://github.com/omin-Kwon/SSM_results/commit/45b54b95f37d20db39dd4d841c3c7b3dc07acce4).
Those artifacts contain traces, not the source implementation.
