# Qwen3.8-Flash-Next GDN DRRQR pruning

## Scope

This port applies per-layer/per-key-head indices produced by
LinearAttentionPruning to `RadixArk/Qwen3.8-Flash-Next-NVFP4`.  It masks Q and K
after the depthwise convolution and SiLU and before L2 normalization.  ReplaySSM
remains enabled and consumes the masked Q/K values.

This is the accuracy-comparison implementation.  It zeros pruned state columns
but does **not** physically shrink the recurrent-state tensor or the kernel's
K dimension, so it must not be presented as a pruning speedup measurement.

## Local inputs verified on 2026-08-30

- Calibration checkpoint: `/disk2/models/Qwen3.8-Flash-Next-FP8`
  - 131 safetensor shards
  - 185,523,317,458 weight bytes (172.782 GiB)
  - `qwen4_exp`, 48 text layers, 36 GDN layers, 16 x 128 key geometry
- Target checkpoint: `/disk2/models/Qwen3.8-Flash-Next-NVFP4`
- All 324 GDN tensors (3.886 GiB) were hashed and are byte-identical between
  the FP8 calibration checkpoint and NVFP4 target.
- Calibration environment:
  `/disk2/omin/miniconda3/envs/qwen38next_rrqr`
  (`torch 2.9.1+cu130`, `transformers 5.16.1`, `accelerate 1.14.0`,
  `datasets 2.21.0`, `scipy 1.17.1`).

No calibration forward or pruned NVFP4 forward was run while preparing this
port.  GPU validation is deliberately pending.

## Extract indices

CPU-only checks:

```bash
/disk2/omin/LinearAttentionPruning/scripts/run_pruning/run_qwen38_flashnext_rrqr_local.sh inspect
/disk2/omin/LinearAttentionPruning/scripts/run_pruning/run_qwen38_flashnext_rrqr_local.sh prepare
/disk2/omin/LinearAttentionPruning/scripts/run_pruning/run_qwen38_flashnext_rrqr_local.sh status
```

Explicit single-B300 run (not started during preparation):

```bash
QWEN_GDN_CUDA_VISIBLE_DEVICES=0 \
  /disk2/omin/LinearAttentionPruning/scripts/run_pruning/run_qwen38_flashnext_rrqr_local.sh run
```

The extractor runs 128 independent 2,048-token forwards with batch size one,
samples 5,000 combined Q/K rows per layer and head, and emits masks for 37.5%,
50%, 62.5%, and 75% sparsity.  It keeps the FP8 checkpoint entirely on one
visible GPU and fails instead of silently offloading modules to CPU or disk.

The FP8 checkpoint stores the roughly 95-GiB runtime PLE embedding as 128
E4M3 shards plus one BF16 global scale. Transformers 5.16.1 concatenates the
shards but has no Qwen4Exp path that applies this scale. The extractor therefore
validates the scale (`0.00019931793212890625`, SHA-256
`c7c58bd6007672362da2106fdbfaf9f50629e4bdf8598169c598027394ef9791`) and
restores the PLE weight to scaled BF16 before calibration. The chosen action is
part of `protocol.json`, so a run cannot silently omit it.

Expected output root:

```text
/disk2/omin/SSM_results/qwen_gdn_drrqr_indices/qwen3.8-flash-next-fp8/
  indices-manifest.json
  ratios/sparsity-0.375/indices.pt
  ratios/sparsity-0.500/indices.pt
  ratios/sparsity-0.625/indices.pt
  ratios/sparsity-0.750/indices.pt
```

## Apply to NVFP4 + ReplaySSM

`NS_GDN_PRUNE` must point at one ratio-specific `indices.pt`.  The port fails
closed unless it sees exactly 36 model-layer keys with H=16 and K=128.  It also
requires tensor parallel size one because the global head masks have not been
sharded for TP execution.

When pruning is armed, the fused CUDA GDN decode path is disabled because it
would bypass the Python-visible post-convolution mask.  The standard Triton
packed decode and ReplaySSM decode paths both apply the mask.  Spec tokens in
the standard path are masked as well.  ROCm, XPU, CPU, TP>1, missing layers,
invalid indices, and a first mask allocation during CUDA graph capture all fail
closed.

Prepared breakdown wrapper:

```bash
# CPU-only readiness check; reports WAITING_INDICES until extraction finishes.
/disk2/omin/nemotron_profile/qwen38_flashnext/run_pruned_pipeline.sh check 0.375

# Explicit GPU run after indices exist.
GPU=0 INPUT_LEN=1024 \
  /disk2/omin/nemotron_profile/qwen38_flashnext/run_pruned_pipeline.sh run 0.375
```

The wrapper reuses the established Flash-Next NVFP4 protocol, including
ReplaySSM buffer length 16, BF16 QSA KV, disabled FlashInfer autotune, v2 model
runner, and the existing CUDA-graph profiling stages.

## Q-Mamba DSQ baseline

This tree also carries an accuracy-only Q-Mamba baseline for the materialized
GDN recurrent state. It applies the paper's dynamic separable quantization
(DSQ) to the last two state axes, `(value_dim, key_dim)`, using the
square-root-mean channel scale and the state-axis maximum. Channel and state
scales are each rounded to FP16 before multiplication, matching the existing
`vllm_gdn` Q-Mamba protocol.

The hook fake-quantizes the FP32 cache in place: it rounds to the selected
integer grid and immediately dequantizes back to FP32. It therefore measures
accuracy effects only; it does not allocate a packed integer cache and cannot
be used to claim state-memory or throughput savings. No calibration or
projector checkpoint is required.

Supported widths are 10, 8, 6, and 4 bits:

```bash
NS_GDN_QBITS=8 \
NS_GDN_QGRAN=dsq_qm \
  <existing Qwen3.8-Flash-Next-NVFP4 vLLM command> \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-mode none \
  --no-enable-prefix-caching
```

ReplaySSM must be off for this arm. Speculative decoding and stochastic
rounding/ESR are not implemented. Unsupported bit widths, another
`NS_GDN_QGRAN`, `NS_GDN_QSR`, ReplaySSM, a non-FP32 SSM cache, or a Mamba cache
mode other than `none` fail at startup.

Scheduler chunking does not change the method: the v2 runner supplies the exact
prompt length and computed-token count, the GDN metadata marks only the logical
end of prefill, and the hook quantizes once there and once after every ordinary
decode step. Intermediate prefill chunks and CUDA-graph padding rows are
masked. CUDA graph capture uses the same fixed-address runtime mask that is
updated before each replay.

`NS_GDN_PRUNE` and `NS_GDN_QBITS` are technically composable in this tree, but
a standalone Q-Mamba baseline should leave `NS_GDN_PRUNE` unset. Conversely,
the DRRQR + ReplaySSM measurements above must leave `NS_GDN_QBITS` unset.

The runtime environment
`/disk2/omin/miniconda3/envs/vllm029_q38next` is an editable install pointing at
`/disk2/omin/vllm-qwen38next`, so this source change is already the environment
implementation; no site-packages overlay is needed.

## Required GPU validation before reporting results

1. Run the extractor and confirm all four `validation.json` files pass.
2. Start an eager one-prompt NVFP4 smoke with `NS_GDN_PRUNE_VERIFY`-equivalent
   tensor checks if a verifier is added; confirm pruned Q/K columns are zero in
   prefill and every decode path.
3. Run ReplaySSM ON twice: unpruned and pruned, recording the index SHA-256,
   vLLM commit, model config hash, selected kernel, and generated output.
4. Only then enable CUDA graphs and run the prepared breakdown wrapper.

For Q-Mamba, add a separate eager smoke before accuracy evaluation: confirm the
`[gdnq]` startup banner, compare one-shot and forced multi-chunk prefill, then
compare eager and CUDA-graph decode. No Q-Mamba GPU forward was run while this
port was prepared.
