# GDN ReplaySSM performance baseline provenance

## Correction to the September 6 timing labels

The B200 B128 flush measurements of approximately 181–182 us are the dense
dispatch of our CUDA rewrite, not the original GDN ReplaySSM implementation.
`benchmark_gdn_sketch.py --replayssm` calls `gdn_flush_cuda` without sketch,
frozen-state or beta arguments. Disabling approximation does not undo the
implementation optimizations in that CUDA kernel.

The CUDA step and flush files first entered this repository in
`14ff737a6508cf6f884e694286901984733a05cf` (September 2). The measured flush
source was last changed in `52217fc6eeae6d17b5e942af2b926ec84981b395`:

- Path: `vllm/third_party/flash_linear_attention/ops/gdn_flush_cuda.py`
- Git blob: `576e92af638ddde83728b8149c7a2f5694d253bc`
- SHA256: `203510ac67ab0a81a124145deca44b23c03e8e1694947c48b30dda1084b34734`

The historical B300 dense step/refresh pair, 145.4/346.1 us at B256, also
belongs to our CUDA implementation. Neither pair establishes original-kernel
performance. See `nested_ssm/scale/docs/latch/GDN_STEP_KERNEL_OPT_20260902.md`
and `GDN_FS_KERNEL_OPT_20260903.md` for that optimization history. The Triton
**v2** numbers in those documents must not automatically be relabeled original
either: that version already contains project changes.

## Original GDN port to preserve for paper comparisons

Use the original GDN ReplaySSM port before the approximation and CUDA rewrite,
with the runner cursor fix included:

- Repository: `vllm-qwen38next`
- Commit: `9df6640e8baba10fbf3bc39b311ea639aa073500` (August 29)
- Kernel: `vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py`
- Kernel Git blob: `674e480c84b2cf40f839afe1f4988e1230395144`
- Kernel SHA256: `56e448bd675558621de5d6707f16b5284122baf3abc3d0bc8e39113a29bf1f52`
- Configuration: `vllm/model_executor/layers/mamba/ops/replayssm_config.py`
- Configuration SHA256: `9f03fb6be6f02c3a85531bee64bd4c5b7e9fbbd6d60f2706ddf44e7ee516ad9d`

The initial port is `82966123014688841acb3621d360a7b0c9216c7a` (August 28).
The subsequent kernel diff adds host-side argument validation; it does not
rewrite the Triton device algorithm. The cursor fix is in the runner wiring.
This is this project's GDN port of ReplaySSM, not a claim that an upstream
ReplaySSM release supplied this GDN kernel.

Paper results should distinguish **ReplaySSM (original GDN port)**,
**ReplaySSM (our optimized CUDA)** and **Ours (exact Z)**. The optimized dense
path remains useful as an additional control for how much improvement comes
from implementation work. It must not silently replace the original baseline.

## Measurement boundaries for the next comparison

The original Triton kernel fuses the current token's output and checkpoint
update when `write_pos == W-1`. Its flush-token latency therefore includes the
online step. The 181–182 us CUDA result measures the additional checkpoint
flush only. Those two quantities must not share a column without explanation.

Report non-flush positions `0..W-2`, the complete flush token at `W-1`, and a
complete W-token window for every implementation. Measure the window directly;
for position-dependent times its arithmetic check is
`(sum(nonflush_position_times) + complete_flush_token_time) / W`.
Keep extra flush-only stage timings as a separate diagnostic.

Use the same GPU, batch, head geometry, W, input/state dtypes, inputs, graph
capture procedure and timing statistic. Pin the original source/configuration
and record compiler versions and launch settings. Before timing, compare
per-token outputs and boundary states against the exact recurrence, including
the original delta-ring convention. Any required compatibility or correctness
fix to the archived port must be recorded explicitly.

No new original-port B200 timings were collected in this provenance audit.
The saved September 6 JSONs retain their historical field names; their
`replayssm_flush_us` values denote our optimized CUDA dense dispatch.
