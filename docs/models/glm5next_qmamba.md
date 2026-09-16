# GLM-5.3 Q-Mamba DSQ baseline

This branch restores the GLM KDA state-only DSQ baseline from
`glm53-baseline` (`e3aedf6003`). It supports 4, 6, 8, and 10 bits through
`NS_GDN_QBITS`; unset it or set it to `0` for the ordinary model.

This is **fake quantization for accuracy evaluation**, not a packed low-bit
cache. State values are quantized and immediately dequantized into the existing
FP32 cache. It does not reduce cache capacity or implement ESR/retraining.

## Run

Use the V2 runner, synchronous scheduling, no speculative decoding, and no
Mamba prefix caching. Do not enable ReplaySSM or RecoverSSM.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
NS_GDN_QBITS=8 NS_GDN_QGRAN=dsq_qm NS_GDN_QSR=0 \
vllm serve /path/to/GLM-5.3-Flash-NVFP4 \
  --mamba-cache-mode none \
  --no-async-scheduling \
  --additional-config '{"kda_prefill_backend":"triton"}'
```

Keep your existing model, parallelism, weight-quantization and memory options.
The command above only shows the relevant baseline settings.

Quantization runs once at the logical end of the prompt and after each decode
update. Intermediate prompt chunks and CUDA-graph padding rows are excluded.
The strided recurrent kernel and the latest router optimizations remain intact.

Both Triton and FlashKDA prefill are supported. The command selects Triton for
comparison with the old baseline: FlashKDA maintains BF16 state internally even
when its final cache is FP32. Use the **same prefill backend** for dense and
Q-Mamba accuracy runs. Select `auto` or `flashkda` explicitly when evaluating
that backend; enabling DSQ does not change backend selection.

GHOST pruning is not included in this change.

## Validation

On a CUDA installation matching this branch:

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/kernels/mamba/test_gdn_quant.py \
  tests/models/glm5next/test_kda_recurrent.py -q
```

Run the same model evaluation with `NS_GDN_QBITS=0` and the desired bit width,
holding the prefill backend, sampling, prompts, batch limits and parallelism
fixed. Kernel/reference and metadata tests do not establish full-model accuracy.

### Port validation (2026-09-17)

- CPU: 40 tests passed (DSQ reference/settings, GDN metadata and kpool mapping).
- CUDA: 9 DSQ kernel/graph tests skipped; no accessible CUDA server was available.
- Full GLM inference and model-accuracy evaluation have not been run for this port.
- The source-only macOS test environment used `--confcutdir` to avoid unrelated
  full-engine fixtures. Metadata tests explicitly selected the CPU device type
  and disabled pinned host memory; no CUDA computation was emulated.

The V2 runtime also disables generic token-to-slot mapping for the one-block
kpool tail cache. Its metadata builder owns the circular addressing; the
generic mapper could otherwise access beyond that block table.
