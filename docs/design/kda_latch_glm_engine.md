# Experimental GLM KDA latch engine adapter

The opt-in `additional_config={"kda_latch": {"rank": 128}}` connects the
paper-audited r=128, W=16 implementation to GLM's ordinary decode path.
Without this option the original dense KDA path remains the accuracy baseline.

## Runtime contract

- Require eager execution, FP32 recurrent state, prefix caching disabled,
  `mamba_cache_mode=none`, and no speculation, ReplaySSM, Q-Mamba or DRRQR.
- Pure decode reads the latch, carries projected f factors and raw replay,
  and publishes the exact checkpoint at each 16-token boundary.
- A batch containing prefill uses the existing dense chunk kernel for all
  its rows. Before gathering initial states, continuing slots materialize
  pending writes from raw key/value/log-decay/beta only. Their latch windows
  are discarded and lazily rebuilt on their next pure decode.
- A new request (`has_initial_state=False`) discards the old sidecar without
  replaying an abandoned window. Physical null slot 0 is rejected.
- Sidecars are allocated lazily per physical slot and kept until that slot
  next participates in prefill. Their memory is additional to engine caches;
  the scheduler does not yet budget it. Use small batches and memory headroom.
- Host synchronization, per-row kernel launches, separate checkpoint copies
  at boundaries and FP64 refresh remain. This is an accuracy adapter, with
  no speedup or fused-refresh traffic claim.

G128 uses the identity basis. Lower G requires a `basis_path` to a trusted
`torch.save` dictionary mapping integer layer IDs to tensors of shape
`(global_heads, 128, G)`. Each TP rank selects its corresponding head slice.
No native-channel truncation or anchor approximation is enabled. A GLM
calibration dataset and lower-rank task-accuracy campaign are separate work.

## Validation

The existing GPU suite now has 11 tests. New coverage checks partial-window
handoff against independent dense transitions with poisoned approximate
metadata, repeated flushes, mixed latch/dense heads, reordered engine slots,
slot reuse and preservation of null slot 0. All 11 passed on 2026-09-05.

```bash
PYTHONPATH=$PWD TRITON_CACHE_DIR=/disk2/omin/.cache/triton-kda-latch \
  .venv/bin/python tests/models/glm5next/test_kda_latch.py

PYTHONPATH=$PWD VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .venv/bin/python benchmarks/kernels/evaluate_glm_kda_latch.py \
  --model /disk2/models/GLM-5.3-Flash-NVFP4 --tp 8 --rank 128 \
  --output /disk2/omin/kda-latch-results/glm_resume/g128.json
```

Use `--rank 0` for the original dense arm with the same prompts/settings.
The smoke runner generates two 40-token continuations twice to exercise
slot reuse, and saves per-worker/layer counters to verify latch execution.
It is not a task-accuracy benchmark. Add `--audit-dense` with a nonzero rank
to maintain an independent dense shadow state on the same live decode inputs.
This diagnostic is installed after engine warmup and resets at prefill.

### Full-model results, 2026-09-05

Dense and G128 both completed on TP8. All 34 KDA layers on all eight workers
executed the latch (161 rows per layer, including warmup). Each arm generated
160 tokens across two prompts repeated twice. Repeated calls within each arm
were identical, but the arms differ: the first differences occur at generated
token 5 for the France prompt and token 16 for the arithmetic prompt. There
are 44 matching token positions out of 160; after the first difference these
are different autoregressive trajectories, not a teacher-forced accuracy score.

The separate dense-shadow audit compared 42,893,312 KDA output elements on
identical live inputs across all 272 worker/layer pairs. Maximum absolute
error was 1.52587890625e-5; aggregate relative RMS error was
1.749581862921867e-5. Enabling the diagnostic preserved the unaudited G128
token sequences. This establishes small KDA output differences in this run;
it does not establish the downstream cause of token divergence or preserved
task accuracy. No low-G quality or speedup claim follows from this smoke.

Artifacts under `/disk2/omin/kda-latch-results/glm_resume`:

- `dense.json`, `g128.json`: generated text, token IDs and execution counters.
- `g128_audit.json`: same-input dense-shadow error statistics per worker/layer.
- `comparison.json`: token divergence and aggregate numerical audit.
- `dense.log`, `g128.log`, `g128_audit.log`: successful inference runs.

The processes wrote their results and exited with code 0. Engine teardown
still emitted forced-worker-cleanup/shared-memory warnings. Repeated generation
within each process passed; graceful teardown remains a runtime limitation.
Next evaluation must use a fixed task dataset and scoring protocol for both
arms before assessing accuracy preservation or calibrating lower G.

## Local environment restoration

The original standalone-test checkout lacked full engine build artifacts.
For the resumed run, CUDA extensions, FlashAttention CuTe sources,
FlashMLA's generated Python interface and DeepGEMM were reused from the local
`/disk2/omin/vllm-qwen38next` build (source HEAD
`1f23395b19ce4d89c795da605526463800a18de2`). These are local runtime artifacts,
not changes to the KDA equations. The successful GLM smoke establishes that
this local combination runs these inputs; it is not a clean-build guarantee.

The donor's main CUDA and MoE libraries contain SM100 code, not SM120 code.
They cannot execute on these RTX devices. The two local symlinks were
therefore replaced with `_C_stable_libtorch.abi3.so` and
`_moe_C_stable_libtorch.abi3.so` from the existing
`/disk2/omin/miniconda3/envs/vllm027/lib/python3.12/site-packages/vllm` install,
whose ELF lists include SM120. The delayed launch error from the incompatible
libraries initially appeared at a later MHC call. Independent MHC post and
fused-post-pre suites (8 cases each), plus a minimal eight-GPU MHC invocation,
passed; changing the TileLang execution backend alone did not fix the model.

The donor lacked `vocab_parallel_embedding`. Its unchanged CUDA source from
this checkout was compiled into the supplemental library
`kda-latch-results/glm_resume/compat/build/kda_engine_embedding.so` (under
`/disk2/omin`), registered through the run-specific `compat/sitecustomize.py`.
All 26 existing `tests/kernels/core/test_vocab_parallel_embedding.py` GPU
tests passed, including bit-exact TP shard reconstruction. To reproduce this
local environment, prepend that `compat` directory to `PYTHONPATH`.

TileLang also needs an explicit SM120 target and a separate cache in this
environment:

```bash
export TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_120"}'
export TILELANG_CACHE_DIR=/disk2/omin/.cache/tilelang-kda-sm120
```

The local launcher `/disk2/omin/kda-latch-results/glm_resume/run_smoke.sh`
sets these variables, the supplemental library path, and the existing
environment's Ninja/CUDA compiler paths.

## GLM NoPE on the RTX sparse MLA backend

GLM has `qk_rope_head_dim=0`, while the SM120 V32-style packed cache/kernel
requires 64 physical RoPE channels. The backend now pads both keys and queries
with 64 zeros and passes the physical dimension to FlashInfer. The model's
logical dimensions and attention scale remain unchanged. Padding contributes
zero to the attention dot product.

The nine sparse MLA API tests pass, including GPU comparisons against
independent attention over dequantized cached KV and the backend's existing
FP8 query quantization. The initial BF16-query reference failed; inspection of
FlashInfer's `common/fp8_quant.cuh` established the per-128-channel, power-of-two
query scaling. The corrected reference passes with the same 0.02 absolute and
relative tolerances. The test separately requires zero physical RoPE bytes.

GLM key pooling widens the index buffer to 2176 entries, exceeding the
SM120 decode dispatch's 2048-entry maximum. The backend uses the actual
buffer width, splits wider rows into supported capacities, and merges partial
outputs with their base-2 log-sum-exp normalizers. Prefill tails are padded to
the supported 2048-entry capacity. Padding is -1 and no selected token is
discarded. Empty query rows return zero. Tests cover decode and prefill batch
sizes, a valid token at the end of the widened buffer, and empty rows.

Both engine arms use the SM120 backend's packed FP8 MLA KV cache and internal
FP8 query computation; KDA recurrent state is FP32. These runs must not be
presented as reproducing the earlier B300 campaign's MLA cache precision or
absolute task scores.

Eight RTX PRO 6000 Blackwell Server Edition devices were visible, each with
97,887 MiB. The config and weight-index SHA256 match the existing evaluation
manifest. Earlier failed environment bring-up attempts are retained separately
from the final successful logs listed above.

## Calibrated allocation and graph evaluation

The subsequent [calibration campaign](glm_kda_calibration_campaign.md) adds
paired-Fisher head allocations, a fixed-capacity GPU cache, and an opt-in
`graph: true` path. The eager adapter described above remains the numerical
reference. Graph decode uses GPU ownership and conditional refresh, supports
null padding and partial prefill handoff, and avoids host synchronization.
See the campaign document and saved timing artifacts for its validation;
the original eager smoke results above are not graph performance results.
