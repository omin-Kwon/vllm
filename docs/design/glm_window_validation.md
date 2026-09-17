# GLM KDA window integration

Base: official vLLM ec4a3a537068db40afbc9374a67da719c8c9b964.
Native dense is the default. Replay and Sketch share ownership and raw-write
replay, preserve native unrotated FP32 state, and use W16 with BF16 raw k/v
and FP32 log decay/beta. Sketch keeps exact-Z K128 and Full-Gram offline
allocation independent of inference P4/P6.

## Test design (before implementation)

- Recurrence contract: post-conv q/k/v, raw gate/beta and physical page IDs in;
  output and updated state out. Compare Replay to the official fused recurrent
  kernel across full/partial windows, shuffled rows, padding and strided state.
- Ownership contract: release discards pending writes before physical page reuse;
  prefill materializes only continuing prefill requests. Mixed batches keep
  decode rows in the window cache without resetting window positions. Exercise replacement, unscheduled
  owners, preemption, new prompts and mixed prefill/decode at kernel level.
- Graph contract: dynamic IDs/positions are device inputs, never capture constants.
  Replay a captured graph across thousands of steps and changing requests.
- Approximation contract: compare P4/P6 to independent FP64 pivot oracle; exact
  state and flush output must remain independent of sketch metadata.
- Runner contract: release precedes request removal for finish, preempt and
  streaming update. Extend existing runner unit tests.
- Model contract: run packed NVFP4 model smoke after kernel gates; compare native
  and Replay with identical prompts/teacher-forced continuations. Accuracy
  equivalence is not established by a smoke test alone.

No published speedup claim is made by this correctness-first Replay path.

## Configuration

Build the Python checkout with precompiled native artifacts from the unmodified
official base (the native extensions have not changed):

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ec4a3a537068db40afbc9374a67da719c8c9b964 \
  uv pip install -e . --torch-backend=auto
```

Use model runner v2, synchronous scheduling, TP1/PP1/DP1, BF16 activations,
FP32 SSM state, prefix caching disabled and no speculative decoding. The
existing `use_replayssm` flag remains the Mamba2 feature; leave it disabled.
Default GLM Dense is unchanged when `kda_window` is absent.

```python
# additional_config for exact raw-write ReplaySSM:
{"kda_prefill_backend": "auto", "kda_window": {"mode": "replay", "window": 16}}

# Same state/lifecycle with SketchSSM reads:
{"kda_prefill_backend": "auto", "kda_window": {
    "mode": "sketch", "window": 16,
    "checkpoint": "/path/to/glm/fullgram/g4/runtime.pt", "pivots": 4,
}}
```

The checkpoint must declare `glm_kda_fullgram_allocation_v1`, exact-Z K128,
no anchors or embedded state, and P-independent Full-Gram allocation.
Pivots can be 4 or 6 without changing the checkpoint. Native GLM has 64
query/key/state heads: group size is one. Ranks 0 and 128 select exact dense
heads; ranks 1--127 select sketch reads.

## Mixed batches and ownership

The official metadata builder puts single-token decode requests first and
provides separately rebased prefill offsets, page IDs and initial-state flags.
`window_attention` consumes these fields rather than deriving chunk offsets
from token counts. Only prefill rows pass through FlashKDA. Decode rows retain
their slot, ring and modulo-16 position on mixed steps. Output slices preserve
the builder's original decode-first order. The native convolution is unchanged.

`finish_requests` invalidates window owners before removing finished or
preempted request metadata, without writing freed physical pages. Continuing
streaming prompts first materialize their pending state. Prefill handoff commits
only rows with an initial state; new prompts discard stale ownership. Graph
capture ownership is reset without writes before real requests start.

An unscheduled live owner remains valid. If a slot must be evicted, its pending
updates are materialized to its own physical page before acquiring another page;
empty slots are preferred. The production pool covers the scheduler's maximum
sequence count and graph capture size.

Physical page zero remains padding. The official runner zeros newly allocated
physical pages before execution, including one-token new prompts classified as
decode. Released ownership must be invalidated before that zeroing/reassignment.

## Validation commands

```bash
.venv/bin/python -m pytest --confcutdir=tests/models/glm5next \
  tests/models/glm5next/test_kda_recurrent.py -v
.venv/bin/python -m pytest --confcutdir=tests/v1/worker \
  tests/v1/worker/test_gpu_model_runner_v2.py -v
PYTHONPATH=examples/offline_inference VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python examples/offline_inference/glm_window_smoke.py \
  --model /path/to/GLM-5.3-Flash-NVFP4 --mode replay --out replay_smoke.json
```

Repeat the model gate with `--mode dense` and `--mode sketch --checkpoint ...`
in separate processes. Four staggered requests force mixed batches and generate
384 tokens across multiple window boundaries. A read-only worker audit records
all 34 KDA layers, packed NVFP4 bytes, ring dtypes and mixed decode counts.

The implementation is ported from the existing `glm_kda_tuning` direct-decay,
small-rank and fused WY-flush kernels, without the experimental `runpy` or global
patching adapters. Replay has no sketch metadata work. Sketch computes metadata
in the flush kernel; neither path dequantizes the model weights.

## B300 verification, 2026-09-16

- KDA suite: 18 tests passed, including existing native tests, independent FP64
  P4/P6 solves, zero/dependent sketches, poisoned metadata at exact flush, and
  4096-step CUDA graph lifecycle runs with 64 slots and 96 physical pages.
- Runner suite: 10 tests passed, including finished/preempted release,
  continuing-stream materialization and existing graph-capture behavior.
- Real packed GLM: Dense, Replay W16 and Sketch G4/P4 each completed four
  staggered requests and 384 generated tokens. All 34 KDA layers used FlashKDA
  prefill; both window modes recorded mixed decode through their own cache.
  The packed uint8 parameter count remained 152,202,903,552 bytes.
- A separate eager `--probe-replay` run shadowed every decode in all 34 layers
  with the official recurrent kernel on identical layer inputs. Maximum output
  absolute error was 6.1035e-5; output relative L2 error was 6.8237e-4; flush-state
  relative L2 error was 1.6939e-7. The shadow does not replace our outputs.
- All applicable repository pre-commit checks passed.

The short greedy sequences were not token-identical: Replay/Dense common prefixes
were 16, 34, 75 and 44 tokens out of 96. These are separate full-model runs with
staggered admission, not a teacher-forced logit equivalence test. The layer probe
establishes bounded local arithmetic error; it does not establish identical
benchmark accuracy. "Exact" here denotes the unapproximated raw-update recurrence
in finite precision, not bitwise equivalence of whole-model generation.

Full benchmark accuracy has not been remeasured on the new window path. P6 is
covered by the kernel/oracle/mixed-routing tests; the real Sketch model smoke
uses P4. Machine-readable results are in
[`glm_window_results_20260916.json`](glm_window_results_20260916.json).

### Concurrent branch integration

Before publishing, integrated remote commit `3daf64c2fa` (GLM Q-Mamba DSQ and
circular kpool-tail mapping) while preserving its changes. Window replay/sketch
now explicitly rejects simultaneous Q-Mamba state quantization.

After integration, the KDA suite passed 20 tests (two new method-combination
guards); GDN metadata and kpool-tail tests passed 32; Q-Mamba tests passed 17,
including CUDA cases. The real G4/P4 mixed-batch model gate also passed again.
The earlier Dense/Replay/shadow measurements above precede this remote merge;
they are not a benchmark accuracy claim for the combined branch.

## Opt-in Flash-style Sketch flush, 2026-09-17

Set `additional_config.kda_window.flush_backend="flash_wy"` for the experimental
Sketch backend. The default is `original`; Replay rejects this Sketch-only key.
This backend follows the deployed FlashKDA pin
`b59532f1f464fbd536272780e30df5bf6a2ccc02`, not the older FP16 inverse implementation.

- Prepare decay-adjusted keys and the W16 inverse before accessing full state.
  Two FP32 8x8 triangular solves and BF16 block products construct the inverse.
- Use BF16 matrix operands and FP32 accumulation. Updated full state is rounded
  to BF16 internally and widened into the existing FP32 cache. Raw k/v rings
  remain BF16; already activated decay/beta rings remain FP32.
- For ranks 1 through 4, process V32 tiles, accumulating small ridge-system
  statistics in FP32. Store U and exact full-state flush output from each tile;
  finish Phi in the same kernel without rereading the full state.
- Larger ranks retain the existing P4/P6 pivot formulas, with the prepared WY
  recurrence fused ahead of metadata construction. Register spilling in this
  larger builder remains an optimization target.
- Non-flush coefficient formulas, calibrated frames and allocations are
  unchanged. Dense-fallback heads and partial-window handoff retain FP32
  recurrence. The scratch workspace is per slot/head and follows device worklists.

The raw-logit FlashKDA interface and this already-activated ring interface have
different gate rounding and operation ordering; this is not a bitwise port.
Adopting its precision requires a separately tagged accuracy run. Do not replace
an in-flight evaluation's backend or combine its scores with this candidate.

```bash
.venv/bin/python benchmarks/kernels/benchmark_glm_window_replay.py \
  --batches 64 --checkpoint /path/to/g4/runtime.pt --pivots 4 \
  --flush-backend flash_wy --out flash_wy_kernels.json
PYTHONPATH=examples/offline_inference VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python examples/offline_inference/glm_window_smoke.py \
  --model /path/to/GLM-5.3-Flash-NVFP4 --mode sketch \
  --checkpoint /path/to/g4/runtime.pt --flush-backend flash_wy \
  --out flash_wy_smoke.json
```

The tests compare flush state with an independent CPU precision oracle, metadata
with an FP64 pivot solve, and flush output with the updated full-state read even
when sketch metadata is poisoned. Graph/eager lifecycle tests exercise release,
slot replacement, row reordering and partial prefill handoff.

B300 G4/P4, actual allocations over all 34 layers, B64/W16 synthetic CUDA graphs
(including lifecycle; microseconds per layer):

| Backend | Non-flush | Flush | W16 mean |
| --- | ---: | ---: | ---: |
| Native recurrent Dense | 103.24 | 103.23 | 103.24 |
| Original 51a268f886 | 46.12 | 1198.61 | 118.15 |
| Flash-style candidate | 46.94 | 820.50 | 95.29 |

This is a 31.55% flush reduction and 1.24x window-mean speedup versus the original
Sketch kernel. It is not an end-to-end speedup or an accuracy result. Non-flush
does not improve. Large-rank metadata still spills, empty metadata/flush kernels
remain in the common graph, and Sketch still publishes pooled state to native
pages with a copy. Optimization is incomplete.

The final candidate passes 24 kernel/lifecycle tests and the packed GLM
mixed-batch model gate (34 layers, 384 tokens, 128 engine steps). This integration
gate does not establish benchmark accuracy for the new rounding boundaries.

NSYS graph-node tracing at layer 18, non-flush position 7, B64: four read-body
kernels total 33.825us (55.9%); nine empty initialization/flush dispatches total
16.384us (27.1%); lifecycle and output zeroing total 10.272us (17.0%). The profiled
60.481us total is diagnostic, not a replacement for unprofiled graph timings.
Eliminating empty dispatches alone will not achieve a 10x non-flush speedup.
