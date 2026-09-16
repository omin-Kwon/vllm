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
