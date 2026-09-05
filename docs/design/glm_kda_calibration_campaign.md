# GLM KDA paired-Fisher campaign

Artifacts are in `/disk2/omin/kda-latch-results/glm_calibration`.
The comparison board reuses the Qwen/Super terminal table renderer:

```bash
watch -n 30 bash /disk2/omin/kda-latch-results/glm_calibration/status.sh
```

Dense and Q-Mamba scores come from the existing GLM campaign. Ghost rows
are unrun placeholders. Ours has four allocation rows and four benchmarks.
Missing scores stay pending; smoke generations are excluded. AIME25 uses
`pass@1[avg-of-8]`, never the seed-zero `pass@1` field.

## Calibration and allocation

The differentiable loader dequantizes the same local NVFP4 checkpoint to
BF16 weights and uses BF16 activations, without NVFP4 activation simulation.
The 312.69B text parameters are distributed across eight GPUs; vision and MTP
are excluded. The model's differentiable FLA KDA path supplies the loss
adjoint at each of its 34 KDA layer outputs. This is a calibration arithmetic
choice, distinct from the NVFP4 inference engine.

The corpus is the saved WikiText text used by the earlier campaigns,
retokenized with GLM. Sequences contain 512 tokens with 256 warmup tokens.
Basis fitting uses blocks 0–127; allocation uses blocks 160–191; validation
uses blocks 192–207. Each block has its own next-token loss. The final label
of block 191 is the first input token of block 192, so the latter validation
split is not strictly token-disjoint at its very first token. A second report,
`strict_heldout.json`, excludes block 192 and validates 3,840 measured tokens
from blocks 193–207 without that overlap. Basis fitting
and allocation are disjoint with a gap. None of the four benchmark questions
is used for calibration.

The basis uses the full channel-decay and erase transition for effective
queries, in native coordinates. Allocation uses same-token squared
loss-gradient/residual inner products and ideal least-squares prefix
residuals, with double-precision rank-revealing modified Gram–Schmidt.
This follows the paper's paired-Fisher objective and reuses the Qwen/Super
`make_direct_head_allocation.py` solver. Runtime metric ridge is not inserted
into this ideal-LS objective. Gradient-free output-error tables are also saved.

Independent transition decomposition error was at most 5.42e-7. The maximum
relative discrepancy against BF16 FLA was 0.00627. All 34 layers supplied
nonzero loss gradients. Files `protocol.json`, `probe.json`, `validation.json`,
`basis.pt`, and `paired_000.pt` through `paired_047.pt` contain the evidence.

With K=V=r=128 and W=16, the allocation traffic proxy is
`1024 + 272*G`: KV/W common flush plus G*(V+r+W). There are no separate
query/key anchors, so s_q=s_k=0. Full Phi already covers all 128 coordinates.
Dense is an action of cost KV in the variable budget and is encoded as rank
zero. Integer mean budgets 28, 12, 7, 4 yield modeled reductions 2.01484,
4.05973, 5.94550, 8.24242. Maximum relative solver gap is 0.00806%.
Per-head tables, CSV, and heatmaps are saved alongside the calibration.
These ratios are a paper traffic proxy, not measured runtime speedups;
head padding and refresh intermediates add actual traffic.

## Evaluation contract

MATH-500 has 500 questions, AIME25 30 questions with seeds 0–7, GPQA-D
198 questions, and LCB-v5 315 questions. Temperature is 1.0, top-p is 0.95,
maximum generation is 65,536, model context is 262,144, reasoning effort is
max, and clear_thinking is false. NeMo-Skills reasoning parsing treats a
missing closing think marker as an empty answer. The existing official
runner, symbolic/math and multiple-choice scorers, LCB sandbox, and metric
aggregator are reused.

`eval_protocol_audit.json` records the comparison with the historical GLM
manifest. All prompt hashes, MATH/AIME/LCB data hashes, model config,
generation config, and weight-index hashes match. GPQA's file hash differs;
all 198 rows, every field, and row order match the archived Super input rows.
The historical GPQA byte-hash discrepancy remains unresolved. The relevant
scoring and reasoning-parser sources match pinned NeMo-Skills commit
b620e79aa395076efb529aff9d137c5d033bd64e byte-for-byte. The package installation
metadata is older, but the actual relevant source files match the pinned
reference. LCB uses de8b59485a841a43ee2d66fc058f017e42f638db.

Historical Dense/Q-Mamba ran on B300 DP2/EP2. Current inference runs on RTX
TP8 with packed FP8 MLA. This hardware and arithmetic difference means the
comparison does not isolate KDA as the only changed implementation detail.

## Optimizing the evaluation implementation

The first eager evaluation was stopped at the user's request. Partial
request checkpoints remain intact. No completed task score exists for that
run. The evaluation queue must resume only after graph implementation
validation and timing; graph results have a distinct tag to avoid mixing
responses from different arithmetic paths.

The reference retains FP64 refresh and host-managed slots. The optimized
cache keeps fixed-capacity ownership, eviction, exact raw-write handoff,
window positions, and refresh decisions on GPU. Its refresh uses FP32 IEEE
matrix products and a conditional Gauss–Jordan solve. It skips inactive
heads and non-boundary slots. Independent raw replay still updates the exact
boundary state. Decode has no tensor-to-host reads; only explicit diagnostic
counter queries read back scalars.

Tests compare graph replay, heterogeneous ranks, padding/null slots, slot
reordering/eviction, partial prefill handoff, and ill-conditioned metric-ridge
read operators against the original implementation. This is numerical
agreement under stated tolerances, not a claim of bitwise equivalence.
Full-model graph capture and end-to-end performance must also pass before
restarting the accuracy queue.

The fixed-work full-model benchmark uses 128 prompts, 256 forced output tokens,
and two timed repetitions after warmup. The first timing was invalidated by
live-input validation: GPU handoff originally assumed contiguous engine state
pages. The engine uses padded strides, which must be included in address
calculations. The original timing artifacts remain under `pre_stride_fix`
filenames. They do not establish the performance of a validated implementation.
The new regression uses poisoned padding, non-contiguous state and QKV views,
and CUDA graph replay. Optimized initialization also uses only FP32.

After stride correction, the live GLM audit compared 70,467,584 values across
all 272 TP-worker/KDA-layer pairs. There were zero nonfinite values and zero
values outside the audit tolerance. Maximum absolute output difference was
6.103515625e-5 and aggregate relative RMS difference was 1.06191318e-4.
`live_audit_summary.json` and `engine_benchmark_live_audit.json` retain these
results. The standalone CUDA-graph regression also covers poisoned state
padding and non-contiguous QKV inputs.

Validated fixed-work throughput after the stride fix was 1,778/1,792 tokens/s
versus 1,119/1,124 for reference eager, an aggregate 1.592x improvement.
The final accuracy runtime is `graph_v3`; its source snapshot is saved as
`runtime_graph_v3_manifest.json`. No FP64 refresh is used in this evaluation
path. The four-budget accuracy queue resumes with a distinct graph tag.

## User-requested pause (2026-09-05)

The graph-v3 campaign was stopped at the user's request before completing
the first MATH-500 cell. All campaign and TP worker processes were stopped;
calibration, allocation tables, and request checkpoints remain on disk under
`/disk2/omin/kda-latch-results/glm_calibration`. Further optimization and
evaluation are deferred. No complete four-benchmark accuracy result exists.

A completion-selected snapshot at 18:44 UTC contained 161 responses. The
unchanged official symbolic scorer marked 158 correct (98.14%); mean length
was 713 tokens, median 181, and maximum 8,383. All had a thinking terminator
and stop finish reason. This is not an unbiased estimate of full accuracy:
long-running responses are excluded. The live engine continued generating
about 2,230 tokens/s across 128 requests while the completion count stalled.
Long completed responses included repeated answer verification. Unfinished
response text was unavailable, so their eventual correctness and repetition
remain unknown. Snapshot inputs, scored rows, and metrics are retained in
`partial_audit/`; they must not populate a completed accuracy table cell.

The comparison board now displays a paused job and suppresses stale live
throughput. To resume the same implementation when explicitly requested, run
`bash /disk2/omin/kda-latch-results/glm_calibration/run_campaign.sh`. If runtime
arithmetic changes, use a new implementation tag and validate it first.
