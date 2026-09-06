# Experimental full-dimension GDN step

The existing CUDA Phi implementation at `1f23395b19` is the reference. It has
not been edited or replaced. `gdn_sketch.py` starts a separate, opt-in kernel
implementation; no model hook dispatches to it and no accuracy campaign was
resumed. It currently implements the decode step, not a complete serving path.

## Algebra and storage

In embedded GDN coordinates, let `H=S0.T`, `U=H[:, :m]`,
`Z=(H.T@H + eta*I)[:, :m]`, and `M=Z[:m]`. The full coefficient map
`P=solve(M,Z.T)` has `P[:, :m]=I`. Thus `P@y = y[:m] + P[:, m:]@y[m:]`.
The new step stores only `m*(K-m)` coefficients per compact-slot/value-head.
The packed row offsets come from the fixed host allocation. There is no
truncation rank argument, anchor, EMA, or freeze path. Zero width denotes
an exact dense head. Widths may differ even among value heads sharing a key.
`pack_metadata` is a setup/reference converter, not an online refresh kernel.

The raw-WY write and normalized projected-factor rings use the same contract
as the existing kernel. State-dependent approximation affects only output.
The step does not change the persistent state or advance the caller's position.
This lets the old full-K CUDA step serve as a direct oracle for outputs and all
ring writes. Exact boundary materialization is also checked independently.

Current specialization: K=V=128, W=16, normalized mixed QKV, FP32 rings/state,
FP32 or BF16 I/O, arbitrary integer sketch widths up to 128, and explicit
compact slot mapping. State strides may be padded; other buffers have
contiguous inner dimensions. Active physical slots must be unique and mapped
to valid initialized compact slots. Padding uses nonpositive physical indices.
The experimental entry point checks static shape/dtype/layout contracts without
reading GPU scalar values. It allocates nothing during decode and supports
CUDA graph replay.

## Implementation and validation

One CTA handles one request/value head. Compile-time sketch/dense separation
prevents the dense state matrix from determining sketch-path register usage.
Small sketch widths use one warp. Larger widths currently use four warps.
The matrix load supplies both query and key contractions; the GDN identity
prefix is supplied directly from the normalized input. Replay reductions and
new projected factors follow the old raw-WY equations.

`tests/kernels/mamba/test_gdn_sketch.py` compares every step of a 16-token window
against the unchanged CUDA kernel with `r=K` and zero anchors. Six cases cover
G=8/32/128 with FP32/BF16 I/O, heterogeneous widths including dense/width-one
heads, reordered physical-to-compact mapping, padding, padded state pages,
strided mixed-QKV rows, and CUDA graphs. A separate dense recurrence verifies
that the new rings still materialize the exact boundary under block-WY.
The step leaves the persistent state unchanged. These are kernel tests, not a
full-model accuracy claim. Slot reuse and mixed-prefill integration remain
future adapter work; no production adapter was changed.

An additional comparison against the existing standalone FP64 specification
passed four cases (including full-rank dense equivalence). The first version's
maximum old/new output difference was 2.384e-7, and the largest factor/write
ring differences were 5.96e-8. Final graph regression results are retained under
`/disk2/omin/kda-latch-results/gdn_full_dimension_review/pytest_final.log`.

## B200 measurements and limitations

`benchmark_gdn_sketch.py` measures synthetic fixed-position decode, HV=48,
BF16 I/O, W=16, with CUDA graphs. It excludes model hooks, refresh, and end-to-end
serving. Timings below are median GPU event samples for 32 repeated steps;
the inputs are synthetic and the rings start at zero. `dense_mix` uses 25%
dense heads. These are not full-cycle speedups.

| Batch | G | Dense mix | Existing CUDA, us | New step, us |
| ---: | ---: | :---: | ---: | ---: |
| 1 | 8 | No | 5.88 | 5.42 |
| 32 | 8 | No | 7.09 | 8.76 |
| 128 | 8 | No | 18.34 | 31.39 |
| 128 | 8 | Yes | 41.07 | 63.20 |
| 128 | 32 | Yes | 56.47 | 124.71 |
| 128 | 128 | Yes | 120.47 | 208.79 |

The initial unified four-warp version took 109.73 us at batch128/G8. Splitting
dense/sketch and reducing small-G to one warp improved it to 31.39 us, but
**the new kernel is still slower at useful batch sizes**. It must not replace
the reference on the strength of algebraic simplification alone. Larger-G
geometry and shared-key work duplication remain candidates for investigation;
no profiler-based bottleneck attribution is claimed yet.

## Next implementation boundary and KDA

Implement an anchor-free refresh that writes packed tails directly from the
exact updated state, retaining the existing metric-ridge choice. Reuse the
validated stream kernel's exact block-WY, TMA state loads, and staged register
lifetimes, then benchmark the W-weighted step-plus-refresh path. Preserve the
reference while experimenting with warp-per-value-head CUDA scheduling and
shared key-ring contractions across grouped heads. Add a model adapter only
after refresh, slot lifecycle, prefill handoff, and multi-window tests pass.

KDA can reuse the conceptual metadata/projected-factor/exact-ring separation,
but not GDN's identity-prefix specialization as-is: channel-wise decay changes
the effective-query recursion, and arbitrary rotations do not preserve a
diagonal gate. KDA needs a separate full-metadata contraction/transition path.
This module does not claim KDA support.

## Width buckets and shared key-head preparation (2026-09-06)

`StepPlan(widths, max_batch, key_heads, device, share_keys=True)` is now the
setup-time execution plan; pass it as `plan=` to `step`. Build it before graph
capture, keep the allocation/metadata consistent for its lifetime, and use a
separate plan for concurrent streams. Its workspace is allocated once. Decode
has no host tensor reads or allocations. Calling without a plan retains the
previous experimental implementation for controlled comparisons. The production
CUDA reference remains unchanged and no model hook uses the new implementation.

Two independent changes are implemented:

- Group value heads by dense/sketch mode, next-power-of-two actual sketch width,
  and next-power-of-two remaining input width. A width-five head in a G128 buffer
  uses an eight-row tile for coefficients, factors and U instead of 128 rows.
  Width 64 uses 64 remaining columns instead of 128. Width 128 has no tail and
  eliminates its tail load and contraction at compile time. Width 127 is tested
  separately: it still needs its one remaining column. Kernels launch only for
  the heads in each bucket; empty dense/sketch work grids are not launched.
- A preparation kernel normalizes q/k, reduces prior-key dot products against
  q and k, computes the current key-query dot, and writes the new key once per
  request/key head. All its value heads reuse these FP32 results from a fixed
  workspace. Gating/decay, reconstruction matrices, writes and projected factors
  remain per value head. This distinction is required for correctness. Preparation
  finishes before consumer launches on the same stream, including graph replay.

The workspace is B*H*(2K+2W+1) FP32 values: approximately 2.26 MiB for
B128/H16/K128/W16. It holds current-step intermediates, not persistent per-window
metadata. Preparation latency and workspace traffic are INCLUDED in shared-step
measurements below; setup conversion and plan construction are excluded.
`share_keys=False` measures bucket sizing without the preparation kernel.

| Batch | G | Dense heads | Before, us | Buckets only, us | Buckets + sharing, us | Existing CUDA, us |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 0% | 8.75 | 7.14 | 7.69 | 7.08 |
| 128 | 8 | 0% | 31.42 | 28.97 | 25.06 | 18.34 |
| 128 | 8 | 25% | 63.28 | 62.09 | 59.11 | 41.18 |
| 128 | 32 | 25% | 119.99 | 116.74 | 108.78 | 56.46 |
| 128 | 128 | 25% | 209.44 | 159.73 | 133.79 | 120.77 |

A separate heterogeneous B128/G128-buffer case repeats widths
`[0,1,5,8,20,64,65,127,128]` across 48 heads: 209.60 -> 157.48 -> 145.39 us,
versus existing CUDA 78.82 us. Thus these changes improve the experimental
kernel but still do not justify replacing the established CUDA implementation.
Sharing loses to buckets alone at B32/G8 because it has its own launch/workspace
cost; it is independently selectable rather than assumed universally beneficial.

The final regression has **24 passing cases**, comparing plain, bucket-only and
shared execution against the existing full-K CUDA kernel across G8/32/64/128
and FP32/BF16 I/O. It retains graph, padding, nontrivial slot mapping, ring-write,
and independent exact-boundary checks. Results are in `pytest_shared_final.log`,
`step_benchmark_shared.json` and `step_benchmark_heterogeneous.json` under the
local review result directory.

Remaining arithmetic padding is explicit: non-power-of-two head widths still
round up, and ring reductions still have W=16 lanes even at shorter prefixes.
These changes reduce oversized tiles; they do not claim every masked scalar
operation is eliminated. Large-G occupancy, grouped scheduling/launch overhead,
and the anchor-free refresh kernel remain separate optimization work.

## Full-dimension CUDA specialization: non-flush improvement

`gdn_step_full_cuda.py` now provides a separate CUDA entry point derived from
the unchanged `gdn_step_phi_cuda.py` reference at `1f23395b19`. The reference
still supports truncation, but **every comparison here uses r=K=128 with zero
anchors**, not r=64. The new CUDA kernel keeps its warp-per-value-head scheduling,
shared key-ring contractions, asynchronous bulk copies, double-buffered shared
memory and raw-WY ring contract. It removes rank selection, anchor pointers,
anchor loads/additions, freeze handling and partial-state-read branches.
K=V=128, W=16, HV/H=3 are explicit specialization constraints.

The initial CUDA specialization alone reduced position-eight B128/G8 from
18.33 to 15.64 us. A subsequent exact specialization skips the coefficient
stream entirely for m=K heads, whose embedded coefficient map is identity.
Those heads load normalized q/k directly for the projected-factor recurrence.
Partial-width heads still consume full-K Phi metadata; packed-tail migration
is not part of this CUDA change. Its input contract requires the canonical
embedded reconstruction map. No numerical truncation or new approximation is
introduced. The runtime metric-ridge policy is unchanged.

The exported CUDA `step` accepts neither r nor anchors. There is no preparatory
kernel or intermediate shared-key workspace: one CUDA launch executes the
step. The persistent state is not mutated, and the caller still owns position
advancement and exact boundary flush. Neither production dispatch nor the
existing flush implementation has been changed. The earlier Triton kernel
remains available as an experimental alternative and algebra cross-check.

### Verification

Sixteen final CUDA tests pass: G8/32/64/128, FP32/BF16 I/O, FP32/BF16 gate input
pairs. Each test compares all 16 positions with the unchanged CUDA reference,
including output, transformed writes, keys, gates and projected factors. They
also cover heterogeneous widths, zero/dense and width-one heads, width127
beside width128, compact-slot mapping, padded state pages, strided input rows,
CUDA graph replay and exact boundary reconstruction against an independent
dense recurrence. No full-model or accuracy campaign is claimed or launched.
The final log is `pytest_cuda_final.log` (16 passed, 24 deselected).

### B200 non-flush measurements

The following are means of the per-position median timings over **all positions
0 through 14**. Position 15/flush is excluded. Each position uses five samples
of 32 graph-captured kernel calls. Both implementations use the same synthetic
setup, B/HV/G, full-coordinate coefficient path and BF16 I/O. As with earlier
benchmarks, setup/metadata conversion and model hooks are excluded. Seven
configurations times fifteen positions give 105 paired measurements; every
measured pair favored the new CUDA kernel in this run.

| Batch | G | Allocation | Existing CUDA r=K, us | Full-K CUDA, us | Time reduction |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | 8 | Uniform | 5.81 | 4.89 | 16.0% |
| 32 | 8 | Uniform | 6.98 | 6.16 | 11.7% |
| 128 | 8 | Uniform | 18.44 | 16.60 | 10.0% |
| 128 | 8 | 25% dense | 40.37 | 38.15 | 5.5% |
| 128 | 32 | 25% dense | 55.92 | 54.01 | 3.4% |
| 128 | 128 | 25% dense | 119.81 | 74.22 | 38.1% |
| 128 | 128 | Heterogeneous | 78.19 | 65.91 | 15.7% |

Heterogeneous repeats `[0,1,5,8,20,64,65,127,128]` across 48 value heads.
These are measured non-flush improvements, not full-window or model speedups.
`cuda_benchmark_v1.json`, `cuda_benchmark_v2.json` and
`cuda_benchmark_nonflush.json` retain the intermediate and final measurements
under `/disk2/omin/kda-latch-results/gdn_full_dimension_review`.

From the vllm-qwen38next checkout, reproduce with:

```bash
export PATH="$PWD/.venv/bin:/disk2/omin/miniconda3/envs/vllm029_q38next/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=10.0
export NS_GDN_CUDA_BUILD_DIR=/disk2/omin/.cache/gdn_reference_sm100
export NS_GDN_FULL_BUILD_DIR=/disk2/omin/.cache/gdn_full_cuda_v2
.venv/bin/python -m pytest tests/kernels/mamba/test_gdn_sketch.py -q -k cuda
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --cuda-only --all-nonflush --output /tmp/gdn_nonflush.json
```

## Flush audit after removing input truncation

This is a source and algebra audit, not an implemented or timed flush change.
The production flush still uses `gdn_flush_stream_cuda.py` and the solve source
from `gdn_ls6_epilogue_cuda.py`. Its three launches already fuse exact state
folding and metadata formation in the middle kernel: prep -> stream -> solve.
The preceding non-flush measurements exclude all three launches and model hooks.

In runtime coordinates, let H be the V-by-K boundary state, U=H[:,:m],
E=H^T H, Z_eta=[E+eta I](:,:m), M=Z_eta[:m,:], and P=M^{-1} Z_eta^T.
The existing runtime metric ridge is retained. With all K input coordinates,
an anchor coefficient is M^{-1} Z_eta^T qbar=P qbar. Its residual anchor
therefore vanishes algebraically. The same holds for kbar. This cancellation
does not require changing the ridge or numerical precision. It refers to
flush anchor solves, not to the raw-write replay contribution in the step.

The following work remains in the old code but can be removed in a new flush:

- qbar/kbar normalization, accumulation and EMA in `decode_bookkeep`, and their
  prefill initialization and frozen-tail vectors. Slot ownership and beta-ring
  capture remain necessary for the existing exact raw-WY flush.
- qbar/kbar exponent selection, hi/lo splitting, preparation records and async
  transfers. The anchor record costs 1 KiB per request/key head; its two stream
  buffers cost 2 KiB of shared memory per CTA, excluding other anchor scratch.
- H qbar / H kbar contractions, their fragment conversions, projected anchor
  contractions, partial reductions, and the two anchor scratch vectors.
- Two anchor right-hand sides in forward/back substitution, followed by the
  prefix/tail subtraction and aq/ak stores. Cholesky itself remains for 0<m<K.

The stream contains a particularly costly masked computation: its extra anchor
`mma3(pacc[4], ...)` executes for every 16-row coefficient tile, while only the
subsequent reduction/store tests whether the row is below m. Removing anchors
eliminates these MMA operations, not just masked stores.

For K=V=128, R=128, ordinary execution (dbg=0), and one value-head item, the
following are source-level `mma3` calls **per warp in metadata formation**:

| Actual m | Main coefficient Gram | H times two means | Projected anchors | Removable fraction of these calls |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 8 | 8 | 66.7% |
| 16 | 16 | 8 | 8 | 50.0% |
| 32 | 32 | 8 | 8 | 33.3% |
| 64 | 64 | 8 | 8 | 20.0% |

Each `mma3` contains three MMA instructions for the existing hi/lo arithmetic.
Counts follow eight row tiles, ceil(m/8) column tiles, eight H-times-mean tiles,
and one anchor call per row tile. They exclude exact state folding, prep,
scalar work, memory movement and solve, and are not latency predictions.
For comparison, dropping two solve RHS changes m=8 from 122 to 120 columns
and m=64 from 66 to 64; it does not eliminate the factorization at those widths.

Further opportunities, in implementation order:

1. Preserve the current streaming fold and strip the complete anchor dataflow.
   Re-evaluate register pressure and occupancy after its fragments, reductions
   and buffers are gone. Removing arithmetic alone does not guarantee the
   optimal launch configuration remains unchanged.
2. Specialize m=K: P=I in the canonical embedded representation already used by
   the new step. Skip coefficient Gram formation, its norm/ridge preparation,
   Cholesky, solve and Phi output. Exact raw-WY folding and the U refresh remain
   required. The m=0 dense path already skips metadata; that is not a new gain.
3. Coordinate flush and step metadata layouts: P[:,:m]=I, so only m*(K-m)
   tail values need storage instead of m*K. The current CUDA step still expects
   full-K rows for partial widths, so a flush-only layout change is invalid.
   Size work by actual m rather than the largest G buffer where profitable.
4. The stream computes both halves of the leading symmetric Gram block.
   A triangular tile schedule could save work there, especially at medium m,
   but must preserve MMA efficiency and avoid extra partial-reduction costs.
   The separate standalone Gram kernel already exploits this symmetry.
5. Preallocate the wrapper's scratch/prep/prep_i buffers and remove obsolete
   model-hook work. These allocations are currently outside the CUDA kernel;
   graph replay does not repeat Python allocations, so distinguish graph setup,
   eager execution and replay when reporting the benefit. Beta capture could
   be fused into the step while preserving its input-dtype rounding behavior.

Key-ring Gram sharing across value heads and fusion of state folding with Gram
formation are already implemented. Exact state read/write, raw-WY eraser
correction and the per-value-head gate/beta dependence cannot be dropped.
Folding the inverse into U is an equivalent representation change, not free
removal of the solve; its projected factors must change consistently too.

Validation of a future implementation must exercise actual GPU flushes over
multiple windows, comparing boundary states and refreshed U/Phi, then outputs
in the following windows. Existing independent boundary reconstruction tests
do not substitute for this. Profile prep/stream/solve and the complete W=16
cycle, with the unchanged full-K CUDA path as reference, before quoting a
flush or end-to-end speedup.

## Implemented full-coordinate flush

`gdn_flush_full_cuda.py` now implements an independent flush and initial metadata
refresh. It retains the existing tensor-core hi/lo arithmetic, metric ridge
(default 0.1, honoring `NS_GDN_LS6_RIDGE`) and exact raw-WY state update. The
reference files remain unchanged. Its API contains no input-rank parameter,
mean vectors, frozen-tail state or anchor inputs/outputs.

Implemented removals include the mean-vector preparation records and transfers,
H-times-mean products, projected-anchor MMA calls and reductions, anchor scratch,
two extra solve RHS, residual-anchor contractions and stores. Full-width heads
skip Gram/norm formation and solve entirely: the stream copies U from the exact
updated state, and leaves unused Phi untouched. Dense heads retain their
established exact delta-ring fold. Partial widths still use the existing full-K
Phi layout, including its explicit identity prefix; packed-tail CUDA migration,
triangular Gram scheduling and per-width workspace packing are not implemented.

`FlushWorkspace` allocates scratch/preparation buffers once, resolves the default
grid before graph capture, and caches the extension. Flush and initial refresh
use only device counts/indices. `gdn_step_full_cuda.step(..., beta_ring=...)`
optionally writes its already-rounded beta directly into the compact-slot ring.
This removes a separate sigmoid/scatter from the new kernel pipeline. No q/k
means or EMA bookkeeping are needed by this API. The old model adapter and
production dispatch still target the unchanged reference path; these results
are for the explicitly invoked new kernel pipeline, not a deployed model.

Correctness covers actual GPU flushes across three windows and the resulting
next-window outputs, independently checked against a dense recurrence and
boundary coefficient solves. It includes G4/8/32/64/128, widths zero/one/127/128,
FP32/BF16 gate rounding, CUDA graphs, compact-slot permutations, padded physical
state pages, empty flushes, and subsets of active requests. Full-width Phi is
poisoned with NaNs to prove it is not consumed. A separate test forces persistent
CTA worklists beyond 64 entries and covers HV/H=1/2/3/4. The runtime remains
FP32/hi-lo; FP64 is used only for independent mathematical test oracles.

Final verification: **54 tests passed** (24 Triton step, 16 CUDA step, 10
multi-window GPU flush, four worklist/head-sharing cases). Compute Sanitizer
memcheck on the HV/H=4 worklist-refill case passed with **zero errors**. Logs:
`pytest_full_flush_final.log` and `flush_memcheck.log` in the review result
directory. Selected pre-commit hooks, including mypy, and manual Ruff checks
for the third-party kernel modules passed.

### B200 flush and complete kernel-cycle measurements

Median of five samples, each with 32 graph-captured calls/cycles. A cycle is
16 ordered steps followed by the real flush; positions are preallocated device
inputs. Fixed synthetic BF16 input sequences and FP32 gates provide the same
beta history to both paths. New steps record beta; reference external beta
capture and mean bookkeeping are excluded. Setup/initial refresh, Python model
hooks and sampling are excluded for both paths. Each variant starts from the
same prepared state/rings/metadata, then runs repeatedly without timed resets.
All benchmarks check finite state/output. These are kernel timings, not model
throughput or accuracy measurements.

| Batch | G | Allocation | Old flush, us | New flush, us | Old 16-step cycle, us | New cycle, us |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | 8 | Uniform | 41.91 | 30.82 | 134.93 | 110.96 |
| 32 | 8 | Uniform | 186.94 | 136.88 | 308.53 | 246.17 |
| 128 | 8 | Uniform | 668.92 | 450.49 | 974.77 | 730.78 |
| 128 | 8 | 25% dense | 618.30 | 440.30 | 1271.02 | 1092.10 |
| 128 | 32 | 25% dense | 1598.51 | 1356.23 | 2487.35 | 2244.17 |
| 128 | 128 | 25% dense | 5318.55 | 255.06 | 7329.75 | 1495.39 |
| 128 | 128 | Heterogeneous | 3349.85 | 2441.19 | 4626.92 | 3536.94 |

The large G128 uniform-sketch gain is specifically the removal of an unnecessary
full-width coefficient refresh; it must not be generalized to compressed heads.
At B128/G8 uniform, flush time falls 32.7% and the complete cycle falls 25.0%.
Separate new-stage measurements there are prep 16.72 us, stream 319.94 us and
solve 102.93 us; separately captured stages need not sum to the full-graph time.
`flush_benchmark_v1.json` stores all cases and stage timings under the local
review result directory.

Use the earlier PATH/TORCH_CUDA_ARCH_LIST/reference-cache settings, with the
updated step cache, to reproduce:

```bash
export NS_GDN_FULL_BUILD_DIR=/disk2/omin/.cache/gdn_full_cuda_v3
export NS_GDN_FULL_FLUSH_BUILD_DIR=/disk2/omin/.cache/gdn_full_flush
.venv/bin/python -m pytest tests/kernels/mamba/test_gdn_sketch.py -q
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --flush --output /tmp/gdn_flush.json
```

## Comparison with base ReplaySSM flush

The preceding "old" columns mean the old sketch/anchor implementation, not
base ReplaySSM. A separate B200 run now measures the actual default ReplaySSM
dispatch, `gdn_flush_cuda` with all sketch/frozen/beta arguments absent (G=1,
TV=8, grid=4*SM=592), and the established stream implementation with all widths
zero (grid=2*SM=296). Stream totals include prep and the empty solve launch;
they are not stream-kernel-only measurements. No flush debug/grid overrides
were set. H=16, HV=48, K=V=128, W=16, all batch rows flush, FP32 state/rings.

Historical B300 measurements in
`nested_ssm/scale/docs/latch/GDN_FLUSH_STREAM_KERNEL_20260904.md` used B=256:
the basic CUDA flush was 362 us, while v12's dense stream kernel alone was
281 us. In particular, comparing those B256 numbers with the preceding B128
sketch numbers would mix batch sizes and, for 281 us, timing boundaries.

| Batch | Sketch allocation | Base ReplaySSM, us | ReplaySSM stream total, us | New sketch flush, us | Sketch / base time |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 128 | m=8 uniform | 180.68 | 160.82 | 451.27 | 2.50x |
| 128 | m=32, 25% dense | 180.86 | 160.74 | 1357.04 | 7.50x |
| 256 | m=8 uniform | 349.35 | 311.75 | 867.54 | 2.48x |
| 256 | m=32, 25% dense | 349.46 | 314.81 | 2642.32 | 7.56x |

Each row uses the same initial state and input window for both methods. The
benchmark first generates a true delta ring for base ReplaySSM using dense
steps, while positive-width sketch heads keep raw-WY writes. It checks that
both flushes yield the same exact boundary before timing them; the largest
observed absolute state difference was 3.58e-7. Feeding the same raw-write ring
to the base delta-only flush would be an invalid correctness comparison.
Timing uses the same five samples of 32 graph-captured calls as above.

The new sketch flush is therefore faster than the old sketch implementation,
but has not achieved the original goal of matching base ReplaySSM flush cost.
Its B256/m8 separately measured stages are prep 28.38 us, stream 629.04 us,
solve 200.90 us. The stream includes exact raw-WY eraser correction, U refresh
and Gram formation; base ReplaySSM's eraser contribution is already present in
its delta ring. These costs remain after eliminating input truncation. Both
the stream and the small-m solve require further optimization to close the gap.
The total-cycle columns in this JSON still compare old/new sketch paths; no
base ReplaySSM full-cycle speedup is inferred from this flush-only baseline.

Results: `replayssm_flush_comparison.json` and matching `.log` in the review
result directory. With the same environment as above, reproduce using:

```bash
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --replayssm --output /tmp/gdn_replayssm_flush.json
```

The subsequent [B200 width analysis](gdn_flush_width_analysis.md) uses uniform
heads, fixed-G controls and Nsight Compute to explain the m-dependent costs:
Gram CAS/bank conflicts, excessive generic-solve instructions, allocation-width
penalties, and the separate m128 identity fast path.

## Flush width tuning after the bottleneck audit

The full-coordinate flush now uses warp-private Gram tiles and width-specific
RHS solves. At B128 with uniform heads, m8 flush is 451 -> 263 us and m32 is
1488 -> 591 us; m33 is 2765 -> 768 us. The m8 complete window is 731 -> 545 us.
The m32 25%-dense complete window is 2244 -> 1442 us. Native ReplaySSM flush
remains faster (182 us at B128). Full m128 sees about 10 us of additional empty
bucket launch overhead; partial widths above 64 retain the previous solver.

See [the updated width analysis](gdn_flush_width_analysis.md) for all timings,
fixed-G128 results, tested contracts and remaining limits. The 60-case suite
passed; an expanded persistent-work-list case covers every m=0..128 and passes
Compute Sanitizer memcheck with 0 errors. The U/P representation and ridge are
unchanged; removing state embedding is not required for these improvements.
