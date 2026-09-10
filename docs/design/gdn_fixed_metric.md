# GDN fixed-metric kernel prototype

2026-09-07, B200. This is a research implementation of a changed read
approximation. It does not replace serving dispatch, recalibrate allocations,
resume Super evaluation, or claim language-model accuracy.

## Method and exact-state contract

In the current embedded input coordinates, H=S0.T=[U,C], with U=H[:,:m].
The state-dependent GDN map is

```text
eta_t = ridge * ||H||F^2 / K
P_t = [I_m, solve(U.T U + eta_t I_m, U.T C)]
checkpoint_read(q) = U P_t q.
```

The prototype instead accepts a fixed calibration metric E0 in these same
embedded coordinates and prepares, once offline,

```text
P0 = [I_m, solve(E0[:m,:m], E0[:m,m:])]
checkpoint_read(q) = U_current P0 q.
```

The online state metric is not replaced inside an otherwise equivalent solve:
the map itself is now fixed. In general this changes outputs. Orthogonality of
the calibrated weights does not make P_t=P0 for each current state. See
`nested_ssm/docs/latch/GDN_OMEGA_NO_SOLVE_CONDITIONS_20260907.md` for the exact
conditions and the difference between calibration and runtime whitening.

The positive-width ring stores raw writes and P0-projected WY factors using the
existing CUDA step. This applies all erase operations consistently to the new
checkpoint read. The full state is still updated exactly through the original
raw-WY fold, including its deferred checkpoint contribution. The new approximate
read does not substitute an approximate state into that fold. Identical input
streams produce bitwise-identical full boundary states in the tested cases;
autoregressive outputs/inputs may of course change under the new approximation.

Width zero retains the existing exact dense/delta-ring route. Width K=128
retains implicit identity coefficients; neither route reads P0.

## Implementation

Sources:

- `benchmarks/kernels/gdn_fixed_metric_research.py`: `FixedMetricWorkspace`,
  generated CUDA fold extension, and offline `coefficients_from_metric`.
- `benchmarks/kernels/benchmark_gdn_fixed_metric.py`: fixed/private/shared
  worlds, reference checks and timing.
- `tests/kernels/mamba/test_gdn_sketch.py`: opt-in fixed-metric graph tests.

The CUDA flush retains the verified FP32 hi/lo Tensor Core state arithmetic.
It removes runtime Gram/norm/ridge generation, coefficient factorization/solve,
and per-request coefficient stores. Its stream extracts U as it writes the new
state. Initial refresh is a U-only gather. No Gram scratch buffer is retained;
the 8 KiB shared Gram staging region is removed. Compiled resource inspection
shows only prep, state fold, and U gather kernels; no Gram or solve entry point
is compiled into this extension. The fold has no local-memory spills.

The fixed coefficient tensor is `(HV,G,K)` and is shared across request slots
using a zero slot stride. The existing step supports this stride, so its CUDA
arithmetic is unchanged. A private-copy variant replicates identical P0 across
slots to distinguish refresh savings from coefficient sharing. Both produce
bitwise-identical outputs in the tested cases. All setup precedes CUDA graph
capture; steady steps/flushes allocate no buffers and read no GPU scalars on CPU.

Persistent P0 uses `4*HV*G*K` bytes. With HV48/G32/K128 it is 0.75 MiB, versus
96.75 MiB for the old 129-slot Phi allocation. The 96 MiB Gram scratch in that
case is also removed. U and projected-WY factor rings remain per slot. These
are kernel-buffer counts, not a measured whole-model memory reduction. The
research subclass temporarily constructs the base workspace at initialization
before releasing its Gram scratch; that transient allocation is outside timing.

Coefficient sharing makes P0 reusable from cache across requests. It does not
mean the step stops executing coefficient loads. The kernel-only benchmark has
no interleaved model-weight traffic; real-model throughput must be measured
separately before revising traffic claims or paper speed plots.

## Functionality checks

`NS_GDN_FIXED_METRIC_TESTS=1` enables 12 existing-suite tests: G4/8/20/32/64/128
with FP32/BF16 gates, heterogeneous widths, dense/full endpoints, permuted compact
slots, inactive rows, paged state/padding, empty device flush counts, and three
successive CUDA graph windows. All 12 pass.

The independent read oracle starts each window from
`H_hat=H[:,:m] P0` (or exact H for dense/full heads), then applies the complete
dense GDN recurrence token by token. Every output is checked against that oracle.
The exact-state recurrence is checked separately. FP64 is confined to these
independent test oracles; kernels and persistent state/metadata remain FP32.
P0 is independently checked against the fixed-metric equation and remains
bitwise unchanged after graph replays. Full state/U/raw writes match the current
kernel bitwise. The benchmark also checks three real windows before timing.

Synthetic E0 is positive definite and has nonzero cross terms, so tests exercise
the complete fixed map, not just a convenient identity/diagonal special case.
The synthetic dynamic-versus-fixed output difference is recorded explicitly;
it is not a language-model accuracy score. Existing paired-Fisher curves and
allocation quality are not validated for this changed reconstruction rule.

Compute Sanitizer memcheck also passes with zero errors for G4/FP32 gates and
G128/BF16 gates, including the full CUDA graph, dense/sketch mix and paged storage.

## Performance

Reference is the current dynamic GDN kernel with fixed head scheduling and the
improved coalesced RHS load. It is an internal reference, not the archived
original ReplaySSM paper baseline.

B200, B128, H16, HV48, K=V128, W16, BF16 mixed QKV and FP32 gates, uniform m=G.
Microseconds. Non-flush averages positions 0..14. Each timing uses 32 operations
per CUDA graph and a median of five CUDA-event measurements after three warmups.
Flush/cycle columns are medians of three rounds with alternating method order.
Cycle includes all 16 actual steps and the flush.

| m | Dynamic non-flush | Fixed/shared non-flush | Dynamic flush | Fixed/shared flush | Dynamic cycle | Fixed/shared cycle | Cycle speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 14.87 | 14.73 | 254.78 | 185.55 | 506.11 | 436.42 | 1.160x |
| 8 | 17.31 | 15.17 | 263.74 | 186.62 | 545.44 | 446.82 | 1.221x |
| 16 | 27.46 | 20.52 | 340.28 | 193.89 | 784.84 | 531.50 | 1.477x |
| 32 | 43.09 | 33.41 | 534.04 | 206.22 | 1233.66 | 752.86 | 1.639x |
| 64 | 72.92 | 57.72 | 1374.07 | 224.73 | 2551.31 | 1159.57 | 2.200x |
| 128 | 77.95 | 79.33 | 254.21 | 251.99 | 1538.58 | 1544.70 | 0.996x |

Full m128 already skips Gram/solve/Phi reads and has no measured benefit. The
prototype's flush is not strictly m-independent: U extraction/stores and padding
still scale with m/G, and cache/layout effects remain. It removes the previous
strong factorization/cross-product dependence on m.

Smaller-batch complete cycles, using a single round of the same event method:

| Batch | m | Dynamic cycle | Fixed/shared cycle | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 4 | 97.12 | 91.69 | 1.059x |
| 1 | 8 | 99.49 | 93.55 | 1.063x |
| 1 | 16 | 122.67 | 110.95 | 1.106x |
| 1 | 32 | 168.24 | 141.95 | 1.185x |
| 32 | 4 | 183.65 | 162.35 | 1.131x |
| 32 | 8 | 188.26 | 163.77 | 1.150x |
| 32 | 16 | 230.24 | 182.87 | 1.259x |
| 32 | 32 | 326.70 | 221.40 | 1.476x |

The initial private-P0 experiment isolates refresh removal: at B128/m32 its
cycle was 914.55 us versus dynamic 1238.49 us; sharing P0 reduced it further to
750.94 us in that run. The repeated run above is the primary comparison.

These results justify retaining this kernel as an accuracy-experiment candidate
for partial widths. They do not establish an unconditional replacement benefit:
batch, width mix, weight traffic and accuracy remain material. No accuracy
evaluation has been started and no default runtime has changed.

## Reproduction and artifacts

Results: `/disk2/omin/kda-latch-results/gdn_solver_research/fixed_metric_20260907/`.
`benchmark_repeated.json` contains all three rounds, `small_batches.json` the
smaller cases, and `pytest.log` the independent verification. Generated CUDA
and its separate extension cache are in
`/disk2/omin/.cache/gdn_solver_research/fixed_metric_fold/`.

```bash
cd /disk2/omin/vllm-qwen38next
export PATH="$PWD/.venv/bin:/disk2/omin/miniconda3/envs/vllm029_q38next/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=2
export NS_GDN_FULL_FLUSH_BUILD_DIR=/disk2/omin/.cache/gdn_full_flush_plan
export NS_GDN_FULL_BUILD_DIR=/disk2/omin/.cache/gdn_full_cuda_v2

NS_GDN_FIXED_METRIC_TESTS=1 .venv/bin/python -m pytest \
  tests/kernels/mamba/test_gdn_sketch.py -q -k fixed_metric_graph
.venv/bin/python benchmarks/kernels/benchmark_gdn_fixed_metric.py \
  --widths 4 8 16 32 64 128 --batches 128 --rounds 3 \
  --output /tmp/gdn_fixed_metric_benchmark.json
```
