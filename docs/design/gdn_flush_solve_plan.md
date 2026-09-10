# Fixed-allocation GDN flush solve plan

2026-09-07. This change preserves the current full-coordinate GDN equations,
ridge, FP32 operations, hi/lo tensor-core products, and U/Phi layout. It does
not introduce a fixed reconstruction approximation or an incremental Gram cache.

## Scheduling change

Previously, allocation stride G selected up to six solve launches (capacities
8, 16, 32, 48, 64, 128). Each launch scheduled every request/value-head pair;
heads outside that kernel's width bucket immediately returned.

`FlushWorkspace(..., widths=mh)` now constructs a sorted head list for each
bucket. A solve launch covers only those heads and absent buckets have no launch.
The kernel retains its width predicate and executes the same Cholesky and RHS
instructions. Scratch and Phi still use the original head index and allocation G.
State update, U refresh, and Gram generation are unchanged.

| Allocation / actual widths | Previous solve launches | Planned solve launches |
| --- | ---: | ---: |
| G=128; every m=8 | 6 | 1 |
| G=128; every m=128 or 0 | 6 | 0 |
| G=128; all six partial-width buckets present | 6 | 6, with compact head grids |

These are launch counts, **not measured latency or speedup**. Mixed-width plans
schedule `max_rows * number_of_partial_heads` solve CTAs in total. The old path
scheduled `max_rows * HV * number_of_launched_buckets`. Device-side row counts
still exclude inactive rows during graph replay.

The only new GPU data is one int32 index per partial-width head: at most 192
bytes for HV=48 per workspace, independent of request slots. There is no new
per-request numeric cache. Seven bucket offsets stay on the host.

## Setup and graph contract

The optional fixed widths tensor is inspected once during workspace construction,
before capture. Calls to flush/refresh and graph replays do not copy widths or
counts to the CPU. They do not allocate the plan again. GDN's model adapter passes
its existing fixed allocation to `FullCoordinateRuntime` at buffer setup.

The bound widths tensor must remain immutable for the lifetime of the workspace
and its graphs. Eager calls reject another tensor or a changed PyTorch version
counter. Inference tensors do not have version counters; graph replays also
bypass Python checks, so mutation in those cases is outside the API contract.
Changing allocation requires a new workspace and new graphs. Omitting widths
retains the old device-filtered launch path for callers without a fixed plan.

## Correctness evidence

Artifacts: `/disk2/omin/kda-latch-results/gdn_full_dimension_review/solve_plan_20260907/`.

- `pytest_plan.log`: four cases compare planned/unplanned execution with zero
  tolerance, including m=0 through 128, all-dense/full-width plans, padded G,
  permuted compact slots, NaN sentinels, changing device row counts, and repeated
  CUDA graph folds/refreshes.
- `pytest_integration.log`: 28 cases check the real runtime dispatch, BF16 inputs,
  paged storage and staggered flushes across multiple windows; compare against
  the untouched legacy CUDA path and independent recurrence/coefficient oracles.
- `gdn_flush_full_baseline.py` preserves the source before this change.
  `compare_baseline.py` reruns the bitwise suite against that source in a separate
  extension, rather than treating the new unplanned branch as the only oracle.
  `pytest_baseline_bits.log`: all four cases passed after strengthening the
  comparison to raw int32 bit patterns, including signed zeros and NaN sentinels.

No new language-model evaluation is claimed. Super uses another runtime and is
currently stopped at the user's request while kernel research continues.

## Performance measurement

Uncontended measurements were completed on 2026-09-07 16:15:36–16:17:28 UTC,
after the Super processes had exited. GPU clocks during measurement were
1965 MHz (memory 3996 MHz). `benchmark_gdn_flush_scheduling.py` alternates the
preserved pre-change source and the planned candidate for three rounds. Each
round uses 32 calls per CUDA graph and the median of five event measurements;
the values below are medians across the three rounds, in microseconds.

Batch 128, H=16, HV=48, K=V=128, W=16; uniform actual width:

| m | Allocation G | Previous flush | Planned flush | Previous solve | Planned solve |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 4 | 253.79 | 254.60 | 18.40 | 18.84 |
| 8 | 8 | 262.44 | 263.35 | 26.87 | 27.31 |
| 16 | 16 | 353.17 | 345.75 | 75.86 | 70.34 |
| 32 | 32 | 591.29 | 582.58 | 241.16 | 232.00 |
| 64 | 64 | 1421.81 | 1420.57 | 900.75 | 895.31 |
| 128 | 128 | 265.26 | 236.25 | 28.65 | 0.13* |
| 8 | 128 | 474.03 | 450.47 | 128.06 | 104.92 |
| 32 | 128 | 752.44 | 728.50 | 300.47 | 273.98 |

*No solve kernels run for the full-width plan; 0.13 us is empty timing overhead.
Small m=4/8 does not benefit when G is already minimal: head indirection has a
small cost. Savings mainly come from avoiding empty buckets. The combined state
fold/U/Gram stage is unchanged; this optimization does not remove its arithmetic.

Complete cycles (16 steps plus one flush) confirm that the effect is conditional:

| Case | Previous cycle | Planned cycle |
| --- | ---: | ---: |
| B128, uniform m=8, G=8 | 544.84 | 545.42 |
| B128, G=32, 25% dense m=0 and 75% m=32 | 1444.43 | 1424.69 |
| B128, G=128, 25% dense m=0 and 75% m=128 | 1503.64 | 1477.67 |
| B128, G=128, heterogeneous widths | 2596.68 | 2586.92 |

The heterogeneous pattern is `[0,1,5,8,20,64,65,127,128]` repeated to 48 heads.
All 102 records, hashes and clock observations are in
`solve_plan_20260907/isolated_speed/{samples,metadata}.json` under the artifact
directory above. The step kernel itself is unchanged.

To reproduce a width sweep (keep both runs on the same idle GPU):

```bash
cd /disk2/omin/vllm-qwen38next
export PATH="$PWD/.venv/bin:/disk2/omin/miniconda3/envs/vllm029_q38next/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=10.0
export NS_GDN_FULL_FLUSH_BUILD_DIR=/disk2/omin/.cache/gdn_full_flush_plan
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --width-sweep --allocation-g 128 --output /tmp/gdn_unplanned.json
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --width-sweep --allocation-g 128 --fixed-solve-plan --output /tmp/gdn_planned.json
```

Repeat without `--allocation-g 128` to compare minimally padded allocations.
`--flush --fixed-solve-plan` also exercises mixed-head and 16-step-cycle cases.
This is an internal before/after scheduling comparison. It does not replace
the archived original ReplaySSM kernel required for paper baseline comparisons.

Gram and Cholesky work for active partial-width heads remains. Removing it would
require another derivation/implementation with separate correctness evidence;
this change makes no such claim.
