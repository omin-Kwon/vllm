# GDN research snapshot, September 10

This commit preserves the September 7 working tree on top of `9e1ee3f954`.
It is a research checkpoint, not a new performance or model-accuracy result.
Generated extension binaries and build directories are excluded.

## What was uncommitted

- **Fixed head scheduling in the full-coordinate runtime.** The model adapter
  supplies immutable per-head widths to `FullCoordinateRuntime` and
  `FlushWorkspace`. Solve kernels launch only for heads in their width bucket;
  empty buckets disappear. The coefficient equations, precision, state fold,
  and metadata layout stay the same. This is the only included change wired
  into the existing full-coordinate runtime. See
  [the solve plan](gdn_flush_solve_plan.md).
- **Alternative solver representations.** Benchmark-only modules investigate
  deferred triangular solves, warp-register Cholesky, and coalesced RHS loads.
  Deferred solves reduced flush time but slowed complete windows in the saved
  runs. Coalesced loads improved selected widths. These candidates are not
  selected by the serving adapter. See [solver research](gdn_solver_research.md).
- **Fixed calibration metric.** This benchmark-only candidate removes online
  coefficient preparation, intentionally changing the read approximation while
  retaining exact state updates. Its synthetic tests do not establish language
  model accuracy. It is not the current Nano P4 implementation and is not a
  replacement for state-dependent coefficients. See
  [the fixed-metric prototype](gdn_fixed_metric.md).
- **Regression tests and baseline provenance.** Tests cover fixed scheduling,
  independent recurrence oracles, and opt-in research variants. Documentation
  distinguishes the original ReplaySSM GDN port from our optimized CUDA dense
  baseline; the historical 182-us result is the latter. See
  [baseline provenance](gdn_replayssm_baseline_provenance.md).

## Evidence and current validation

The [evidence manifest](gdn_research_evidence_20260910/manifest.json) records
SHA256 hashes of the original staged sources before lint cleanup and copies
the existing B200 test logs. Historical logs report:

| Check | Historical result |
| --- | --- |
| Fixed-plan versus unplanned graph execution | 4 passed |
| Runtime integration | 28 passed |
| Fixed-plan versus preserved pre-change source, bitwise | 4 passed |
| Warp Cholesky | 25 passed |
| Coalesced RHS | 16 passed |
| Fixed-metric read and exact-state oracles | 12 passed |
| Fixed-metric sanitizer subset | 2 passed; zero memory errors |

These are September 7 B200 records, not new runs on today's GPU.
On September 10, the four fixed-plan cases were attempted on an RTX PRO 6000
Blackwell. All stopped with `cudaErrorNoKernelImageForDevice`: the flush
extension explicitly compiles `compute_100f/sm_100f` for B200, whereas this GPU
is SM120. Setting `TORCH_CUDA_ARCH_LIST=12.0` does not replace that explicit
compiler flag. The [attempt log](gdn_research_evidence_20260910/rtx_plan_attempt.log)
is retained. This checkpoint does not claim a successful RTX correctness run
or silently change the architecture target.

Only lint cleanup was added during preservation: fenced-code language labels,
a separate variable for selecting a test workspace class, and accelerator API
names for test synchronization and benchmark cache clearing. CUDA arithmetic
was not changed. The original benchmark artifacts remain at the locations
recorded in the linked research notes. No model evaluation was launched.
