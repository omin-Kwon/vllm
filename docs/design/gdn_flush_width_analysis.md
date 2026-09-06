# GDN flush width analysis on B200

Source: `gdn_flush_full_cuda.py`, independent full-coordinate kernel after
removal of anchors. No kernel implementation was changed for this analysis.

## What should and should not be constant

The exact raw-WY boundary update for a positive-width sketch head depends on
K, V and W, but not m. Its full-state read/write traffic is also independent
of m. The complete flush additionally refreshes the sketch and coefficient map.
With the runtime state H of shape V-by-K:

```text
U = H[:, :m]
Z_eta = (H^T H + eta I)[:, :m]
M = Z_eta[:m, :]
P = M^{-1} Z_eta^T
```

For 0<m<K, Gram formation costs O(VKm), Cholesky O(m^3), and solving the
nonidentity tail O(m^2(K-m)). The first m coefficient columns are already
identity; only K-m RHS are solved. Removing input rank truncation does not
remove this work. Nearly constant total flush time would require hiding this
additional work behind state traffic; it is not a mathematical invariant.

At m=K=128, P=I in this embedded representation. The implementation therefore
skips Gram/norm formation and solve entirely. Its abrupt drop at 128 is an
intentional algebraic fast path, not evidence that a 128-wide solve is cheap.
U refresh and exact raw-WY folding still execute. The zero-width/dense path
uses a different, exact-delta ring convention.

## Controlled width sweep

GPU: NVIDIA B200, driver 580.126.09. B=128, H=16, HV=48, K=V=128, W=16.
Every head has the same m; no dense/sketch mixtures. FP32 state and rings,
ridge coefficient 0.1. For each m, identical seeded state/key/write data and
normalized keys are used. G=ceil4(m), at least four. Each time is the median
of five samples of 32 graph-captured invocations. Parts are timed separately,
so their sum need not equal the complete flush. Prep remains about 16.7 us.

| m | G | Stream (fold + metadata), us | Solve, us | Full flush, us |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 317.9 | 61.0 | 406.8 |
| 8 | 8 | 319.4 | 102.9 | 451.3 |
| 16 | 16 | 366.9 | 186.2 | 584.2 |
| 17 | 20 | 560.5 | 345.9 | 934.6 |
| 24 | 24 | 542.1 | 449.6 | 1020.0 |
| 32 | 32 | 875.5 | 584.5 | 1488.3 |
| 33 | 36 | 756.3 | 1981.8 | 2765.1 |
| 48 | 48 | 882.1 | 2444.3 | 3351.7 |
| 64 | 64 | 1544.6 | 2646.5 | 4221.8 |
| 80 | 80 | 1491.4 | 2587.0 | 4103.2 |
| 96 | 96 | 2502.2 | 2386.6 | 4912.2 |
| 120 | 120 | 1967.7 | 1866.6 | 3854.1 |
| 127 | 128 | 3306.0 | 2365.5 | 5688.7 |
| 128 | 128 | 230.8 | 7.3 (early returns) | 255.3 |

Previous tables mixed uniform m8 with 25%-dense m32/m128 configurations. This
sweep removes that confound and still shows large discontinuities below 128.
The residual m4-vs-m8 stream similarity also reflects rounding both to an
eight-column Gram tile.

## Bottleneck 1: shared Gram accumulation becomes contended CAS loops

The stream reserves SPHI_MAX=2048 floats for coefficient partials. Its policy is:

```text
mc = ceil8(m)
RBC = max(1, min(8, 2048 / (128 * mc)))
NB = min(8, 2048 / (16 * RBC * mc))
slice = warp % NB
```

NB=8 gives each warp private partials and ordinary stores. With NB<8, warps
share slices and use `red_shared_f32`, then drain and clear the partials.

| m | Padded columns | Private/shared slices NB | Consequence |
| ---: | ---: | ---: | :--- |
| 8 | 8 | 8 | Private stores; four row-chunk drains |
| 16 | 16 | 8 | Private stores; eight row-chunk drains |
| 17 | 24 | 5 | Atomic accumulation starts |
| 32 | 32 | 4 | Two warps share each slice |
| 64 | 64 | 2 | Four warps share each slice |
| 80..127 | 80..128 | 1 | Eight warps share one slice |

Nsight Compute and its correlated SASS show that the source's
`red.shared.add.f32` is lowered to **ATOMS.CAST.SPIN retry sequences** in this
B200 build. Merely changing the source from CUDA atomicAdd to PTX red.shared
did not remove this cost. In the m32 stream, hardware counters report about
49.7 million shared-load bank conflicts (average 4.8-way shared-load pattern).
The dominant source location for excessive shared wavefronts is the Gram
`red_shared_f32` call. These source-derived wavefront totals and hardware bank
conflict counters are different metrics and should not be added together.

m32 stream DRAM throughput is only 15.5% of peak, eligible warps per scheduler
0.41, achieved occupancy 24.8%, and short-scoreboard stalls about 50.5% of
average warp cycles per issued instruction. This is not a saturated HBM path.
The compiled kernel uses 128 registers/thread and about 112 KB dynamic shared
memory per block, limiting its ability to hide these waits. A small stack spill
also exists, but the observed shared-reduction behavior is the clearer target.

The row-major partial layout matters as well: m33's padded width 40 has fewer
bank conflicts than m32's width 32, so its stream is faster despite more Gram
arithmetic. Similar sawtooth behavior occurs at 64/80 and 96/120. It cannot be
explained solely by total FLOPs.

## Bottleneck 2: generic solve geometry multiplies issued instructions

The solver uses this runtime dispatch within one kernel:

| Width range | Columns per warp | Row capacity per RHS | RHS parallelism per CTA |
| :--- | ---: | ---: | ---: |
| m<=16 | 4 | 16 | 16 |
| 17<=m<=32 | 2 | 32 | 8 |
| m>=33 | 1 | 128 | 4 |

The implementation computes forward/back updates for every compiled register
row, including rows outside actual m. Loads clamp those rows to m-1 and the
final stores mask them out; their intermediate arithmetic still executes.
For example, m8 uses a 16-row capacity, while m33 switches from the m32 path's
32-row capacity to 128. Column parallelism halves at the same boundary.
There are also serial per-column k loops, repeated address/min/selection logic,
and a width-four Cholesky panel schedule with block barriers. A stale source
comment says two barriers per panel, but the body contains three.

The discontinuity is measured, not inferred only from source:

| Quantity | m32 | m33 |
| :--- | ---: | ---: |
| Graph solve time | 584.5 us | 1981.8 us |
| m^2(K-m), proportional solve work | 98,304 | 103,455 (+5.2%) |
| NCU issued instructions | 542,380,032 | 1,809,094,656 (3.34x) |
| NCU ALU pipeline utilization | 69.7% | 80.4% |
| NCU DRAM throughput | 4.0% | 1.4% |

Correlated source counters put most of the extra instructions in the generic
forward/back updates and row-selection branches, rather than Cholesky alone.
m8 solve is also instruction-heavy: about 100.9 million issued instructions,
71.1% ALU utilization and 0.9% DRAM throughput. Here "ALU" means integer/logic
pipeline work, not useful matrix-multiply FLOPs. Specializing actual row/column
geometry is more promising than tuning HBM loads in this solve.

## Bottleneck 3: buffer allocation width G affects small-m execution

A second sweep fixes G=128 while varying actual m. This isolates a separate
inefficiency relevant to heterogeneous allocations:

| Actual m | Flush with minimal G, us | Flush with G128, us | Solve with minimal G / G128, us |
| ---: | ---: | ---: | ---: |
| 8 | 451.3 | 790.4 | 102.9 / 335.9 |
| 32 | 1488.3 | 2067.2 | 584.5 / 1074.1 |
| 64 | 4221.8 | 5581.8 | 2646.5 / 3815.2 |

Even m8 receives 82,688 bytes of dynamic solve shared memory in a G128 buffer,
versus 4,160 with G8: the host allocation uses the largest possible width, not
the head's actual m. This limits block residency. The stream also zeros unused
scratch/U columns and the solver writes G*K outputs, including identity and
inactive rows. This is a combined layout/allocation penalty; the sweep does not
attribute the whole difference to shared memory alone.

## What to change next

1. Replace the generic solve's row-capacity/column dispatch with actual-width
   specializations, beginning with m8/16/32/33/64. Eliminate arithmetic for
   nonexistent rows, improve RHS parallelism, and reduce repeated index logic.
   Treat factorization and RHS application as separate profiling targets.
2. Give Gram partials unique writers and use ordinary reductions, or change the
   output tile ownership. Remove CAS retries and fix bank mapping. Simply making
   all private buffers larger may lower occupancy or outgrow the reused ring
   buffer, so it requires a tile/dataflow redesign rather than one constant edit.
3. Allocate/schedule solve work by actual width and stop writing inactive or
   implicit-identity metadata, coordinating any packed layout with the step.
4. Preserve the exact state fold and the m128 identity fast path. Moving the
   inverse into U, as in the paper, changes the representation but does not make
   factorization/application free; projected factors and ridge policy must stay
   consistent if that route is chosen.

The objective can remain low overhead relative to ReplaySSM flush, but exact
flat latency across every m is not guaranteed by the scheme. The present large
overhead includes demonstrated avoidable scheduling and communication costs.

## State embedding versus folding the inverse

These are separate choices. Embedding uses a fixed orthogonal rotation so
U is a prefix of the rotated state. It makes U extraction a copy or view.
Without it, an explicit sketch still needs U=S^T Omega, Z=S U and its Gram
to implement the same reconstruction. Removing embedding does not by itself
remove Gram formation or the coefficient solve. The experimental CUDA path
already copies U into a separate buffer; it does not alias U with the state
allocation or implement the paper's capacity saving from physical aliasing.

Folding the inverse is an algebraic reassociation: the paper stores U M^-1,
whereas this kernel stores P=M^-1 Z_eta^T and keeps U unmodified. Storing the
inverse or a factor separately can move RHS application from flush to each
step, but still requires Gram/factorization and adds per-step metadata traffic
and arithmetic. Such a change should be compared over the whole W-step window,
with the same ridge and projected-factor convention. The optimization below
retains the existing U/P representation and exact raw-WY fold.

## Artifacts and reproduction

Artifacts are under `/disk2/omin/kda-latch-results/gdn_full_dimension_review`:

- `flush_width_sweep.json` and `flush_width_sweep_g128.json` (graph timings).
- `ncu_flush_m8/m32/m33.ncu-rep`, matching details text, and m32/m33 SASS CSV.
- `flush_source_hotspots.json` (correlated source summaries).

NCU uses `--clock-control none --cache-control none`; its kernel replay times
are diagnostic and are not substituted for the graph timings above. Some
unrelated C2C-link metrics were unavailable; required SM/shared-memory/source
counters were collected. No precision or functional-path changes were made.

Use the PATH/TORCH_CUDA_ARCH_LIST settings in `gdn_sketch_full_dimension.md`:

```bash
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --width-sweep --output /tmp/gdn_widths.json
.venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --width-sweep --allocation-g 128 --output /tmp/gdn_widths_g128.json
/usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --kernel-name 'regex:gdn_(flush_stream|ls6_solve)_kernel' --set full --export /tmp/gdn_m32 .venv/bin/python benchmarks/kernels/benchmark_gdn_sketch.py --profile-width 32 --output /tmp/gdn_m32.json
```

## Implemented tuning: private Gram tiles and width-specific solves

The historical tables above describe the previous implementation. The new
stream assigns each warp a private 256-float slice containing two 16x8 MMA
tiles. Fragment-order stores give contiguous lane addresses; an ordinary sum
combines the eight slices. Two buffers alternate between computation and drain.
For m<=8, each pair covers two row tiles; otherwise it covers two column tiles.
There is no shared floating-point atomic accumulation or partial-buffer reset
between tiles. The same three-product FP16 hi/lo arithmetic and FP32 ridge are
retained. Reduction ordering changes, within the tested numerical tolerances.

Tile indices are statically unrolled. An initial runtime-index version caused
a 416-byte stream stack and small-m regressions; it was rejected. The final
stream has a reported 8-byte stack and 128 registers/thread. Its SASS has no
ATOMS.CAST.SPIN sequence; the existing integer work-list ATOMS.ADD remains.
This is not a claim of zero spills or zero synchronization.

For m<=64, one thread solves one tail RHS, retaining its vector in registers.
Separate kernels cover 1..8, 9..16, 17..32, 33..48 and 49..64. Their shared
memory and register limits no longer follow the largest allocated G. The
compiled register counts are 32, 48, 71, 72 and 96 respectively. Width checks
remain on the GPU: no host scalar reads or graph-capture allocations were added.
The >64 path retains the previous row-distributed substitution and allocation
stride; optimizing that solver remains outstanding. The factorization is still
the established FP32 width-four Cholesky with three barriers per panel.

Same B128/H16/HV48/K=V128/W16 conditions and graph timing protocol:

| m | Old stream, us | New stream, us | Old solve, us | New solve, us | Old flush, us | New flush, us | Flush speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 317.9 | 216.5 | 61.0 | 18.4 | 406.8 | 253.3 | 1.61x |
| 8 | 319.4 | 216.5 | 102.9 | 26.9 | 451.3 | 262.5 | 1.72x |
| 16 | 366.9 | 254.5 | 186.2 | 75.9 | 584.2 | 353.5 | 1.65x |
| 17 | 560.5 | 309.0 | 345.9 | 143.6 | 934.6 | 478.8 | 1.95x |
| 24 | 542.1 | 294.8 | 449.6 | 167.1 | 1020.0 | 487.3 | 2.09x |
| 32 | 875.5 | 323.6 | 584.5 | 241.5 | 1488.3 | 590.7 | 2.52x |
| 33 | 756.3 | 382.0 | 1981.8 | 339.2 | 2765.1 | 768.0 | 3.60x |
| 48 | 882.1 | 384.4 | 2444.3 | 443.4 | 3351.7 | 867.9 | 3.86x |
| 64 | 1544.6 | 458.1 | 2646.5 | 900.5 | 4221.8 | 1431.1 | 2.95x |
| 80 | 1491.4 | 529.6 | 2587.0 | 2770.0 | 4103.2 | 3332.3 | 1.23x |
| 96 | 2502.2 | 632.9 | 2386.6 | 2562.8 | 4912.2 | 3224.6 | 1.52x |
| 120 | 1967.7 | 898.2 | 1866.6 | 1971.1 | 3854.1 | 2885.5 | 1.34x |
| 127 | 3306.0 | 1103.5 | 2365.5 | 2472.4 | 5688.7 | 3605.7 | 1.58x |
| 128 | 230.8 | 219.1 | 7.3 | 29.0 | 255.3 | 265.2 | 0.96x |

Solve timings include every launched width bucket, including empty buckets.
At G128 there are six launches. This adds roughly 10 us to the full-m128
flush (255 -> 265 us) despite a faster stream; no Gram or solve arithmetic is
performed for those heads. Eliminating empty-bucket launch overhead requires
an allocation-aware launch plan. Small-G launches are fewer.

With G fixed at 128, m8 full flush falls from 790.4 to 474.7 us, m32 from
2067.2 to 753.2 us, and m64 from 5581.8 to 1702.0 us. The remaining gap against
minimal G includes padded U/Phi writes and strided accesses as well as empty
launches. Width specialization does not make these allocated buffers packed.
Large partial widths still pay substantial factorization/substitution costs.

Validation: 60 kernel cases passed, including expanded window tests at G16,
G20 (m17), G36 (m33), FP32/BF16 gates, next-window outputs, padded state pages,
compact mappings, empty/partial flushes and the m128 identity path. The
work-list test was then expanded to include **every m from 0 through 128**
with HV/H=4 and persistent table refills; it passed under Compute Sanitizer
memcheck with **0 errors**. This tests the standalone kernels; no model-level
accuracy or serving throughput claim is made and production dispatch is unchanged.

Results: `flush_optimized_widths.json`, `flush_optimized_g128.json`,
`flush_optimized_cycles.json`, `pytest_optimized.log`, and
`flush_optimized_memcheck.log` in the review result directory. Build cache:
`/disk2/omin/.cache/gdn_full_flush_optimized` (set
`NS_GDN_FULL_FLUSH_BUILD_DIR` to reproduce without another compile).

The complete 16-step window also improves (same previous full-coordinate
kernel versus this tuning, not the r-truncation reference):

| B | G | Head mix | Previous flush, us | New flush, us | Previous cycle, us | New cycle, us |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | 8 | uniform | 30.8 | 19.5 | 111.0 | 98.7 |
| 32 | 8 | uniform | 136.9 | 78.7 | 246.2 | 187.6 |
| 128 | 8 | uniform | 450.5 | 262.8 | 730.8 | 545.1 |
| 128 | 8 | 25% dense | 440.3 | 257.4 | 1092.1 | 908.5 |
| 128 | 32 | 25% dense | 1356.2 | 543.4 | 2244.2 | 1442.2 |
| 128 | 128 | 25% dense | 255.1 | 276.1 | 1495.4 | 1506.6 |
| 128 | 128 | heterogeneous | 2441.2 | 1511.2 | 3536.9 | 2594.6 |

Native ReplaySSM is separately remeasured: B128 m8 configuration 182.2 us
versus the new sketch flush 262.8 us (1.44x), and the m32/25%-dense
configuration 181.6 us versus 543.4 us (2.99x). Proper delta versus raw-WY
rings are constructed from the same inputs; the boundary-state maximum
absolute difference is 3.58e-7. This is not a claim that the new sketch flush
is faster than the native ReplaySSM flush. The extra reconstruction work
remains visible.
