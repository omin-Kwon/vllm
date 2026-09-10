# GDN flush: solver and representation research on B200

Measured 2026-09-06 on B200, 148 SMs, CUDA 13.0, driver 580.126.09.
These are isolated research variants. The default GDN step/flush dispatch is
unchanged. No model evaluation or allocation campaign was restarted.

## Mathematical contract

Write the runtime boundary state as $H=S^T\in\mathbb R^{V\times K}$, with
$K=V=128$. For an embedded prefix of width $m$, let

$$
U=H[:, :m],\qquad J=[I_m\;0],\qquad
\eta=0.1\,\operatorname{tr}(H^T H)/K.
$$

The current experimental CUDA representation is

$$
M=U^TU+\eta I_m,\qquad Z_\eta^T=U^TH+\eta J,\qquad
P=M^{-1}Z_\eta^T.
$$

The state contribution is $UP\tilde q$. The existing CUDA kernel does not form
an explicit inverse: it uses Cholesky and two triangular substitutions. Because
$P[:, :m]=I_m$, it solves only the $K-m$ tail right-hand sides. The paper folds
the inverse into the other factor; both placements give the same contraction
when the ridge numerator and the projected replay factors are treated consistently.

The calibration rotation makes the selected input coordinates a prefix. It does
not make the current, state-dependent columns of $U$ orthonormal. Full width
$m=128$ is a separate exact path with implicit $P=I$. Width zero uses the exact
dense delta ring; positive widths use the raw-WY ring convention.

## Removing both the inverse and triangular solve

For zero ridge and full column rank, thin QR gives
$U=QR$ and $U(U^TU)^{-1}U^T=QQ^T$. Store $Q$ and $Q^TH$ instead of $U,P$.
No inverse or triangular solve is necessary to construct or apply these factors.

To preserve the actual nonzero ridge, orthogonalize the augmented sketch:

$$
A=\begin{bmatrix}U\\\sqrt\eta I_m\end{bmatrix}
=\begin{bmatrix}Q_v\\Q_b\end{bmatrix}R.
$$

Then the exact identity is

$$
UM^{-1}(U^TH+\eta J)
=Q_v\left(Q_v^TH+\sqrt\eta Q_b^TJ\right).
$$

Thus the replacement buffers are

$$
U'=Q_v,\qquad P'=Q_v^TH+\sqrt\eta Q_b^TJ.
$$

The existing step kernel can consume these two buffers unchanged. Its projected
ring factors must be generated in the new $P'$ coordinates, which happens
naturally when the new metadata is installed at a window boundary. The raw-WY
state update is unchanged. No persistent $m\times m$ buffer is added. Only the
full augmented $Q$ is orthonormal; $Q_v$ alone generally is not.

This is an algebraic identity, not another truncation or iterative approximation.
Floating-point QR and matrix multiplication still introduce rounding. Orthogonal
columns do not eliminate the cost of computing the basis: QR costs roughly
$O((V+m)m^2)$, and the new coefficient product costs $O(VKm)$.

Three implementations were evaluated:

1. cuSolverDx Householder GEQRF + UNGQR, with a shared-memory FP32 coefficient
   product in the same block.
2. The same QR with less shared memory, followed by a Tensor Core product.
3. A custom warp-distributed modified Gram-Schmidt QR, followed by the same
   Tensor Core product. Coalesced state reads feed a padded shared sketch;
   width buckets 8/16/32 use 32/40/64 registers and no spills.

The Tensor Core product uses three TF32 products (`tf32x3`) with FP32 accumulation,
not a single reduced-precision TF32 product. Its reduction is tiled to reduce
register pressure after spills were found in the first width-32 implementation. All tested QR
variants omit the old stream Gram computation and solver. The current prototype
still writes the raw prefix U during folding before QR overwrites it, and rereads
the boundary state for QR and the coefficient product. A fully fused design can
remove some of these costs; the measurements are not a lower bound for QR.

The QR research specializations currently support $G\le32$, every $m\in[0,G]$,
and nonzero-state positive ridge. Zero state is also tested. Arbitrarily small
ridge with nearly dependent columns needs further stability testing before
general production use. Full-width dispatch remains in the original kernel.

The final custom QR prototype remained slower than the optimized current path.
All numbers below are microseconds, at the same batch/head/window dimensions
as the solver comparison. Both QR and coefficient construction are included.

| m | Current flush | Custom QR flush | Current 16-token cycle | Custom QR cycle |
| --- | ---: | ---: | ---: | ---: |
| 8 | 262.8 | 451.9 | 544.6 | 733.1 |
| 16 | 353.9 | 524.4 | 798.3 | 957.9 |
| 32 | 591.6 | 1080.8 | 1287.2 | 1775.9 |

The first fused Householder prototype took 598/769/1758 us per flush at
m=8/16/32. Separating its coefficient product reduced this to 492/580/1201 us
before the final product tiling change. Removing inverse/solve is feasible;
these prototypes do not yet justify replacing the current default for speed.

## Batched library comparison on identical Gram matrices

Batch 128, H=16, HV=48, W=16, uniform width; 6,144 independent systems.
Microseconds, including device packing, all library calls, and coefficient
scatter. Handles, pointer arrays, scratch, and graph setup are outside timing.

| m | Current solve | cuSOLVER POTRF + TRSM | cuBLAS inverse + GEMM | cuSolverDx POSV, fixed 128 threads |
| --- | ---: | ---: | ---: | ---: |
| 8 | 26.8 | 198.3 | 120.7 | 62.5 |
| 16 | 75.9 | 380.8 | 204.0 | 111.9 |
| 32 | 241.7 | 888.9 | 554.5 | 365.6 |
| 64 | 902.3 | 1803.6 | — | 821.3 |
| 80 | 2770.2 | 1911.6 | — | 723.2 |
| 96 | 2563.3 | 1961.7 | — | 1165.8 |

The inverse column uses `matinvBatched`, limited here to m<=32. Separate LU,
LU-inverse, and custom register Gauss-Jordan candidates also ran; their complete
results are in the result directory. The tested Gauss-Jordan kernel was slower
and spilled at larger widths. This does not rule out other implementations.

cuSOLVER's batched POTRS supports one RHS, so the host-library Cholesky candidate
uses two cuBLAS batched TRSM calls. The fused device POSV path avoids global
packing. Using cuSolverDx's suggested block size improves m=8 to 18.9 us and m=16
to 68.8 us, but worsens m=32 to 711.7 us. Suggested size plus identity padding
reduces m=127 from 7,243 to 1,454 us. These configurations use one batch per block;
batches-per-block tuning was not exhausted.

## Moving the inverse out of the coefficient matrix

The unfolded experiment stores raw $Z_\eta^T$ and $M^{-1}$. Since the projected
WY recurrence is linear, only the final query coefficient needs the inverse
matvec; the key-side factor is stored directly in Z coordinates. This reduces
the number of precomputed coefficient RHSs but adds a matrix read/matvec per
token. It preserved the tested outputs and state updates but lost total time:

| m | Current flush | Unfolded flush | Current 16-token cycle | Unfolded cycle |
| --- | ---: | ---: | ---: | ---: |
| 8 | 262.8 | 259.1 | 544.8 | 562.5 |
| 16 | 353.8 | 339.5 | 794.2 | 854.3 |
| 32 | 591.7 | 576.1 | 1288.5 | 1378.6 |

The same real-window harness measured cuSolverDx at m=8: 257.4 us flush and
537.7 us cycle. At m=80, it reduced flush 3332 to 1239 us and cycle 4830 to
2722 us. Large partial widths are useful diagnostics, but at these dimensions
$m(V+K+W)=272m$ already exceeds a dense $KV=16384$ state at m>=61 in the
paper's per-step sketch/cache traffic component. Allocation should consider
that dense exact option as well as latency.

An instrumented current-solver proxy returning immediately after factorization
took 15.5/44.3/138.9/510.6 us at m=8/16/32/80; complete instrumented solves took
27.0/74.5/242.9/2738.5 us. The proxy includes loading/preparation and changes
output work/compiler allocation, so subtraction is not an exact decomposition.
Both factorization and RHS application matter; their balance changes with m.

## Correctness and measurement limits

- Same post-flush Gram inputs were used for library comparisons, including
  independent FP64 solve oracles and backward residuals. FP64 is only a testing
  oracle; these candidate kernels use FP32 arithmetic.
- Actual step and flush comparisons cover three successive W=16 windows,
  mixed dense/sketch widths, nonidentity compact mappings, inactive rows,
  FP32/BF16 gates, poisoned metadata, raw writes, and transformed ring factors.
- The augmented QR projection test independently checks all m=1..32 on random,
  collinear, and zero states at scales 1e-3, 1, and 1e3. Nine pytest cases pass.
- Compute Sanitizer memcheck passes with zero errors for both Householder plus
  Tensor Core and custom QR paths, at m=8/16/32 with both gate dtypes. In these
  three-window comparisons state error is zero. The maximum BF16 output absolute
  difference across the compared QR cases and batch-128 trajectories is 2.44e-4.
- Timings use CUDA graphs: three warmups, 32 calls per graph, median of five
  event measurements. Both the complete flush and complete 16-token cycle are
  measured. Metadata-only timing for QR includes QR plus the coefficient product,
  despite the legacy JSON field being named `solve_us`.
- These are kernel experiments with synthetic inputs, not end-to-end model
  accuracy or serving-throughput results. Production dispatch has not changed.

## 2026-09-07: deferred solve, register Cholesky and memory access

Super evaluation is stopped at the user's request. New experiments compare
against the fixed-allocation solve plan described in `gdn_flush_solve_plan.md`.
They do not change the serving dispatch. The same B128/H16/HV48/W16 dimensions
and graph/event timing method above apply.

### Cache a factor and solve just one vector per token

`gdn_deferred_research.py` stores raw `C=Z_eta.T=[M,B]` instead of solved P,
plus a Cholesky factor. Linearity permits raw-C projected WY factors, with
`M^-1` applied only to the final query coefficient. This replaces K-m RHS
solves at flush with one vector solve per token. No explicit inverse is built;
the ridge numerator and exact state fold remain unchanged. It adds a persistent
FP32 G-by-G factor per slot/value head (4 KiB at G32).

The first version read the factor globally during substitutions. The second
asynchronously preloads it into shared memory before the existing step work.
Both passed three-window comparisons, including transformed projected factors.
Small mixed cases cover FP32/BF16 gates and m8/16/32/64. Maximum output error is
1.22e-4 there and 2.44e-4 in B128 runs; exact state error is zero. Reassociation
means output bitwise identity is not claimed. These checks do not establish
long model trajectory accuracy or numerical behavior for arbitrary ridge.

| m | Current flush | Factor-cache flush | Current cycle | Global-factor cycle | Shared-factor cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 263.47 | 256.86 | 546.53 | 598.16 | 582.72 |
| 16 | 345.67 | 319.60 | 793.05 | 945.17 | 815.60 |
| 32 | 582.32 | 490.28 | 1278.71 | 1665.12 | 1419.46 |
| 64 | 1411.91 | 901.89 | 2605.93 | 4030.94 | 3636.80 |

Current/flush columns are from the shared-factor run; global-factor cycles are
from its separately recorded run. Moving solves successfully reduced flush,
but increased whole-cycle time at every measured width. It is not a deployment
candidate on this evidence. Extra cache reads and serial triangular dependency
are plausible costs; they were not separately profiled in these variants.

### Small Cholesky in warp registers

`gdn_warp_cholesky_research.py` factors m<=32 within one warp using register rows
and shuffles, leaving RHS substitutions and the stored representation intact.
It replaces per-panel block barriers, but the measured gain is small. For
m8/16/32, solve changes 27.25/70.39/232.19 to 23.47/66.11/225.50 us. Complete
cycles change 546.42/793.26/1278.27 to 542.26/780.39/1279.70 us. Thus m32 has
no observed whole-cycle benefit in this run.

All three-window benchmark comparisons reported zero output/state/Phi error.
The opt-in existing pytest suite adds CUDA graph trajectories, dense/full-width
endpoints, compact mappings and independent recurrence oracles, and stresses
all m1..32 on random/collinear/zero states at scales 1e-3/1/1e3: 25 cases pass.
The independent oracle uses FP64; the kernel remains FP32. Resource inspection
finds 32/48/64 registers at m8/16/32, with no local spills.

### Profiling found strided RHS loads

Nsight Compute on current m32 reports 72% excessive global sectors and only
5.1 useful bytes per 32-byte sector for global loads. RHS loading assigned
adjacent lanes adjacent RHS columns, while scratch stores each column's m
entries contiguously. Thus those lanes read addresses separated by G floats.
The `coalesced` candidate changes only the assignment of load iterations:

```text
old: i=o/ncol, c=o%ncol
new: i=o%m,    c=o/m
same assignment: sX[i*NCS+c] = scratch[(m+c)*G+i]
```

Each destination receives exactly the same FP32 bits before the same block
barrier. Cholesky, RHS arithmetic, layout, precision and persistent capacity
are unchanged. This candidate does not use the warp-factor change above.

| m | Current solve | Coalesced solve | Current flush | Coalesced flush | Current cycle | Coalesced cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 27.30 | 27.18 | 263.67 | 263.64 | 546.54 | 546.23 |
| 16 | 70.40 | 64.16 | 346.56 | 340.26 | 793.31 | 777.93 |
| 32 | 232.32 | 183.84 | 582.40 | 533.25 | 1278.36 | 1233.00 |
| 64 | 889.57 | 849.38 | 1416.67 | 1374.62 | 2608.73 | 2556.86 |

The profiler replay ran at different SM clocks (about 1.12 GHz); its 396 us
duration is diagnostic and is not used in the timing table. Benchmark clocks
and ordinary graph/event timings must be used for speed comparisons.

The coalesced candidate passes all 16 opt-in graph/window regression cases
(G4/8/16/20/32/36/64/128, FP32/BF16 gates), including independent recurrence
and boundary coefficient checks. The B128 three-window comparisons above
report zero output and state error. No model evaluation was resumed.

Sources are research modules in `benchmarks/kernels`; raw JSON, logs, generated
global-factor source, resource/profile artifacts and the algebra oracle are in
`/disk2/omin/kda-latch-results/gdn_solver_research/deferred_20260907/`.
The coalesced source is a promising measured implementation fix, not evidence
that the runtime Gram or solve can be discarded.

Exact conditions on Omega, calibration whitening versus runtime whitening,
and invariant-subspace cancellation are derived in
`nested_ssm/docs/latch/GDN_OMEGA_NO_SOLVE_CONDITIONS_20260907.md`.

## Why other apparent shortcuts are not drop-in identities

For $H'=aH+XB^T$ with update rank at most W, the new Gram differs from $a^2M$
by rank at most 2W plus $(\eta'-a^2\eta)I$. The trace-based ridge changes each
window, so ignoring that full-rank shift in a Woodbury update changes the method.
For m<=32, 2W is also not smaller than m. Freezing the Gram or using only its
diagonal similarly changes the reconstruction. A fixed number of inverse
iterations requires an explicit residual/convergence policy and is not an
exact replacement merely because it uses matrix multiplication.

## Reproduction

Sources live in `benchmarks/kernels/gdn_solver_candidates.cu`,
`gdn_solver_gauss_jordan.cu`, `gdn_qr_candidates.cu`, `gdn_qr_mgs.cu`,
`gdn_unfolded_research.py`, `gdn_qr_research.py`, `benchmark_gdn_solvers.py`, and
`benchmark_gdn_solver_windows.py`. Generated extensions use separate caches.
Results and logs are under
`/disk2/omin/kda-latch-results/gdn_solver_research`.

```bash
cd /disk2/omin/vllm-qwen38next
export PATH="$PWD/.venv/bin:/disk2/omin/miniconda3/envs/vllm029_q38next/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=4
export NS_GDN_FULL_FLUSH_BUILD_DIR=/disk2/omin/.cache/gdn_full_flush_optimized
export NS_GDN_FULL_BUILD_DIR=/disk2/omin/.cache/gdn_full_cuda_v3
export GDN_RESEARCH_CACHE=/disk2/omin/.cache/gdn_solver_research
export GDN_RESEARCH_RESULTS=/disk2/omin/kda-latch-results/gdn_solver_research
export GDN_MATHDX="$GDN_RESEARCH_CACHE/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06"

nvcc -shared -Xcompiler=-fPIC -rdc=true -dlto -std=c++17 -arch=sm_100 -O3 -lineinfo \
  -I"$GDN_MATHDX/include" -I"$GDN_MATHDX/external/cutlass/include" \
  benchmarks/kernels/gdn_solver_candidates.cu -L"$GDN_MATHDX/lib" \
  -lcusolverdx -lcusolver -lcublas -o "$GDN_RESEARCH_CACHE/libgdn_solver_candidates_tuned.so"
nvcc -shared -Xcompiler=-fPIC -std=c++17 -arch=sm_100 -O3 -lineinfo \
  benchmarks/kernels/gdn_solver_gauss_jordan.cu -o "$GDN_RESEARCH_CACHE/libgdn_solver_gauss_jordan.so"
nvcc -shared -Xcompiler=-fPIC -rdc=true -dlto -std=c++17 -arch=sm_100 -O3 -lineinfo \
  -I"$GDN_MATHDX/include" -I"$GDN_MATHDX/external/cutlass/include" \
  benchmarks/kernels/gdn_qr_candidates.cu -L"$GDN_MATHDX/lib" \
  -lcusolverdx -o "$GDN_RESEARCH_CACHE/libgdn_qr_candidates.so"
nvcc -shared -Xcompiler=-fPIC -std=c++17 -arch=sm_100 -O3 -lineinfo \
  benchmarks/kernels/gdn_qr_mgs.cu -o "$GDN_RESEARCH_CACHE/libgdn_qr_mgs.so"

NS_GDN_QR_RESEARCH_TESTS=1 .venv/bin/python -m pytest \
  tests/kernels/mamba/test_gdn_sketch.py -q -k augmented_qr
.venv/bin/python benchmarks/kernels/benchmark_gdn_solver_windows.py \
  --library "$GDN_RESEARCH_CACHE/libgdn_solver_candidates_tuned.so" \
  --kinds current qr_mgs --widths 8 16 32 --bench \
  --output "$GDN_RESEARCH_RESULTS/qr_mgs_tuned_benchmark.json"
```

Omit `--bench` for mixed-width window validation with both gate dtypes. The MGS
path does not load or require the MathDx library; the harness's `--library` option
is only consumed for `dx` candidates. MathDx was downloaded into the cache from
NVIDIA's official distribution. On CUDA 13.0 the working build uses `-rdc=true
-dlto` and `libcusolverdx.a`; the attempted monolithic fatbin link was rejected.

## Primary references

- [NVIDIA cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [cuSolverDx fused POSV example](https://docs.nvidia.com/cuda/cusolverdx/get_started/introduction.html)
- [cuSolverDx Householder QR](https://docs.nvidia.com/cuda/cusolverdx/get_started/geqrf.html)
- [cuSolverDx installation and linking](https://docs.nvidia.com/cuda/cusolverdx/get_started/installation.html)
- [NVIDIA MathDx samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx)
- [Variable-size batched Gauss-Jordan research, 2018](https://icl.utk.edu/files/publications/2018/icl-utk-1068-2018.pdf)

The augmented ridge identity above is a derivation from the current runtime
contract. The references establish available algorithms/APIs; they do not claim
the B200 performance measured here.
