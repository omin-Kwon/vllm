# KDA LatchSSM r=128: implementation and paper audit

Date: 2026-09-05. Paper snapshot: `latchssm_paper@1f461a9`.
Engine base: `glm53-baseline@e3aedf6003`. Work branch: `kda-latch-r128`.

## Scope and acceptance

The accuracy baseline is the existing dense KDA model with ReplaySSM OFF.
There was no previously validated KDA ReplaySSM baseline. The new ring and
flush are part of the implementation under test; neither is a trusted oracle.

This first implementation provides a standalone eager Triton kernel API and
an AFM pretrained-model evaluation adapter. It does **not** yet wire the new
cache into GLM's vLLM scheduler, prefill metadata, mixed batches, prefix cache,
CUDA graphs, or speculative decode. Those paths must not silently select it.

The current flush is exact arithmetic over a register-resident state tile.
Its **subsequent torch FP64 latch/Phi epilogue rereads the updated state**.
It therefore does not yet satisfy the paper's fused-refresh traffic/performance
target. No measured speedup or production readiness is claimed.

## Historical failure and the required regression

`vllm-qwen38next@575c2a875f` (2026-09-02), "Restore projected f_s path for exact
GDN writes", restores both the key-side recurrence and the query-side erase
correction. It also removes the approximate checkpoint key-read from the
online write. Earlier tests could pass with a full-state fallback even when
the latch contribution was missing.

The new tests therefore use nonzero initial checkpoints, small G, correlated
keys and queries, nonuniform channel decay, and multiple flush boundaries.
They compare the *approximate* low-rank output to the paper projection, not to
an exact dense output it should not reproduce. They separately compare the
boundary checkpoint to dense. Deleting previous f_s is an explicit negative
control that must produce a detectable error.

## Equations and code correspondence

Let `H = S0.T` have shape `(V,K)`; buffers add `(slot,head)` axes.
`K=V=r=128`, `W=16`. Omega is a fixed full-column-rank `(K,G)` matrix.
The initial experiment uses a uniform rank across heads, with optional
per-head exact dense fallback. There is no calibrated head allocator yet.

| Paper object/equation | Implementation |
| --- | --- |
| Update: `M_t = (I - beta_t k_t k_t.T) D_t` | `_flush`: decay state first, then delta erase/write |
| GLM decay | `log(a)=-5*sigmoid(exp(A_log)*(raw_g+bias))` in `_step` |
| AFM decay | Optional original `-exp(A_log)*softplus(raw_g+bias)`; never substitute GLM gate in pretrained AFM |
| Query/key normalization | FP32 `x/sqrt(sum(x*x)+1e-6)`; query additionally scaled by `K^-0.5` |
| Latch `U=H Omega` | `KDALatchState.refresh`, explicit `(V,G)` buffer, no coordinate embedding |
| Metric ridge (paper fn:ridge) | `Phi=(H.T H + eta I)Omega [Omega.T(H.T H+eta I)Omega]^-1` |
| `Z.T` contractions | Use `Phi.T` instead, folding the inverse into coefficient metadata as in latest GDN |
| `d_[t]` | `exp(prefix_t)`, with prefix of raw channel log-decays stored in FP32 |
| `ell_s(t)` / eq. ringmerge | `k_s * exp(prefix_t-prefix_s)`; never divide by tiny `d_s` |
| eq. wyread, key | `f=beta*(sum(phi*d*k)-sum(f_past*kk))` |
| eq. wyread, query | `c=sum(phi*d*q)-sum(f_past*kq)-f_current*dot(k,q)` |
| Exact raw-write replay | `u_t=beta_t*(v_t-sum(u_past*kk))`; output `sum(u_past*kq)+u_t*dot(k,q)` |
| Read split | `out=U*c+replay` on every step, including the last step of a window |
| Exact boundary update | `_flush` accepts only raw `k/v/log_a/beta`, State, slots, position and head mask |
| Truncation | Disabled (`r=K`); no anchors, channel selection, band approximation or rotation |
| Offline basis | Full-transition effective-query covariance and averaged state metric, whitened PCA |

The replay `u_t` is an algebraically transformed *state-independent* raw write.
The original `v_t` and beta are also retained so the first flush can use the
literal causal recurrence without relying on the transformed ring.

Ridge is added to the state metric, not simply to `U.T U`: the denominator
contains `eta*Omega.T Omega`, and the numerator contains `eta*Omega`. The tests
use nonorthogonal Omega to distinguish these formulations. For a full-rank
basis, the state-read projection is the identity even with this ridge.

Boundary exactness is conditional on the **same supplied raw inputs**. It does
not assert identical model-wide states to a separate dense inference whose
earlier layer outputs, and hence later layer inputs, differ under approximation.

## Independent validation

`tests/models/glm5next/test_kda_latch.py` uses:

1. The original GLM `fused_recurrent_kda` with physical cache slots 1 and 2
   (slot 0 is vLLM NULL_BLOCK_ID), over 49 decode steps at G=128.
2. A separate FP64 oracle multiplying full `KxK` transition matrices. It solves
   the augmented least-squares problem `[H Omega; sqrt(eta) Omega] c =
   [H q_effective; sqrt(eta) q_effective]`, without Phi or projected f recurrence.
3. Direct verification of every stored `f_t = Phi.T pi_t`, where
   `pi_t = beta_t M_(1:t-1).T D_t k_t` is formed with the full transition.
4. Low-rank G=5/8/32 with dense fallback heads and multiple boundary checks.
5. A negative control erasing the previous f buffers.
6. A flush independence check poisoning f/u/U/Phi before the final step.
7. Shuffled, asynchronous slot progression and reset after an abandoned window.
8. Strong alternating channel decay, including cumulative log-decay -80.
9. The unmodified small-model softplus gate path.

The two initial GLM reference-call failures were harness errors: missing
state-index metadata, followed by using reserved slot 0. Corrected by allocating
a sentinel and supplying indices `[1,2]`; no baseline kernel was changed and
no tolerance was loosened. The first seven tests then all passed on RTX PRO
6000 Blackwell Server Edition. Additional f-factor checks are recorded with
the final test report.

## Reproduction

Use this checkout's `.venv` (uv-created, inheriting the existing CUDA 13.0
PyTorch environment). GPU access may require running outside the filesystem
sandbox. Set writable caches explicitly.

```bash
PYTHONPATH=$PWD \
TRITON_CACHE_DIR=/disk2/omin/.cache/triton-kda-latch \
KDA_TEST_REPORT=/disk2/omin/kda-latch-results/kernel_verification.json \
.venv/bin/python tests/models/glm5next/test_kda_latch.py

HF_HOME=/disk2/omin/.cache/huggingface \
HF_MODULES_CACHE=/disk2/omin/.cache/huggingface/modules \
PYTHONPATH=$PWD TRITON_CACHE_DIR=/disk2/omin/.cache/triton-kda-latch \
.venv/bin/python benchmarks/kernels/evaluate_kda_latch.py \
  --output /disk2/omin/kda-latch-results/afm_r128_w16.json
```

AFM evaluation: original dense FLA baseline, FP32 state, BF16 weights and
activations; 2 calibration segments and 4 disjoint evaluation segments from
the local WikiText text (hash and token offsets saved in JSON). Each has 128
prefill tokens and 65 scored continuation tokens (the first is predicted by
prefill, followed by 64 actual decode steps). Report NLL/perplexity, next-token
accuracy and dense-logit agreement. This 260-token teacher-forced run is a smoke
evaluation, **not** GSM8K/MATH/AIME/GLM task accuracy or a statistical guarantee.

Before larger accuracy campaigns: complete GLM cache integration tests,
verify original gate/normalization/cache ABI at each adapter, and repeat the
paper audit after any ring, f_s, or flush optimization. Before speed claims:
fuse the refresh epilogue and measure actual traffic and non-flush/flush timing.

## Results of the first run

Hardware: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported VRAM.
PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.16.1, FLA 0.5.2.
AFM snapshot: `295be1a2acd9a84a72ec1f1e267e25b5af4e7f56` (5.00B loaded
parameters; 27 KDA layers). Baseline prefill/decode loaded successfully.

Final GPU suite: **9 tests passed, 0 failures/errors**. In addition to the
checks listed above, a test changes only the persistent checkpoint after
refresh and confirms a non-flush output is unchanged; zeroing the latch
instead changes the output. This detects a hidden dense read substituting
for a nonfunctional latch.

| Check | Measured maximum absolute difference |
| --- | ---: |
| G128 versus original GLM dense, output/checkpoint comparisons combined | 6.10e-5 |
| Stored f_t versus direct full-transition projection | 2.11e-8 |
| Deliberately omit previous f_s (must differ) | 6.84e-3 |
| Deliberately zero the latch (must differ) | 1.42e-2 |
| Poison f/u/U/Phi before flush | Boundary state remained bitwise identical |

Low-rank output tests use atol=8e-4, rtol=6e-3 (BF16 output); boundary states
use atol=3e-5, rtol=2e-4 (FP32). The saved JSON also reports observed errors
for the other independent reference cases.

AFM WikiText smoke, 260 scored tokens, r=128, W=16, uniform G:

| Arm | NLL | Perplexity | Next-token accuracy | Dense top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Original dense, ReplaySSM OFF | 2.77522 | 16.0421 | 39.62% | 100% |
| G128 | 2.77808 | 16.0881 | 39.62% | 99.23% |
| G32 | 2.77691 | 16.0693 | 39.62% | 99.62% |
| G8 | 2.78955 | 16.2736 | 39.62% | 97.31% |

G128 is algebraically exact but **not bitwise identical model-wide**: BF16
outputs and changed FP32 reduction order propagate through later layers.
Its mean KL from dense is 4.90e-4; G8 is 5.32e-3. Equal top-1 correctness
counts on this small sample do not establish unchanged task accuracy.

Full local artifacts (no GLM full-model run was launched):

- `/disk2/omin/kda-latch-results/kernel_verification.json` and `.log`
- `/disk2/omin/kda-latch-results/afm_r128_w16.json` and `.log`
- `/disk2/omin/kda-latch-results/afm_r128_w16.basis.pt`
- `/disk2/omin/kda-latch-results/precommit.log`
