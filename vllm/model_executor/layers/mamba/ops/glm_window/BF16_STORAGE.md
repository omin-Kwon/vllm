# BF16 sketch storage

This branch (`port/kda-bf16-sketch`) extends `kda-latest` at `51a268f886`.
SketchSSM P4/P6 now store U (`latch`), the coefficient map (`phi`), and
projected erase history (`f`) in BF16. Reads explicitly convert to FP32 before
products and reductions. Metadata construction and solves remain FP32.
The recurrent state, gate/prefix buffers, and existing raw replay ring precision
are unchanged. Exact Replay is unaffected.

`kda_window.sketch_dtype` defaults to `"bfloat16"`; use `"float32"` for a
reference run. Calibration/checkpoints and rank allocation do not change.
The engine smoke audit reports all three sketch buffer dtypes.

Validation at publication (2026-09-18): Ruff and diff checks passed; 19 window
test cases collected. GPU numerical tests are queued behind another GPU job
and have **not passed yet**. No BF16 model accuracy result is claimed.

Run:

```bash
.venv/bin/python -m pytest tests/models/glm5next/test_kda_recurrent.py -k window -q
```

The tests cover independent FP64 pivot oracles, both storage dtypes and P4/P6,
BF16-vs-FP32 state and map comparisons, mixed routing, flush output independent
of sketch metadata, and long graph/slot lifecycle checks.

The companion GDN implementation and GPU queue are in `omin-Kwon/nested_ssm`,
branch `port/machine-agnostic-paths`, `scale/research/qwen_bf16_storage`.

## Validation update (2026-09-23, B300)

Completed GLM RULER runs used BF16 U/C/projected erase, as recorded in the
archived `SSM_results/data/ruler_recall_16k_2026-09-20/results/glm/` engine audits.
This does not establish the storage dtype of older reasoning evaluations.

The new low-rank reasoning queue initially stopped on a bit-exact comparison
between direct BF16 coefficient storage and FP32 coefficients cast to BF16.
Separate output-dtype specializations can straddle a rounding boundary; that
comparison is stricter than the intended numerical contract. The test now
bounds map error by half the BF16 relative spacing plus a per-head FP32
arithmetic tolerance near zero. State and rounded U comparisons remain exact.
Independent FP64 coefficient checks now also cover two flushes, not just
initialization. No production kernel arithmetic was changed.

All eight selected P4/P6 tests passed: independent FP64 oracle (both storage
dtypes), BF16 storage/state, and flush output with corrupted sketch metadata.
An additional 128-step diagnostic with reorder/release/reset found bit-identical
full states and maximum output relative norms of 0.002474 (P4) and 0.002550 (P6).
These are kernel checks with shared inputs, not model accuracy guarantees.

Local evidence:
`/disk2/omin/kda-latch-results/accuracy_lowrank_20260923/glm/debug_bf16/`.
