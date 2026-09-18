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
