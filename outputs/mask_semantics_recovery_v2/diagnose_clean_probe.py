"""Aggregate-only diagnosis of the one M2 clean-probe mismatch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import src.mask_semantics_recovery_v2 as recovery
from src import ningbo_confirmation_v1 as ningbo

test, _, _ = recovery.load_parts()
indices = np.linspace(0, len(test) - 1, recovery.PROBE_RECORDS, dtype=np.int64)
raw = np.stack([ningbo.raw(test[i]) for i in indices]).astype(np.float32)
model, mean, std, _ = recovery.load_model("physics_aware_m2", 104729)
inputs = recovery.prepare_clean_input(raw, mean, std)
run_a = recovery.infer_batches(model, inputs, "cuda", 128, True)
run_b = recovery.infer_batches(model, inputs, "cuda", 128, True)
old = recovery.old_scores("physics_aware_m2", 104729, "clean")[indices]
delta = np.abs(old - run_a)
bad = ~np.isclose(old, run_a, rtol=recovery.CLEAN_TOL_RTOL, atol=recovery.CLEAN_TOL_ATOL)
coords = np.argwhere(bad)
worst = tuple(coords[np.argmax(delta[bad])]) if len(coords) else (0, 0)
print({
    "repeat_inference_bitwise_equal": bool(np.array_equal(run_a, run_b)),
    "old_vs_new_max_abs": float(delta.max()),
    "old_vs_new_mean_abs": float(delta.mean()),
    "failed_comparisons": int(bad.sum()),
    "total_comparisons": int(bad.size),
    "worst_old_score": float(old[worst]),
    "worst_new_score": float(run_a[worst]),
    "repeated_run_max_abs": float(np.abs(run_a - run_b).max()),
    "sample_shape": list(raw.shape),
})
