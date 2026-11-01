# Amendment 003 — reproduction packaging corrections

This amendment addresses two reproduction-only defects reported in `outputs/seed_corrected_review_20261101/REVIEW.json`. No model, checkpoint, prediction, calibration, threshold, point metric, bootstrap draw, or scientific conclusion was changed.

## Bootstrap archive assembly

`evaluate_full.py` now keeps cohort-prefixed keys exactly once and calls `combine_cohort_draws()` / `write_full_draw_archive()`. Before writing, the helper requires the two expected cohorts, the exact 700-key cartesian product, no key collisions, and a finite numeric vector of shape `(1000,)` for every key. The separate `verify_archive_repack.py` used the existing 350-key Ningbo and 350-key HEEDB archives and tested the exact writer through a temporary file. It passed with 700 unique keys and zero numerical differences against the active full archive. It did not recompute bootstrap statistics.

## Frozen input validation

The original `control.py`, frozen execution protocol and protocol SHA sidecar are unchanged. The three updated root documentation files still carry their original hashes inside the frozen protocol. Exact historical snapshots were recovered from verified Git `HEAD` blobs, each snapshot SHA equals its frozen hash, and current documentation has a separate explicit hash record.

The new supported fresh-run wrapper uses `control_replay.py`. It verifies the original protocol and runner bindings, membership/split snapshots, all 65 frozen non-document inputs at their original hashes, each historical documentation snapshot at its original hash, and each current documentation file at its separately recorded hash. Its code and the supported `train_all_fresh.py` entrypoint are bound in `INPUT_VALIDATION_AMENDMENT_001.json` plus the SHA-256 sidecar. `train_all_fresh.py` still calls the original `control.train_one()` implementation; the original `control.py` hash remains the checkpoint's training-runner binding.

Validation was executed through the supported training command's own check mode:

```powershell
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\train_all_fresh.py --check-inputs
```

It passed: 65 strict scientific input checks, 3 original documentation snapshots, 3 current documentation hashes, and all original runner/protocol hashes. No training was invoked. Running the command without `--check-inputs` is for a fresh prepared copy only; the entrypoint refuses to overwrite any existing checkpoints.

The curated `REPRODUCIBILITY_MANIFEST.json` is rebuilt from an explicit list of 300 protocol, code, scientific output, snapshot, checkpoint and prediction files. It excludes `__pycache__`, temporary repack files, invalid attempt payload directories and review materials. Historical output hashes and the nine checkpoint-to-original-runner bindings remain intact.
