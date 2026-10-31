# M2 finalizer-lock incident

## What happened

The trainer/evaluator completed all three frozen 794-condition seed evaluations
(2,382 total conditions), then invoked `mitigation_m2_finalize_v1.py run` before
its own `mitigation_m2_ningbo_v1.lock` was released.  The finalizer correctly
refused the overlap with `RuntimeError: M2 trainer/evaluator remains active`.

## Scientific impact

This was orchestration-only.  No training, validation, checkpoint selection,
test inference, masks, thresholds, endpoints, bootstrap indices, or predictions
were rerun or changed.  The original runner stdout/stderr logs remain preserved.
All completed per-seed evaluation artifacts are inputs to the subsequent
cached-prediction-only finalizer.

## Correction

The trainer now releases its lock in `finally` before calling a dedicated
`launch_finalizer_after_trainer_release` hand-off.  The hand-off refuses to
start when that lock exists, and `tests/test_m2_orchestration.py` verifies both
the rejection-before-release and launch-after-release paths.
