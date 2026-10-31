# Amendment 008 — bootstrap source alias

The evaluation runner completed and saved all 96 full prediction files for Ningbo test and HEEDB but stopped before metric aggregation because the bootstrap index reference used an undefined local alias `N`. This is recorded in `FAILED_ATTEMPT_007.json`. No summary or interval from the failed call was retained as complete.

The evaluator now takes the frozen Ningbo root from the shared control module. On rerun it verifies and reuses the saved predictions, then computes the same frozen metrics and synchronized bootstrap intervals. No inference or training is repeated.
