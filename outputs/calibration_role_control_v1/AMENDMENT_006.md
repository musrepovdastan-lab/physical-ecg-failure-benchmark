# Amendment 006 — use the exact historical Platt objective

Before test/HEEDB scoring, the first clean A/B fit lock was reviewed against the frozen numerical specification. Its objective was numerically equivalent but used clipped sigmoid/log terms rather than the exact checked `fit_platt` implementation. The first lock and CSV are preserved as `CALIBRATION_AND_THRESHOLD_LOCK_PRECISE_FORMULA_REVIEW.*`.

The final lock is recomputed from the already saved, hash-bound clean A/B prediction files using the exact checked implementation and `numpy` linear q90. No model weights, memberships, labels, scores, thresholds-by-outcome, or scientific scope changed. The final fit/threshold lock is recorded in `CALIBRATION_AND_THRESHOLD_LOCK.json` before opening either test cohort.
