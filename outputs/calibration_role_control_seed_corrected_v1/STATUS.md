# Package status

**COMPLETE_REVIEWED** вЂ” all 9 fits passed their predeclared gates; all 36 calibration/threshold locks are complete; all 96 held-out predictions, 700 paired bootstrap groups and old/new comparisons are complete. Initialization, RNG resume, historical integrity, detector-cache, prediction-binding and numerical audits passed.

No additional cohorts, seeds, models, thresholds, calibration choices, or training are part of this package. Raw inputs, checkpoints and individual predictions remain local. The original initialization states are unknown, so old/new differences are sensitivity findings rather than a pure causal estimate of the seed-order defect.

The follow-up reproduction review is also closed: the existing 700-draw archives repack exactly with 700 canonical keys; the supported `train_all_fresh.py --check-inputs` passes all 65 strict scientific hashes, the original runner/protocol hashes, and all three historical/current documentation bindings. No training or scientific recalculation was run during this follow-up.

See [FOLLOWUP_REPORT_RU.md](FOLLOWUP_REPORT_RU.md), [FINAL_REPORT_RU.md](FINAL_REPORT_RU.md), [TRAINING_REPORT_RU.md](TRAINING_REPORT_RU.md), [AMENDMENT_002.md](AMENDMENT_002.md), [AMENDMENT_003.md](AMENDMENT_003.md), and [SUPPORTED_REPRODUCTION.md](SUPPORTED_REPRODUCTION.md). The reviewed results are incorporated in [manuscript v3](../submission_evidence_v3/MANUSCRIPT_DRAFT.md). Documentation publication provenance is recorded in [AMENDMENT_004.md](AMENDMENT_004.md).
