# Amendment 009 — parse locked calibration numbers

The resumed metric calculation stopped at its first metric because coefficient and threshold values read from the locked CSV remained strings. No partial summary tables were finalized. The saved predictions remain unchanged and are reused.

The evaluator now explicitly casts slope, intercept, raw q90, and mapped calibrated threshold to float when building the calibration lookup, then reruns metrics from the verified prediction files.
