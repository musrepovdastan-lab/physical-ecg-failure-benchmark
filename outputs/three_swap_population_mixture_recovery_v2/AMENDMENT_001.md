# Amendment 001 — float32 engineering comparison

The initial independent-formula check used 1e-5 raw units and stopped at 6.1035e-5 on derived limb leads. This is float32 rounding when deriving the displayed leads from I/II at the source signal scale. The engineering check now uses 1e-4 raw units, still below the protocol's separately specified 2e-4 known-class inverse bound. It does not change any scientific input, inference method, classifier probability tolerance, predictions, threshold, or analysis. The initial failure is preserved in `FAILED_ATTEMPT_001.json`.
