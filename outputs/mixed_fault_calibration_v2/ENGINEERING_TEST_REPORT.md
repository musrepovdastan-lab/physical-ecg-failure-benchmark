# Engineering test report (v2)

**Status: PASS**

- Validation order and 3,488 labels match all frozen diagnostic-model validation artifacts; the unchanged A/B indices partition all 3,488 records into 1,744 A and 1,744 B records.
- Calibration optimizer, clipping, precision and bounds are the existing v1 implementation; coefficients have positive slopes and each fit uses A only. Thresholds use only negative B scores.
- Detector and all three diagnostic checkpoints replay the existing 64-record test subset with exact probability agreement at preregistered tolerances; diagnostic replay used the frozen 128-record inference batch size.
- All three pre-existing methods retain exactly the v1 point estimates; Huang now has complete full-pipeline historical-raw, validation-B raw and validation-B calibrated coverage.
- q=0 and q=1 endpoint reference checks passed for all Huang seeds, TWA, cohorts and threshold modes.
- Raw/calibrated Huang fixed operating-point decisions agree: maximum absolute difference across all five call-rate metrics and q values is 0.
- The frozen RNG stream reproduces all 882,000 saved v1 absolute metric draws exactly, then supplies paired Huang intervals. Every v2 interval contains 1,000 replicates.
- Validation detector output/labels are finite and ordered; test/HEEDB labels were not used for fitting. No diagnostic training or reserved-test assessment occurred.

See `ENGINEERING_TEST_REPORT.json` for machine-readable counts and hashes.
