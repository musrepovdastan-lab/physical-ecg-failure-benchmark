# Four-method calibration completion and paired uncertainty (v2)

## Scope

This is a versioned completion of the previously accessed Ningbo and HEEDB analyses. It is not a new untouched validation. No diagnostic network or Huang detector was trained; no weights, labels, splits, q values, or training data were changed. The v1 directory is retained unchanged. The only new model outputs are frozen Huang full-pipeline predictions for the existing 3,488 Ningbo validation records.

The published detector was run on clean validation ECGs and its argmax class controlled correction for every record, including records predicted clean. The same clean-trained ResNet checkpoints, preprocessing and normalization were then used for all three seeds. The test adapter replay matched stored detector and diagnostic probabilities exactly on the preregistered 64-record subset, using the historical diagnostic batch size 128. The initial 64-sized partial-batch replay differed by 0.000244; source inspection identified batch shape as the cause, the 0.0001 tolerance was left unchanged, and amendment 001 records the repeat. The detector marked 48/3,488 clean validation records for correction (1.38%); this is descriptive and does not change the always-predict-correction policy.

Huang calibrators were fitted on clean Ningbo validation A only, using v1's unchanged Platt implementation and ordinary unweighted log loss. Raw thresholds are the 90th percentile of negative full-pipeline scores on validation B; calibrated thresholds are those same thresholds transformed through the fit. Test and HEEDB labels were not used for fitting. Historical Huang thresholds remain visible as a separate mode. The legacy v1 contrast, where physics-aware uses its validation-B raw threshold while Huang uses the historical baseline threshold, is retained with an explicit label.

All model selection/checkpoint selection preceded this A/B split and historically used full validation, so the split does not make checkpoint selection independent. Calibration is fixed in all test/HEEDB and bootstrap calculations. HEEDB outcomes are physician-code proxies, and both HEEDB analyses share the Emory source. Results are exploratory, q is a hypothetical mixture share rather than a clinical prevalence estimate, and intervals condition on selected checkpoints/calibrators.

## TWA ranking over the full q grid

Entries are full-sample weighted pooled AUPRC, averaged across seed-specific metrics (prediction vectors are not ensembled). Huang values use its raw validation-B threshold; the score mapping itself is raw. The full seed-level table covers all endpoints and operating threshold modes in `MIXTURE_RESULTS_BY_SEED.csv`.

### Ningbo

| q | Ordinary | Physics-aware | Chest-only | Huang raw validation-B |
|---:|---:|---:|---:|---:|
| 0 | 0.676 | 0.687 | 0.655 | 0.675 |
| 0.001 | 0.673 | 0.687 | 0.655 | 0.675 |
| 0.005 | 0.663 | 0.687 | 0.655 | 0.675 |
| 0.01 | 0.651 | 0.687 | 0.655 | 0.675 |
| 0.05 | 0.569 | 0.686 | 0.655 | 0.675 |
| 0.1 | 0.496 | 0.685 | 0.655 | 0.674 |
| 1 | 0.182 | 0.667 | 0.655 | 0.664 |

### HEEDB_I0006_independent_confirmation

| q | Ordinary | Physics-aware | Chest-only | Huang raw validation-B |
|---:|---:|---:|---:|---:|
| 0 | 0.573 | 0.576 | 0.537 | 0.572 |
| 0.001 | 0.570 | 0.576 | 0.537 | 0.572 |
| 0.005 | 0.561 | 0.576 | 0.537 | 0.572 |
| 0.01 | 0.550 | 0.576 | 0.537 | 0.572 |
| 0.05 | 0.482 | 0.575 | 0.537 | 0.572 |
| 0.1 | 0.421 | 0.575 | 0.537 | 0.573 |
| 1 | 0.165 | 0.568 | 0.537 | 0.581 |

## Physics-aware minus Huang TWA AUPRC

`Full-sample difference` is the central estimate from the entire cohort; the bootstrap mean is stored separately in `PAIRED_CONTRASTS.csv`. Intervals use the same 1,000 synchronized draws and paired clean/reversal cluster multiplicities.

| Source | q | Threshold mode | Full-sample difference | 95% percentile interval |
|---|---:|---|---:|---:|
| Ningbo | 0 | raw_historical_threshold | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0 | raw_validation_B_90pct | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0 | calibrated_validation_B_90pct | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0.001 | raw_historical_threshold | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0.001 | raw_validation_B_90pct | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0.001 | calibrated_validation_B_90pct | 0.0115 | [-0.0010, 0.0232] |
| Ningbo | 0.01 | raw_historical_threshold | 0.0114 | [-0.0011, 0.0231] |
| Ningbo | 0.01 | raw_validation_B_90pct | 0.0114 | [-0.0011, 0.0231] |
| Ningbo | 0.01 | calibrated_validation_B_90pct | 0.0114 | [-0.0011, 0.0231] |
| Ningbo | 0.05 | raw_historical_threshold | 0.0110 | [-0.0016, 0.0229] |
| Ningbo | 0.05 | raw_validation_B_90pct | 0.0110 | [-0.0016, 0.0229] |
| Ningbo | 0.05 | calibrated_validation_B_90pct | 0.0110 | [-0.0016, 0.0229] |
| Ningbo | 1 | raw_historical_threshold | 0.0029 | [-0.0131, 0.0190] |
| Ningbo | 1 | raw_validation_B_90pct | 0.0029 | [-0.0131, 0.0190] |
| Ningbo | 1 | calibrated_validation_B_90pct | 0.0029 | [-0.0131, 0.0190] |
| HEEDB_I0006_independent_confirmation | 0 | raw_historical_threshold | 0.0040 | [-0.0074, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0 | raw_validation_B_90pct | 0.0040 | [-0.0074, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0 | calibrated_validation_B_90pct | 0.0040 | [-0.0074, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0 | raw_legacy_v1_Huang_historical | 0.0040 | [-0.0074, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0.001 | raw_historical_threshold | 0.0039 | [-0.0075, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0.001 | raw_validation_B_90pct | 0.0039 | [-0.0075, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0.001 | calibrated_validation_B_90pct | 0.0039 | [-0.0075, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0.001 | raw_legacy_v1_Huang_historical | 0.0039 | [-0.0075, 0.0160] |
| HEEDB_I0006_independent_confirmation | 0.01 | raw_historical_threshold | 0.0037 | [-0.0077, 0.0158] |
| HEEDB_I0006_independent_confirmation | 0.01 | raw_validation_B_90pct | 0.0037 | [-0.0077, 0.0158] |
| HEEDB_I0006_independent_confirmation | 0.01 | calibrated_validation_B_90pct | 0.0037 | [-0.0077, 0.0158] |
| HEEDB_I0006_independent_confirmation | 0.01 | raw_legacy_v1_Huang_historical | 0.0037 | [-0.0077, 0.0158] |
| HEEDB_I0006_independent_confirmation | 0.05 | raw_historical_threshold | 0.0028 | [-0.0087, 0.0149] |
| HEEDB_I0006_independent_confirmation | 0.05 | raw_validation_B_90pct | 0.0028 | [-0.0087, 0.0149] |
| HEEDB_I0006_independent_confirmation | 0.05 | calibrated_validation_B_90pct | 0.0028 | [-0.0087, 0.0149] |
| HEEDB_I0006_independent_confirmation | 0.05 | raw_legacy_v1_Huang_historical | 0.0028 | [-0.0087, 0.0149] |
| HEEDB_I0006_independent_confirmation | 1 | raw_historical_threshold | -0.0128 | [-0.0263, 0.0021] |
| HEEDB_I0006_independent_confirmation | 1 | raw_validation_B_90pct | -0.0128 | [-0.0263, 0.0021] |
| HEEDB_I0006_independent_confirmation | 1 | calibrated_validation_B_90pct | -0.0128 | [-0.0263, 0.0021] |
| HEEDB_I0006_independent_confirmation | 1 | raw_legacy_v1_Huang_historical | -0.0128 | [-0.0263, 0.0021] |

Intervals crossing zero leave the relative difference unresolved; they do not establish equivalence. Across the tested q values, these intervals do not establish a physics-aware AUPRC advantage over Huang. This removes a basis for claiming superiority over Huang in this analysis. At higher q the point estimates generally remain close, with uncertainty spanning zero.

## Source-validation calibration at q=0

| Source | Method | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---:|---:|---:|---:|
| Ningbo | Ordinary | 0.0990 | 0.0737 | 0.0978 | 0.0117 |
| Ningbo | Physics-aware | 0.1077 | 0.0730 | 0.1247 | 0.0115 |
| Ningbo | Chest-only | 0.1131 | 0.0772 | 0.1338 | 0.0140 |
| Ningbo | Huang | 0.0993 | 0.0739 | 0.0984 | 0.0118 |
| HEEDB_I0006_independent_confirmation | Ordinary | 0.0994 | 0.0826 | 0.0768 | 0.0253 |
| HEEDB_I0006_independent_confirmation | Physics-aware | 0.1152 | 0.0840 | 0.1120 | 0.0227 |
| HEEDB_I0006_independent_confirmation | Chest-only | 0.1193 | 0.0877 | 0.1172 | 0.0215 |
| HEEDB_I0006_independent_confirmation | Huang | 0.0997 | 0.0827 | 0.0772 | 0.0246 |

The monotone mapping leaves AUPRC/AUROC essentially unchanged except for measured clipping-created ties. It improves Brier/ECE on Ningbo and the measured HEEDB proxy labels. The latter is not evidence that source calibration is valid probability calibration for HEEDB's proxy outcome construct.

Raw validation-B versus calibrated validation-B use exactly one operating point: sensitivity, specificity, predicted-positive fraction, false positives and false negatives are identical at every q. Comparisons to historical thresholds are separate because the Huang historical threshold differs from its full-pipeline validation-B threshold; changes in those call rates arise from the operating-point standardization, not monotone calibration.

## Huang validation calibration coefficients

| Seed | Endpoint | a | b | Raw B threshold | Calibrated B threshold |
|---:|---|---:|---:|---:|---:|
| 104729 | TWA | 0.71028 | -1.12111 | 0.672852 | 0.352306 |
| 104729 | SB | 1.04715 | -1.34487 | 0.00760651 | 0.00158486 |
| 104729 | STach | 0.806741 | -2.17463 | 0.0440399 | 0.00940116 |
| 130363 | TWA | 0.752171 | -1.41682 | 0.747314 | 0.354066 |
| 130363 | SB | 1.12719 | 0.610838 | 0.0026722 | 0.00231839 |
| 130363 | STach | 1.07518 | -2.30675 | 0.106177 | 0.00997829 |
| 155921 | TWA | 0.888557 | -1.09108 | 0.618652 | 0.340476 |
| 155921 | SB | 1.28673 | -1.74981 | 0.0440063 | 0.0032988 |
| 155921 | STach | 1.22658 | -0.629428 | 0.0402527 | 0.0107768 |

## Conclusions

1. **Relative ranking:** Physics-aware retains the strongest TWA point estimates under increasingly frequent simulated RA–LL faults. Ordinary deteriorates substantially as q rises; chest-only is nearly invariant because the simulated reversal changes limb leads. Paired intervals comparing physics-aware with Huang cross zero, so the analysis does not resolve a superiority claim over Huang.
2. **Calibration:** Huang's full-pipeline validation scores fill the previous calibration gap. Source calibration improves probability-error metrics, while the strictly monotone mapping preserves ranking to within measured clipping/tie effects. It leaves fixed operating decisions unchanged.
3. **Rare faults:** At q=0 and q=0.001, physics-aware and Huang/ordinary rankings remain close; confidence intervals do not resolve a physics-aware advantage over Huang. With larger q, ordinary worsens while physics-aware remains relatively stable.
4. **Clean ECG tradeoff:** Huang's false-correction rate is 1.38% on this clean validation cohort under the mandated detector-argmax correction policy. This is an observed validation behavior, not a clinical harm estimate. Full clean-cohort metrics are in the seed-level CSV.
5. **HEEDB:** The relative ranking is broadly consistent with the Ningbo pattern, but endpoint labels are proxies and the source is one Emory system. It is not multi-center clinical confirmation.
6. **Next experiment:** No `p_aug` value is selected by this inference-time mixture analysis. A focused masking/combined Ningbo replication remains useful to ask whether independently retrained arms reproduce M2's reversal-versus-masking tradeoff and preserve the q response. No such training was started.

## Reproducibility and engineering checks

- `FROZEN_COMPLETION_PROTOCOL.json` and amendments 001–003 bind cohorts, record order, inputs, replay tolerances, A/B split, calibration and bootstrap rules.
- The exact v1 RNG stream was replayed in cohort order. All 882,000 v1 absolute metric draws matched with maximum absolute delta 0; existing v1 intervals were reused, the missing Ningbo Huang paired interval was filled, and new bootstrap metrics were computed only for Huang's B-threshold and calibrated modes plus the new matched historical operating-point comparisons.
- All 2,709 bootstrap summary groups contain 1,000 draws. Point estimates, bootstrap means and intervals are separately labeled.
- See `ENGINEERING_TEST_REPORT.md`, `HUANG_VALIDATION_INFERENCE_AUDIT.json`, `BOOTSTRAP_COMPLETION_AUDIT.json`, and `REPRODUCIBILITY_MANIFEST.json`.

## Main files

- `HUANG_VALIDATION_INFERENCE_AUDIT.md` and `.json`: exact full-pipeline inference, replay, record/label bindings and clean false-correction count.
- `HUANG_CALIBRATION_AND_PROVENANCE.json` and `CALIBRATION_AND_PROVENANCE.json`: fit coefficients and unchanged existing calibrators.
- `MIXTURE_RESULTS_BY_SEED.csv`, `PAIRED_CONTRASTS.csv`, `BOOTSTRAP_METRICS_SUMMARY.csv`, `BOOTSTRAP_METRICS_REPLICATES.npz`.
- `SCORE_COVERAGE_AUDIT.csv`, `RELIABILITY_BINS.csv`, `TWA_*.png`, `ENGINEERING_TEST_REPORT.md`, `REPRODUCIBILITY_MANIFEST.json`.

Reserved_test was not evaluated. Stop here for review; no subsequent training or experiment was launched.
