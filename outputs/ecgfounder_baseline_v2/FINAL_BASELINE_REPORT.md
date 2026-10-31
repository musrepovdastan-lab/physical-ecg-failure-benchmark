# ECGFounder baseline v2 — exploratory paired comparison

This comparison uses the previously examined Chapman and Ningbo test sets. The v2 protocol was frozen after auditing test **inputs** for paths, hashes, headers, and missing samples, and before evaluating test **model outcomes**. Head settings and thresholds were selected from each dataset's existing training and validation splits. The ECGFounder backbone stayed frozen.

## Exact-label trained-head comparison

TWA is primary. Values are clean / RA–LL / clean-minus-reversal drop. Brackets give 95% paired patient/record bootstrap intervals for the drop (1,000 synchronized resamples). ResNet values are the mean of three saved seed **metrics**, not an ensemble.

| Dataset | Label | Metric | ECGFounder head clean / RA–LL / drop [95%] | Saved ResNet mean clean / RA–LL / drop [95%] | Paired drop difference [95%] |
|---|---|---|---:|---:|---:|
| Chapman | TWA | AUPRC | 0.626 / 0.316 / 0.311 [0.256, 0.365] | 0.653 / 0.328 / 0.325 [0.284, 0.364] | -0.015 [-0.069, 0.037] |
| Chapman | TWA | AUROC | 0.881 / 0.637 / 0.244 [0.214, 0.277] | 0.916 / 0.661 / 0.254 [0.230, 0.283] | -0.010 [-0.046, 0.022] |
| Chapman | TWA | SENSITIVITY | 0.727 / 0.772 / -0.045 [-0.097, 0.010] | 0.775 / 0.875 / -0.101 [-0.139, -0.064] | 0.056 [0.002, 0.112] |
| Chapman | SB | AUPRC | 1.000 / 1.000 / 0.000 [0.000, 0.001] | 0.999 / 0.994 / 0.006 [0.003, 0.008] | -0.005 [-0.008, -0.003] |
| Chapman | SB | AUROC | 1.000 / 1.000 / 0.000 [0.000, 0.000] | 1.000 / 0.997 / 0.003 [0.002, 0.004] | -0.003 [-0.004, -0.002] |
| Chapman | SB | SENSITIVITY | 1.000 / 1.000 / 0.000 [0.000, 0.000] | 1.000 / 1.000 / 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Chapman | STach | AUPRC | 0.995 / 0.902 / 0.093 [0.058, 0.129] | 0.986 / 0.756 / 0.230 [0.185, 0.271] | -0.137 [-0.182, -0.094] |
| Chapman | STach | AUROC | 0.996 / 0.975 / 0.021 [0.011, 0.031] | 0.997 / 0.950 / 0.046 [0.039, 0.055] | -0.025 [-0.037, -0.013] |
| Chapman | STach | SENSITIVITY | 0.997 / 0.917 / 0.080 [0.050, 0.110] | 0.995 / 0.894 / 0.101 [0.076, 0.127] | -0.021 [-0.057, 0.013] |
| Ningbo | TWA | AUPRC | 0.629 / 0.209 / 0.420 [0.391, 0.448] | 0.676 / 0.182 / 0.494 [0.467, 0.521] | -0.074 [-0.100, -0.048] |
| Ningbo | TWA | AUROC | 0.903 / 0.587 / 0.316 [0.299, 0.335] | 0.923 / 0.562 / 0.360 [0.343, 0.377] | -0.044 [-0.061, -0.025] |
| Ningbo | TWA | SENSITIVITY | 0.675 / 0.824 / -0.149 [-0.181, -0.117] | 0.733 / 0.763 / -0.031 [-0.060, -0.002] | -0.118 [-0.150, -0.083] |
| Ningbo | SB | AUPRC | 0.998 / 0.965 / 0.033 [0.024, 0.043] | 0.996 / 0.957 / 0.039 [0.033, 0.046] | -0.006 [-0.013, 0.001] |
| Ningbo | SB | AUROC | 0.998 / 0.986 / 0.013 [0.010, 0.015] | 0.998 / 0.977 / 0.021 [0.018, 0.023] | -0.008 [-0.011, -0.005] |
| Ningbo | SB | SENSITIVITY | 0.997 / 0.947 / 0.050 [0.042, 0.059] | 0.997 / 0.963 / 0.035 [0.030, 0.040] | 0.015 [0.006, 0.025] |
| Ningbo | STach | AUPRC | 0.991 / 0.971 / 0.020 [0.015, 0.027] | 0.986 / 0.702 / 0.284 [0.260, 0.307] | -0.263 [-0.285, -0.241] |
| Ningbo | STach | AUROC | 0.997 / 0.988 / 0.009 [0.005, 0.012] | 0.997 / 0.905 / 0.092 [0.085, 0.099] | -0.083 [-0.090, -0.076] |
| Ningbo | STach | SENSITIVITY | 0.996 / 0.980 / 0.016 [0.009, 0.024] | 0.995 / 0.842 / 0.152 [0.137, 0.167] | -0.136 [-0.151, -0.121] |

The primary TWA AUPRC loss appears in both datasets with positive patient/record bootstrap intervals. This supports transfer of the RA–LL vulnerability to this frozen ECGFounder feature plus linear-head pipeline in these exploratory sets.
TWA sensitivity can rise under reversal even while AUPRC and AUROC fall, because the fixed threshold also admits many more false positives. The paired drop difference compares losses under the two model pipelines; it does not rank overall clinical utility or establish SOTA superiority.

## Label mapping and direct zero-shot outputs

The official 150-class task list contains **SINUS BRADYCARDIA** (zero-based index 4) and **SINUS TACHYCARDIA** (index 6), which are name-level matches only. It has no exact TWA output. Its **NONSPECIFIC T WAVE ABNORMALITY** output is excluded from the exact-label and zero-shot comparisons. The table below is separate from the trained-head result; thresholds for sensitivity came from clean validation negatives.

| Dataset | Named output | Clean AUPRC | RA–LL AUPRC | Clean AUROC | RA–LL AUROC | Clean sensitivity | RA–LL sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|
| Chapman | SB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Chapman | STach | 0.968 | 0.946 | 0.994 | 0.990 | 0.990 | 0.990 |
| Ningbo | SB | 0.991 | 0.983 | 0.997 | 0.995 | 0.999 | 0.996 |
| Ningbo | STach | 0.940 | 0.953 | 0.992 | 0.992 | 0.996 | 0.992 |

## Changed predictions

Examples use the ECGFounder trained head and its validation-fixed threshold. Truth is the exact benchmark label; probabilities are clean → RA–LL.

| Dataset | Record | Label | Truth | Head probability | Threshold | Direction |
|---|---|---|---:|---:|---:|---|
| Chapman | `g5/JS04359` | TWA | 0 | 0.012 → 0.962 | 0.337 | negative_to_positive |
| Chapman | `g2/JS01632` | STach | 1 | 1.000 → 0.003 | 0.056 | positive_to_negative |
| Ningbo | `g10/JS20279` | TWA | 0 | 0.008 → 0.968 | 0.331 | negative_to_positive |
| Ningbo | `g10/JS20205` | STach | 1 | 0.995 → 0.004 | 0.026 | positive_to_negative |

## Missing samples, paths, and reproducibility

All 45,133 frozen split records resolved. WFDB NaN physical samples were replaced with zero before the electrical reversal or filtering, following the pinned authors' `dataset.py`; infinities and unexpected schema errors remained rejection conditions. No records were excluded.

| Dataset / split | Records | Records with NaNs | NaN samples | ID-folder differences |
|---|---:|---:|---:|---:|
| Chapman / train | 7163 | 0 | 0 | 0 |
| Chapman / validation | 1021 | 0 | 0 | 0 |
| Chapman / test | 2050 | 0 | 0 | 0 |
| Ningbo / train | 24430 | 59 | 1861 | 24 |
| Ningbo / validation | 3488 | 12 | 1003 | 6 |
| Ningbo / test | 6981 | 26 | 1227 | 4 |

Known adapter cases `g1/JS10767`, `g11/JS21646`, and `g13/JS23074` passed clean/reversal smoke inference. The last case uses a distinct header stem that references its canonical waveform. Electrical RA–LL equations and unchanged chest leads passed the v2 tests.

Official code commit: `04edac702b61c91face519774ddcc0cd712fef23`; 12-lead checkpoint SHA-256: `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`. Protocol SHA-256: `2f1e7600c26a6466c29e9bbc5ddbe6b826486cc05c84edaa3350433b3981ad81`. Input: 12 leads in I, II, III, aVR, aVL, aVF, V1–V6 order; 500 Hz, 10 seconds, WFDB physical mV. After NaN→0 and the physical RA–LL terminal permutation: official 50 Hz notch (Q=30), 0.67–40 Hz fourth-order Butterworth bandpass, 201-sample median baseline removal, global z-score, float32. The frozen 1024-dimensional features feed three L2 logistic heads trained on clean training records, with C selected from 0.01/0.1/1 by clean validation AUPRC and thresholds fixed at the 90th percentile of validation negative probabilities.

[Frozen protocol](../../configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json) · [reproducibility manifest](REPRODUCIBILITY_MANIFEST.json) · [full comparison](MAIN_COMPARISON.csv) · [compact comparison](COMPACT_RESULTS.csv) · [per-seed comparison](PER_SEED_COMPARISON.csv) · [paired differences](PAIRED_DROP_DIFFERENCES.csv) · [zero-shot outputs](ZERO_SHOT_NAME_MATCHED.csv) · [changed predictions](CHANGED_PREDICTIONS.csv) · [missing/path audit](MISSING_PATH_AUDIT.json).

Intervals resample canonical records as patient units under the project's one-record-per-patient assumption; the split tables have no independent patient identifier. The existing test sets were previously examined, and public pretraining overlap has not been independently ruled out. The honest next step is a new untouched, patient-linked external cohort with the same exact labels and electrical fault protocol, frozen before outcomes. This run does not develop a robustness method or claim superiority over published models.
