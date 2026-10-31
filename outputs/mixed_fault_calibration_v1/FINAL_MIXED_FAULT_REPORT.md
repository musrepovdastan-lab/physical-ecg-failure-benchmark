# Mostly-clean simulated RA–LL mixtures with source-validation calibration

## Scope and interpretation

This is an exploratory reanalysis of Ningbo and previously studied HEEDB I0006 records, not a new untouched confirmation. q is a hypothetical simulated-fault share, not an estimated clinical frequency. Each ECG contributes clean and exact RA–LL-reversed score/label copies weighted 1−q and q; metrics are calculated from the pooled weighted empirical mixture.

No diagnostic model was trained. The only newly fitted parameters are monotone Platt calibrators fit on clean Ningbo validation A. Thresholds use the 90th percentile of negative validation B scores. Checkpoints had historically been selected using full validation, so this A/B split does not make their selection independent. Calibrators and thresholds were held fixed on test/HEEDB. Calibration does not change the underlying ordering except for extreme clipping ties.

The Huang detector→predicted correction→clean ResNet pipeline has no complete frozen Ningbo-validation score set in the available artifacts. It is therefore shown raw only, using its existing corresponding baseline Ningbo validation threshold. No Huang calibration or standardized validation-B operating point is claimed. The already completed Ningbo corrected-score caches were added through [protocol amendment 001](FROZEN_PROTOCOL_AMENDMENT_001.json), with order/hash bindings.

HEEDB is external relative to the Chinese training sources, but these HEEDB cohorts come from the same Emory source and are not two independent external centers. HEEDB outcomes are physician-code proxy constructs; source calibration does not establish probability calibration for those proxies.

## Primary TWA ranking by q

Values are three-seed means from full-cohort weighted pooled mixtures. The bootstrap intervals below are 1,000 synchronized record/patient resamples, paired across clean/reversal states and methods; they are conditional on the selected checkpoints, fitted calibrators and fixed thresholds. Full intervals for all nine TWA metrics are in [BOOTSTRAP_METRICS_SUMMARY.csv](BOOTSTRAP_METRICS_SUMMARY.csv), with replicate arrays in `BOOTSTRAP_METRICS_REPLICATES.npz`.

| Source | q | Ordinary | Physics-aware | Chest-only | Huang |
|---|---:|---:|---:|---:|---:|
| Ningbo test (previously accessed) | 0 | 0.676 | 0.687 | 0.655 | 0.675 |
| Ningbo test (previously accessed) | 0.001 | 0.673 | 0.687 | 0.655 | 0.675 |
| Ningbo test (previously accessed) | 0.005 | 0.663 | 0.687 | 0.655 | 0.675 |
| Ningbo test (previously accessed) | 0.01 | 0.651 | 0.687 | 0.655 | 0.675 |
| Ningbo test (previously accessed) | 0.05 | 0.569 | 0.686 | 0.655 | 0.675 |
| Ningbo test (previously accessed) | 0.1 | 0.496 | 0.685 | 0.655 | 0.674 |
| Ningbo test (previously accessed) | 1 | 0.182 | 0.667 | 0.655 | 0.664 |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | 0.573 | 0.576 | 0.537 | 0.572 |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | 0.570 | 0.576 | 0.537 | 0.572 |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | 0.561 | 0.576 | 0.537 | 0.572 |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | 0.550 | 0.576 | 0.537 | 0.572 |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | 0.482 | 0.575 | 0.537 | 0.572 |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | 0.421 | 0.575 | 0.537 | 0.573 |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | 0.165 | 0.568 | 0.537 | 0.581 |

### Paired primary contrast: physics-aware minus each comparator

Percentile 95% intervals for TWA AUPRC; an interval crossing zero is unresolved and does not establish equality. Huang is raw-only.

| Source | q | Score mode | Contrast | Mean difference | 95% interval |
|---|---:|---|---|---:|---:|
| Ningbo test (previously accessed) | 0 | raw | physics_aware_minus_ordinary_resnet | 0.0107 | [-0.0015, 0.0225] |
| Ningbo test (previously accessed) | 0 | raw | physics_aware_minus_chest_only_resnet | 0.0313 | [0.0189, 0.0440] |
| Ningbo test (previously accessed) | 0 | raw | physics_aware_minus_huang_pipeline | 0.0114 | [-0.0010, 0.0232] |
| Ningbo test (previously accessed) | 0 | calibrated | physics_aware_minus_ordinary_resnet | 0.0107 | [-0.0015, 0.0225] |
| Ningbo test (previously accessed) | 0 | calibrated | physics_aware_minus_chest_only_resnet | 0.0313 | [0.0189, 0.0440] |
| Ningbo test (previously accessed) | 0.001 | raw | physics_aware_minus_ordinary_resnet | 0.0133 | [0.0009, 0.0250] |
| Ningbo test (previously accessed) | 0.001 | raw | physics_aware_minus_chest_only_resnet | 0.0313 | [0.0188, 0.0440] |
| Ningbo test (previously accessed) | 0.001 | raw | physics_aware_minus_huang_pipeline | 0.0114 | [-0.0010, 0.0232] |
| Ningbo test (previously accessed) | 0.001 | calibrated | physics_aware_minus_ordinary_resnet | 0.0133 | [0.0009, 0.0250] |
| Ningbo test (previously accessed) | 0.001 | calibrated | physics_aware_minus_chest_only_resnet | 0.0313 | [0.0188, 0.0440] |
| Ningbo test (previously accessed) | 0.005 | raw | physics_aware_minus_ordinary_resnet | 0.0236 | [0.0113, 0.0349] |
| Ningbo test (previously accessed) | 0.005 | raw | physics_aware_minus_chest_only_resnet | 0.0312 | [0.0188, 0.0439] |
| Ningbo test (previously accessed) | 0.005 | raw | physics_aware_minus_huang_pipeline | 0.0113 | [-0.0010, 0.0231] |
| Ningbo test (previously accessed) | 0.005 | calibrated | physics_aware_minus_ordinary_resnet | 0.0236 | [0.0113, 0.0349] |
| Ningbo test (previously accessed) | 0.005 | calibrated | physics_aware_minus_chest_only_resnet | 0.0312 | [0.0188, 0.0439] |
| Ningbo test (previously accessed) | 0.01 | raw | physics_aware_minus_ordinary_resnet | 0.0358 | [0.0236, 0.0469] |
| Ningbo test (previously accessed) | 0.01 | raw | physics_aware_minus_chest_only_resnet | 0.0311 | [0.0187, 0.0437] |
| Ningbo test (previously accessed) | 0.01 | raw | physics_aware_minus_huang_pipeline | 0.0113 | [-0.0011, 0.0231] |
| Ningbo test (previously accessed) | 0.01 | calibrated | physics_aware_minus_ordinary_resnet | 0.0358 | [0.0236, 0.0469] |
| Ningbo test (previously accessed) | 0.01 | calibrated | physics_aware_minus_chest_only_resnet | 0.0311 | [0.0187, 0.0437] |
| Ningbo test (previously accessed) | 0.05 | raw | physics_aware_minus_ordinary_resnet | 0.1165 | [0.1031, 0.1292] |
| Ningbo test (previously accessed) | 0.05 | raw | physics_aware_minus_chest_only_resnet | 0.0302 | [0.0178, 0.0429] |
| Ningbo test (previously accessed) | 0.05 | raw | physics_aware_minus_huang_pipeline | 0.0110 | [-0.0016, 0.0229] |
| Ningbo test (previously accessed) | 0.05 | calibrated | physics_aware_minus_ordinary_resnet | 0.1165 | [0.1031, 0.1292] |
| Ningbo test (previously accessed) | 0.05 | calibrated | physics_aware_minus_chest_only_resnet | 0.0302 | [0.0178, 0.0429] |
| Ningbo test (previously accessed) | 0.1 | raw | physics_aware_minus_ordinary_resnet | 0.1889 | [0.1737, 0.2040] |
| Ningbo test (previously accessed) | 0.1 | raw | physics_aware_minus_chest_only_resnet | 0.0292 | [0.0170, 0.0417] |
| Ningbo test (previously accessed) | 0.1 | raw | physics_aware_minus_huang_pipeline | 0.0105 | [-0.0020, 0.0225] |
| Ningbo test (previously accessed) | 0.1 | calibrated | physics_aware_minus_ordinary_resnet | 0.1889 | [0.1737, 0.2040] |
| Ningbo test (previously accessed) | 0.1 | calibrated | physics_aware_minus_chest_only_resnet | 0.0292 | [0.0170, 0.0417] |
| Ningbo test (previously accessed) | 1 | raw | physics_aware_minus_ordinary_resnet | 0.4841 | [0.4570, 0.5090] |
| Ningbo test (previously accessed) | 1 | raw | physics_aware_minus_chest_only_resnet | 0.0114 | [-0.0012, 0.0240] |
| Ningbo test (previously accessed) | 1 | raw | physics_aware_minus_huang_pipeline | 0.0033 | [-0.0131, 0.0190] |
| Ningbo test (previously accessed) | 1 | calibrated | physics_aware_minus_ordinary_resnet | 0.4841 | [0.4570, 0.5090] |
| Ningbo test (previously accessed) | 1 | calibrated | physics_aware_minus_chest_only_resnet | 0.0114 | [-0.0012, 0.0240] |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | raw | physics_aware_minus_ordinary_resnet | 0.0036 | [-0.0081, 0.0150] |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | raw | physics_aware_minus_chest_only_resnet | 0.0393 | [0.0279, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | raw | physics_aware_minus_huang_pipeline | 0.0042 | [-0.0074, 0.0160] |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | calibrated | physics_aware_minus_ordinary_resnet | 0.0036 | [-0.0081, 0.0150] |
| HEEDB I0006 (Emory proxy-label cohort) | 0 | calibrated | physics_aware_minus_chest_only_resnet | 0.0393 | [0.0279, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | raw | physics_aware_minus_ordinary_resnet | 0.0059 | [-0.0056, 0.0172] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | raw | physics_aware_minus_chest_only_resnet | 0.0392 | [0.0279, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | raw | physics_aware_minus_huang_pipeline | 0.0042 | [-0.0075, 0.0160] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | calibrated | physics_aware_minus_ordinary_resnet | 0.0059 | [-0.0056, 0.0172] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.001 | calibrated | physics_aware_minus_chest_only_resnet | 0.0392 | [0.0279, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | raw | physics_aware_minus_ordinary_resnet | 0.0149 | [0.0037, 0.0262] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | raw | physics_aware_minus_chest_only_resnet | 0.0392 | [0.0278, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | raw | physics_aware_minus_huang_pipeline | 0.0041 | [-0.0075, 0.0159] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | calibrated | physics_aware_minus_ordinary_resnet | 0.0149 | [0.0037, 0.0262] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.005 | calibrated | physics_aware_minus_chest_only_resnet | 0.0392 | [0.0278, 0.0513] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | raw | physics_aware_minus_ordinary_resnet | 0.0255 | [0.0143, 0.0369] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | raw | physics_aware_minus_chest_only_resnet | 0.0391 | [0.0278, 0.0512] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | raw | physics_aware_minus_huang_pipeline | 0.0040 | [-0.0077, 0.0158] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | calibrated | physics_aware_minus_ordinary_resnet | 0.0255 | [0.0143, 0.0369] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.01 | calibrated | physics_aware_minus_chest_only_resnet | 0.0391 | [0.0278, 0.0512] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | raw | physics_aware_minus_ordinary_resnet | 0.0937 | [0.0826, 0.1060] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | raw | physics_aware_minus_chest_only_resnet | 0.0385 | [0.0273, 0.0504] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | raw | physics_aware_minus_huang_pipeline | 0.0031 | [-0.0087, 0.0149] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | calibrated | physics_aware_minus_ordinary_resnet | 0.0937 | [0.0826, 0.1060] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.05 | calibrated | physics_aware_minus_chest_only_resnet | 0.0385 | [0.0273, 0.0504] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | raw | physics_aware_minus_ordinary_resnet | 0.1537 | [0.1413, 0.1678] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | raw | physics_aware_minus_chest_only_resnet | 0.0378 | [0.0267, 0.0496] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | raw | physics_aware_minus_huang_pipeline | 0.0020 | [-0.0100, 0.0138] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | calibrated | physics_aware_minus_ordinary_resnet | 0.1537 | [0.1413, 0.1678] |
| HEEDB I0006 (Emory proxy-label cohort) | 0.1 | calibrated | physics_aware_minus_chest_only_resnet | 0.0378 | [0.0267, 0.0496] |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | raw | physics_aware_minus_ordinary_resnet | 0.4031 | [0.3809, 0.4264] |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | raw | physics_aware_minus_chest_only_resnet | 0.0315 | [0.0202, 0.0434] |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | raw | physics_aware_minus_huang_pipeline | -0.0127 | [-0.0263, 0.0021] |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | calibrated | physics_aware_minus_ordinary_resnet | 0.4031 | [0.3809, 0.4264] |
| HEEDB I0006 (Emory proxy-label cohort) | 1 | calibrated | physics_aware_minus_chest_only_resnet | 0.0315 | [0.0202, 0.0434] |

## Clean, rare-fault and all-fault tradeoffs

At q=0, physics-aware TWA AUPRC is 0.687 on Ningbo and 0.576 on HEEDB, versus 0.676/0.573 for ordinary ResNet, 0.655/0.537 for chest-only, and 0.675/0.572 for Huang. Its paired intervals versus ordinary and Huang include zero at q=0; the comparison with chest-only favors physics-aware on both sources. As q rises, ordinary ResNet ranking falls sharply while physics-aware and chest-only remain steadier. At q=1, physics-aware remains well above ordinary, is close to chest-only on Ningbo, and is below Huang on HEEDB by 0.0127 AUPRC (95% paired interval −0.0263 to +0.0021), which does not establish a difference. These are scenario results, not clinical-frequency claims.

Source-validation calibration reduces TWA Brier/ECE on both sources at q=0: for physics-aware, Ningbo Brier/ECE changes from 0.1077/0.1247 raw to 0.0730/0.0115 calibrated; HEEDB changes from 0.1152/0.1120 to 0.0840/0.0227. On Ningbo, maximum raw/calibrated AUPRC and AUROC differences across arms/q are below 4e-16. On HEEDB, clipping created ties: the maximum differences are 0.000062 AUPRC and 0.000142 AUROC. These are measured and reported as clipping effects. The calibrated threshold is the same monotone transform of the raw validation-B threshold; all five operating-point measures matched exactly (maximum absolute difference 0). Raw historical-threshold results, raw validation-B threshold results, and calibrated equivalents are distinct rows in the result CSV. See [ranking/calibration audit](RANKING_CALIBRATION_AUDIT.json).

The chest-only model is exactly invariant to RA–LL in this input path because only V1–V6 enter the model; its score-based discrimination is constant across q. This makes it a useful simple comparator, while not implying superiority on every source or endpoint.

## Required engineering checks

Synthetic checks covered q=0 and q=1 identity, weighted AP/AUROC and all nine grouped-bootstrap metrics against scikit-learn reference calculations, ties and zero weights, paired bootstrap record multiplicity, Brier linear-mixture identity, raw/calibrated threshold equivalence, and monotonicity/clipping ties. See [ENGINEERING_TEST_REPORT.md](ENGINEERING_TEST_REPORT.md).

## Positioning correction

A separate correction note records the existing matched CPU request-to-response benchmark (10.59 ms physics-aware vs 31.51 ms Huang chain; 2.97×, specific to the measured implementation/hardware/workload), the HEEDB same-Emory-source limitation, and the full-PDF access caveat for Cao. See [POSITIONING_CORRECTION_001.md](../novelty_reassessment_v1/POSITIONING_CORRECTION_001.md). No new latency benchmark was run.

## Follow-up experiments

This packet does not make a new p_aug sweep mandatory. If the retained question is whether the training exposure rate itself has a dose-response relation, a future experiment should freeze a small p_aug grid, matched seeds, validation selection and test transfer before training. The current q-mixture answers inference-time scenario sensitivity for existing fixed models, not an optimal training exposure rate.

A focused Ningbo masking/combined-training replication remains scientifically useful because the prior corrected M2 Ningbo result passed its frozen external replication gate. Its concrete question should be whether independently retrained masking-only and combined arms reproduce M2's reversal-vs-clean/mask tradeoffs and whether they preserve the mostly-clean q curves on held-out data. It is not needed to complete this reanalysis and was not started.

## Files

- `SCORE_COVERAGE_AUDIT.csv`: model/source/seed coverage and explicit missing-data notes.
- `FROZEN_ANALYSIS_PROTOCOL.json`, `FROZEN_PROTOCOL_AMENDMENT_001.json`, `FROZEN_PROTOCOL_AMENDMENT_002.json`: frozen methods, cohorts, thresholds, hashes and versioned score/bootstrap coverage additions.
- `CALIBRATION_AND_PROVENANCE.json`, `SPLIT_PROVENANCE.json`: Platt fit coefficients, A/B split, threshold and source hashes.
- `RANKING_CALIBRATION_AUDIT.json`: measured raw/calibrated ranking deltas, clipping-created ties and operating-point invariance.
- `MIXTURE_RESULTS_BY_SEED.csv`, `PAIRED_CONTRASTS.csv`: primary/secondary full-sample metrics and paired contrasts.
- `BOOTSTRAP_SUMMARY.csv`, `BOOTSTRAP_REPLICATES.csv`: primary TWA AUPRC uncertainty draws and intervals.
- `BOOTSTRAP_METRICS_SUMMARY.csv`, `BOOTSTRAP_METRICS_REPLICATES.npz`, `BOOTSTRAP_METRICS_COMPLETE.json`: all nine TWA metrics across q, 1,000 paired draws and intervals.
- `RELIABILITY_BINS.csv`, `TWA_AUPRC_BY_Q.png`, `TWA_CALIBRATION_BY_Q.png`, `TWA_RELIABILITY_*.png`: reliability bins and q curves.
