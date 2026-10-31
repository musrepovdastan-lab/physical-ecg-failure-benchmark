# Corrected external six-method summary

This table is reconstructed directly from the independently checked extension CSVs. No inference, calibration fitting, or bootstrap was rerun. The report in the historical package is preserved unchanged. Cohort: 10,000 patient-disjoint records from HEEDB I0006, Emory; labels are physician-code proxies. Primary endpoint: TWA; SB and STach are secondary.

## TWA results by method and simulated reversal share

Each row is one method x q. AUPRC/AUROC and raw Brier/ECE come only from `score_mode=raw`; calibrated Brier/ECE and operating metrics come only from `score_mode=calibrated`, all at each method/seed/endpoint full-pipeline validation-B threshold. Values are three-seed mean +/- SD. `q_fault` is a scenario weight in pooled clean/reversed copies, not an observed prevalence.

| Method | q_fault | Raw AUPRC | Raw AUROC | Raw Brier | Raw ECE-15 | Calibrated Brier | Calibrated ECE-15 | Sensitivity | Specificity | Predicted-positive fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary ResNet | 0 | 0.573 +/- 0.012 | 0.903 +/- 0.007 | 0.099 +/- 0.004 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.621 +/- 0.038 | 0.917 +/- 0.008 | 0.161 +/- 0.012 |
| Masking | 0 | 0.568 +/- 0.012 | 0.901 +/- 0.006 | 0.112 +/- 0.010 | 0.103 +/- 0.014 | 0.083 +/- 0.002 | 0.018 +/- 0.004 | 0.649 +/- 0.013 | 0.907 +/- 0.008 | 0.174 +/- 0.009 |
| Physics-aware ResNet | 0 | 0.576 +/- 0.011 | 0.893 +/- 0.006 | 0.115 +/- 0.018 | 0.112 +/- 0.045 | 0.084 +/- 0.002 | 0.023 +/- 0.005 | 0.656 +/- 0.055 | 0.904 +/- 0.018 | 0.178 +/- 0.023 |
| Combined | 0 | 0.592 +/- 0.012 | 0.905 +/- 0.003 | 0.104 +/- 0.004 | 0.098 +/- 0.012 | 0.081 +/- 0.001 | 0.019 +/- 0.000 | 0.661 +/- 0.021 | 0.910 +/- 0.006 | 0.173 +/- 0.008 |
| Chest-only ResNet | 0 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0 | 0.572 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.617 +/- 0.034 | 0.919 +/- 0.007 | 0.159 +/- 0.011 |
| Ordinary ResNet | 0.001 | 0.570 +/- 0.012 | 0.903 +/- 0.007 | 0.100 +/- 0.004 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.621 +/- 0.038 | 0.917 +/- 0.008 | 0.161 +/- 0.012 |
| Masking | 0.001 | 0.565 +/- 0.011 | 0.900 +/- 0.006 | 0.112 +/- 0.010 | 0.103 +/- 0.014 | 0.083 +/- 0.002 | 0.019 +/- 0.004 | 0.649 +/- 0.013 | 0.907 +/- 0.008 | 0.174 +/- 0.009 |
| Physics-aware ResNet | 0.001 | 0.576 +/- 0.011 | 0.893 +/- 0.006 | 0.115 +/- 0.018 | 0.112 +/- 0.045 | 0.084 +/- 0.002 | 0.023 +/- 0.005 | 0.656 +/- 0.055 | 0.904 +/- 0.018 | 0.178 +/- 0.023 |
| Combined | 0.001 | 0.592 +/- 0.012 | 0.905 +/- 0.003 | 0.104 +/- 0.004 | 0.098 +/- 0.012 | 0.081 +/- 0.001 | 0.019 +/- 0.000 | 0.661 +/- 0.021 | 0.910 +/- 0.006 | 0.173 +/- 0.008 |
| Chest-only ResNet | 0.001 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0.001 | 0.572 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.617 +/- 0.034 | 0.919 +/- 0.007 | 0.159 +/- 0.011 |
| Ordinary ResNet | 0.005 | 0.561 +/- 0.011 | 0.901 +/- 0.007 | 0.101 +/- 0.004 | 0.079 +/- 0.008 | 0.084 +/- 0.002 | 0.026 +/- 0.006 | 0.621 +/- 0.038 | 0.915 +/- 0.008 | 0.163 +/- 0.013 |
| Masking | 0.005 | 0.556 +/- 0.010 | 0.898 +/- 0.006 | 0.114 +/- 0.010 | 0.105 +/- 0.014 | 0.084 +/- 0.002 | 0.019 +/- 0.003 | 0.649 +/- 0.013 | 0.904 +/- 0.008 | 0.176 +/- 0.008 |
| Physics-aware ResNet | 0.005 | 0.576 +/- 0.011 | 0.893 +/- 0.006 | 0.115 +/- 0.018 | 0.112 +/- 0.045 | 0.084 +/- 0.002 | 0.023 +/- 0.005 | 0.656 +/- 0.055 | 0.904 +/- 0.018 | 0.178 +/- 0.023 |
| Combined | 0.005 | 0.591 +/- 0.012 | 0.905 +/- 0.003 | 0.104 +/- 0.004 | 0.099 +/- 0.012 | 0.081 +/- 0.001 | 0.019 +/- 0.000 | 0.661 +/- 0.021 | 0.910 +/- 0.006 | 0.173 +/- 0.008 |
| Chest-only ResNet | 0.005 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0.005 | 0.572 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.617 +/- 0.034 | 0.919 +/- 0.007 | 0.159 +/- 0.011 |
| Ordinary ResNet | 0.01 | 0.550 +/- 0.010 | 0.899 +/- 0.007 | 0.103 +/- 0.004 | 0.082 +/- 0.008 | 0.085 +/- 0.002 | 0.027 +/- 0.005 | 0.622 +/- 0.038 | 0.912 +/- 0.009 | 0.166 +/- 0.013 |
| Masking | 0.01 | 0.545 +/- 0.012 | 0.896 +/- 0.006 | 0.116 +/- 0.010 | 0.108 +/- 0.014 | 0.085 +/- 0.002 | 0.020 +/- 0.003 | 0.649 +/- 0.013 | 0.901 +/- 0.007 | 0.179 +/- 0.008 |
| Physics-aware ResNet | 0.01 | 0.576 +/- 0.011 | 0.893 +/- 0.006 | 0.115 +/- 0.018 | 0.112 +/- 0.045 | 0.084 +/- 0.002 | 0.023 +/- 0.005 | 0.656 +/- 0.055 | 0.904 +/- 0.018 | 0.178 +/- 0.023 |
| Combined | 0.01 | 0.591 +/- 0.012 | 0.905 +/- 0.003 | 0.104 +/- 0.004 | 0.099 +/- 0.011 | 0.081 +/- 0.001 | 0.019 +/- 0.000 | 0.661 +/- 0.021 | 0.910 +/- 0.006 | 0.173 +/- 0.008 |
| Chest-only ResNet | 0.01 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0.01 | 0.572 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.617 +/- 0.034 | 0.919 +/- 0.007 | 0.159 +/- 0.011 |
| Ordinary ResNet | 0.05 | 0.482 +/- 0.006 | 0.882 +/- 0.007 | 0.119 +/- 0.004 | 0.100 +/- 0.009 | 0.093 +/- 0.002 | 0.034 +/- 0.005 | 0.624 +/- 0.039 | 0.890 +/- 0.012 | 0.185 +/- 0.016 |
| Masking | 0.05 | 0.476 +/- 0.034 | 0.878 +/- 0.008 | 0.135 +/- 0.010 | 0.128 +/- 0.014 | 0.094 +/- 0.003 | 0.032 +/- 0.004 | 0.651 +/- 0.011 | 0.876 +/- 0.003 | 0.201 +/- 0.003 |
| Physics-aware ResNet | 0.05 | 0.575 +/- 0.012 | 0.892 +/- 0.006 | 0.115 +/- 0.017 | 0.113 +/- 0.044 | 0.084 +/- 0.002 | 0.022 +/- 0.004 | 0.657 +/- 0.052 | 0.903 +/- 0.017 | 0.178 +/- 0.022 |
| Combined | 0.05 | 0.590 +/- 0.010 | 0.904 +/- 0.002 | 0.105 +/- 0.003 | 0.100 +/- 0.011 | 0.081 +/- 0.001 | 0.019 +/- 0.000 | 0.662 +/- 0.022 | 0.909 +/- 0.007 | 0.174 +/- 0.009 |
| Chest-only ResNet | 0.05 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0.05 | 0.572 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.025 +/- 0.006 | 0.618 +/- 0.034 | 0.918 +/- 0.007 | 0.160 +/- 0.011 |
| Ordinary ResNet | 0.1 | 0.421 +/- 0.006 | 0.862 +/- 0.007 | 0.139 +/- 0.004 | 0.124 +/- 0.009 | 0.102 +/- 0.003 | 0.046 +/- 0.006 | 0.627 +/- 0.040 | 0.863 +/- 0.017 | 0.208 +/- 0.020 |
| Masking | 0.1 | 0.418 +/- 0.045 | 0.856 +/- 0.012 | 0.157 +/- 0.012 | 0.153 +/- 0.016 | 0.105 +/- 0.006 | 0.048 +/- 0.007 | 0.653 +/- 0.009 | 0.844 +/- 0.005 | 0.228 +/- 0.003 |
| Physics-aware ResNet | 0.1 | 0.575 +/- 0.013 | 0.892 +/- 0.007 | 0.116 +/- 0.017 | 0.113 +/- 0.044 | 0.084 +/- 0.002 | 0.022 +/- 0.004 | 0.658 +/- 0.049 | 0.903 +/- 0.015 | 0.179 +/- 0.020 |
| Combined | 0.1 | 0.587 +/- 0.009 | 0.903 +/- 0.002 | 0.105 +/- 0.003 | 0.101 +/- 0.010 | 0.081 +/- 0.001 | 0.019 +/- 0.001 | 0.662 +/- 0.023 | 0.908 +/- 0.009 | 0.175 +/- 0.011 |
| Chest-only ResNet | 0.1 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 0.1 | 0.573 +/- 0.013 | 0.903 +/- 0.008 | 0.100 +/- 0.005 | 0.077 +/- 0.008 | 0.083 +/- 0.002 | 0.024 +/- 0.006 | 0.619 +/- 0.034 | 0.918 +/- 0.007 | 0.160 +/- 0.011 |
| Ordinary ResNet | 1 | 0.165 +/- 0.016 | 0.538 +/- 0.033 | 0.492 +/- 0.022 | 0.548 +/- 0.034 | 0.281 +/- 0.034 | 0.333 +/- 0.047 | 0.685 +/- 0.066 | 0.373 +/- 0.094 | 0.635 +/- 0.088 |
| Masking | 1 | 0.159 +/- 0.034 | 0.491 +/- 0.066 | 0.567 +/- 0.079 | 0.614 +/- 0.083 | 0.304 +/- 0.057 | 0.370 +/- 0.072 | 0.693 +/- 0.050 | 0.277 +/- 0.122 | 0.718 +/- 0.110 |
| Physics-aware ResNet | 1 | 0.568 +/- 0.031 | 0.890 +/- 0.012 | 0.119 +/- 0.012 | 0.125 +/- 0.031 | 0.085 +/- 0.004 | 0.019 +/- 0.007 | 0.675 +/- 0.009 | 0.895 +/- 0.016 | 0.188 +/- 0.015 |
| Combined | 1 | 0.554 +/- 0.026 | 0.888 +/- 0.012 | 0.120 +/- 0.015 | 0.127 +/- 0.029 | 0.087 +/- 0.005 | 0.024 +/- 0.011 | 0.669 +/- 0.051 | 0.887 +/- 0.035 | 0.194 +/- 0.037 |
| Chest-only ResNet | 1 | 0.537 +/- 0.016 | 0.876 +/- 0.010 | 0.119 +/- 0.004 | 0.117 +/- 0.007 | 0.088 +/- 0.002 | 0.022 +/- 0.005 | 0.634 +/- 0.016 | 0.902 +/- 0.006 | 0.176 +/- 0.007 |
| Huang pipeline | 1 | 0.581 +/- 0.012 | 0.907 +/- 0.006 | 0.098 +/- 0.005 | 0.077 +/- 0.009 | 0.081 +/- 0.002 | 0.023 +/- 0.006 | 0.638 +/- 0.037 | 0.916 +/- 0.008 | 0.164 +/- 0.012 |

## Historical Huang threshold sensitivity analysis

The historical operating point is shown separately and only for the Huang pipeline. Its scores remain raw; this threshold was selected in the earlier exploratory HEEDB analysis and is not the primary operating point. The main Huang operating point above uses the Huang detector -> predicted correction -> clean ResNet full-pipeline validation-B threshold, frozen before this extension. Do not mix the two threshold rows.

| q_fault | Sensitivity | Specificity | Predicted-positive fraction |
|---:|---:|---:|---:|
| 0 | 0.636 +/- 0.038 | 0.914 +/- 0.009 | 0.166 +/- 0.013 |
| 0.001 | 0.636 +/- 0.038 | 0.914 +/- 0.009 | 0.166 +/- 0.013 |
| 0.005 | 0.636 +/- 0.038 | 0.914 +/- 0.009 | 0.166 +/- 0.013 |
| 0.01 | 0.636 +/- 0.038 | 0.914 +/- 0.009 | 0.166 +/- 0.013 |
| 0.05 | 0.637 +/- 0.039 | 0.914 +/- 0.009 | 0.166 +/- 0.014 |
| 0.1 | 0.638 +/- 0.039 | 0.914 +/- 0.009 | 0.166 +/- 0.014 |
| 1 | 0.655 +/- 0.043 | 0.912 +/- 0.009 | 0.171 +/- 0.014 |

## Selected paired contrasts

AUPRC differences and patient-bootstrap 95% intervals are from `PAIRED_CONTRASTS_BOOTSTRAP.csv` (1,000 synchronized patient draws; conditional on frozen checkpoints/calibrators). Differences below are combined/left method minus comparator. Point estimates use full-sample seed means; intervals are not training-seed uncertainty intervals.

| q_fault | Contrast | Delta AUPRC | Conditional 95% interval |
|---:|---|---:|---:|
| 0 | masking_minus_ordinary_resnet | -0.005 | [-0.015, +0.005] |
| 0 | combined_minus_physics_aware_resnet | +0.016 | [+0.008, +0.023] |
| 0 | combined_minus_huang_pipeline | +0.020 | [+0.009, +0.030] |
| 0 | combined_minus_chest_only_resnet | +0.055 | [+0.042, +0.068] |
| 0.1 | masking_minus_ordinary_resnet | -0.003 | [-0.010, +0.003] |
| 0.1 | combined_minus_physics_aware_resnet | +0.013 | [+0.005, +0.020] |
| 0.1 | combined_minus_huang_pipeline | +0.015 | [+0.004, +0.024] |
| 0.1 | combined_minus_chest_only_resnet | +0.051 | [+0.038, +0.063] |
| 1 | masking_minus_ordinary_resnet | -0.006 | [-0.011, -0.001] |
| 1 | combined_minus_physics_aware_resnet | -0.014 | [-0.024, -0.004] |
| 1 | combined_minus_huang_pipeline | -0.027 | [-0.042, -0.011] |
| 1 | combined_minus_chest_only_resnet | +0.017 | [+0.004, +0.031] |

## Interpretation and provenance

- At q=0, combined TWA AUPRC is 0.592 versus 0.576 physics-aware and 0.572 Huang. At q=1 it is 0.554 versus 0.568 physics-aware and 0.581 Huang; the q=1 combined-minus-Huang interval excludes zero in the unfavorable direction. This is not a universal ranking.
- At q=0.01, combined sensitivity/specificity are 0.661/0.910, compared with 0.617/0.919 for Huang. Report both and the predicted-positive fraction; this is a source-selected operating point, not a clinical threshold.
- Monotone source-validation Platt calibration, paired with the corresponding transformed threshold, preserves classification decisions. The audit reports zero decision mismatches in 108 rows, though clipping can create ties. Better calibrated Brier/ECE therefore does not mean more cases were detected; AUPRC/AUROC are separately reported from raw scores.
- Raw ranking metrics are unchanged by the monotone calibration except possible tie behavior from clipping; the saved audit confirms no ordering violations and no observed operating-decision mismatch.
- Combined saw 25% physical-reversal training examples and physics-aware 50%; q_fault is the evaluation mixture weight and is distinct from training augmentation probability p_aug. The comparison does not isolate masking or estimate an optimal p_aug.
- Three seeds and patient bootstrap intervals condition on selected checkpoints/calibrators; they provide limited coverage of training uncertainty. The two HEEDB cohorts share Emory and are not independent centers. No real miswirings, held-out reversal compositions, clinical adjudication, or reserved test were assessed.
- Source rows: `../heedb_four_arm_extension_v1/MIXTURE_MEAN_SUMMARY.csv`, `../heedb_four_arm_extension_v1/PAIRED_CONTRASTS_BOOTSTRAP.csv`, `../heedb_four_arm_extension_v1/CALIBRATION_RANKING_AUDIT.csv`. Independent review: `../heedb_four_arm_extension_v1/REVIEW_NOTES_20261017.md`; frozen protocol SHA-256 `0cc31881f9342b74ea7da8165220858aa95161de9e256fb068751eefd2a898aa`.
