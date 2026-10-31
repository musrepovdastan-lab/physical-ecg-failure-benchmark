# External six-method HEEDB comparison

**Protocol fingerprint:** `0cc31881f9342b74ea7da8165220858aa95161de9e256fb068751eefd2a898aa`  
**Status:** Completed exploratory extension using the frozen 10,000-patient HEEDB I0006 independent-confirmation cohort. This cohort and its outcomes were previously accessed in mixed_fault_calibration_v2; this is not new independent confirmation.

## Results

TWA is primary; SB and STach are secondary. The full seed-level raw/calibrated results for all seven q values are in `MIXTURE_RESULTS_BY_SEED.csv`; three-seed means and SDs are in `MIXTURE_MEAN_SUMMARY.csv`. Values below are mean across seed-specific metrics, never an ensemble. q represents weighted copies of the same cohort, not measured fault prevalence.

| Method | q | Raw TWA AUPRC | Calibrated Brier | Calibrated sensitivity |
|---|---:|---:|---:|---:|
| chest_only_resnet | 0 | 0.537 ± 0.016 | 0.088 ± 0.002 | 0.634 ± 0.016 |
| chest_only_resnet | 0 | 0.537 ± 0.016 | 0.119 ± 0.004 | 0.634 ± 0.016 |
| combined | 0 | 0.592 ± 0.012 | 0.081 ± 0.001 | 0.661 ± 0.021 |
| combined | 0 | 0.592 ± 0.012 | 0.104 ± 0.004 | 0.661 ± 0.021 |
| huang_pipeline | 0 | 0.572 ± 0.013 | 0.083 ± 0.002 | 0.617 ± 0.034 |
| huang_pipeline | 0 | 0.572 ± 0.013 | 0.100 ± 0.005 | 0.617 ± 0.034 |
| huang_pipeline | 0 | 0.572 ± 0.013 | 0.100 ± 0.005 | 0.636 ± 0.038 |
| masking | 0 | 0.568 ± 0.012 | 0.083 ± 0.002 | 0.649 ± 0.013 |
| masking | 0 | 0.568 ± 0.012 | 0.112 ± 0.010 | 0.649 ± 0.013 |
| ordinary_resnet | 0 | 0.573 ± 0.012 | 0.083 ± 0.002 | 0.621 ± 0.038 |
| ordinary_resnet | 0 | 0.573 ± 0.012 | 0.099 ± 0.004 | 0.621 ± 0.038 |
| physics_aware_resnet | 0 | 0.576 ± 0.011 | 0.084 ± 0.002 | 0.656 ± 0.055 |
| physics_aware_resnet | 0 | 0.576 ± 0.011 | 0.115 ± 0.018 | 0.656 ± 0.055 |
| chest_only_resnet | 0.01 | 0.537 ± 0.016 | 0.088 ± 0.002 | 0.634 ± 0.016 |
| chest_only_resnet | 0.01 | 0.537 ± 0.016 | 0.119 ± 0.004 | 0.634 ± 0.016 |
| combined | 0.01 | 0.591 ± 0.012 | 0.081 ± 0.001 | 0.661 ± 0.021 |
| combined | 0.01 | 0.591 ± 0.012 | 0.104 ± 0.004 | 0.661 ± 0.021 |
| huang_pipeline | 0.01 | 0.572 ± 0.013 | 0.083 ± 0.002 | 0.617 ± 0.034 |
| huang_pipeline | 0.01 | 0.572 ± 0.013 | 0.100 ± 0.005 | 0.617 ± 0.034 |
| huang_pipeline | 0.01 | 0.572 ± 0.013 | 0.100 ± 0.005 | 0.636 ± 0.038 |
| masking | 0.01 | 0.545 ± 0.012 | 0.085 ± 0.002 | 0.649 ± 0.013 |
| masking | 0.01 | 0.545 ± 0.012 | 0.116 ± 0.010 | 0.649 ± 0.013 |
| ordinary_resnet | 0.01 | 0.550 ± 0.010 | 0.085 ± 0.002 | 0.622 ± 0.038 |
| ordinary_resnet | 0.01 | 0.550 ± 0.010 | 0.103 ± 0.004 | 0.622 ± 0.038 |
| physics_aware_resnet | 0.01 | 0.576 ± 0.011 | 0.084 ± 0.002 | 0.656 ± 0.055 |
| physics_aware_resnet | 0.01 | 0.576 ± 0.011 | 0.115 ± 0.018 | 0.656 ± 0.055 |
| chest_only_resnet | 0.1 | 0.537 ± 0.016 | 0.088 ± 0.002 | 0.634 ± 0.016 |
| chest_only_resnet | 0.1 | 0.537 ± 0.016 | 0.119 ± 0.004 | 0.634 ± 0.016 |
| combined | 0.1 | 0.587 ± 0.009 | 0.081 ± 0.001 | 0.662 ± 0.023 |
| combined | 0.1 | 0.587 ± 0.009 | 0.105 ± 0.003 | 0.662 ± 0.023 |
| huang_pipeline | 0.1 | 0.573 ± 0.013 | 0.083 ± 0.002 | 0.619 ± 0.034 |
| huang_pipeline | 0.1 | 0.573 ± 0.013 | 0.100 ± 0.005 | 0.619 ± 0.034 |
| huang_pipeline | 0.1 | 0.573 ± 0.013 | 0.100 ± 0.005 | 0.638 ± 0.039 |
| masking | 0.1 | 0.418 ± 0.045 | 0.105 ± 0.006 | 0.653 ± 0.009 |
| masking | 0.1 | 0.418 ± 0.045 | 0.157 ± 0.012 | 0.653 ± 0.009 |
| ordinary_resnet | 0.1 | 0.421 ± 0.006 | 0.102 ± 0.003 | 0.627 ± 0.040 |
| ordinary_resnet | 0.1 | 0.421 ± 0.006 | 0.139 ± 0.004 | 0.627 ± 0.040 |
| physics_aware_resnet | 0.1 | 0.575 ± 0.013 | 0.084 ± 0.002 | 0.658 ± 0.049 |
| physics_aware_resnet | 0.1 | 0.575 ± 0.013 | 0.116 ± 0.017 | 0.658 ± 0.049 |
| chest_only_resnet | 1 | 0.537 ± 0.016 | 0.088 ± 0.002 | 0.634 ± 0.016 |
| chest_only_resnet | 1 | 0.537 ± 0.016 | 0.119 ± 0.004 | 0.634 ± 0.016 |
| combined | 1 | 0.554 ± 0.026 | 0.087 ± 0.005 | 0.669 ± 0.051 |
| combined | 1 | 0.554 ± 0.026 | 0.120 ± 0.015 | 0.669 ± 0.051 |
| huang_pipeline | 1 | 0.581 ± 0.012 | 0.081 ± 0.002 | 0.638 ± 0.037 |
| huang_pipeline | 1 | 0.581 ± 0.012 | 0.098 ± 0.005 | 0.638 ± 0.037 |
| huang_pipeline | 1 | 0.581 ± 0.012 | 0.098 ± 0.005 | 0.655 ± 0.043 |
| masking | 1 | 0.159 ± 0.034 | 0.304 ± 0.057 | 0.693 ± 0.050 |
| masking | 1 | 0.159 ± 0.034 | 0.567 ± 0.079 | 0.693 ± 0.050 |
| ordinary_resnet | 1 | 0.165 ± 0.016 | 0.281 ± 0.034 | 0.685 ± 0.066 |
| ordinary_resnet | 1 | 0.165 ± 0.016 | 0.492 ± 0.022 | 0.685 ± 0.066 |
| physics_aware_resnet | 1 | 0.568 ± 0.031 | 0.085 ± 0.004 | 0.675 ± 0.009 |
| physics_aware_resnet | 1 | 0.568 ± 0.031 | 0.119 ± 0.012 | 0.675 ± 0.009 |

### Paired primary and secondary contrasts

Full-sample differences and bootstrap intervals are reported separately in `PAIRED_CONTRASTS_BOOTSTRAP.csv` for all q, requested metrics, and raw/calibrated modes. Intervals use 1,000 synchronized patient bootstrap draws.

| q | Mode | Contrast | Metric | Full-sample difference | 95% interval |
|---:|---|---|---|---:|---:|
| 0 | calibrated | combined_minus_chest_only_resnet | auprc | +0.055 | [+0.042, +0.068] |
| 0 | calibrated | combined_minus_chest_only_resnet | brier | -0.007 | [-0.008, -0.005] |
| 0 | calibrated | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.011, +0.043] |
| 0 | calibrated | combined_minus_huang_pipeline | auprc | +0.020 | [+0.009, +0.030] |
| 0 | calibrated | combined_minus_huang_pipeline | brier | -0.002 | [-0.003, -0.001] |
| 0 | calibrated | combined_minus_huang_pipeline | sensitivity | +0.045 | [+0.033, +0.056] |
| 0 | calibrated | combined_minus_physics_aware_resnet | auprc | +0.016 | [+0.008, +0.023] |
| 0 | calibrated | combined_minus_physics_aware_resnet | brier | -0.003 | [-0.004, -0.002] |
| 0 | calibrated | combined_minus_physics_aware_resnet | sensitivity | +0.005 | [-0.005, +0.016] |
| 0 | calibrated | masking_minus_ordinary_resnet | auprc | -0.005 | [-0.015, +0.005] |
| 0 | calibrated | masking_minus_ordinary_resnet | brier | +0.000 | [-0.001, +0.001] |
| 0 | calibrated | masking_minus_ordinary_resnet | sensitivity | +0.028 | [+0.015, +0.040] |
| 0 | raw | combined_minus_chest_only_resnet | auprc | +0.055 | [+0.042, +0.068] |
| 0 | raw | combined_minus_chest_only_resnet | brier | -0.016 | [-0.018, -0.013] |
| 0 | raw | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.011, +0.043] |
| 0 | raw | combined_minus_huang_pipeline | auprc | +0.020 | [+0.009, +0.030] |
| 0 | raw | combined_minus_huang_pipeline | brier | +0.004 | [+0.002, +0.005] |
| 0 | raw | combined_minus_huang_pipeline | sensitivity | +0.045 | [+0.033, +0.056] |
| 0 | raw | combined_minus_physics_aware_resnet | auprc | +0.016 | [+0.008, +0.023] |
| 0 | raw | combined_minus_physics_aware_resnet | brier | -0.011 | [-0.013, -0.010] |
| 0 | raw | combined_minus_physics_aware_resnet | sensitivity | +0.005 | [-0.005, +0.016] |
| 0 | raw | masking_minus_ordinary_resnet | auprc | -0.005 | [-0.015, +0.005] |
| 0 | raw | masking_minus_ordinary_resnet | brier | +0.012 | [+0.011, +0.014] |
| 0 | raw | masking_minus_ordinary_resnet | sensitivity | +0.028 | [+0.015, +0.040] |
| 0.01 | calibrated | combined_minus_chest_only_resnet | auprc | +0.055 | [+0.042, +0.067] |
| 0.01 | calibrated | combined_minus_chest_only_resnet | brier | -0.007 | [-0.008, -0.005] |
| 0.01 | calibrated | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.011, +0.043] |
| 0.01 | calibrated | combined_minus_huang_pipeline | auprc | +0.019 | [+0.009, +0.030] |
| 0.01 | calibrated | combined_minus_huang_pipeline | brier | -0.002 | [-0.003, -0.001] |
| 0.01 | calibrated | combined_minus_huang_pipeline | sensitivity | +0.045 | [+0.033, +0.056] |
| 0.01 | calibrated | combined_minus_physics_aware_resnet | auprc | +0.015 | [+0.008, +0.023] |
| 0.01 | calibrated | combined_minus_physics_aware_resnet | brier | -0.003 | [-0.004, -0.002] |
| 0.01 | calibrated | combined_minus_physics_aware_resnet | sensitivity | +0.005 | [-0.005, +0.015] |
| 0.01 | calibrated | masking_minus_ordinary_resnet | auprc | -0.006 | [-0.015, +0.003] |
| 0.01 | calibrated | masking_minus_ordinary_resnet | brier | +0.000 | [-0.000, +0.001] |
| 0.01 | calibrated | masking_minus_ordinary_resnet | sensitivity | +0.027 | [+0.015, +0.039] |
| 0.01 | raw | combined_minus_chest_only_resnet | auprc | +0.055 | [+0.042, +0.067] |
| 0.01 | raw | combined_minus_chest_only_resnet | brier | -0.015 | [-0.018, -0.013] |
| 0.01 | raw | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.011, +0.043] |
| 0.01 | raw | combined_minus_huang_pipeline | auprc | +0.019 | [+0.009, +0.030] |
| 0.01 | raw | combined_minus_huang_pipeline | brier | +0.004 | [+0.003, +0.006] |
| 0.01 | raw | combined_minus_huang_pipeline | sensitivity | +0.045 | [+0.033, +0.056] |
| 0.01 | raw | combined_minus_physics_aware_resnet | auprc | +0.015 | [+0.008, +0.023] |
| 0.01 | raw | combined_minus_physics_aware_resnet | brier | -0.011 | [-0.013, -0.010] |
| 0.01 | raw | combined_minus_physics_aware_resnet | sensitivity | +0.005 | [-0.005, +0.015] |
| 0.01 | raw | masking_minus_ordinary_resnet | auprc | -0.006 | [-0.015, +0.003] |
| 0.01 | raw | masking_minus_ordinary_resnet | brier | +0.013 | [+0.011, +0.015] |
| 0.01 | raw | masking_minus_ordinary_resnet | sensitivity | +0.027 | [+0.015, +0.039] |
| 0.1 | calibrated | combined_minus_chest_only_resnet | auprc | +0.051 | [+0.038, +0.063] |
| 0.1 | calibrated | combined_minus_chest_only_resnet | brier | -0.006 | [-0.008, -0.005] |
| 0.1 | calibrated | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.012, +0.043] |
| 0.1 | calibrated | combined_minus_huang_pipeline | auprc | +0.015 | [+0.004, +0.024] |
| 0.1 | calibrated | combined_minus_huang_pipeline | brier | -0.001 | [-0.002, -0.000] |
| 0.1 | calibrated | combined_minus_huang_pipeline | sensitivity | +0.043 | [+0.032, +0.054] |
| 0.1 | calibrated | combined_minus_physics_aware_resnet | auprc | +0.013 | [+0.005, +0.020] |
| 0.1 | calibrated | combined_minus_physics_aware_resnet | brier | -0.003 | [-0.003, -0.002] |
| 0.1 | calibrated | combined_minus_physics_aware_resnet | sensitivity | +0.004 | [-0.005, +0.014] |
| 0.1 | calibrated | masking_minus_ordinary_resnet | auprc | -0.003 | [-0.010, +0.003] |
| 0.1 | calibrated | masking_minus_ordinary_resnet | brier | +0.003 | [+0.002, +0.003] |
| 0.1 | calibrated | masking_minus_ordinary_resnet | sensitivity | +0.026 | [+0.015, +0.036] |
| 0.1 | raw | combined_minus_chest_only_resnet | auprc | +0.051 | [+0.038, +0.063] |
| 0.1 | raw | combined_minus_chest_only_resnet | brier | -0.014 | [-0.016, -0.012] |
| 0.1 | raw | combined_minus_chest_only_resnet | sensitivity | +0.028 | [+0.012, +0.043] |
| 0.1 | raw | combined_minus_huang_pipeline | auprc | +0.015 | [+0.004, +0.024] |
| 0.1 | raw | combined_minus_huang_pipeline | brier | +0.006 | [+0.004, +0.007] |
| 0.1 | raw | combined_minus_huang_pipeline | sensitivity | +0.043 | [+0.032, +0.054] |
| 0.1 | raw | combined_minus_physics_aware_resnet | auprc | +0.013 | [+0.005, +0.020] |
| 0.1 | raw | combined_minus_physics_aware_resnet | brier | -0.010 | [-0.011, -0.009] |
| 0.1 | raw | combined_minus_physics_aware_resnet | sensitivity | +0.004 | [-0.005, +0.014] |
| 0.1 | raw | masking_minus_ordinary_resnet | auprc | -0.003 | [-0.010, +0.003] |
| 0.1 | raw | masking_minus_ordinary_resnet | brier | +0.019 | [+0.017, +0.020] |
| 0.1 | raw | masking_minus_ordinary_resnet | sensitivity | +0.026 | [+0.015, +0.036] |
| 1 | calibrated | combined_minus_chest_only_resnet | auprc | +0.017 | [+0.004, +0.031] |
| 1 | calibrated | combined_minus_chest_only_resnet | brier | -0.001 | [-0.002, +0.001] |
| 1 | calibrated | combined_minus_chest_only_resnet | sensitivity | +0.035 | [+0.021, +0.050] |
| 1 | calibrated | combined_minus_huang_pipeline | auprc | -0.027 | [-0.042, -0.011] |
| 1 | calibrated | combined_minus_huang_pipeline | brier | +0.006 | [+0.004, +0.007] |
| 1 | calibrated | combined_minus_huang_pipeline | sensitivity | +0.032 | [+0.017, +0.046] |
| 1 | calibrated | combined_minus_physics_aware_resnet | auprc | -0.014 | [-0.024, -0.004] |
| 1 | calibrated | combined_minus_physics_aware_resnet | brier | +0.002 | [+0.001, +0.003] |
| 1 | calibrated | combined_minus_physics_aware_resnet | sensitivity | -0.006 | [-0.018, +0.006] |
| 1 | calibrated | masking_minus_ordinary_resnet | auprc | -0.006 | [-0.011, -0.001] |
| 1 | calibrated | masking_minus_ordinary_resnet | brier | +0.023 | [+0.021, +0.026] |
| 1 | calibrated | masking_minus_ordinary_resnet | sensitivity | +0.008 | [-0.006, +0.023] |
| 1 | raw | combined_minus_chest_only_resnet | auprc | +0.017 | [+0.004, +0.031] |
| 1 | raw | combined_minus_chest_only_resnet | brier | +0.001 | [-0.001, +0.003] |
| 1 | raw | combined_minus_chest_only_resnet | sensitivity | +0.035 | [+0.021, +0.050] |
| 1 | raw | combined_minus_huang_pipeline | auprc | -0.027 | [-0.042, -0.011] |
| 1 | raw | combined_minus_huang_pipeline | brier | +0.022 | [+0.020, +0.024] |
| 1 | raw | combined_minus_huang_pipeline | sensitivity | +0.032 | [+0.017, +0.046] |
| 1 | raw | combined_minus_physics_aware_resnet | auprc | -0.014 | [-0.024, -0.004] |
| 1 | raw | combined_minus_physics_aware_resnet | brier | +0.001 | [-0.001, +0.003] |
| 1 | raw | combined_minus_physics_aware_resnet | sensitivity | -0.006 | [-0.018, +0.006] |
| 1 | raw | masking_minus_ordinary_resnet | auprc | -0.006 | [-0.011, -0.001] |
| 1 | raw | masking_minus_ordinary_resnet | brier | +0.075 | [+0.072, +0.079] |
| 1 | raw | masking_minus_ordinary_resnet | sensitivity | +0.008 | [-0.006, +0.023] |

## Interpretation

- **Masking transfer:** masking does not show an external RA–LL protection advantage over ordinary training. Its TWA AUPRC difference is small at q=0 (−0.005; 95% CI −0.015 to +0.005) and negative at q=1 (−0.006; −0.011 to −0.001); at q=0.1 both methods are near 0.42. The expected missing-lead robustness does not transfer to this distinct reversal challenge.
- **Combined versus physics-aware:** combined is higher on clean TWA AUPRC (+0.016; +0.008 to +0.023 at q=0), but lower at full reversal share (−0.014; −0.024 to −0.004 at q=1). This resembles the Ningbo tradeoff direction at high reversal exposure; q is not a prevalence estimate. Combined saw half as many physical-reversal training examples as physics-aware, so this comparison does not isolate masking.
- **Combined versus Huang and chest-only:** at q=0 combined exceeds Huang by +0.020 AUPRC (+0.009 to +0.030) and chest-only by +0.055 (+0.042 to +0.068). At q=1 combined is below Huang by −0.027 (−0.042 to −0.011), while it remains above chest-only by +0.018 (+0.004 to +0.031). Read sensitivity and Brier alongside AUPRC; improvement in ranking is not evidence of more patients detected or clinical benefit.
- **Clean ECG and rare faults:** q=0.001 and 0.005 are available in the complete table and plots. Ordinary TWA AUPRC declines quickly as q grows; physics-aware remains comparatively stable. Masking also degrades at larger q. Combined retains higher clean ranking than physics-aware, while its Brier/sensitivity and reversal performance vary with q.
- **Calibration:** source-validation Platt maps are monotone. See `CALIBRATION_RANKING_AUDIT.csv` for clipping-created ties, ordering and exact operating-decision checks. Raw and calibrated metrics use corresponding validation-B thresholds; historical Huang threshold appears only as a separately labelled sensitivity mode.

## Methods and limitations

The six already-trained Ningbo masking/combined checkpoints were applied to the same 10,000 HEEDB records with frozen record order and physician-code proxy labels. The frozen 100 Hz conversion, checkpoint-specific training normalization, batch size 128, evaluation mode and no-gradient CUDA autocast path were used. Clean and physically correct RA–LL reversal predictions were saved per checkpoint, endpoint, patient and condition. Existing ordinary, physics-aware, chest-only and full Huang detector→predicted correction→diagnostic predictions were reused after bindings and the replay gate passed.

Mixture metrics weight each patient’s clean and reversed copies by 1−q and q; AUPRC/AUROC are weighted pooled metrics. Bootstrap draws use the same 20261017 NumPy stream after the 1,000 Ningbo draws from the prior analysis; each patient multiplicity is shared across conditions, methods and seeds. Seed metrics are averaged within each draw before differences. Full-sample estimates are separate from bootstrap means. Intervals are conditional on chosen checkpoints, fitted calibrators and fixed thresholds; they exclude training uncertainty. Crossing zero does not establish equivalence, and no multiplicity correction was prespecified.

HEEDB labels are proxy constructs rather than clinical adjudication. The HEEDB analyses share the Emory source, so they are not two independent centers. Findings are exploratory and do not establish clinical utility or a universal winner.

## Audit and artifacts

- Frozen protocol SHA-256: `0cc31881f9342b74ea7da8165220858aa95161de9e256fb068751eefd2a898aa`; six checkpoints and preprocessing bindings: `FROZEN_EXTENSION_PROTOCOL.json`.
- Replay: 18/18 rows pass; maximum absolute difference 0.0 at the unchanged 1e-4 tolerance. The initial legacy-helper diagnostic is retained separately.
- Inference: 12 condition files for six checkpoint/seed pairs; hashes and row bindings are in `PREDICTION_BINDINGS.json`.
- Calibration and bootstrap provenance: `CALIBRATION_PROVENANCE.json`, `HEEDB_BOOTSTRAP_INDICES.npy`, `PAIRED_BOOTSTRAP_REPLICATES.npz`.
- Reserved test was not evaluated; no new training, model selection, parameter search, or 792-mask evaluation was performed.
