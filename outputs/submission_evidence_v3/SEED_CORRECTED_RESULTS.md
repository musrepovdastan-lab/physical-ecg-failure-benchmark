# Corrected initialization series: manuscript tables

These tables summarize completed fits; generating them does not run models or statistics. TWA is primary. AP means and SD are calculated over three seed-specific metrics, not an ensemble. q is a constructed evaluation mixture weight, not observed fault prevalence.

## Table 1. Raw TWA AP in the equal-three-swap scenario

| Cohort | Pipeline | q=0 | q=0.1 | q=1 |
|---|---|---:|---:|---:|
| Ningbo_test | Ordinary | 0.6766 ± 0.0036 | 0.5505 ± 0.0245 | 0.2239 ± 0.0165 |
| Ningbo_test | Physics-aware p050 | 0.6911 ± 0.0015 | 0.6883 ± 0.0009 | 0.6642 ± 0.0036 |
| Ningbo_test | Combined | 0.6731 ± 0.0074 | 0.6693 ± 0.0078 | 0.6384 ± 0.0140 |
| Ningbo_test | Huang pipeline | 0.6763 ± 0.0035 | 0.6757 ± 0.0036 | 0.6701 ± 0.0040 |
| HEEDB | Ordinary | 0.5855 ± 0.0114 | 0.4904 ± 0.0194 | 0.2190 ± 0.0104 |
| HEEDB | Physics-aware p050 | 0.5951 ± 0.0166 | 0.5922 ± 0.0156 | 0.5723 ± 0.0100 |
| HEEDB | Combined | 0.5912 ± 0.0084 | 0.5855 ± 0.0092 | 0.5436 ± 0.0304 |
| HEEDB | Huang pipeline | 0.5849 ± 0.0128 | 0.5857 ± 0.0131 | 0.5941 ± 0.0160 |

## Table 2. Differences from ordinary at equal-three q=0.1

Entries are full-cohort mean-seed differences [unadjusted 95% paired percentile interval]. Higher AP/sensitivity is favorable; lower Brier/false positives is favorable. False positives are per 1,000 evaluated ECGs, not per 1,000 negatives.

| Cohort | Pipeline | AP raw | Brier calibrated | Sensitivity raw | FP per 1,000 raw |
|---|---|---:|---:|---:|---:|
| Ningbo_test | Physics-aware p050 | +0.1378 [+0.1243, +0.1503] | -0.0141 [-0.0152, -0.0129] | +0.0276 [+0.0133, +0.0413] | -29.1 [-31.9, -26.3] |
| Ningbo_test | Combined | +0.1187 [+0.1066, +0.1313] | -0.0122 [-0.0132, -0.0111] | +0.0064 [-0.0061, +0.0194] | -29.9 [-32.5, -27.0] |
| HEEDB | Physics-aware p050 | +0.1018 [+0.0905, +0.1130] | -0.0106 [-0.0118, -0.0094] | +0.0407 [+0.0292, +0.0521] | -12.2 [-14.7, -9.7] |
| HEEDB | Combined | +0.0951 [+0.0839, +0.1058] | -0.0108 [-0.0120, -0.0098] | +0.0341 [+0.0229, +0.0454] | -14.7 [-17.0, -12.4] |

## Table 3. AP comparisons against Huang

Positive values favor the named training method over the complete Huang pipeline.

| Cohort | Pipeline minus Huang | equal-three q=0.1 | equal-three q=1 |
|---|---|---:|---:|
| Ningbo_test | Physics-aware p050 | +0.0126 [-0.0002, +0.0248] | -0.0060 [-0.0227, +0.0088] |
| Ningbo_test | Combined | -0.0064 [-0.0176, +0.0043] | -0.0317 [-0.0463, -0.0172] |
| HEEDB | Physics-aware p050 | +0.0065 [-0.0056, +0.0174] | -0.0218 [-0.0362, -0.0100] |
| HEEDB | Combined | -0.0002 [-0.0123, +0.0105] | -0.0505 [-0.0657, -0.0374] |

At q=0.1, neither p050 nor combined has an AP interval excluding zero against Huang in either cohort. The historical positive comparisons must not be carried forward as evidence of the corrected series. At q=1 Huang is the point leader in both cohorts; its difference from p050 on Ningbo includes zero. Crossing zero does not establish equivalence.

## Table 4. Point leaders before and after corrected initialization

| Cohort | Scenario | q | Original leader | Corrected leader |
|---|---|---:|---|---|
| HEEDB | RA_LL_only | 0 | Physics-aware p050 | Physics-aware p050 |
| HEEDB | RA_LL_only | 0.1 | Physics-aware p050 | Physics-aware p050 |
| HEEDB | RA_LL_only | 1 | Huang pipeline | Huang pipeline |
| HEEDB | equal_three_swap | 0 | Physics-aware p050 | Physics-aware p050 |
| HEEDB | equal_three_swap | 0.1 | Physics-aware p050 | Physics-aware p050 |
| HEEDB | equal_three_swap | 1 | Huang pipeline | Huang pipeline |
| Ningbo_test | RA_LL_only | 0 | Combined | Physics-aware p050 |
| Ningbo_test | RA_LL_only | 0.1 | Combined | Physics-aware p050 |
| Ningbo_test | RA_LL_only | 1 | Huang pipeline | Huang pipeline |
| Ningbo_test | equal_three_swap | 0 | Combined | Physics-aware p050 |
| Ningbo_test | equal_three_swap | 0.1 | Combined | Physics-aware p050 |
| Ningbo_test | equal_three_swap | 1 | Huang pipeline | Huang pipeline |

## Scope and provenance

The complete CSVs contain both RA–LL-only and equal-three-swap scenarios at q = 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1. The 700 contrast groups use 1,000 synchronized bootstrap replicates each, with records resampled for Ningbo and patients for HEEDB. Intervals condition on the fitted models/calibrators and are not multiplicity-adjusted. The original calibration-role series and the corrected series use the same data roles and scientific recipes; original initialization states are unknown. The comparison is a sensitivity analysis on previously examined data, not an isolated causal estimate or independent external validation.

Sources: [corrected mean/SD table](../calibration_role_control_seed_corrected_v1/MIXTURE_METRICS_MEAN_SD.csv), [paired contrasts](../calibration_role_control_seed_corrected_v1/TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv), [old/new leaders](../calibration_role_control_seed_corrected_v1/AP_LEADERSHIP_OLD_VS_CORRECTED.csv).
