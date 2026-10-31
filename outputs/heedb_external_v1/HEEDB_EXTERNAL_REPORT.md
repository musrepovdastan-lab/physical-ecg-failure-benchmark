# HEEDB I0006 external transportability test v1

## Decision

The frozen external evaluation completed on 10,000 patient-disjoint Emory ECGs. It is a fair, label-blind external transportability test of the frozen Ningbo baseline and physics-aware classifiers under the prespecified HEEDB physician-overread proxy labels. It is **not** an exact SNOMED-label replication, so construct mismatch remains a limitation.

## Primary core-TWA comparisons

Positive values for `physics-aware minus baseline` favor the physics-aware arm. Positive within-arm `clean minus RA–LL` values indicate degradation under reversal.

| Metric | Contrast | Estimate | Paired patient bootstrap 95% CI |
|---|---|---:|---:|
| auprc | baseline_clean_minus_ra_ll | -0.0036 | [-0.0108, 0.0041] |
| auprc | physics_aware_clean_minus_ra_ll | -0.0052 | [-0.0084, -0.0021] |
| auprc | physics_aware_minus_baseline_clean | -0.0139 | [-0.0194, -0.0084] |
| auprc | physics_aware_minus_baseline_ra_ll | -0.0123 | [-0.0180, -0.0063] |
| sensitivity_at_frozen_threshold | baseline_clean_minus_ra_ll | 0.0584 | [0.0478, 0.0691] |
| sensitivity_at_frozen_threshold | physics_aware_clean_minus_ra_ll | 0.0048 | [0.0013, 0.0086] |
| sensitivity_at_frozen_threshold | physics_aware_minus_baseline_clean | -0.0658 | [-0.0755, -0.0562] |
| sensitivity_at_frozen_threshold | physics_aware_minus_baseline_ra_ll | -0.0122 | [-0.0176, -0.0069] |

The physics-aware arm therefore reduced the RA–LL sensitivity loss (0.0048 versus 0.0584) but did not improve absolute core-TWA performance: its clean and RA–LL AUPRC and sensitivity were all lower than baseline. AUPRC rose slightly rather than falling under RA–LL in both arms, so the prespecified AUPRC degradation contrast does not show a reversal penalty in this proxy-label cohort.

## Three-seed mean performance

| Mapping/outcome | Arm | Condition | AUPRC | Sensitivity at frozen threshold |
|---|---|---|---:|---:|
| core TWA | baseline | clean | 0.1579 | 0.0809 |
| core TWA | baseline | RA–LL | 0.1616 | 0.0225 |
| core TWA | physics-aware | clean | 0.1440 | 0.0151 |
| core TWA | physics-aware | RA–LL | 0.1493 | 0.0103 |
| core SB | baseline | clean | 0.1851 | 0.9550 |
| core SB | baseline | RA–LL | 0.1611 | 0.8551 |
| core SB | physics-aware | clean | 0.2094 | 0.9900 |
| core SB | physics-aware | RA–LL | 0.2082 | 0.9934 |
| core STach | baseline | clean | 0.0870 | 0.3432 |
| core STach | baseline | RA–LL | 0.0582 | 0.2184 |
| core STach | physics-aware | clean | 0.0650 | 0.4889 |
| core STach | physics-aware | RA–LL | 0.0620 | 0.4364 |
| strict TWA | baseline | clean | 0.0965 | 0.1030 |
| strict TWA | baseline | RA–LL | 0.0956 | 0.0268 |
| strict TWA | physics-aware | clean | 0.0911 | 0.0186 |
| strict TWA | physics-aware | RA–LL | 0.0965 | 0.0139 |
| strict SB | baseline | clean | 0.1619 | 0.9526 |
| strict SB | baseline | RA–LL | 0.1390 | 0.8545 |
| strict SB | physics-aware | clean | 0.1797 | 0.9899 |
| strict SB | physics-aware | RA–LL | 0.1786 | 0.9933 |

For the core-TWA 24-mask subset, mean AUPRC was 0.1596 for baseline and 0.1553 for physics-aware. The paired physics-aware-minus-baseline contrast was -0.0044 (95% CI -0.0077 to -0.0004). This is explicitly a subset result, not the all-792-mask mean.

## Aggregate label counts

All counts use `codes_physician` only.

| Mapping/outcome | Positive | Total |
|---|---:|---:|
| core_SB | 1,665 | 10,000 |
| core_STach | 673 | 10,000 |
| core_TWA | 1,524 | 10,000 |
| strict_SB | 1,448 | 10,000 |
| strict_TWA | 770 | 10,000 |

## Secondary results

Core SB/STach and strict TWA/SB clean and RA–LL results are in `AGGREGATE_RESULTS_BY_SEED.csv` and `AGGREGATE_CONTRASTS.csv`. The mask comparison is the prespecified deterministic **24-mask subset**, not the 792-mask mean; each displayed lead occurs in exactly 10 masks. Only its core-TWA arm contrast receives the prespecified paired-bootstrap interval.

## Cohort, exclusions, and transfers

- Joined table rows: 998,844.
- Excluded for missing patient ID: 27,328.
- Excluded after patient linkage for empty physician codes: 675.
- Excluded readiness-sample patients: 23.
- Eligible patients after one deterministic ECG per patient: 349,482.
- Evaluated patients/recordings: 10,000 / 10,000.
- Unreadable attempts: 0; replacements: 0.
- Exact waveform bytes downloaded/retained for this test: 1,178,093,385 (1.097 GiB), against the 2 GiB cap.
- Remaining disk space at report completion: 169,455,800,320 bytes (157.818 GiB).
- Patient- and recording-level manifests, waveforms, cached tensors, and predictions remain under `data/heedb_restricted/I0006/external_v1/`, which is git-ignored.

## Label interpretation

Primary labels are physician-overread `codes_physician`: core TWA 1140/1145/1150/1151/1160/1161/1170/1171/1172/1180/1181, SB 21/24, and sinus tachycardia 23. Strict sensitivity mappings are TWA 1140 and SB 21. Software codes and v24 labels were never merged. Mixed ST–T, suggestive/fragment, serial-change T-wave codes and probable SB code 265 were excluded from the positive definitions. These clinical constructs overlap with, but are not identical to, the Chapman/Ningbo SNOMED outcomes.

## Frozen implementation and provenance

- Protocol SHA-256: `a1409359fe89e6e93f1cfdc734d82b212807e8ebe6e8aa7adb181c55b527d2c1`.
- Evaluation code SHA-256: `6c7186779578a91240a6137fb563f6b37a0aab1aa46b5684bf7672af76ea3450`.
- Engineering smoke: PASS on 23 excluded readiness records, all six models and 26 conditions.
- Frozen inference elapsed: 11.4 minutes.
- Three matched seeds were aggregated as means of seed-level metrics/contrasts; prediction vectors were not averaged across seeds.
- Uncertainty uses 1,000 synchronized patient-level bootstrap resamples.

## Reproducibility artifacts

Aggregate results and all hashes are in this output directory. Restricted manifests and predictions are intentionally excluded. The exact checkpoint, validation-threshold, preprocessing, source, and upstream protocol hashes are recorded in the frozen protocol and `RUN_PROVENANCE.json`.
