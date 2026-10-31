# Independent review: HEEDB probability extension

Review date: 2026-10-19. Status: accepted as a completed exploratory external extension. This note and the companion audit JSON were added after the original package manifest. Original predictions, code, protocols, tables and reports were preserved.

## Independent checks

- Verified all 53 output-file hashes and all 83 frozen input hashes, including the original 2.4 GB waveform cache. Verified the unchanged HEEDB bootstrap index file against the preceding extension.
- Reconstructed proxy labels from the original diagnosis-code file and joined them to the original selection. Confirmed 10,000 unique patient IDs and exact labels/order in new prediction bundles and reused masking/combined bundles.
- Checked all 12 new prediction checkpoint/preprocessing bindings. Inspected the saved 18/18 successful GPU replay checks; GPU inference was not independently repeated.
- Checked all 18 new calibration coefficient/threshold entries against the recovered Ningbo source artifact. No new fitting was performed.
- Recomputed all 1,071 seed-specific mixture rows: 9,639 scalar values spanning AP, AUROC, Brier, ECE15, sensitivity, specificity, call fraction and FP/FN per 1,000. Maximum error: 5.552e-16.
- Independently reproduced 4,998 mean/SD summary values. Maximum error: 3.331e-16.
- Recomputed all 144 calibration decision checks with zero mismatches.
- Reproduced all 48 full-cohort contrast points; maximum error: 4.219e-15.
- Reproduced all 144 mean/percentile summaries from the 48 saved 1,000-draw arrays, with zero difference.
- Independently recomputed actual repeated-patient contrasts for draws 0, 417 and 999 in all 48 groups: 144 checks, maximum error 2.367e-15.
- Confirmed preservation of all 819 historical metric rows.

The complete 1,000-draw metric-generation computation was not repeated; saved summaries and the stated real-data resample spots were checked independently. Exact counts and limits are recorded in `INDEPENDENT_REVIEW_AUDIT_20261019.json`.

## Interpretation to preserve

At hypothetical q=0.01, new p010/p025 mean raw TWA AP is about 0.5852/0.5849, versus p050 0.5759, combined 0.5913 and Huang 0.5721. The p025-minus-p050 AP interval is positive, while p010-minus-p050 includes zero. Do not turn both point advantages into two established advantages. Neither new arm exceeds combined in mean AP, and the p025-minus-combined low-q AP interval crossing zero is not equivalence.

At q=1, new p010/p025 AP is about 0.5367/0.5288, below p050 0.5683 and Huang 0.5811. Calibrated Brier also favors p050/Huang over these new arms at q=1. Results support a fault-prevalence-dependent tradeoff, not a universal optimal augmentation probability.

The low-q operating-point narrative needs a comparator-specific qualification. At q=0.01, calibrated TWA sensitivity/specificity is about:

| Method | Sensitivity | Specificity | FP per 1,000 |
| --- | --- | --- | --- |
| p010 | 0.65236 | 0.91288 | 74.436 |
| p025 | 0.63555 | 0.91344 | 73.955 |
| p050 | 0.65610 | 0.90359 | 82.373 |
| combined | 0.66148 | 0.90999 | 76.901 |
| Huang | 0.61697 | 0.91852 | 69.619 |

Consequently, the report sentence that operating decisions did not worsen at low q must not be generalized to every comparator or both sensitivity and specificity. Relative to Huang, new arms gain sensitivity and lose specificity; relative to p050 they have lower sensitivity and higher specificity. Preserve those tradeoffs in consolidation. This is an interpretation clarification, not an invalid underlying CSV or a reason to rerun the experiment.

HEEDB remains the same previously examined external cohort; new probabilities have not acquired a fresh independent test population. q is an assumed mixture scenario, not a measured hospital fault rate. Calibrators and validation-B thresholds are source-derived, but the preceding full-validation checkpoint selection remains a dependence limitation. Proxy labels and the meaning of absent codes still require clinical interpretation.

The five documented amendments preserve explicit inference-versus-analysis protocol lineage. Do not describe the final analysis code as though it were unchanged throughout the run. The audit-coefficient fix did not alter the verified metric rows or probabilities.

## Next work

The planned computational sequence is complete, with its positive and mixed results. Next consolidate the mentor checklist, experiment chronology and defensible contribution/claims from the corrected packages. Do not automatically launch another training run, tune a probability on HEEDB, or treat noise/contact stress tests and real recorded miswirings as prerequisites for a simulation-scoped paper. Actual clinician endpoint/code-mapping decisions remain outstanding; do not fabricate sign-off or silently revise labels.
