# Independent review of augmentation probability recovery

Date: 2026-10-18. Added after the original recovery manifest; original package outputs were preserved. Review status: accepted for progressing to the planned architectural control, with the interpretation limits below.

## Independently completed checks

- All 358 files listed in the recovery manifest have the recorded SHA256. The canonical recovery protocol fingerprint also matches.
- All 12 recovered validation ID sequences and label matrices match canonical split records. Recomputed validation TWA AP/AUROC match the best-checkpoint gates exactly. All 36 full-validation negative q90 thresholds reproduce exactly.
- 720 threshold metric values, covering sensitivity, specificity, call rate and accuracy on five fixed conditions per new model, were independently recalculated. Maximum absolute error: 4.997e-16.
- All 252 new Ningbo mixture rows were recalculated independently: 2,268 values covering raw/calibrated ranking, probability, threshold and decision metrics. Maximum absolute error: 5.552e-16.
- All 18 calibration fits reproduce their recorded training NLL exactly; maximum absolute optimization gradient component is 7.772e-9. All 18 raw/calibrated decision checks match, using the transformed raw threshold.
- All 300 cached scalar AUROC/Brier full-cohort points were checked against probabilities. For replicates 0, 417 and 999, 900 actual metric draws were independently recalculated. Maximum error: 2.221e-16.
- All 320 numerical summary values in the 80 new interval groups reproduce; maximum error: 4.997e-16. All 48 preserved AP groups agree with v1.

The companion `INDEPENDENT_REVIEW_AUDIT_20261018.json` records these checks. No new training, GPU replay or full 792-mask inference/bootstrap was performed by this reviewer. The saved restoration audit reports 12 exact state-dictionary matches and 24 exact test replay checks in the original batch context; those GPU/tensor checks were inspected, not independently rerun.

## Interpretation

The checkpoint/validation mismatch has been repaired. Raw ranking/probability results remain valid, and the new operating-point and calibrated metrics can now be used with their stated validation-selection limitations. The 10/25/50% extension remains exploratory on previously examined cohorts. Epoch selection used all validation records before calibration A/B splitting; therefore calibration fitting and threshold estimation are not independent of checkpoint selection. This is distinct from fitting calibration on test data, which the code does not do.

The AUROC/Brier analysis adds an important tradeoff: p050 has higher clean TWA AP than p010/p025 on Ningbo, but p025 has lower raw clean Brier (about 0.0944 versus 0.1077). Consequently a ranking advantage is not a universal probability-quality advantage. After proper calibration, new-arm clean TWA Brier is about 0.07418 for p010 and 0.07363 for p025; these are source-cohort results, not new external evidence.

At fixed 25% reversal exposure, combined has higher mean-mask TWA AP in both cohorts than p025. Benefits and costs remain condition-dependent; historical training/random-stream differences limit a causal interpretation.

## Minor reporting clarifications, not blockers

1. The report's calibration-change table is averaged across TWA, SB and STach as well as seeds. This is evident in `recover_ningbo_calibration_mixtures.py` grouping, but not stated beside the table. Do not quote its deltas as TWA-specific or as a primary endpoint improvement. Use endpoint-specific rows in the underlying CSV instead.
2. The manifest's `new_test_model_inference: false` means no new full test/condition evaluation. Small original-batch test replays did occur for auditing and are explicitly documented in the restoration report. Preserve that distinction in future summaries.
3. The validation table has duplicated Markdown column-header rows. This is presentation-only.

None of these requires another recovery loop or retraining. Preserve them as notes when interpreting/reusing the package.

## Next work

Proceed to the planned ECGFounder architectural control: establish input/checkpoint/pretraining provenance and resource feasibility, then compare matched clean versus physical-augmentation head training with a frozen backbone under a separate fixed protocol. Prior ECGFounder baseline results demonstrate vulnerability only; they do not already establish mitigation transfer. Keep the mixed Transformer result visible and do not present a frozen-head experiment as full backbone fine-tuning or a search for the best architecture.
