# Ningbo four-arm training replication

**Protocol:** `4718893819c305c0c8efbbe2f9a1e5760d2f622f3f2038fce3685a894acb74ae`. **Scope:** follow-up on Ningbo after prior test access; this is exploratory corrected reanalysis, not untouched confirmation.

## Training and validation

The only newly trained arms were masking and combined (three fixed seeds each). Clean and physics-aware checkpoints were reused after checkpoint, preprocessing, split, corrected mask-cache and condition bindings were checked. The six unit summaries, stop reasons, elapsed time and measured GPU process allocation are in `TRAINING_STATUS_BY_ARM_SEED.csv`. Historical artifacts did not retain per-run wall time or GPU peak memory, so those fields are explicitly marked unavailable for reused models.

| Arm | Seed | Status | Best epoch | Validation TWA AUPRC | Validation TWA AUROC | Stop reason | Wall seconds | Peak allocated GPU bytes | Checkpoint |
|---|---:|---|---:|---:|---:|---|---:|---:|---|
| clean | 104729 | PASS | 14 | 0.6605 | 0.9203 | historical validation early stopping (patience 5) | not recorded in historical clean run artifacts | not recorded in historical clean run artifacts | `outputs/ningbo_confirmation_v2/seed_104729/checkpoint.pt` |
| clean | 130363 | PASS | 11 | 0.6532 | 0.9211 | historical validation early stopping (patience 5) | not recorded in historical clean run artifacts | not recorded in historical clean run artifacts | `outputs/ningbo_confirmation_v2/seed_130363/checkpoint.pt` |
| clean | 155921 | PASS | 6 | 0.6541 | 0.9199 | historical validation early stopping (patience 5) | not recorded in historical clean run artifacts | not recorded in historical clean run artifacts | `outputs/ningbo_confirmation_v2/seed_155921/checkpoint.pt` |
| masking | 104729 | PASS | 25 | 0.6783 | 0.9217 | early stopping: validation TWA AUPRC patience 5 | 2041.115083 | 282212864 | `outputs/ningbo_four_arm_replication_v1/masking/seed_104729/checkpoint.pt` |
| masking | 130363 | PASS | 16 | 0.6639 | 0.9228 | early stopping: validation TWA AUPRC patience 5 | 1301.576676 | 283986432 | `outputs/ningbo_four_arm_replication_v1/masking/seed_130363/checkpoint.pt` |
| masking | 155921 | PASS | 25 | 0.6769 | 0.9228 | early stopping: validation TWA AUPRC patience 5 | 1894.310626 | 283986432 | `outputs/ningbo_four_arm_replication_v1/masking/seed_155921/checkpoint.pt` |
| physics_aware | 104729 | PASS | 17 | 0.6715 | 0.9223 | historical validation early stopping (patience 5) | not recorded in historical M2 run artifacts | not recorded in historical M2 run artifacts | `outputs/mitigation_m2_ningbo_v1/seed_104729/checkpoint.pt` |
| physics_aware | 130363 | PASS | 20 | 0.6704 | 0.9178 | historical validation early stopping (patience 5) | not recorded in historical M2 run artifacts | not recorded in historical M2 run artifacts | `outputs/mitigation_m2_ningbo_v1/seed_130363/checkpoint.pt` |
| physics_aware | 155921 | PASS | 11 | 0.6634 | 0.9185 | historical validation early stopping (patience 5) | not recorded in historical M2 run artifacts | not recorded in historical M2 run artifacts | `outputs/mitigation_m2_ningbo_v1/seed_155921/checkpoint.pt` |
| combined | 104729 | PASS | 13 | 0.6704 | 0.9216 | early stopping: validation TWA AUPRC patience 5 | 1242.97919 | 283986432 | `outputs/ningbo_four_arm_replication_v1/combined/seed_104729/checkpoint.pt` |
| combined | 130363 | PASS | 12 | 0.6807 | 0.9236 | early stopping: validation TWA AUPRC patience 5 | 1107.840377 | 283986432 | `outputs/ningbo_four_arm_replication_v1/combined/seed_130363/checkpoint.pt` |
| combined | 155921 | PASS | 14 | 0.6682 | 0.9257 | early stopping: validation TWA AUPRC patience 5 | 1242.999365 | 283986432 | `outputs/ningbo_four_arm_replication_v1/combined/seed_155921/checkpoint.pt` |

All 792 masks were evaluated separately. The exact mask removes I, II, III, aVR and aVF (indices `[0,1,2,3,5]`). No prediction vectors were averaged. TWA is primary; SB and STach are secondary. Full metrics for every seed and condition are in `RESULTS_BY_ARM_SEED.csv`, with seed means and SDs in `RESULTS_MEAN_BY_ARM.csv`; exact-mask metrics and mask distributions are in the per-seed and mean-summary CSVs.

## Prespecified TWA comparisons

Intervals below use the same 1,000 canonical-record resamples across both arms, all seeds and conditions. Central differences use full-cohort point estimates; bootstrap means are reported separately. A 95% interval crossing zero leaves the difference unresolved and does not establish equivalence. Comparisons were prespecified, but no multiplicity adjustment was defined; interpret the family as exploratory.

| Contrast | Estimand | Full-cohort difference | Bootstrap mean | 95% interval |
|---|---|---:|---:|---:|
| masking_minus_clean | clean | -0.0021 | -0.0021 | [-0.0119, +0.0076] |
| masking_minus_clean | RA_LL_reversal | -0.0130 | -0.0129 | [-0.0192, -0.0060] |
| masking_minus_clean | mean_mask | +0.0598 | +0.0597 | [+0.0502, +0.0696] |
| combined_minus_physics_aware | clean | -0.0101 | -0.0101 | [-0.0196, -0.0009] |
| combined_minus_physics_aware | RA_LL_reversal | -0.0218 | -0.0220 | [-0.0325, -0.0122] |
| combined_minus_physics_aware | mean_mask | +0.0371 | +0.0369 | [+0.0288, +0.0446] |

The seed-level primary results are:

| Arm | Seed | Clean | RA–LA | RA–LL | LA–LL | Mean of 792 masks | Exact mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 104729 | 0.6736 | 0.2095 | 0.1773 | 0.6262 | 0.5857 | 0.3973 |
| clean | 130363 | 0.6756 | 0.3210 | 0.1878 | 0.6291 | 0.5884 | 0.4723 |
| clean | 155921 | 0.6791 | 0.2533 | 0.1814 | 0.6462 | 0.5849 | 0.4378 |
| masking | 104729 | 0.6791 | 0.3109 | 0.1576 | 0.6207 | 0.6470 | 0.5992 |
| masking | 130363 | 0.6651 | 0.4029 | 0.1380 | 0.6300 | 0.6402 | 0.6233 |
| masking | 155921 | 0.6778 | 0.3471 | 0.2119 | 0.6357 | 0.6511 | 0.6310 |
| physics_aware | 104729 | 0.6955 | 0.6808 | 0.6741 | 0.6779 | 0.6088 | 0.6285 |
| physics_aware | 130363 | 0.6761 | 0.6590 | 0.6647 | 0.6504 | 0.5929 | 0.6001 |
| physics_aware | 155921 | 0.6890 | 0.6684 | 0.6607 | 0.6789 | 0.6131 | 0.6493 |
| combined | 104729 | 0.6678 | 0.6373 | 0.6508 | 0.6562 | 0.6369 | 0.6487 |
| combined | 130363 | 0.6898 | 0.6564 | 0.6557 | 0.6649 | 0.6508 | 0.6404 |
| combined | 155921 | 0.6727 | 0.6482 | 0.6277 | 0.6579 | 0.6383 | 0.6487 |

## Mostly-clean RA–LL mixtures and calibration

The new masking and combined Platt fits used only clean validation A; raw thresholds were the 90th percentile of negative clean validation B scores, and calibrated thresholds were transformed through the same fitted monotone map. Clean and physics-aware fits were reused unchanged from mixed-fault calibration v2. No test data were used for fitting. Per-seed raw/calibrated metrics at every frozen q are in `MIXTURE_RESULTS_BY_SEED.csv`, with cross-seed means/SDs in `MIXTURE_MEAN_SUMMARY.csv`; paired TWA bootstrap intervals are in `MIXTURE_BOOTSTRAP_SUMMARY.csv`.

| Arm | TWA q=0 raw AUPRC | Raw Brier | Calibrated Brier | Raw ECE15 | Calibrated ECE15 |
|---|---:|---:|---:|---:|---:|
| clean | 0.6761 | 0.0990 | 0.0737 | 0.0978 | 0.0117 |
| masking | 0.6740 | 0.1079 | 0.0741 | 0.1162 | 0.0121 |
| physics_aware | 0.6869 | 0.1077 | 0.0730 | 0.1247 | 0.0115 |
| combined | 0.6768 | 0.1024 | 0.0733 | 0.1175 | 0.0122 |

## Interpretation


- **Does masking protect against RA–LL?** Masking minus clean-trained TWA AUPRC under RA–LL was -0.013 (95% interval -0.019 to -0.006). Judge the size, interval and seed signs together; this comparison does not imply complete protection.
- **Is combined more balanced than physics-aware?** Combined minus physics-aware was -0.010 (95% interval -0.020 to -0.001) on clean ECG, -0.022 (95% interval -0.032 to -0.012) on RA–LL and +0.037 (95% interval +0.029 to +0.045) on mean-mask performance. Combined uses 25% reversal examples versus 50% in physics-aware, so this contrast cannot isolate the effect of adding masks.
- **Clean cost and calibration:** clean-condition raw test calibration and validation-calibrated mostly-clean results should be read separately in `RESULTS_BY_ARM_SEED.csv` and `MIXTURE_RESULTS_BY_SEED.csv`; the calibration map is monotone and preserves operating decisions except measured clipping/ties.
- **Seed stability:** see the seed-level table above. Three seeds provide limited precision; a result that changes direction by seed is not stable evidence.
- **Chapman transfer:** Chapman M1 had passed its frozen gate for physics-aware as primary, while masking/combined were secondary. This Ningbo study is a follow-up, not fresh confirmation. The exact Chapman-to-Ningbo combined comparison is tabulated below; Chapman source results remain unchanged.

Chapman M1 combined versus cached clean baseline TWA AUPRC (full-cohort test estimates; source M1 values):

| Seed | Condition | Combined | Clean baseline | Difference |
|---:|---|---:|---:|---:|
| 104729 | clean | 0.6635 | 0.6472 | +0.0163 |
| 104729 | RA_LL_reversal | 0.6394 | 0.2753 | +0.3641 |
| 130363 | clean | 0.6643 | 0.6513 | +0.0130 |
| 130363 | RA_LL_reversal | 0.6298 | 0.3403 | +0.2895 |
| 155921 | clean | 0.6480 | 0.6609 | -0.0129 |
| 155921 | RA_LL_reversal | 0.6250 | 0.3679 | +0.2572 |
- **Reversal exposure:** if combined differs from physics-aware, both mask exposure and halved reversal exposure change together. No mechanism can be attributed from this comparison.
- **HEEDB:** a future external analysis may be justified if the new fixed arms show reproducible tradeoffs, but this package does not run HEEDB inference and Ningbo findings alone do not establish external validity.
- **Competing results:** retain chest-only and Huang comparisons from `outputs/mixed_fault_calibration_v2/FINAL_FOUR_METHOD_COMPARISON.md`. The corrected recovery reports and the TimePatchTransformer M3 Chapman `FAIL_M3_ARCHITECTURE_SPECIFIC` remain in force; no global winner is claimed.

No hyperparameter search, retraining of clean/physics-aware, reserved-test evaluation, new data, new split, or HEEDB inference was performed.
