# Independent review: augmentation probability study

Review date: 2026-10-18. This note is a post-manifest addition. Original models, predictions, code, reports and manifests were not edited. No new training or GPU inference was performed in this review.

## Status: recovery required before calibration or operating-point interpretation

All 12 training runs have recorded PASS gates and 796 evaluated conditions per run. This does not establish correctness of the downstream analysis.

### Critical: validation predictions belong to the last epoch, not the selected checkpoint

In `run_probability_study.py:135`, `validation.npz` is overwritten every epoch. At line 140 the best checkpoint is restored and validation is recomputed, but that best-checkpoint validation output is never saved back to `validation.npz`. Test inference loads the best checkpoint correctly.

For all 12 runs, independently recomputed TWA AP from the saved validation file matches the last history epoch and differs from the best-checkpoint validation gate. For example, Chapman p010 seed 104729 has best epoch 9 / AP 0.746246, while the validation file has last epoch 14 / AP 0.714255. Ningbo p025 seed 155921 has best epoch 21 / AP 0.683343, while the saved validation file has last epoch 26 / AP 0.651462.

Consequences:

- Raw test AP, AUROC, Brier and ECE are not functions of this validation artifact and can be assessed separately.
- All new-arm validation-derived thresholds and resulting sensitivity, specificity, call rate and accuracy require recovery.
- New Ningbo Platt fits use this mismatched validation file and are then applied to best-checkpoint test scores. Calibrated scores/metrics and calibrated thresholds require recovery. This is not test-set fitting, but it is a checkpoint mismatch.
- Recovery needs clean validation inference from 12 existing best checkpoints, not retraining or reevaluation of the 792 test masks.

### Additional analysis and reporting problems

1. `analyze_ningbo_mixtures.py:52` takes the quantile after calibration. The prior recipe transforms the raw validation-B quantile through the fitted monotone calibrator. These are not exactly identical with linear quantile interpolation. Use the prior recipe and audit decision invariance, including clipping/ties.
2. The report's opening section claims Chapman p025 has higher mean-mask AP than combined. Its own bootstrap CSV says p025 minus combined is **-0.0474**, CI approximately [-0.0585, -0.0372]. Ningbo is also negative: **-0.0377**, CI approximately [-0.0449, -0.0304]. Correct the direction and the associated cross-cohort narrative.
3. Main comparison tables omit ordinary, p050 and combined, displaying dashes even though their predictions and metric rows are available. Build complete tables from the all-arm data.
4. The saved paired interval table only covers TWA AP. Complete the requested secondary TWA AUROC and raw Brier intervals for the same four contrasts and five scalar conditions (80 groups across two cohorts). These are completion of the requested analysis, not new outcome selection.
5. The report says no new models were trained, contrary to its own 12-run training table. State exactly which models were trained and which were reused.
6. Ningbo `test_labels_order.npz` stores string representations of record dictionaries instead of canonical IDs. Validate the order against the canonical split and provide an explicit ID/order binding in the recovery package without rewriting historical prediction files.

## Preliminary raw AP interpretation

Even p010 substantially improves RA-LL and both cycles relative to ordinary training in both sources. p050 generally provides stronger protection than p010/p025. Reducing augmentation probability has not established a clean-AP benefit; on Ningbo, p050 clean AP exceeds both new arms.

At fixed 25% reversal exposure, combined exceeds p025 for mean-mask AP in both cohorts. Its Ningbo AP advantages also include RA-LL, both cycles and the exact mask. Chapman differences vary by condition. Thus, masks cannot simply be described as harmful: the comparison is more favorable to combined than comparing it only with p050 suggests. Historical implementation and random-stream differences remain limitations of a causal interpretation.

This is an exploratory extension on previously examined Chapman/Ningbo test cohorts. It is not a new independent external replication or proof of an optimal augmentation probability.

## Recovery scope

Create a separate `outputs/augmentation_probability_recovery_v2/` package. Preserve all original checkpoints, test predictions, bootstrap indices and historical comparator artifacts. Regenerate validation predictions bound to the inference checkpoint; repair threshold-dependent metrics, new-arm Ningbo calibration and mixtures; complete missing intervals and tables. Preserve valid raw AP results and verify them against v1. Do not start HEEDB, Founder training or manuscript work during this recovery.

The companion `INDEPENDENT_REVIEW_AUDIT_20261018.json` records the completed numerical checks and their explicit limits.

Completed independently: 28 tracked output hashes; 60 new scalar prediction hashes; 540 raw AP/AUROC/Brier values; 450 scalar bootstrap draw checks against sklearn; 120 per-mask AP spot checks; and all 192 values in the 48 AP contrast summaries. All numerical differences in these comparisons were zero. Mean-mask contrast summaries were reproduced from saved draws and points; the complete 1000-draw, 792-mask computation was not rerun. Best-checkpoint GPU replay and checkpoint tensor equality are deferred to recovery.
