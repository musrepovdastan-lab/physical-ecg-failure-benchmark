# Review and corrections to the consolidation

Date: 2026-10-19. Original v3 documents and manifest are preserved in `archive_before_review_20261019/`. Only consolidation documents were edited; no experimental code, models, predictions, labels or historical reports were changed. No training or inference was performed.

All 31 original manifest entries matched their recorded hashes before edits. Existing Markdown links resolved, but several content/provenance statements required correction:

1. Two completed tests had been copied into the future-work list: held-out cycles and matched 25% reversal exposure. They are now explicitly recorded as completed. Only genuinely untested families/stresses remain optional future work.
2. The chronology omitted the 12 new p010/p025 trainings preceding validation recovery. These trainings are now distinguished from subsequent recovery, which did not retrain the models.
3. Mask recovery was incorrectly described as merely reusing saved mask predictions. The corrected Ningbo package generated 4,752 new mask predictions on existing checkpoints; only verified clean/RA-LL caches were reused. No new training occurred.
4. The Huang row incorrectly called the reported calibrated operating point a legacy threshold. The quoted primary calibrated sensitivity/specificity uses the full-pipeline source-validation B threshold; the legacy threshold is a separate sensitivity analysis.
5. The HEEDB source-index row pointed to Ningbo mixture/interval filenames. It now points to the actual HEEDB CSVs and their correct method/q/score-mode filters.
6. The contribution paragraph could be read as saying the models trained on cycles. It now states that training used individual swaps and the unseen cycles were evaluation conditions.
7. The original clean-HEEDB probability/ranking costs raised by the mentor were restored explicitly: ordinary versus p050 raw Brier 0.099399 versus 0.115225, AUROC 0.903118 versus 0.892727; calibrated Brier 0.082636 versus 0.084022. These are distinct score regimes, not interchangeable results.
8. Matched-exposure evidence is now visible in the checklist and source index: combined beats p025 on mean-mask AP in both cohorts, while remaining implementation/exposure differences limit a purely causal interpretation.
9. The English mentor update now treats confirmed real miswirings as an optional validation extension, separately from outstanding clinical interpretation of endpoint definitions and proxy mappings.

All local Markdown links were converted to verified absolute paths. The revised manifest records the current documents and additional directly cited numerical sources, with lineage to the archived original manifest. Numerical spot checks and link/manifest validation are recorded in `INDEPENDENT_REVIEW_AUDIT_20261019.json`.

## Status and next step

The corrected consolidation is suitable for sharing as a research status brief. It does not establish acceptance readiness for a particular venue or guarantee that the empirical contribution meets its novelty bar. The original computational follow-up is complete with positive and mixed results. The next useful action is mentor agreement on the principal contribution and claim scope, alongside clinician review of endpoint/code interpretation. No further experiment, full manuscript, publication or message to another person has been initiated by this review.

## Clarification after comparison with every planning stage

`PLAN_COMPLETION_AUDIT_20261019.md` distinguishes completion of the bounded later series from the broader 16 October wish list. The wording "original computational follow-up is complete" above was too broad if read as every item in that original plan. Full Cao access/direct-baseline decision, a calibration set independent of epoch selection, and three-swap prevalence mixtures are not fully supplied. Transformer Ningbo/HEEDB was conditional on a gate that failed; leave-one-swap-out was optional and was not performed. The audit does not invalidate the completed experiments or automatically authorize another training series.
