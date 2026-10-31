# Verification report

**Review update, 2026-10-18:** Independent consistency checks found and corrected a Brier/sensitivity/AUPRC transposition in the manuscript, ambiguous cohort reuse language, and a bootstrap-mean versus full-cohort contrast label. The corrected external main table itself passed all 42-row/756-number checks. See `REVIEW_NOTES_20261018.md` for the corrections and their scope; original documents are retained in `archive_before_review_20261018/`. The original checks below should be read together with that review, rather than as a claim that the initial manuscript had no transcription errors.

## Scope

This package synthesizes existing artifacts only. No training, GPU inference, new experiment, calibration fit, or bootstrap recomputation was run. Historical evidence and source directories were read as-is; this output is written under the new `outputs/submission_evidence_v2/` directory.

## Direct source checks

- Re-read the post-mentor plan and current terminal reports for mask recovery, M3, chest-only, novelty review, mixed-fault calibration, Ningbo four-arm, HEEDB six-method extension and integrated runtime.
- Confirmed the HEEDB external review note explicitly requires separating raw, calibrated and historical-Huang threshold rows; it independently verified 48 output hashes, 30 frozen input hashes (one input >250 MB was not rehashed), 819 seed-level metric rows, 280 full-sample contrasts, sampled paired bootstrap contrasts, and the complete bootstrap RNG stream. Its saved GPU replay was not repeated by that review.
- Read `MIXTURE_MEAN_SUMMARY.csv` directly: 273 rows, six methods, three endpoints, seven q values, and three score modes. For each of the 42 TWA method × q pairs, the corrected table joins only `raw` and `calibrated` rows with `threshold_type=validation_B_90pct`; raw/calibrated AUPRC, AUROC and threshold decisions match as expected for the source-calibration transform.
- Read `CALIBRATION_RANKING_AUDIT.csv` directly: all 108 rows have `exact_threshold_decision_mismatches=0` and `max_monotonicity_violation=0`. The independent review notes that clipping can introduce ties; strict rank invariance is not claimed without qualification.
- The historical Huang sensitivity table is sourced only from rows tagged `score_mode=raw_historical_threshold` and `threshold_type=historical_HEEDB_exploratory_v1`. It is excluded from the primary operating-point table.
- The paired contrast table in the corrected summary reads the saved `PAIRED_CONTRASTS_BOOTSTRAP.csv`; no interval was regenerated.
- Confirmed endpoint mappings and ambiguity handling against `src/heedb_external_v1.py`, its frozen config, the independent-confirmation config, and the source protocol descriptions. No mapping or code was changed.
- Cross-checked headline mask, chest-only, four-arm, M3 and runtime numbers against their current final reports and linked source CSVs. Superseded old exact-mask values are described only as historical values and explicitly marked invalid for the intended mask.

## Scope and wording checks

- The manuscript separates raw ranking and raw calibration values from calibrated Brier/ECE and operating metrics.
- It defines `q_fault` as an evaluation mixture weight and `p_aug` as training allocation, never as measured prevalence.
- It preserves the unequal 25% versus 50% reversal exposure, the Ningbo/HEEDB clean-direction disagreement, M3 failure, untested Ningbo M3, Founder frozen-backbone scope, proxy labels, shared Emory source, and conditional bootstrap limitations.
- It makes no universal architecture, equivalence, SOTA, disease-diagnosis, or clinical-benefit claim.
- The clinician packet includes no patient-level identifiers, waveforms, restricted cohort manifests or individual predictions; it records the mappings as unresolved pending physician review.

## Reproducibility identifier

Composite frozen-method fingerprint: `3b31f1c1a50fb82e98b86f93dbf35820ebfec7c9bd5f4438e1c10bba2b1348aa`. Its constituent protocol paths and hashes are in `METHOD_FINGERPRINT.json`. This fingerprint identifies the existing method artifacts referenced by the synthesis; it is not a new scientific protocol or a readiness decision.

## Limitations of this verification

The report is a consistency review, not an independent reproduction of training, inference, bootstrap, or clinical validity. It inherits the explicitly recorded scope of each source audit. In particular, one large HEEDB input was not rehashed in the external independent review, GPU replay was not repeated in that review, complete primary-text verification of Cao 2020 remains open, and no clinician has signed off on endpoints. Scientific and clinical readiness are therefore unresolved; no automatic PASS is assigned.
