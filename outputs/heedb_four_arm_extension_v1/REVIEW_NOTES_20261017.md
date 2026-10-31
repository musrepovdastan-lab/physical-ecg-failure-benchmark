# Independent review of the HEEDB six-method extension

Review date: 2026-10-17. This note was added after the completed package; it is not part of its original artifact manifest. No training, GPU inference, calibration fitting, or changes to the completed package were performed during this review.

## Numerical checks

- Verified all 48 output artifact hashes listed in the original reproducibility manifest.
- Verified 30 frozen input hashes. One input larger than 250 MB was not rehashed; this review does not independently verify that waveform file or repeat GPU replay.
- Checked all 12 new prediction files for shared labels/patient order and checkpoint hashes; the cohort contains 10,000 unique patient identifiers.
- Independently recalculated all 819 seed-level metric rows (7,371 scalar values) using weighted sklearn AP/AUROC, explicit operating-point counts, Brier and ECE15.
- Reproduced all 280 full-sample contrasts and summaries of their saved 1,000-draw distributions.
- Recalculated 360 individual paired bootstrap contrasts from patient multiplicities: draws 0, 417 and 999; q=0, 0.01 and 1; both modes, four contrasts and five metrics.
- Reproduced the complete 1,000-draw HEEDB RNG stream after the preceding Ningbo draws.
- Confirmed preservation of all 567 applicable legacy metric rows, accounting for the explicit Huang threshold-mode mapping.
- All 108 calibration-audit rows report zero operating-decision mismatches and zero ordering violations; clipping introduces ties, so strict rank invariance should not be asserted without qualification.

These checks passed. Saved replay reports also show 18/18 comparisons passing with zero probability difference; those GPU comparisons were not independently rerun in this review.

## Required presentation correction

The first Results table in FINAL_EXTERNAL_SIX_METHOD_REPORT.md iterates through raw, calibrated and historical-Huang rows without filtering score_mode, but labels every Brier and sensitivity value as calibrated. It therefore repeats each method/q and mislabels raw Brier values (e.g. combined clean 0.104 alongside calibrated 0.081).

The underlying CSVs and paired contrasts passed independent checks. Preserve the historical report and provide a corrected table in the next versioned synthesis package: exactly one row per method/q, raw AUPRC selected from raw mode, calibrated Brier/sensitivity/specificity selected from calibrated mode, and any historical Huang threshold kept in a separate table. Do not modify bound historical scripts merely to regenerate their outputs.

## Interpretation to preserve

- Combined clean HEEDB AUPRC is 0.591685, physics-aware 0.576011 and Huang 0.572058. At q=0.01 these are 0.591252, 0.575862 and 0.572142.
- At q=1 combined falls to 0.554224, versus physics-aware 0.568320 and Huang 0.581132. Combined-minus-Huang is negative with a conditional patient-bootstrap interval excluding zero; this does not make Huang universally superior across scenarios.
- Combined clean HEEDB calibrated Brier is 0.080808, versus Huang 0.082731; its raw Brier is worse than Huang. Separate the score regimes explicitly.
- At q=0.01 combined sensitivity/specificity are 0.661479/0.909994, versus Huang 0.616967/0.918517. Higher sensitivity comes with lower specificity at these source-selected operating points.
- The clean combined-versus-physics-aware direction differs between Ningbo (negative) and HEEDB (positive). Only the high-fault reversal disadvantage has the same direction; do not claim the complete Ningbo tradeoff replicated externally.
- Combined receives 25% reversal training examples, physics-aware 50%. Their contrast cannot identify the causal effect of adding masks or select an optimal augmentation probability.
- Three-seed aggregate patient-bootstrap intervals condition on the selected checkpoints/calibrators and do not measure all training variability. Keep per-seed values visible.
- The existing submission_evidence_v1 manuscript and mentor brief predate mask recovery and these extensions. Their Ningbo mask values and some conclusions require versioned replacement.

Next work: evidence consolidation, a revised manuscript/mentor brief, and an endpoint/code-mapping packet for an actual clinician. Clinician sign-off, unseen-fault generalization, and real recorded miswiring validation have not been supplied by this extension.
