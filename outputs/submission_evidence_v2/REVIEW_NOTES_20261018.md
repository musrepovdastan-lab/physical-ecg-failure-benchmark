# Independent synthesis review and corrections

Review date: 2026-10-18. Original package files were copied to `archive_before_review_20261018/` before editing. No training, inference, calibration fitting or bootstrap rerun was performed. Historical experimental packages remain unchanged.

## Checks and corrections

- All ten original package-file hashes and all five referenced protocol-file hashes matched the original manifests.
- All 42 rows and 756 displayed numerical values in the main corrected external table matched the controlling CSV, including the separate raw/calibrated source modes. That table was numerically correct before this review.
- Fixed the manuscript's masking q=1 example: raw Brier is 0.567 and calibrated Brier is 0.304. The previously written 0.693 and 0.159 are sensitivity and AUPRC, respectively, not Brier values.
- Distinguished the full-cohort mean-mask-minus-reversal contrast reduction (0.465748, displayed as 0.4657) from the bootstrap mean (0.464858, historically displayed as 0.4649). Existing percentile intervals were retained; no distribution was recomputed. Corrected the manuscript, claims ledger and Russian mentor brief consistently.
- Clarified HEEDB cohort reuse: the initial corrected external evaluation and subsequent patient-disjoint confirmation used different Emory cohorts. The six-method extension reused the latter 10,000-patient cohort after access to its outcomes. Extension and confirmation must not be counted as two different samples or new confirmations.
- Added a direct Ningbo ordinary/physics-aware table for clean, reversal, mean-mask and exact-mask AUPRC, with the main baseline contrasts. These controls were previously discussed without their central values.
- Documented the fact that the validation A/B split followed historical full-validation checkpoint selection; it does not make calibration independent of model selection.
- Fixed the source index to identify the Ningbo mask recovery separately from the unaffected Chapman ResNet mask analyses.
- Replaced literal question-mark separators in the external table with unambiguous ASCII notation (`+/-`, `x`, `Delta`, `->`).
- Added a clinician question about the provenance of `codes_physician`: a field name alone does not prove adjudication of each statement.
- Corrected the assertion that operating-metric intervals were in the condensed summary; their actual source is the paired-contrast CSV.

## Remaining work

The current package is appropriate for mentor/coauthor discussion after these corrections. It remains a working draft rather than a submission-formatted paper. Methods need complete cohort/split/training details from the frozen configurations, and the article needs an actual bibliography and sufficient quantitative presentation of the second architecture.

The official ICASSP conference entry for Cao et al. was checked again during this review. Direct access to the proceedings PDF timed out again; indexed primary-text content remains available, as previously recorded. A locally retained complete PDF has not been obtained by this review. Source: https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=3120 ; attempted PDF: https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000951.pdf . This does not replace the existing literature review or establish priority for the proposed contribution.

Clinical endpoint review and author decisions remain pending. No clinician approval, broad architecture superiority, unseen-reversal generalization or real-miswiring validation should be inferred from this editorial review.
