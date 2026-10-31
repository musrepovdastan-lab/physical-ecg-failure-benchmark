# Readiness and remaining work

## Ready for coauthor review

- Versioned evidence synthesis with the controlling source and score/threshold mode for each main result.
- Corrected, one-row-per-method/q HEEDB TWA summary directly rebuilt from reviewed CSV rows; raw and calibrated metrics are separated and the historical Huang threshold is separate.
- Claim/evidence ledger, response to mentor concerns, English manuscript draft and one-page Russian mentor brief.
- Russian clinician packet describes current SNOMED/HEEDB mappings and implementation rules without changing them or treating a model-generated interpretation as clinical review.
- Historical output directories remain intact. This package contains no new model, inference or experiment outputs.

## Requires author or mentor decision

- Final title, novelty/contribution language, and how prominently to describe the mask-semantic incident.
- Which exploratory components belong in the main paper versus supplement (M3 failure, chest-only, Founder, four-arm, calibration mixtures and runtime).
- Complete primary-text verification of Cao 2020 and a targeted related-work check; the retained review notes that direct PDF access failed.
- Whether the current data/analysis access and proxy endpoint framing fit the intended venue and disclosure requirements.

## Requires clinician review

- Intended target: preservation of an ECG interpretation, detection of a coded feature, or disease diagnosis.
- SNOMED meaning and context for `164934002`, `426177001`, `427084000`.
- Clinical meanings and grouping of HEEDB core/strict codes; excluded/ambiguous codes; unknown values; and whether absent core codes in a nonempty record should count as negative.
- Whether the proxy mapping is acceptable for descriptive external evaluation and what terminology should be used.

## Scientific evidence still absent

- Validation with real, documented lead miswirings and held-out reversal compositions.
- Independent external centers and clinician-adjudicated/endpoint-validated labels.
- Prospective operating thresholds, workflow outcomes and clinical utility.
- Additional training seeds or a prospectively specified confirmatory analysis if authors need stronger training-uncertainty or generalization claims.
- Ningbo M3, Founder mitigation, and a causal design that holds physical-reversal exposure fixed while varying masking, if those claims are desired.
- Deployment measurements on target clinical equipment/software and broader workload conditions.

These are potential future studies, not work required to assemble this coauthor-review packet. The current package does not mark scientific or clinical readiness as PASS. No follow-up experiment was run.
