# Positioning correction 001 — 2026-10-17

This note supplements the existing novelty reassessment. The original ledger, search log, review and prior reports are retained unchanged.

## Runtime comparison already in the project

`outputs/integrated_deployment_v4/` contains a matched warm, batch-1 CPU request-to-diagnostic-response comparison. Reported medians are 10.59 ms for the physics-aware ResNet and 31.51 ms for the integrated Huang detector → predicted correction → clean ResNet chain, a 2.97× ratio. The benchmark used seven runs on the same 128-record balanced Ningbo fixture, one framework thread and CPU affinity; output agreement was checked first. This supports only an implementation-, hardware- and workload-specific runtime statement. It does not establish universal efficiency, clinical utility, Huang-paper latency superiority, or a comparison with chest-only or Cao. No new latency benchmark is needed to make this correction.

## HEEDB external-site wording

HEEDB/Emory is external relative to the Chinese Chapman/Ningbo development sources. The earlier and later patient-disjoint HEEDB cohorts both come from the same Emory source; they are not two independent external centers. Appropriate wording is “external Emory evaluation with a later patient-disjoint confirmation from the same source.” The HEEDB endpoints remain physician-code proxy constructs, with the documented mapping limitations.

## Cao full-text access limitation

The Cao review records that direct access to the complete PDF failed and the review used indexed text. Preserve this access limitation. Do not state that every paper detail was independently verified from the original PDF. The reviewed primary methods describe ordinary training and reversal test sets; the available review does not support attributing reversal augmentation to Cao. Missing code, ablations or calibration should be phrased as “not reported/found in the reviewed material,” not as an absolute claim. The existence of Cao or 3KG alone does not show that either paper demonstrated this project’s exact three-swap intervention. Priority for that exact intervention remains unestablished without a targeted search.

## References used for this correction

- `outputs/novelty_reassessment_v1/REVIEW_NOTES_20261017.md`
- `outputs/integrated_deployment_v4/FINAL_DECISION.md`
- `outputs/integrated_deployment_v4/INTEGRATED_BENCHMARK.json`
- `outputs/integrated_deployment_v4/OUTPUT_AGREEMENT.json`
- `outputs/integrated_deployment_v4/REPRODUCIBILITY_MANIFEST.json`
- `outputs/novelty_reassessment_v1/CAO2020_FULLTEXT_REVIEW.md`
