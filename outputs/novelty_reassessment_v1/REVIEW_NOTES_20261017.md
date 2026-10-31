# Review of novelty reassessment — 2026-10-17

The proposed next package (mostly-clean scenarios and validation-only calibration) is well aligned with the remaining evidence gaps. Preserve the simulation-bounded evaluation/comparison framing, chest-only results, M3 retention failure, proxy endpoints and post-hoc status of Huang.

Three corrections are required before using the reassessment as manuscript evidence.

## 1. An integrated runtime comparison already exists

The efficiency row of CLAIMS_LEDGER.md incorrectly states that no controlled matched inference-cost comparison exists. See:

- outputs/integrated_deployment_v4/FINAL_DECISION.md
- outputs/integrated_deployment_v4/INTEGRATED_BENCHMARK.json
- outputs/integrated_deployment_v4/OUTPUT_AGREEMENT.json
- outputs/integrated_deployment_v4/REPRODUCIBILITY_MANIFEST.json

The reported warm batch-1 CPU request-to-diagnostic-response medians are 10.59 ms for physics-aware ResNet and 31.51 ms for the integrated Huang detector, predicted correction and clean ResNet chain (2.97x ratio). Seven runs use the same 128-record balanced Ningbo fixture, one framework thread and CPU affinity on the documented machine. Output agreement was checked before timing. This is an implementation/hardware/workload-specific result, not universal efficiency, clinical utility, or superiority to Huang's own published latency. It does not compare chest-only or Cao. No repeat benchmark is necessary solely to correct the ledger.

## 2. External-site wording must distinguish the comparisons

HEEDB/Emory is external to the Chapman/Ningbo training sources. Two patient-disjoint HEEDB samples from the same Emory source do not constitute two independent external centers. Therefore say "external Emory evaluation with a later patient-disjoint confirmation from the same source," rather than implying that HEEDB is not external to the Chinese development datasets. Proxy-label and historical-access limitations still apply.

## 3. Full-text access confidence and novelty attribution

The Cao review transparently states that direct PDF access failed and indexed text was used. Preserve that limitation; retain an accessible full PDF or complete primary-text export before treating all page/table details as independently checked. Missing code, ablations or calibration should be described as not reported/found in the reviewed material, rather than absolute absence.

The accessible primary methods support ordinary training and reversal test sets; do not attribute reversal augmentation to Cao. The existence of Cao or 3KG alone does not establish priority for the exact three-swap intervention. Avoid first/novel-augmentation claims without a targeted search, but distinguish "priority not established" from "this exact method was demonstrated in those papers."

## Scope of the next computational package

First inventory score coverage. Calibration requires validation outputs from the complete diagnostic system, especially Huang-to-correction-to-classifier. HEEDB currently lacks the chest-only comparison. Generate missing scores only with existing frozen models and already available data; do not silently omit a competitor or fit calibration on test/HEEDB labels.

Use one deployable calibrator per model/seed/endpoint, applied to both clean and reversed inputs. A condition-specific calibrator selected using the true simulated fault would leak unavailable information. Monotone calibration should preserve ranking, so its purpose here is probability calibration and operating-point analysis, not manufacturing AUPRC gains.
