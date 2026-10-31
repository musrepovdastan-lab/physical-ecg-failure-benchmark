# Independent evidence verification — 26 September 2026

**Decision: PASS.** No discrepancy was found in the frozen scientific artifacts. No model inference, data download, threshold selection, or patient selection was performed for this package. Verification command: `.venv\Scripts\python.exe -m src.submission_evidence_audit_v1`; verifier SHA-256 `10e6a7f28c0d218387677aad6ffb2d15cdc80a9802d2945de4acbe91813a66fc`.

| Verification | Independent check |
|---|---|
| Protocol integrity | SHA-256 of all six Chapman, Ningbo, HEEDB-independent, and Huang frozen protocols matched their recorded bindings; full values below. |
| Cohorts | Restricted manifest hashes matched. The original HEEDB v1/corrected-v2 group and the independent-confirmation group each contained 10,000 patients; the readiness group contained 23. All three pairwise overlaps were zero. No identifiers were exported. |
| Source labels and metrics | Chapman TWA 377/2,050; Ningbo TWA 1,034/6,981. Recomputed cached-score AUPRC for clean, exact simulated RA-LL, and all 792 five-lead masks for every seed/arm, and checked the saved cross-dataset figure values. HEEDB independent physician-code core TWA was independently parsed as 1,456/10,000; 18 arm-condition-seed rows of AUPRC, AUROC, sensitivity, specificity, predicted-positive fraction, mean score, Brier and ECE15 matched saved aggregates. |
| Uncertainty | Recomputed 1,000 saved synchronized resamples for Chapman/Ningbo baseline and mitigation summaries. Independently regenerated 1,000 HEEDB patient-bootstrap paired contrasts for independent-confirmation AUPRC/sensitivity/specificity and Huang AUPRC/AUROC/sensitivity/specificity. Values matched reported estimates and percentile intervals within artifact rounding. |
| Huang detector and aggregates | Detector calls matched 9,701/10,000 clean and 9,919/10,000 simulated RA-LL; five independent and five Huang aggregate artifact hashes passed. |
| Figures | `FIGURE_DATA.csv` has 64 aggregate-only rows, traceable to the checked cross-dataset and Huang summary/interval CSVs. `FIGURE_MANIFEST.json` hashes each source, generator and rendered SVG/PNG. No waveform or record-level row enters figure data. |

| Frozen protocol | SHA-256 |
|---|---|
| Chapman baseline | `312dabb9a0400675097d492f66df58c9da5ead69623fff0563f4daece0944cb7` |
| Chapman M1 | `248976f691003daa45c5f188c53f7cf75891f19f8723223477fc41e1d65667ce` |
| Ningbo baseline | `1fe829e4357b4efde0e5f227baf5bf94e8d22d12af0959a38cee12cce1f40d71` |
| Ningbo M2 | `2d7b605af175ef6d5b59b223a9ea37bb3f7d675c2c1ac3e27fff23fcd43e96b8` |
| HEEDB independent confirmation | `199344e9f7cbb0eb55564f52ce320bca4c85478ceb68174a3d11568611a58760` |
| HEEDB Huang exploratory | `50974d7cdb7a286d7d3cd4831b29df2a80edaac31a274afd697d931bb76cd53c` |

Verified primary numbers: Chapman baseline mask-minus-RA-LL TWA AUPRC 95% CI [0.2096, 0.2796], Chapman mitigation reduction [0.2771, 0.3508]; Ningbo baseline mask-minus-RA-LL [0.3113, 0.3503], Ningbo mitigation reduction [0.4348, 0.4839]. Independent HEEDB physics-aware-minus-baseline exact simulated RA-LL TWA AUPRC +0.4036 [0.3785, 0.4255]. Exploratory Huang contrast (physics-aware minus Huang) -0.0128 [-0.0280, +0.0019]. These are distinct cohorts/analyses; HEEDB cohorts were not pooled.

The figure generator is [submission_figures_v1.py](../../src/submission_figures_v1.py), SHA-256 `3419e30e8ea148eb4b6ab11c56270cc9565d90c0337eed20c7f1d95be4ae2f82`. This report verifies arithmetic/provenance, not the clinical validity of physician-code labels or real-world acquisition-error frequency.

An independent read-only [package check](../../src/submission_package_check_v1.py), SHA-256 `79b541d2a2847c3707fabbe2edbebadc222f874b86c7dfcc4862eec3f9778bd3`, also passed: all 64 published figure-data values and intervals matched the frozen aggregate CSV rows exactly (within floating-point serialization), and three source hashes and seven generated-artifact hashes matched the figure manifest. The output schema contains no patient or recording identifier.
