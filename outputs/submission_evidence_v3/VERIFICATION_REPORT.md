# Manuscript publication verification

1 November 2026. This verifies an editorial/aggregate export update, not another experiment.

| Check | Result |
|---|---|
| Corrected-series aggregate export | 336 TWA seed rows and 112 mean/SD rows agree with the completed source tables. |
| Paired contrasts | All 700 point contrasts agree with means of the paired seed-specific metric differences; maximum discrepancy 5.69e-14 in the verification run. The exported CI table retains exact original source bytes. |
| Key interpretation | All four p050/combined-minus-Huang AP intervals at equal-three q=0.1 include zero; all four corresponding augmentation-minus-ordinary AP intervals are positive. |
| Local regression | 45 passed, 1 platform skip; 15 subtests passed. |
| Public copy regression | 39 passed, 4 skips for omitted external source/platform support; 15 subtests passed. |
| Public-only manuscript verifier | Passes using included aggregate CSVs without ECG data, weights or individual predictions. |
| Documentation input validation | Supported `train_all_fresh.py --check-inputs` and direct provenance check pass for 65 scientific inputs, frozen runner/protocol and both historical/current documentation versions. |
| Historical preservation | Manuscript v2 and original corrected-series scientific outputs retain the pre-update hashes. Only explicit documentation/provenance/manifest status files changed. |
| Publication selection | Explicit allowlist; no raw ECGs, individual exports, membership contents, weights, caches, resample arrays or third-party payloads added. |
| Figure | PNG inspected for readable axes, titles, legend and footnote; SVG provided for typesetting. Lines are descriptive means, not significance claims. |

The first regression attempts used an inaccessible host temporary location and failed four temporary-directory tests. Directing TEMP/TMP to writable workspace folders resolved the failures without changes to tests or scientific code. The local and clean-copy runs used the existing Python 3.12 research environment; dependencies were not freshly installed.

The original nine complete trajectories were not independently repeated during this work. No scientific training, inference, calibrator fit, threshold selection or bootstrap generation was performed for the manuscript. Table/figure builders use only completed aggregates. The manuscript remains a working author-review draft.

Commands: `python outputs/submission_evidence_v3/verify_package.py` verifies the distributed synthesis; `build_tables.py` re-exports tables; `build_figure.py` rebuilds the descriptive figure from those tables. The full research prerequisites remain separate from this public aggregate check.
