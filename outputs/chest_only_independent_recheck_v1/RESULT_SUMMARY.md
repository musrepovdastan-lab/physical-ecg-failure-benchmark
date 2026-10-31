# Independent chest-only comparison recheck

Date: 2026-10-18. Existing predictions, three seeds, TWA AP. No training or inference. All six 1,000-draw conditional paired-bootstrap intervals were independently recalculated and matched the historical results.

| Dataset | Condition | Physics-aware AP | Chest-only AP | Difference | 95% interval |
|---|---|---:|---:|---:|---|
| Chapman | clean | 0.6641 | 0.6503 | +0.0137 | [-0.0017, +0.0299] |
| Chapman | RA_LL_reversal | 0.6372 | 0.6503 | -0.0131 | [-0.0282, +0.0008] |
| Ningbo | clean | 0.6869 | 0.6554 | +0.0315 | [+0.0178, +0.0438] |
| Ningbo | RA_LL_reversal | 0.6665 | 0.6554 | +0.0111 | [-0.0014, +0.0231] |
| HEEDB | clean | 0.5760 | 0.5367 | +0.0393 | [+0.0279, +0.0513] |
| HEEDB | RA_LL_reversal | 0.5683 | 0.5367 | +0.0316 | [+0.0202, +0.0434] |

An interval crossing zero does not establish equality. Under RA-LL, an AP advantage over chest-only is unresolved in Chapman/Ningbo and supported on the previously accessed HEEDB proxy cohort. Positive evidence is limited to the endpoint, models, cohort and conditional uncertainty evaluated.

See SEED_LEVEL_RECHECK.csv, PAIRED_RECHECK.csv, BOOTSTRAP_RECHECK_DRAWS.npz and AUDIT.json.
