# Reproducibility audit (corrected)

This correction used cached JSON, CSV, NPZ and checkpoint/prediction files only. It did not load raw ECGs, invoke a model, run inference, evaluate masks, or rerun bootstrap.

| Check | Chapman v7 | Ningbo v2 |
|---|---:|---:|
| Frozen protocol SHA-256 matches recorded conclusion | PASS | PASS |
| All three baseline gates passed | PASS | PASS |
| Test not accessed before frozen evaluation specification | PASS | PASS |
| Exactly 1,000 valid synchronized bootstrap replicates | PASS | PASS |

Ningbo's three checkpoints, three evaluation manifests, and 2,382 prediction files match their recorded hashes: **PASS**. Ningbo `WORKFLOW_STATUS.json` remains at `BOOTSTRAP` with 1,000/1,000 complete; it is stale telemetry only, not evidence against `CONFIRMED_GENERAL_FAILURE`.

## Corrected Chapman pre-test evidence

| Seed | Baseline source | Frozen specification source | Baseline gate | Test not accessed before evaluation freeze |
|---:|---|---|---:|---:|
| 104729 | `outputs/phase1_chapman_seed1_v4/BASELINE_ADEQUACY_v4.json` | `outputs/phase1_chapman_seed1_v4/FAILURE_ANALYSIS_FREEZE_v4.json` | PASS | PASS |
| 130363 | `outputs/phase2_chapman_multiseed_v7/seed_130363/baseline.json` | `outputs/phase2_chapman_multiseed_v7/seed_130363/pretest_freeze.json` | PASS | PASS |
| 155921 | `outputs/phase2_chapman_multiseed_v7/seed_155921/baseline.json` | `outputs/phase2_chapman_multiseed_v7/seed_155921/pretest_freeze.json` | PASS | PASS |

The prior field name was semantically inverted. The corrected machine-readable field is `test_not_accessed_before_evaluation_freeze`; it is based on each source's explicit pre-evaluation field, never on a later post-evaluation `test_accessed:true` record.
