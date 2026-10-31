# Endpoint amendment v3 — frozen before model outcomes

## Change

The v2 prevalence-selected co-primary endpoints, sinus bradycardia and sinus tachycardia, are moved to prespecified **rate-based negative controls**. They remain scientifically useful: a morphology-specific acquisition-fault effect should be stronger for the primary morphology-sensitive endpoint than for either simple rate-recognition endpoint.

The new sole primary endpoint is **T-wave abnormality** (SNOMED CT `164934002`): 1,876 Chapman–Shaoxing and 5,167 Ningbo positives in the official Challenge 2021 source-wise label table. It passed the original prevalence gates but had been secondary only.

## Fixed roles

| Role | Endpoint | SNOMED CT |
|---|---|---|
| Primary morphology-sensitive | T-wave abnormality | 164934002 |
| Negative control | Sinus bradycardia | 426177001 |
| Negative control | Sinus tachycardia | 427084000 |

AF and individual bundle/conduction labels remain rejected because they failed the previously frozen shared-source count criteria. Rate controls are not failed primary endpoints and will not be treated as such.

The SHA-256 of `configs/FROZEN_PHASE1_PILOT_PROTOCOL_v3.json` is recorded in `ENVIRONMENT_AUDIT.json` before any model prediction is inspected.
