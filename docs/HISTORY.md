# Physical ECG Failure Benchmark — Phase 0

## Current status: Phase 1 Chapman-only pilot v3 — acquisition paused

### v5 interpretation audit — mixed signal, no confirmation claim

A read-only audit of the completed v4 cached predictions found that the automatic
multiseed decision used an absolute signed contrast and that 70 nominal
independent masks collapse to 50 unique prediction outcomes. V1–V4 was less,
not more, damaging than the average comparator mask. The honest status is
`MIXED_SIGNAL_REQUIRES_NEW_CONFIRMATORY_PROTOCOL`; no multi-seed work or Ningbo
access is authorized. See `outputs/phase1_interpretation_audit_v5/`.

Phase 0 and its v2 amendment remain historical feasibility artifacts and have not
been altered. The currently frozen protocol is
`configs/FROZEN_PHASE1_PILOT_PROTOCOL_v3.json` (SHA-256
`289838FC500DDD118D664A9F3909D71476FE09CF7A43C597AD198313C0489C09`) and changes
the primary endpoint to morphology-sensitive T-wave abnormality (SNOMED
`164934002`), with sinus bradycardia and sinus tachycardia as prespecified rate
negative controls. It permits only a one-seed, Chapman-Shaoxing-only pilot.

The project-local CUDA/mixed-precision, physics-test, JSON-integrity, and disk
gates passed; see `outputs/phase1_chapman_seed1_v3/ENVIRONMENT_AUDIT.md`.
No Ningbo waveform has been downloaded, accessed, or inferred on. Official
Chapman acquisition completed (10,247 records and 20,494 SHA-256-verified files,
with no `.part` files). The strict ingestion audit returned
`STOP_DATA_INTEGRITY`: 13 pairs of distinct record IDs have exact duplicate
waveform hashes. The frozen protocol forbids silent repair or exclusion, so no
split, model training, internal-test access, or failure analysis was performed.
See `outputs/phase1_chapman_seed1_v3/DATA_INGESTION_AUDIT.md` and
`outputs/phase1_chapman_seed1_v3/FINAL_PHASE1_PILOT_DECISION.md`.

## Historical Phase 0 decision: NO-GO

**Research question.** Do evaluations based on random whole-lead masking systematically overestimate the robustness of 12-lead ECG classifiers to physically realizable electrode failures?

This is a self-contained Phase-0 feasibility, novelty, and design audit. It does not train a model, download waveforms, or access sealed confirmation outcomes. The decisive blocker is in [outputs/FINAL_PHASE0_DECISION.md](outputs/FINAL_PHASE0_DECISION.md): the proposed non-PTB replication sources in the official PhysioNet/Computing in Cardiology Challenge 2020 release do not provide authoritative patient identifiers, and the official documentation states that patient overlap cannot be ruled out. They are therefore ineligible for the required confirmatory patient-cluster inference.

The project records a rigorously specified, electrically constrained fault representation and tests the exact transformations that can be derived from a conventional 12-lead recording. It deliberately does **not** pretend that a disconnected electrode or spatial precordial misplacement can be reconstructed from a clean 12-lead ECG alone.

## Contents

- `FEASIBILITY_AUDIT.md` — gate-by-gate audit, legacy evidence, compute, and blocker.
- `RELATED_WORK_GAP.md` — 2022–2026 search and narrow novelty position.
- `DATA_MANIFEST.json` — official-source metadata-only audit; no waveforms downloaded.
- `PHYSICAL_FAULT_LIBRARY_SPEC.md` — equations, eligible transformations, and forbidden heuristics.
- `configs/FROZEN_PHASE1_PROTOCOL.json` — a contingent protocol, explicitly not authorised to run while NO-GO holds.
- `src/ecg_physics.py`, `tests/test_ecg_physics.py` — reference equations and unit tests.

## Check the mathematical implementation

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The test suite checks Einthoven and Goldberger identities, terminal-permutation transformations, and every implemented unavailable-measurement mask. It uses synthetic signals only.

## What would change the decision

Provide an official, public, non-PTB 12-lead source with authoritative patient IDs (or a documented one-record-per-patient guarantee), label-compatible endpoint counts, and no PTB-XL overlap. The protocol must then be amended and re-frozen before any outcome access.


<!-- V7_STATUS_START -->
## Chapman multiseed v7

Final frozen decision: `FREEZE_NINGBO_GENERAL_FAILURE_CONFIRMATION`. Chapman results remain exploratory; Ningbo was not accessed. See `outputs/phase2_chapman_multiseed_v7/`.
<!-- V7_STATUS_END -->

<!-- NINGBO_V1_STATUS_START -->
## Ningbo confirmation v1

The frozen post-acquisition confirmation runner is implemented in
`src/ningbo_confirmation_v1.py`. It verifies the frozen protocol and acquisition
artifacts before any waveform access, then uses a sealed three-seed validation
gate before internal-test evaluation. Its status is recorded in
`outputs/ningbo_confirmation_v1/WORKFLOW_STATUS.json`; no decision is reported
until a complete run writes `FINAL_NINGBO_CONFIRMATION_DECISION.md`.
<!-- NINGBO_V1_STATUS_END -->

<!-- NINGBO_V2_STATUS_START -->
## Ningbo confirmation v2

V1 is permanently recorded as `STOP_DUPLICATE_CONFLICT` before any split,
training, or test evaluation. The separately frozen v2 protocol applies the
prespecified amendment: it excludes every member of the three audited
metadata-conflicting duplicate waveform groups (six records) before splitting,
independent of endpoint labels. See `PROTOCOL_V2_AMENDMENT.md` and
`outputs/ningbo_confirmation_v2/`.
<!-- NINGBO_V2_STATUS_END -->

## Paper readiness v1

The completed Ningbo v2 decision is `CONFIRMED_GENERAL_FAILURE`. A read-only
cross-dataset audit verified frozen protocol bindings, all three baseline and
pre-test gates, all 2,382 Ningbo prediction hashes, and exactly 1,000 valid
synchronized bootstrap replicates. The Ningbo `WORKFLOW_STATUS.json` field
remained at `BOOTSTRAP` after finalization; its completed count is 1,000 and it
is documented as stale telemetry rather than contradictory evidence.

The publication package is in
[`outputs/paper_readiness_v1/`](outputs/paper_readiness_v1/):
[`REPRODUCIBILITY_AUDIT.md`](outputs/paper_readiness_v1/REPRODUCIBILITY_AUDIT.md),
[`CROSS_DATASET_RESULTS.csv`](outputs/paper_readiness_v1/CROSS_DATASET_RESULTS.csv),
[`MECHANISM_AUDIT.md`](outputs/paper_readiness_v1/MECHANISM_AUDIT.md),
[`CLAIMS_AND_LIMITATIONS.md`](outputs/paper_readiness_v1/CLAIMS_AND_LIMITATIONS.md),
[`WORKSHOP_READINESS_DECISION.md`](outputs/paper_readiness_v1/WORKSHOP_READINESS_DECISION.md),
and the not-yet-executed
[`NEXT_EXPERIMENT_PROTOCOL_DRAFT.md`](outputs/paper_readiness_v1/NEXT_EXPERIMENT_PROTOCOL_DRAFT.md).


<!-- M1_STATUS_START -->
## Mitigation M1

Chapman-only physics-aware mitigation is a prespecified follow-up. Final decision: `PASS_M1_FREEZE_NINGBO_M2`. See `outputs/mitigation_m1_v1/FINAL_M1_DECISION.md`; Ningbo M2 has not been started.
<!-- M1_STATUS_END -->


<!-- M2_STATUS_START -->
## Mitigation M2

Locked Ningbo external follow-up of the Chapman-frozen physics-aware method. Final decision: `PASS_M2_EXTERNAL_REPLICATION`. See `outputs/mitigation_m2_ningbo_v1/FINAL_M2_DECISION.md`.
<!-- M2_STATUS_END -->
