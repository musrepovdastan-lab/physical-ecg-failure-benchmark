# Incidents and telemetry audit

## M2 validation writer

The first M2 run stopped before validation-gate creation because the atomic NPZ helper path parameter was named `p`, colliding with the legitimate prediction-array keyword `p`. The helper parameter was renamed, a save/reload regression test for `y`, `p` and `thresholds` passed, and seed 104729 was recovered by validation-only inference from its existing best checkpoint. No training epoch or test inference was repeated.

## M2 finalizer lock and ECE implementation

The trainer first invoked the finalizer while retaining its lock; the finalizer correctly rejected overlap. The hand-off now releases the trainer lock first and has a regression test. A separate ECE generator-scoping error stopped the first finalizer attempt before bootstrap; the imperative 15-bin implementation has a regression test. These were orchestration/implementation fixes, not changes to scientific definitions.

## Invariants verified after completion

- Frozen M2 protocol SHA-256: `2D7B605AF175EF6D5B59B223A9EA37BB3F7D675C2C1AC3E27FFF23FCD43E96B8`.
- Frozen Ningbo bootstrap-index SHA-256: `6D186026B73C6A0AD31DF3C20F16C6F18AD9970176E9C78B0235E6A64E1861D1`.
- Current bootstrap-state SHA-256: `8DE562E38F9AB637640D491EFAEF67B323B2DFBA54FCC210FEE5C37C8092C891`; 1,000/1,000 replicates complete.
- All 2,382 saved M2 prediction files still match their per-seed evaluation manifests and pass shape/finite checks.
- Endpoints, thresholds, 794-condition definition, model predictions and bootstrap definitions were not recomputed or changed.

The only mutable historical record updated in this phase is `WORKFLOW_STATUS.json`, after binding its `bootstrap_completed: 1000` field to the existing bootstrap-state hash. Future finalizer COMPLETE writes now include this same telemetry field.
