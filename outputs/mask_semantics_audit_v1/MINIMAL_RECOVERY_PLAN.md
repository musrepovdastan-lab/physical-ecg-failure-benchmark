# Minimal recovery plan for L1

L0 found an evaluation semantic defect in current Ningbo v1 delegated evaluation, Ningbo M2, and Transformer M3. Historical provenance is incomplete for most saved panels, so preserve old outputs and produce corrected results in a separate version. No training is indicated by this evidence.

## Evaluator contract and versioning

Create a new evaluator version and output root such as `outputs/mask_semantics_recovery_v2/`; do not overwrite historical outputs. Define every `mask5_*` set as channels to replace by train mean. Assert that `mask5_0_1_2_3_5` maps to I, II, III, aVR, aVF. Replace in raw/display units before normalization; require the sentinel to normalize to zero and all other seven signals to remain bitwise unchanged.

Cache keys must bind evaluator semantic version and source SHA-256, protocol hash, split/test membership hash, checkpoint hash, preprocessing mean/std hashes, condition name/indices, canonical lead order, and normalization stage. Old caches must fail closed even if shapes, timestamps, checkpoint hash, and prediction hash look valid. Freeze these bindings before inference and copy them to each completion manifest.

## Reuse and regenerate

| Experiment | Reuse | Regenerate | Recompute downstream |
|---|---|---|---|
| Ningbo v2 baseline | Three checkpoints, preprocessing, validation thresholds, splits/bootstrap indices, clean and RA–LL predictions | 792 masks × 3 seeds | Per-mask metrics; panel mean/distribution; mask minus RA–LL and synchronized bootstrap; confirmation decision |
| Chapman baseline v6/v7 | Checkpoints, preprocessing, clean/reversal predictions; verified v7 panels for seeds 130363/155921 | Verify inherited v6 seed 104729 source; otherwise 792 masks × 1 | Three-seed aggregate panel metrics/contrasts/intervals/gate and dependent plots/claims |
| Chapman M1 (physics-aware, masking, combined) | Nine checkpoints (3 arms × 3 seeds), preprocessing, validation artifacts/thresholds, clean/reversal predictions | 792 masks × each arm/seed reported, unless frozen source hash proves panel semantics | Per-condition metrics, panel distribution, mask contrasts/bootstrap, decisions, tables and plots |
| Ningbo M2 | Three physics-aware checkpoints, preprocessing, thresholds, split/bootstrap indices, clean/reversal outputs | 792 masks × 3 M2 seeds; share corrected v2 baseline panels above | Baseline/M2 panels, matched contrasts/reductions, intervals and every mask-dependent gate/PASS decision |
| Transformer M3 Chapman | Six checkpoints (clean/physics-aware × 3 seeds), preprocessing, validation artifacts, clean/reversal predictions | 792 masks × 6 model units | Per-condition metrics, panel means/distributions, contrasts/intervals, Stage M3A decision and dependent figures/tables |

M2/M3 use reversal-only physics-aware augmentation, so their checkpoints can be reused. Ningbo v2 baseline and Chapman baselines are clean-trained. M1 masking and combined models were trained with correct replacement in current source, but confirm source provenance before calling historical training verified.

## Recalculation and reporting

Keep clean and RA–LL prediction vectors and their direct metrics if their existing integrity checks pass. Keep thresholds, labels, splits, checkpoints and bootstrap indices. Recalculate all outputs that consume mask predictions: individual AUPRC/AUROC/sensitivity/specificity/ECE; exact-mask values; panel mean, median, quantiles and distribution; mask-minus-RA–LL contrasts; paired intervals; gates; summary tables; mask-related figures; and manuscript language. Refresh figure captions and evidence tables to identify corrected evaluator version.

Retain legacy values with a note that source linkage is unresolved. Do not rename predictions to make seven suppressed leads appear to be five. The Ningbo exact-mask AUPRC ≈0.527 must be retired as a five-lead-removal result unless corrected evaluation verifies it.

## Checks required before full recovery

1. Synthetic test for selected-five replacement, unmasked-seven preservation, zero sentinel, lead-name mapping, and 792 unique masks.
2. Synthetic evaluator smoke test captures the tensor passed to the model for the exact RA–LL mask and compares it exactly with expected normalized values.
3. Freeze manifests bind all cache inputs above; semantic-version mismatch invalidates cache.
4. Validate one synthetic record per mask and aggregate channel transformations before batched inference.
5. Check prediction hashes, expected dimensions, finite scores, 794 conditions per seed, and all 792 names exactly once.
6. Recalculate frozen decisions without changing thresholds or gates; preserve gate failures.

## Workload

Maximum new mask panels if no unresolved panel is source-verified: Chapman inherited baseline seed 104729 (1); M1 (9 arm/seed units); Ningbo baseline (3); Ningbo M2 (3 new M2 units, sharing the corrected baseline panel above); and M3 (6). That is at most 22 model/seed panels, or 17,424 model-condition evaluations. M1 panels may be reusable if an exact historical source snapshot is found. Each condition processes each test record once per model; Ningbo has 6,981 test records. Chapman record counts are in its evaluation manifests.

No reliable end-to-end timing for corrected inference, aggregation, or bootstrap was found. Run a short benchmark on one model/seed and a fixed small set of masks before scheduling the full panel. Do not extrapolate time from file counts.

L1 should first implement and validate the new evaluator contract, then run one model/seed pilot with its retained checkpoint. Start the full panel only after the pilot confirms tensor semantics, cache invalidation, runtime, and output schema. L1 is not started by this audit.
