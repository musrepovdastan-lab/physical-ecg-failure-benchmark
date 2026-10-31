# Audit of lead-mask semantics and result provenance

Audit scope: L0 only. No training, data download, full inference, bootstrap, checkpoint, prediction, protocol, source file, or historical report was changed. Audit outputs are isolated in this directory. No applicable `AGENTS.md` was found in the project directory or its parent directories. The controlling plan is `RESEARCH_PLAN_AFTER_MENTOR_20261016.md`; its Priority 0 calls for this audit and cautions that current source is not proof of historical execution.

## Finding

Current evaluators in Ningbo confirmation v1, Ningbo M2, and Transformer M3 use `np.where(mask, x, mean)`. Here `mask` denotes the five enumerated indices, so the expression keeps those five signals and replaces the complement of seven with train mean. The label `mask5_i_j_k_l_m` therefore does not describe the current transformation in these evaluator paths. Under canonical order I, II, III, aVR, aVL, aVF, V1–V6, RA–LL indices `[0,1,2,3,5]` keep I, II, III, aVR, aVF and suppress aVL plus V1–V6. This reverses the intended missingness condition.

Chapman v7 and M1 have the intended current branch: selected indices are replaced by train mean before per-channel normalization. That sentinel maps to zero. All of these paths normalize after the replacement using train-derived per-lead mean/std. Static inspection establishes current source semantics; it does not establish historical execution unless a frozen run manifest binds the source.

## Current implementation and training scope

| Workflow | Current evaluation | Training path | Historical evidence |
|---|---|---|---|
| Ningbo confirmation v1/v2 | Inverted in v1 `evaluate`; v2 delegates to it | Clean-trained baseline; no mask augmentation | v1 stopped before split/training/evaluation; v2 predictions have no evaluator-code hash |
| Chapman v6/matched reversal | Correct in current source | Clean baseline | Saved outputs are not source-hash-bound |
| Chapman v7 | Correct in current source | Clean baseline | New seeds 130363/155921 bind evaluator hash; inherited seed 104729 comes from v6 |
| Chapman M1 | Correct in current source | Masking and combined arms replace five selected leads in training; physics-aware arm uses reversals | Historical training/evaluator source hash absent |
| Ningbo M2 | Inverted in current evaluation | Physics-aware reversal augmentation only | Evaluation manifests lack evaluator hash |
| Transformer M3 | Inverted in current evaluation | Clean or physics-aware reversal augmentation only | Evaluation manifests lack evaluator hash |

Evidence supports an evaluation-only issue for Ningbo baseline, M2, and M3. It does not support retraining these models. M1's current masking and combined training branches have correct semantics, though historical source provenance is incomplete. Clean and RA–LL branches do not use the faulty mask expression and can be retained if their independent integrity checks pass.

## Historical provenance and status

Statuses in `AFFECTED_ARTIFACTS.csv` apply to historical artifacts. Ningbo v1 is `NOT_APPLICABLE`: its conclusion says `STOP_DUPLICATE_CONFLICT` and explicitly records no split, training, or test evaluation. Ningbo v2 freezes bind protocol, split, checkpoint, physics, test membership, and condition definition; completion manifests bind prediction hashes, but none bind evaluator source. Current v1 is demonstrably inverted, but historical semantics remain `UNRESOLVED_PROVENANCE`.

The exact-mask Ningbo TWA AUPRC rows in `CONDITION_RESULTS_BY_SEED.csv` are 0.5185, 0.5290, and 0.5349 (mean ≈0.5275). The current evaluator's exact condition would preserve the five listed leads and suppress aVL/V1–V6. Since the stored run is not linked to source, the earlier ≈0.527 value is **not fit to cite as removal of the five RA–LL leads** until recomputed.

M2's completion files bind prediction hashes/checkpoints but not evaluator code. Its `PASS_M2_EXTERNAL_REPLICATION` consumes mask contrasts and noninferiority checks, so the decision must be suspended. M2 training is reversal-only. M3 manifests similarly omit evaluator code; its training is clean/reversal-only. Neither is evidence for retraining.

Chapman v7 freezes for seeds 130363 and 155921 include `evaluator_code_sha256` that matches the current v7 source; the branch replaces selected leads with mean before normalization. Their panels are `VERIFIED_CORRECT`. The aggregate reuses seed 104729 from v6, whose current code is correct but whose historical output lacks evaluator hash. The complete three-seed aggregate remains `UNRESOLVED_PROVENANCE` until that panel is verified or regenerated. M1 current source is correct, but its historical freezes omit source hashes, so stored panels remain unresolved.

No historical panel is assigned `VERIFIED_INVERTED` because none of the artifacts reviewed binds its generating process to the inverted evaluator source. The inversion itself is confirmed in the current evaluator source for the three affected code paths. Protocol hashes, timestamps, condition counts, prediction hashes, and checkpoint hashes establish identity/integrity, not mask branch order. See the CSV for artifact-level status and action.

## Synthetic check

`synthetic_mask_check.py` exercises the evaluator expressions on a synthetic `(12 leads, 4 samples)` array with distinct channel values and nonzero per-lead means/stds. It records source hashes. It is an expression-level test because importing the full evaluator entails model/data dependencies and inference; the relevant `np.where` branch is reproduced directly and the limitation is recorded in the JSON result.

The check asserts that the inverted expression changes `[4,6,7,8,9,10,11]` (aVL and V1–V6), while the intended expression changes `[0,1,2,3,5]`. Selected signals are retained in the inverted branch; the other seven remain unchanged in the intended branch. Mean sentinels normalize to zero. The panel contains exactly 792 unique combinations of five distinct indices.

## Impact

Clean and simulated RA–LL scores are outside the mask branch and remain provisionally usable subject to their existing integrity/provenance controls. HEEDB clean and RA–LL values likewise do not depend on this evaluator. Direct clean-to-reversal contrasts remain usable on the same basis.

Do not rely on exact-mask scores, individual mask scores, panel means/distributions, mask-minus-reversal contrasts, their bootstrap intervals, or decisions consuming these contrasts as evidence about removal of five leads. This affects Ningbo v2 `CONFIRMED_GENERAL_FAILURE`, M2 `PASS_M2_EXTERNAL_REPLICATION`, Chapman v6/v7 aggregates, M1 mask outcomes, M3 mask outcomes, and manuscript plots/tables/claims using the comparator. Clean and RA–LL cells from these analyses can be kept separately.

The current evidence does not establish that reversal damage exceeds the impact of removing five selected leads. Clean-versus-RA–LL may still support a physical-reversal failure for clean-trained models, but the comparative missingness claim and associated gates await corrected evaluation.

## Dependency path

`checkpoint + preprocessing` → `condition predictions (clean / RA–LL / 792 masks)` → `condition metrics` → `panel mean/distribution` → `mask minus reversal contrast` → `paired bootstrap intervals` → `PASS/FAIL gate` → `summary tables, figures, manuscript claims`.

For M2, both baseline and M2 mask panels feed contrast-reduction gates, so both need source verification or regeneration. For baseline v2, Chapman, M1 and M3, regenerate mask metrics and all downstream panel contrasts. Clean and RA–LL predictions, metrics, thresholds, and direct clean-to-reversal comparisons do not need recalculation solely because of this mask issue. Mask-derived operating-point metrics and plots do.

## Next step

See `MINIMAL_RECOVERY_PLAN.md`. L1 should use a separately versioned evaluator/output root, invalidate all mask caches by semantic version and source hash, and reuse validated checkpoints. Keep historical results immutable. Do not infer that seven suppressed leads can be repaired by renaming files.
