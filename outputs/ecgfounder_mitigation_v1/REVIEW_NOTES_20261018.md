# Independent review: Founder mitigation engineering stop

Date: 2026-10-18. No head training, new test predictions or diagnostic metric results exist in this package. The engineering pilot passed, but the extraction run stopped under the previous prompt's six-hour projected resource ceiling. This is not a negative mitigation result.

## Independently checked

- Canonical frozen protocol fingerprint reproduces exactly.
- All 42 frozen source/artifact hashes match.
- The 32-mask panel reproduces exactly from the specified RNG seed, including the exact affected-lead mask.
- Reused Chapman train and validation clean feature copies match their recorded source hashes and dimensions. Their committed values are finite.
- The partial Chapman RA_LA training feature cache contains 2,464 committed finite rows, shape 7,163 by 1,024, with the expected checkpoint/protocol/order binding. The uncommitted suffix is not valid output.
- No heads or predictions directories exist. The engineering report's assertions about GPU feature replay and unchanged encoder state were inspected, not independently rerun by this reviewer.

Details are in `INDEPENDENT_REVIEW_AUDIT_20261018.json`, a new review artifact outside the original frozen source set.

## Critical implementation fix required before test inference

`run_mitigation.py:548`, inside `make_test_predictions`, calls:

`predict(head, features, np.zeros(1024, np.float32), np.ones(1024, np.float32))`

The head is trained and validated on `(features - clean_train_mean) / clean_train_std`. Passing zeros and ones at test time bypasses that standardization. The current head weights have not folded in the scaler. This would evaluate a different input mapping and invalidate the intended comparison. Load each cohort's saved scaler and use it for every head/condition exactly as in validation. Bind it to test predictions and add a train/validation-only round-trip check through the actual test prediction path before any test outcomes.

## Resume fix required before a sustained run

`train_unit` loads an early-stopped resume with `stale_epochs >= 5`, then enters the loop at `completed_epoch + 1` if fewer than 50 epochs were completed. Rerunning `--stage all` could therefore train an already-finished head for another epoch. A completed valid head must be verified and reused without an optimizer step; a resume that has reached its stopping rule must finalize the stored best state without another epoch. Bind epoch history and optimizer/current/best state consistently. No heads have yet been trained, so there are no head outcomes to recover.

## Resource interpretation and continuation

The six-hour ceiling came from the preceding assistant prompt; it was not a separately stated user hardware limit. Do not describe it as an independently user-specified constraint.

The approximately 8.67-hour estimate is projected end-to-end single-device pipeline wall time at the first live extraction rate. It is not measured CUDA-active time. The pilot's roughly 51 records/s excluded portions of the production I/O/hash/write path and should not be used as production throughput without qualification. No reliable final completion time has yet been measured across all condition types.

An execution amendment can preserve all scientific choices while extending the operational budget and addressing the two implementation bugs. Preserve the old protocol, runner and stop report, and record exact code changes and cache lineage before continuation. Resume the committed prefix rather than discarding it.

A numerically checked optimization is possible for masks: official clean preprocessing is identical across all 32 masks, since masking happens afterward. Preprocess a record block once and zero the appropriate channels for each mask, preserving each condition's original 16-record encoder batch context, record order and precision. Reuse one frozen encoder instance. This reduces repeated preprocessing and I/O without changing the intended inputs. Check equivalence using train/validation records before applying the optimized path to test.

No architecture-generality or novelty claim changes until the actual clean-versus-augmented head comparison is completed.
