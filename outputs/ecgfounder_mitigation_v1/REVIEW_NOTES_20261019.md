# Independent review: completed Founder mitigation

Review date: 2026-10-19. Status: accepted as an exploratory frozen-encoder/head-training result. This note and its companion JSON were added after the results manifest; original scientific artifacts were preserved.

## Completed independently

- Verified 338 recorded hashes: result/source files, preserved artifacts, 12 best checkpoints, 76 test prediction bundles and metadata, and 76 test feature caches and metadata. The original method protocol fingerprint matches.
- Recomputed all 1,368 endpoint/seed/condition metric rows: 9,576 AP, AUROC, Brier, ECE15 and operating-point values. Maximum differences are below 5e-13, consistent with CSV rounding.
- Verified all canonical test IDs and labels, six-model order, protocol and validation-threshold bindings in all 76 bundles.
- Independently reproduced all 36 thresholds from best-checkpoint clean validation negatives. Validation TWA AP/AUROC match training results exactly; best epochs match the earliest maximum in each history, and stopping histories are consistent with patience 5.
- Re-derived both 1,024-dimensional feature scalers from the clean train embedding arrays exactly.
- Independently loaded all 12 trained linear heads on CPU. Using NumPy/scipy prediction on cached embeddings and the proper scaler, reproduced 12 validation samples and 108 test head/condition samples, 32 records per sample. Maximum validation difference 5.365e-7; maximum test difference 9.537e-7, below the preselected CPU/CUDA FP32 arithmetic tolerance of 1e-5. No encoder or GPU inference was required for this check.
- Recomputed all 44 full-cohort bootstrap contrast points; maximum difference below 4.875e-13.
- Independently calculated 132 actual bootstrap contrast spots using sklearn on repeated records (draws 0, 417 and 999), including the full 32-mask panel. Compared those values with the runner's optimized bootstrap algorithm; maximum difference below 8.882e-16.

Limits: the encoder GPU replay was not rerun independently. Full 1,000-draw percentile intervals were not all recomputed in this review; their implementation was inspected and checked against real resample contrasts as described above. See `INDEPENDENT_RESULT_REVIEW_20261019.json` for exact counts and limits.

## Resolved implementation issues

The continuation prediction path now uses the correct cohort clean-train scaler exactly once. Completed heads are verified/reused without further optimizer steps, and stopped resumes finalize the best checkpoint without another training epoch. The original runner/stop report remain preserved and bound by the amendment. The previous engineering stop is superseded by actual completed results, not rewritten as though it never happened.

## Scientific interpretation

Mean TWA AP, clean-trained versus augmented heads:

| Cohort | Clean inputs | RA-LL | Cycle 120 | Cycle 201 |
| --- | --- | --- | --- | --- |
| Chapman | 0.655735 -> 0.653585 | 0.335416 -> 0.518864 | 0.283125 -> 0.509356 | 0.520276 -> 0.586592 |
| Ningbo | 0.633513 -> 0.621858 | 0.247365 -> 0.459847 | 0.243066 -> 0.459014 | 0.380989 -> 0.522708 |

The reported RA-LL and cycle AP-gain intervals are positive in both cohorts. This adds evidence that augmentation can help a classifier on substantially different pretrained ECG features, beyond the original small ResNet. It is not full Founder fine-tuning, independent encoder replication, a claim that Founder is the best architecture, or universal robustness.

Preserve the limitations:

- Clean Ningbo AP decreases by about 0.01166, with a reported interval excluding zero; clean Chapman is uncertain.
- RA-LA benefits occur in both sources, but LA-LL benefit is not established in Ningbo.
- Exact-mask gains occur in both sources; the average 32-mask effect is uncertain in both. This is not a 792-mask result and does not establish broad missing-channel robustness.
- Augmented reversal AP remains below the same head's clean-input AP; robustness is not complete recovery.
- The mixed/failed Transformer M3 gate remains part of the evidence.
- Previously examined cohorts, one pretrained encoder, label/proxy limitations and pretraining-overlap uncertainty remain relevant.

## Next bounded work

The planned Founder control is complete. Do not retrain it merely to obtain a cleaner result. A remaining part of the original probability/calibration plan is external evaluation of the already frozen p010/p025 Ningbo ResNet models, using their recovered validation-only calibration on the existing 10,000-patient HEEDB cohort. Reuse historical comparators, include both new probabilities without selecting between them, and disclose that HEEDB has already been examined. This needs no new model training or cohort acquisition.

After that external check, consolidate the mentor-response evidence and outstanding clinician decisions. Noise/contact stress tests and real miswiring recordings are optional strengthening work, not automatic prerequisites for a simulation-scoped paper.
