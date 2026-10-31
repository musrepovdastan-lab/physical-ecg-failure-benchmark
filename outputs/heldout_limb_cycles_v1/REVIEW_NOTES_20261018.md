# Independent review of held-out cycle results

Review date: 2026-10-18. This note was added after the original package manifest. No training, GPU inference or change to the historical package was performed during this review.

## Checks completed

- All 84 original output-file hashes matched the saved manifest before importing the adapter; bytecode writing was disabled for the review.
- The canonical frozen-protocol fingerprint was independently reproduced.
- Both cycle adapters matched independently written displayed-lead formulas exactly after float32 conversion on 128 real Ningbo waveforms; chest channels were unchanged.
- Checkpoint, preprocessing, gate and validation-artifact hashes were checked for all 30 model units. All 60 new prediction files had matching condition, checkpoint, protocol, label and threshold bindings.
- All 360 metric rows were independently recalculated: 2,520 values covering AP, AUROC, Brier, ECE15, sensitivity, specificity and predicted-positive fraction.
- All 192 saved bootstrap contrast groups contained 1,000 draws; their means and percentile intervals matched the summary CSV.
- All 192 full-cohort contrast points were independently reproduced. For draws 0, 417 and 999, all contrasts were recomputed using ordinary sklearn on repeated records: 576 checks passed.
- All 12 chest-only cycle predictions matched their corresponding historical clean predictions exactly.
- The saved replay audit reports 30 model units / 60 condition checks with zero probability difference. GPU replay itself was not rerun during this review.

## Interpretation

Physics-aware and combined improve TWA ranking relative to ordinary training on both new cycles in both previously accessed cohorts. This supports transfer to these two unseen compositions within the trained fault family. It does not establish that the network learned a particular physical mechanism or that performance transfers to real acquisition errors.

Physics-aware exceeds combined on both cycles in both cohorts, with all four conditional AP intervals excluding zero. Reversal exposure remains confounded: 50% for physics-aware versus 25% for combined. The next planned probability study is relevant to this unresolved question.

Chest-only remains competitive. On Chapman cycle_slots_120, physics-aware minus chest-only AP is -0.0252 with interval [-0.0409, -0.0105]. For the other three cohort/cycle comparisons, the corresponding intervals cross zero. Do not claim universal superiority over chest-only.

The engineering report records a nonzero identity-reconstruction difference against historical displayed signals (maximum 3.5 raw units; RMSE about 0.89). The cycle adapter rederives limb leads from I/II, while historical clean inputs remain untouched. The reported clean-to-fault changes therefore use the established historical signal path; the identity audit should remain visible. This review did not independently measure model sensitivity to identity rederivation.

The training-transform audit supports that a single training view did not apply multiple swaps. It remains an audit of the retained implementations and protocols, not proof about every unretained historical execution.

## Next step

Freeze the bounded 10%/25% reversal-training study before new training outcomes. Compare 25%-only reversal training with combined at the same nominal reversal exposure, and compare the probability grid without test-based recipe changes. Keep the current cycle package as a source of verified transforms, cohort bindings and cached comparator predictions.
