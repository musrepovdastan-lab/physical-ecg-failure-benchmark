from pathlib import Path

out = Path(__file__).resolve().parent
report = """# Ningbo mask semantics recovery: final report

This is a corrected reanalysis after prior access to the Ningbo test set. It is not an untouched or independent confirmation.

Protocol binding: `b3c050887f99b2979ef20122374552298185e5a8f8004a769650783c1b77b28a`.
Engineering integrity: **PASS**, covering six model/seed units and all 4,752 corrected mask predictions.

## Main results

For baseline TWA, mean five-lead removal AUPRC exceeded RA-LL AUPRC by `0.4084`, `0.4006`, and `0.4035` across the three seeds (mean `0.4042`; synchronized canonical-record bootstrap 95% CI `0.3809` to `0.4257`). The frozen Ningbo confirmation criterion is **CONFIRMED_GENERAL_FAILURE**.

The exact five-lead mask also exceeded RA-LL in every baseline seed: contrasts `0.2200`, `0.2846`, and `0.2564` (mean `0.2536`; bootstrap 95% CI `0.2316` to `0.2778`). Thus the exact-mask gap persists after correcting the mask semantics.

For M2, mean five-lead mask minus RA-LL contrasts were `-0.0652`, `-0.0719`, and `-0.0476` (mean `-0.0616`; bootstrap 95% CI `-0.0707` to `-0.0526`). The exact-mask mean contrast was `-0.0405`. M2 therefore reverses the baseline ordering: RA-LL AUPRC is above the mean and exact-mask AUPRC. Under the unchanged frozen gates, M2 **PASS_M2_EXTERNAL_REPLICATION**.

M2 raised RA-LL TWA AUPRC by `0.4836` on average versus baseline (paired bootstrap 95% CI `0.4564` to `0.5090`). Mean mask TWA AUPRC changed by `+0.0188` (CI `+0.0091` to `+0.0282`); clean TWA changed by `+0.0107` (CI `-0.0009` to `+0.0225`). The mean mask-minus-RA-LL contrast fell by `0.4649` (reduction CI `0.4400` to `0.4891`).

## Historical claims

Preserve the direct clean and RA-LL observations, their frozen thresholds, and the conclusion that the clean-trained baseline is highly vulnerable to RA-LL. The corrected analysis also supports the stronger mean-mask and exact-mask comparisons described above.

Replace prior mask means, exact-mask scores, mask-minus-reversal intervals, and mask-dependent PASS/FAIL claims with this recovery package. The three historical baseline exact-mask scores average about `0.5275`; the seed 104729 legacy score matches an inverted-evaluator replay exactly for that single condition. Provenance for the other historical exact-mask entries remains unresolved, and the one forensic match does not validate the other 791 masks. Historical M2 mask scores also remain unresolved. See `OLD_VS_CORRECTED.csv`.

No training or data acquisition occurred. Clean/RA-LL caches were checked against model, record-order, completion, and probe bindings; prior mask caches were not reused. Because the Ningbo test set had already been accessed, report this as a corrected reanalysis, not an untouched confirmation.

## Next step: Transformer M3

Apply the same versioned evaluator and cache contract to M3's six existing Chapman checkpoints in a separate recovery package. Validate model, preprocessing, and record-order bindings; regenerate the 792-mask panel; recompute Stage M3A criteria and downstream figures. Do not retrain M3 unless a separate audit identifies a training-path defect.
"""
(out / "FINAL_RECOVERY_REPORT.md").write_text(report, encoding="utf-8")

pilot = out / "PILOT_REPORT.md"
if pilot.is_file():
    pilot_report = """# Ningbo mask recovery pilot

Decision: **PASS**

The frozen 512-record clean and RA-LL probe used `atol=1e-5`, `rtol=1e-4`. Ten of twelve saved-score checks passed within the unchanged tolerance. The two failed model/seed checks had their complete clean and RA-LL arrays re-inferred in the recovery cache.

Five-mask inference took 4.38 seconds (0.88 seconds per mask). The estimated 4,752-mask inference time was 1.16 hours. The exact weighted-bootstrap benchmark took 3.72 seconds for 20 resamples, 792 masks, and one endpoint; the initial full-statistics estimate was 0.93 hours.

Peak CUDA allocation was 92.9 MiB; the raw test matrix used 319.6 MiB. The baseline seed 104729 forensic inverted exact replay matched the historical cache exactly, for that single condition only.

No model training or data acquisition occurred.
"""
    pilot.write_text(pilot_report, encoding="utf-8")
