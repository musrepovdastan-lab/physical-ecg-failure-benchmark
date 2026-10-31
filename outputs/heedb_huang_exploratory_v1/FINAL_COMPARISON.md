# Exploratory Huang comparison on the completed HEEDB independent cohort

## Decision

The published Huang detector **can be applied without changing its weights or fitting to HEEDB**. This operational detect→predicted-correct→clean-ResNet pipeline materially repairs the clean ResNet's simulated RA–LL TWA failure. Its RA–LL TWA AUPRC is **0.5811**, versus **0.5683** for the physics-aware ResNet and **0.1647** unprotected. The paired physics-aware-minus-Huang AUPRC difference is **−0.0128** (1,000 patient-bootstrap 95% interval **−0.0280 to +0.0019**). This is an *unresolved difference*, not evidence of equivalence or Huang superiority on AUPRC. Per-seed physics-aware-minus-Huang RA–LL AUPRC changes sign (+0.0367, −0.0308, −0.0443).

Other metrics show meaningful tradeoffs: Huang has higher RA–LL AUROC, specificity and better calibration; physics-aware has higher sensitivity at its own unchanged Ningbo threshold. Neither approach is established as clinically superior. This comparison was planned **after viewing the HEEDB independent results** and is exploratory, not a new independent confirmation or an exact reproduction of Huang et al.'s paper scores.

## Fair adapter and frozen analysis

The exact same 10,000 one-ECG-per-patient independent cohort and locally cached waveforms were reused; no records were selected, replaced or downloaded. The frozen cache is canonical I, II, III, aVR, aVL, aVF, V1–V6 in microvolts, 10 seconds at 500 Hz (original 250-Hz records were already resampled). The existing Huang path applies a 500-Hz fifth-order 0.5–35 Hz SOS Butterworth filter, Fourier resampling to 1,200 samples (120 Hz), per-record/per-lead z-score, then inputs III/aVR/V6 as `(1200,3)`. Because of within-lead z-scoring, the mV→microvolt scale does not change the detector input; a scale-invariance test passed. The tests also checked channel isolation and exact equality to the existing Huang preprocessing implementation.

Both clean and faulted 500-Hz-equivalent waveforms use `src.ecg_physics.named_reversal` and `derive_display`; tests check all 12 leads, including the aVR/aVF exchange, and the real-cache adapter smoke passed on 16 waveforms. The published detector's **unmodified softmax argmax** alone chooses whether and how to invert the displayed raw ECG. The true simulated fault never chooses the deployed correction. The resulting ECG is sampled `::5` to exactly `(12,1000)`, normalized by each original baseline checkpoint's Ningbo mean/std, and passed through all three saved clean ResNets with their unchanged Ningbo validation thresholds. The two completed independent arms are read from their original hashed predictions, not re-inferred. `codes_physician` core TWA remains the sole outcome (1,456/10,000; prevalence 0.1456); this is a HEEDB 12SL physician-code **proxy**, not the exact Chapman/Ningbo SNOMED label.

The [frozen protocol](FROZEN_PROTOCOL.json) SHA-256 is `50974d7cdb7a286d7d3cd4831b29df2a80edaac31a274afd697d931bb76cd53c`; its sidecar, code/test/weight hashes and source-cache hashes bind the comparison. Three adapter tests passed **before** protocol freeze and new comparator predictions; the 16-waveform input smoke also passed. Original detector checkpoint bytes were verified against the prior published-source audit. No model, threshold, label, detector policy or sample was adapted.

## Three-seed mean TWA results

One patient per observation. AUPRC, AUROC, sensitivity, specificity, predicted-positive fraction and calibration are means of three seed-level values. Sensitivity/specificity use each arm's pre-existing Ningbo validation threshold; Huang uses the corresponding **baseline** threshold. ECE uses 15 fixed equal-width bins; lower Brier/ECE is better.

| Arm | Input | AUPRC | AUROC | Sens. | Spec. | Pred. + | Brier | ECE15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unprotected baseline | Clean | .5726 | .9031 | .6362 | .9140 | .1661 | .0994 | .0768 |
| Physics-aware | Clean | .5760 | .8927 | .6646 | .8993 | .1828 | .1152 | .1120 |
| Huang→clean ResNet | Clean | .5721 | .9025 | .6358 | .9141 | .1659 | .0997 | .0772 |
| Unprotected baseline | RA–LL | .1647 | .5383 | .6969 | .3599 | .6483 | .4917 | .5477 |
| Physics-aware | RA–LL | .5683 | .8901 | .6866 | .8908 | .1933 | .1195 | .1253 |
| Huang→clean ResNet | RA–LL | .5811 | .9066 | .6548 | .9117 | .1708 | .0985 | .0769 |

Under RA–LL, physics-aware minus Huang is **−.0128** AUPRC [−.0280, +.0019], **−.0166** AUROC [−.0204, −.0124], **+.0318** sensitivity [+.0189, +.0457], **−.0209** specificity [−.0246, −.0172], and **+.0210** Brier [+.0188, +.0232]. Huang minus unprotected baseline is **+.4164** AUPRC [+.3935, +.4407], **−.0421** sensitivity [−.0698, −.0195], and **+.5518** specificity [+.5421, +.5616]. Thus ranking gain must not be described as a clinical sensitivity gain. On clean ECGs, Huang-minus-baseline AUPRC is −.0006 [−.0045, +.0033]. All intervals use 1,000 synchronized with-replacement patient resamples and seed-mean paired contrasts; [full metric intervals](PAIRED_PATIENT_BOOTSTRAP.csv), [three-seed means](TWA_THREE_SEED_MEANS.csv), and [each seed](TWA_BY_SEED.csv) are aggregate-only.

## Detector errors and diagnostic consequences

The detector correctly called **9,701/10,000 clean ECGs** (97.01% clean specificity) and **9,919/10,000 simulated RA–LL ECGs** (99.19% RA–LL recall). Of 299 clean false corrections, 61 were called RA–LA, 15 RA–LL and 223 LA–LL. Of 81 RA–LL mistakes, 17 were called clean, 48 RA–LA and 16 LA–LL. These are post-hoc error classifications only; true condition was never supplied to the correction policy. [Aggregate confusion](DETECTOR_CONFUSION_AGGREGATE.csv) contains no identifiers.

When the detector correctly leaves clean ECGs untouched, diagnostic predictions are exactly the reused baseline predictions. On the 299 false-corrected clean ECGs, mean TWA Brier rises from .1869 to .1979, and 14.5% of seed-record fixed-threshold decisions differ from the uncorrected clean reference. On the 81 wrongly classified RA–LL ECGs, mean Brier is .2608 after predicted correction versus .2304 for the clean reference (the uncorrected reversed value is .3274), and 28.8% of seed-record threshold decisions differ from clean. These small error-subset diagnostics are descriptive, not subgroup efficacy estimates; no per-patient output is published. The [aggregate diagnostic breakdown](DETECTOR_ERROR_DIAGNOSIS_AGGREGATE.csv) shows that most global benefit comes from the 9,919 correctly inverted RA–LL ECGs; detector mistakes prevent perfect recovery, while even wrong corrections can sometimes improve on no correction.

## Provenance and limits

Zero new downloads; restricted detector and diagnostic prediction arrays (320,128 and 720,128 bytes) remain in `data/heedb_restricted/I0006/huang_exploratory_v1/`, covered by `.gitignore`. Detector inference took 185.219 s on CPU; three clean diagnostic checkpoint passes plus three faulted passes took 72.985 s; bootstrap analysis took 126.094 s. The source waveform and independent prediction hashes, the two new prediction hashes, code/weight hashes and aggregate result hashes are in [the frozen protocol](FROZEN_PROTOCOL.json), [analysis completion](ANALYSIS_COMPLETE.json), [detector completion](DETECTOR_COMPLETE.json) and [diagnostic completion](HUANG_DIAGNOSIS_COMPLETE.json). The reused baseline/M2 means match the original independent aggregate table in all eight metrics for both conditions (zero mismatches).

This is **not** a SOTA claim or proof that either protected approach is equivalent. It evaluates our unchanged clean ResNet downstream of Huang's published detector, not Huang's own paper score. It tests electrically simulated cable permutations, not verified real-world wiring errors, and physician-overread HEEDB proxy labels, not exact Chapman/Ningbo SNOMED labels. The comparison was post-hoc and only TWA was analyzed here. The previously reported Ningbo integrated latency comparison is a separate hardware/software benchmark, not a HEEDB runtime-performance or clinical claim.
