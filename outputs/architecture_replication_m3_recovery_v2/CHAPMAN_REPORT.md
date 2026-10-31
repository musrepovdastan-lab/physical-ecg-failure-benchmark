# TimePatchTransformer M3 corrected evaluation

**Decision: FAIL_M3_ARCHITECTURE_SPECIFIC**. The frozen Stage M3A gate was not passed, so Stage B (Ningbo training/evaluation) was not run.

This is a corrected reanalysis of the previously accessed Chapman test set, not an untouched confirmation. No Chapman model was retrained. Historical inverted mask predictions were not reused.

Recovery protocol binding: `e4a3287aa726325360845ffe200f905c963c7a51ec44241a198a1b434d3dc3de`. Original M3 protocol SHA-256: `16D554B2C18532792FE534107F5CBD28EAC0F4B8547FEA12FA40F97718FAAEFD`.
Integrity: **PASS**; six units, 4752/4,752 corrected mask predictions, 14,292 condition/endpoint metric rows, and 18/18 synchronized bootstrap tasks.

## Stage M3A decision

| Frozen criterion | Result | Estimate | Threshold |
|---|---|---:|---:|
| aggregate_contrast_reduction_min_0_15 | PASS | `0.4050692048702242` | `0.15` |
| all_integrity_gates | PASS | `True` | `integrity` |
| any_clean_decline_max_0_04 | PASS | `[0.02009548823965912, -0.0032531487420337513, -0.03967669365734228]` | `-0.04` |
| any_mask_decline_max_0_04 | FAIL | `[0.02653334749628844, 0.029606188982806425, -0.11256703322433398]` | `-0.04` |
| mean_clean_decline_max_0_02 | PASS | `[0.02009548823965912, -0.0032531487420337513, -0.03967669365734228]` | `-0.02` |
| mean_mask_decline_max_0_02 | PASS | `[0.02653334749628844, 0.029606188982806425, -0.11256703322433398]` | `-0.02` |
| physics_contrast_upper95_max_0_15 | PASS | `-0.080032978951931` | `0.15` |
| ra_ll_twa_improvement_each_seed_min_0_15 | PASS | `[0.4285650784377727, 0.3580782262593452, 0.37213681316831543]` | `0.15` |

The sole failed criterion is the per-seed mean-mask retention limit: seed 155921 declined by 0.1126 TWA AUPRC, exceeding the allowed 0.04 decline. The mean across seeds was a 0.0188 decline and passed the 0.02 limit. This is a mask-performance cost at one seed, not evidence that the RA–LL protection effect is absent.
RA–LL TWA AUPRC improved by at least 0.15 in all three seeds; aggregate mask-minus-RA–LL contrast reduction passed; the physics-aware contrast upper 95% limit also passed. Mean and per-seed clean-performance limits passed.

## Per-seed TWA AUPRC

| Seed | Clean arm clean | Clean arm RA–LL | Clean arm exact mask | Clean arm 792-mask mean | Physics-aware clean | Physics-aware RA–LL | Physics-aware exact mask | Physics-aware 792-mask mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 104729 | 0.6255 | 0.2090 | 0.4089 | 0.4940 | 0.6456 | 0.6376 | 0.5341 | 0.5206 |
| 130363 | 0.6245 | 0.2250 | 0.3739 | 0.5062 | 0.6213 | 0.5830 | 0.5718 | 0.5358 |
| 155921 | 0.6606 | 0.2209 | 0.5385 | 0.5594 | 0.6209 | 0.5930 | 0.4772 | 0.4469 |

The exact mask removes I, II, III, aVR, and aVF (indices [0,1,2,3,5]); it matches the corresponding column in the lexicographic 792-mask panel. Every mask metric was computed separately before taking the panel mean.

## Synchronized bootstrap

| TWA estimand | Bootstrap mean | Percentile 95% interval | Replicates |
|---|---:|---:|---:|
| RA–LL TWA AUPRC improvement | +0.3861 | [+0.3467, +0.4249] | 1,000 |
| Mean-mask minus RA–LL contrast reduction | +0.4050 | [+0.3678, +0.4455] | 1,000 |
| Physics-aware mean-mask minus RA–LL contrast | -0.1032 | [-0.1254, -0.0800] | 1,000 |
| Clean TWA AUPRC change | -0.0077 | [-0.0271, +0.0119] | 1,000 |
| Mean-mask TWA AUPRC change | -0.0189 | [-0.0338, -0.0045] | 1,000 |

Bootstrap resamples use the existing Chapman multiseed indices (1000 × 2050, canonical test record order; synchronized across conditions, seeds, and arms). Central estimates use the complete test sample; bootstrap means are reported separately.

## Interpretation and next steps

The corrected Chapman results show a large RA–LL benefit for physics-aware training and a substantial reduction in the mask-versus-RA–LL contrast. The frozen Stage M3A gate nevertheless fails because seed 155921 misses the mean-mask retention bound. Under the protocol, this is `FAIL_M3_ARCHITECTURE_SPECIFIC`; it is not a passed architecture replication.
Ningbo Transformer training and test evaluation were not run because Stage M3A did not pass. No new method, architecture, threshold, split, or probability was selected.
Chapman ResNet corrected mask panel status remains unavailable/not fully verified; this package did not restore it. Ningbo ResNet values below are from the separate corrected recovery package, after historical Ningbo test access.

| Model family | Chapman corrected mask status | Ningbo corrected panel/status | Interpretation |
|---|---|---|---|
| ECGResNet-small | Not fully verified; not regenerated here | Corrected recovery complete for baseline and M2 (6 units, 4,752 predictions); M2 retained its reported frozen-gate PASS | Ningbo is corrected reanalysis after prior test access, not independent untouched confirmation. |
| TimePatchTransformer | Corrected 6-unit Chapman panel complete; Stage M3A **FAIL** on one-seed mean-mask retention | **Not run** because Stage M3A gate failed | Ningbo transfer of the intervention remains untested. |

Open research questions remain chest-only performance, calibration, published baselines, and clinical interpretation. This work does not rank Transformer versus ResNet as an architecture in general.

Detailed tables: `CONDITION_RESULTS_BY_SEED.csv`, `EXACT_MASK_AND_DISTRIBUTION_BY_SEED.csv`, `MASK_DISTRIBUTION_SUMMARY.csv`, `PRIMARY_CONTRASTS.csv`, `BOOTSTRAP_SUMMARY.csv`, and `BOOTSTRAP_REPLICATES.csv`.
