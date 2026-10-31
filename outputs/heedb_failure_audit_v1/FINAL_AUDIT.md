# HEEDB v1 post-hoc failure audit

## Decision

The v1 numbers are arithmetically reproducible, but **v1 did not execute the intended pipeline-matched, electrically exact RA–LL test**. Two implementation errors affect every evaluation record: the classifier received 5,000 time steps where Ningbo training supplied 1,000, and the v1 RA–LL routine computed aVR and aVF incorrectly. The near-prevalence core-TWA AUPRC and near-zero frozen-threshold sensitivity are therefore valid descriptions of the *v1 implementation*, not a definitive test of diagnostic transportability. Preserve v1; a single, separately frozen corrected v2 on the same local cohort is justified. No corrected inference was run in this audit.

## Independent integrity and metric checks

- All 23 checked bindings passed: v1 protocol, evaluator and physics source; six checkpoints, six preprocessing artifacts and six validation/threshold artifacts; restricted selection and prediction cache. The prediction array has axes `(baseline, physics-aware) × (104729, 130363, 155921) × 26 conditions × 10,000 records × (TWA, SB, STach)` and contains only finite values.
- The three baseline checkpoint headers encode SNOMED channels `[164934002, 426177001, 427084000]`, corresponding to TWA, SB and STach. M2 uses the same training label order, and its validation label arrays equal baseline's for each seed. On Ningbo validation, the matched channels had three-seed mean AUROC 0.920/0.919 for TWA, 0.9985/0.9986 for SB and 0.9981/0.9989 for STach (baseline/M2), well above cross-channel alternatives. The HEEDB analysis indexed channels 0, 1 and 2 accordingly. No channel swap was found.
- Parsing `codes_physician` again produced 1,524 core TWA, 1,665 core SB, 673 core STach, 770 strict TWA and 1,448 strict SB positives among 10,000 patients, with zero differences from the frozen label columns. Core TWA prevalence is **0.1524**. Software and v24 fields were not used.
- Recomputing every core clean/RA–LL seed AUPRC and sensitivity from the cached probabilities and original validation thresholds agreed with the v1 CSV to within `9.6×10⁻¹⁷`. Recomputed primary three-seed contrasts agreed to within `9.8×10⁻¹⁷`. In particular, physics-aware minus baseline RA–LL core-TWA AUPRC is −0.012310 and sensitivity is −0.012248. The arithmetic is not the cause of the discrepancy.

## Signal and preprocessing checks

The Ningbo source headers and HEEDB headers both specify 12 leads in `I, II, III, aVR, aVL, aVF, V1–V6` order and gain **1,000 digital counts/mV**, with baseline zero. Ningbo loads the `.mat` digital `val`; HEEDB v1 reads WFDB physical mV and multiplies by 1,000. In 20 local records at each HEEDB sampling rate, digital counts minus physical mV × 1,000 differed by at most `4.6×10⁻¹³` counts; rebuilding the cached waveform from WFDB yielded **zero** float32 difference. This rules out a factor-of-1,000 conversion error. The 250 Hz path resampled 2,500 samples to 5,000 at 500 Hz as coded; 506 selected records were 250 Hz and 9,494 were 500 Hz.

The time axis diverged after that step. The Ningbo baseline and M2 training loaders both call `raw(...)=wave(... )[:, ::5]`: a 500 Hz, 5,000-sample source becomes **1,000 samples at an effective 100 Hz**. HEEDB v1 supplied all **5,000 samples** to each model. ECGResNet's global temporal pooling accepts either length, which explains why the engineering smoke passed. Its temporal convolution filters nevertheless see five times as many samples per waveform and different frequencies per step. This is a material, reproducible preprocessing error on both clean and reversed inputs, including both original sampling rates.

The amplitude scale is broadly plausible but not identical. In deterministic, independent 256-record samples, Ningbo training versus HEEDB after the *proposed* matching decimation had the following pooled 99th percentile absolute amplitude and normalized standard deviation using the frozen Ningbo mean/std:

| Lead | Ningbo p99 µV | HEEDB p99 µV | Ningbo normalized SD | HEEDB normalized SD |
|---|---:|---:|---:|---:|
| I | 615 | 639 | 1.033 | 0.979 |
| II | 756 | 727 | 0.944 | 0.957 |
| V1 | 893 | 854 | 0.924 | 0.824 |
| V3 | 1,479 | 1,069 | 0.934 | 0.630 |
| V6 | 1,488 | 839 | 0.995 | 0.411 |

Median whole-record RMS was 205.6 µV for Ningbo and 153.3 µV for HEEDB. HEEDB v1's un-decimated amplitude quantiles and normalized SDs were essentially the same as its decimated values, so this smaller chest-lead scale is a cohort/acquisition shift rather than the fivefold time-axis error. It may still contribute to transportability after correction; this audit cannot assign causal shares without corrected inference. Einthoven residual p99 was about 1 µV in both samples, consistent with rounding in the derived leads.

The RA–LL implementation has a separate electrical error. With original I and II, exact cable reversal gives `I′=I−II`, `II′=−II`, `III′=−I`, `aVR′=II−I/2`, `aVL′=I−II/2`, `aVF′=−(I+II)/2`, and unchanged V1–V6. V1 instead used `aVR′=(II−I)/2` and `aVF′=I/2−II`; those same wrong equations are recorded in the frozen v1 protocol. Against the project's hashed `ecg_physics.py` on 32 cached ECGs, aVR differed by p99 400 µV (maximum 946.5 µV), and aVF by p99 541.5 µV (maximum 1,634.5 µV). The other leads agreed to float precision. Consequently the v1 reversal condition is **not electrically exact**; its RA–LL contrasts cannot answer the stated reversal question.

## Cached score diagnostics

The table gives three-seed means. `Score p50/p90/p99` are patient score quantiles averaged across seeds, not pooled probabilities. ECE uses 15 equal-width bins. Sensitivity, specificity and predicted-positive fraction use each seed's original Ningbo validation threshold, with no new threshold chosen.

| Label | Arm | Input | Prev. | AUROC | AUPRC | Sens. | Spec. | Pred. + | Score p50/p90/p99 | ECE15 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| TWA | baseline | clean | .152 | .484 | .158 | .081 | .917 | .083 | .331/.615/.759 | .230 |
| TWA | baseline | v1 RA–LL | .152 | .509 | .162 | .023 | .983 | .018 | .192/.451/.687 | .137 |
| TWA | physics-aware | clean | .152 | .452 | .144 | .015 | .981 | .018 | .172/.411/.581 | .189 |
| TWA | physics-aware | v1 RA–LL | .152 | .458 | .149 | .010 | .994 | .007 | .139/.382/.628 | .154 |
| SB | baseline | clean | .167 | .540 | .185 | .955 | .068 | .936 | .274/.613/.828 | .225 |
| SB | baseline | v1 RA–LL | .167 | .506 | .161 | .855 | .173 | .832 | .077/.297/.625 | .141 |
| SB | physics-aware | clean | .167 | .592 | .209 | .990 | .019 | .982 | .253/.621/.856 | .166 |
| SB | physics-aware | v1 RA–LL | .167 | .599 | .208 | .993 | .017 | .985 | .243/.596/.826 | .149 |
| STach | baseline | clean | .067 | .562 | .087 | .343 | .738 | .267 | .025/.150/.438 | .055 |
| STach | baseline | v1 RA–LL | .067 | .445 | .058 | .218 | .737 | .260 | .024/.239/.481 | .089 |
| STach | physics-aware | clean | .067 | .465 | .065 | .489 | .479 | .519 | .042/.140/.266 | .076 |
| STach | physics-aware | v1 RA–LL | .067 | .454 | .062 | .436 | .545 | .454 | .033/.116/.224 | .071 |

The TWA clean AUPRCs are near the 0.152 prevalence, and AUROC is near or below 0.5. The transferred TWA thresholds generally lie above most HEEDB scores; mean predicted-positive fraction is 8.3% for baseline clean and 1.8% for physics-aware clean. Yet low sensitivity is not solely an operating-point issue: the poor AUROC and prevalence-level AUPRC show weak ranking as well. By comparison, on Ningbo validation at a similar TWA prevalence (0.148), baseline/M2 mean TWA AUROC was 0.920/0.920, AUPRC 0.656/0.668, and sensitivity at the same frozen thresholds about 0.719/0.726. Mean clean HEEDB TWA scores were 0.344/0.202 versus observed prevalence 0.152 (baseline/M2), another sign that the validation operating points and calibration did not transfer in v1. The detailed per-seed score quantiles, means, Brier scores and calibration errors are in `AGGREGATE_SCORE_DIAGNOSTICS.csv`.

This is not a single-seed anomaly. Clean core-TWA metrics by seed were:

| Arm | Seed | AUROC | AUPRC | Sensitivity at frozen threshold |
|---|---:|---:|---:|---:|
| baseline | 104729 | .533 | .187 | .058 |
| baseline | 130363 | .518 | .164 | .102 |
| baseline | 155921 | .400 | .123 | .082 |
| physics-aware | 104729 | .419 | .129 | .028 |
| physics-aware | 130363 | .497 | .160 | .001 |
| physics-aware | 155921 | .442 | .143 | .016 |

## Exploratory label and rate checks

Among 8,476 frozen core-TWA negatives, **627** carry at least one prespecified ambiguous T-wave code. Across all 10,000 records, 263 carry mixed ST–T code 1141 and 586 carry a serial/fragment code (categories may overlap). Strict 1140 labels select 770 positives; 754 core positives are outside this strict definition. These are clinically meaningful definition differences, not permission to relabel the primary test.

| TWA definition, exploratory except frozen core | Positive / total | Prev. | Baseline clean AUPRC / AUROC | M2 clean AUPRC / AUROC |
|---|---:|---:|---:|---:|
| Frozen core | 1,524 / 10,000 | .152 | .158 / .484 | .144 / .452 |
| Strict 1140 | 770 / 10,000 | .077 | .096 / .558 | .091 / .541 |
| Core plus all ambiguous codes | 2,151 / 10,000 | .215 | .219 / .479 | .202 / .453 |
| Exclude ambiguous core negatives | 1,524 / 9,373 | .163 | .169 / .482 | .153 / .450 |

The broad AUPRC rises mainly with prevalence; removing ambiguous negatives does not restore ranking. Strict TWA shows modest above-chance ranking, but not Ningbo-like discrimination. The v1 reversal analogues are in `AGGREGATE_LABEL_SCENARIOS.csv` and remain affected by the electrical error.

The result is also not confined to the 250 Hz path. Core TWA had 73 positives among 506 records at 250 Hz and 1,451 among 9,494 at 500 Hz. Baseline clean mean AUROC/AUPRC was .448/.151 at 250 Hz and .485/.159 at 500 Hz; M2 clean was .401/.126 and .455/.145, respectively. Frozen-threshold clean sensitivity was .046/.083 for baseline and .005/.016 for M2 (250/500 Hz). These rate analyses are descriptive; all v1 records share the temporal mismatch.

## Next step

The separately hashed [corrected-v2 plan](CORRECTED_V2_PLAN.md) specifies the same cohort, labels, models, thresholds and mask panel, with only the 1,000-step input and exact RA–LL algebra corrected. Plan SHA-256: `fd36873f9e276cb1c6a4bdbf8ba7f9e9e4948c98a4701e1f741e6543b884e58e`. Before any corrected inference, create and hash a separate executable v2 protocol and evaluator and pass an engineering smoke that checks input length and all 12 reversal leads. Save v2 results alongside v1 and identify them as a post-hoc implementation correction. Additional HEEDB model development or threshold tuning is not warranted by this audit; evaluate the correction first. If corrected clean and exact RA–LL ranking remain near prevalence, conclude that these frozen classifiers did not transport to the HEEDB physician-code proxy outcomes.

This audit reused only local tables, waveforms and predictions. It performed no training, threshold selection, waveform download, or corrected model inference. All patient- and recording-level inputs remain in the git-ignored restricted directory; the new outputs contain aggregates only. Audit code: `src/heedb_failure_audit_v1.py`, SHA-256 `c5b324c8f93cc626bea526ff571df7e2e846a58520a17de9e58fd4d651ff869e`.
