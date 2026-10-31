# Feasibility, novelty, and design audit

## Phase-0 verdict: NO-GO

The research question is feasible in principle, and the physical-fault formalism is implementable without inventing waveform transformations. The project nevertheless fails the prespecified replication gate: two usable, non-PTB, patient-separable confirmation datasets have not been established from the official Challenge 2020 release. CPSC/CPSC-Extra and Georgia are public and label-rich, but the release’s headers have only recording identifiers; the official documentation says repeat patients between sets may occur and does not provide a patient-ID field. Per the prespecified rule, a source without authoritative patient IDs where repeats cannot be ruled out is ineligible for confirmatory patient-cluster inference.

This is an exact blocking fact, not a claim that these sources are unusable for exploratory record-level analysis. The frozen Phase-1 protocol is retained only as a contingent design. No outcome data from a sealed confirmation source were accessed.

## Gate audit

| Required GO condition | Finding | Status |
|---|---|---|
| Official public data without private assistance | Challenge 2020 v1.0.2 is public under CC BY 4.0. | PASS |
| Two non-PTB sources usable for confirmation | CPSC/CPSC-Extra and Georgia lack authoritative patient IDs; repeats cannot be excluded. | **FAIL** |
| Sufficient prevalence for two relevant endpoint families | Record-level counts are ample for AF, RBBB, LBBB, sinus rhythm, and conduction/rhythm labels; not sufficient evidence for patient-level inference without patient IDs. | BLOCKED |
| Patient/source separation defensible | Source separation is clear by folder; patient-level separation is not. PTB and PTB-XL are explicitly excluded from replication. | **FAIL** |
| Physical transformations unit-testable | Exact limb terminal permutations and explicit unavailability patterns are implemented and tested. Open-circuit waveform behaviour and positional misplacement are correctly not fabricated. | PASS |
| No same matched physical-fault vs random benchmark, 2024–26 | Targeted primary-literature search found adjacent work, but no paper matching the complete controlled comparison. This is a bounded search result, not an impossibility claim. | PASS (narrow) |
| Lightweight baseline within limit | Reused 12-channel residual CNN is ~3.9 MB/checkpoint and ran on the local RTX 5070 Laptop GPU; projected pilot is within limit. | PASS |
| Meaningful claim without augmentation success | The estimand is an evaluation/failure-analysis contrast; no augmentation claim is required. | PASS |

## Audit of the legacy ECG projects (read-only)

### Data, splits, identities, and models

`ecg_selective_12lead_pilot` audited official PTB-XL v1.0.2: 21,801 valid records, 18,869 `patient_id` values, no patient-fold violations, 100 Hz × 10 s waveforms, and the conventional displayed lead order. Its models were lightweight 12-channel residual CNNs (three seeds: 104729, 130363, 155921), trained folds 1–8 and evaluated on fold 9. Fold 10 was not accessed by either read-only audit. The prior project’s complete lead-derived channel representation is a key limitation for physical-fault inference.

`structured_missingness_biomedical_audit` reused those three checkpoints and fold-9 data. Its corrected v2 audit fixed an important comparator error: the original random condition had averaged probability vectors across masks, producing an artificial ensemble. Corrected v2 instead computed each random-mask metric separately inside each patient bootstrap replicate, then averaged the metrics. It additionally enumerated every fixed 4-of-12 displayed-lead removal subset (495 subsets). No model was retrained in corrected v2.

### What PTB-XL supports

The corrected v2 fold-9 result is exploratory but technically meaningful for its stated abstraction: with four displayed channels set to normalized zero, removal of V1–V4 reduced MI raw AUPRC versus the fair per-record random four-lead comparator by **18.1, 19.4, and 11.0 percentage points** across the three seeds. Patient-cluster 95% intervals were respectively −21.1 to −14.6, −22.7 to −15.6, and −14.2 to −7.9 points. Its macro-AUPRC fixed-subset percentiles were 4.4, 6.9, and 18.8. This supports the motivating observation that a coherent displayed precordial block can be substantially worse than an equally sized random whole-lead mask.

### What it does not support

- It is fold-9 development evidence, not a sealed or external result.
- A displayed 12-lead input contains derived limb leads; removing four stored channels is not equivalent to four independent electrode failures.
- A zero-filled channel is an availability abstraction, not an open-circuit impedance, motion artefact, electrode displacement, or cable reversal waveform.
- The v3 structured-mask training experiment **failed** its frozen gate and closed the augmentation topic. Its mean held-out V1–V4 MI AUPRC was lower for structured than uniform training (0.6389 vs 0.7161); it must not be described as a successful repair.
- No observed fault prevalence or prospective outage log exists in PTB-XL.

## Official Challenge 2020 v1.0.2 metadata-only replication audit

No file was downloaded in Phase 0. The facts below come from the official project page and Challenge paper.

| Source folder | Records | Sampling / typical duration | Candidate endpoints from official training+validation figure | Identifier and repeat-patient assessment | Status |
|---|---:|---|---|---|---|
| `cpsc_2018` | 6,877 | 500 Hz; 6–60 s | AF 722; RBBB 1,857; sinus rhythm 918; IAVB 616 | Recording filename only in header; no patient ID. Repeats cannot be ruled out. | exploratory only |
| `cpsc_2018_extra` | 3,453 | 500 Hz; 6–60 s | AF 106; RBBB 1; sinus rhythm 4; IAVB 73 | Same limitation; small counts for many proposed endpoints. | exploratory only |
| `georgia` | 10,344 | 500 Hz; 10 s | AF 769; RBBB 542; sinus rhythm 18,092; IAVB 459; LBBB 552 | Recording filename only in header; no patient ID. Repeats cannot be ruled out. | exploratory only |
| `st_petersburg_incart` | 74 recordings from 32 Holters | 257 Hz; 30 min | too small / different episodic annotation structure | recording identifiers; repeated source records explicit | ineligible |
| `ptb`, `ptb-xl` | 516; 21,837 | 1 kHz, 110.8 s; 500 Hz, 10 s | label-rich | forbidden by replication scope; PTB-XL overlap is direct | excluded |

Challenge documentation gives 13,256 total CPSC records (including a 2,926-record hidden component) and 20,678 total Georgia records (including 10,344 hidden). Only the listed training folders are public. The record-level figure counts are across public training and validation material and are not recalculated header counts; they are adequate only for provisional endpoint selection. Headers contain number, lead count, sampling frequency, sample count, age, sex, Dx, Rx, Hx, and Sx—not a documented patient identifier. The Challenge paper explicitly says the test set may include a small number of patients also in training; this is an explicit overlap risk, not a resolved one.

The project page gives public access under CC BY 4.0 and 7.5 GB uncompressed / 4.5 GB ZIP for the whole release. Metadata/header inspection is possible from the public files, but the source-wide header list is not a small manifest; a full waveform download is neither needed nor permitted in Phase 0.

## External-replication conclusion

There are two **candidate record-level** external sources (CPSC and Georgia) and several label families with hundreds of positives, but there are zero eligible sources for the prespecified **patient-level confirmatory** analysis. The project therefore cannot honestly claim two usable external confirmation datasets or freeze class counts for patient-level inference. A different source must be identified and audited before resuming.

## Compute and storage estimate

The local GPU is an NVIDIA GeForce RTX 5070 Laptop GPU with 8,151 MiB VRAM. The legacy 12-channel CNN uses widths 32/64/128/256, 10-second 100-Hz inputs, mixed precision, and three ~3.9 MB checkpoints. Masked inference plus patient bootstrap of three existing checkpoints took about 408 seconds in the legacy audit. A conservative Phase-1 pilot with one new source, one seed, two light architectures, clean + exact-fault + matched-random inference, 1,000 cluster bootstraps, and no foundation model is estimated at **8–11 GPU-hours** and **12–18 wall-clock hours** (including CPU I/O and bootstrap). The three-seed, two-source program is estimated at **36–45 GPU-hours** and **3–5 elapsed days** on this laptop, below the requested 48 GPU-hour ceiling but only if data are stored locally and training remains lightweight.

Whole Challenge 2020 is 4.5 GB compressed / 7.5 GB uncompressed. A source-selective two-folder working copy, derived 100-Hz cache, checkpoints, and predictions is projected at 3–7 GB depending on compression; reserve 15 GB. Phase 0 downloaded 0 MB.

## Sources

- [Official Challenge 2020 v1.0.2 project page](https://physionet.org/content/challenge-2020/1.0.2/) — access, license, folders, public counts, file size, header format, and sources.
- [Reyna et al., Challenge 2020 paper](https://physionet.org/files/challenge-2020/1.0.1/papers/2020ChallengePaper.pdf) — source sizes, duration, sampling frequencies, labels, and overlap caveat.
- [Wagner et al., PTB-XL](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/) — PTB-XL provenance and conventional ECG acquisition context.
