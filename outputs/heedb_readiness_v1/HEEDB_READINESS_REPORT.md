# HEEDB v5.0 data-readiness report

Updated: 2026-09-24 (Asia/Qyzylorda)

## Decision

**Data-readiness decision: PASS TO FREEZE PROTOCOL.** A fair HEEDB external-validation protocol can now be frozen, provided it uses the physician-overread field as the primary label source, names the endpoints as an explicit HEEDB 12SL operational crosswalk rather than claiming exact SNOMED equivalence, groups or samples by the real patient identifier, and specifies sampling-rate handling before model outcomes are examined.

This task did not train, tune, run inference, or report model performance. The full external evaluation has not started. The 23-patient engineering sample described below should be excluded from the later untouched evaluation.

## Access and provenance

The prior 2026-09-23 attempt stopped before S3 access because profile `bdsp` returned the AWS root principal. That profile was not used in this resumed audit.

The new credential gate passed:

- Command: `aws sts get-caller-identity --profile bdsp-reader`
- Account: `552883519608`
- ARN: `arn:aws:iam::552883519608:user/bdsp-heedb-reader`
- Required identity: exact match

All S3 operations used only the previously recorded large-repository Access Point alias `bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias`. No publication-project bucket, root profile, recursive `ECG/` listing, or recursive `WFDB/` listing was used.

## Downloads and cap

### I0006 tables

| Remote object | Remote bytes | Local SHA-256 |
|---|---:|---|
| `ECG/I0006/12SL_diagnoses/diagnoses_dictionary.csv` | 18,958 | `903c02c4304c95049647f324817f552c2ffafe41c7dfda419b67e8f639dd4b34` |
| `ECG/I0006/12SL_diagnoses/diagnoses_acquisition.csv` | 101,266,857 | `32eddfbd813c69138850344ff0f141a1b07c3716ef11bb113cf10c254f66a296` |
| `ECG/I0006/12SL_diagnoses/diagnoses_v24.csv` | 60,927,709 | `f2ee79dfc7be0b22ae172cb36807f20da0ccf7d7800e2647f95ea08d4ce0eb66` |
| `ECG/I0006/metadata/metadata.csv` | 169,535,260 | `a52601522471d2a881271749cf7698817d395cb8ecc9d01e0eac36950dabe2c0` |
| **Table total** | **331,748,784** | |

Remote sizes were recorded with object-specific `HeadObject` calls before transfer. Local byte counts matched exactly.

### Waveform readability sample

- Deterministic sample: 23 patients and 23 recordings, one recording per patient.
- Coverage: three recordings from each of seven recording-level label patterns and two from the rare all-three-positive pattern, which has only two physician-coded recordings in the full table.
- Objects: 23 `.hea` plus 23 `.dat` files.
- `.hea`: 19,415 bytes total; 826–854 bytes per file.
- `.dat`: 2,700,000 bytes total; 60,000–120,000 bytes per file.
- Waveform sample total: **2,719,415 bytes**.

Total new HEEDB download: **334,468,199 bytes (318.97 MiB)**. Cap use: **15.57% of 2,147,483,648 bytes**. Restricted files and record-level manifests remain under `data/heedb_restricted/`, which is covered by the repository `.gitignore`. Available space on `C:` at final verification was **176,332,771,328 bytes (164.22 GiB)**.

## Table schemas and joins

| Table | Rows | Columns used in this audit |
|---|---:|---|
| Diagnosis dictionary | 503 | `codes`, `acronym`, `diagnoses` |
| Acquisition diagnoses | 998,844 | `FileName`, `codes_software`, `codes_physician`, `jaccard_index` |
| v24 diagnoses | 974,172 | `FileName`, `codes` |
| Metadata | 998,844 | patient, file, demographic, and date/age fields |

`FileName` is unique and complete in acquisition diagnoses and metadata. All 998,844 acquisition rows join one-to-one to metadata. `FileID` is also unique and complete. The v24 table joins one-to-one for all 974,172 rows but omits 24,672 acquisition/metadata recordings.

`BDSPPatientID` is missing for 27,328 recordings. The patient-linked eligible pool before waveform QC contains **971,516 recordings from 349,547 patients**. Of these patients, 174,035 have multiple recordings; median is 1, 75th percentile is 3, 99th percentile is 17, and maximum is 106 recordings. Record-level random splitting would therefore leak patients and overweight frequent users.

Other material missingness:

- Physician code field: 676 empty rows.
- Software code field: 77 empty rows.
- Sex: 733 empty rows.
- Age/date of birth: 3,030 empty rows.
- Race: 8,353 empty rows.

The software-versus-physician code-set Jaccard value is present for every acquisition row: mean 0.8600, median 1.0, and 675,241 rows equal 1.0.

## Label source and mapping

The primary source should be `diagnoses_acquisition.csv:codes_physician`, which explicitly represents physician-overread codes. `codes_software` must remain a separate software-generated sensitivity/QC source. `diagnoses_v24.csv:codes` has no source indicator in its schema, omits 24,672 records, contains the nonnumeric token `None` 5,492 times, and is not suitable as the primary label source without additional provenance documentation.

The following mapping is a predeclared HEEDB 12SL operational crosswalk. It is not an assertion that 12SL codes are ontology-equivalent to the Chapman/Ningbo SNOMED labels.

### T-wave abnormality — SNOMED reference `164934002`

- Strict sensitivity definition: code `1140` (`NT`, nonspecific T wave abnormality).
- Recommended core definition: `1140`, `1145`, `1150`, `1151`, `1160`, `1161`, `1170`, `1171`, `1172`, `1180`, `1181`.
- The added core codes explicitly say T-wave abnormality or marked T-wave abnormality with inferior, lateral, anterior, inferolateral, or anterolateral localization and often “consider ... ischemia.” They are clinically narrower/more etiologic than a generic SNOMED T-wave-abnormality label.
- Excluded as ambiguous: `1141` (combined nonspecific ST and T abnormality), `1142` (abnormal QRS-T angle, consider primary T-wave abnormality), and `1182` (incomplete “T wave inversion in” phrase).
- Excluded longitudinal/fragments: `1200`, `1201`, `1203`, `1207`, `1208`, `1210`–`1224`. These describe T waves or changes versus a prior ECG and should not define a current cross-sectional label.

### Sinus bradycardia — SNOMED reference `426177001`

- Strict definition: `21` (`SBRAD`, sinus bradycardia).
- Recommended core definition: `21` plus `24` (`MSBRAD`, marked sinus bradycardia).
- Excluded as ambiguous: `265` (`PR-SBRAD`, probable sinus bradycardia, verify AV conduction).

### Sinus tachycardia — SNOMED reference `427084000`

- Strict and recommended core definition: `23` (`STACH`, sinus tachycardia).
- Atrial, junctional, supraventricular, ventricular, low-atrial, wide-QRS, and narrow-QRS tachycardia codes are not substitutes and remain excluded.

Repeated code tokens occur within 130,465 physician rows and 136,437 software rows. Label membership was therefore deduplicated within each recording. Ambiguous-code counts below are flags and may overlap core-positive recordings.

## Exact aggregate label counts

### Primary physician-overread source

| Outcome | Strict recordings, all | Strict patient-linked recordings | Strict patients | Core recordings, all | Core patient-linked recordings | Core patients | Ambiguous recordings, all | Ambiguous patient-linked recordings | Ambiguous patients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-wave abnormality | 75,251 | 73,128 | 46,959 | 175,578 | 170,173 | 81,645 | 93,384 | 91,352 | 54,792 |
| Sinus bradycardia | 135,277 | 131,730 | 80,881 | 152,881 | 148,773 | 86,215 | 87 | 86 | 84 |
| Sinus tachycardia | 76,072 | 74,390 | 40,546 | 76,072 | 74,390 | 40,546 | 0 | 0 | 0 |

### Software-generated source

| Outcome | Core recordings, all | Core patient-linked recordings | Core patients | Agreement with physician core | Both positive | Software only | Physician only |
|---|---:|---:|---:|---:|---:|---:|---:|
| T-wave abnormality | 176,645 | 171,249 | 81,327 | 98.7849% | 170,043 | 6,602 | 5,535 |
| Sinus bradycardia | 152,751 | 148,646 | 86,305 | 99.7719% | 151,677 | 1,074 | 1,204 |
| Sinus tachycardia | 78,319 | 76,620 | 42,079 | 99.4362% | 74,380 | 3,939 | 1,692 |

### v24 source with unresolved provenance

| Outcome | Core recordings, all | Core patient-linked recordings | Core patients |
|---|---:|---:|---:|
| T-wave abnormality | 178,378 | 172,979 | 83,893 |
| Sinus bradycardia | 147,304 | 143,273 | 84,033 |
| Sinus tachycardia | 76,273 | 74,609 | 41,176 |

The differences are real source disagreement, not rounding. The protocol must not merge physician, software, and v24 labels.

## Waveform smoke test

All 23 selected recordings:

- joined exactly to their metadata and acquisition-diagnosis rows;
- reproduced the stored physician core-label bits;
- loaded successfully with WFDB;
- had 12 signals in `I, II, III, aVR, aVL, aVF, V1–V6` order;
- used mV units;
- contained no NaN or infinite physical samples.

Twenty-two records were 500 Hz with 5,000 samples. One was 250 Hz with 2,500 samples. Both represent 10 seconds, but this observed heterogeneity invalidates a blanket 500 Hz assumption.

## Protocol that can now be frozen

1. Use I0006/Emory only for the first untouched external test. Do not mix I0001/MGH into protocol development.
2. Define labels from physician-overread codes using the core mapping above. Report strict definitions as prespecified sensitivity analyses and software labels only as a separate provenance sensitivity analysis. Do not use v24 as primary.
3. Exclude records without `BDSPPatientID`, empty physician codes, unreadable waveforms, wrong lead count/order, unsupported units, or duration outside a predeclared 10-second tolerance. Report every exclusion count.
4. Exclude the 23 readiness-sample patients from the final untouched test.
5. Select one recording per remaining patient by a frozen SHA-256 rank based only on a public seed and stable identifiers, without using labels. This aligns the unit with the prior one-record-per-patient evaluations and avoids overweighting patients with up to 106 ECGs.
6. Accept 250 Hz and 500 Hz 10-second recordings. Resample 250 Hz to 500 Hz with one frozen polyphase implementation before the existing physical-reversal and ECGFounder/ResNet preprocessing. Stop on any other sampling rate until a rule is frozen without outcome access.
7. Apply the already frozen clean and RA-LL electrical transformation and the existing model/checkpoints without HEEDB training, fine-tuning, threshold selection, or mapping changes.
8. Reuse validation-frozen thresholds from the development cohorts. Report the same metrics and paired patient bootstrap contrasts, with a separately labeled label-source sensitivity analysis.
9. Freeze the complete eligible-record manifest and its hash before inference. The full waveform transfer or in-account computation requires a separate approved storage plan because it is outside this 2 GiB readiness task.

## Final readiness statement

HEEDB I0006 has usable patient identifiers, complete acquisition-to-metadata joins, sufficient physician-overread positives for all three outcomes, and readable paired WFDB files. The data are ready for a fair external-validation protocol to be frozen under the explicit mapping and sampling rules above. They are not yet a completed external test, and the label crosswalk still needs clinical interpretation in any publication because it is not an official 12SL-to-SNOMED equivalence map.
