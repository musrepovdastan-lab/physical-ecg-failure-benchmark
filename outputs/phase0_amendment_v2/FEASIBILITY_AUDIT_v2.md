# Phase-0 amendment v2 — Chapman–Shaoxing / Ningbo

## Decision: GO

The v1 NO-GO remains correct for the specific Challenge 2020 sources it audited. It is not a final NO-GO for the research question. This amendment identifies two separately usable public Challenge 2021 cohorts: Chapman–Shaoxing as development and Ningbo as sealed external confirmation.

## Source-eligibility evidence

The primary Challenge 2021 paper’s Table 2 reports 10,247 patients / 10,247 training recordings for Chapman–Shaoxing and 34,905 / 34,905 for Ningbo. Equal counts are the authoritative one-record-per-patient guarantee for this protocol. Table 1 identifies distinct institutions—Shaoxing People’s Hospital and Ningbo First Hospital—in different cities in Zhejiang, China. They are separate from PTB and PTB-XL sources in the same table. The official release reports complete twelve-lead public training waveforms, 500 Hz, 10 seconds, in WFDB header + MATLAB v4 format and CC BY 4.0.

The sources must never be pooled: Chapman is development only; Ningbo is a sealed external confirmation cohort. The source-level data are public, but this is a study-specific analysis seal, not a claim that the public records are technically inaccessible.

## Metadata and label audit

The official paper gives exact source-wise counts for 30 scored approximate SNOMED labels in Figure 4, plus the lead order, sample rate, duration, mean age, and sex summaries in Table 4. `LABEL_COMPATIBILITY_AUDIT.csv` preserves the key endpoint decisions.

| Candidate | Chapman | Ningbo | Decision |
|---|---:|---:|---|
| AF (164889003) | 1,780 | 0 | reject; not shared |
| Atrial flutter (164890007) | 445 | 7,615 | reject; Chapman <500 |
| First-degree AV block (270492004) | 247 | 893 | reject; Chapman <500 |
| RBBB (59118001) | 454 | 195 | reject; both source gates fail |
| Sinus bradycardia (426177001) | 3,889 | 12,670 | **primary** |
| Sinus tachycardia (427084000) | 1,568 | 5,687 | **primary** |
| T-wave abnormality (164934002) | 1,876 | 5,167 | eligible secondary only |

The primary endpoints are two clinically meaningful, distinct rate-rhythm conditions. This is less aligned with the initial preference for AF/conduction but is the honest result of applying the frozen count gates: AF is absent from Ningbo and individual conduction labels are too sparse in Chapman. Do not replace them with a post hoc union label without a further amendment and exact header-level union audit.

With the frozen 70/10/20 Chapman record allocation, the stratification target is exactly 778 sinus-bradycardia and 313 sinus-tachycardia positive records in internal test, both above 200. Because every recording is a different patient, record-level counts are patient counts.

The headers separately contain age, sex, diagnoses, lead order, sample rate, and samples; a headers-only audit script is included. It was not allowed to continue after the environment detached its foreground network process. Thus exact header-level age/sex missingness, multiple-diagnosis count, and malformed-header count are pending the Phase-1 ingestion checkpoint. Non-finite waveform count cannot be learned from headers and requires reading signals; it remains zero-claim/unknown in Phase 0. This is documented rather than fabricated and is not one of the amended GO gates.

## Fault and control audit

The v1 exact terminal equations remain unchanged. V2 resolves the comparator ambiguity:

- **Family A, availability:** operate only on the eight independent signals `I`, `II`, `V1`–`V6`; reconstruct displays consistently; match random conditions by unavailable independent measurement count; calculate each random metric separately inside each bootstrap draw.
- **Family B, connection error:** RA↔LA, RA↔LL, and LA↔LL terminal permutations preserve information and receive no dropout control. Compare against clean performance and test whether ranks under random dropout predict ranks under each exact reversal.

No open circuits, impedance changes, motion artefacts, or spatial chest-electrode misplacements are simulated.

The existing unit tests specify Einthoven/Goldberger identities and every implemented transformation. All four named test functions passed during this amendment using the project runtime plus a transient read-only NumPy path; no legacy artefact was modified. The host still lacks pytest and its runtime lacks `venv`, so `pytest` command-line execution remains a Phase-1 setup checkpoint. This is an environment limitation, not a transformation change.

## Deterministic splits and seal

The frozen implementation allocates Chapman record IDs with deterministic multilabel stratification, seed `20260828`, target proportions 0.70/0.10/0.20, and exact primary-label test allocation targets described above. It writes record IDs, source, split, primary-label bits, and SHA-256 of the sorted membership table before any internal-test prediction. No model outcome is inspected during split generation.

Ningbo may be read only to audit header integrity and eligibility labels at ingestion. Its record-level labels, fault outcomes, predictions, thresholds, and ranking results remain sealed until the Chapman model, fault library, code hash, and reporting plan are frozen.

## Compute audit

The first pilot trains a single lightweight residual CNN on Chapman only (10,247 10-second × 500-Hz recordings, downsampled to 100 Hz), one seed, maximum 30 epochs with mixed precision. Clean + Family-A random/physical availability + Family-B reversal inference occurs only on Chapman internal test. Estimate: **4–6 GPU-hours**, **6–10 wall-clock hours**, <6 GB VRAM; this meets the ≤8 GPU-hour and ≤12-hour target on the local RTX 5070 Laptop GPU (8 GB).

The later protocol is two architectures × three seeds trained only on Chapman, followed by one sealed Ningbo inference pass per frozen model and bootstrap. Estimate: **27–39 GPU-hours** and **3–5 wall-clock days**, below the earlier 48-GPU-hour ceiling. Reserve 15–20 GB working storage; the official whole Challenge release is 12.6 GB uncompressed, while selective Chapman/Ningbo acquisition is expected to be substantially smaller. Exact per-folder bytes must be recorded from the official SHA256 manifest at ingestion.

## GO-gate audit

| Gate | Result |
|---|---|
| One-record-per-patient claims verified | PASS |
| Distinct non-PTB cohorts | PASS |
| Two common meaningful endpoints pass counts | PASS: SB and STach |
| Compatible waveform/label representation | PASS: 12-lead, 500 Hz, 10 s, WFDB/SNOMED |
| Physical transformations / matched controls | PASS as specified; execution check required at environment setup |
| Chapman one-seed pilot ≤8 GPU-hours | PASS |
| Different from located 2024–26 work | PASS, narrowly; it is an evaluation protocol, not an augmentation claim |

## Authoritative sources

- [Official Challenge 2021 v1.0.3](https://physionet.org/content/challenge-2021/1.0.3/)
- [Challenge source/population paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/)
- [Challenge paper PDF, Tables 1–5 and Figure 4](https://physionet.org/files/challenge-2021/1.0.2/papers/2021ChallengePaperPMEA.pdf?download=)
