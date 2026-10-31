# Final Phase-0 decision: NO-GO

## Exact research question

Do evaluations based on random whole-lead masking systematically overestimate the robustness of 12-lead ECG classifiers to physically realizable electrode failures?

## Decision and exact blocker

**NO-GO.** The required independent replication cannot be prespecified honestly with official Challenge 2020 v1.0.2 sources. The usable non-PTB folders (CPSC/CPSC-Extra and Georgia) publish recording IDs and clinical header fields but no authoritative patient IDs. The official Challenge documentation also says that a small number of patients may appear in both training and test data. Because repeated patients cannot be ruled out, these sources are ineligible under this project’s rule requiring patient-cluster inference whenever patient IDs exist and exclusion from confirmatory inference when authoritative IDs are unavailable. Therefore there are **zero**, not two, eligible non-PTB confirmation sources.

No waveform files, full downloads, training, background jobs, or sealed confirmation outcomes were accessed in Phase 0.

## What the PTB-XL evidence already supports

The corrected read-only legacy analysis supports an exploratory motivation result, not the physical-failure claim: in PTB-XL fold 9, V1–V4 displayed-lead zeroing reduced MI raw AUPRC by 18.1, 19.4, and 11.0 percentage points relative to a fair random four-displayed-lead comparator across three seeds. The corrected analysis used patient-cluster bootstrap, separately calculated random-mask metrics within each bootstrap draw, and all 495 fixed four-lead removal controls.

The previous structured-mask training method failed its held-out stressor gate; it did not repair the vulnerability and is not presented as successful.

## What remains unverified

- Whether the effect survives an external, sealed, patient-separable non-PTB dataset.
- Whether a physically exact limb reversal or true electrode availability failure has the same or greater impact than random corruption matched by independent measurement impact.
- Fault prevalence and clinical consequences in actual acquisition workflows.
- Any beneficial mitigation/augmentation method.

## Strongest related work and difference

- [Oh et al. 2022](https://proceedings.mlr.press/v174/oh22a.html): random lead masking for SSL, not an exact physical-fault-versus-matched-random diagnostic robustness benchmark.
- [Bo et al. 2022](https://arxiv.org/abs/2211.07889): adversarial temporal masking for SSL, not electrode/cable transformations or clinical-fault degradation.
- [Huang et al. 2024](https://pubmed.ncbi.nlm.nih.gov/38663434/): detects lead misplacement, rather than comparing a separate diagnostic classifier’s fault degradation with matched random corruption.

The targeted search did not locate a 2024–2026 paper with the full matched physical-fault-versus-random design. That is a bounded literature finding, not a claim that no equivalent work exists.

## Candidate external data and counts

At record level, CPSC (6,877) and Georgia (10,344) are public, non-PTB candidates. Official Figure-1 counts include CPSC IAVB 722 / AF 1,221 / RBBB 1,857 and Georgia IAVB 769 / AF 570 / RBBB 542 / sinus rhythm 1,752. CPSC-Extra has 3,453 recordings but sparse endpoint overlap. These are not patient counts, and none of these sources is eligible for confirmatory patient-level inference. PTB and PTB-XL are excluded for external replication.

## Resources and user action

Phase 0 used 0 MB of downloads. The full Challenge release is 4.5 GB compressed / 7.5 GB uncompressed; reserve 15 GB for a later eligible two-source working set. A redesigned one-seed pilot is estimated at 8–11 GPU-hours (12–18 wall-clock hours) and the full program at 36–45 GPU-hours (3–5 days) on the RTX 5070 Laptop GPU.

**Action required from you:** identify or authorize a search for two public non-PTB 12-lead datasets with authoritative patient IDs (or explicit one-record-per-patient guarantees) and compatible labels. Do not download Challenge 2020 waveforms for confirmation; doing so cannot repair the patient-separation blocker.
