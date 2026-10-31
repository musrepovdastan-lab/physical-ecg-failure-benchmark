# Final Phase-0 decision v2: GO

## Amended question

> Do conventional random lead-dropout evaluations predict the diagnostic robustness of 12-lead ECG classifiers under a preregistered library of physically realizable acquisition faults?

## Why the decision changed

The original NO-GO was correctly limited to Challenge 2020 CPSC/CPSC-Extra/Georgia sources, which lack authoritative patient linkage. The official Challenge 2021 population table adds two eligible non-PTB cohorts: Chapman–Shaoxing (10,247 records from 10,247 patients) and Ningbo (34,905 from 34,905), from distinct hospitals. Both are public, CC BY 4.0, full 12-lead, 500-Hz, 10-second records and use compatible approximate SNOMED labels.

## Frozen source roles

- **Development:** Chapman–Shaoxing only.
- **Sealed confirmation:** Ningbo only; no Phase-0 waveform access, inference, tuning, thresholds, or fault selection.

## Frozen primary endpoints

AF and individual conduction labels fail the source-count gates. The two primary outcomes are therefore **sinus bradycardia** (3,889 Chapman; 12,670 Ningbo) and **sinus tachycardia** (1,568; 5,687). The Chapman internal-test allocation targets are 778 and 313 positive patients, respectively. T-wave abnormality is eligible secondary only.

## What Phase 1 may do

After completing the headers-only integrity checkpoint and a standard `pytest` environment check (the four mathematical test functions pass in the present audit), train one lightweight model/one seed on Chapman only. Evaluate separately:

1. availability failures versus random controls matched by unavailable **independent** measurement count; and
2. exact limb-terminal reversals versus clean performance, testing whether random-dropout robustness ranks predict reversal robustness.

This is GO for an evaluation/failure-analysis pilot. It makes no assertion that random dropout is necessarily optimistic and no augmentation-success claim.

## Remaining pre-ingestion checks

Exact age/sex missingness, multi-diagnosis count, malformed-header count, exact folder byte totals/checksum matches, and waveform non-finite count are intentionally not invented. The supplied headers-only audit can obtain the header fields without `.mat` files; non-finite checking must be done only when waveform acquisition is authorised for Phase 1.

No scientific training, external inference, waveform download, or background job was started in this amendment.
