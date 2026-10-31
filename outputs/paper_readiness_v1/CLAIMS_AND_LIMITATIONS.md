# Claims and limitations (corrected)

## Supported empirical claim

In frozen Chapman v7 and independent Ningbo v2 evaluations, exact RA–LL terminal reversal was more damaging to AUPRC than the mean of all 792 conventional five-displayed-lead masking conditions for TWA in every seed. The numeric results and bootstrap intervals are unchanged in `CROSS_DATASET_RESULTS.csv`.

## Corrected fair comparison

RA–LL reversal uses terminal permutation `(2, 1, 0)`. The changed displayed leads are **I, II, III, aVR and aVF**. **aVL and V1–V6 are unchanged** to numerical tolerance; the Wilson central terminal is invariant to terminal permutation.

The comparator spans all C(12,5)=792 conventional five-displayed-lead masks because the reversal changes exactly five displayed channels. It is **consequence-matched**, not waveform-equivalent: masking substitutes a frozen train-mean sentinel, while a physical terminal reversal preserves signals and changes lead algebra. An explicit missing-lead sentinel is separate unavailability representation, not a reversal.

## Limitations

- Results concern the two frozen datasets, architecture, endpoints and deterministic transformation.
- Cached predictions establish output sensitivity, not the causal model feature responsible for it.
- No causal conclusion about STach is warranted from performance loss alone.
