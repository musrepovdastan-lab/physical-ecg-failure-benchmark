# ECGFounder baseline v1 — pre-run audit

The new comparison is exploratory because the existing Chapman and Ningbo held-out
sets and ResNet results have already been examined. No new split, label, record
exclusion, or robustness method is introduced.

The frozen Chapman v4 membership contains 7,163 train, 1,021 validation, and 2,050
test records; the frozen Ningbo v2 membership contains 24,430, 3,488, and 6,981.
The endpoint order is TWA, SB, STach. The canonical split artifacts have no record
ID overlap; Ningbo has no waveform-hash overlap, and the earlier Chapman split
manifest reports no waveform-hash overlap after excluding 13 duplicate exports.
The new run follows the project's existing one-record-per-patient/canonical-cluster
interpretation. Exact counts and SHA-256 bindings are in `PRE_RUN_AUDIT.json` and
the frozen protocol.

Saved ECGResNet-small clean and RA–LL prediction arrays exist for seeds 104729,
130363, and 155921 on both held-out sets. Their row order follows the sorted
split membership used by the original evaluators. The read-only audit verified
each array's shape and finite values and recomputed AUPRC and AUROC from the
unchanged split labels. The exact saved-file hashes are frozen in the protocol.

M3 is unfinished: its `WORKFLOW_STATUS.json` remains at Chapman `EVALUATING`,
seed 155921, physics-aware arm, condition 794/794, with bootstrap 0/1,000.
Its recorded PID was not running at this audit. No M3 file or process was changed.

The authors' official code is pinned to commit
`04edac702b61c91face519774ddcc0cd712fef23`. The public 12-lead checkpoint
SHA-256 is `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`.
The authors' `dataset.py` requests `filter_bandpass` followed by one global z-score
over the signal. Its classification branch contains an undefined `sample_rate`
variable and an unused resampling result. These recordings are already 500 Hz
and exactly 5,000 samples long, so the baseline implements the documented filter
and normalization operations directly and performs no resampling. The authors'
PTB-XL validation script has a different path without filtering; this run follows
the repository README's instruction to follow `dataset.py` for pretrained use.
The test suite compares our filter numerically against the function extracted
from pinned `util.py`.

The new protocol was frozen before ECGFounder evaluation on held-out records.
Only a train-record preprocessing/inference smoke test follows the freeze.
