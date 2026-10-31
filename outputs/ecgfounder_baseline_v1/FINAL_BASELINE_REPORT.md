# ECGFounder baseline v1 — stopped before a valid comparison

**Decision: `STOP_PREPROCESSING_MISMATCH`.** This exploratory run cannot answer
whether RA–LL reversal causes a diagnostic loss in ECGFounder on both datasets.
No ECGFounder–ResNet effect estimate or patient-level interval is reported.

The authors' 12-lead checkpoint loaded strictly into the official 150-class
architecture. The protocol was frozen before new held-out extraction; three
electrical/preprocessing tests and a one-record train smoke test passed. Chapman
training and validation feature extraction, head selection, and held-out feature
extraction completed. Ningbo stopped in **training** after 80/24,430 records;
neither Ningbo validation nor held-out ECGFounder recordings were inferred on.
Chapman held-out ECGFounder metrics are withheld because the required paired
two-dataset baseline did not complete.

## Exact obstacle

At Ningbo `g1/JS10767`, four V6 digital samples equal `-32768`. WFDB converts
them to NaN physical values. The frozen v1 adapter stops on nonfinite input,
but the authors' `dataset.py` explicitly replaces NaNs with zero **before**
filtering. A training-only scan found 59/24,430 records with 1,861 such samples
([audit](NINGBO_TRAIN_MISSING_AUDIT.json)). This is a real preprocessing
disagreement, so continuing v1 would violate its frozen input rule.

The v1 adapter also derives Ningbo waveform paths from record IDs. The canonical
mapping shows 24 training records whose actual waveform folder differs from
the ID's group, such as `g11/JS21646` at `training/ningbo/g12/JS21646.mat`.
The existing project correctly uses the canonical waveform path. This adapter
would therefore have stopped again even if the NaN rule had been changed.
Both are local adapter issues, not evidence of a checkpoint failure or model
robustness. No v1 protocol, prior experiment, or M3 artifact was changed to
work around them.

## Saved ResNet reference, with ECGFounder unevaluated

Values are mean **metrics** over the three saved clean-trained ECGResNet-small
seeds on each existing held-out set. Drop is clean minus RA–LL AUPRC. These are
existing ResNet results, not new ECGFounder findings. The companion
[side-by-side CSV](BLOCKED_SIDE_BY_SIDE.csv) contains AUROC and sensitivity at
each seed's validation-fixed threshold for all three labels.

| Dataset | Label | ResNet clean AUPRC | ResNet RA–LL AUPRC | ResNet drop | ECGFounder |
|---|---|---:|---:|---:|---|
| Chapman | TWA | 0.653 | 0.328 | 0.325 | Not evaluated |
| Chapman | Sinus bradycardia | 0.999 | 0.994 | 0.006 | Not evaluated |
| Chapman | Sinus tachycardia | 0.986 | 0.756 | 0.230 | Not evaluated |
| Ningbo | TWA | 0.676 | 0.182 | 0.494 | Not evaluated |
| Ningbo | Sinus bradycardia | 0.996 | 0.957 | 0.039 | Not evaluated |
| Ningbo | Sinus tachycardia | 0.986 | 0.702 | 0.284 | Not evaluated |

The [label audit](LABEL_MAPPING_AUDIT.md) found no exact 150-class zero-shot TWA
output. The named sinus bradycardia and tachycardia outputs are name-level
matches and were slated for separate reporting. No zero-shot or trained-head
ECGFounder outcome metrics are released from this stopped run.

Examples of changes in the **existing ResNet** predictions are in
[SAVED_RESNET_CHANGED_EXAMPLES.csv](SAVED_RESNET_CHANGED_EXAMPLES.csv); they must
not be interpreted as ECGFounder examples. For instance, Chapman
`g3/JS02774` has TWA truth 0, ResNet seed 104729 clean probability 0.0053 and
RA–LL probability 0.9712 at its fixed 0.7041 threshold. The available evidence
does not establish whether this failure generalizes to ECGFounder, and it makes
no claim of superiority over published models.

Reproducibility: [frozen protocol](../../configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v1.json),
[manifest](REPRODUCIBILITY_MANIFEST.json), [pre-run audit](PRE_RUN_AUDIT.md),
[progress logs](ningbo_stdout.log), and [final status](FINAL_STATUS.json).
