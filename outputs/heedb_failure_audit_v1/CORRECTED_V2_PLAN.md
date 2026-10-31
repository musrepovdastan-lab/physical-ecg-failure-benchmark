# Corrected HEEDB external evaluation v2: frozen plan

Status: plan frozen after the read-only v1 failure audit; **no corrected inference has run**. A separate executable protocol and code hash must be sealed before any v2 inference. Preserve v1 exactly as published locally.

## Fixed population and artifacts

- Reuse the same 10,000 patients, one ECG per patient, physician `codes_physician` core and strict mappings, exclusions, cached waveforms, and label-blind order in the v1 final selection. Do not replace records or inspect predictions to change the cohort.
- Reuse the exact six Ningbo ECGResNet checkpoints, their own preprocessing arrays and validation thresholds, the clean/RA–LL conditions, and the same deterministic 24-mask subset. Verify all against `configs/FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json` and `outputs/heedb_external_v1/RUN_PROVENANCE.json` before use.
- No S3 access, new data, training, fine-tuning, recalibration, threshold tuning, or changes to label mappings. Keep record-level predictions in a new git-ignored restricted directory.

## The two corrections only

1. Temporal input: the Ningbo training loader uses `loadmat(...)["val"][:, ::5]`, yielding 1,000 samples from each 500 Hz, 5,000-sample waveform. The cached HEEDB v1 waveform is already in microvolts at 500 Hz and has 5,000 samples. Feed `cache[:, :, ::5]` to every v2 condition, yielding exactly 1,000 samples and an effective 100 Hz. This also implements the prespecified 250 Hz path: original 2,500 samples → `resample_poly(up=2, down=1)` → 5,000 → `::5` → 1,000. Preserve the existing mV × 1000 conversion and per-lead Ningbo normalization.
2. RA–LL algebra: use the existing, hashed `src.ecg_physics.named_reversal` followed by `derive_display` on independent leads I, II, V1–V6. From original I and II, the correct displayed leads are I′=I−II, II′=−II, III′=−I, aVR′=II−I/2, aVL′=I−II/2, aVF′=−(I+II)/2; V1–V6 are unchanged. The v1 aVR′ and aVF′ equations were wrong. Apply reversal before per-lead normalization. Verify all 12 outputs against the physics library on synthetic basis signals and cached sample ECGs.

## Freeze and run gates

1. Implement a new v2 evaluator without modifying v1 code or outputs. Freeze and SHA-256 hash a separate executable protocol, evaluator source, selected manifest, source cache, six checkpoints, six preprocessing files, six validation threshold files, physics library, and mask panel before any inference.
2. Smoke test the 23 evaluation-excluded readiness ECGs, including both sampling rates, all six checkpoints and all 26 conditions. Require 12×1,000 finite normalized inputs, finite three-channel probabilities, and exact RA–LL algebra within float32 tolerance. A model's variable-length acceptance is not a sufficient smoke gate.
3. Evaluate the unchanged 10,000-patient sample once; keep progress outcome-blind. Recompute the original primary and separate secondary contrasts with the original frozen thresholds and 1,000 synchronized patient bootstraps. Do not use v1 outcomes to choose settings.
4. Save v2 under `outputs/heedb_external_corrected_v2/` and restricted record-level material under `data/heedb_restricted/I0006/external_corrected_v2/`. Report v1 and corrected v2 side by side, identify v2 as a post-hoc implementation correction, and do not replace v1. Judge HEEDB transportability only after this corrected analysis.
