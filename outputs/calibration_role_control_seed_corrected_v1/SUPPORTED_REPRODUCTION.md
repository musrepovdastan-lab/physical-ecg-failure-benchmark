# Supported reproduction commands

Run these from the project root with the existing `.venv`.

## Input-only check (safe in the completed package)

```powershell
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\train_all_fresh.py --check-inputs
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\verify_input_provenance.py
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\verify_archive_repack.py
```

These commands validate the exact supported trainer input path and repack the two existing per-cohort bootstrap archives to a temporary location for exact comparison. They do not train, run inference, fit calibration, set thresholds, or recalculate bootstrap statistics.

## Full reproduction on a fresh prepared copy only

Use an isolated copy of the prepared package with no existing `fits/**/checkpoint.pt`; the launcher deliberately refuses to replace completed checkpoints.

```powershell
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\initialization_audit.py
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\rng_resume_audit.py
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\train_all_fresh.py --check-inputs
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\train_all_fresh.py
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\postprocess.py
.venv\Scripts\python.exe outputs\calibration_role_control_seed_corrected_v1\evaluate_full.py
```

The frozen original `control.py` holds the `train_one()` implementation used for the nine saved fits and remains byte-bound to the original protocol and checkpoint metadata. `control_replay.py` only validates input provenance and returns the unchanged frozen protocol to `train_all_fresh.py`; it does not change training behavior. The post-fit NPZ writer in the original `control.py run` is known to fail after the nine training gates, so that historical launcher is not the supported reproduction command. `postprocess.py` is the supported clean A/B scoring and calibration stage. `evaluate_full.py` now enforces the canonical 700-key archive before writing.

The final report's scientific conclusions, checkpoints, predictions, calibration values, thresholds and paired bootstrap draws were created before this packaging amendment and have not been regenerated here.
