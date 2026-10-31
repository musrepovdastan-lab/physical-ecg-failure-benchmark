"""Freeze the v2 analysis extension before generating Huang validation scores."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "mixed_fault_calibration_v2"
V1 = ROOT / "outputs" / "mixed_fault_calibration_v1"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
SEEDS = (104729, 130363, 155921)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_hash(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "FROZEN_COMPLETION_PROTOCOL.json").exists():
        raise FileExistsError("v2 protocol already exists; refusing to overwrite a frozen protocol")
    canonical_path = N / "CANONICAL_RECORDS.json"
    membership_path = N / "SPLIT_MEMBERSHIP.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))["records"]
    membership = json.loads(membership_path.read_text(encoding="utf-8"))["rows"]
    by_id = {r["record_id"]: r for r in canonical}
    validation = [by_id[r["record_id"]] for r in membership if r["split"] == "validation"]
    test = [by_id[r["record_id"]] for r in membership if r["split"] == "test"]
    if len(validation) != 3488 or len(test) != 6981:
        raise RuntimeError(f"unexpected frozen Ningbo split sizes: validation={len(validation)} test={len(test)}")
    split = np.load(V1 / "calibration_split_indices.npz", allow_pickle=False)
    if len(split["A"]) != 1744 or len(split["B"]) != 1744:
        raise RuntimeError("v1 calibration A/B split sizes differ from protocol")
    input_paths = [
        canonical_path,
        membership_path,
        V1 / "FROZEN_ANALYSIS_PROTOCOL.json",
        V1 / "FROZEN_PROTOCOL_AMENDMENT_001.json",
        V1 / "FROZEN_PROTOCOL_AMENDMENT_002.json",
        V1 / "calibration_split_indices.npz",
        V1 / "CALIBRATION_AND_PROVENANCE.json",
        V1 / "BOOTSTRAP_METRICS_REPLICATES.npz",
        V1 / "BOOTSTRAP_METRICS_SUMMARY.csv",
        ROOT / "src" / "published_detector_pipeline_v1.py",
        ROOT / "src" / "huang_detector_infer_v1.py",
        ROOT / "src" / "huang_detector_infer_v2.py",
        ROOT / "src" / "ecg_physics.py",
        ROOT / "src" / "chapman_phase1.py",
        ROOT / "src" / "ningbo_confirmation_v1.py",
        ROOT / "outputs" / "mixed_fault_calibration_v1" / "run_analysis.py",
        ROOT / "outputs" / "mixed_fault_calibration_v1" / "extend_bootstrap_metrics.py",
        OUT / "infer_validation.py",
        OUT / "detector_validation_worker.py",
        OUT / "freeze_protocol.py",
        ROOT / "outputs" / "published_detector_pipeline_v1" / "FROZEN_PROTOCOL.json",
        ROOT / "outputs" / "published_detector_pipeline_v1" / "REPRODUCIBILITY_MANIFEST.json",
        ROOT / "outputs" / "published_detector_pipeline_v1" / "DETECTOR_AUDIT.json",
        ROOT / "outputs" / "published_detector_pipeline_v1" / "cache" / "detector_test_probs.npz",
    ]
    for seed in SEEDS:
        input_paths.extend([
            ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/checkpoint.pt",
            ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz",
            ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz",
            ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz",
            ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz",
        ])
    detector_python = ROOT / "outputs/published_detector_pipeline_v1/detector_env/Scripts/python.exe"
    audit = subprocess.run(
        [str(detector_python), str(ROOT / "src/huang_detector_infer_v2.py"), "--audit-only"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    audit_line = next(line for line in reversed(audit.stdout.splitlines()) if line.startswith("{"))
    audit_data = json.loads(audit_line)
    checkpoint = Path(audit_data["checkpoint"])
    tf_version = audit_data["tensorflow_version"]
    detector_assets = [
        ROOT / "outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable/LDenseNet.index",
        ROOT / "outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable/LDenseNet.data-00000-of-00001",
        ROOT / "outputs/published_detector_pipeline_v1/official_source/ECG_lead_check_main.zip",
    ]
    input_paths.extend(detector_assets)
    missing = [str(p) for p in input_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("required frozen inputs missing: " + ", ".join(missing))
    files = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in input_paths}
    validation_binding = [{"record_id": r["record_id"], "labels": r["labels"], "waveform_path": r["waveform_path"]} for r in validation]
    test_binding = [{"record_id": r["record_id"], "labels": r["labels"], "waveform_path": r["waveform_path"]} for r in test]
    config = {
        "version": "mixed_fault_calibration_v2",
        "status": "FROZEN_BEFORE_VALIDATION_INFERENCE",
        "purpose": "complete four-method Huang Ningbo validation calibration and missing paired TWA intervals; exploratory extension of previously accessed results",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "v1_immutable": True,
        "no_diagnostic_training": True,
        "seeds": list(SEEDS),
        "population": {
            "dataset": "Ningbo frozen v2",
            "validation_n": len(validation),
            "validation_order_sha256": rows_hash(validation_binding),
            "test_n": len(test),
            "test_order_sha256": rows_hash(test_binding),
            "validation_split": "SPLIT_MEMBERSHIP.json rows with split=validation, preserved order",
            "calibration_split": "reuse v1 calibration_split_indices.npz unchanged; A=1744, B=1744, global validation positions",
        },
        "pipeline": {
            "detector": "published Huang LDenseNet; frozen weights, v2 preprocessing (SOS Butterworth 0.5-35Hz order 5, sosfiltfilt, resample to 1200, per-lead z-score; channels III/aVR/V6); softmax argmax",
            "correction": "always correct using predicted detector class, including clean predictions; class map clean, RA_LA, RA_LL, LA_LL; exact frozen published_detector_pipeline_v1 correction function",
            "diagnostic": "three frozen Ningbo clean-trained ResNet checkpoints, seeds 104729/130363/155921; 100Hz ECG, same normalization and CUDA autocast path as prior Huang inference",
            "detector_runtime": "published_detector_pipeline_v1/detector_env, TensorFlow " + str(tf_version),
            "diagnostic_runtime": "project .venv PyTorch; CUDA device if available",
            "replay_tolerances": {"detector_max_abs_probability": 1e-5, "diagnostic_max_abs_probability": 1e-4, "frozen_before_replay": True},
            "input_handoff": "float32 four-class detector probabilities; argmax integer class vector in frozen record order; indices passed to existing correct_by_class; corrected 100Hz arrays normalized by each seed's frozen training mean/std",
        },
        "calibration": {
            "method": "Platt sigmoid(a*logit(p)+b), a>0; exact v1 implementation, clipping, precision, optimizer, bounds, convergence rules",
            "fit": "clean Huang full-pipeline scores on validation A only; ordinary unweighted log loss",
            "raw_standardized_threshold": "90th percentile of negative Huang full-pipeline raw scores on validation B",
            "calibrated_threshold": "same validation-B raw threshold transformed by that fitted Huang calibrator",
            "threshold_modes": ["raw_historical_baseline", "raw_validation_B_90pct", "calibrated_validation_B_90pct"],
            "apply": "each seed and endpoint calibrator/threshold applied unchanged to clean and RA_LL-reversed Ningbo and HEEDB scores at every q",
            "existing_three_method_calibrators_and_split_unchanged": True,
        },
        "mixture": {"q": [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0], "sources": ["Ningbo", "HEEDB_I0006_independent_confirmation"], "primary": "TWA", "metrics": ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "false_positives_per_1000", "false_negatives_per_1000"]},
        "bootstrap": {
            "replicates": 1000,
            "clusters": {"Ningbo": "canonical record", "HEEDB_I0006_independent_confirmation": "patient"},
            "rng": "numpy.default_rng(20261017); regenerate source-ordered draws (Ningbo first, HEEDB second) and require exact reproduction of saved v1 absolute metric replicates for all three existing methods before using the v1 saved absolute Huang Ningbo historical-threshold replicates and calculating previously missing Huang paired contrasts",
            "seed_aggregation": "metric per seed then arithmetic mean; never average prediction vectors",
            "fixed": "calibrators, thresholds and checkpoints; paired clean/reversal multiplicity shared",
            "interval": "percentile 2.5%,97.5%; conditional on selected models and fitted calibrators",
            "reuse": "preserve all available v1 summaries and replicates; add only missing Ningbo Huang paired contrasts and new Huang validation-B/calibrated Huang intervals",
        },
        "sources": files,
        "detector_checkpoint": {"path": str(checkpoint), "sha256": sha(Path(str(checkpoint) + ".index"))},
        "output_predictions": "huang_validation_scores.npz in this directory",
        "reserved_test": "not evaluated beyond previously cached short adapter replay",
    }
    path = OUT / "FROZEN_COMPLETION_PROTOCOL.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"FROZEN {path} sha256={sha(path)} validation={len(validation)} test={len(test)} sources={len(files)}")


if __name__ == "__main__":
    main()
