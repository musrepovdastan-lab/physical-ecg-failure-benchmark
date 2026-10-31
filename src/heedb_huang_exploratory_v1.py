"""Post-hoc HEEDB independent-cohort comparison with the unchanged Huang detector.

This never mutates the completed confirmation. Record-level arrays remain private.
Freeze must run before detector or diagnostic comparator inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import heedb_independent_confirmation_v1 as previous
from src import heedb_external_corrected_v2 as corrected_v2
from src import heedb_external_v1 as heedb_v1
from src.heedb_huang_adapter_v1 import (
    HUANG_CLASSES, corrected_batch, detector_input, diagnostic_input,
    electrical_reversal, electrical_reversal_batch,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_huang_exploratory_v1"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "huang_exploratory_v1"
SOURCE = previous.PRIVATE / "signals_10000_float32.npy"
PRIOR_PREDICTIONS = previous.PRIVATE / "predictions_float32.npy"
DETECTOR = PRIVATE / "detector_probabilities_float32.npy"
HUANG_PREDICTIONS = PRIVATE / "huang_diagnostic_probabilities_float32.npy"
PROTOCOL = OUT / "FROZEN_PROTOCOL.json"
SEEDS = previous.SEEDS
CONDITIONS = ("clean", "RA_LL_reversal")
ARMS = ("baseline", "physics_aware", "huang_detect_correct_clean_resnet")
METRICS = ("auprc", "auroc", "sensitivity", "specificity", "predicted_positive_fraction", "mean_score", "brier", "ece15")
BOOT = 1000
BOOT_SEED = 20260926


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path.name}")
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log(stage: str, **fields: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"utc": timestamp(), "stage": stage, **fields}
    with (OUT / "PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def artifacts() -> dict[str, str]:
    audit = json.loads((ROOT / "outputs" / "published_detector_multifault_v2" / "WEIGHT_AND_SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    weights = audit["weight_files"]["windows_loadable_copy"]
    expected = {item["path"].replace("\\", "/"): item["sha256"].lower() for item in weights.values()}
    if len(expected) != 2 or any(digest(ROOT / rel) != value for rel, value in expected.items()):
        raise RuntimeError("published unmodified detector weights no longer match source audit")
    return expected


def upstream_gate() -> tuple[dict, pd.DataFrame]:
    frozen, selected = previous.frozen()  # identifiers only, no outcome read
    completed = json.loads((previous.OUT / "ANALYSIS_COMPLETE.json").read_text(encoding="utf-8"))
    provenance = json.loads((previous.OUT / "RUN_PROVENANCE.json").read_text(encoding="utf-8"))
    if completed["decision"] != "PASS" or len(selected) != 10000:
        raise RuntimeError("independent confirmation is not complete")
    if digest(SOURCE) != provenance["waveform_cache_sha256"] or digest(PRIOR_PREDICTIONS) != provenance["prediction_artifact_sha256"]:
        raise RuntimeError("completed independent waveform or prediction cache hash mismatch")
    if "data/heedb_restricted/" not in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        raise RuntimeError("restricted patient-level data are not git-ignored")
    raw = np.load(SOURCE, mmap_mode="r")
    prior = np.load(PRIOR_PREDICTIONS, mmap_mode="r")
    if raw.shape != (10000, 12, 5000) or prior.shape != (2, 3, 2, 10000, 3):
        raise RuntimeError("frozen independent cache geometry changed")
    return frozen, selected


def freeze() -> None:
    if PROTOCOL.exists() or (OUT / "FROZEN_PROTOCOL_SHA256.txt").exists() or DETECTOR.exists() or HUANG_PREDICTIONS.exists():
        raise RuntimeError("comparator already frozen or run; refusing a second freeze")
    if subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_heedb_huang_adapter_v1.py"], cwd=ROOT).returncode:
        raise RuntimeError("adapter automated tests failed")
    upstream, selected = upstream_gate()
    source = np.load(SOURCE, mmap_mode="r")
    for index in range(16):
        raw = np.asarray(source[index])
        processed = detector_input(raw)
        swapped = electrical_reversal(raw, "RA_LL_reversal")
        if processed.shape != (1200, 3) or detector_input(swapped).shape != (1200, 3):
            raise RuntimeError("real-cache Huang adapter smoke failed")
        if diagnostic_input(raw, np.zeros(12), np.ones(12)).shape != (12, 1000):
            raise RuntimeError("real-cache diagnostic geometry smoke failed")
    weight_hashes = artifacts()
    protocol = {
        "version": "heedb_huang_exploratory_v1",
        "status": "POST_HOC_EXPLORATORY_FROZEN_BEFORE_NEW_COMPARATOR_OUTCOMES",
        "created_utc": timestamp(),
        "population": "exact completed HEEDB I0006 independent-confirmation 10000 patients, one ECG each; no reselection",
        "selected_manifest_sha256": digest(previous.COHORT),
        "independent_protocol_sha256": digest(previous.PROTOCOL),
        "independent_predictions_sha256": digest(PRIOR_PREDICTIONS),
        "cache_sha256": digest(SOURCE),
        "cache_shape": [10000, 12, 5000],
        "source_units": "physical mV x1000 -> microvolts; 250Hz upsampled to 500Hz, or native 500Hz, in frozen independent cache",
        "source_leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "conditions": list(CONDITIONS),
        "reversal": "src.ecg_physics.named_reversal plus derive_display at raw 500Hz-equivalent resolution; exact cable algebra, not observed clinical wiring error",
        "huang_adapter": "Existing Huang detector preprocessing: 500Hz SOS Butterworth 0.5-35Hz order 5, scipy Fourier resample 5000->1200, per-lead within-record z-score, select III/aVR/V6 in that order, transpose to (1200,3). No HEEDB-derived scaling, adaptation or weight change.",
        "detector_classes": list(HUANG_CLASSES),
        "detector_policy": "unmodified published four-class softmax argmax; predicted class alone determines inverse correction, never the known simulated condition",
        "correction": "apply predicted inverse to raw displayed 500Hz-equivalent waveform before Ningbo diagnostic preprocessing; clean false alarms are corrected too",
        "diagnostic_adapter": "raw corrected displayed waveform ::5 -> exact (12,1000) at effective 100Hz; frozen baseline checkpoint's per-lead Ningbo mean/std; use all three original baseline checkpoints and unchanged Ningbo validation thresholds",
        "comparison": "reuse completed independent baseline and physics-aware predictions without rerunning; Huang corrected ECG to clean-trained baseline ResNet; same 3 matched seeds",
        "outcome": "HEEDB diagnoses_acquisition.csv codes_physician core TWA only; no software or v24 merge; proxy construct not exact Chapman/Ningbo SNOMED label",
        "core_twa_codes": sorted(heedb_v1.CORE["TWA"]),
        "metrics": list(METRICS),
        "calibration": "mean score, Brier, 15 equal-width bin ECE",
        "uncertainty": "1000 synchronized with-replacement patient-index bootstrap replicates, numpy default_rng seed 20260926; mean seed-level metric then paired arm contrasts; percentile 2.5/97.5 intervals; no equivalence inference from zero-crossing interval",
        "detector_error_audit": "post-hoc aggregate confusion against clean/simulated RA-LL; compare TWA diagnostic scores/operating points conditional on detector correctness, suppress small subsets; true condition never enters deployed correction",
        "interpretation": "exploratory comparison planned after seeing HEEDB independent results; no SOTA, no clinical real-error claim",
        "no_adaptation": "no download, training, fine-tuning, threshold choice, patient selection, outcome remapping or modification of independent results",
        "seeds": list(SEEDS),
        "thresholds": upstream["thresholds"],
        "detector_weights_sha256": weight_hashes,
        "adapter_sha256": digest(ROOT / "src" / "heedb_huang_adapter_v1.py"),
        "adapter_tests_sha256": digest(ROOT / "tests" / "test_heedb_huang_adapter_v1.py"),
        "worker_sha256": digest(ROOT / "src" / "heedb_huang_detector_worker_v1.py"),
        "evaluator_sha256": digest(Path(__file__)),
        "huang_preprocessing_source_sha256": digest(ROOT / "src" / "huang_detector_infer_v2.py"),
        "physics_source_sha256": digest(ROOT / "src" / "ecg_physics.py"),
    }
    write_json(PROTOCOL, protocol)
    (OUT / "FROZEN_PROTOCOL_SHA256.txt").write_text(digest(PROTOCOL) + "\n", encoding="utf-8")
    log("PROTOCOL_FROZEN", patients=len(selected), adapter_tests="PASS", real_cache_adapter_smoke=16, protocol_sha256=digest(PROTOCOL))


def frozen() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(PROTOCOL) != (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("exploratory protocol hash mismatch")
    for key, rel in (("adapter_sha256", "src/heedb_huang_adapter_v1.py"),
                     ("adapter_tests_sha256", "tests/test_heedb_huang_adapter_v1.py"),
                     ("worker_sha256", "src/heedb_huang_detector_worker_v1.py"),
                     ("evaluator_sha256", "src/heedb_huang_exploratory_v1.py"),
                     ("huang_preprocessing_source_sha256", "src/huang_detector_infer_v2.py"),
                     ("physics_source_sha256", "src/ecg_physics.py")):
        if digest(ROOT / rel) != protocol[key]:
            raise RuntimeError(f"frozen comparator code changed: {rel}")
    if artifacts() != protocol["detector_weights_sha256"]:
        raise RuntimeError("detector weights changed")
    previous_protocol, _ = upstream_gate()
    if protocol["selected_manifest_sha256"] != digest(previous.COHORT) or protocol["thresholds"] != previous_protocol["thresholds"]:
        raise RuntimeError("selection or thresholds changed")
    return protocol


def detector_gate() -> np.ndarray:
    complete = json.loads((OUT / "DETECTOR_COMPLETE.json").read_text(encoding="utf-8"))
    if complete["decision"] != "PASS" or digest(DETECTOR) != complete["detector_probabilities_sha256"]:
        raise RuntimeError("detector prediction hash mismatch")
    p = np.load(DETECTOR, mmap_mode="r")
    if p.shape != (2, 10000, 4) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=2), 1, atol=1e-4):
        raise RuntimeError("invalid detector probability cache")
    return p


def diagnose() -> None:
    protocol = frozen()
    if HUANG_PREDICTIONS.exists() or (OUT / "HUANG_DIAGNOSIS_COMPLETE.json").exists():
        raise RuntimeError("Huang diagnostic inference already run")
    detector = detector_gate()
    raw_cache = np.load(SOURCE, mmap_mode="r")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    temporary = PRIVATE / "huang_diagnostic_probabilities_float32.npy.partial"
    if temporary.exists():
        raise RuntimeError("partial prior Huang diagnostic run exists; refusing duplicate")
    started = time.monotonic()
    predictions = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(3, 2, 10000, 3))
    import torch
    for seed_index, seed in enumerate(SEEDS):
        model, mean, std, thresholds = corrected_v2.load_frozen_model("baseline", seed)
        if not np.array_equal(thresholds, np.asarray(protocol["thresholds"]["baseline"][str(seed)], dtype=np.float32)):
            raise RuntimeError("baseline validation threshold changed")
        for condition_index, condition in enumerate(CONDITIONS):
            classes = np.asarray(detector[condition_index]).argmax(axis=1)
            for start in range(0, 10000, 128):
                stop = min(start + 128, 10000)
                raw = np.asarray(raw_cache[start:stop], dtype=np.float32)
                if condition_index:
                    raw = electrical_reversal_batch(raw, "RA_LL_reversal")
                corrected = corrected_batch(raw, classes[start:stop])
                model_input = np.stack([diagnostic_input(item, mean, std) for item in corrected])
                if model_input.shape != (stop - start, 12, 1000):
                    raise RuntimeError("Huang-corrected diagnostic model input shape mismatch")
                p = heedb_v1.predict_array(model, model_input)
                if p.shape != (stop - start, 3) or not np.isfinite(p).all():
                    raise RuntimeError("Huang-corrected diagnostic prediction invalid")
                predictions[seed_index, condition_index, start:stop] = p
                if stop % 1280 == 0 or stop == 10000:
                    predictions.flush()
                    log("HUANG_DIAGNOSIS_PROGRESS", seed=seed, condition=condition, completed=stop, total=10000)
        del model
        torch.cuda.empty_cache()
    predictions.flush()
    del predictions
    os.replace(temporary, HUANG_PREDICTIONS)
    write_json(OUT / "HUANG_DIAGNOSIS_COMPLETE.json", {
        "decision": "PASS", "patients": 10000, "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "prediction_sha256": digest(HUANG_PREDICTIONS), "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    log("HUANG_DIAGNOSIS_COMPLETE", patients=10000, elapsed_seconds=round(time.monotonic() - started, 3))


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 15).astype(int), 14)
    return float(sum(np.mean(bins == b) * abs(np.mean(y[bins == b]) - np.mean(p[bins == b]))
                     for b in range(15) if np.any(bins == b)))


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    positive = y == 1
    predicted = p >= threshold
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "sensitivity": float(predicted[positive].mean()),
        "specificity": float((~predicted[~positive]).mean()),
        "predicted_positive_fraction": float(predicted.mean()),
        "mean_score": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece15": ece15(y, p),
    }


def analyze() -> None:
    protocol = frozen()
    if (OUT / "FINAL_COMPARISON.md").exists() or (OUT / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("comparison already analyzed")
    detector = detector_gate()
    diagnosis_complete = json.loads((OUT / "HUANG_DIAGNOSIS_COMPLETE.json").read_text(encoding="utf-8"))
    if diagnosis_complete["decision"] != "PASS" or digest(HUANG_PREDICTIONS) != diagnosis_complete["prediction_sha256"]:
        raise RuntimeError("Huang diagnostic prediction hash mismatch")
    _upstream, selected = previous.frozen()
    y = previous.labels_after_freeze(selected)[("core", "TWA")].astype(np.uint8)
    if int(y.sum()) != 1456:
        raise RuntimeError("frozen independent TWA label count changed")
    prior = np.load(PRIOR_PREDICTIONS, mmap_mode="r")
    huang = np.load(HUANG_PREDICTIONS, mmap_mode="r")
    if huang.shape != (3, 2, 10000, 3) or not np.isfinite(huang).all():
        raise RuntimeError("invalid Huang diagnostic cache")
    p = np.empty((3, 3, 2, 10000), dtype=np.float32)
    p[:2] = prior[:, :, :, :, 0]
    p[2] = huang[:, :, :, 0]
    thresholds = np.asarray([[protocol["thresholds"]["baseline"][str(seed)][0] for seed in SEEDS],
                             [protocol["thresholds"]["physics_aware"][str(seed)][0] for seed in SEEDS],
                             [protocol["thresholds"]["baseline"][str(seed)][0] for seed in SEEDS]], dtype=float)
    rows = []
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            for condition_index, condition in enumerate(CONDITIONS):
                rows.append({"arm": arm, "seed": seed, "condition": condition, "positive": int(y.sum()),
                             "prevalence": float(y.mean()), "frozen_threshold": float(thresholds[arm_index, seed_index]),
                             **metric(y, p[arm_index, seed_index, condition_index], thresholds[arm_index, seed_index])})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "TWA_BY_SEED.csv", index=False)
    means = frame.groupby(["arm", "condition"], sort=False)[list(METRICS)].mean().reset_index()
    means.to_csv(OUT / "TWA_THREE_SEED_MEANS.csv", index=False)
    error_rows = []
    error_diagnosis = []
    for condition_index, condition in enumerate(CONDITIONS):
        predicted_class = np.asarray(detector[condition_index]).argmax(axis=1)
        expected = 0 if condition_index == 0 else 2
        for class_index, name in enumerate(HUANG_CLASSES):
            error_rows.append({"condition": condition, "predicted_class": name,
                               "patients": int((predicted_class == class_index).sum()),
                               "correct": bool(class_index == expected)})
        for status, mask in (("correct", predicted_class == expected), ("mistake", predicted_class != expected)):
            count, positives = int(mask.sum()), int(y[mask].sum())
            if count < 50:
                continue
            for seed_index, seed in enumerate(SEEDS):
                b = np.asarray(p[0, seed_index, condition_index, mask])
                h = np.asarray(p[2, seed_index, condition_index, mask])
                base_threshold = thresholds[0, seed_index]
                entry = {"condition": condition, "detector_status": status, "seed": seed,
                         "patients": count, "twa_positive": positives,
                         "baseline_mean_score": float(b.mean()), "huang_mean_score": float(h.mean()),
                         "mean_score_shift_huang_minus_baseline": float((h - b).mean()),
                         "brier_shift_huang_minus_baseline": float(np.mean((h - y[mask]) ** 2 - (b - y[mask]) ** 2)),
                         "predicted_positive_fraction_baseline": float((b >= base_threshold).mean()),
                         "predicted_positive_fraction_huang": float((h >= base_threshold).mean())}
                if positives >= 50 and count - positives >= 50:
                    entry["auprc_baseline"] = float(average_precision_score(y[mask], b))
                    entry["auprc_huang"] = float(average_precision_score(y[mask], h))
                    entry["sensitivity_baseline"] = float((b[y[mask] == 1] >= base_threshold).mean())
                    entry["sensitivity_huang"] = float((h[y[mask] == 1] >= base_threshold).mean())
                error_diagnosis.append(entry)
    pd.DataFrame(error_rows).to_csv(OUT / "DETECTOR_CONFUSION_AGGREGATE.csv", index=False)
    pd.DataFrame(error_diagnosis).to_csv(OUT / "DETECTOR_ERROR_DIAGNOSIS_AGGREGATE.csv", index=False)
    comparisons = ((1, 2), (2, 0), (1, 0))  # M2-Huang, Huang-baseline, M2-baseline
    rng = np.random.default_rng(BOOT_SEED)
    boot_arm = np.empty((BOOT, 3, 2, len(METRICS)), dtype=np.float32)
    started = time.monotonic()
    for replicate in range(BOOT):
        idx = rng.integers(0, 10000, 10000)
        labels = y[idx]
        for arm_index in range(3):
            for condition_index in range(2):
                seed_metrics = [metric(labels, p[arm_index, seed_index, condition_index, idx], thresholds[arm_index, seed_index])
                                for seed_index in range(3)]
                boot_arm[replicate, arm_index, condition_index] = [np.mean([item[name] for item in seed_metrics]) for name in METRICS]
        if (replicate + 1) % 25 == 0:
            log("PAIRED_PATIENT_BOOTSTRAP", completed=replicate + 1, total=BOOT)
    intervals = []
    for arm_index, arm in enumerate(ARMS):
        for condition_index, condition in enumerate(CONDITIONS):
            for metric_index, name in enumerate(METRICS):
                values = boot_arm[:, arm_index, condition_index, metric_index]
                intervals.append({"type": "arm", "left": arm, "right": "", "condition": condition, "metric": name,
                                  "estimate": float(means[(means.arm == arm) & (means.condition == condition)][name].iloc[0]),
                                  "low95": float(np.quantile(values, .025)), "high95": float(np.quantile(values, .975)),
                                  "replicates": BOOT, "unit": "patient", "paired": True})
    for left, right in comparisons:
        for condition_index, condition in enumerate(CONDITIONS):
            for metric_index, name in enumerate(METRICS):
                values = boot_arm[:, left, condition_index, metric_index] - boot_arm[:, right, condition_index, metric_index]
                estimate = float(means[(means.arm == ARMS[left]) & (means.condition == condition)][name].iloc[0] -
                                 means[(means.arm == ARMS[right]) & (means.condition == condition)][name].iloc[0])
                intervals.append({"type": "paired_contrast", "left": ARMS[left], "right": ARMS[right],
                                  "condition": condition, "metric": name, "estimate": estimate,
                                  "low95": float(np.quantile(values, .025)), "high95": float(np.quantile(values, .975)),
                                  "replicates": BOOT, "unit": "patient", "paired": True})
    pd.DataFrame(intervals).to_csv(OUT / "PAIRED_PATIENT_BOOTSTRAP.csv", index=False)
    write_json(OUT / "ANALYSIS_COMPLETE.json", {"decision": "PASS", "patients": 10000, "twa_positive": int(y.sum()),
        "bootstrap_replicates": BOOT, "bootstrap_seed": BOOT_SEED, "elapsed_seconds": round(time.monotonic() - started, 3),
        "protocol_sha256": digest(PROTOCOL), "detector_prediction_sha256": digest(DETECTOR),
        "huang_diagnosis_prediction_sha256": digest(HUANG_PREDICTIONS),
        "aggregate_hashes": {name: digest(OUT / name) for name in (
            "TWA_BY_SEED.csv", "TWA_THREE_SEED_MEANS.csv", "DETECTOR_CONFUSION_AGGREGATE.csv",
            "DETECTOR_ERROR_DIAGNOSIS_AGGREGATE.csv", "PAIRED_PATIENT_BOOTSTRAP.csv")}})
    log("ANALYSIS_COMPLETE", patients=10000, bootstrap_replicates=BOOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "diagnose", "analyze"))
    args = parser.parse_args()
    {"freeze": freeze, "diagnose": diagnose, "analyze": analyze}[args.command]()


if __name__ == "__main__":
    main()
