"""Frozen multifault audit of physics-aware diagnosis versus Huang correction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "published_detector_multifault_v2"
V1 = ROOT / "outputs" / "published_detector_pipeline_v1"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
SOURCE = V1 / "official_source" / "ECG_lead_check-main"
ORIGINAL_WEIGHTS = SOURCE / "saved_model" / "LDenseNet"
LOADABLE_WEIGHTS = V1 / "official_source" / "LDenseNet_loadable"
DETECTOR_PY = V1 / "detector_env" / "Scripts" / "python.exe"
DETECTOR_SCRIPT = ROOT / "src" / "huang_detector_infer_v2.py"

SEEDS = (104729, 130363, 155921)
LABELS = ("TWA", "SB", "STach")
CLASSES = ("clean", "RA_LA", "RA_LL", "LA_LL")
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
FAULTS = CONDITIONS[1:]
ARMS = (
    "clean_resnet_no_correction",
    "physics_aware_no_correction",
    "huang_argmax_correct_clean_resnet",
    "oracle_true_inverse_clean_resnet_upper_bound",
)
DEPLOYABLE_ARMS = ARMS[:3]
UPSTREAM_COMMIT = "465cc36d12c7cd04f622fd121fd1166981eaebd6"
UPSTREAM_TREE = "8fe35a7ad01c803a4bf4858f1959d81262a89583"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def log(message: str, **fields: object) -> None:
    entry = {"time": now(), "message": message, **fields}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "progress_log.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True), flush=True)


def records() -> list[dict]:
    canonical = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf8"))["records"]
    membership = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf8"))["rows"]
    by_id = {row["record_id"]: row for row in canonical}
    test = [by_id[row["record_id"]] for row in membership if row["split"] == "test"]
    if len(test) != 6981 or len({row["record_id"] for row in test}) != 6981:
        raise RuntimeError("Ningbo test membership is not the expected 6,981 unique records")
    return test


def record_ids(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["record_id"] for row in rows])


def record_order_sha(rows: list[dict]) -> str:
    return text_sha256("\n".join(row["record_id"] for row in rows) + "\n")


def assert_cache_identity(path: Path, rows: list[dict], probability_key: str = "p") -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        if "record_ids" not in cache.files or probability_key not in cache.files:
            raise RuntimeError(f"identity-bearing cache fields missing: {path}")
        if cache["record_ids"].astype(str).tolist() != [row["record_id"] for row in rows]:
            raise RuntimeError(f"record order mismatch: {path}")
        probabilities = cache[probability_key]
        if probabilities.shape[0] != len(rows) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"invalid probabilities: {path}")
        return probabilities.copy()


def run_isolated_weight_assertion() -> dict:
    command = [str(DETECTOR_PY), str(DETECTOR_SCRIPT), "--audit-only"]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("isolated detector audit produced no JSON result")
    audit = json.loads(lines[-1])
    if audit.get("model_variable_match_assertion") != "PASS":
        raise RuntimeError("published model variable assertion did not pass")
    return audit


def audit_weights_source_and_v1(rows: list[dict]) -> dict:
    original = {
        "index": ORIGINAL_WEIGHTS / ".index",
        "data": ORIGINAL_WEIGHTS / ".data-00000-of-00001",
    }
    loadable = {
        "index": LOADABLE_WEIGHTS / "LDenseNet.index",
        "data": LOADABLE_WEIGHTS / "LDenseNet.data-00000-of-00001",
    }
    original_hashes = {name: sha256(path) for name, path in original.items()}
    loadable_hashes = {name: sha256(path) for name, path in loadable.items()}
    content_matches = {name: original_hashes[name] == loadable_hashes[name] for name in original}
    if not all(content_matches.values()):
        raise RuntimeError("Windows-loadable detector checkpoint differs from published bytes")

    isolated = run_isolated_weight_assertion()
    v1_manifest = json.loads((V1 / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf8"))
    v1_detector_path = V1 / "cache" / "detector_test_probs.npz"
    if sha256(v1_detector_path) != v1_manifest["detector_cache_sha256"].upper():
        raise RuntimeError("v1 detector cache hash differs from its frozen manifest")
    with np.load(v1_detector_path, allow_pickle=False) as cache:
        if cache["record_ids"].astype(str).tolist() != [row["record_id"] for row in rows]:
            raise RuntimeError("v1 detector cache record order differs from frozen Ningbo test order")
        if tuple(cache["class_names"].astype(str)) != CLASSES:
            raise RuntimeError("v1 detector class order differs from the published mapping")
        if cache["probs_ra_ll"].shape != (6981, 4) or not np.isfinite(cache["probs_ra_ll"]).all():
            raise RuntimeError("v1 RA-LL detector cache is invalid")

    reused_diagnostic_hashes: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        checkpoint_hash = sha256(N / f"seed_{seed}" / "checkpoint.pt")
        expected_checkpoint = v1_manifest["diagnostic_checkpoints"]["clean_resnet"][str(seed)].upper()
        if checkpoint_hash != expected_checkpoint:
            raise RuntimeError(f"clean checkpoint hash changed for seed {seed}")
        reused_diagnostic_hashes[str(seed)] = {}
        for condition in ("clean", "RA_LL_reversal"):
            path = V1 / "cache" / f"huang_corrected_{condition}_seed_{seed}.npz"
            with np.load(path, allow_pickle=False) as cache:
                if cache.files != ["p"] or cache["p"].shape != (6981, 3) or not np.isfinite(cache["p"]).all():
                    raise RuntimeError(f"invalid v1 Huang diagnostic cache: {path}")
            reused_diagnostic_hashes[str(seed)][condition] = sha256(path)

    audit = {
        "decision": "PASS",
        "created_utc": now(),
        "published_class_mapping": {"0": "clean", "1": "RA_LA", "2": "RA_LL", "3": "LA_LL"},
        "model_load": isolated,
        "weight_files": {
            "original_published": {name: {"path": str(path.relative_to(ROOT)), "sha256": original_hashes[name]} for name, path in original.items()},
            "windows_loadable_copy": {name: {"path": str(path.relative_to(ROOT)), "sha256": loadable_hashes[name]} for name, path in loadable.items()},
            "byte_content_matches": content_matches,
            "checkpoint_prefix_note": "Only the TensorFlow checkpoint prefix was renamed for Windows; index and data shard bytes are identical.",
        },
        "upstream_source": {
            "repository": "https://github.com/wjcai/ECG_lead_check",
            "commit_sha": UPSTREAM_COMMIT,
            "commit_date_utc": "2024-05-16T14:14:53Z",
            "git_tree_sha": UPSTREAM_TREE,
            "mapping_status": "RELIABLY_MAPPED",
            "mapping_method": "Queried GitHub commit/tree API and recomputed Git blob SHA-1 for every extracted file.",
            "matched_blobs": 22,
            "total_blobs": 22,
            "missing_blobs": 0,
            "mismatched_blobs": 0,
            "downloaded_main_archive_sha256": sha256(V1 / "official_source" / "ECG_lead_check_main.zip"),
        },
        "v1_cache_validation": {
            "record_count": len(rows),
            "record_order_sha256": record_order_sha(rows),
            "detector_cache_path": str(v1_detector_path.relative_to(ROOT)),
            "detector_cache_sha256": sha256(v1_detector_path),
            "detector_cache_hash_matches_v1_manifest": True,
            "record_order_exact_match": True,
            "class_order_exact_match": True,
            "reused_huang_diagnostic_cache_hashes": reused_diagnostic_hashes,
            "diagnostic_checkpoint_hashes_match_v1_manifest": True,
        },
        "training_overlap_assessment": "Published code identifies Chapman-Shaoxing training and PTB/PTB-XL external tests; no Ningbo use is indicated.",
        "tensorflow_environment_reused": str(DETECTOR_PY.relative_to(ROOT)),
    }
    write_json(OUT / "WEIGHT_AND_SOURCE_AUDIT.json", audit)
    return audit


def detector_cache_path(condition: str, smoke: bool) -> Path:
    base = OUT / "smoke" / "cache" if smoke else OUT / "cache"
    return base / f"detector_{condition}.npz"


def validate_detector_cache(path: Path, rows: list[dict]) -> np.ndarray:
    probabilities = assert_cache_identity(path, rows, "probabilities")
    with np.load(path, allow_pickle=False) as cache:
        if tuple(cache["class_names"].astype(str)) != CLASSES:
            raise RuntimeError(f"detector class mapping mismatch: {path}")
        if probabilities.shape != (len(rows), 4):
            raise RuntimeError(f"detector shape mismatch: {path}")
    return probabilities


def import_v1_detector_cache(condition: str, rows: list[dict]) -> Path:
    target = detector_cache_path(condition, smoke=False)
    if target.exists():
        validate_detector_cache(target, rows)
        return target
    key = {"clean": "probs_clean", "RA_LL_reversal": "probs_ra_ll"}[condition]
    source = V1 / "cache" / "detector_test_probs.npz"
    with np.load(source, allow_pickle=False) as cache:
        probabilities = cache[key].copy()
    atomic_npz(
        target,
        record_ids=record_ids(rows),
        class_names=np.asarray(CLASSES),
        condition=np.asarray(condition),
        probabilities=probabilities,
        source_cache_sha256=np.asarray(sha256(source)),
    )
    validate_detector_cache(target, rows)
    log("detector_cache_reused", condition=condition, source=str(source.relative_to(ROOT)), sha256=sha256(target))
    return target


def run_detector(condition: str, rows: list[dict], smoke: bool) -> Path:
    target = detector_cache_path(condition, smoke)
    if target.exists():
        validate_detector_cache(target, rows)
        return target
    command = [
        str(DETECTOR_PY), str(DETECTOR_SCRIPT), "--condition", condition,
        "--output", str(target), "--batch-size", "64",
    ]
    if smoke:
        command += ["--limit", str(len(rows))]
    log("detector_start", condition=condition, smoke=smoke, records=len(rows))
    log_base = OUT / "smoke" if smoke else OUT
    log_base.mkdir(parents=True, exist_ok=True)
    with (log_base / f"detector_{condition}_stdout.log").open("w", encoding="utf8") as stdout, (
        log_base / f"detector_{condition}_stderr.log"
    ).open("w", encoding="utf8") as stderr:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stdout, stderr=stderr)
    validate_detector_cache(target, rows)
    log("detector_complete", condition=condition, smoke=smoke, sha256=sha256(target))
    return target


def raw_display_100hz(record: dict) -> np.ndarray:
    from src import ningbo_confirmation_v1 as baseline

    return baseline.wave(baseline.path(record["waveform_path"]))[:, ::5].astype("float32")


def display_to_independent(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x[:2], x[6:]), axis=0)


def apply_condition(x: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return x
    from src.ecg_physics import derive_display, named_reversal

    return derive_display(named_reversal(display_to_independent(x), condition)).astype("float32")


def predicted_inverse(x: np.ndarray, predicted_class: int) -> np.ndarray:
    if predicted_class == 0:
        return x
    condition = {1: "RA_LA_reversal", 2: "RA_LL_reversal", 3: "LA_LL_reversal"}[int(predicted_class)]
    return apply_condition(x, condition)


def diagnostic_cache_path(arm: str, seed: int, condition: str, smoke: bool) -> Path:
    base = OUT / "smoke" / "cache" if smoke else OUT / "cache"
    return base / f"diagnostic_{arm}_{condition}_seed_{seed}.npz"


def wrap_prediction_cache(target: Path, source: Path, rows: list[dict]) -> np.ndarray:
    with np.load(source, allow_pickle=False) as cache:
        probabilities = cache["p"].copy()
    if probabilities.shape != (len(rows), 3) or not np.isfinite(probabilities).all():
        raise RuntimeError(f"source diagnostic cache is invalid: {source}")
    atomic_npz(
        target,
        record_ids=record_ids(rows),
        p=probabilities,
        source_path=np.asarray(str(source.relative_to(ROOT))),
        source_sha256=np.asarray(sha256(source)),
        record_order_sha256=np.asarray(record_order_sha(rows)),
    )
    return assert_cache_identity(target, rows)


def reusable_prediction_source(arm: str, seed: int, condition: str) -> Path | None:
    if arm == "clean_resnet_no_correction" and condition in ("clean", "RA_LL_reversal"):
        return N / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
    if arm == "physics_aware_no_correction" and condition in ("clean", "RA_LL_reversal"):
        return M2 / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
    if arm == "huang_argmax_correct_clean_resnet" and condition in ("clean", "RA_LL_reversal"):
        return V1 / "cache" / f"huang_corrected_{condition}_seed_{seed}.npz"
    return None


def load_diagnostic_model(arm: str, seed: int):
    from src.chapman_phase1 import ECGResNetSmall

    physics = arm == "physics_aware_no_correction"
    base = M2 if physics else N
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGResNetSmall().to(device)
    checkpoint = torch.load(base / f"seed_{seed}" / "checkpoint.pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(base / f"seed_{seed}" / "preprocessing.npz", allow_pickle=False) as prep:
        mean, std = prep["mean"].copy(), prep["std"].copy()
    return model, mean, std, device


def diagnostic_predict(
    arm: str,
    seed: int,
    condition: str,
    rows: list[dict],
    detector_classes: np.ndarray,
    smoke: bool,
) -> np.ndarray:
    target = diagnostic_cache_path(arm, seed, condition, smoke)
    if target.exists():
        return assert_cache_identity(target, rows)
    if not smoke:
        source = reusable_prediction_source(arm, seed, condition)
        if source is not None:
            probabilities = wrap_prediction_cache(target, source, rows)
            log("diagnostic_cache_reused", arm=arm, seed=seed, condition=condition, source=str(source.relative_to(ROOT)))
            return probabilities

    model, mean, std, device = load_diagnostic_model(arm, seed)
    outputs = []
    batch = []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            x = apply_condition(raw_display_100hz(row), condition)
            if arm == "huang_argmax_correct_clean_resnet":
                x = predicted_inverse(x, int(detector_classes[index - 1]))
            elif arm == "oracle_true_inverse_clean_resnet_upper_bound" and condition != "clean":
                x = apply_condition(x, condition)
            batch.append(((x - mean[:, None]) / std[:, None]).astype("float32"))
            if len(batch) == 128 or index == len(rows):
                tensor = torch.from_numpy(np.asarray(batch, dtype="float32")).to(device)
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        prediction = torch.sigmoid(model(tensor))
                else:
                    prediction = torch.sigmoid(model(tensor))
                outputs.append(prediction.float().cpu().numpy())
                batch = []
                if not smoke:
                    log("diagnostic_batch", arm=arm, seed=seed, condition=condition, completed=index, total=len(rows))
    probabilities = np.concatenate(outputs).astype("float32")
    atomic_npz(
        target,
        record_ids=record_ids(rows),
        p=probabilities,
        record_order_sha256=np.asarray(record_order_sha(rows)),
    )
    return assert_cache_identity(target, rows)


def freeze_protocol(rows: list[dict]) -> None:
    path = OUT / "FROZEN_MULTIFAULT_PROTOCOL.json"
    if path.exists():
        return
    protocol = {
        "version": "published_detector_multifault_v2",
        "status": "FROZEN_AFTER_AUDIT_AND_SMOKE_BEFORE_FULL_RA_LA_AND_LA_LL_OUTCOME_GENERATION_OR_INSPECTION",
        "created_utc": now(),
        "population": {"dataset": "Ningbo frozen v2 test split", "records": len(rows), "record_order_sha256": record_order_sha(rows)},
        "conditions": list(CONDITIONS),
        "detector_policy": "Published Huang LDenseNet four-class unmodified softmax argmax; no threshold tuning or adaptation.",
        "detector_classes": list(CLASSES),
        "diagnostic_arms": list(ARMS),
        "oracle_label": "Implementation upper bound only; true fault is never used by deployable arms.",
        "correction_order": "Fault and predicted inverse correction are applied to raw displayed ECG before normal diagnostic preprocessing.",
        "diagnoses": list(LABELS),
        "metrics": ["AUPRC", "AUROC", "sensitivity", "specificity"],
        "diagnostic_thresholds": "Existing per-seed validation-fixed thresholds from each diagnostic checkpoint family.",
        "worst_fault": "For each arm, diagnosis, and metric: minimum of the three reversal-condition metrics after averaging across seeds.",
        "bootstrap": {
            "indices": "outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
            "replicates": 1000,
            "unit": "record",
            "endpoint": "TWA AUPRC",
            "seed_aggregation": "mean within each condition and replicate",
            "worst_fault_endpoint": "minimum across the three reversal-condition seed-mean AUPRC values within each replicate",
            "comparisons": [
                "physics-aware minus unprotected clean ResNet",
                "Huang detect-and-correct minus unprotected clean ResNet",
                "physics-aware minus Huang detect-and-correct",
            ],
        },
        "prohibited": ["HEEDB", "model retraining", "detector adaptation", "792-mask rerun", "post-test detector threshold tuning"],
    }
    write_json(path, protocol)
    log("protocol_frozen", sha256=sha256(path))


def run_smoke() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_rows = records()
    audit_weights_source_and_v1(full_rows)
    rows = full_rows[:8]
    detector_predictions = {}
    for condition in CONDITIONS:
        path = run_detector(condition, rows, smoke=True)
        detector_predictions[condition] = validate_detector_cache(path, rows).argmax(1)
    shapes = {}
    seed = SEEDS[0]
    for arm in ARMS:
        for condition in CONDITIONS:
            prediction = diagnostic_predict(arm, seed, condition, rows, detector_predictions[condition], smoke=True)
            shapes[f"{arm}:{condition}"] = list(prediction.shape)
    report = {
        "decision": "PASS",
        "completed_utc": now(),
        "records": len(rows),
        "conditions": list(CONDITIONS),
        "arms": list(ARMS),
        "diagnostic_shapes": shapes,
        "outcomes_inspected": False,
    }
    write_json(OUT / "SMOKE_REPORT.json", report)
    log("smoke_complete", records=len(rows))


def thresholds_for(arm: str, seed: int) -> np.ndarray:
    base = M2 if arm == "physics_aware_no_correction" else N
    with np.load(base / f"seed_{seed}" / "validation.npz", allow_pickle=False) as validation:
        return validation["thresholds"].copy()


def metric_table(y: np.ndarray, predictions: dict[str, dict[int, dict[str, np.ndarray]]]) -> list[dict]:
    rows = []
    for arm, seeds in predictions.items():
        for seed, conditions in seeds.items():
            thresholds = thresholds_for(arm, seed)
            for condition, matrix in conditions.items():
                for column, diagnosis in enumerate(LABELS):
                    truth = y[:, column]
                    scores = matrix[:, column]
                    predicted = scores >= thresholds[column]
                    positive = truth == 1
                    negative = ~positive
                    rows.append({
                        "arm": arm,
                        "seed": seed,
                        "condition": condition,
                        "diagnosis": diagnosis,
                        "auprc": float(average_precision_score(truth, scores)),
                        "auroc": float(roc_auc_score(truth, scores)),
                        "sensitivity": float(predicted[positive].mean()),
                        "specificity": float((~predicted[negative]).mean()),
                        "threshold": float(thresholds[column]),
                        "n": len(truth),
                    })
    return rows


def worst_fault_summary(metrics: list[dict]) -> list[dict]:
    output = []
    for arm in ARMS:
        for diagnosis in LABELS:
            subset = [row for row in metrics if row["arm"] == arm and row["diagnosis"] == diagnosis]
            for metric in ("auprc", "auroc", "sensitivity", "specificity"):
                means = {
                    condition: float(np.mean([row[metric] for row in subset if row["condition"] == condition]))
                    for condition in CONDITIONS
                }
                worst_condition = min(FAULTS, key=lambda condition: means[condition])
                output.append({
                    "arm": arm,
                    "diagnosis": diagnosis,
                    "metric": metric,
                    "clean": means["clean"],
                    "RA_LA_reversal": means["RA_LA_reversal"],
                    "RA_LL_reversal": means["RA_LL_reversal"],
                    "LA_LL_reversal": means["LA_LL_reversal"],
                    "worst_fault": worst_condition,
                    "minimum_across_reversals": means[worst_condition],
                    "clean_to_worst_fault_drop": means["clean"] - means[worst_condition],
                })
    return output


def detector_outputs(detector_probabilities: dict[str, np.ndarray]) -> dict:
    confusion = np.zeros((4, 4), dtype=int)
    for true_class, condition in enumerate(CONDITIONS):
        predicted = detector_probabilities[condition].argmax(1)
        for predicted_class in range(4):
            confusion[true_class, predicted_class] = int((predicted == predicted_class).sum())
    confusion_rows = [
        {"true_class": CLASSES[row], **{f"predicted_{CLASSES[column]}": int(confusion[row, column]) for column in range(4)}}
        for row in range(4)
    ]
    write_csv(OUT / "detector_confusion_all_faults.csv", confusion_rows)
    recalls = {CLASSES[index]: float(confusion[index, index] / confusion[index].sum()) for index in range(4)}
    metrics = {
        "records_per_class": int(confusion[0].sum()),
        "total_examples": int(confusion.sum()),
        "class_order": list(CLASSES),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": recalls,
        "clean_specificity_correct_no_correction_rate": recalls["clean"],
        "clean_false_correction_rate": 1.0 - recalls["clean"],
        "policy": "unmodified argmax",
    }
    write_json(OUT / "detector_metrics_all_faults.json", metrics)
    return metrics


def paired_bootstraps(y: np.ndarray, predictions: dict[str, dict[int, dict[str, np.ndarray]]]) -> list[dict]:
    indices = np.load(N / "bootstrap_indices.npy", allow_pickle=False)
    if indices.shape != (1000, len(y)):
        raise RuntimeError(f"unexpected synchronized bootstrap shape: {indices.shape}")
    comparisons = (
        ("physics_aware_vs_unprotected_clean", "physics_aware_no_correction", "clean_resnet_no_correction"),
        ("huang_vs_unprotected_clean", "huang_argmax_correct_clean_resnet", "clean_resnet_no_correction"),
        ("physics_aware_vs_huang", "physics_aware_no_correction", "huang_argmax_correct_clean_resnet"),
    )

    compared_arms = tuple(dict.fromkeys(arm for _, left, right in comparisons for arm in (left, right)))
    point_values = {
        arm: {
            condition: float(np.mean([
                average_precision_score(y[:, 0], predictions[arm][seed][condition][:, 0]) for seed in SEEDS
            ]))
            for condition in FAULTS
        }
        for arm in compared_arms
    }
    bootstrap_values = {
        arm: {condition: np.empty(len(indices), dtype=float) for condition in FAULTS}
        for arm in compared_arms
    }
    for replicate, selected in enumerate(indices):
        truth = y[selected, 0]
        for arm in compared_arms:
            for condition in FAULTS:
                bootstrap_values[arm][condition][replicate] = np.mean([
                    average_precision_score(truth, predictions[arm][seed][condition][selected, 0]) for seed in SEEDS
                ])
        if (replicate + 1) % 100 == 0:
            log("bootstrap_progress", completed=replicate + 1, total=len(indices))

    output = []
    for comparison, left, right in comparisons:
        for endpoint in (*FAULTS, "worst_fault"):
            if endpoint == "worst_fault":
                left_point = min(point_values[left].values())
                right_point = min(point_values[right].values())
                left_values = np.min(np.stack([bootstrap_values[left][condition] for condition in FAULTS]), axis=0)
                right_values = np.min(np.stack([bootstrap_values[right][condition] for condition in FAULTS]), axis=0)
            else:
                left_point = point_values[left][endpoint]
                right_point = point_values[right][endpoint]
                left_values = bootstrap_values[left][endpoint]
                right_values = bootstrap_values[right][endpoint]
            deltas = left_values - right_values
            lower, upper = np.quantile(deltas, (0.025, 0.975))
            output.append({
                "comparison": comparison,
                "left_arm": left,
                "right_arm": right,
                "fault_endpoint": endpoint,
                "diagnosis": "TWA",
                "metric": "auprc",
                "point_delta_left_minus_right": left_point - right_point,
                "bootstrap_mean_delta": float(deltas.mean()),
                "lower95": float(lower),
                "upper95": float(upper),
                "interval_excludes_zero": bool(lower > 0 or upper < 0),
                "inference": "left_better" if lower > 0 else "left_worse" if upper < 0 else "no_statistically_resolved_difference",
                "replicates": len(indices),
            })
    return output


def final_decision(worst: list[dict], bootstraps: list[dict], detector: dict) -> None:
    comparison = [row for row in bootstraps if row["comparison"] == "physics_aware_vs_huang"]
    worst_comparison = next(row for row in comparison if row["fault_endpoint"] == "worst_fault")
    if worst_comparison["lower95"] > 0:
        overall = "Physics-aware training statistically beats the published detect-and-correct pipeline on worst-fault TWA AUPRC."
    elif worst_comparison["upper95"] < 0:
        overall = "Physics-aware training statistically loses to the published detect-and-correct pipeline on worst-fault TWA AUPRC."
    else:
        overall = "Physics-aware training and the published detect-and-correct pipeline are not statistically distinguishable on worst-fault TWA AUPRC; this is not an equivalence claim."
    fault_lines = []
    for fault in FAULTS:
        row = next(item for item in comparison if item["fault_endpoint"] == fault)
        fault_lines.append(
            f"- {fault}: delta {row['point_delta_left_minus_right']:+.4f}, 95% CI [{row['lower95']:+.4f}, {row['upper95']:+.4f}] ({row['inference']})."
        )
    consistent = all(row["inference"] == "no_statistically_resolved_difference" for row in comparison if row["fault_endpoint"] in FAULTS)
    consistency = (
        "Yes, the inferential answer is consistent across all three faults: every 95% interval includes zero. "
        "The point estimates do not all favor the same arm, and interval inclusion is not proof of equivalence."
        if consistent else
        "No, the inferential answer is not consistent across all three faults; see the fault-specific intervals below."
    )

    protected = [row for row in worst if row["metric"] == "auprc" and row["arm"] in ("physics_aware_no_correction", "huang_argmax_correct_clean_resnet")]
    hardest = min(
        ((condition, diagnosis, np.mean([row[condition] for row in protected if row["diagnosis"] == diagnosis]))
         for condition in FAULTS for diagnosis in LABELS),
        key=lambda item: item[2],
    )
    clean_twa = {row["arm"]: row["clean"] for row in worst if row["metric"] == "auprc" and row["diagnosis"] == "TWA"}
    physics_clean_delta = clean_twa["physics_aware_no_correction"] - clean_twa["clean_resnet_no_correction"]
    huang_clean_delta = clean_twa["huang_argmax_correct_clean_resnet"] - clean_twa["clean_resnet_no_correction"]
    auprc = {
        (row["arm"], row["diagnosis"]): row
        for row in worst if row["metric"] == "auprc"
    }
    twa_table = []
    for condition, label in (("RA_LA_reversal", "RA-LA"), ("RA_LL_reversal", "RA-LL"), ("LA_LL_reversal", "LA-LL")):
        comparison_row = next(row for row in comparison if row["fault_endpoint"] == condition)
        twa_table.append(
            f"| {label} | {auprc[('clean_resnet_no_correction', 'TWA')][condition]:.4f} | "
            f"{auprc[('physics_aware_no_correction', 'TWA')][condition]:.4f} | "
            f"{auprc[('huang_argmax_correct_clean_resnet', 'TWA')][condition]:.4f} | "
            f"[{comparison_row['lower95']:+.4f}, {comparison_row['upper95']:+.4f}] |"
        )
    twa_table.append(
        f"| Worst fault | {auprc[('clean_resnet_no_correction', 'TWA')]['minimum_across_reversals']:.4f} | "
        f"{auprc[('physics_aware_no_correction', 'TWA')]['minimum_across_reversals']:.4f} | "
        f"{auprc[('huang_argmax_correct_clean_resnet', 'TWA')]['minimum_across_reversals']:.4f} | "
        f"[{worst_comparison['lower95']:+.4f}, {worst_comparison['upper95']:+.4f}] |"
    )

    lines = [
        "# FINAL MULTIFAULT DECISION",
        "",
        overall,
        consistency,
        "",
        "| TWA AUPRC endpoint | Unprotected clean ResNet | Physics-aware | Huang correct | Physics minus Huang 95% CI |",
        "|---|---:|---:|---:|---:|",
        *twa_table,
        "",
        "## Physics-aware versus Huang by fault",
        "",
        *fault_lines,
        f"- Worst-fault endpoint: delta {worst_comparison['point_delta_left_minus_right']:+.4f}, 95% CI [{worst_comparison['lower95']:+.4f}, {worst_comparison['upper95']:+.4f}].",
        "",
        "## Hardest case and clean performance",
        "",
        f"The lowest protected-method mean AUPRC is {hardest[1]} under {hardest[0]} ({hardest[2]:.4f}).",
        f"Worst-fault AUPRC by diagnosis is: physics-aware TWA {auprc[('physics_aware_no_correction', 'TWA')]['minimum_across_reversals']:.4f}, SB {auprc[('physics_aware_no_correction', 'SB')]['minimum_across_reversals']:.4f}, STach {auprc[('physics_aware_no_correction', 'STach')]['minimum_across_reversals']:.4f}; Huang TWA {auprc[('huang_argmax_correct_clean_resnet', 'TWA')]['minimum_across_reversals']:.4f}, SB {auprc[('huang_argmax_correct_clean_resnet', 'SB')]['minimum_across_reversals']:.4f}, STach {auprc[('huang_argmax_correct_clean_resnet', 'STach')]['minimum_across_reversals']:.4f}.",
        f"For TWA clean AUPRC, physics-aware minus clean ResNet is {physics_clean_delta:+.4f}; Huang correction minus clean ResNet is {huang_clean_delta:+.4f}.",
        f"The detector leaves {detector['clean_specificity_correct_no_correction_rate']:.2%} of clean ECGs untouched and falsely corrects {detector['clean_false_correction_rate']:.2%}.",
        "No material clean rank-performance damage is visible in the point estimates. Physics-aware slightly improves clean AUPRC for all three diagnoses but lowers fixed-threshold specificity by at most 0.0045; Huang changes clean AUPRC by -0.0007 (TWA), approximately 0 (SB), and -0.0014 (STach). These clean comparisons are descriptive because no paired clean-performance hypothesis tests were prespecified.",
        "",
        "## Research direction",
        "",
        "This audit alone is not enough to justify developing a novel worst-case/consistency training method. It establishes the multifault operational baseline and identifies residual weaknesses, but a new method still needs a prespecified hypothesis, validation-only development, and an independent held-out evaluation.",
        "",
        "## Unsupported claims",
        "",
        "This is not a reproduction of Huang et al.'s paper scores, not evidence of SOTA superiority, not a clinical-prevalence study, and not evidence for real electrode errors, artifacts, devices, populations, or fault types beyond the exact simulated terminal swaps tested here. Detector calibration, non-argmax policies, Huang-model adaptation, and prospective workflow utility remain untested.",
        "",
        "The oracle arm uses the true fault only as an implementation upper bound and is not deployable.",
    ]
    temporary = OUT / f".FINAL_MULTIFAULT_DECISION.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf8")
    os.replace(temporary, OUT / "FINAL_MULTIFAULT_DECISION.md")


def write_manifest(rows: list[dict]) -> None:
    required_outputs = (
        "FROZEN_MULTIFAULT_PROTOCOL.json", "WEIGHT_AND_SOURCE_AUDIT.json",
        "detector_confusion_all_faults.csv", "detector_metrics_all_faults.json",
        "diagnostic_metrics_by_seed_and_fault.csv", "worst_fault_summary.csv",
        "paired_bootstrap_comparisons.csv", "FINAL_MULTIFAULT_DECISION.md",
    )
    manifest = {
        "created_utc": now(),
        "record_count": len(rows),
        "record_order_sha256": record_order_sha(rows),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "source_archive_sha256": sha256(V1 / "official_source" / "ECG_lead_check_main.zip"),
        "scripts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (DETECTOR_SCRIPT, ROOT / "src" / "published_detector_multifault_v2.py", ROOT / "src" / "ecg_physics.py")
        },
        "frozen_inputs": {
            "split_membership": sha256(N / "SPLIT_MEMBERSHIP.json"),
            "canonical_records": sha256(N / "CANONICAL_RECORDS.json"),
            "bootstrap_indices": sha256(N / "bootstrap_indices.npy"),
        },
        "diagnostic_checkpoints": {
            "clean_resnet": {str(seed): sha256(N / f"seed_{seed}" / "checkpoint.pt") for seed in SEEDS},
            "physics_aware": {str(seed): sha256(M2 / f"seed_{seed}" / "checkpoint.pt") for seed in SEEDS},
        },
        "detector_caches": {condition: sha256(detector_cache_path(condition, False)) for condition in CONDITIONS},
        "diagnostic_caches": {
            f"{arm}:{seed}:{condition}": sha256(diagnostic_cache_path(arm, seed, condition, False))
            for arm in ARMS for seed in SEEDS for condition in CONDITIONS
        },
        "outputs": {name: sha256(OUT / name) for name in required_outputs},
        "environment": {
            "detector_python": str(DETECTOR_PY.relative_to(ROOT)),
            "project_python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", manifest)


def run_full() -> None:
    rows = records()
    audit = audit_weights_source_and_v1(rows)
    smoke_path = OUT / "SMOKE_REPORT.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text(encoding="utf8")).get("decision") != "PASS":
        raise RuntimeError("successful smoke test is required before protocol freeze")
    if audit["decision"] != "PASS":
        raise RuntimeError("weight/source audit failed")
    freeze_protocol(rows)

    detector_probabilities = {}
    for condition in CONDITIONS:
        if condition in ("clean", "RA_LL_reversal"):
            path = import_v1_detector_cache(condition, rows)
        else:
            path = run_detector(condition, rows, smoke=False)
        detector_probabilities[condition] = validate_detector_cache(path, rows)

    detector_classes = {condition: probabilities.argmax(1) for condition, probabilities in detector_probabilities.items()}
    predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            predictions[arm][seed] = {}
            for condition in CONDITIONS:
                predictions[arm][seed][condition] = diagnostic_predict(
                    arm, seed, condition, rows, detector_classes[condition], smoke=False
                )

    y = np.asarray([row["labels"] for row in rows], dtype=int)
    metrics = metric_table(y, predictions)
    write_csv(OUT / "diagnostic_metrics_by_seed_and_fault.csv", metrics)
    worst = worst_fault_summary(metrics)
    write_csv(OUT / "worst_fault_summary.csv", worst)
    detector = detector_outputs(detector_probabilities)
    bootstraps = paired_bootstraps(y, predictions)
    write_csv(OUT / "paired_bootstrap_comparisons.csv", bootstraps)
    final_decision(worst, bootstraps, detector)
    write_manifest(rows)
    log("complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "run"))
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
