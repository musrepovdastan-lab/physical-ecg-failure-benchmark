"""Corrected HEEDB v2 evaluation using the frozen v1 cohort and models.

Only two input changes are made: Ningbo's ::5 time sampling, and RA-LL through
the project's existing electrical physics library. Restricted predictions stay
in the git-ignored HEEDB data tree; public outputs contain aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import heedb_external_v1 as v1
from src.ecg_physics import derive_display, named_reversal


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_external_corrected_v2"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "external_corrected_v2"
V1_DATA = ROOT / "data" / "heedb_restricted" / "I0006" / "external_v1"
V1_OUT = ROOT / "outputs" / "heedb_external_v1"
PROTOCOL = ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_CORRECTED_V2_PROTOCOL.json"
PLAN = ROOT / "outputs" / "heedb_failure_audit_v1" / "CORRECTED_V2_PLAN.md"
ARMS = ("baseline", "physics_aware")
SEEDS = (104729, 130363, 155921)
LABELS = ("TWA", "SB", "STach")
VARIANTS = (("core", "TWA"), ("core", "SB"), ("core", "STach"), ("strict", "TWA"), ("strict", "SB"))
SHAPE = (12, 1000)
BOOT = 1000
BOOT_SEED = 20260924


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def progress(stage: str, **fields: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = {"updated_utc": now(), "stage": stage, **fields}
    with (OUT / "PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    print(json.dumps(line, sort_keys=True), flush=True)


def frozen() -> dict:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip()
    if hash_file(PROTOCOL) != expected:
        raise RuntimeError("v2 protocol hash mismatch")
    if hash_file(Path(__file__)) != value["evaluator_source_sha256"]:
        raise RuntimeError("v2 evaluator source hash mismatch")
    if hash_file(ROOT / "tests" / "test_heedb_external_corrected_v2.py") != value["tests_source_sha256"]:
        raise RuntimeError("v2 engineering tests source hash mismatch")
    if hash_file(PLAN) != value["plan_sha256"]:
        raise RuntimeError("v2 plan hash mismatch")
    if hash_file(ROOT / "src" / "ecg_physics.py") != value["physics_sha256"]:
        raise RuntimeError("physics source hash mismatch")
    panel_digest = hashlib.sha256(json.dumps(value["mask_panel_zero_based"], separators=(",", ":")).encode("utf-8")).hexdigest()
    if panel_digest != value["mask_panel_sha256"]:
        raise RuntimeError("mask panel hash mismatch")
    return value


def to_ningbo_time(raw_500hz_uv: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_500hz_uv, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (12, 5000) or not np.isfinite(raw).all():
        raise ValueError("source waveform must be finite (batch, 12, 5000)")
    sampled = np.ascontiguousarray(raw[:, :, ::5])
    if sampled.shape[1:] != SHAPE:
        raise ValueError("Ningbo-matched model input must have exact (12, 1000) shape")
    return sampled


def exact_ra_ll(displayed: np.ndarray) -> np.ndarray:
    x = np.asarray(displayed, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != SHAPE or not np.isfinite(x).all():
        raise ValueError("RA-LL input must be finite (batch, 12, 1000)")
    independent = np.concatenate((x[:, :2, :], x[:, 6:, :]), axis=1)
    reversed_display = derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)
    if reversed_display.shape[1:] != SHAPE or not np.isfinite(reversed_display).all():
        raise ValueError("RA-LL output shape or finiteness failed")
    return reversed_display


def normalized_conditions(raw_500hz_uv: np.ndarray, mean: np.ndarray, std: np.ndarray, masks: tuple[tuple[int, ...], ...]) -> list[np.ndarray]:
    x = to_ningbo_time(raw_500hz_uv)
    clean = ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    reverse = ((exact_ra_ll(x) - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    arrays = [clean, reverse]
    for mask in masks:
        masked = clean.copy()
        masked[:, list(mask), :] = 0.0  # v1's train-mean sentinel, after normalization
        arrays.append(masked)
    if any(array.shape[1:] != SHAPE or not np.isfinite(array).all() for array in arrays):
        raise ValueError("a model input failed exact (12, 1000) shape or finiteness")
    return arrays


def verify_bindings() -> tuple[dict, pd.DataFrame, np.ndarray, tuple[tuple[int, ...], ...]]:
    protocol = frozen()
    old_protocol = json.loads((ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json").read_text(encoding="utf-8"))
    old_provenance = json.loads((V1_OUT / "RUN_PROVENANCE.json").read_text(encoding="utf-8"))
    if hash_file(ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json") != protocol["v1_protocol_sha256"]:
        raise RuntimeError("v1 protocol changed")
    if hash_file(ROOT / "src" / "heedb_external_v1.py") != protocol["v1_evaluator_source_sha256"]:
        raise RuntimeError("v1 evaluator changed")
    for name, expected in protocol["v1_public_artifact_sha256"].items():
        if hash_file(V1_OUT / name) != expected:
            raise RuntimeError(f"v1 aggregate artifact changed: {name}")
    for arm in ARMS:
        for seed in SEEDS:
            unit = old_protocol["models"][arm][str(seed)]
            for path_key, hash_key in (
                ("checkpoint", "checkpoint_sha256"),
                ("preprocessing", "preprocessing_sha256"),
                ("validation_threshold_source", "validation_threshold_source_sha256"),
            ):
                path = ROOT / unit[path_key]
                if hash_file(path) != unit[hash_key] or unit[hash_key] != protocol["model_artifact_sha256"][arm][str(seed)][hash_key]:
                    raise RuntimeError(f"frozen model artifact changed: {arm} {seed} {path_key}")
    source_cache = V1_DATA / "signals_10000_float32.npy"
    selection = V1_DATA / "selection_final.csv"
    if hash_file(source_cache) != protocol["source_cache_sha256"] or hash_file(selection) != protocol["selection_sha256"]:
        raise RuntimeError("source cache or selection hash mismatch")
    cache = np.load(source_cache, mmap_mode="r")
    frame = pd.read_csv(selection, dtype=str, keep_default_na=False)
    if cache.shape != (10000, 12, 5000) or len(frame) != 10000 or frame.BDSPPatientID.nunique() != 10000:
        raise RuntimeError("population or source waveform shape mismatch")
    if int(frame["replaced_initial"].astype(int).sum()) != 0:
        raise RuntimeError("unexpected cohort replacements")
    old_counts = json.loads((V1_OUT / "AGGREGATE_LABEL_COUNTS.json").read_text(encoding="utf-8"))["outcomes"]
    for variant in VARIANTS:
        name = "_".join(variant)
        if int(frame[name].astype(int).sum()) != old_counts[name]["positive"]:
            raise RuntimeError("label counts changed")
    masks = tuple(tuple(mask) for mask in protocol["mask_panel_zero_based"])
    if masks != tuple(tuple(mask) for mask in old_protocol["mask_panel"]["masks_zero_based"]):
        raise RuntimeError("mask panel changed")
    names = ["clean", "RA_LL_reversal"] + ["mask5_" + "_".join(map(str, mask)) for mask in masks]
    if names != json.loads((V1_OUT / "EVALUATION_FREEZE.json").read_text(encoding="utf-8"))["conditions"]:
        raise RuntimeError("condition order changed")
    if protocol["arms"] != list(ARMS) or protocol["seeds"] != list(SEEDS) or protocol["label_order"] != list(LABELS):
        raise RuntimeError("arm, seed or label channel order changed")
    if old_provenance["selection_sha256"] != protocol["selection_sha256"] or old_provenance["waveform_cache_sha256"] != protocol["source_cache_sha256"]:
        raise RuntimeError("v1 provenance binding mismatch")
    return protocol, frame, cache, masks


def load_frozen_model(arm: str, seed: int):
    # v1 loader verifies the existing checkpoint, normalization and threshold
    # hashes. It performs no input transform and does not invoke v1 reversal.
    model, mean, std, thresholds = v1.load_model(arm, seed)
    if mean.shape != (12,) or std.shape != (12,) or thresholds.shape != (3,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise RuntimeError("frozen normalization or thresholds invalid")
    return model, mean, std, thresholds


def smoke() -> None:
    protocol, frame, _cache, masks = verify_bindings()
    if (OUT / "ENGINEERING_SMOKE.json").exists() or (OUT / "EVALUATION_FREEZE.json").exists():
        raise RuntimeError("smoke or evaluation already exists; refusing duplicate run")
    import torch

    readiness_dir = ROOT / "data" / "heedb_restricted" / "I0006"
    readiness = pd.read_csv(readiness_dir / "audit_manifests" / "waveform_sample_candidates.csv", dtype=str, keep_default_na=False)
    objects = pd.read_csv(readiness_dir / "audit_manifests" / "waveform_object_manifest.csv", dtype=str, keep_default_na=False)
    if len(readiness) != 23 or set(readiness.BDSPPatientID) & set(frame.BDSPPatientID):
        raise RuntimeError("readiness sample size or exclusion gate failed")
    old_wave_root = v1.WAVES
    signals = []
    rates = []
    try:
        v1.WAVES = readiness_dir / "waveform_sample"
        for name in readiness.FileName:
            signals.append(v1.read_waveform(name))
            size = objects[(objects.FileName == name) & objects.remote_key.str.endswith(".dat")]
            if len(size) != 1:
                raise RuntimeError("readiness waveform manifest mismatch")
            rates.append(250 if int(size.iloc[0].remote_bytes) == 60000 else 500)
    finally:
        v1.WAVES = old_wave_root
    if sorted(set(rates)) != [250, 500]:
        raise RuntimeError("smoke did not cover both original sampling rates")
    raw = np.stack(signals).astype(np.float32)
    sampled = to_ningbo_time(raw)
    if sampled.shape != (23, *SHAPE):
        raise RuntimeError("smoke exact model input shape failed")
    independent = np.concatenate((sampled[:, :2], sampled[:, 6:]), axis=1)
    reference = derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)
    if not np.allclose(exact_ra_ll(sampled), reference, rtol=0, atol=0):
        raise RuntimeError("all-lead electrical reversal check failed")
    started = time.monotonic()
    checked = 0
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, std, _thresholds = load_frozen_model(arm, seed)
            for model_input in normalized_conditions(raw, mean, std, masks):
                if model_input.shape != (23, *SHAPE):
                    raise RuntimeError("smoke exact model input shape failed")
                p = v1.predict_array(model, model_input)
                if p.shape != (23, 3) or not np.isfinite(p).all():
                    raise RuntimeError("smoke prediction shape or finiteness failed")
                checked += 1
            del model
            torch.cuda.empty_cache()
    result = {
        "decision": "PASS",
        "protocol_sha256": hash_file(PROTOCOL),
        "excluded_records": 23,
        "original_rate_counts": {"250": rates.count(250), "500": rates.count(500)},
        "exact_model_input_shape": list(SHAPE),
        "reversal_all_12_leads_exact": True,
        "models": 6,
        "conditions_per_model": 26,
        "prediction_batches_checked": checked,
        "finite_three_channel_predictions": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_utc": now(),
    }
    save_json(OUT / "ENGINEERING_SMOKE.json", result)
    progress("ENGINEERING_SMOKE_PASS", records=23, model_input_shape=list(SHAPE), conditions=26, models=6)


def evaluate() -> None:
    protocol, frame, cache, masks = verify_bindings()
    smoke_file = OUT / "ENGINEERING_SMOKE.json"
    if not smoke_file.exists() or json.loads(smoke_file.read_text(encoding="utf-8")).get("decision") != "PASS":
        raise RuntimeError("engineering smoke gate did not pass")
    if (OUT / "EVALUATION_FREEZE.json").exists() or (OUT / "INFERENCE_COMPLETE.json").exists():
        raise RuntimeError("evaluation already started or completed; refusing duplicate run")
    import torch

    PRIVATE.mkdir(parents=True, exist_ok=True)
    temp = PRIVATE / "predictions_float32.npy.partial"
    final = PRIVATE / "predictions_float32.npy"
    if temp.exists() or final.exists():
        raise RuntimeError("prediction artifact already exists; refusing duplicate run")
    lock = OUT / ".evaluation.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("evaluation lock exists; refusing duplicate run") from exc
    started = time.monotonic()
    try:
        save_json(OUT / "EVALUATION_FREEZE.json", {
            "protocol_sha256": hash_file(PROTOCOL),
            "evaluator_source_sha256": hash_file(Path(__file__)),
            "selection_sha256": protocol["selection_sha256"],
            "source_cache_sha256": protocol["source_cache_sha256"],
            "patients": len(frame),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "conditions": 26,
            "labels_or_predictions_used_to_change_plan": False,
            "started_utc": now(),
        })
        predictions = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(2, 3, 26, len(frame), 3))
        batch_size = 128
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                model, mean, std, _thresholds = load_frozen_model(arm, seed)
                total_batches = math.ceil(len(frame) / batch_size)
                for batch_number, start in enumerate(range(0, len(frame), batch_size), 1):
                    end = min(len(frame), start + batch_size)
                    raw = np.asarray(cache[start:end], dtype=np.float32)
                    arrays = normalized_conditions(raw, mean, std, masks)
                    for condition_index, array in enumerate(arrays):
                        p = v1.predict_array(model, array)
                        if p.shape != (end - start, 3) or not np.isfinite(p).all():
                            raise RuntimeError("evaluation prediction shape or finiteness failed")
                        predictions[arm_index, seed_index, condition_index, start:end] = p
                    if batch_number % 10 == 0 or batch_number == total_batches:
                        predictions.flush()
                        progress("FROZEN_EVALUATION", arm=arm, seed=seed, completed_batches=batch_number, total_batches=total_batches, elapsed_seconds=round(time.monotonic() - started, 1))
                del model
                torch.cuda.empty_cache()
        predictions.flush()
        del predictions
        os.replace(temp, final)
        save_json(OUT / "INFERENCE_COMPLETE.json", {
            "decision": "PASS",
            "patients": len(frame),
            "models": 6,
            "conditions_per_model": 26,
            "prediction_artifact_sha256": hash_file(final),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_utc": now(),
        })
        progress("INFERENCE_COMPLETE", patients=len(frame), elapsed_seconds=round(time.monotonic() - started, 1))
    finally:
        lock.unlink(missing_ok=True)


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 15).astype(np.int32), 14)
    return float(sum(selected.mean() * abs(y[selected].mean() - p[selected].mean()) for index in range(15) if (selected := bins == index).any()))


def condition_metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    positive = y == 1
    negative = ~positive
    thresholded = p >= threshold
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "sensitivity_at_frozen_threshold": float(thresholded[positive].mean()),
        "specificity_at_frozen_threshold": float((~thresholded[negative]).mean()),
        "predicted_positive_fraction": float(thresholded.mean()),
        "mean_score": float(p.mean()),
        "ece15": ece15(y, p),
    }


def analyze() -> None:
    protocol, frame, _cache, masks = verify_bindings()
    complete = json.loads((OUT / "INFERENCE_COMPLETE.json").read_text(encoding="utf-8"))
    pred_path = PRIVATE / "predictions_float32.npy"
    if complete.get("decision") != "PASS" or hash_file(pred_path) != complete["prediction_artifact_sha256"]:
        raise RuntimeError("completed prediction artifact integrity gate failed")
    if (OUT / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("analysis already completed; refusing overwrite")
    pred = np.load(pred_path, mmap_mode="r")
    if pred.shape != (2, 3, 26, 10000, 3) or not np.isfinite(pred).all():
        raise RuntimeError("prediction array shape or finiteness failed")
    thresholds = {}
    for arm in ARMS:
        for seed in SEEDS:
            _checkpoint, _preprocessing, validation = v1.model_paths(arm, seed)
            with np.load(validation, allow_pickle=False) as loaded:
                thresholds[(arm, seed)] = loaded["thresholds"].astype(np.float32)
            if not np.array_equal(thresholds[(arm, seed)], np.asarray(protocol["thresholds"][arm][str(seed)], dtype=np.float32)):
                raise RuntimeError("frozen threshold changed")
    labels = {variant: frame["_".join(variant)].astype(np.uint8).to_numpy() for variant in VARIANTS}
    conditions = ["clean", "RA_LL_reversal"] + ["mask5_" + "_".join(map(str, mask)) for mask in masks]
    rows = []
    for variant in VARIANTS:
        mapping, label = variant
        label_index = LABELS.index(label)
        y = labels[variant]
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                threshold = float(thresholds[(arm, seed)][label_index])
                for condition_index, name in enumerate(conditions):
                    p = np.asarray(pred[arm_index, seed_index, condition_index, :, label_index])
                    rows.append({"mapping": mapping, "label": label, "arm": arm, "seed": seed, "condition": name, "patients": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()), "frozen_threshold": threshold, **condition_metric(y, p, threshold)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "AGGREGATE_RESULTS_BY_SEED.csv", index=False)
    summary = results.groupby(["mapping", "label", "arm", "condition"], sort=False)[["auprc", "auroc", "sensitivity_at_frozen_threshold", "specificity_at_frozen_threshold", "predicted_positive_fraction", "mean_score", "ece15"]].mean().reset_index()
    summary.to_csv(OUT / "THREE_SEED_MEAN_METRICS.csv", index=False)
    save_json(OUT / "AGGREGATE_LABEL_COUNTS.json", {"patients": len(frame), "outcomes": {"_".join(variant): int(labels[variant].sum()) for variant in VARIANTS}})

    contrasts = []
    for variant in VARIANTS:
        mapping, label = variant
        subset = results[(results.mapping == mapping) & (results.label == label)]
        for metric_name in ("auprc", "sensitivity_at_frozen_threshold"):
            by = {}
            for arm in ARMS:
                by[(arm, "clean")] = subset[(subset.arm == arm) & (subset.condition == "clean")].sort_values("seed")[metric_name].to_numpy()
                by[(arm, "RA_LL_reversal")] = subset[(subset.arm == arm) & (subset.condition == "RA_LL_reversal")].sort_values("seed")[metric_name].to_numpy()
                contrasts.append({"mapping": mapping, "label": label, "metric": metric_name, "contrast": f"{arm}_clean_minus_ra_ll", "estimate": float(np.mean(by[(arm, "clean")] - by[(arm, "RA_LL_reversal")]))})
            for condition, suffix in (("clean", "clean"), ("RA_LL_reversal", "ra_ll")):
                contrasts.append({"mapping": mapping, "label": label, "metric": metric_name, "contrast": f"physics_aware_minus_baseline_{suffix}", "estimate": float(np.mean(by[("physics_aware", condition)] - by[("baseline", condition)]))})
        for arm in ARMS:
            mask_rows = subset[(subset.arm == arm) & subset.condition.str.startswith("mask5_")]
            contrasts.append({"mapping": mapping, "label": label, "metric": "auprc", "contrast": f"{arm}_24_mask_subset_mean", "estimate": float(mask_rows.groupby("seed").auprc.mean().mean())})
    contrast_table = pd.DataFrame(contrasts)
    progress("AGGREGATE_METRICS_COMPLETE", patients=len(frame), conditions=26)

    boot_values = {(mapping, label, metric_name, contrast): [] for mapping, label in VARIANTS for metric_name in ("auprc", "sensitivity_at_frozen_threshold") for contrast in ("baseline_clean_minus_ra_ll", "physics_aware_clean_minus_ra_ll", "physics_aware_minus_baseline_clean", "physics_aware_minus_baseline_ra_ll")}
    panel_key = ("core", "TWA", "auprc", "physics_aware_minus_baseline_24_mask_subset_mean")
    boot_values[panel_key] = []
    random = np.random.default_rng(BOOT_SEED)
    for replicate in range(BOOT):
        index = random.integers(0, len(frame), len(frame))
        for variant in VARIANTS:
            mapping, label = variant
            label_index = LABELS.index(label)
            y = labels[variant][index]
            positive = y == 1
            by = {}
            for arm_index, arm in enumerate(ARMS):
                for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                    ap, sens = [], []
                    for seed_index, seed in enumerate(SEEDS):
                        p = np.asarray(pred[arm_index, seed_index, condition_index, index, label_index])
                        ap.append(float(average_precision_score(y, p)))
                        sens.append(float((p[positive] >= thresholds[(arm, seed)][label_index]).mean()))
                    by[(arm, condition, "auprc")] = np.asarray(ap)
                    by[(arm, condition, "sensitivity_at_frozen_threshold")] = np.asarray(sens)
            for metric_name in ("auprc", "sensitivity_at_frozen_threshold"):
                for arm in ARMS:
                    boot_values[(mapping, label, metric_name, f"{arm}_clean_minus_ra_ll")].append(float(np.mean(by[(arm, "clean", metric_name)] - by[(arm, "RA_LL_reversal", metric_name)])))
                for condition, suffix in (("clean", "clean"), ("RA_LL_reversal", "ra_ll")):
                    boot_values[(mapping, label, metric_name, f"physics_aware_minus_baseline_{suffix}")].append(float(np.mean(by[("physics_aware", condition, metric_name)] - by[("baseline", condition, metric_name)])))
        y = labels[("core", "TWA")][index]
        panel = {}
        for arm_index, arm in enumerate(ARMS):
            seed_means = []
            for seed_index, _seed in enumerate(SEEDS):
                scores = [float(average_precision_score(y, np.asarray(pred[arm_index, seed_index, condition_index, index, 0]))) for condition_index in range(2, 26)]
                seed_means.append(float(np.mean(scores)))
            panel[arm] = float(np.mean(seed_means))
        boot_values[panel_key].append(panel["physics_aware"] - panel["baseline"])
        if (replicate + 1) % 25 == 0:
            progress("PAIRED_BOOTSTRAP", completed=replicate + 1, total=BOOT)
    intervals = []
    for (mapping, label, metric_name, contrast), values in boot_values.items():
        numbers = np.asarray(values)
        intervals.append({"mapping": mapping, "label": label, "metric": metric_name, "contrast": contrast, "bootstrap_mean": float(numbers.mean()), "low95": float(np.quantile(numbers, .025)), "high95": float(np.quantile(numbers, .975)), "replicates": BOOT, "unit": "patient", "paired": True})
    interval_table = pd.DataFrame(intervals)
    interval_table.to_csv(OUT / "PAIRED_BOOTSTRAP_INTERVALS.csv", index=False)
    combined = contrast_table.merge(interval_table.drop(columns=["replicates", "unit", "paired"]), how="left", on=["mapping", "label", "metric", "contrast"])
    combined.to_csv(OUT / "AGGREGATE_CONTRASTS.csv", index=False)
    save_json(OUT / "ANALYSIS_COMPLETE.json", {"decision": "PASS", "patients": len(frame), "bootstrap_replicates": BOOT, "bootstrap_unit": "patient", "paired_across_arms_seeds_conditions": True, "completed_utc": now()})
    progress("ANALYSIS_COMPLETE", patients=len(frame), bootstrap_replicates=BOOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "smoke", "evaluate", "analyze"))
    command = parser.parse_args().command
    if command == "preflight":
        protocol, frame, cache, masks = verify_bindings()
        progress("PREFLIGHT_PASS", patients=len(frame), waveform_shape=list(cache.shape), masks=len(masks), protocol_sha256=hash_file(PROTOCOL))
    else:
        {"smoke": smoke, "evaluate": evaluate, "analyze": analyze}[command]()


if __name__ == "__main__":
    main()
