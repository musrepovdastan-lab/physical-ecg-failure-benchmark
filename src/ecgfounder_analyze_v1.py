"""Paired held-out analysis of frozen ECGFounder features and saved ResNets."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ecgfounder_audit_v1 import LABELS, ROOT, SEEDS, pred_path, rows, sha
from src.ecgfounder_baseline_v1 import CFG, OUT, SingleRun, config, extract_result, label_matrix, split, status

MODELS = ("ECGFounder_head", "ResNet_104729", "ResNet_130363", "ResNet_155921")
CONDS = ("clean", "RA_LL_reversal")
METRICS = ("auprc", "auroc", "sensitivity")
N_BOOT = 1000


def csvwrite(path: Path, data: list[dict]):
    if not data:
        return
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    tmp.replace(path)


def scores(y: np.ndarray, p: np.ndarray, threshold: float) -> np.ndarray:
    if np.unique(y).size != 2:
        return np.full(3, np.nan)
    return np.asarray((average_precision_score(y, p), roc_auc_score(y, p),
                       np.mean(p[y == 1] >= threshold)), dtype=np.float64)


def head_prob(dataset: str, condition: str):
    freeze_path = OUT / f"{dataset.lower()}_validation_freeze.json"
    hp = OUT / f"{dataset.lower()}_head.npz"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != sha(CFG) or frozen["head_sha256"] != sha(hp) or frozen["test_accessed_before_freeze"] is not False:
        raise RuntimeError("invalid validation freeze")
    head = np.load(hp, allow_pickle=False)
    feature, _ = extract_result(dataset, "test", condition)
    z = (np.asarray(feature, np.float64) - head["mean"]) / head["std"]
    p = expit(z @ head["coef"].T + head["intercept"])
    return p, head["thresholds"]


def read_dataset(dataset: str):
    y = label_matrix(dataset, "test")
    predictions = np.empty((len(y), len(LABELS), len(MODELS), len(CONDS)), np.float64)
    thresholds = np.empty((len(LABELS), len(MODELS)), np.float64)
    for k, condition in enumerate(CONDS):
        predictions[:, :, 0, k], t = head_prob(dataset, condition)
        thresholds[:, 0] = t
        for j, seed in enumerate(SEEDS, 1):
            predictions[:, :, j, k] = np.load(pred_path(dataset, seed, condition), allow_pickle=False)["p"]
            vp = (ROOT / (f"outputs/phase1_chapman_seed1_v4/validation_predictions_v4.npz" if dataset == "Chapman" and seed == 104729 else
                          f"outputs/phase2_chapman_multiseed_v7/seed_{seed}/validation.npz" if dataset == "Chapman" else
                          f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz"))
            thresholds[:, j] = np.load(vp, allow_pickle=False)["thresholds"]
    if predictions.shape != (len(y), 3, 4, 2) or not np.isfinite(predictions).all():
        raise RuntimeError("prediction shape or finite check failed")
    return y, predictions, thresholds


def zero_shot(dataset: str, y: np.ndarray):
    tasks = (OUT / "official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/tasks.txt").read_text().splitlines()
    if len(tasks) != 150 or tasks[4] != "SINUS BRADYCARDIA" or tasks[6] != "SINUS TACHYCARDIA":
        raise RuntimeError("official output index mismatch")
    val_logits = extract_result(dataset, "validation", "clean")[1]
    outputs = []
    for j, idx in ((1, 4), (2, 6)):
        val_y = label_matrix(dataset, "validation")[:, j]
        val_p = expit(np.asarray(val_logits[:, idx], np.float64))
        threshold = float(np.quantile(val_p[val_y == 0], 0.9))
        for condition in CONDS:
            logits = extract_result(dataset, "test", condition)[1]
            p = expit(np.asarray(logits[:, idx], np.float64))
            ap, au, se = scores(y[:, j], p, threshold)
            outputs.append({"dataset": dataset, "label": LABELS[j], "official_task": tasks[idx],
                            "official_zero_based_index": idx, "condition": condition,
                            "records": len(y), "positives": int(y[:, j].sum()),
                            "validation_threshold": threshold, "auprc": ap, "auroc": au,
                            "sensitivity": se, "mapping_scope": "name-level match; separate from trained-head comparison"})
    return outputs


def bootstrap(dataset: str, y: np.ndarray, p: np.ndarray, thresholds: np.ndarray):
    ids = [r["record_id"] for r in split(dataset, "test")]
    ids_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    ip = OUT / f"{dataset.lower()}_bootstrap_indices.npy"
    if ip.exists():
        indices = np.load(ip, mmap_mode="r")
        if indices.shape != (N_BOOT, len(y)):
            raise RuntimeError("bootstrap index shape mismatch")
    else:
        rng = np.random.default_rng(20260921 + (0 if dataset == "Chapman" else 1))
        indices = rng.integers(0, len(y), size=(N_BOOT, len(y)), dtype=np.int32)
        np.save(ip, indices)
    bp = OUT / f"{dataset.lower()}_bootstrap_metrics.npy"
    mp = OUT / f"{dataset.lower()}_bootstrap_progress.json"
    binding = {"dataset": dataset, "protocol_sha256": sha(CFG), "test_ids_sha256": ids_sha,
               "indices_sha256": sha(ip), "replicates": N_BOOT,
               "model_order": list(MODELS), "condition_order": list(CONDS), "metric_order": list(METRICS)}
    if mp.exists():
        metadata = json.loads(mp.read_text(encoding="utf-8"))
        if any(metadata.get(k) != v for k, v in binding.items()):
            raise RuntimeError("bootstrap progress binding mismatch")
        values = np.load(bp, mmap_mode="r+")
        start = int(metadata["completed_replicates"])
        if start == N_BOOT and metadata.get("metrics_sha256") == sha(bp):
            return np.asarray(values)
    else:
        if bp.exists():
            raise RuntimeError("unbound bootstrap cache")
        values = np.lib.format.open_memmap(bp, mode="w+", dtype=np.float64,
                                            shape=(N_BOOT, 3, 4, 2, 3))
        start = 0
        mp.write_text(json.dumps({**binding, "completed_replicates": 0}, indent=2), encoding="utf-8")
    for b in range(start, N_BOOT):
        ix = indices[b]
        for label in range(3):
            yy = y[ix, label]
            for model in range(4):
                for condition in range(2):
                    values[b, label, model, condition] = scores(
                        yy, p[ix, label, model, condition], thresholds[label, model])
        if (b + 1) % 25 == 0 or b + 1 == N_BOOT:
            values.flush()
            mp.write_text(json.dumps({**binding, "completed_replicates": b + 1}, indent=2), encoding="utf-8")
            status("BOOTSTRAP", dataset=dataset, completed_replicates=b + 1, total_replicates=N_BOOT)
    mp.write_text(json.dumps({**binding, "completed_replicates": N_BOOT,
                              "metrics_sha256": sha(bp)}, indent=2), encoding="utf-8")
    return np.asarray(values)


def summarize(dataset: str, y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, boot: np.ndarray):
    point = np.empty((3, 4, 2, 3))
    for j in range(3):
        for m in range(4):
            for c in range(2):
                point[j, m, c] = scores(y[:, j], p[:, j, m, c], thresholds[j, m])
    summary = []
    per_seed = []
    paired = []
    for j, label in enumerate(LABELS):
        for q, metric in enumerate(METRICS):
            for m, model in enumerate(MODELS):
                clean, reversal = point[j, m, :, q]
                drops = boot[:, j, m, 0, q] - boot[:, j, m, 1, q]
                row = {"dataset": dataset, "label": label, "model": model, "metric": metric,
                       "records": len(y), "positives": int(y[:, j].sum()),
                       "clean": clean, "ra_ll_reversal": reversal, "clean_minus_ra_ll_drop": clean - reversal,
                       "clean_low95": np.nanquantile(boot[:, j, m, 0, q], .025),
                       "clean_high95": np.nanquantile(boot[:, j, m, 0, q], .975),
                       "ra_ll_low95": np.nanquantile(boot[:, j, m, 1, q], .025),
                       "ra_ll_high95": np.nanquantile(boot[:, j, m, 1, q], .975),
                       "drop_low95": np.nanquantile(drops, .025),
                       "drop_high95": np.nanquantile(drops, .975),
                       "validation_threshold": thresholds[j, m]}
                per_seed.append(row)
                if m == 0:
                    summary.append(row)
            # A metric mean across the three individual seeds, never a probability ensemble.
            mean_point = point[j, 1:, :, q].mean(axis=0)
            mean_boot = boot[:, j, 1:, :, q].mean(axis=1)
            drop = mean_boot[:, 0] - mean_boot[:, 1]
            summary.append({"dataset": dataset, "label": label, "model": "ResNet_mean_of_3_seed_metrics",
                            "metric": metric, "records": len(y), "positives": int(y[:, j].sum()),
                            "clean": mean_point[0], "ra_ll_reversal": mean_point[1],
                            "clean_minus_ra_ll_drop": mean_point[0] - mean_point[1],
                            "clean_low95": np.nanquantile(mean_boot[:, 0], .025),
                            "clean_high95": np.nanquantile(mean_boot[:, 0], .975),
                            "ra_ll_low95": np.nanquantile(mean_boot[:, 1], .025),
                            "ra_ll_high95": np.nanquantile(mean_boot[:, 1], .975),
                            "drop_low95": np.nanquantile(drop, .025),
                            "drop_high95": np.nanquantile(drop, .975),
                            "validation_threshold": "per-seed validation thresholds"})
            effect = (boot[:, j, 0, 0, q] - boot[:, j, 0, 1, q]) - drop
            delta_point = (point[j, 0, 0, q] - point[j, 0, 1, q]) - (mean_point[0] - mean_point[1])
            paired.append({"dataset": dataset, "label": label, "metric": metric,
                           "ecgfounder_drop_minus_resnet_mean_drop": delta_point,
                           "low95": np.nanquantile(effect, .025),
                           "high95": np.nanquantile(effect, .975),
                           "unit": "canonical patient/record", "replicates": N_BOOT})
    return summary, per_seed, paired


def changed_examples(dataset: str, y: np.ndarray, p: np.ndarray, thresholds: np.ndarray):
    ids = [r["record_id"] for r in split(dataset, "test")]
    out = []
    for j, label in enumerate(LABELS):
        changed = np.flatnonzero((p[:, j, 0, 0] >= thresholds[j, 0]) !=
                                 (p[:, j, 0, 1] >= thresholds[j, 0]))
        ranked = sorted(changed, key=lambda i: abs(p[i, j, 0, 0] - p[i, j, 0, 1]), reverse=True)
        for i in ranked[:3]:
            out.append({"dataset": dataset, "record_id": ids[i], "label": label,
                        "truth": int(y[i, j]), "head_clean": p[i, j, 0, 0],
                        "head_ra_ll": p[i, j, 0, 1], "head_threshold": thresholds[j, 0],
                        "resnet_104729_clean": p[i, j, 1, 0],
                        "resnet_104729_ra_ll": p[i, j, 1, 1],
                        "direction": "positive_to_negative" if p[i, j, 0, 0] >= thresholds[j, 0] else "negative_to_positive"})
    return out


def main():
    config()
    all_main, all_seeds, all_paired, all_zero, all_examples = [], [], [], [], []
    with SingleRun():
        for dataset in ("Chapman", "Ningbo"):
            status("ANALYZING", dataset=dataset)
            y, p, thresholds = read_dataset(dataset)
            all_zero += zero_shot(dataset, y)
            boot = bootstrap(dataset, y, p, thresholds)
            main, seeds, paired = summarize(dataset, y, p, thresholds, boot)
            all_main += main
            all_seeds += seeds
            all_paired += paired
            all_examples += changed_examples(dataset, y, p, thresholds)
        csvwrite(OUT / "MAIN_COMPARISON.csv", all_main)
        csvwrite(OUT / "PER_SEED_COMPARISON.csv", all_seeds)
        csvwrite(OUT / "PAIRED_DROP_DIFFERENCES.csv", all_paired)
        csvwrite(OUT / "ZERO_SHOT_NAME_MATCHED.csv", all_zero)
        csvwrite(OUT / "CHANGED_PREDICTIONS.csv", all_examples)
        status("COMPLETE", datasets=["Chapman", "Ningbo"], bootstrap_replicates=N_BOOT)


if __name__ == "__main__":
    main()
