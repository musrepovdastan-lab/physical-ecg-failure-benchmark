"""Read-only ResNet reference table after the frozen ECGFounder v1 stop."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ecgfounder_audit_v1 import LABELS, ROOT, SEEDS, pred_path, rows

OUT = ROOT / "outputs/ecgfounder_baseline_v1"


def write(name, data):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    table, examples = [], []
    for dataset in ("Chapman", "Ningbo"):
        _, parts, labelfn = rows(dataset)
        y = np.array([labelfn(r) for r in parts["test"]], np.int8)
        ids = [r["record_id"] for r in parts["test"]]
        by_label = {name: [] for name in LABELS}
        for seed in SEEDS:
            base = (ROOT / "outputs/phase1_chapman_seed1_v4/validation_predictions_v4.npz"
                    if dataset == "Chapman" and seed == 104729 else
                    ROOT / f"outputs/phase2_chapman_multiseed_v7/seed_{seed}/validation.npz"
                    if dataset == "Chapman" else
                    ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz")
            thresholds = np.load(base, allow_pickle=False)["thresholds"]
            clean = np.load(pred_path(dataset, seed, "clean"), allow_pickle=False)["p"]
            fault = np.load(pred_path(dataset, seed, "RA_LL_reversal"), allow_pickle=False)["p"]
            for j, label in enumerate(LABELS):
                metrics = []
                for p in (clean, fault):
                    metrics.append((float(average_precision_score(y[:, j], p[:, j])),
                                    float(roc_auc_score(y[:, j], p[:, j])),
                                    float(np.mean(p[y[:, j] == 1, j] >= thresholds[j]))))
                by_label[label].append(metrics)
                if seed == 104729:
                    changed = np.flatnonzero((clean[:, j] >= thresholds[j]) != (fault[:, j] >= thresholds[j]))
                    selected = sorted(changed, key=lambda i: abs(float(clean[i, j] - fault[i, j])), reverse=True)[:2]
                    for i in selected:
                        examples.append({"dataset": dataset, "record_id": ids[i], "label": label,
                                         "truth": int(y[i, j]), "resnet_seed": seed,
                                         "clean_probability": float(clean[i, j]),
                                         "ra_ll_probability": float(fault[i, j]),
                                         "validation_threshold": float(thresholds[j]),
                                         "scope": "existing ResNet prediction; ECGFounder v1 stopped"})
        for label in LABELS:
            z = np.array(by_label[label])
            for q, metric in enumerate(("AUPRC", "AUROC", "sensitivity")):
                clean = float(z[:, 0, q].mean())
                fault = float(z[:, 1, q].mean())
                table.append({"dataset": dataset, "label": label, "metric": metric,
                              "records": len(y), "positives": int(y[:, LABELS.index(label)].sum()),
                              "saved_resnet_mean_clean": clean,
                              "saved_resnet_mean_ra_ll": fault,
                              "saved_resnet_mean_drop": clean - fault,
                              "ecgfounder_clean": "NOT_EVALUATED",
                              "ecgfounder_ra_ll": "NOT_EVALUATED",
                              "ecgfounder_drop": "NOT_EVALUATED",
                              "reason": "STOP_PREPROCESSING_MISMATCH"})
    write("BLOCKED_SIDE_BY_SIDE.csv", table)
    write("SAVED_RESNET_CHANGED_EXAMPLES.csv", examples)
    print(json.dumps({"reference_rows": len(table), "existing_resnet_examples": len(examples)}))


if __name__ == "__main__":
    main()
