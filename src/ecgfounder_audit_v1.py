"""Read-only binding audit for the exploratory ECGFounder baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v1"
LABELS = ("TWA", "SB", "STach")
SEEDS = (104729, 130363, 155921)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(dataset: str):
    if dataset == "Chapman":
        src = "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json"
        data = read(src)["rows"]
        label = lambda r: [int(c) for c in r["bits_twa_sb_stach"]]
    else:
        src = "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"
        data = read(src)["rows"]
        label = lambda r: r["labels"]
    parts = {s: [r for r in data if r["split"] == s] for s in ("train", "validation", "test")}
    assert sum(len(v) for v in parts.values()) == len(data)
    assert len({r["record_id"] for r in data}) == len(data)
    if dataset == "Ningbo":
        assert len({r["waveform_sha256"] for r in data}) == len(data)
    return src, parts, label


def pred_path(dataset: str, seed: int, condition: str) -> Path:
    if dataset == "Chapman":
        if seed == 104729:
            return ROOT / "outputs/phase1_chapman_seed1_v4/test_predictions_v4" / f"{condition}.npz"
        return ROOT / f"outputs/phase2_chapman_multiseed_v7/seed_{seed}/test_predictions/{condition}.npz"
    return ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/test_predictions/{condition}.npz"


def main():
    result = {"datasets": {}, "bindings": {}}
    for dataset in ("Chapman", "Ningbo"):
        split_path, parts, label = rows(dataset)
        result["bindings"][split_path] = sha(ROOT / split_path)
        counts = {s: {"records": len(v), **{name: int(sum(label(r)[j] for r in v)) for j, name in enumerate(LABELS)}} for s, v in parts.items()}
        ids = {s: {r["record_id"] for r in v} for s, v in parts.items()}
        assert all(not ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
        y = np.asarray([label(r) for r in parts["test"]], dtype=np.int8)
        predictions = {}
        for seed in SEEDS:
            metrics = {}
            for condition in ("clean", "RA_LL_reversal"):
                pth = pred_path(dataset, seed, condition)
                p = np.load(pth, allow_pickle=False)["p"]
                assert p.shape == y.shape and np.isfinite(p).all()
                result["bindings"][str(pth.relative_to(ROOT)).replace("\\", "/")] = sha(pth)
                metrics[condition] = {name: {"auprc": float(average_precision_score(y[:, j], p[:, j])), "auroc": float(roc_auc_score(y[:, j], p[:, j]))} for j, name in enumerate(LABELS)}
            predictions[str(seed)] = metrics
        result["datasets"][dataset] = {"counts": counts, "record_overlap": 0, "saved_prediction_metrics": predictions}
    for path in (
        "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json",
        "src/ecg_physics.py",
        "outputs/ecgfounder_baseline_v1/official/12_lead_ECGFounder.pth",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/tasks.txt",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/util.py",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/dataset.py",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/net1d.py",
    ):
        result["bindings"][path] = sha(ROOT / path)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PRE_RUN_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({d: result["datasets"][d]["counts"] for d in ("Chapman", "Ningbo")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
