"""Replay audit for the previously saved clean and RA-LL comparator predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import chest_only_baseline_v1 as run


def source(dataset, model, seed):
    if dataset == "Chapman" and model == "clean_12lead" and seed == 104729:
        base = run.V4 / "test_predictions_v4"
        manifest = json.loads((run.V4 / "test_predictions_manifest_v4.json").read_text(encoding="utf-8"))
        return base, manifest, None
    if dataset == "Chapman" and model == "clean_12lead":
        base = run.V7 / f"seed_{seed}" / "test_predictions"
        manifest = json.loads((base.parent / "evaluation_complete.json").read_text(encoding="utf-8"))
        return base, manifest, "prediction_files"
    if dataset == "Chapman":
        base = run.M1 / f"seed_{seed}" / "test_predictions"
        manifest = json.loads((base.parent / "evaluation_complete.json").read_text(encoding="utf-8"))
        return base, manifest, "prediction_sha256"
    if model == "clean_12lead":
        base = run.N / f"seed_{seed}" / "test_predictions"
        manifest = json.loads((run.N / f"seed_{seed}" / "evaluation_complete.json").read_text(encoding="utf-8"))
        return base, manifest, "prediction_sha256"
    base = run.NM2TRAIN / f"seed_{seed}" / "test_predictions"
    manifest = json.loads((run.NM2TRAIN / f"seed_{seed}" / "evaluation_complete.json").read_text(encoding="utf-8"))
    return base, manifest, "prediction_sha256"


def main():
    checks = []
    for dataset in run.DATASETS:
        for model in run.MODELS[:2]:
            for seed in run.SEEDS:
                olddir, oldmanifest, hash_key = source(dataset, model, seed)
                newdir = run.OUT / "comparators" / dataset / model / f"seed_{seed}"
                for condition in ("clean", "RA_LL_reversal"):
                    oldpath = olddir / f"{condition}.npz"; newpath = newdir / f"{condition}.npz"
                    if not oldpath.is_file() or not newpath.is_file(): raise RuntimeError(f"missing saved replay file: {oldpath}")
                    oldhash = run.sha(oldpath)
                    expected = oldmanifest.get(hash_key, {}).get(condition) if hash_key else None
                    if expected and oldhash.lower() != str(expected).lower(): raise RuntimeError(f"saved prediction file hash mismatch: {oldpath}")
                    with np.load(oldpath, allow_pickle=False) as z: oldp = z["p"]
                    with np.load(newpath, allow_pickle=False) as z: newp = z["p"]
                    if oldp.shape != newp.shape: raise RuntimeError(f"saved replay shape mismatch: {dataset}/{model}/{seed}/{condition}")
                    delta = np.abs(oldp.astype(np.float64) - newp.astype(np.float64))
                    checks.append({"dataset": dataset, "model": model, "seed": seed, "condition": condition,
                                   "saved_prediction_path": str(oldpath.relative_to(run.ROOT)), "saved_prediction_sha256": oldhash,
                                   "saved_manifest_hash_checked": expected is not None, "shape": list(oldp.shape),
                                   "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
                                   "frozen_tolerance": {"atol": 1e-5, "rtol": 1e-4},
                                   "pass": bool(np.allclose(oldp, newp, atol=1e-5, rtol=1e-4))})
    report = {"decision": "PASS" if all(x["pass"] for x in checks) else "MISMATCH_REPLAYED_FROM_BOUND_CHECKPOINTS",
              "protocol_sha256": run.sha(run.PROTO), "checks": checks,
              "total_replays": len(checks), "manifest_hash_verified": sum(x["saved_manifest_hash_checked"] for x in checks),
              "interpretation": "The comparator outputs used in the final analysis are fresh inference from provenance-audited checkpoints, preprocessing and validation thresholds, on this package's bound canonical test order."}
    run.atomic_json(run.OUT / "SAVED_COMPARATOR_REPLAY_AUDIT.json", report)
    print(json.dumps({"decision": report["decision"], "total_replays": len(checks), "manifest_hash_verified": report["manifest_hash_verified"],
                      "max_abs_delta": max(x["max_abs_delta"] for x in checks)}, indent=2))


if __name__ == "__main__": main()
