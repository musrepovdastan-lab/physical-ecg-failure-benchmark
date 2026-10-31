"""Fair operational baseline using the published Huang et al. limb detector."""
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
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "published_detector_pipeline_v1"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
SOURCE = OUT / "official_source" / "ECG_lead_check-main"
DETECTOR_PY = OUT / "detector_env" / "Scripts" / "python.exe"
SEEDS = (104729, 130363, 155921)
LABELS = ("TWA", "SB", "STach")
DETECTOR_CLASSES = ("clean", "RA_LA", "RA_LL", "LA_LL")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def log(message: str, **fields: object) -> None:
    entry = {"time": now(), "message": message, **fields}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "progress_log.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True), flush=True)


def records() -> list[dict]:
    can = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf8"))["records"]
    split = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf8"))["rows"]
    by = {r["record_id"]: r for r in can}
    test = [by[r["record_id"]] for r in split if r["split"] == "test"]
    if len(test) != 6981:
        raise RuntimeError("unexpected test count")
    return test


def load_100hz_display(record: dict) -> np.ndarray:
    from src import ningbo_confirmation_v1 as b

    return b.wave(b.path(record["waveform_path"]))[:, ::5].astype("float32")


def display_to_independent(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x[:2], x[6:]), axis=0)


def derive_display(independent: np.ndarray) -> np.ndarray:
    from src.ecg_physics import derive_display as dd

    return dd(independent).astype("float32")


def correct_by_class(x: np.ndarray, cls: int) -> np.ndarray:
    from src.ecg_physics import named_reversal

    if cls == 0:
        return x
    fault = {1: "RA_LA_reversal", 2: "RA_LL_reversal", 3: "LA_LL_reversal"}[int(cls)]
    return derive_display(named_reversal(display_to_independent(x), fault))


def apply_ra_ll(x: np.ndarray) -> np.ndarray:
    from src.ecg_physics import named_reversal

    return derive_display(named_reversal(display_to_independent(x), "RA_LL_reversal"))


def run_detector(limit: int | None = None) -> Path:
    name = "detector_smoke_probs.npz" if limit else "detector_test_probs.npz"
    out = OUT / "cache" / name
    if out.exists():
        return out
    cmd = [
        str(DETECTOR_PY),
        "src/huang_detector_infer_v1.py",
        "--output",
        str(out),
        "--batch-size",
        "64",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    log("detector_start", limit=limit, output=str(out))
    with (OUT / ("detector_smoke_stdout.log" if limit else "detector_stdout.log")).open("w", encoding="utf8") as so, (
        OUT / ("detector_smoke_stderr.log" if limit else "detector_stderr.log")
    ).open("w", encoding="utf8") as se:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=so, stderr=se)
    log("detector_complete", output=str(out), sha256=sha(out))
    return out


def freeze_protocol() -> None:
    p = {
        "version": "published_detector_pipeline_v1",
        "status": "FROZEN_AFTER_AUDIT_AND_SMOKE_BEFORE_NINGBO_TEST_OUTCOME_ANALYSIS",
        "created_utc": now(),
        "population": {"dataset": "Ningbo frozen v2", "test_records": 6981},
        "arms": [
            "clean-trained ResNet, no correction",
            "physics-aware ResNet, no correction",
            "published Huang LDenseNet detector argmax -> correction -> clean-trained ResNet",
        ],
        "detector": {
            "source": "https://github.com/wjcai/ECG_lead_check",
            "local_source": str(SOURCE.relative_to(ROOT)),
            "classes": list(DETECTOR_CLASSES),
            "input_shape": [1200, 3],
            "channels": ["III", "aVR", "V6"],
            "preprocessing": "500 Hz SOS Butterworth 0.5-35 Hz order 5, resample to 1200, per-lead z-score",
            "policy_primary": "unmodified softmax argmax",
            "threshold_selection": "none for primary; no validation tuning used",
        },
        "prohibited": ["HEEDB", "diagnostic retraining", "new detector training", "792-mask rerun"],
        "bootstrap": {
            "indices": "outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
            "replicates": 1000,
            "unit": "record",
            "primary": "TWA AUPRC paired differences",
        },
    }
    write_json(OUT / "FROZEN_PROTOCOL.json", p)


def detector_audit() -> None:
    hashes = {}
    for path in [
        SOURCE / "limb" / "data_preprocess.py",
        SOURCE / "limb" / "LDenseNet.py",
        SOURCE / "limb" / "train_model_limb.py",
        OUT / "official_source" / "LDenseNet_loadable" / "LDenseNet.index",
        OUT / "official_source" / "LDenseNet_loadable" / "LDenseNet.data-00000-of-00001",
        OUT / "official_source" / "ECG_lead_check_main.zip",
    ]:
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = sha(path)
    write_json(
        OUT / "DETECTOR_AUDIT.json",
        {
            "decision": "PASS",
            "class_mapping_from_code": {"0": "clean", "1": "RA_LA", "2": "RA_LL", "3": "LA_LL"},
            "published_weights_loaded": True,
            "checkpoint_note": "Published checkpoint prefix '.' was byte-copied to prefix 'LDenseNet' for Windows TensorFlow loading; .data/.index hashes are preserved by content.",
            "training_data_overlap_assessment": "Repository limb training code names Chapman-Shaoxing for training and PTB/PTB-XL for external tests; no Ningbo training use is indicated in code.",
            "hashes": hashes,
            "tensorflow_env": str(DETECTOR_PY.relative_to(ROOT)),
        },
    )


def load_cached_predictions(base: Path, condition: str) -> dict[int, np.ndarray]:
    return {
        seed: np.load(base / f"seed_{seed}" / "test_predictions" / f"{condition}.npz", allow_pickle=False)["p"]
        for seed in SEEDS
    }


def load_thresholds(base: Path) -> dict[int, np.ndarray]:
    return {seed: np.load(base / f"seed_{seed}" / "validation.npz", allow_pickle=False)["thresholds"] for seed in SEEDS}


def diagnostic_predict(seed: int, rows: list[dict], classes: np.ndarray, source_condition: str) -> np.ndarray:
    from src.chapman_phase1 import ECGResNetSmall

    cache = OUT / "cache" / f"huang_corrected_{source_condition}_seed_{seed}.npz"
    if cache.exists():
        return np.load(cache, allow_pickle=False)["p"]
    z = np.load(N / f"seed_{seed}" / "preprocessing.npz", allow_pickle=False)
    mean, std = z["mean"], z["std"]
    model = ECGResNetSmall().cuda()
    model.load_state_dict(torch.load(N / f"seed_{seed}" / "checkpoint.pt", weights_only=True, map_location="cuda")["state_dict"])
    out = []
    batch = []
    model.eval()
    with torch.no_grad():
        for i, record in enumerate(rows, 1):
            x = load_100hz_display(record)
            if source_condition == "RA_LL_reversal":
                x = apply_ra_ll(x)
            x = correct_by_class(x, int(classes[i - 1]))
            batch.append(((x - mean[:, None]) / std[:, None]).astype("float32"))
            if len(batch) == 128 or i == len(rows):
                a = torch.from_numpy(np.asarray(batch, dtype="float32")).cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(torch.sigmoid(model(a)).float().cpu().numpy())
                batch = []
                log("diagnostic_batch", seed=seed, condition=source_condition, completed=i, total=len(rows))
    p = np.concatenate(out).astype("float32")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, p=p)
    return p


def metric_rows(y: np.ndarray, predictions: dict[str, dict[int, dict[str, np.ndarray]]], thresholds: dict[str, dict[int, np.ndarray]]) -> list[dict]:
    rows = []
    for arm, by_seed in predictions.items():
        for seed, by_condition in by_seed.items():
            t = thresholds[arm][seed]
            for condition, pmat in by_condition.items():
                for j, label in enumerate(LABELS):
                    p = pmat[:, j]
                    z = y[:, j]
                    pred = p >= t[j]
                    pos = z == 1
                    neg = ~pos
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "condition": condition,
                            "endpoint": label,
                            "auprc": average_precision_score(z, p),
                            "auroc": roc_auc_score(z, p),
                            "sensitivity": float(pred[pos].mean()),
                            "specificity": float((~pred[neg]).mean()),
                            "threshold": float(t[j]),
                            "n": len(z),
                        }
                    )
    return rows


def bootstrap_twa(y: np.ndarray, predictions: dict[str, dict[int, dict[str, np.ndarray]]]) -> list[dict]:
    ix = np.load(N / "bootstrap_indices.npy", allow_pickle=False)
    out = []
    clean_base = predictions["clean_resnet_no_correction"]
    for compare in ("physics_aware_no_correction", "huang_argmax_correct_clean_resnet"):
        deltas = []
        for b in range(ix.shape[0]):
            vals = []
            for seed in SEEDS:
                ref = average_precision_score(y[ix[b], 0], clean_base[seed]["RA_LL_reversal"][ix[b], 0])
                cmp = average_precision_score(y[ix[b], 0], predictions[compare][seed]["RA_LL_reversal"][ix[b], 0])
                vals.append(cmp - ref)
            deltas.append(float(np.mean(vals)))
        a = np.asarray(deltas)
        out.append(
            {
                "comparison": f"{compare}_minus_clean_resnet_no_correction",
                "endpoint": "TWA",
                "mean_delta_auprc": float(a.mean()),
                "lower95": float(np.quantile(a, 0.025)),
                "upper95": float(np.quantile(a, 0.975)),
                "replicates": int(ix.shape[0]),
            }
        )
    return out


def detector_metrics(detector_npz: Path) -> dict:
    d = np.load(detector_npz, allow_pickle=False)
    pc, pr = d["probs_clean"], d["probs_ra_ll"]
    yc = np.zeros(len(pc), dtype=int)
    yr = np.full(len(pr), 2, dtype=int)
    pred = np.concatenate((pc.argmax(1), pr.argmax(1)))
    truth = np.concatenate((yc, yr))
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    rows = [
        {"true": DETECTOR_CLASSES[i], **{f"pred_{DETECTOR_CLASSES[j]}": int(cm[i, j]) for j in range(4)}}
        for i in range(4)
    ]
    write_csv(OUT / "detector_confusion_matrix.csv", rows)
    return {
        "confusion_matrix": cm.tolist(),
        "ra_ll_recall": float((pr.argmax(1) == 2).mean()),
        "clean_specificity": float((pc.argmax(1) == 0).mean()),
        "clean_incorrectly_corrected_fraction": float((pc.argmax(1) != 0).mean()),
    }


def run(limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start", limit=limit)
    detector_audit()
    if limit is None:
        freeze_protocol()
    det_path = run_detector(limit)
    if limit:
        write_json(OUT / "SMOKE_REPORT.json", {"decision": "PASS", "detector_cache": str(det_path), "completed_utc": now()})
        return
    rows = records()
    d = np.load(det_path, allow_pickle=False)
    ids = d["record_ids"].astype(str).tolist()
    if ids != [r["record_id"] for r in rows]:
        raise RuntimeError("detector record order mismatch")
    clean_cls = d["probs_clean"].argmax(1)
    ra_cls = d["probs_ra_ll"].argmax(1)
    y = np.asarray([r["labels"] for r in rows], dtype=int)
    pred = {
        "clean_resnet_no_correction": {
            seed: {
                "clean": np.load(N / f"seed_{seed}" / "test_predictions" / "clean.npz", allow_pickle=False)["p"],
                "RA_LL_reversal": np.load(N / f"seed_{seed}" / "test_predictions" / "RA_LL_reversal.npz", allow_pickle=False)["p"],
            }
            for seed in SEEDS
        },
        "physics_aware_no_correction": {
            seed: {
                "clean": np.load(M2 / f"seed_{seed}" / "test_predictions" / "clean.npz", allow_pickle=False)["p"],
                "RA_LL_reversal": np.load(M2 / f"seed_{seed}" / "test_predictions" / "RA_LL_reversal.npz", allow_pickle=False)["p"],
            }
            for seed in SEEDS
        },
        "huang_argmax_correct_clean_resnet": {
            seed: {
                "clean": diagnostic_predict(seed, rows, clean_cls, "clean"),
                "RA_LL_reversal": diagnostic_predict(seed, rows, ra_cls, "RA_LL_reversal"),
            }
            for seed in SEEDS
        },
    }
    thresholds = {
        "clean_resnet_no_correction": load_thresholds(N),
        "physics_aware_no_correction": load_thresholds(M2),
        "huang_argmax_correct_clean_resnet": load_thresholds(N),
    }
    metrics = metric_rows(y, pred, thresholds)
    write_csv(OUT / "diagnostic_metrics_by_seed.csv", metrics)
    write_csv(OUT / "bootstrap_twa_primary.csv", bootstrap_twa(y, pred))
    det = detector_metrics(det_path)
    write_json(OUT / "detector_metrics.json", det)
    manifest = {
        "created_utc": now(),
        "source_zip_sha256": sha(OUT / "official_source" / "ECG_lead_check_main.zip"),
        "detector_cache_sha256": sha(det_path),
        "protocol_sha256": sha(OUT / "FROZEN_PROTOCOL.json"),
        "ningbo_split_sha256": sha(N / "SPLIT_MEMBERSHIP.json"),
        "diagnostic_checkpoints": {
            "clean_resnet": {str(seed): sha(N / f"seed_{seed}" / "checkpoint.pt") for seed in SEEDS},
            "physics_aware": {str(seed): sha(M2 / f"seed_{seed}" / "checkpoint.pt") for seed in SEEDS},
        },
    }
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", manifest)
    summarize(metrics, det)
    log("complete")


def summarize(metrics: list[dict], det: dict) -> None:
    compact = []
    for arm in ("clean_resnet_no_correction", "physics_aware_no_correction", "huang_argmax_correct_clean_resnet"):
        for condition in ("clean", "RA_LL_reversal"):
            for endpoint in LABELS:
                rs = [r for r in metrics if r["arm"] == arm and r["condition"] == condition and r["endpoint"] == endpoint]
                compact.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "endpoint": endpoint,
                        "mean_auprc": float(np.mean([r["auprc"] for r in rs])),
                        "mean_auroc": float(np.mean([r["auroc"] for r in rs])),
                        "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rs])),
                        "mean_specificity": float(np.mean([r["specificity"] for r in rs])),
                    }
                )
    write_csv(OUT / "compact_result_tables.csv", compact)
    twa = [r for r in compact if r["condition"] == "RA_LL_reversal" and r["endpoint"] == "TWA"]
    best = max(twa, key=lambda r: r["mean_auprc"])
    decision = [
        "# FINAL_DECISION",
        "",
        f"Detector RA-LL recall: {det['ra_ll_recall']:.4f}. Clean specificity: {det['clean_specificity']:.4f}.",
        "",
        "Mean TWA AUPRC under RA-LL reversal:",
    ]
    for r in twa:
        decision.append(f"- {r['arm']}: {r['mean_auprc']:.4f}")
    decision += [
        "",
        f"Best deployable arm by mean RA-LL TWA AUPRC: {best['arm']}.",
        "This is an operational evaluation of the published pretrained detector plus our frozen clean diagnostic model, not a direct reproduction of Huang et al.'s reported paper scores and not a SOTA claim.",
        "No diagnostic model, split, or 792-mask benchmark was retrained or rerun.",
        "",
        "What remains untested: detector calibration or threshold policies beyond argmax, real-world low-prevalence workflow costs beyond the optional illustrative mixture, other cable faults as clinical prevalence scenarios, and any adaptation of Huang's detector to Ningbo.",
    ]
    (OUT / "FINAL_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "run"))
    args = parser.parse_args()
    run(limit=16 if args.command == "smoke" else None)


if __name__ == "__main__":
    main()
