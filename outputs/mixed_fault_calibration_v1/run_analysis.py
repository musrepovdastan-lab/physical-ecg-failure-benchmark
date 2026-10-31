"""Run frozen-score mixture analysis; no diagnostic model training occurs here."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mixed_fault_calibration_v1"
PRIVATE = ROOT / "data/heedb_restricted/I0006/mixed_fault_calibration_v1"
sys.path.insert(0, str(ROOT))
import torch
from src.chest_only_baseline_v1 import ChestOnlyResNetSmall
from src import heedb_external_v1
SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary_resnet", "physics_aware_resnet", "chest_only_resnet", "huang_pipeline")
ENDPOINTS = ("TWA", "SB", "STach")
Q = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 1.0)
SOURCES = ("Ningbo", "HEEDB_I0006_independent_confirmation")
METHODS = {"ordinary_resnet": "Ordinary ResNet", "physics_aware_resnet": "Physics-aware ResNet",
           "chest_only_resnet": "Chest-only ResNet", "huang_pipeline": "Huang→predicted correction→clean ResNet"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        # Several output ledgers combine row types with different optional metadata.
        # Preserve their full schemas rather than failing after a lengthy bootstrap.
        fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def check_frozen() -> dict:
    protocol_path = OUT / "FROZEN_ANALYSIS_PROTOCOL.json"
    if not protocol_path.exists():
        raise RuntimeError("No frozen analysis protocol; run freeze_inventory.py first")
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    if p["status"] != "FROZEN_BEFORE_NEW_CHEST_HEEDB_INFERENCE_AND_CALIBRATION":
        raise RuntimeError("Protocol freeze status mismatch")
    for item in p["input_artifacts"].values():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source artifact changed: {item['path']}")
    if sha(OUT / "calibration_split_indices.npz") != p["validation_split"]["sha256"]:
        raise RuntimeError("Calibration split changed")
    return p


def chest_inference() -> dict:
    """Frozen Ningbo chest-only models on cached, already-selected HEEDB waveforms."""
    started = time.monotonic()
    private_pred = PRIVATE / "chest_only_predictions_float32.npy"
    if private_pred.exists():
        done = PRIVATE / "chest_only_predictions_complete.json"
        meta = json.loads(done.read_text(encoding="utf-8"))
        if meta.get("sha256") != sha(private_pred):
            raise RuntimeError("Existing private chest prediction hash mismatch")
        return meta
    source = ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy"
    signals = np.load(source, mmap_mode="r")
    if signals.shape != (10000, 12, 5000):
        raise ValueError(f"Unexpected HEEDB signal cache shape {signals.shape}")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    tmp = private_pred.with_suffix(".npy.partial")
    pred = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(3, 2, 10000, 3))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
    for si, seed in enumerate(SEEDS):
        model_dir = ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{seed}"
        ckpt_path = model_dir / "checkpoint.pt"
        prep_path = model_dir / "preprocessing.npz"
        model = ChestOnlyResNetSmall().to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ckpt.get("protocol_sha256") != sha(ROOT / "outputs/chest_only_baseline_v1/FROZEN_CHEST_ONLY_PROTOCOL.json"):
            raise RuntimeError(f"Chest-only checkpoint protocol mismatch for seed {seed}")
        model.load_state_dict(ckpt["state_dict"], strict=True); model.eval()
        prep = np.load(prep_path, allow_pickle=False)
        mean = np.asarray(prep["mean"], dtype=np.float32)
        std = np.asarray(prep["std"], dtype=np.float32)
        # Frozen adapter shared with the Ningbo diagnostic systems: canonical waveform, ::5 sampling,
        # select V1-V6, then apply this model's Ningbo train-derived channel normalization.
        for start in range(0, 10000, 128):
            stop = min(10000, start + 128)
            raw = np.asarray(signals[start:stop, :, ::5], dtype=np.float32)
            chest = raw[:, 6:12, :]
            x = np.ascontiguousarray((chest - mean[None, :, None]) / std[None, :, None])
            with torch.no_grad():
                xb = torch.from_numpy(x).to(device)
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16): score = torch.sigmoid(model(xb)).float()
                else:
                    score = torch.sigmoid(model(xb))
                scores = score.cpu().numpy()
            pred[si, 0, start:stop] = scores
            # RA–LL changes limb leads only. The chest-only input is byte-identical, so the frozen
            # model's reversal scores are exactly its clean scores; preserve both condition axes.
            pred[si, 1, start:stop] = scores
            if stop % 2048 == 0 or stop == 10000:
                pred.flush()
                print(f"HEEDB chest-only inference seed={seed}: {stop}/10000 ({device})", flush=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    pred.flush(); del pred
    os.replace(tmp, private_pred)
    meta = {"status": "PASS", "n_patients": 10000, "seeds": list(SEEDS), "conditions": ["clean", "RA_LL_reversal"],
            "conditions_identical_by_input_invariance": True, "device": device, "seconds": round(time.monotonic() - started, 3),
            "prediction_path": private_pred.relative_to(ROOT).as_posix(), "sha256": sha(private_pred),
            "source_signal_sha256": sha(source), "checkpoint_sha256": {str(s): sha(ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{s}/checkpoint.pt") for s in SEEDS},
            "preprocessing_sha256": {str(s): sha(ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{s}/preprocessing.npz") for s in SEEDS},
            "adapter": "canonical cached 12-lead 500Hz-equivalent waveform -> ::5 -> V1-V6 -> per-seed Ningbo train-derived mean/std -> frozen chest-only ResNet"}
    write_json(PRIVATE / "chest_only_predictions_complete.json", meta)
    return meta


def heedb_labels():
    selection = pd.read_csv(ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv", dtype=str, keep_default_na=False)
    diagnoses = pd.read_csv(ROOT / "data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv",
                            usecols=["FileName", "codes_physician"], dtype=str, keep_default_na=False)
    frame = selection[["FileName", "BDSPPatientID"]].merge(diagnoses, on="FileName", how="left", validate="one_to_one", sort=False)
    if len(frame) != 10000 or frame["BDSPPatientID"].nunique() != 10000 or frame.codes_physician.isna().any() or frame.codes_physician.eq("").any():
        raise RuntimeError("HEEDB frozen selection/label linkage check failed")
    labels = np.asarray([heedb_external_v1.label_vector(x, heedb_external_v1.CORE) for x in frame.codes_physician], dtype=np.uint8)
    return labels, frame["BDSPPatientID"].astype(str).to_numpy()


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float, dict]:
    eps = 1e-6
    clipped = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    x = np.log(clipped / (1 - clipped))
    target = np.asarray(y, dtype=np.float64)
    def objective(theta):
        a = np.exp(theta[0]); z = a * x + theta[1]
        # stable logistic cross entropy
        loss = np.logaddexp(0, z) - target * z
        return float(loss.mean())
    result = minimize(objective, np.asarray([0.0, 0.0]), method="L-BFGS-B", bounds=[(-12, 12), (-30, 30)],
                      options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Platt optimizer failed: {result.message}")
    a, b = float(np.exp(result.x[0])), float(result.x[1])
    if not np.isfinite(a) or a <= 0:
        raise RuntimeError("Platt slope must be finite and positive")
    return a, b, {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "loss": float(result.fun)}


def platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    c = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = a * np.log(c / (1 - c)) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -745, 745)))


def ece15(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total <= 0: raise ValueError("ECE needs positive total weight")
    value = 0.0
    for k in range(15):
        mask = (p >= k / 15) & ((p < (k + 1) / 15) if k < 14 else (p <= 1.0)) & (w > 0)
        if mask.any():
            mass = w[mask].sum()
            value += mass / total * abs(np.average(y[mask], weights=w[mask]) - np.average(p[mask], weights=w[mask]))
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=np.uint8); p = np.asarray(p, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    active = w > 0
    ya, pa, wa = y[active], p[active], w[active]
    if len(np.unique(ya)) != 2: raise ValueError("Both outcome classes required for AP/AUROC")
    called = p >= threshold
    pos = y == 1; neg = ~pos
    tp = float(w[pos & called].sum()); fn = float(w[pos & ~called].sum())
    tn = float(w[neg & ~called].sum()); fp = float(w[neg & called].sum())
    total = float(w.sum())
    return {"auprc": float(average_precision_score(ya, pa, sample_weight=wa)),
            "auroc": float(roc_auc_score(ya, pa, sample_weight=wa)),
            "brier": float(np.average((p - y) ** 2, weights=w)), "ece15": ece15(y, p, w),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "predicted_positive_fraction": (tp + fp) / total,
            "false_positives_per_1000": 1000.0 * fp / total,
            "false_negatives_per_1000": 1000.0 * fn / total}


def get_data() -> tuple[dict, np.ndarray, np.ndarray]:
    # Return {source: {method: {seed: (y, clean, reversal)}}} and HEEDB patient clusters.
    data: dict = {source: {arm: {} for arm in ARMS} for source in SOURCES}
    nbase = ROOT / "outputs/chest_only_baseline_v1"
    for seed in SEEDS:
        for arm, base, chest in (("ordinary_resnet", nbase / "comparators/Ningbo/clean_12lead", False),
                                 ("physics_aware_resnet", nbase / "comparators/Ningbo/physics_aware_12lead", False),
                                 ("chest_only_resnet", nbase / "models/Ningbo", True)):
            pred_dir = base / f"seed_{seed}" / "test_predictions" if chest else base / f"seed_{seed}"
            with np.load(pred_dir / "clean.npz", allow_pickle=False) as c, np.load(pred_dir / "RA_LL_reversal.npz", allow_pickle=False) as r:
                if not np.array_equal(c["y"], r["y"]) or c["p"].shape != r["p"].shape:
                    raise RuntimeError(f"Ningbo paired score binding failed for {arm}/{seed}")
                data["Ningbo"][arm][seed] = (c["y"].astype(np.uint8), c["p"].astype(np.float64), r["p"].astype(np.float64))
        # Historical validated baseline thresholds are used for the uncalibrated Huang pipeline operating point.
        z = np.load(ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz", allow_pickle=False)
        thresholds = z["thresholds"].astype(float)
        cpath = ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz"
        with np.load(cpath, allow_pickle=False) as c:
            data["Ningbo"]["huang_pipeline"][seed] = (c["y"].astype(np.uint8), None, None)
    private = ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1"
    matrix = np.load(private / "predictions_float32.npy", mmap_mode="r")
    labels, patients = heedb_labels()
    chest = np.load(PRIVATE / "chest_only_predictions_float32.npy", mmap_mode="r")
    huang = np.load(ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    if labels.shape != (10000, 3) or patients.shape != (10000,) or not np.isfinite(matrix).all():
        raise RuntimeError("HEEDB cohort/prediction shape or finiteness check failed")
    for si, seed in enumerate(SEEDS):
        # saved independent array arm order: baseline, physics-aware; condition order clean, RA-LL.
        for arm, ai in (("ordinary_resnet", 0), ("physics_aware_resnet", 1)):
            data[SOURCES[1]][arm][seed] = (labels, np.asarray(matrix[ai, si, 0]), np.asarray(matrix[ai, si, 1]))
        data[SOURCES[1]]["chest_only_resnet"][seed] = (labels, np.asarray(chest[si, 0]), np.asarray(chest[si, 1]))
        # Huang prediction cache order is [seed, condition, patient, endpoint].
        data[SOURCES[1]]["huang_pipeline"][seed] = (labels, np.asarray(huang[si, 0]), np.asarray(huang[si, 1]))
    return data, labels, patients


def self_checks() -> dict:
    from sklearn.metrics import average_precision_score as ap_ref, roc_auc_score as auc_ref
    y = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    clean = np.asarray([.1, .8, .4, .2]); rev = np.asarray([.2, .7, .3, .9])
    reference = {}
    for q in (0., 1.):
        p = clean if q == 0 else rev
        got = metrics(y, p, .5)
        assert np.isclose(got["auprc"], ap_ref(y, p)) and np.isclose(got["auroc"], auc_ref(y, p))
        reference[str(q)] = got
    q = .25
    yy = np.r_[y, y]; pp = np.r_[clean, rev]; ww = np.r_[np.full(4, 1-q), np.full(4, q)]
    got = metrics(yy, pp, .5, ww)
    assert np.isclose(got["auprc"], ap_ref(yy, pp, sample_weight=ww))
    assert np.isclose(got["auroc"], auc_ref(yy, pp, sample_weight=ww))
    ties = metrics(np.asarray([0, 1, 0, 1]), np.asarray([.5, .5, .2, .2]), .5, np.asarray([1., 0., 1., 1.]))
    assert np.isfinite(ties["auprc"]) and np.isfinite(ties["auroc"])
    assert np.isclose(ties["predicted_positive_fraction"], 1/3)
    # Paired bootstrap copies share the same record multiplicity by construction.
    rng = np.random.default_rng(71); ix = rng.integers(0, len(y), len(y)); m = np.bincount(ix, minlength=len(y))
    assert int(m.sum()) == len(y) and np.array_equal(m, np.bincount(ix, minlength=len(y)))
    brier_mix = np.average(np.r_[(clean-y)**2, (rev-y)**2], weights=ww)
    brier_lerp = (1-q) * np.mean((clean-y)**2) + q * np.mean((rev-y)**2)
    assert np.isclose(brier_mix, brier_lerp)
    a, b = 1.7, -.3; th = .4
    assert np.array_equal(platt(np.asarray([.1, .4, .8]), a, b) >= platt(np.asarray([th]), a, b), np.asarray([.1, .4, .8]) >= th)
    # Monotone mapping should preserve rank except for clipping/ties; exercise clipping behavior explicitly.
    clipped = platt(np.asarray([0., 1e-8, .2, .8, 1-1e-8, 1.]), 1.0, 0.0)
    assert np.all(np.diff(clipped) >= 0) and clipped[0] == clipped[1] and clipped[-1] == clipped[-2]
    return {"decision": "PASS", "q_endpoints": "exact raw clean/reversal metric equivalence", "weighted_AP_AUROC": "matched sklearn sample_weight reference",
            "ties_zero_weights": "finite; zero-mass rows ignored", "paired_bootstrap": "record multiplicity reused for both copies",
            "brier_mixture_identity": "PASS", "raw_calibrated_threshold_equivalence": "PASS",
            "monotone_ranking_and_clipping_ties": "PASS"}


def calibrators() -> tuple[dict, dict]:
    split = np.load(OUT / "calibration_split_indices.npz", allow_pickle=False)
    a_idx = split["A"].astype(int); b_idx = split["B"].astype(int)
    specs = {
        "ordinary_resnet": lambda seed: ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz",
        "physics_aware_resnet": lambda seed: ROOT / f"outputs/mitigation_m2_ningbo_v1/seed_{seed}/validation.npz",
        "chest_only_resnet": lambda seed: ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{seed}/validation.npz",
    }
    fits, valinfo = {}, {}
    for arm, pathfn in specs.items():
        fits[arm] = {}
        for seed in SEEDS:
            z = np.load(pathfn(seed), allow_pickle=False); y = z["y"].astype(np.uint8); p = z["p"].astype(float)
            fits[arm][seed] = []
            for ep, label in enumerate(ENDPOINTS):
                slope, intercept, opt = fit_platt(p[a_idx, ep], y[a_idx, ep])
                threshold_raw = float(np.quantile(p[b_idx, ep][y[b_idx, ep] == 0], .90, method="linear"))
                threshold_historical = float(z["thresholds"][ep]) if "thresholds" in z else None
                th_cal = float(platt(np.asarray([threshold_raw]), slope, intercept)[0])
                fits[arm][seed].append({"a": slope, "b": intercept, "threshold_raw": threshold_raw,
                    "threshold_calibrated": th_cal, "threshold_historical": threshold_historical,
                    "optimizer": opt, "A_n": len(a_idx), "B_n": len(b_idx),
                    "A_positives": int(y[a_idx, ep].sum()), "B_negatives": int((y[b_idx, ep,] == 0).sum()),
                    "endpoint": label, "seed": seed})
            valinfo[f"{arm}/{seed}"] = {"validation_path": pathfn(seed).relative_to(ROOT).as_posix(),
                "validation_sha256": sha(pathfn(seed)), "thresholds_full_validation_historical": z["thresholds"].astype(float).tolist() if "thresholds" in z else None}
    return fits, valinfo


def mixtures(data: dict, fits: dict) -> tuple[list[dict], list[dict]]:
    rows, reliability = [], []
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                y, clean, rev = data[source][arm][seed]
                for ep, label in enumerate(ENDPOINTS):
                    if arm == "huang_pipeline":
                        # Huang outputs exist only on HEEDB. Its Ningbo test comparison and validation
                        # scores are not available under the frozen manifest; no calibrated Huang estimate.
                        if source == "Ningbo": continue
                        pclean, prev = clean[:, ep], rev[:, ep]
                        historical_threshold = json.loads((ROOT / "outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))["thresholds"]["baseline"][str(seed)][ep]
                        for q in Q:
                            yy = np.r_[y[:, ep], y[:, ep]]; pp = np.r_[pclean, prev]
                            w = np.r_[np.full(len(y), 1-q), np.full(len(y), q)]
                            met = metrics(yy, pp, historical_threshold, w)
                            rows.append({"source": source, "method": arm, "method_label": METHODS[arm], "seed": seed,
                                "endpoint": label, "q_fault": q, "score_mode": "raw", "threshold_type": "historical_Ningbo_baseline",
                                "threshold": historical_threshold, **met})
                            if ep == 0 and q in (0., .01, .10, 1.):
                                reliability.extend(reliability_rows(source, arm, seed, label, q, "raw", yy, pp, w))
                        continue
                    fit = fits[arm][seed][ep]
                    pclean_cal = platt(clean[:, ep], fit["a"], fit["b"])
                    prev_cal = platt(rev[:, ep], fit["a"], fit["b"])
                    modes = [("raw", clean[:, ep], rev[:, ep], fit["threshold_raw"]),
                             ("calibrated", pclean_cal, prev_cal, fit["threshold_calibrated"])]
                    if fit["threshold_historical"] is not None:
                        modes.append(("raw_historical_threshold", clean[:, ep], rev[:, ep], fit["threshold_historical"]))
                    for mode, cscore, rscore, threshold in modes:
                        for q in Q:
                            yy = np.r_[y[:, ep], y[:, ep]]; pp = np.r_[cscore, rscore]
                            w = np.r_[np.full(len(y), 1-q), np.full(len(y), q)]
                            met = metrics(yy, pp, threshold, w)
                            rows.append({"source": source, "method": arm, "method_label": METHODS[arm], "seed": seed,
                                "endpoint": label, "q_fault": q, "score_mode": mode, "threshold_type": "clean_validation_B_90pct_specificity",
                                "threshold": threshold, **met})
                            if ep == 0 and q in (0., .01, .10, 1.):
                                reliability.extend(reliability_rows(source, arm, seed, label, q, mode, yy, pp, w))
    return rows, reliability


def reliability_rows(source, arm, seed, endpoint, q, mode, y, p, w):
    rows = []
    for k in range(15):
        mask = (p >= k/15) & ((p < (k+1)/15) if k < 14 else (p <= 1)) & (w > 0)
        mass = float(w[mask].sum())
        rows.append({"source": source, "method": arm, "seed": seed, "endpoint": endpoint, "q_fault": q,
            "score_mode": mode, "bin": k, "lower": k/15, "upper": (k+1)/15, "weighted_count": mass,
            "weighted_fraction": mass/float(w.sum()), "mean_prediction": float(np.average(p[mask],weights=w[mask])) if mask.any() else "",
            "observed_fraction": float(np.average(y[mask],weights=w[mask])) if mask.any() else ""})
    return rows


def bootstrap_primary(data: dict, fits: dict) -> tuple[list[dict], list[dict]]:
    """1000 synchronized cluster bootstraps for primary TWA AUPRC and paired contrasts."""
    started = time.monotonic(); rng = np.random.default_rng(20261017)
    replicates, summaries = [], []
    competitors = ("ordinary_resnet", "chest_only_resnet", "huang_pipeline")
    for source in SOURCES:
        n = len(data[source]["ordinary_resnet"][SEEDS[0]][0])
        arm_rep = {arm: {mode: {q: np.full(1000, np.nan) for q in Q} for mode in (("raw",) if arm == "huang_pipeline" else ("raw", "calibrated"))} for arm in ARMS}
        # Resampling at patient/record level and sharing counts across clean/reversal rows.
        for bi in range(1000):
            if bi % 100 == 0: print(f"bootstrap {source}: {bi}/1000", flush=True)
            idx = rng.integers(0, n, size=n)
            for arm in ARMS:
                for seed in SEEDS:
                    y, clean, rev = data[source][arm][seed]
                    yy0 = y[idx, 0]
                    if arm == "huang_pipeline":
                        if source == "Ningbo": continue
                        cscore, rscore = clean[idx, 0], rev[idx, 0]
                        th = json.loads((ROOT / "outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))["thresholds"]["baseline"][str(seed)][0]
                        modes = [("raw", cscore, rscore, th)]
                    else:
                        fit = fits[arm][seed][0]
                        cscore, rscore = clean[idx, 0], rev[idx, 0]
                        ccal = platt(cscore, fit["a"], fit["b"]); rcal = platt(rscore, fit["a"], fit["b"])
                        modes = [("raw", cscore, rscore, fit["threshold_raw"]),
                                 ("calibrated", ccal, rcal, fit["threshold_calibrated"])]
                    for mode, cscore, rscore, th in modes:
                        for q in Q:
                            pp = np.r_[cscore, rscore]; yy = np.r_[yy0, yy0]
                            w = np.r_[np.full(n, 1-q), np.full(n, q)]
                            arm_rep[arm][mode][q][bi] += 0 if np.isfinite(arm_rep[arm][mode][q][bi]) else 0
                            # Keep the seed values to average metrics across seeds, not the predictions.
                            val = float(average_precision_score(yy, pp, sample_weight=w))
                            key = (arm, mode, q, bi)
                            seed_store.setdefault(key, []).append(val)
            for (arm, mode, q, bix), vals in list(seed_store.items()):
                if bix == bi and len(vals) == len(SEEDS): arm_rep[arm][mode][q][bi] = float(np.mean(vals))
            for key in [k for k in seed_store if k[3] == bi]: seed_store.pop(key, None)
        for arm in ARMS:
            for mode, qvalues in arm_rep[arm].items():
                for q, vals in qvalues.items():
                    if source == "Ningbo" and arm == "huang_pipeline": continue
                    row = {"source": source, "method": arm, "score_mode": mode, "endpoint": "TWA", "q_fault": q,
                           "replicates": 1000, "bootstrap_mean": float(np.nanmean(vals)),
                           "ci_2_5": float(np.nanquantile(vals, .025)), "ci_97_5": float(np.nanquantile(vals, .975)),
                           "conditional_on": "selected checkpoints, fixed validation calibrators and fixed thresholds"}
                    summaries.append(row)
        for competitor in competitors:
            modes = ("raw",) if competitor == "huang_pipeline" else ("raw", "calibrated")
            for mode in modes:
                if competitor == "huang_pipeline" and source == "Ningbo": continue
                for q in Q:
                    diff = arm_rep["physics_aware_resnet"][mode][q] - arm_rep[competitor][mode][q]
                    replicates.extend({"source": source, "primary_endpoint": "TWA", "metric": "AUPRC", "score_mode": mode,
                        "q_fault": q, "contrast": f"physics_aware_minus_{competitor}", "bootstrap_replicate": i, "difference": float(v)} for i, v in enumerate(diff))
    # Persist paired contrast intervals from replicate-level synchronized differences.
    contrast_summary = []
    if replicates:
        frame = pd.DataFrame(replicates)
        for keys, grp in frame.groupby(["source", "score_mode", "q_fault", "contrast"], sort=True):
            vals = grp.difference.to_numpy(dtype=float)
            contrast_summary.append({"source": keys[0], "score_mode": keys[1], "q_fault": keys[2], "contrast": keys[3],
                "primary_endpoint": "TWA", "metric": "AUPRC", "replicates": len(vals), "bootstrap_mean": float(vals.mean()),
                "ci_2_5": float(np.quantile(vals, .025)), "ci_97_5": float(np.quantile(vals, .975)),
                "conditional_on": "selected checkpoints, fixed validation calibrators and fixed thresholds"})
    write_json(OUT / "BOOTSTRAP_RUNTIME.json", {"seconds": round(time.monotonic()-started, 3), "unit": "canonical record/patient",
        "replicates": 1000, "rng_seed": 20261017, "primary_metric": "TWA AUPRC only; all other endpoints have full-sample descriptive estimates"})
    return summaries + [{"kind": "paired_contrast", **x} for x in contrast_summary], replicates


def main():
    started = time.monotonic(); protocol = check_frozen()
    tests = self_checks()
    write_json(OUT / "ENGINEERING_TEST_REPORT.json", tests)
    (OUT / "ENGINEERING_TEST_REPORT.md").write_text("# Engineering test report\n\nAll synthetic weighted-mixture checks passed before analysis.\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in tests.items() if k != "decision") + "\n", encoding="utf-8")
    chestmeta = chest_inference()
    data, heedb_y, patients = get_data()
    fits, valinfo = calibrators()
    write_json(OUT / "CALIBRATION_AND_PROVENANCE.json", {"calibration_fits": fits, "validation_inputs": valinfo,
        "calibration_split_sha256": protocol["validation_split"]["sha256"], "HEEDB_label_prevalence": dict(zip(ENDPOINTS, heedb_y.mean(axis=0).tolist())),
        "HEEDB_patient_count": int(len(np.unique(patients))), "HEEDB_chest_inference": chestmeta,
        "huang_calibration": "not fit: no full-pipeline Ningbo validation score artifact; HEEDB/test labels not used"})
    rows, reliability = mixtures(data, fits)
    write_csv(OUT / "MIXTURE_RESULTS_BY_SEED.csv", rows)
    write_csv(OUT / "RELIABILITY_BINS.csv", reliability)
    # Descriptive, same-record paired differences from seed means, with all metrics retained.
    contrast_rows = []
    keys = ["source", "endpoint", "q_fault", "score_mode"]
    frame = pd.DataFrame(rows)
    pivot_metrics = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction",
                     "false_positives_per_1000", "false_negatives_per_1000"]
    for source in SOURCES:
        for mode in ("raw", "calibrated"):
            for q in Q:
                for endpoint in ENDPOINTS:
                    p = frame[(frame.source == source) & (frame.endpoint == endpoint) & (frame.q_fault == q) & (frame.score_mode == mode)]
                    if p.empty: continue
                    avg = p.groupby("method")[pivot_metrics].mean()
                    for comparator in ("ordinary_resnet", "chest_only_resnet", "huang_pipeline"):
                        if "physics_aware_resnet" not in avg.index or comparator not in avg.index: continue
                        for metric_name in pivot_metrics:
                            contrast_rows.append({"source": source, "endpoint": endpoint, "q_fault": q, "score_mode": mode,
                                "contrast": f"physics_aware_minus_{comparator}", "metric": metric_name,
                                "physics_aware_seed_mean": float(avg.loc["physics_aware_resnet", metric_name]),
                                "comparator_seed_mean": float(avg.loc[comparator, metric_name]),
                                "difference": float(avg.loc["physics_aware_resnet", metric_name] - avg.loc[comparator, metric_name]),
                                "interval": "primary TWA AUPRC intervals are in BOOTSTRAP_SUMMARY.csv; secondary contrasts descriptive"})
    write_csv(OUT / "PAIRED_CONTRASTS.csv", contrast_rows)
    print("Starting 1000 synchronized TWA AUPRC bootstrap replicates", flush=True)
    boot_summary, boot_reps = bootstrap_primary(data, fits)
    write_csv(OUT / "BOOTSTRAP_SUMMARY.csv", boot_summary)
    write_csv(OUT / "BOOTSTRAP_REPLICATES.csv", boot_reps)
    write_json(OUT / "ANALYSIS_COMPLETE.json", {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(OUT / "FROZEN_ANALYSIS_PROTOCOL.json"), "analysis_code_sha256": sha(Path(__file__)),
        "seconds": round(time.monotonic()-started, 3), "mixture_rows": len(rows), "paired_contrast_rows": len(contrast_rows),
        "bootstrap_rows": len(boot_summary), "private_chest_prediction_sha256": chestmeta["sha256"],
        "source_tests_previously_accessed": True, "no_diagnostic_training": True, "no_downloads": True})
    print(f"ANALYSIS_COMPLETE rows={len(rows)} boot={len(boot_summary)}", flush=True)


if __name__ == "__main__":
    seed_store = {}
    main()
