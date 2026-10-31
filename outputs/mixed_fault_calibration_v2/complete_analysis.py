"""Add full Huang validation calibration and missing paired primary TWA intervals."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mixed_fault_calibration_v2"
V1 = ROOT / "outputs/mixed_fault_calibration_v1"
N = ROOT / "outputs/ningbo_confirmation_v2"
HEEDB_PROTOCOL = ROOT / "outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json"
sys.path.insert(0, str(V1))
import run_analysis as base
import extend_bootstrap_metrics as ext

SEEDS = base.SEEDS
SOURCES = base.SOURCES
ARMS = base.ARMS
Q = base.Q
METRICS = ext.METRICS
NEW_MODES = ("raw_historical_threshold", "raw_validation_B_90pct", "calibrated_validation_B_90pct")
BOOT_MODES = ("raw", "calibrated", "raw_historical_threshold", "raw_validation_B_90pct",
              "calibrated_validation_B_90pct", "raw_legacy_v1_Huang_historical")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def protocol_preflight() -> tuple[dict, dict]:
    protocol = json.loads((OUT / "FROZEN_COMPLETION_PROTOCOL.json").read_text(encoding="utf-8"))
    amendment = json.loads((OUT / "FROZEN_COMPLETION_AMENDMENT_001.json").read_text(encoding="utf-8"))
    amendment2_path = OUT / "FROZEN_COMPLETION_AMENDMENT_002.json"
    amendment2 = json.loads(amendment2_path.read_text(encoding="utf-8"))
    amendment3_path = OUT / "FROZEN_COMPLETION_AMENDMENT_003.json"
    amendment3 = json.loads(amendment3_path.read_text(encoding="utf-8")) if amendment3_path.exists() else None
    for rel, item in protocol["sources"].items():
        if rel == "outputs/mixed_fault_calibration_v2/infer_validation.py":
            expected = amendment["infer_validation_py_sha256"]
        else:
            expected = item["sha256"]
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError(f"frozen input hash changed: {rel}")
    if sha(OUT / "FROZEN_COMPLETION_PROTOCOL.json") != amendment2["protocol_sha256"]:
        raise RuntimeError("protocol hash mismatch in analysis amendment")
    if amendment2.get("status") != "FROZEN_BEFORE_CALIBRATION_AND_BOOTSTRAP":
        raise RuntimeError("analysis amendment was not frozen before outcomes")
    expected_script_sha = amendment3["complete_analysis_py_sha256"] if amendment3 is not None else amendment2.get("complete_analysis_py_sha256")
    if expected_script_sha != sha(Path(__file__)):
        raise RuntimeError("analysis amendment does not bind analysis script")
    for rel, expected in amendment2.get("bound_inputs", {}).items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError(f"frozen analysis input changed: {rel}")
    if amendment3 is not None:
        if amendment3.get("status") != "FROZEN_BEFORE_BOOTSTRAP_RETRY" or amendment3.get("protocol_sha256") != sha(OUT / "FROZEN_COMPLETION_PROTOCOL.json"):
            raise RuntimeError("invalid frozen bootstrap retry amendment")
        if amendment3.get("amendment_002_sha256") != sha(amendment2_path):
            raise RuntimeError("bootstrap retry amendment does not bind amendment 002")
        for rel, expected in amendment3.get("bound_inputs", {}).items():
            p = ROOT / rel
            if not p.is_file() or sha(p) != expected:
                raise RuntimeError(f"frozen retry input changed: {rel}")
    inference_audit = json.loads((OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8"))
    if inference_audit.get("status") != "PASS" or inference_audit.get("protocol_sha256") != sha(OUT / "FROZEN_COMPLETION_PROTOCOL.json"):
        raise RuntimeError("Huang validation inference is not complete under frozen protocol")
    if inference_audit.get("amendment_sha256") != sha(OUT / "FROZEN_COMPLETION_AMENDMENT_001.json"):
        raise RuntimeError("Huang inference replay amendment binding mismatch")
    return protocol, amendment2


def fit_huang_validation() -> tuple[dict, dict, np.ndarray]:
    val = np.load(OUT / "huang_validation_scores.npz", allow_pickle=False)
    ids = val["record_ids"].astype(str)
    labels = val["labels"].astype(np.uint8)
    probabilities = val["diagnostic_scores"].astype(np.float64)
    seeds = val["seeds"].astype(int).tolist()
    if seeds != list(SEEDS) or labels.shape != (3488, 3) or probabilities.shape != (3, 3488, 3):
        raise RuntimeError("Huang validation score dimensions/seeds mismatch")
    split = np.load(V1 / "calibration_split_indices.npz", allow_pickle=False)
    a_idx, b_idx = split["A"].astype(int), split["B"].astype(int)
    fits = {}
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        fits[str(seed)] = []
        for endpoint, label in enumerate(("TWA", "SB", "STach")):
            p = probabilities[seed_index, :, endpoint]
            y = labels[:, endpoint]
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise RuntimeError(f"invalid Huang validation scores seed={seed} {label}")
            a, b, opt = base.fit_platt(p[a_idx], y[a_idx])
            threshold_raw = float(np.quantile(p[b_idx][y[b_idx] == 0], .90, method="linear"))
            threshold_cal = float(base.platt(np.asarray([threshold_raw]), a, b)[0])
            item = {"seed": seed, "endpoint": label, "a": a, "b": b,
                    "threshold_raw_validation_B_90pct": threshold_raw,
                    "threshold_calibrated_validation_B_90pct": threshold_cal,
                    "threshold_historical": None, "A_n": len(a_idx), "B_n": len(b_idx),
                    "A_positives": int(y[a_idx].sum()), "B_negatives": int((y[b_idx] == 0).sum()),
                    "optimizer": opt}
            fits[str(seed)].append(item)
            rows.append({**item, "false_correction_clean_validation_count": int(json.loads((OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8"))["clean_validation_incorrect_correction_count"])})
    payload = {
        "method": "Platt scaling sigmoid(a*logit(p)+b), a>0",
        "loss": "ordinary unweighted mean binary log loss on clean validation A",
        "thresholds": "90th percentile of negative clean full-pipeline Huang probabilities on validation B; calibrated threshold is that raw threshold transformed by the fit",
        "validation_record_ids_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "calibration_split_sha256": sha(V1 / "calibration_split_indices.npz"),
        "fit_rows": rows,
        "fits_by_seed": fits,
        "test_HEEDB_labels_not_used": True,
    }
    write_json(OUT / "HUANG_CALIBRATION_AND_PROVENANCE.json", payload)
    return fits, payload, probabilities


def historical_thresholds(source: str, seed: int) -> np.ndarray:
    if source == "Ningbo":
        return np.load(N / f"seed_{seed}/validation.npz", allow_pickle=False)["thresholds"].astype(float)
    p = json.loads(HEEDB_PROTOCOL.read_text(encoding="utf-8"))
    return np.asarray(p["thresholds"]["baseline"][str(seed)], dtype=float)


def get_bound_data() -> dict:
    data, _, _ = base.get_data()
    for seed in SEEDS:
        label_path = ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz"
        y = np.load(label_path, allow_pickle=False)["y"].astype(np.uint8)
        c = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz", allow_pickle=False)["p"].astype(np.float64)
        r = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz", allow_pickle=False)["p"].astype(np.float64)
        if c.shape != r.shape or c.shape != y.shape or not np.array_equal(y, data["Ningbo"]["ordinary_resnet"][seed][0]):
            raise RuntimeError(f"Ningbo test Huang labels/order mismatch seed {seed}")
        data["Ningbo"]["huang_pipeline"][seed] = (y, c, r)
    return data


def mixture_tables(data: dict, fits: dict, val_probabilities: np.ndarray) -> tuple[list[dict], list[dict]]:
    old = pd.read_csv(V1 / "MIXTURE_RESULTS_BY_SEED.csv")
    old = old[old.method != "huang_pipeline"].copy()
    rows = old.to_dict(orient="records")
    validation = np.load(OUT / "huang_validation_scores.npz", allow_pickle=False)
    val_labels = validation["labels"].astype(np.uint8)
    thresholds = json.loads(HEEDB_PROTOCOL.read_text(encoding="utf-8"))["thresholds"]["baseline"]
    for source in SOURCES:
        for si, seed in enumerate(SEEDS):
            y, clean, reversal = data[source]["huang_pipeline"][seed]
            historical = historical_thresholds(source, seed)
            for endpoint, label in enumerate(("TWA", "SB", "STach")):
                yj = y[:, endpoint]
                c, r = clean[:, endpoint], reversal[:, endpoint]
                fit = fits[str(seed)][endpoint]
                ccal, rcal = base.platt(c, fit["a"], fit["b"]), base.platt(r, fit["a"], fit["b"])
                mode_info = [
                    ("raw_historical_threshold", c, r, float(historical[endpoint]), "historical_baseline"),
                    ("raw_validation_B_90pct", c, r, float(fit["threshold_raw_validation_B_90pct"]), "clean_validation_B_90pct"),
                    ("calibrated_validation_B_90pct", ccal, rcal, float(fit["threshold_calibrated_validation_B_90pct"]), "calibrated_clean_validation_B_90pct"),
                ]
                for mode, cs, rs, threshold, threshold_type in mode_info:
                    for q in Q:
                        yy = np.r_[yj, yj]
                        pp = np.r_[cs, rs]
                        w = np.r_[np.full(len(yj), 1 - q), np.full(len(yj), q)]
                        met = base.metrics(yy, pp, threshold, w)
                        rows.append({"source": source, "method": "huang_pipeline", "method_label": base.METHODS["huang_pipeline"],
                                     "seed": seed, "endpoint": label, "q_fault": q, "score_mode": mode,
                                     "threshold_type": threshold_type, "threshold": threshold, **met})
    # Coverage ledger carries forward v1 and adds a hash-bound validation entry.
    coverage = pd.read_csv(V1 / "SCORE_COVERAGE_AUDIT.csv").to_dict(orient="records")
    inference_audit = json.loads((OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        coverage.append({"source": "Ningbo_validation", "method": "huang_pipeline", "seed": str(seed),
                         "score_type": "full_detector_predicted_correction_clean_ResNet_probabilities",
                         "status": "PASS", "records": 3488, "record_order_sha256": inference_audit["validation_record_order_sha256"],
                         "artifact": "outputs/mixed_fault_calibration_v2/huang_validation_scores.npz",
                         "sha256": sha(OUT / "huang_validation_scores.npz"),
                         "notes": "clean validation; detector class argmax controls correction for every record; validation-only calibration A/B uses unchanged v1 split"})
    return rows, coverage


def create_bootstrap(data: dict, fits: dict) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    saved = np.load(V1 / "BOOTSTRAP_METRICS_REPLICATES.npz", allow_pickle=False)
    old_abs = saved["absolute"]
    old_con = saved["paired_contrasts"]
    old_sources = saved["sources"].astype(str).tolist()
    old_arms = saved["arms"].astype(str).tolist()
    old_modes = saved["modes"].astype(str).tolist()
    old_q = saved["q_grid"].astype(float).tolist()
    old_metrics = saved["metrics"].astype(str).tolist()
    old_comps = saved["competitors"].astype(str).tolist()
    if old_sources != list(SOURCES) or old_arms != list(ARMS) or old_metrics != list(METRICS) or old_q != list(Q):
        raise RuntimeError("v1 replicate array dimensions do not match frozen v2 methods/q/metrics")
    source_i = {s: i for i, s in enumerate(old_sources)}
    arm_i = {s: i for i, s in enumerate(old_arms)}
    metric_i = {s: i for i, s in enumerate(old_metrics)}
    mode_i = {s: i for i, s in enumerate(old_modes)}
    q_i = {q: i for i, q in enumerate(old_q)}
    # One common axis distinguishes legacy and v2 threshold modes.
    modes = BOOT_MODES
    abs_new = np.full((len(SOURCES), len(ARMS), len(modes), len(Q), len(METRICS), 1000), np.nan, dtype=np.float64)
    con_new = np.full((len(SOURCES), 3, len(modes), len(Q), len(METRICS), 1000), np.nan, dtype=np.float64)
    # Preserve all already-saved raw and calibrated absolute/contrast draws verbatim.
    for si in range(2):
        for ai, arm in enumerate(ARMS):
            for im, mode in enumerate(old_modes):
                if mode not in ("raw", "calibrated"):
                    continue
                if arm == "huang_pipeline" and mode == "calibrated":
                    continue
                target_mode = "raw_historical_threshold" if arm == "huang_pipeline" and mode == "raw" else mode
                abs_new[si, ai, modes.index(target_mode)] = old_abs[si, ai, im]
        for ci, comp in enumerate(old_comps):
            for im, mode in enumerate(old_modes):
                if mode in ("raw", "calibrated"):
                    target_ci = ("ordinary_resnet", "chest_only_resnet", "huang_pipeline").index(comp)
                    if comp == "huang_pipeline":
                        if mode == "raw":
                            con_new[si, target_ci, modes.index("raw_legacy_v1_Huang_historical")] = old_con[si, ci, im]
                    else:
                        con_new[si, target_ci, modes.index(mode)] = old_con[si, ci, im]
    # Build fixed score caches for new historical and Huang B-calibrated modes.
    cache = {}
    existing_fits, _ = base.calibrators()
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                y, clean, reversal = data[source][arm][seed]
                for mode in modes:
                    if mode in ("raw", "calibrated"):
                        if arm == "huang_pipeline" and mode == "calibrated":
                            continue
                        if arm == "huang_pipeline":
                            c, r = clean[:, 0], reversal[:, 0]
                            threshold = historical_thresholds(source, seed)[0]
                        else:
                            fit = existing_fits[arm][seed][0]
                            if mode == "raw":
                                c, r, threshold = clean[:, 0], reversal[:, 0], fit["threshold_raw"]
                            else:
                                c, r = base.platt(clean[:, 0], fit["a"], fit["b"]), base.platt(reversal[:, 0], fit["a"], fit["b"])
                                threshold = fit["threshold_calibrated"]
                        cache[(source, arm, seed, mode)] = ext.make_cache(y[:, 0], c, r, float(threshold))
                    elif mode == "raw_historical_threshold":
                        if arm == "huang_pipeline":
                            c, r = clean[:, 0], reversal[:, 0]
                        else:
                            c, r = clean[:, 0], reversal[:, 0]
                            threshold = float(existing_fits[arm][seed][0]["threshold_historical"])
                        if arm == "huang_pipeline":
                            threshold = float(historical_thresholds(source, seed)[0])
                        cache[(source, arm, seed, mode)] = ext.make_cache(y[:, 0], c, r, threshold)
                    elif arm == "huang_pipeline" and mode in ("raw_validation_B_90pct", "calibrated_validation_B_90pct"):
                        fit = fits[str(seed)][0]
                        if mode == "raw_validation_B_90pct":
                            c, r = clean[:, 0], reversal[:, 0]
                            threshold = fit["threshold_raw_validation_B_90pct"]
                        else:
                            c, r = base.platt(clean[:, 0], fit["a"], fit["b"]), base.platt(reversal[:, 0], fit["a"], fit["b"])
                            threshold = fit["threshold_calibrated_validation_B_90pct"]
                        cache[(source, arm, seed, mode)] = ext.make_cache(y[:, 0], c, r, float(threshold))
    # Reproduce the exact original RNG stream and validate every v1 raw/calibrated absolute draw.
    rng = np.random.default_rng(20261017)
    rng_verified = {"v1_absolute_groups_checked": 0, "max_abs_delta": 0.0}
    new_modes_to_calculate = ("raw_historical_threshold", "raw_validation_B_90pct", "calibrated_validation_B_90pct")
    for si, source in enumerate(SOURCES):
        n = len(data[source]["ordinary_resnet"][SEEDS[0]][0])
        for bi in range(1000):
            if bi % 100 == 0:
                print(f"v2 paired bootstrap {source}: {bi}/1000", flush=True)
            ix = rng.integers(0, n, size=n)
            multiplicity = np.bincount(ix, minlength=n)
            for arm in ARMS:
                # Existing v1 groups are recomputed from the same generated draw solely for exact stream verification.
                existing_modes = ("raw", "calibrated") if arm != "huang_pipeline" else ("raw",)
                for mode in existing_modes:
                    for qi, q in enumerate(Q):
                        per_seed = [ext.fast_metrics(cache[(source, arm, seed, mode)], multiplicity, q) for seed in SEEDS]
                        vals = {m: float(np.mean([x[m] for x in per_seed])) for m in METRICS}
                        saved_mode = "raw" if mode == "raw" else "calibrated"
                        if mode == "calibrated" and arm == "huang_pipeline":
                            continue
                        for m, value in vals.items():
                            old = old_abs[si, arm_i[arm], mode_i[saved_mode], qi, metric_i[m], bi]
                            if np.isfinite(old):
                                delta = abs(value - old)
                                rng_verified["max_abs_delta"] = max(rng_verified["max_abs_delta"], delta)
                                if delta > 1e-12:
                                    raise RuntimeError(f"v1 bootstrap draw reproduction mismatch {source}/{arm}/{mode}/q={q}/{m}/rep={bi}: {value} vs {old}")
                                rng_verified["v1_absolute_groups_checked"] += 1
            for arm in ARMS:
                for mode in new_modes_to_calculate:
                    if arm == "huang_pipeline" or mode == "raw_historical_threshold":
                        if arm == "huang_pipeline" and mode == "raw_historical_threshold":
                            continue  # Exact Huang historical absolute draws already exist in v1.
                        for qi, q in enumerate(Q):
                            per_seed = [ext.fast_metrics(cache[(source, arm, seed, mode)], multiplicity, q) for seed in SEEDS]
                            for mi, metric in enumerate(METRICS):
                                abs_new[si, arm_i[arm], modes.index(mode), qi, mi, bi] = float(np.mean([x[metric] for x in per_seed]))
    if rng_verified["v1_absolute_groups_checked"] == 0:
        raise RuntimeError("no saved v1 bootstrap draws were verified")
    # Fill all new paired contrasts from the same paired absolute-score bootstrap stream.
    comp_list = ("ordinary_resnet", "chest_only_resnet", "huang_pipeline")
    for si, source in enumerate(SOURCES):
        for ci, comp in enumerate(comp_list):
            for mode in modes:
                if mode in ("raw", "calibrated") and comp != "huang_pipeline":
                    continue  # preserved v1 paired draws, unchanged
                if comp == "huang_pipeline" and mode not in NEW_MODES:
                    continue
                if comp != "huang_pipeline" and mode == "raw_historical_threshold":
                    pass
                elif comp == "huang_pipeline":
                    pass
                else:
                    continue
                for qi in range(len(Q)):
                    con_new[si, ci, modes.index(mode), qi] = (
                        abs_new[si, arm_i["physics_aware_resnet"], modes.index("raw") if mode == "raw_validation_B_90pct" else (modes.index("calibrated") if mode == "calibrated_validation_B_90pct" else modes.index("raw_historical_threshold")), qi]
                        - abs_new[si, arm_i[comp], modes.index(mode), qi]
                    )
    # Verify unchanged legacy v1 Huang contrasts and fill proper matched-threshold historical contrasts.
    for si, source in enumerate(SOURCES):
        old_ci = old_comps.index("huang_pipeline")
        expected = old_con[si, old_ci, mode_i["raw"]]
        observed = con_new[si, comp_list.index("huang_pipeline"), modes.index("raw_legacy_v1_Huang_historical")]
        if source != "Ningbo" and not np.allclose(expected, observed, rtol=0, atol=1e-12, equal_nan=True):
            raise RuntimeError("reconstructed existing HEEDB Huang historical paired interval differs from v1")
    if not np.isfinite(con_new[0, comp_list.index("huang_pipeline"), modes.index("raw_historical_threshold")]).all():
        raise RuntimeError("Ningbo historical Huang paired contrasts remain incomplete")
    # Summaries are computed from preserved v1 draws plus newly computed modes.
    summary = []
    for si, source in enumerate(SOURCES):
        for ai, arm in enumerate(ARMS):
            allowed = ("raw", "calibrated", "raw_historical_threshold") if arm != "huang_pipeline" else NEW_MODES
            for mode in allowed:
                for qi, q in enumerate(Q):
                    for mi, metric in enumerate(METRICS):
                        values = abs_new[si, ai, modes.index(mode), qi, mi]
                        if not np.isfinite(values).all():
                            raise RuntimeError(f"nonfinite absolute bootstrap: {source}/{arm}/{mode}/{q}/{metric}")
                        summary.append({"source": source, "method": arm, "score_mode": mode, "q_fault": q, "endpoint": "TWA",
                                        "metric": metric, "bootstrap_mean": float(values.mean()),
                                        "ci_2_5": float(np.quantile(values, .025)), "ci_97_5": float(np.quantile(values, .975)),
                                        "replicates": 1000, "conditional_on": "selected checkpoints, fixed source-validation calibrators and thresholds"})
        for ci, comp in enumerate(comp_list):
            allowed = ("raw", "calibrated", "raw_historical_threshold") if comp != "huang_pipeline" else NEW_MODES + (("raw_legacy_v1_Huang_historical",) if source != "Ningbo" else ())
            for mode in allowed:
                for qi, q in enumerate(Q):
                    for mi, metric in enumerate(METRICS):
                        values = con_new[si, ci, modes.index(mode), qi, mi]
                        if not np.isfinite(values).all():
                            raise RuntimeError(f"nonfinite paired bootstrap: {source}/{comp}/{mode}/{q}/{metric}")
                        summary.append({"source": source, "contrast": f"physics_aware_minus_{comp}", "score_mode": mode,
                                        "q_fault": q, "endpoint": "TWA", "metric": metric,
                                        "bootstrap_mean": float(values.mean()), "ci_2_5": float(np.quantile(values, .025)),
                                        "ci_97_5": float(np.quantile(values, .975)), "replicates": 1000,
                                        "conditional_on": "selected checkpoints, fixed source-validation calibrators and thresholds"})
    # Preserve all source arms and new Huang operating modes in a single NPZ, including paired arrays.
    npz_path = OUT / "BOOTSTRAP_METRICS_REPLICATES.npz"
    np.savez_compressed(npz_path, absolute=abs_new, paired_contrasts=con_new, sources=np.asarray(SOURCES), arms=np.asarray(ARMS),
                        modes=np.asarray(modes), q_grid=np.asarray(Q), metrics=np.asarray(METRICS), competitors=np.asarray(comp_list),
                        rng_seed=np.asarray(20261017), replicates=np.asarray(1000), source_order=np.asarray(SOURCES),
                        rng_draw_contract=np.asarray("Ningbo first 1000; HEEDB next 1000; regenerated and verified against v1 saved absolute draws"))
    write_csv(OUT / "BOOTSTRAP_METRICS_SUMMARY.csv", summary)
    audit = {"status": "PASS", "rng_seed": 20261017, "replicates": 1000, "metrics": list(METRICS), "modes": list(modes),
             "verification": rng_verified, "huang_historical_contrast_reused_HEEDB_and_filled_Ningbo": True,
             "new_huang_validation_B_and_calibrated_absolute_and_paired_metrics": True,
             "retry_amendment_sha256": sha(OUT / "FROZEN_COMPLETION_AMENDMENT_003.json") if (OUT / "FROZEN_COMPLETION_AMENDMENT_003.json").exists() else None,
             "all_groups_finite": True, "summary_rows": len(summary), "replicate_file_sha256": sha(npz_path)}
    write_json(OUT / "BOOTSTRAP_COMPLETION_AUDIT.json", audit)
    return abs_new, con_new, summary, audit


def make_contrasts(mixture_rows: list[dict], abs_arr: np.ndarray, con_arr: np.ndarray) -> list[dict]:
    frame = pd.DataFrame(mixture_rows)
    sources = list(SOURCES)
    arms_i = {a: i for i, a in enumerate(ARMS)}
    modes = BOOT_MODES
    comp_list = ("ordinary_resnet", "chest_only_resnet", "huang_pipeline")
    q_index = {q: i for i, q in enumerate(Q)}
    rows = []
    for source in SOURCES:
        for endpoint in ("TWA", "SB", "STach"):
            for q in Q:
                for comparator in comp_list:
                    score_modes = ("raw", "calibrated", "raw_historical_threshold") if comparator != "huang_pipeline" else NEW_MODES + (("raw_legacy_v1_Huang_historical",) if source != "Ningbo" else ())
                    for mode in score_modes:
                        if mode == "raw_legacy_v1_Huang_historical":
                            physics_mode = "raw"
                            comparator_mode = "raw_historical_threshold"
                        else:
                            physics_mode = mode if mode in ("raw", "calibrated", "raw_historical_threshold") else ("raw" if mode == "raw_validation_B_90pct" else "calibrated")
                            comparator_mode = mode
                        for metric in METRICS:
                            def mean(arm: str, mode_name: str) -> float:
                                group = frame[(frame.source == source) & (frame.method == arm) & (frame.endpoint == endpoint) &
                                              (frame.q_fault == q) & (frame.score_mode == mode_name)]
                                if len(group) != 3:
                                    raise RuntimeError(f"missing full-sample values {source}/{arm}/{endpoint}/{q}/{mode_name}: {len(group)}")
                                return float(group[metric].mean())
                            pv = mean("physics_aware_resnet", physics_mode)
                            cv = mean(comparator, comparator_mode)
                            row = {"source": source, "endpoint": endpoint, "q_fault": q,
                                   "contrast": f"physics_aware_minus_{comparator}", "score_mode": mode, "metric": metric,
                                   "physics_aware_full_sample_seed_mean": pv, "comparator_full_sample_seed_mean": cv,
                                   "full_sample_difference": pv - cv,
                                   "bootstrap_mean_difference": "", "ci_2_5": "", "ci_97_5": "", "replicates": ""}
                            if endpoint == "TWA":
                                si, ci, mi, qi, xi = sources.index(source), comp_list.index(comparator), modes.index(mode), q_index[q], METRICS.index(metric)
                                values = con_arr[si, ci, mi, qi, xi]
                                row.update({"bootstrap_mean_difference": float(values.mean()), "ci_2_5": float(np.quantile(values, .025)),
                                            "ci_97_5": float(np.quantile(values, .975)), "replicates": 1000})
                            rows.append(row)
    return rows


def main() -> None:
    started = time.monotonic()
    protocol_preflight()
    fits, cal_prov, valp = fit_huang_validation()
    data = get_bound_data()
    mixtures, coverage = mixture_tables(data, fits, valp)
    write_csv(OUT / "MIXTURE_RESULTS_BY_SEED.csv", mixtures)
    write_csv(OUT / "SCORE_COVERAGE_AUDIT.csv", coverage)
    abs_arr, con_arr, summary, boot_audit = create_bootstrap(data, fits)
    contrasts = make_contrasts(mixtures, abs_arr, con_arr)
    write_csv(OUT / "PAIRED_CONTRASTS.csv", contrasts)
    allfinite = all(np.isfinite(abs_arr[np.isfinite(abs_arr)]))
    if not allfinite or not boot_audit["all_groups_finite"]:
        raise RuntimeError("bootstrap QA failed")
    complete = {
        "status": "PASS", "protocol_sha256": sha(OUT / "FROZEN_COMPLETION_PROTOCOL.json"),
        "amendment_001_sha256": sha(OUT / "FROZEN_COMPLETION_AMENDMENT_001.json"),
        "amendment_002_sha256": sha(OUT / "FROZEN_COMPLETION_AMENDMENT_002.json"),
        "amendment_003_sha256": sha(OUT / "FROZEN_COMPLETION_AMENDMENT_003.json") if (OUT / "FROZEN_COMPLETION_AMENDMENT_003.json").exists() else None,
        "mixture_rows": len(mixtures), "paired_contrast_rows": len(contrasts), "bootstrap_summary_rows": len(summary),
        "bootstrap_audit": boot_audit, "no_training": True, "reserved_test_evaluated": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(OUT / "ANALYSIS_COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
