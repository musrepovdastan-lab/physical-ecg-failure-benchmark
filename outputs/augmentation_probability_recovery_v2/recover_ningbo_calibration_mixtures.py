from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
SPLIT_PATH = ROOT / "outputs/mixed_fault_calibration_v1/calibration_split_indices.npz"
sys.path.insert(0, str(ROOT))
from outputs.augmentation_probability_study_v1.analyze_ningbo_mixtures import fit_platt, weighted_ece

SEEDS = (104729, 130363, 155921)
ARMS = ("physics_p010", "physics_p025")
LABELS = ("TWA", "SB", "STach")
Q = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = np.log(clipped / (1 - clipped))
    return expit(a * z + b)


def weighted_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    pos = y == 1
    neg = ~pos
    return {
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "ece15": weighted_ece(y, p, w),
        "sensitivity": float(w[pred & pos].sum() / w[pos].sum()),
        "specificity": float(w[(~pred) & neg].sum() / w[neg].sum()),
        "predicted_positive_fraction": float(w[pred].sum() / w.sum()),
        "accuracy": float(np.average((pred == y).astype(np.float64), weights=w)),
        "threshold": float(threshold),
    }


def main() -> None:
    status = json.loads((OUT / "RECOVERED_METRICS_STATUS.json").read_text(encoding="utf-8"))
    if status["stage"] != "RAW_METRICS_AND_OPERATING_POINTS_RECOVERED":
        raise RuntimeError("Recovered test metrics are not ready")
    with np.load(SPLIT_PATH, allow_pickle=False) as z:
        A, B = z["A"].astype(np.int64), z["B"].astype(np.int64)
    old = pd.read_csv(STUDY / "MIXTURE_RESULTS_NEW_BY_SEED.csv")
    rows, fits, decision_audits = [], {}, []
    for arm in ARMS:
        for seed in SEEDS:
            valfile = OUT / "validation" / "Ningbo" / arm / f"seed_{seed}.npz"
            with np.load(valfile, allow_pickle=False) as z:
                yv, pv = z["y"].astype(np.uint8), z["p"].astype(np.float64)
                val_ids = z["record_ids"].astype(str).tolist()
            if len(yv) != 3488 or yv.shape[1] != 3 or pv.shape != yv.shape:
                raise RuntimeError(f"wrong frozen Ningbo validation shape for {arm}/{seed}")
            if len(A) != len(B) or set(A.tolist()) & set(B.tolist()) or sorted(np.r_[A, B].tolist()) != list(range(len(yv))):
                raise RuntimeError("frozen validation A/B indices must be disjoint and cover validation")
            cal_specs, raw_thresholds, cal_thresholds = [], [], []
            val_cal = np.empty_like(pv)
            for j, endpoint in enumerate(LABELS):
                a, b, opt = fit_platt(pv[A, j], yv[A, j])
                val_cal[:, j] = apply_platt(pv[:, j], a, b)
                t_raw = float(np.quantile(pv[B][yv[B, j] == 0, j], .9, method="linear"))
                t_cal = float(apply_platt(np.asarray([t_raw]), a, b)[0])
                cal_specs.append({"endpoint": endpoint, "a": a, "b": b, "optimizer": opt, "fit_rows": int(len(A)), "fit_positive_count": int(yv[A, j].sum()), "threshold_raw_on_B_negative": t_raw, "threshold_calibrated_transform_of_raw": t_cal})
                raw_thresholds.append(t_raw); cal_thresholds.append(t_cal)
            fit_key = f"{arm}/{seed}"
            fit_record = {
                "fit_split": "A from outputs/mixed_fault_calibration_v1/calibration_split_indices.npz",
                "threshold_split": "B from the same frozen file",
                "A_indices_sha256": hashlib.sha256(A.tobytes()).hexdigest(),
                "B_indices_sha256": hashlib.sha256(B.tobytes()).hexdigest(),
                "validation_order_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
                "A_fit_rows": int(len(A)), "B_threshold_rows": int(len(B)),
                "calibration_fits": cal_specs,
                "fit_formula": "a=exp(theta[0]) > 0; g(p)=expit(a*logit(clip(p,1e-6,1-1e-6))+b)",
                "optimization": {"method": "L-BFGS-B", "initial_theta": [0.0, 0.0], "bounds_theta0": [-12, 12], "bounds_b": [-30, 30], "maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
                "full_validation_used_for_checkpoint_selection_before_A_B_split": True,
            }
            fits[fit_key] = fit_record

            pred_dir = STUDY / "predictions/Ningbo" / arm / f"seed_{seed}"
            with np.load(pred_dir / "clean.npz", allow_pickle=False) as z:
                yc, pc = z["y"].astype(np.uint8), z["p"].astype(np.float64)
            with np.load(pred_dir / "RA_LL_reversal.npz", allow_pickle=False) as z:
                yr, pr = z["y"].astype(np.uint8), z["p"].astype(np.float64)
            if not np.array_equal(yc, yr):
                raise RuntimeError(f"Ningbo clean/RA_LL label order differs: {fit_key}")
            cal_c = np.column_stack([apply_platt(pc[:, j], cal_specs[j]["a"], cal_specs[j]["b"]) for j in range(3)])
            cal_r = np.column_stack([apply_platt(pr[:, j], cal_specs[j]["a"], cal_specs[j]["b"]) for j in range(3)])
            for j, endpoint in enumerate(LABELS):
                t_raw, t_cal = raw_thresholds[j], cal_thresholds[j]
                raw_decisions = np.r_[pc[:, j] >= t_raw, pr[:, j] >= t_raw]
                cal_decisions = np.r_[cal_c[:, j] >= t_cal, cal_r[:, j] >= t_cal]
                mismatch = int(np.count_nonzero(raw_decisions != cal_decisions))
                if mismatch:
                    raise RuntimeError(f"raw/calibrated threshold decisions differ for {fit_key}/{endpoint}: {mismatch}")
                decision_audits.append({
                    "arm": arm, "seed": seed, "endpoint": endpoint,
                    "test_rows_both_conditions": int(len(raw_decisions)),
                    "decision_mismatches": mismatch,
                    "raw_clean_ties_at_threshold": int(np.count_nonzero(pc[:, j] == t_raw)),
                    "raw_fault_ties_at_threshold": int(np.count_nonzero(pr[:, j] == t_raw)),
                    "cal_clean_ties_at_threshold": int(np.count_nonzero(cal_c[:, j] == t_cal)),
                    "cal_fault_ties_at_threshold": int(np.count_nonzero(cal_r[:, j] == t_cal)),
                    "clean_clipped_low": int(np.count_nonzero(pc[:, j] <= 1e-6)),
                    "clean_clipped_high": int(np.count_nonzero(pc[:, j] >= 1 - 1e-6)),
                    "fault_clipped_low": int(np.count_nonzero(pr[:, j] <= 1e-6)),
                    "fault_clipped_high": int(np.count_nonzero(pr[:, j] >= 1 - 1e-6)),
                    "positive_slope_a": cal_specs[j]["a"],
                })
                for q in Q:
                    weights = np.r_[np.full(len(yc), 1 - q), np.full(len(yr), q)]
                    y_mix = np.r_[yc[:, j], yr[:, j]]
                    for mode, cscore, rscore, threshold in (
                        ("raw", pc[:, j], pr[:, j], t_raw),
                        ("calibrated", cal_c[:, j], cal_r[:, j], t_cal),
                    ):
                        scores = np.r_[cscore, rscore]
                        rows.append({"arm": arm, "seed": seed, "endpoint": endpoint, "q": q, "score_mode": mode, **weighted_metrics(y_mix, scores, weights, threshold), "test_positive_prevalence": float(np.average(y_mix, weights=weights)), "calibration_a": cal_specs[j]["a"], "calibration_b": cal_specs[j]["b"], "threshold_source": "validation B negative raw 90th percentile; calibrated threshold transformed from raw", "mixture_source": "weighted pooled clean/RA_LL test observations"})

    new = pd.DataFrame(rows)
    old_keys = ["arm", "seed", "endpoint", "q", "score_mode"]
    compare = new.merge(old, on=old_keys, suffixes=("_recovered", "_v1"), validate="one_to_one")
    # v1 did not emit accuracy for the mixture table; recover it algebraically
    # from its sensitivity/specificity and the same fixed weighted prevalence.
    compare["accuracy_recovered"] = compare["accuracy"]
    compare["accuracy_v1"] = compare["sensitivity_v1"] * compare["test_positive_prevalence"] + compare["specificity_v1"] * (1 - compare["test_positive_prevalence"])
    fields = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "accuracy", "threshold"]
    for field in fields:
        compare[f"delta_{field}"] = compare[f"{field}_recovered"] - compare[f"{field}_v1"]
    compare.to_csv(OUT / "NINGBO_MIXTURE_CHANGES_V1_TO_RECOVERED.csv", index=False, float_format="%.15g")
    new.to_csv(OUT / "NINGBO_MIXTURES_RECOVERED_BY_SEED.csv", index=False, float_format="%.15g")
    (OUT / "NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json").write_text(json.dumps({
        "calibration_split_file": str(SPLIT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "calibration_split_sha256": sha(SPLIT_PATH),
        "new_fits": fits,
        "threshold_decision_audit": decision_audits,
        "all_raw_and_calibrated_decisions_identical": True,
        "decision_check_conditions": ["clean", "RA_LL_reversal"],
        "q_fault": list(Q),
        "test_data_used_for_fit_or_threshold": False,
        "historical_comparator_calibrators_refit": False,
        "full_validation_checkpoint_selection_dependence": "All validation records had already influenced epoch/checkpoint selection. The subsequent A/B split is used for a consistent post-selection calibration procedure but does not make the fit/threshold statistically independent of model selection.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Concise aggregate changes for the final report.
    summary = compare.groupby(["arm", "q", "score_mode"], as_index=False)[[f"delta_{m}" for m in fields if m in ("brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "accuracy", "threshold")]].mean()
    summary.to_csv(OUT / "NINGBO_MIXTURE_CHANGE_SUMMARY.csv", index=False, float_format="%.8g")
    print(json.dumps({"new_mixture_rows": len(new), "calibrators": len(fits), "decision_audits": len(decision_audits), "max_abs_threshold_decision_mismatch": max(x["decision_mismatches"] for x in decision_audits), "old_v1_rows_compared": len(compare)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
