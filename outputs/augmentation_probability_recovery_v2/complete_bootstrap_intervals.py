from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
sys.path.insert(0, str(ROOT))
from outputs.augmentation_probability_study_v1 import paired_bootstrap as ap_source

SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary", "physics_p010", "physics_p025", "physics_aware", "combined")
CONTRASTS = (
    ("physics_p025", "combined"),
    ("physics_p025", "physics_aware"),
    ("physics_p010", "ordinary"),
    ("physics_p010", "physics_aware"),
)
SCALAR_CONDITIONS = ("clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201", "mask5_0_1_2_3_5")


def counts_from_indices(indices: np.ndarray, n: int) -> np.ndarray:
    return np.stack([np.bincount(row, minlength=n) for row in indices]).astype(np.uint16)


def weighted_auc_draws(y: np.ndarray, p: np.ndarray, counts: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    c = counts[:, order].astype(np.float64, copy=False)
    yy = y[order].astype(np.float64)
    pos_by_score = np.add.reduceat(c * yy[None, :], starts, axis=1)
    neg_by_score = np.add.reduceat(c * (1.0 - yy)[None, :], starts, axis=1)
    neg_before = np.cumsum(neg_by_score, axis=1) - neg_by_score
    total_pos = np.sum(counts * y[None, :], axis=1).astype(np.float64)
    total_neg = counts.sum(axis=1).astype(np.float64) - total_pos
    numerator = np.sum(pos_by_score * (neg_before + 0.5 * neg_by_score), axis=1)
    return np.divide(numerator, total_pos * total_neg, out=np.full(len(counts), np.nan), where=(total_pos * total_neg) > 0)


def brier_draws(y: np.ndarray, p: np.ndarray, counts: np.ndarray) -> np.ndarray:
    err = (np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2
    return (counts.astype(np.float64) @ err) / counts.sum(axis=1)


def ap_point_and_draw(cohort: str, arm: str, seed: int, condition: str, indices: np.ndarray, cache: dict) -> tuple[float, np.ndarray]:
    f = STUDY / "bootstrap_draws" / f"{cohort}_{arm}_{seed}_{condition}.npz"
    if not f.is_file():
        raise FileNotFoundError(f"Required frozen v1 AUPRC draws are absent (no recomputation allowed): {f}")
    with np.load(f, allow_pickle=False) as z:
        draws = z["draws"].astype(np.float64)
        if condition == "mean_792_masks":
            point = float(z["point"])
        else:
            y, p = ap_source.arrays(cohort, arm, seed, condition)
            point = float(average_precision_score(y, p))
    if len(draws) != 1000:
        raise RuntimeError(f"Invalid cached AP draw count: {f}")
    cache[(cohort, arm, seed, condition)] = (point, draws)
    return point, draws


def main() -> None:
    gate = json.loads((OUT / "VALIDATION_RESTORATION_AUDIT.json").read_text(encoding="utf-8"))
    if not gate["all_units_gate_reproduced"] or not gate["all_test_subsamples_reproduced"]:
        raise RuntimeError("Cannot bootstrap before validation and test replay pass")
    v1_ap = pd.read_csv(STUDY / "PAIRED_BOOTSTRAP_TWA_AUPRC.csv")
    if len(v1_ap) != 48 or set(v1_ap.metric) != {"AUPRC"}:
        raise RuntimeError("Expected the 48 existing TWA AUPRC comparisons")
    ap_cache: dict = {}
    ap_audits = []
    ap_indices = {}
    for cohort in ("Chapman", "Ningbo"):
        index_path = ROOT / ("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if cohort == "Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy")
        idx = np.load(index_path, allow_pickle=False).astype(np.int32)
        if idx.shape[0] != 1000:
            raise RuntimeError(f"Unexpected pre-existing bootstrap shape: {cohort}: {idx.shape}")
        ap_indices[cohort] = idx
    for row in v1_ap.to_dict("records"):
        a, b = row["contrast"].split("_minus_")
        condition = row["condition"]
        points, draws = [], []
        for seed in SEEDS:
            pa, da = ap_point_and_draw(row["cohort"], a, seed, condition, ap_indices[row["cohort"]], ap_cache)
            pb, db = ap_point_and_draw(row["cohort"], b, seed, condition, ap_indices[row["cohort"]], ap_cache)
            points.append(pa - pb)
            draws.append(da - db)
        diff_draws = np.mean(np.stack(draws), axis=0)
        checks = {
            "point_abs_error": abs(float(np.mean(points)) - float(row["full_cohort_point_difference"])),
            "bootstrap_mean_abs_error": abs(float(diff_draws.mean()) - float(row["bootstrap_mean"])),
            "ci_low_abs_error": abs(float(np.quantile(diff_draws, .025)) - float(row["ci95_low"])),
            "ci_high_abs_error": abs(float(np.quantile(diff_draws, .975)) - float(row["ci95_high"])),
        }
        ap_audits.append({"cohort": row["cohort"], "contrast": row["contrast"], "condition": condition, **checks})
    max_ap_error = max(max(x[k] for x in ap_audits) for k in ("point_abs_error", "bootstrap_mean_abs_error", "ci_low_abs_error", "ci_high_abs_error"))
    if max_ap_error > 1e-6:
        raise RuntimeError(f"Cached AUPRC intervals do not reproduce the saved 48 rows: max error {max_ap_error}")
    v1_ap.to_csv(OUT / "PAIRED_BOOTSTRAP_TWA_AUPRC_PRESERVED.csv", index=False)

    index_counts, index_meta = {}, {}
    for cohort in ("Chapman", "Ningbo"):
        path = ROOT / ("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if cohort == "Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy")
        idx = ap_indices[cohort]
        n = idx.shape[1]
        index_counts[cohort] = counts_from_indices(idx, n)
        index_meta[cohort] = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "shape": list(idx.shape), "sample_size": n}
        if np.min(idx) < 0 or np.max(idx) >= n:
            raise RuntimeError(f"Bootstrap index out of bounds: {cohort}")

    # Audit optimized tied-rank AUC against sklearn on a fixed subset of real
    # predictions and the first 20 already-frozen cohort resamples.
    auc_algorithm_audits = []
    for cohort in ("Chapman", "Ningbo"):
        y, p = ap_source.arrays(cohort, "physics_p010", SEEDS[0], "clean")
        d = weighted_auc_draws(y, p, index_counts[cohort][:20])
        sklearn_d = np.asarray([roc_auc_score(y[index], p[index]) for index in ap_indices[cohort][:20]])
        b = brier_draws(y, p, index_counts[cohort][:20])
        sklearn_b = np.asarray([np.mean((p[index] - y[index]) ** 2) for index in ap_indices[cohort][:20]])
        err_auc = float(np.nanmax(np.abs(d - sklearn_d)))
        err_brier = float(np.max(np.abs(b - sklearn_b)))
        auc_algorithm_audits.append({"cohort": cohort, "real_model_condition": "physics_p010 seed 104729 clean TWA", "replicates": 20, "auc_max_abs_error_vs_sklearn": err_auc, "brier_max_abs_error_vs_direct": err_brier})
        if err_auc > 1e-12 or err_brier > 1e-12:
            raise RuntimeError(f"Bootstrap draw implementation audit failed for {cohort}: {err_auc=} {err_brier=}")

    draw_cache = OUT / "bootstrap_draws_auroc_brier"
    draw_cache.mkdir(exist_ok=True)
    model_points: dict[tuple[str, str, int, str, str], float] = {}
    model_draws: dict[tuple[str, str, int, str, str], np.ndarray] = {}
    metrics_by_key = {}
    full_point_v1_max_error = {"auroc": 0.0, "brier": 0.0}
    v1_all = pd.read_csv(STUDY / "METRICS_ALL_ARMS_BY_SEED.csv")
    # Use the existing full-row table to bind every full-cohort point to v1.
    v1_point = {}
    for r in v1_all[v1_all.endpoint == "TWA"].to_dict("records"):
        for metric in ("auprc", "auroc", "brier"):
            v1_point[(r["cohort"], r["arm"], int(r["seed"]), r["condition"], metric)] = float(r[metric])

    for cohort in ("Chapman", "Ningbo"):
        indices = ap_indices[cohort]
        counts = index_counts[cohort]
        for arm in ARMS:
            source_arm = "physics_aware" if arm == "physics_aware" else arm
            for seed in SEEDS:
                for condition in SCALAR_CONDITIONS:
                    y, p = ap_source.arrays(cohort, source_arm, seed, condition)
                    if len(y) != counts.shape[1]:
                        raise RuntimeError(f"test N and frozen bootstrap index sample size differ: {cohort}/{arm}/{condition}")
                    condition_cache = {}
                    for metric in ("auroc", "brier"):
                        if metric == "auroc":
                            point = float(roc_auc_score(y, p))
                            draws = weighted_auc_draws(y, p, counts)
                        else:
                            point = float(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2))
                            draws = brier_draws(y, p, counts)
                        expected = v1_point[(cohort, source_arm, seed, condition, metric)]
                        err = abs(point - expected)
                        full_point_v1_max_error[metric] = max(full_point_v1_max_error[metric], err)
                        if err > 1e-7:
                            raise RuntimeError(f"full-cohort raw {metric} differs from v1: {cohort}/{arm}/{seed}/{condition}: {point} vs {expected}")
                        f = draw_cache / f"{cohort}_{arm}_{seed}_{condition}_{metric}.npz"
                        np.savez_compressed(f, draws=draws.astype(np.float64), point=np.asarray(point))
                        model_points[(cohort, arm, seed, condition, metric)] = point
                        model_draws[(cohort, arm, seed, condition, metric)] = draws.astype(np.float64)
                        condition_cache[metric] = point
                    metrics_by_key[(cohort, arm, seed, condition)] = condition_cache

    out_rows = []
    for cohort in ("Chapman", "Ningbo"):
        for arm_a, arm_b in CONTRASTS:
            for condition in SCALAR_CONDITIONS:
                for metric in ("auroc", "brier"):
                    seed_points, seed_draw_diffs = [], []
                    for seed in SEEDS:
                        p_a = model_points[(cohort, arm_a, seed, condition, metric)]
                        p_b = model_points[(cohort, arm_b, seed, condition, metric)]
                        seed_points.append(p_a - p_b)
                        seed_draw_diffs.append(model_draws[(cohort, arm_a, seed, condition, metric)] - model_draws[(cohort, arm_b, seed, condition, metric)])
                    replicate_diffs = np.mean(np.stack(seed_draw_diffs), axis=0)
                    out_rows.append({
                        "cohort": cohort, "contrast": f"{arm_a}_minus_{arm_b}", "arm_a": arm_a, "arm_b": arm_b,
                        "condition": condition, "endpoint": "TWA", "metric": "AUROC" if metric == "auroc" else "raw Brier",
                        "full_cohort_point_difference": float(np.mean(seed_points)),
                        "bootstrap_mean": float(np.mean(replicate_diffs)),
                        "ci95_low": float(np.quantile(replicate_diffs, .025)),
                        "ci95_high": float(np.quantile(replicate_diffs, .975)),
                        "replicates": 1000, "paired_synchronized": True,
                        "seeds_averaged_within_replicate": True,
                        "interpretation": "negative Brier difference favors arm_a (lower Brier)" if metric == "brier" else "higher AUROC favors arm_a",
                    })
    if len(out_rows) != 80:
        raise RuntimeError(f"Expected 80 new bootstrap groups, got {len(out_rows)}")
    pd.DataFrame(out_rows).to_csv(OUT / "PAIRED_BOOTSTRAP_TWA_AUROC_BRIER.csv", index=False, float_format="%.15g")
    audit = {
        "preserved_existing_TWA_AUPRC_rows": len(v1_ap),
        "cached_AP_draw_verification_rows": len(ap_audits),
        "cached_AP_draw_max_abs_summary_error": max_ap_error,
        "cached_792_mask_AP_draw_files_reused": sum("mean_792_masks" in key[3] for key in ap_cache),
        "full_792_mask_AP_bootstrap_recomputed": False,
        "new_AUROC_Brier_groups": len(out_rows),
        "groups_expected": 80,
        "full_cohort_point_max_abs_difference_vs_v1": full_point_v1_max_error,
        "full_cohort_point_source": "recomputed from the exact frozen prediction probabilities; small raw Brier differences up to 1e-7 are allowed for float32/float64 reduction precision",
        "bootstrap_indices": index_meta,
        "auc_brier_implementation_audits": auc_algorithm_audits,
        "Brier_sign_convention": "negative first-minus-second difference means lower Brier and favors first arm",
    }
    (OUT / "BOOTSTRAP_RECOVERY_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"preserved_AP_rows": len(v1_ap), "AP_draw_max_error": max_ap_error, "new_groups": len(out_rows), "auc_brier_audit_max": max(max(x["auc_max_abs_error_vs_sklearn"], x["brier_max_abs_error_vs_direct"]) for x in auc_algorithm_audits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
