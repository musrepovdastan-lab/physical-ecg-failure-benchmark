"""Complete only the missing paired bootstrap groups from frozen predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "outputs" / "calibration_role_control_v1"
OUT = ROOT / "outputs" / "calibration_role_control_completion_v2"
COHORTS = ("Ningbo_test", "HEEDB")
METHODS = ("ordinary", "physics_aware_p050", "combined", "Huang_full_pipeline")
SEEDS = (104729, 130363, 155921)
# Canonical order in frozen control.py: ('clean', *FAULTS).
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
SCENARIOS = ("RA_LL_only", "equal_three_swap")
Q_ALL = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
Q_NEW = (0.0, 0.001, 0.005, 0.05)
CONTRASTS = (
    ("physics_aware_p050", "ordinary"),
    ("combined", "ordinary"),
    ("Huang_full_pipeline", "ordinary"),
    ("physics_aware_p050", "Huang_full_pipeline"),
    ("combined", "Huang_full_pipeline"),
)
METRICS = ("AP_raw", "Brier_raw", "Brier_calibrated", "sensitivity_raw", "FP_per_1000_raw")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = a * (np.log(p / (1 - p))) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def tie_group_ap(group_desc: np.ndarray, yy: np.ndarray, weight: np.ndarray, n_groups: int) -> float:
    # Zero-weight groups add neither TP nor total; cumulative precision stays exact.
    positive = np.bincount(group_desc, weights=weight * yy, minlength=n_groups).astype(np.float64)
    total = np.bincount(group_desc, weights=weight, minlength=n_groups).astype(np.float64)
    positives = positive.sum()
    if positives <= 0:
        return float("nan")
    tp = np.cumsum(positive)
    seen = np.cumsum(total)
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    return float(np.sum((positive / positives) * precision))


def scenario_weights(scenario: str, q: float) -> np.ndarray:
    if scenario == "RA_LL_only":
        return np.asarray([1 - q, 0, q, 0], np.float64)
    return np.asarray([1 - q, q / 3, q / 3, q / 3], np.float64)


def main() -> None:
    frozen_bindings = json.loads((OUT / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    source_changes = []
    for item in frozen_bindings["files"]:
        path = ROOT / Path(item["path"])
        if not path.exists() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            source_changes.append(item["path"])
    if source_changes:
        raise RuntimeError(f"frozen source changed before calculation: {source_changes[:5]}")

    # Verify accepted 300-group coverage and preserve its vectors verbatim.
    old_rows = read_csv(V1 / "TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv")
    old_draws: dict[str, dict[str, np.ndarray]] = {}
    for cohort in COHORTS:
        with np.load(V1 / f"{cohort}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz", allow_pickle=False) as archive:
            old_draws[cohort] = {key: archive[key].copy() for key in archive.files}
    old_expected = {
        f"{cohort}|{scenario}|q{q:g}|{left}-{right}|{metric}"
        for cohort in COHORTS
        for scenario in SCENARIOS
        for q in (0.01, 0.1, 1.0)
        for left, right in CONTRASTS
        for metric in METRICS
    }
    old_keys = {key for cohort in COHORTS for key in old_draws[cohort]}
    if len(old_rows) != 300 or old_keys != old_expected or any(v.shape != (1000,) for d in old_draws.values() for v in d.values()):
        raise RuntimeError("existing 300 bootstrap groups failed scope/geometry validation")
    old_row_map = {
        (r["cohort"], r["scenario"], float(r["q_fault"]), r["left"], r["right"], r["metric"]): r
        for r in old_rows
    }
    for cohort in COHORTS:
        for key, values in old_draws[cohort].items():
            _, scenario, qstr, contrast, metric = key.split("|", 4)
            q = float(qstr[1:])
            left, right = contrast.rsplit("-", 1)
            row = old_row_map[(cohort, scenario, q, left, right, metric)]
            checks = (np.mean(values), np.quantile(values, 0.025), np.quantile(values, 0.975))
            saved = tuple(float(row[k]) for k in ("bootstrap_mean", "ci95_low", "ci95_high"))
            if max(abs(a - b) for a, b in zip(checks, saved)) > 1e-12:
                raise RuntimeError(f"existing interval summary does not match draws: {cohort}|{key}")

    # Load prediction payloads only; this performs no model construction or inference.
    predictions: dict[tuple[str, str, int, str], np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        for method in METHODS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    path = V1 / "predictions" / cohort / method / f"seed_{seed}" / f"{condition}.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        p = archive["p"].astype(np.float64)
                        y = archive["y"][:, 0].astype(np.uint8)
                        rid = archive["ids"].astype(str)
                        if p.shape[1] != 3 or not np.isfinite(p).all():
                            raise RuntimeError(f"invalid prediction payload {path}")
                        if cohort in labels and (not np.array_equal(y, labels[cohort]) or not np.array_equal(rid, ids[cohort])):
                            raise RuntimeError(f"prediction label/order binding mismatch {path}")
                        labels[cohort], ids[cohort] = y, rid
                        predictions[cohort, method, seed, condition] = p[:, 0]

    # Calibration locks are per method/seed/endpoint and were fitted on calibration_A.
    lock = read_csv(V1 / "CALIBRATION_AND_THRESHOLD_LOCK.csv")
    cal = {}
    for row in lock:
        if row["endpoint"] == "TWA":
            cal[row["method"], int(row["seed"])] = (
                float(row["slope_a"]), float(row["intercept_b"]), float(row["raw_q90"])
            )
    if len(cal) != 12:
        raise RuntimeError("expected 12 TWA calibration/threshold bindings")

    mixture = read_csv(V1 / "MIXTURE_METRICS_BY_SEED.csv")
    mix_map = {
        (r["cohort"], r["method"], int(r["seed"]), r["endpoint"], r["scenario"], float(r["q_fault"])): r
        for r in mixture
    }
    # Recompute the complete set of TWA point metrics from the locked predictions.
    point_check_rows = []
    for cohort in COHORTS:
        yy0 = labels[cohort]
        yy = np.tile(yy0, 4)
        for scenario in SCENARIOS:
            for q in Q_ALL:
                w_cond = scenario_weights(scenario, q)
                weights = np.repeat(w_cond, len(yy0))
                for method in METHODS:
                    for seed in SEEDS:
                        pp = np.concatenate([predictions[cohort, method, seed, condition] for condition in CONDITIONS])
                        a, b, threshold = cal[method, seed]
                        cp = platt(pp, a, b)
                        pos = yy == 1
                        neg = ~pos
                        dec = pp >= threshold
                        total = weights.sum()
                        vals = {
                            "AP_raw": float(average_precision_score(yy, pp, sample_weight=weights)),
                            "AUROC_raw": _weighted_auc(yy, pp, weights),
                            "Brier_raw": float(np.sum(weights * (pp - yy) ** 2) / total),
                            "Brier_calibrated": float(np.sum(weights * (cp - yy) ** 2) / total),
                            "sensitivity_raw": float(np.sum(weights * dec * pos) / np.sum(weights * pos)),
                            "FP_per_1000_raw": float(1000 * np.sum(weights * dec * neg) / total),
                        }
                        saved = mix_map[(cohort, method, seed, "TWA", scenario, q)]
                        for name, value in vals.items():
                            delta = abs(value - float(saved[name]))
                            point_check_rows.append({"cohort": cohort, "scenario": scenario, "q_fault": q,
                                                     "method": method, "seed": seed, "metric": name,
                                                     "recalculated": value, "saved": float(saved[name]), "abs_delta": delta})

    # Precompute tied-score rank buckets once for the four conditions combined.
    rank_cache = {}
    yy = {cohort: np.tile(labels[cohort], 4) for cohort in COHORTS}
    for cohort in COHORTS:
        for method in METHODS:
            for seed in SEEDS:
                pp = np.concatenate([predictions[cohort, method, seed, condition] for condition in CONDITIONS])
                _, inverse = np.unique(pp, return_inverse=True)
                rank_cache[cohort, method, seed] = (pp, len(np.unique(inverse)) - 1 - inverse,
                                                     len(np.unique(inverse)), platt(pp, *cal[method, seed][:2]),
                                                     pp >= cal[method, seed][2])

    new_draws: dict[str, dict[str, np.ndarray]] = {cohort: {} for cohort in COHORTS}
    sklearn_checks = []
    for cohort, idx_path in (
        ("Ningbo_test", ROOT / "outputs/ningbo_confirmation_v2/bootstrap_indices.npy"),
        ("HEEDB", ROOT / "outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy"),
    ):
        idx = np.load(idx_path, mmap_mode="r", allow_pickle=False)
        n = len(labels[cohort])
        if idx.shape != (1000, n):
            raise RuntimeError(f"bootstrap index geometry mismatch {cohort}: {idx.shape}")
        for scenario in SCENARIOS:
            for q in Q_NEW:
                wc = scenario_weights(scenario, q)
                by_method = {m: np.full((1000, len(SEEDS), len(METRICS)), np.nan, np.float64) for m in METHODS}
                for draw_i, sample in enumerate(idx):
                    counts = np.bincount(np.asarray(sample, dtype=np.int64), minlength=n).astype(np.float64)
                    multiplicity = np.concatenate([counts * w for w in wc])
                    for si, seed in enumerate(SEEDS):
                        for method in METHODS:
                            pp, group_desc, ng, cp, dec = rank_cache[cohort, method, seed]
                            y = yy[cohort]
                            positive = y == 1
                            negative = ~positive
                            w_sum = multiplicity.sum()
                            vals = (
                                tie_group_ap(group_desc, y, multiplicity, ng),
                                float(np.sum(multiplicity * (pp - y) ** 2) / w_sum),
                                float(np.sum(multiplicity * (cp - y) ** 2) / w_sum),
                                float(np.sum(multiplicity * dec * positive) / np.sum(multiplicity * positive)),
                                float(1000 * np.sum(multiplicity * dec * negative) / w_sum),
                            )
                            by_method[method][draw_i, si] = vals
                            if draw_i in (0, 417, 999):
                                ap_sklearn = float(average_precision_score(y, pp, sample_weight=multiplicity))
                                sklearn_checks.append({"cohort": cohort, "scenario": scenario, "q_fault": q,
                                                       "draw": draw_i, "method": method, "seed": seed,
                                                       "ap_tie_group": vals[0], "ap_sklearn_weighted": ap_sklearn,
                                                       "absolute_delta": abs(vals[0] - ap_sklearn),
                                                       "zero_weight_rows": int(np.count_nonzero(multiplicity == 0))})
                for left, right in CONTRASTS:
                    for mi, metric in enumerate(METRICS):
                        values = np.mean(by_method[left][:, :, mi] - by_method[right][:, :, mi], axis=1)
                        key = f"{cohort}|{scenario}|q{q:g}|{left}-{right}|{metric}"
                        new_draws[cohort][key] = values

    # Preserve old vectors and combine with exactly the 400 missing groups.
    combined_draws = {}
    for cohort in COHORTS:
        combined_draws.update(old_draws[cohort])
        combined_draws.update(new_draws[cohort])
    expected_all = {
        f"{cohort}|{scenario}|q{q:g}|{left}-{right}|{metric}"
        for cohort in COHORTS for scenario in SCENARIOS for q in Q_ALL
        for left, right in CONTRASTS for metric in METRICS
    }
    if set(combined_draws) != expected_all or len(combined_draws) != 700:
        raise RuntimeError(f"700-group draw coverage mismatch: groups={len(combined_draws)}")
    if max(x["absolute_delta"] for x in sklearn_checks) > 1e-12:
        raise RuntimeError("new tie-group AP failed actual sklearn weighted checks")

    point_lookup = {}
    for r in point_check_rows:
        point_lookup[(r["cohort"], r["scenario"], float(r["q_fault"]), r["method"], int(r["seed"]), r["metric"])] = r["recalculated"]
    full_rows = []
    group_audit = []
    old_key_set = {key for c in COHORTS for key in old_draws[c]}
    for cohort in COHORTS:
        for scenario in SCENARIOS:
            for q in Q_ALL:
                for left, right in CONTRASTS:
                    for metric in METRICS:
                        key = f"{cohort}|{scenario}|q{q:g}|{left}-{right}|{metric}"
                        values = combined_draws[key]
                        point = float(np.mean([
                            point_lookup[(cohort, scenario, q, left, seed, metric)] -
                            point_lookup[(cohort, scenario, q, right, seed, metric)] for seed in SEEDS
                        ]))
                        old = old_row_map.get((cohort, scenario, q, left, right, metric))
                        if key in old_key_set:
                            # Reuse the original row fields and leave its values untouched.
                            row = dict(old)
                            if abs(point - float(row["full_cohort_point_mean_seed_contrast"])) > 1e-12:
                                raise RuntimeError(f"saved point contrast differs from recomputed points: {key}")
                        else:
                            row = {"cohort": cohort, "scenario": scenario, "q_fault": q,
                                   "left": left, "right": right, "metric": metric,
                                   "full_cohort_point_mean_seed_contrast": point,
                                   "bootstrap_mean": float(np.mean(values)),
                                   "ci95_low": float(np.quantile(values, 0.025)),
                                   "ci95_high": float(np.quantile(values, 0.975)),
                                   "n_draws": 1000,
                                   "unit": "record" if cohort == "Ningbo_test" else "patient",
                                   "conditional_on_fitted_models_and_calibrators": True}
                        full_rows.append(row)
                        group_audit.append({"cohort": cohort, "scenario": scenario, "q_fault": q,
                                            "left": left, "right": right, "metric": metric,
                                            "group_key": key, "origin": "reused_v1" if key in old_key_set else "new_v2",
                                            "draw_count": int(len(values)), "all_finite": bool(np.isfinite(values).all())})

    # q=0 scenarios use identical weights; ensure both rows are present and numerically equal.
    q0_deltas = []
    for cohort in COHORTS:
        for q0_key, q0_value in new_draws[cohort].items():
            if "|q0|" not in q0_key:
                continue
            other = q0_key.replace("|RA_LL_only|", "|equal_three_swap|") if "|RA_LL_only|" in q0_key else q0_key.replace("|equal_three_swap|", "|RA_LL_only|")
            q0_deltas.append(float(np.max(np.abs(q0_value - new_draws[cohort][other]))))
    if len(q0_deltas) != 100 or max(q0_deltas) > 1e-14:
        raise RuntimeError(f"q=0 coverage/equality check failed: {len(q0_deltas)} groups")

    # Persist full paired tables and compact documented draw archive.
    write_csv(OUT / "PAIRED_POINTS_CI_ALL_700.csv", full_rows)
    write_csv(OUT / "BOOTSTRAP_COVERAGE_700.csv", group_audit)
    write_csv(OUT / "NEW_GROUP_SKLEARN_CHECKS.csv", sklearn_checks)
    write_csv(OUT / "TWA_POINT_RECALCULATION.csv", point_check_rows)
    np.savez_compressed(OUT / "PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz", **combined_draws)
    for cohort in COHORTS:
        per_cohort = {k: v for k, v in combined_draws.items() if k.startswith(cohort + "|")}
        np.savez_compressed(OUT / f"{cohort}_PAIRED_DRAWS_700.npz", **per_cohort)

    # These hashes demonstrate that training, fit, thresholds and prediction inputs were not altered.
    post_changes = []
    for item in frozen_bindings["files"]:
        path = ROOT / Path(item["path"])
        if not path.exists() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            post_changes.append(item["path"])
    if post_changes:
        raise RuntimeError(f"frozen sources changed during analysis: {post_changes[:5]}")

    old_delta = 0.0
    for cohort in COHORTS:
        for key, values in old_draws[cohort].items():
            old_delta = max(old_delta, float(np.max(np.abs(values - combined_draws[key]))))
    point_max = max(r["abs_delta"] for r in point_check_rows)
    sklearn_max = max(r["absolute_delta"] for r in sklearn_checks)
    audit = {
        "status": "PASS",
        "metric_scope": "TWA endpoint; all q/scenario/method/seed point rows recalculated from saved predictions",
        "twa_point_rows_recalculated": len(point_check_rows),
        "twa_point_max_abs_delta_vs_saved_mixture_metrics": point_max,
        "old_group_count": len(old_key_set), "old_draw_vectors_preserved_max_abs_delta": old_delta,
        "new_group_count": sum(x["origin"] == "new_v2" for x in group_audit),
        "total_group_count": len(group_audit), "draws_per_group": 1000,
        "new_sklearn_checks": len(sklearn_checks), "new_sklearn_max_abs_delta": sklearn_max,
        "q0_scenario_pair_groups_checked": len(q0_deltas), "q0_max_abs_delta": max(q0_deltas),
        "all_full_draws_finite": all(x["all_finite"] for x in group_audit),
        "frozen_source_changes": post_changes,
        "no_training_or_inference": True,
        "no_calibration_refit_or_threshold_selection": True,
    }
    (OUT / "NUMERICAL_VERIFICATION.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = {
        "status": "PASS", "cohorts": list(COHORTS), "scenarios": list(SCENARIOS),
        "q_fault": list(Q_ALL), "contrasts": [f"{a}-{b}" for a, b in CONTRASTS], "metrics": list(METRICS),
        "expected_group_count": 700, "observed_group_count": len(group_audit),
        "reused_v1_groups": len(old_key_set), "new_v2_groups": sum(x["origin"] == "new_v2" for x in group_audit),
        "draws_per_group": 1000, "expected_draws": 700000,
        "groups_by_q": {str(q): 100 for q in Q_ALL},
        "new_groups_by_q": {str(q): 100 for q in Q_NEW},
        "q0_scenarios_equal_weights_but_both_covered": True,
        "unit": {"Ningbo_test": "record", "HEEDB": "patient"},
        "group_manifest": "BOOTSTRAP_COVERAGE_700.csv",
        "draw_archive_schema": {"format": "NPZ", "array_key": "cohort|scenario|q{q:g}|left-right|metric", "shape": [1000], "dtype": "float64", "draw_index": "same frozen cohort bootstrap index row across views/methods/seeds"},
    }
    (OUT / "BOOTSTRAP_COVERAGE_AUDIT.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "groups": len(group_audit), "new": coverage["new_v2_groups"],
                      "old": coverage["reused_v1_groups"], "sklearn_checks": len(sklearn_checks),
                      "point_rows": len(point_check_rows), "point_max_delta": point_max,
                      "sklearn_max_delta": sklearn_max}, indent=2))


def _weighted_auc(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> float:
    """Weighted Mann-Whitney AUROC with average tie credit."""
    u, inv = np.unique(p, return_inverse=True)
    pos = np.bincount(inv, weights=weight * y, minlength=len(u))
    total = np.bincount(inv, weights=weight, minlength=len(u))
    neg = total - pos
    neg_below = np.cumsum(neg) - neg
    concordant = np.sum(pos * (neg_below + 0.5 * neg))
    ptotal, ntotal = pos.sum(), neg.sum()
    return float(concordant / (ptotal * ntotal))


if __name__ == "__main__":
    main()
