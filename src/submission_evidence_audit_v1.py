"""Read-only independent arithmetic/integrity audit for the submission package.

Does not infer, train, access S3, or write patient-level or public files.
Only aggregate verification facts are printed. Raises on the first discrepancy.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006"
SEEDS = (104729, 130363, 155921)
METRICS = ("auprc", "auroc", "sensitivity", "specificity", "predicted_positive_fraction", "mean_score", "brier", "ece15")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().lower()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, context: str, tol: float = 2e-6) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"audit discrepancy: {context}: observed={actual}, reported={expected}")


def interval(values: np.ndarray, reported: dict, context: str, tol: float = 2e-6) -> None:
    reported_mean = next((reported[key] for key in ("mean", "bootstrap_mean", "mean_contrast") if key in reported), None)
    if reported_mean is None:
        raise RuntimeError(f"audit verifier missing reported mean for {context}")
    close(float(np.mean(values)), float(reported_mean), context + " mean", tol)
    close(float(np.quantile(values, .025)), float(reported.get("low95", reported.get("lower95"))), context + " low95", tol)
    close(float(np.quantile(values, .975)), float(reported.get("high95", reported.get("upper95"))), context + " high95", tol)


def protocols_and_cohorts() -> dict:
    bindings = {
        "Chapman baseline": ("configs/FROZEN_CHAPMAN_MULTISEED_PROTOCOL_v7.json", "312dabb9a0400675097d492f66df58c9da5ead69623fff0563f4daece0944cb7"),
        "Chapman M1": ("configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json", "248976f691003daa45c5f188c53f7cf75891f19f8723223477fc41e1d65667ce"),
        "Ningbo baseline": ("configs/FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json", "1fe829e4357b4efde0e5f227baf5bf94e8d22d12af0959a38cee12cce1f40d71"),
        "Ningbo M2": ("configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json", "2d7b605af175ef6d5b59b223a9ea37bb3f7d675c2c1ac3e27fff23fcd43e96b8"),
        "HEEDB independent": ("configs/FROZEN_HEEDB_INDEPENDENT_CONFIRMATION_v1.json", "199344e9f7cbb0eb55564f52ce320bca4c85478ceb68174a3d11568611a58760"),
        "HEEDB Huang": ("outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json", "50974d7cdb7a286d7d3cd4831b29df2a80edaac31a274afd697d931bb76cd53c"),
    }
    for name, (relative, expected) in bindings.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"audit discrepancy: {name} protocol hash")
    independent = read_json(OUT / "heedb_independent_confirmation_v1" / "RUN_PROVENANCE.json")
    huang = read_json(OUT / "heedb_huang_exploratory_v1" / "ANALYSIS_COMPLETE.json")
    for relative, expected in independent["aggregate_output_sha256"].items():
        if sha(OUT / "heedb_independent_confirmation_v1" / relative) != expected.lower():
            raise RuntimeError(f"audit discrepancy: independent aggregate hash {relative}")
    for relative, expected in huang["aggregate_hashes"].items():
        if sha(OUT / "heedb_huang_exploratory_v1" / relative) != expected.lower():
            raise RuntimeError(f"audit discrepancy: Huang aggregate hash {relative}")
    for path, expected in (
        (PRIVATE / "independent_confirmation_v1" / "selection_initial.csv", independent["selected_manifest_sha256"]),
        (PRIVATE / "independent_confirmation_v1" / "signals_10000_float32.npy", independent["waveform_cache_sha256"]),
        (PRIVATE / "independent_confirmation_v1" / "predictions_float32.npy", independent["prediction_artifact_sha256"]),
        (PRIVATE / "huang_exploratory_v1" / "detector_probabilities_float32.npy", huang["detector_prediction_sha256"]),
        (PRIVATE / "huang_exploratory_v1" / "huang_diagnostic_probabilities_float32.npy", huang["huang_diagnosis_prediction_sha256"]),
    ):
        if sha(path) != expected.lower():
            raise RuntimeError(f"audit discrepancy: restricted artifact hash {path.name}")
    old = pd.read_csv(PRIVATE / "external_v1" / "selection_final.csv", usecols=["BDSPPatientID"], dtype=str, keep_default_na=False).BDSPPatientID
    new = pd.read_csv(PRIVATE / "independent_confirmation_v1" / "selection_initial.csv", usecols=["BDSPPatientID"], dtype=str, keep_default_na=False).BDSPPatientID
    ready = pd.read_csv(PRIVATE / "audit_manifests" / "waveform_sample_candidates.csv", usecols=["BDSPPatientID"], dtype=str, keep_default_na=False).BDSPPatientID
    if (len(old), old.nunique(), len(new), new.nunique(), ready.nunique()) != (10000, 10000, 10000, 10000, 23):
        raise RuntimeError("audit discrepancy: HEEDB cohort cardinalities")
    if set(old) & set(new) or set(old) & set(ready) or set(new) & set(ready):
        raise RuntimeError("audit discrepancy: HEEDB cohort overlap")
    return {"protocol_hashes": {name: expected for name, (_, expected) in bindings.items()},
            "heedb_original_patients": 10000, "heedb_independent_patients": 10000,
            "readiness_excluded_patients": 23, "pairwise_patient_overlap": 0,
            "independent_aggregate_hashes_checked": len(independent["aggregate_output_sha256"]),
            "huang_aggregate_hashes_checked": len(huang["aggregate_hashes"])}


def saved_state_contrasts() -> dict:
    files = {
        "Chapman": OUT / "phase2_chapman_multiseed_v7",
        "Ningbo": OUT / "ningbo_confirmation_v2",
    }
    observations = {}
    for dataset, directory in files.items():
        summary = pd.read_csv(directory / ("MULTISEED_BOOTSTRAP_SUMMARY.csv" if dataset == "Chapman" else "BOOTSTRAP_SUMMARY.csv"))
        with np.load(directory / "bootstrap_state.npz", allow_pickle=False) as state:
            if state["done"].shape != (1000,) or not state["done"].all() or state["contrasts"].shape != (1000, 3, 3):
                raise RuntimeError(f"audit discrepancy: {dataset} bootstrap state completeness")
            for label_index, label in enumerate(("TWA", "SB", "STach")):
                row = summary.loc[summary.label == label].iloc[0].to_dict()
                values = state["contrasts"][:, :, label_index].mean(axis=1)
                interval(values, row, dataset + " " + label)
            low_key, high_key = (("low95", "high95") if dataset == "Chapman" else ("lower95", "upper95"))
            observations[dataset + "_TWA_mask_minus_RA_LL_CI"] = [float(x) for x in summary.loc[summary.label == "TWA", [low_key, high_key]].iloc[0]]
    for dataset, directory, arm_key in (
        ("Chapman", OUT / "mitigation_m1_v1", "arm_contrasts"),
        ("Ningbo", OUT / "mitigation_m2_ningbo_v1", "m2_contrasts"),
    ):
        summary = pd.read_csv(directory / "BOOTSTRAP_SUMMARY.csv")
        with np.load(directory / "bootstrap_state.npz", allow_pickle=False) as state:
            if state["done"].shape != (1000,) or not state["done"].all():
                raise RuntimeError(f"audit discrepancy: {dataset} mitigation bootstrap completeness")
            for label_index, label in enumerate(("TWA", "SB", "STach")):
                base = state["baseline_contrasts"][:, :, label_index].mean(axis=1)
                protected = (state[arm_key][:, 0, :, label_index] if dataset == "Chapman" else state[arm_key][:, :, label_index]).mean(axis=1)
                subset = summary.loc[summary.endpoint == label]
                key = "matched_contrast" if dataset == "Chapman" else "m2_matched_contrast"
                row = subset.loc[subset.metric == key].iloc[0].to_dict()
                interval(protected, row, dataset + " physics " + label)
                reduction = subset.loc[subset.metric == "contrast_reduction_vs_baseline"].iloc[0].to_dict()
                interval(base - protected, reduction, dataset + " reduction " + label)
            observations[dataset + "_TWA_mitigation_reduction_CI"] = [float(x) for x in summary.loc[(summary.endpoint == "TWA") & (summary.metric == "contrast_reduction_vs_baseline"), ["low95", "high95"]].iloc[0]]
    return observations


def chapman_ningbo_figure_data() -> dict:
    from src.chapman_phase1 import labels as chapman_labels

    chapman_split = read_json(OUT / "phase1_chapman_seed1_v4" / "SPLIT_MEMBERSHIP_v4.json")["rows"]
    chapman_ids = [r["record_id"] for r in chapman_split if r["split"] == "test"]
    yc = np.stack([chapman_labels(record) for record in chapman_ids]).astype(np.uint8)
    ningbo_records = {r["record_id"]: r for r in read_json(OUT / "ningbo_confirmation_v2" / "CANONICAL_RECORDS.json")["records"]}
    ningbo_split = read_json(OUT / "ningbo_confirmation_v2" / "SPLIT_MEMBERSHIP.json")["rows"]
    yn = np.asarray([ningbo_records[r["record_id"]]["labels"] for r in ningbo_split if r["split"] == "test"], dtype=np.uint8)
    if yc.shape != (2050, 3) or yn.shape != (6981, 3):
        raise RuntimeError("audit discrepancy: Chapman/Ningbo test-label geometry")
    output = {}
    cross = pd.read_csv(OUT / "paper_readiness_v1" / "CROSS_DATASET_RESULTS.csv")
    post = pd.read_csv(OUT / "post_m2_paper_readiness_v2" / "CROSS_DATASET_PRIMARY_RESULTS.csv")
    for dataset, y, baseline_dir, mitigation_dir, baseline_csv, mitigation_csv in (
        ("Chapman", yc, OUT / "phase2_chapman_multiseed_v7", OUT / "mitigation_m1_v1", "CONDITION_RESULTS_BY_SEED.csv", "RESULTS_BY_ARM_SEED.csv"),
        ("Ningbo", yn, OUT / "ningbo_confirmation_v2", OUT / "mitigation_m2_ningbo_v1", "CONDITION_RESULTS_BY_SEED.csv", "RESULTS_BY_SEED.csv"),
    ):
        base = pd.read_csv(baseline_dir / baseline_csv)
        miti = pd.read_csv(mitigation_dir / mitigation_csv)
        if int(y[:, 0].sum()) != (377 if dataset == "Chapman" else 1034):
            raise RuntimeError(f"audit discrepancy: {dataset} TWA prevalence")
        for seed in SEEDS:
            b = base.loc[(base.seed == seed) & (base.label == "TWA")]
            m = miti.loc[(miti.seed == seed) & (miti.endpoint == "TWA") &
                         (miti.arm == ("physics_aware" if dataset == "Chapman" else "physics_aware_m2"))]
            if len(b) != 794 or len(m) != 794 or b.condition.nunique() != 794 or m.condition.nunique() != 794:
                raise RuntimeError(f"audit discrepancy: {dataset} mask panel size or uniqueness")
            for arm, table, location in (("baseline", b, baseline_dir), ("physics_aware", m, mitigation_dir)):
                if arm == "baseline" and dataset == "Chapman" and seed == 104729:
                    pred_dir = OUT / "phase1_chapman_seed1_v4" / "test_predictions_v4"
                elif arm == "baseline":
                    pred_dir = location / f"seed_{seed}" / "test_predictions"
                elif dataset == "Chapman":
                    pred_dir = location / "physics_aware" / f"seed_{seed}" / "test_predictions"
                else:
                    pred_dir = location / f"seed_{seed}" / "test_predictions"
                for condition in ("clean", "RA_LL_reversal"):
                    reported = float(table.loc[table.condition == condition, "auprc"].iloc[0])
                    with np.load(pred_dir / f"{condition}.npz", allow_pickle=False) as z:
                        computed = float(average_precision_score(y[:, 0], z["p"][:, 0]))
                    close(computed, reported, f"{dataset} {arm} {seed} {condition} cached AUPRC")
                mean_mask = float(table.loc[table.condition.str.startswith("mask5_"), "auprc"].mean())
                clean = float(table.loc[table.condition == "clean", "auprc"].iloc[0])
                ra = float(table.loc[table.condition == "RA_LL_reversal", "auprc"].iloc[0])
                old = cross.loc[(cross.dataset == dataset) & (cross.seed == seed) & (cross.endpoint == "TWA")].iloc[0]
                page = post.loc[(post.dataset == dataset) & (post.seed == seed) & (post.arm == ("cached_clean_baseline" if arm == "baseline" else "physics_aware"))].iloc[0]
                for row, fields in ((page, ("clean_twa_auprc", "ra_ll_twa_auprc", "mean_five_lead_mask_twa_auprc", "matched_contrast")),):
                    for field, value in zip(fields, (clean, ra, mean_mask, mean_mask - ra)):
                        close(float(row[field]), value, f"figure data {dataset} {arm} {seed} {field}")
                if arm == "baseline":
                    for field, value in (("clean_auprc", clean), ("ra_ll_reversal_auprc", ra), ("mean_792_mask_auprc", mean_mask), ("matched_contrast_mask_minus_ra_ll", mean_mask - ra)):
                        close(float(old[field]), value, f"legacy figure data {dataset} {seed} {field}")
        output[dataset + "_TWA_test_positive"] = int(y[:, 0].sum())
        output[dataset + "_TWA_test_size"] = len(y)
        output[dataset + "_conditions_per_seed"] = 794
    return output


def ece(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 15).astype(np.int32), 14)
    return float(sum((bins == k).mean() * abs(y[bins == k].mean() - p[bins == k].mean()) for k in range(15) if np.any(bins == k)))


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    positive = y == 1
    hits = p >= threshold
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "sensitivity": float(hits[positive].mean()), "specificity": float((~hits[~positive]).mean()),
            "predicted_positive_fraction": float(hits.mean()), "mean_score": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)), "ece15": ece(y, p)}


def heedb_labels() -> tuple[np.ndarray, np.ndarray]:
    selected = pd.read_csv(PRIVATE / "independent_confirmation_v1" / "selection_initial.csv", usecols=["FileName"], dtype=str, keep_default_na=False)
    source = pd.read_csv(PRIVATE / "12SL_diagnoses" / "diagnoses_acquisition.csv", usecols=["FileName", "codes_physician"], dtype=str, keep_default_na=False)
    joined = selected.merge(source, on="FileName", how="left", validate="one_to_one", sort=False)
    codes = {1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181}
    y = np.asarray([bool({int(token) for token in re.findall(r"\d+", value)} & codes) for value in joined.codes_physician], dtype=np.uint8)
    if len(joined) != 10000 or joined.codes_physician.isna().any() or int(y.sum()) != 1456:
        raise RuntimeError("audit discrepancy: HEEDB independent physician-code mapping/count")
    return y, selected.FileName.to_numpy()


def heedb_aggregates_and_intervals() -> dict:
    y, _ = heedb_labels()
    independent = np.load(PRIVATE / "independent_confirmation_v1" / "predictions_float32.npy", mmap_mode="r")
    huang = np.load(PRIVATE / "huang_exploratory_v1" / "huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    detector = np.load(PRIVATE / "huang_exploratory_v1" / "detector_probabilities_float32.npy", mmap_mode="r")
    if independent.shape != (2, 3, 2, 10000, 3) or huang.shape != (3, 2, 10000, 3) or detector.shape != (2, 10000, 4):
        raise RuntimeError("audit discrepancy: HEEDB private prediction geometry")
    frozen = read_json(ROOT / "configs" / "FROZEN_HEEDB_INDEPENDENT_CONFIRMATION_v1.json")
    thresholds = np.asarray([[frozen["thresholds"][arm][str(seed)][0] for seed in SEEDS] for arm in ("baseline", "physics_aware", "baseline")])
    p = np.empty((3, 3, 2, 10000), dtype=np.float32)
    p[:2] = independent[:, :, :, :, 0]
    p[2] = huang[:, :, :, 0]
    names = ("baseline", "physics_aware", "huang_detect_correct_clean_resnet")
    old = pd.read_csv(OUT / "heedb_independent_confirmation_v1" / "AGGREGATE_RESULTS_BY_SEED.csv")
    hrows = pd.read_csv(OUT / "heedb_huang_exploratory_v1" / "TWA_BY_SEED.csv")
    hmeans = pd.read_csv(OUT / "heedb_huang_exploratory_v1" / "TWA_THREE_SEED_MEANS.csv")
    for arm_idx, arm in enumerate(names):
        for seed_idx, seed in enumerate(SEEDS):
            for cond_idx, condition in enumerate(("clean", "RA_LL_reversal")):
                row = (old.loc[(old.mapping == "core") & (old.label == "TWA") & (old.arm == arm) & (old.seed == seed) & (old.condition == condition)].iloc[0]
                       if arm_idx < 2 else hrows.loc[(hrows.arm == arm) & (hrows.seed == seed) & (hrows.condition == condition)].iloc[0])
                values = metric(y, p[arm_idx, seed_idx, cond_idx], thresholds[arm_idx, seed_idx])
                for name, value in values.items():
                    close(value, float(row[name]), f"HEEDB {arm} {seed} {condition} {name}")
    for arm in names:
        for condition in ("clean", "RA_LL_reversal"):
            subset = hrows.loc[(hrows.arm == arm) & (hrows.condition == condition)]
            meanrow = hmeans.loc[(hmeans.arm == arm) & (hmeans.condition == condition)].iloc[0]
            for name in METRICS:
                close(float(subset[name].mean()), float(meanrow[name]), f"Huang figure data {arm} {condition} {name}")
    if (detector[0].argmax(axis=1) == 0).sum() != 9701 or (detector[1].argmax(axis=1) == 2).sum() != 9919:
        raise RuntimeError("audit discrepancy: detector correctness counts")
    old_ci = pd.read_csv(OUT / "heedb_independent_confirmation_v1" / "PAIRED_BOOTSTRAP_INTERVALS.csv")
    huang_ci = pd.read_csv(OUT / "heedb_huang_exploratory_v1" / "PAIRED_PATIENT_BOOTSTRAP.csv")
    checks = (("auprc", average_precision_score), ("auroc", roc_auc_score),
              ("sensitivity", None), ("specificity", None))
    def point(metric_name: str, labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
        if metric_name == "auprc":
            return float(average_precision_score(labels, scores))
        if metric_name == "auroc":
            return float(roc_auc_score(labels, scores))
        positive = labels == 1
        prediction = scores >= threshold
        return float(prediction[positive].mean() if metric_name == "sensitivity" else (~prediction[~positive]).mean())
    ci_values = {kind: [] for kind, _ in checks}
    old_values = {kind: [] for kind in ("auprc", "sensitivity", "specificity")}
    rng_old = np.random.default_rng(20260924)
    rng_huang = np.random.default_rng(20260926)
    for replicate in range(1000):
        i_old = rng_old.integers(0, 10000, 10000)
        i_h = rng_huang.integers(0, 10000, 10000)
        for kind, _ in checks:
            for index, cache in ((i_old, old_values), (i_h, ci_values)):
                if kind not in cache:
                    continue
                labels = y[index]
                by = []
                for seed_idx in range(3):
                    m2 = point(kind, labels, p[1, seed_idx, 1, index], thresholds[1, seed_idx])
                    base = point(kind, labels, p[0, seed_idx, 1, index], thresholds[0, seed_idx])
                    h = point(kind, labels, p[2, seed_idx, 1, index], thresholds[2, seed_idx])
                    by.append((m2 - base, m2 - h, h - base))
                cache[kind].append(np.mean(by, axis=0))
    for kind, values in old_values.items():
        series = np.asarray(values)[:, 0]
        row = old_ci.loc[(old_ci.mapping == "core") & (old_ci.label == "TWA") & (old_ci.metric == kind) & (old_ci.contrast == "physics_aware_minus_baseline_ra_ll")].iloc[0].to_dict()
        interval(series, row, "HEEDB independent " + kind, tol=3e-5)
    for kind, values in ci_values.items():
        series = np.asarray(values)
        for column, (left, right) in enumerate((("physics_aware", "baseline"), ("physics_aware", "huang_detect_correct_clean_resnet"), ("huang_detect_correct_clean_resnet", "baseline"))):
            row = huang_ci.loc[(huang_ci.type == "paired_contrast") & (huang_ci.left == left) & (huang_ci.right == right) & (huang_ci.condition == "RA_LL_reversal") & (huang_ci.metric == kind)].iloc[0]
            close(float(np.quantile(series[:, column], .025)), float(row.low95), "Huang " + kind + " low95 " + left + " vs " + right, tol=3e-5)
            close(float(np.quantile(series[:, column], .975)), float(row.high95), "Huang " + kind + " high95 " + left + " vs " + right, tol=3e-5)
    return {"heedb_core_TWA_positive": int(y.sum()), "heedb_seed_condition_metric_rows_checked": 18,
            "independent_primary_interval_metrics_recomputed": list(old_values),
            "huang_paired_interval_metrics_recomputed": list(ci_values),
            "bootstrap_replicates_each": 1000, "detector_clean_correct": 9701, "detector_RA_LL_correct": 9919}


def main() -> None:
    result = {"status": "PASS", "hash_and_cohort": protocols_and_cohorts(),
              "saved_state_intervals": saved_state_contrasts(),
              "source_figure_data": chapman_ningbo_figure_data(),
              "heedb": heedb_aggregates_and_intervals()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
