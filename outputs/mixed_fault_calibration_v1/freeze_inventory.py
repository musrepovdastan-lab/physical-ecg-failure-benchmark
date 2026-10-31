"""Freeze the mixed-fault reanalysis inventory before calibration/scoring."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mixed_fault_calibration_v1"
SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary_resnet", "physics_aware_resnet", "chest_only_resnet", "huang_pipeline")
ENDPOINTS = ("TWA", "SB", "STach")
Q = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 1.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ningbo_models = ROOT / "outputs/chest_only_baseline_v1/models/Ningbo"
    ningbo_comps = ROOT / "outputs/chest_only_baseline_v1/comparators/Ningbo"
    n_confirm = ROOT / "outputs/ningbo_confirmation_v2"
    heedb_private = ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1"
    heedb_huang = ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1"
    heedb_out = ROOT / "outputs/heedb_independent_confirmation_v1"
    chest_out = ROOT / "outputs/chest_only_baseline_v1"
    files: dict[str, Path] = {}
    # New-model validation, checkpoints, preprocessing and clean/reversal scores.
    for seed in SEEDS:
        sd = str(seed)
        for name in ("validation.npz", "checkpoint.pt", "preprocessing.npz", "training_complete.json"):
            files[f"chest_only/{sd}/{name}"] = ningbo_models / f"seed_{sd}" / name
        for cond in ("clean", "RA_LL_reversal"):
            files[f"chest_only/{sd}/{cond}"] = ningbo_models / f"seed_{sd}/test_predictions/{cond}.npz"
        for model in ("clean_12lead", "physics_aware_12lead"):
            for cond in ("clean", "RA_LL_reversal"):
                files[f"{model}/{sd}/{cond}"] = ningbo_comps / model / f"seed_{sd}/{cond}.npz"
        files[f"ordinary_val/{sd}"] = n_confirm / f"seed_{sd}/validation.npz"
        files[f"physics_val/{sd}"] = ROOT / f"outputs/mitigation_m2_ningbo_v1/seed_{sd}/validation.npz"
    files.update({
        "ningbo_cohort": n_confirm / "SPLIT_MEMBERSHIP.json",
        "ningbo_protocol": n_confirm / "FROZEN_PROTOCOL_HASH.json",
        "chest_protocol": chest_out / "FROZEN_CHEST_ONLY_PROTOCOL.json",
        "chest_report_manifest": chest_out / "REPORTING_MANIFEST.json",
        "heedb_cohort_freeze": heedb_out / "COHORT_FREEZE.json",
        "heedb_protocol": heedb_out / "FROZEN_PROTOCOL_SHA256.txt",
        "heedb_run_provenance": heedb_out / "RUN_PROVENANCE.json",
        "heedb_inference_complete": heedb_out / "INFERENCE_COMPLETE.json",
        "heedb_selection": heedb_private / "selection_initial.csv",
        "heedb_signal_cache": heedb_private / "signals_10000_float32.npy",
        "heedb_prediction_cache": heedb_private / "predictions_float32.npy",
        "heedb_huang_prediction": heedb_huang / "huang_diagnostic_probabilities_float32.npy",
        "heedb_huang_protocol": ROOT / "outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json",
        "heedb_chest_protocol": ROOT / "outputs/chest_only_baseline_v1/FROZEN_CHEST_ONLY_PROTOCOL.json",
        "heedb_labels_source": ROOT / "data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv",
    })
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inventory inputs: {missing}")

    # Verify corresponding cohort labels/order across all Ningbo model outputs.
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=20261017)
    split_meta = {}
    ref_y = None
    for seed in SEEDS:
        for arm, valpath in (("ordinary", n_confirm / f"seed_{seed}/validation.npz"),
                             ("physics_aware", ROOT / f"outputs/mitigation_m2_ningbo_v1/seed_{seed}/validation.npz"),
                             ("chest_only", ningbo_models / f"seed_{seed}/validation.npz")):
            z = np.load(valpath, allow_pickle=False)
            y = z["y"].astype(np.uint8)
            if y.ndim != 2 or y.shape[1] != 3 or not set(np.unique(y)).issubset({0, 1}):
                raise ValueError(f"Invalid validation labels in {valpath}")
            if ref_y is None:
                ref_y = y
            elif not np.array_equal(y, ref_y):
                raise ValueError(f"Validation labels/order mismatch in {valpath}")
            split_meta[f"{arm}/{seed}"] = {"path": valpath.relative_to(ROOT).as_posix(), "n": len(y),
                "validation_twa_prevalence": float(y[:, 0].mean()), "score_shape": list(z["p"].shape),
                "thresholds": z["thresholds"].astype(float).tolist() if "thresholds" in z else None}
    if ref_y is None:
        raise AssertionError
    strata = np.asarray(["".join(map(str, row.tolist())) for row in ref_y])
    a, b = next(split.split(np.zeros(len(ref_y)), strata))
    if len(np.unique(strata[a])) != len(np.unique(strata)) or len(np.unique(strata[b])) != len(np.unique(strata)):
        raise RuntimeError("Validation A/B split dropped a label-combination stratum")
    np.savez_compressed(OUT / "calibration_split_indices.npz", A=a.astype(np.int32), B=b.astype(np.int32))
    split_hash = sha(OUT / "calibration_split_indices.npz")

    # Test records and labels are bound to canonical source artifacts. Existing HEEDB model predictions are
    # already frozen; reading them here is an inventory operation, not fitting/selecting any analysis choice.
    n_tests = {}
    for seed in SEEDS:
        n_tests[str(seed)] = {}
        for arm, base in (("ordinary_resnet", ningbo_comps / "clean_12lead"),
                          ("physics_aware_resnet", ningbo_comps / "physics_aware_12lead"),
                          ("chest_only_resnet", ningbo_models)):
            clean = base / f"seed_{seed}/test_predictions/clean.npz" if arm == "chest_only_resnet" else base / f"seed_{seed}/clean.npz"
            rev = base / f"seed_{seed}/test_predictions/RA_LL_reversal.npz" if arm == "chest_only_resnet" else base / f"seed_{seed}/RA_LL_reversal.npz"
            if not clean.exists() or not rev.exists():
                raise FileNotFoundError(f"Missing Ningbo {arm} condition scores seed={seed}")
            with np.load(clean, allow_pickle=False) as c, np.load(rev, allow_pickle=False) as r:
                if not np.array_equal(c["y"], r["y"]):
                    raise ValueError(f"Ningbo condition labels/order differ: {clean} {rev}")
                n_tests[str(seed)][arm] = {"n": len(c["y"]), "labels_shape": list(c["y"].shape),
                    "record_order_sha256": json.loads(str(c["binding"].item())).get("test_record_order_sha256")}
    hp = np.load(heedb_private / "predictions_float32.npy", mmap_mode="r")
    hhuang = np.load(heedb_huang / "huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    if hp.shape != (2, 3, 2, 10000, 3) or hhuang.shape != (3, 2, 10000, 3):
        raise ValueError(f"Unexpected HEEDB saved prediction shapes: {hp.shape} {hhuang.shape}")
    # confirm arm/condition label order and the exact 10k selected patient cohort
    selection = files["heedb_selection"]
    with selection.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    if "FileName" not in header or "BDSPPatientID" not in header:
        raise ValueError("HEEDB selection lacks record/patient binding columns")

    coverage = []
    for arm in ARMS:
        for source in ("Ningbo", "HEEDB_I0006_independent_confirmation"):
            for seed in SEEDS:
                if source == "Ningbo":
                    validation = "available" if arm != "huang_pipeline" else "unavailable (no frozen full detector→predicted correction→classifier validation predictions found)"
                    clean_test = "available"
                    reversal_test = "available"
                    checkpoint = "available"
                    preproc = "available"
                elif arm == "huang_pipeline":
                    validation, clean_test, reversal_test = "unavailable", "available", "available"
                    checkpoint, preproc = "published frozen detector + existing clean ResNet; see Huang protocol", "frozen Huang protocol"
                elif arm == "chest_only_resnet":
                    validation, clean_test, reversal_test = "available (Ningbo source validation)", "missing; scheduled frozen-model inference", "missing; limb reversal does not alter V1–V6; infer identically"
                    checkpoint, preproc = "available", "available (V1–V6 train-derived normalization; 500Hz→100Hz ::5)"
                else:
                    validation = "available (Ningbo source validation)"
                    clean_test, reversal_test = "available", "available"
                    checkpoint, preproc = "available", "available"
                coverage.append({"method": arm, "source": source, "seed": seed, "validation": validation,
                                 "clean_test": clean_test, "RA_LL_test": reversal_test,
                                 "checkpoint": checkpoint, "preprocessing": preproc,
                                 "record_order_and_labels": "bound; see protocol and prediction artifacts"})
    with (OUT / "SCORE_COVERAGE_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0])); w.writeheader(); w.writerows(coverage)

    input_hashes = {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for k, p in files.items()}
    protocol = {
        "title": "Mostly-clean RA-LL simulated-fault mixtures and source-validation-only calibration",
        "status": "FROZEN_BEFORE_NEW_CHEST_HEEDB_INFERENCE_AND_CALIBRATION",
        "analysis_class": "exploratory_reanalysis_of_previously_accessed_test_sources",
        "primary_source": "Ningbo; same split and canonical test records as prior Ningbo confirmation",
        "external_source": "HEEDB I0006 independent-confirmation cohort, one record per patient, 10,000; same Emory source as earlier HEEDB cohort; not an independent center",
        "methods": list(ARMS), "seeds": list(SEEDS), "endpoints": list(ENDPOINTS), "primary_endpoint": "TWA",
        "fault": "RA_LL_reversal", "q_fault_scenarios": list(Q),
        "mixture": "For each record, clean and exact RA–LL reversal score/label copies receive weights 1-q and q; assign fault independently of labels and patients; this is an explicitly hypothetical scenario, not measured prevalence.",
        "Ningbo_test_n_and_bindings": n_tests,
        "validation_split": {"unit": "existing Ningbo validation record", "stratification": "joint three-label binary combination", "seed": 20261017,
            "A_fraction": 0.5, "B_fraction": 0.5, "indices_artifact": "calibration_split_indices.npz", "sha256": split_hash,
            "model_selection_caveat": "checkpoints were historically selected using the full validation set; A/B does not remove that selection dependence"},
        "calibration": {"methods": "ordinary ResNet, physics-aware ResNet and chest-only ResNet only; no frozen full-Huang-pipeline Ningbo validation scores were found, so Huang calibration and standardized B operating point are deferred",
            "mapping": "sigmoid(a*logit(p)+b), a>0", "fit_data": "clean validation A", "loss": "ordinary unweighted binary log loss; fit separately per method×seed×endpoint",
            "clip_raw_probability": [1e-6, 1-1e-6], "precision": "float64", "optimizer": "scipy.optimize.minimize L-BFGS-B; parameters theta=log(a), b; theta bounded [-12,12], b bounded [-30,30]",
            "convergence": "accept only success=True, finite objective/parameters, and positive a; otherwise fail and do not score that fit",
            "threshold": "90th percentile of negative clean validation B raw scores, NumPy linear quantile method; calibrated threshold is the same calibrator applied to that raw threshold; both modes therefore share the operating point",
            "no_refit": "No full-validation refit; no condition-specific mapping or HEEDB calibration"},
        "estimands": {"weighted_AP_AUROC": "exact weighted pooled empirical mixture, never average condition metrics", "probability": "weighted Brier and 15 equal-width ECE; reliability bins fixed at [k/15,(k+1)/15), last includes 1", "operating_point": "sensitivity, specificity, predicted-positive fraction, FP and FN per 1000 records", "summaries": "full-cohort central estimates; metrics calculated per seed then arithmetic mean across three seeds; do not average prediction vectors"},
        "bootstrap": {"replicates": 1000, "seed": 20261017, "Ningbo_cluster": "canonical record; same resampled multiplicity used for clean and reversal copies", "HEEDB_cluster": "patient; same multiplicity for both copies", "calibrators_and_thresholds": "fixed", "interval": "percentile 2.5%,97.5%; conditional on selected checkpoints and calibrators; excludes model/calibrator fitting uncertainty"},
        "secondary_endpoints": ["SB", "STach"], "HEEDB_label_limit": "physician-code proxy constructs; source SNOMED calibration does not establish proxy-label calibration validity",
        "scope_limits": ["No diagnostic model training", "No downloads", "No new latency benchmark", "No reserved_test evaluation", "No clinical cost or utility assignment"],
        "input_artifacts": input_hashes,
        "analysis_code_sha256": None,
        "preexisting_positioning_correction": {"integrated_deployment_v4": "warm batch-1 CPU request-to-response median 10.59ms physics-aware vs 31.51ms integrated Huang chain (2.97×) on documented matched fixture, 7 runs; implementation/hardware/workload-specific only", "HEEDB": "external relative to Chinese training sources; same Emory source across cohorts, not two external centers", "Cao": "direct full-PDF access failed; indexed text was used; retain limitation"}
    }
    dump(OUT / "FROZEN_ANALYSIS_PROTOCOL.json", protocol)
    dump(OUT / "SPLIT_PROVENANCE.json", {"n": len(ref_y), "joint_label_strata_counts": {str(k): int(v) for k, v in zip(*np.unique(strata, return_counts=True))},
        "A_indices_sha256": hashlib.sha256(a.astype(np.int32).tobytes()).hexdigest(), "B_indices_sha256": hashlib.sha256(b.astype(np.int32).tobytes()).hexdigest(),
        "artifact_sha256": split_hash, "seed": 20261017})
    print(f"FROZEN protocol={sha(OUT / 'FROZEN_ANALYSIS_PROTOCOL.json')} split={split_hash} rows={len(coverage)}")


if __name__ == "__main__":
    main()
