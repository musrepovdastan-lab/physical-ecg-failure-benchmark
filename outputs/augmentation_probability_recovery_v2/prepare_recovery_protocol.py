from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
SEEDS = (104729, 130363, 155921)
NEW_ARMS = ("physics_p010", "physics_p025")
SCALAR = ("clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201", "mask5_0_1_2_3_5")
HIST_ARMS = ("ordinary", "physics_aware", "combined")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def historical_prediction(cohort: str, arm: str, seed: int, condition: str) -> Path:
    if condition.startswith("cycle_slots_"):
        return ROOT / "outputs/heldout_limb_cycles_v1/predictions" / cohort / arm / f"seed_{seed}" / f"{condition}.npz"
    if cohort == "Chapman":
        if arm == "ordinary":
            if seed == 104729:
                if condition.startswith("mask5_"):
                    return ROOT / "outputs/phase1b_matched_reversal_development_v6/predictions" / f"{condition}.npz"
                return ROOT / "outputs/phase1_chapman_seed1_v4/test_predictions_v4" / f"{condition}.npz"
            return ROOT / "outputs/phase2_chapman_multiseed_v7" / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
        return ROOT / "outputs/mitigation_m1_v1" / arm / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
    if arm == "ordinary":
        if condition.startswith("mask5_"):
            return ROOT / "outputs/mask_semantics_recovery_v2/baseline_v2" / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
        return ROOT / "outputs/ningbo_confirmation_v2" / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
    if arm == "physics_aware":
        if condition.startswith("mask5_"):
            return ROOT / "outputs/mask_semantics_recovery_v2/physics_aware_m2" / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
        return ROOT / "outputs/mitigation_m2_ningbo_v1" / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"
    return ROOT / "outputs/ningbo_four_arm_replication_v1" / arm / f"seed_{seed}" / "test_predictions" / f"{condition}.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, str] = {}

    def add(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = path.relative_to(ROOT).as_posix()
        if rel in inputs:
            return
        inputs[rel] = sha(path)

    # Frozen recovery and review materials.
    for rel in [
        "outputs/augmentation_probability_study_v1/REVIEW_NOTES_20261018.md",
        "outputs/augmentation_probability_study_v1/INDEPENDENT_REVIEW_AUDIT_20261018.json",
        "outputs/augmentation_probability_study_v1/FROZEN_PROTOCOL.json",
        "outputs/augmentation_probability_study_v1/run_probability_study.py",
        "outputs/augmentation_probability_study_v1/analyze_ningbo_mixtures.py",
        "outputs/augmentation_probability_study_v1/paired_bootstrap.py",
        "outputs/augmentation_probability_study_v1/build_final_report.py",
        "outputs/augmentation_probability_study_v1/FINAL_PROBABILITY_STUDY_REPORT_RU.md",
        "outputs/augmentation_probability_study_v1/RUN_MANIFEST.json",
        "outputs/augmentation_probability_study_v1/PAIRED_BOOTSTRAP_TWA_AUPRC.csv",
        "outputs/augmentation_probability_study_v1/PAIRED_CONTRASTS_POINT_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/METRICS_ALL_ARMS_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/METRICS_CHAPMAN_NEW_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/METRICS_NINGBO_NEW_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/METRICS_CHAPMAN_SUMMARY.csv",
        "outputs/augmentation_probability_study_v1/METRICS_NINGBO_SUMMARY.csv",
        "outputs/augmentation_probability_study_v1/MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/MIXTURE_RESULTS_NEW_BY_SEED.csv",
        "outputs/augmentation_probability_study_v1/CALIBRATION_PROVENANCE_NEW.json",
        "outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_001.json",
        "outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_002.json",
        "outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_003.json",
        "outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_004.json",
        "outputs/mixed_fault_calibration_v1/calibration_split_indices.npz",
        "outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json",
        "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json",
        "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json",
        "outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
        "outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy",
        "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json",
        "outputs/phase1_chapman_seed1_v4/preprocessing_v4.npz",
        "src/chapman_phase1.py",
        "src/chapman_stage23.py",
        "src/ecg_physics.py",
        "src/ningbo_confirmation_v1.py",
        "src/mask_semantics_recovery_v2.py",
        "outputs/heldout_limb_cycles_v1/cycle_adapter.py",
        "REMAINING_WORK_PLAN_20261018.md",
    ]:
        add(ROOT / rel)

    # Fingerprint the actual waveform/header files read by clean validation and
    # the fixed test replay. Only validation and test records are needed here.
    chapman_split = json.loads((ROOT / "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json").read_text(encoding="utf-8"))["rows"]
    chapman_ids = sorted({r["record_id"] for r in chapman_split if r["split"] in ("validation", "test")})
    chapman_data = ROOT / "data/challenge2021_v103/chapman_shaoxing"
    for record_id in chapman_ids:
        add(chapman_data / f"{record_id}.hea")
        add(chapman_data / f"{record_id}.mat")
    ningbo_base = ROOT / "data/challenge2021_v103/ningbo"
    ningbo_records = json.loads((ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    ningbo_by_id = {r["record_id"]: r for r in ningbo_records}
    ningbo_split = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    ningbo_ids = sorted({r["record_id"] for r in ningbo_split if r["split"] in ("validation", "test")})
    for record_id in ningbo_ids:
        rec = ningbo_by_id[record_id]
        add(ningbo_base / Path(rec["waveform_path"]).relative_to("training/ningbo"))

    units = []
    for cohort in ("Chapman", "Ningbo"):
        for arm in NEW_ARMS:
            for seed in SEEDS:
                unit = STUDY / "training" / cohort / arm / f"seed_{seed}"
                pred = STUDY / "predictions" / cohort / arm / f"seed_{seed}"
                for fn in ("best.pt", "checkpoint.pt", "validation.npz", "validation_gate.json", "evaluation_complete.json"):
                    add(unit / fn)
                evaluation = json.loads((unit / "evaluation_complete.json").read_text(encoding="utf-8"))
                listed = evaluation.get("prediction_hashes", {})
                if len(listed) != 797 or "test_labels_order.npz" not in listed:
                    raise RuntimeError(f"Expected 796 condition predictions plus test_labels_order.npz: {cohort}/{arm}/{seed}; got {len(listed)}")
                for filename, expected_hash in listed.items():
                    path = pred / filename
                    add(path)
                    if inputs[path.relative_to(ROOT).as_posix()] != expected_hash:
                        raise RuntimeError(f"test prediction does not match original evaluation manifest: {path}")
                units.append({"cohort": cohort, "arm": arm, "seed": seed, "test_conditions": len(listed) - 1, "prediction_files_in_manifest": len(listed)})

    # Historical comparators and cached AP draws needed for the intervals.
    for cohort in ("Chapman", "Ningbo"):
        for arm in HIST_ARMS:
            for seed in SEEDS:
                for condition in SCALAR:
                    add(historical_prediction(cohort, arm, seed, condition))
        for cache in (STUDY / "bootstrap_draws").glob(f"{cohort}_*_*.npz"):
            add(cache)

    protocol = {
        "package": "augmentation_probability_recovery_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RECOVERY_PROTOCOL_FROZEN_BEFORE_RECOVERY_COMPUTATIONS",
        "purpose": "Post-hoc recovery of validation-dependent analysis for the completed augmentation_probability_study_v1; no training.",
        "important_temporal_limit": "This recovery protocol is post hoc, after original test outcomes were examined. It is not presented as preregistered or fixed before the original test evaluation.",
        "diagnosis": "The v1 runner overwrote validation.npz on each epoch; after loading best.pt and recomputing best-checkpoint validation, it wrote validation_gate.json but did not replace validation.npz. Therefore v1 validation.npz reflects the last epoch while test predictions used checkpoint.pt restored from best.pt. Raw test ranking and probability metrics are unaffected; thresholds and new Ningbo calibration based on the mismatched validation predictions require recovery.",
        "scope": {"cohorts": ["Chapman", "Ningbo"], "arms": list(NEW_ARMS), "seeds": list(SEEDS), "units": units, "recovery_data_record_counts": {"Chapman_validation_and_test": len(chapman_ids), "Ningbo_validation_and_test": len(ningbo_ids)}, "no_training": True, "no_full_792_mask_model_reinference": True, "no_HEEDB_or_Founder": True},
        "validation_recovery": {"checkpoint": "best.pt; state_dict must exactly equal checkpoint.pt used for test inference", "data": "existing validation split, canonical order, original source_data/raw_for/labels_for, preprocessing means/stds, batch size 128, CUDA autocast float16 sigmoid inference", "comparison": "recomputed TWA AUPRC/AUROC against validation_gate.json", "predeclared_absolute_tolerance": 1e-6, "tolerance_reason": "Same saved weights, records, preprocessing, inference code and deterministic metric definitions should reproduce to float precision; 1e-6 allows harmless floating-point reduction variation, while materially larger deviations trigger a stop before dependent analysis.", "test_probability_replay": {"indices": [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987], "conditions": ["clean", "RA_LL_reversal"], "predeclared_max_absolute_tolerance": 1e-6}},
        "threshold_recovery": {"method": "per model and endpoint, linear 90th percentile of raw scores among clean-validation negatives; apply to stored test probabilities", "metrics": ["sensitivity", "specificity", "predicted_positive_fraction", "accuracy"], "raw_test_metrics_preserved": ["AUPRC", "AUROC", "Brier", "ECE15"], "raw_metric_comparison_tolerance": 1e-12, "test_conditions_per_unit": 796},
        "ningbo_calibration": {"split_file": "outputs/mixed_fault_calibration_v1/calibration_split_indices.npz", "A": "Platt fitting only", "B": "negative-only raw 90th percentile threshold", "calibrated_threshold": "transform raw B threshold through fitted calibrator; do not independently quantile calibrated scores", "fit": "same mixed_fault_calibration_v2 formula, clipping, positive slope exp(theta0), bounds, L-BFGS-B parameters", "mixture_q_fault": [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1], "calibration_leakage_note": "The full validation set was used during checkpoint selection before A/B was applied to refit calibration; report this dependence explicitly.", "comparators": "Reuse historical calibrators and thresholds; do not refit."},
        "bootstrap": {"reuse_existing_indices": True, "replicates": 1000, "paired_synchronized_indices": True, "aggregate_seed_specific_differences_within_each_replicate": True, "preserve_existing_48_TWA_AUPRC_comparisons_and_cached_draws": True, "new_metrics": ["TWA AUROC", "TWA raw Brier"], "contrasts": ["physics_p025_minus_combined", "physics_p025_minus_physics_aware", "physics_p010_minus_ordinary", "physics_p010_minus_physics_aware"], "conditions": list(SCALAR), "new_groups": 80, "brier_sign": "negative differences favor the first arm (lower Brier)."},
        "original_inputs_sha256": dict(sorted(inputs.items())),
        "original_input_count": len(inputs),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    protocol["recovery_protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = OUT / "RECOVERY_PROTOCOL.json"
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(path), "inputs": len(inputs), "units": len(units), "protocol_sha256": protocol["recovery_protocol_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
