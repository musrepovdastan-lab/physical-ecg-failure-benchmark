"""Aggregate, read-only post-hoc audit of the frozen HEEDB v1 predictions.

Only new aggregate diagnostics are written. No model is loaded for inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "heedb_external_v1"
NEW = ROOT / "outputs" / "heedb_failure_audit_v1"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "external_v1"
BASELINE = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
PROTO_PATH = ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json"
ARMS = ("baseline", "physics_aware")
SEEDS = (104729, 130363, 155921)
LABELS = ("TWA", "SB", "STach")
CORE = {"TWA": {1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181}, "SB": {21, 24}, "STach": {23}}
STRICT = {"TWA": {1140}, "SB": {21}}
AMBIGUOUS_TWA = {1141, 1142, 1182, 1200, 1201, 1203, 1207, 1208, *range(1210, 1225)}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def codes(text: str) -> set[int]:
    return {int(piece) for piece in re.findall(r"\d+", text)}


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 15).astype(np.int32), 14)
    total = 0.0
    for index in range(15):
        selected = bins == index
        if selected.any():
            total += selected.mean() * abs(float(y[selected].mean() - p[selected].mean()))
    return float(total)


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pos = y == 1
    neg = ~pos
    return {
        "n": int(len(y)),
        "positive": int(pos.sum()),
        "prevalence": float(pos.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float((p[pos] >= threshold).mean()),
        "specificity": float((p[neg] < threshold).mean()),
        "predicted_positive_fraction": float((p >= threshold).mean()),
        "threshold": float(threshold),
        "mean_score": float(p.mean()),
        "score_q10": float(np.quantile(p, .10)),
        "score_q50": float(np.quantile(p, .50)),
        "score_q90": float(np.quantile(p, .90)),
        "score_q99": float(np.quantile(p, .99)),
        "positive_score_q50": float(np.median(p[pos])),
        "negative_score_q50": float(np.median(p[neg])),
        "brier": float(brier_score_loss(y, p)),
        "ece15": ece15(y, p),
    }


def waveform_summary(records: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> dict:
    x = np.stack(records).astype(np.float32)
    z = (x - mean[None, :, None]) / std[None, :, None]
    q = {}
    for lead, index in (("I", 0), ("II", 1), ("V1", 6), ("V3", 8), ("V6", 11)):
        raw = x[:, index, :]
        norm = z[:, index, :]
        q[lead] = {
            "raw_abs_p50_uv": float(np.quantile(abs(raw), .50)),
            "raw_abs_p95_uv": float(np.quantile(abs(raw), .95)),
            "raw_abs_p99_uv": float(np.quantile(abs(raw), .99)),
            "normalized_mean": float(norm.mean()),
            "normalized_std": float(norm.std()),
            "normalized_abs_p95": float(np.quantile(abs(norm), .95)),
        }
    einthoven = x[:, 2, :] - (x[:, 1, :] - x[:, 0, :])
    return {
        "records": len(x),
        "samples_per_record": x.shape[2],
        "lead_stats": q,
        "record_median_abs_uv_q50": float(np.median(np.median(abs(x), axis=(1, 2)))),
        "record_rms_uv_q50": float(np.median(np.sqrt(np.mean(x * x, axis=(1, 2))))),
        "einthoven_abs_residual_p99_uv": float(np.quantile(abs(einthoven), .99)),
    }


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=True)
    proto = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
    expected_proto = (OLD / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip()
    provenance = json.loads((OLD / "RUN_PROVENANCE.json").read_text(encoding="utf-8"))
    hashes = {"protocol_matches": hash_file(PROTO_PATH) == expected_proto == provenance["protocol_sha256"]}
    hashes["v1_evaluator_source_matches"] = hash_file(ROOT / "src" / "heedb_external_v1.py") == proto["source_hashes"]["evaluation_code"]
    hashes["physics_source_matches"] = hash_file(ROOT / "src" / "ecg_physics.py") == proto["source_hashes"]["physics_source"]
    for arm, base in (("baseline", BASELINE), ("physics_aware", M2)):
        for seed in SEEDS:
            frozen = proto["models"][arm][str(seed)]
            unit = base / f"seed_{seed}"
            hashes[f"{arm}_{seed}_checkpoint_matches"] = hash_file(unit / "checkpoint.pt") == frozen["checkpoint_sha256"]
            hashes[f"{arm}_{seed}_preprocessing_matches"] = hash_file(unit / "preprocessing.npz") == frozen["preprocessing_sha256"]
            hashes[f"{arm}_{seed}_validation_matches"] = hash_file(unit / "validation.npz") == frozen["validation_threshold_source_sha256"]
    hashes["prediction_cache_matches"] = hash_file(PRIVATE / "predictions_float32.npy") == provenance["prediction_artifact_sha256"]
    hashes["final_selection_matches"] = hash_file(PRIVATE / "selection_final.csv") == provenance["selection_sha256"]
    if not all(hashes.values()):
        raise RuntimeError("Frozen artifact hash verification failed")
    import torch
    baseline_checkpoint_labels = []
    for seed in SEEDS:
        checkpoint = torch.load(BASELINE / f"seed_{seed}" / "checkpoint.pt", map_location="cpu", weights_only=True)
        baseline_checkpoint_labels.append(checkpoint["labels"])
    expected_snomed = ["164934002", "426177001", "427084000"]
    if baseline_checkpoint_labels != [expected_snomed] * 3:
        raise RuntimeError("Baseline checkpoint output labels are not TWA, SB, STach")

    frame = pd.read_csv(PRIVATE / "selection_final.csv", dtype=str, keep_default_na=False)
    parsed = [codes(text) for text in frame["codes_physician"]]
    recomputed = {}
    for name, mapping in (("core", CORE), ("strict", STRICT)):
        for label, accepted in mapping.items():
            y = np.array([bool(row & accepted) for row in parsed], dtype=np.uint8)
            column = frame[f"{name}_{label}"].astype(np.uint8).to_numpy()
            recomputed[f"{name}_{label}"] = {"positive": int(y.sum()), "column_mismatches": int((y != column).sum())}
    if any(value["column_mismatches"] for value in recomputed.values()):
        raise RuntimeError("Frozen label columns do not agree with physician code parsing")

    predictions = np.load(PRIVATE / "predictions_float32.npy", mmap_mode="r")
    if predictions.shape != (2, 3, 26, 10000, 3) or not np.isfinite(predictions).all():
        raise RuntimeError("Prediction cache shape or finiteness mismatch")
    freeze = json.loads((OLD / "EVALUATION_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["arms"] != list(ARMS) or freeze["seeds"] != list(SEEDS) or freeze["conditions"][:2] != ["clean", "RA_LL_reversal"] or proto["models"]["label_order"] != list(LABELS):
        raise RuntimeError("Prediction axis or label order mismatch")

    score_rows = []
    threshold_map = {}
    validation_cross_channel = []
    for arm_index, arm in enumerate(ARMS):
        base = BASELINE if arm == "baseline" else M2
        for seed_index, seed in enumerate(SEEDS):
            with np.load(base / f"seed_{seed}" / "validation.npz", allow_pickle=False) as file:
                thresholds = file["thresholds"]
                val_y, val_p = file["y"], file["p"]
            if not np.array_equal(thresholds, np.array(proto["models"][arm][str(seed)]["thresholds"], dtype=np.float32)):
                raise RuntimeError("Frozen threshold mismatch")
            threshold_map[(arm, seed)] = thresholds
            for true_index, true_label in enumerate(LABELS):
                for score_index, score_label in enumerate(LABELS):
                    validation_cross_channel.append({"arm": arm, "seed": seed, "true_label": true_label, "score_channel": score_label, "auroc": float(roc_auc_score(val_y[:, true_index], val_p[:, score_index])), "auprc": float(average_precision_score(val_y[:, true_index], val_p[:, score_index]))})
            for label_index, label in enumerate(LABELS):
                y = frame[f"core_{label}"].astype(np.uint8).to_numpy()
                for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                    p = np.asarray(predictions[arm_index, seed_index, condition_index, :, label_index])
                    score_rows.append({"arm": arm, "seed": seed, "condition": condition, "label": label, **metric(y, p, float(thresholds[label_index]))})
            # Validation score distribution is a reference, not a HEEDB calibration set.
            y = val_y[:, 0].astype(np.uint8)
            p = val_p[:, 0]
            score_rows.append({"arm": arm, "seed": seed, "condition": "ningbo_validation", "label": "TWA", **metric(y, p, float(thresholds[0]))})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(NEW / "AGGREGATE_SCORE_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(validation_cross_channel).groupby(["arm", "true_label", "score_channel"])[["auroc", "auprc"]].mean().reset_index().to_csv(NEW / "NINGBO_VALIDATION_CHANNEL_CHECK.csv", index=False)

    frozen_rows = pd.read_csv(OLD / "AGGREGATE_RESULTS_BY_SEED.csv")
    check_errors = []
    for row in scores[scores.condition != "ningbo_validation"].itertuples(index=False):
        match = frozen_rows[(frozen_rows.mapping == "core") & (frozen_rows.label == row.label) & (frozen_rows.arm == row.arm) & (frozen_rows.seed == row.seed) & (frozen_rows.condition == row.condition)]
        if len(match) != 1:
            raise RuntimeError("Frozen result row missing or duplicated")
        check_errors.extend((abs(row.auprc - match.iloc[0].auprc), abs(row.sensitivity - match.iloc[0].sensitivity_at_frozen_threshold)))
    frozen_contrasts = pd.read_csv(OLD / "AGGREGATE_CONTRASTS.csv")
    contrast_errors = []
    reconstructed_contrasts = {}
    twa = scores[(scores.label == "TWA") & (scores.condition != "ningbo_validation")]
    for statistic, column in (("auprc", "auprc"), ("sensitivity_at_frozen_threshold", "sensitivity")):
        values = {(row.arm, row.seed, row.condition): getattr(row, column) for row in twa.itertuples(index=False)}
        for name, left, right in (
            ("physics_aware_minus_baseline_ra_ll", ("physics_aware", "RA_LL_reversal"), ("baseline", "RA_LL_reversal")),
            ("physics_aware_minus_baseline_clean", ("physics_aware", "clean"), ("baseline", "clean")),
            ("baseline_clean_minus_ra_ll", ("baseline", "clean"), ("baseline", "RA_LL_reversal")),
            ("physics_aware_clean_minus_ra_ll", ("physics_aware", "clean"), ("physics_aware", "RA_LL_reversal")),
        ):
            value = float(np.mean([values[(left[0], seed, left[1])] - values[(right[0], seed, right[1])] for seed in SEEDS]))
            match = frozen_contrasts[(frozen_contrasts.mapping == "core") & (frozen_contrasts.label == "TWA") & (frozen_contrasts.metric == statistic) & (frozen_contrasts.contrast == name)]
            if len(match) != 1:
                raise RuntimeError("Frozen primary contrast missing or duplicated")
            contrast_errors.append(abs(value - float(match.iloc[0].estimate)))
            reconstructed_contrasts[f"{statistic}:{name}"] = value

    # Exploratory code-definition diagnostics.  No label or threshold is changed.
    core = frame["core_TWA"].astype(np.uint8).to_numpy()
    strict = frame["strict_TWA"].astype(np.uint8).to_numpy()
    ambiguous = np.array([bool(row & AMBIGUOUS_TWA) for row in parsed], dtype=bool)
    broad = np.asarray(core | ambiguous, dtype=np.uint8)
    keep_without_ambiguous_negatives = (core == 1) | ~ambiguous
    scenarios = {
        "frozen_core": (np.ones(len(core), bool), core),
        "strict_1140": (np.ones(len(core), bool), strict),
        "exploratory_broad_core_plus_ambiguous": (np.ones(len(core), bool), broad),
        "exploratory_drop_ambiguous_core_negatives": (keep_without_ambiguous_negatives, core),
    }
    scenario_rows = []
    for name, (keep, y_full) in scenarios.items():
        y = y_full[keep]
        for arm_index, arm in enumerate(ARMS):
            for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                aps, aucs = [], []
                for seed_index, _seed in enumerate(SEEDS):
                    p = np.asarray(predictions[arm_index, seed_index, condition_index, keep, 0])
                    aps.append(float(average_precision_score(y, p)))
                    aucs.append(float(roc_auc_score(y, p)))
                scenario_rows.append({"scenario": name, "arm": arm, "condition": condition, "n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()), "mean_three_seed_auprc": float(np.mean(aps)), "mean_three_seed_auroc": float(np.mean(aucs))})
    pd.DataFrame(scenario_rows).to_csv(NEW / "AGGREGATE_LABEL_SCENARIOS.csv", index=False)
    code_overlap = {
        "ambiguous_any": int(ambiguous.sum()),
        "ambiguous_core_negative": int((ambiguous & (core == 0)).sum()),
        "mixed_st_t_1141": int(sum(1141 in row for row in parsed)),
        "other_suggestive_1142_or_1182": int(sum(bool(row & {1142, 1182}) for row in parsed)),
        "serial_or_fragment_1200_to_1224": int(sum(bool(row & (AMBIGUOUS_TWA - {1141, 1142, 1182})) for row in parsed)),
        "core_without_strict_1140": int(((core == 1) & (strict == 0)).sum()),
    }

    manifest = pd.read_csv(PRIVATE / "waveform_object_manifest.csv")
    dat = manifest[manifest.remote_key.str.endswith(".dat")].copy()
    if len(dat) != 10000 or dat.FileName.duplicated().any():
        raise RuntimeError("WFDB object manifest mismatch")
    rate_by_filename = dict(zip(dat.FileName, np.where(dat.remote_bytes.to_numpy() == 60000, 250, 500)))
    rates = np.array([rate_by_filename[name] for name in frame.FileName])
    rate_rows = []
    for rate in (250, 500):
        chosen = rates == rate
        y = core[chosen]
        if len(y) < 30 or y.sum() < 20:
            rate_rows.append({"rate_hz": rate, "n": int(len(y)), "positive": int(y.sum()), "metrics_suppressed": True})
            continue
        for arm_index, arm in enumerate(ARMS):
            for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                by_seed = []
                for seed_index, seed in enumerate(SEEDS):
                    p = np.asarray(predictions[arm_index, seed_index, condition_index, chosen, 0])
                    by_seed.append(metric(y, p, float(threshold_map[(arm, seed)][0])))
                rate_rows.append({"rate_hz": rate, "n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()), "arm": arm, "condition": condition, "metrics_suppressed": False, "mean_three_seed_auroc": float(np.mean([row["auroc"] for row in by_seed])), "mean_three_seed_auprc": float(np.mean([row["auprc"] for row in by_seed])), "mean_three_seed_sensitivity": float(np.mean([row["sensitivity"] for row in by_seed]))})
    pd.DataFrame(rate_rows).to_csv(NEW / "AGGREGATE_RATE_DIAGNOSTICS.csv", index=False)

    with np.load(BASELINE / "seed_104729" / "preprocessing.npz", allow_pickle=False) as f:
        train_mean, train_std = f["mean"], f["std"]
    split = json.loads((BASELINE / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))
    train_ids = {row["record_id"] for row in split["rows"] if row["split"] == "train"}
    canonical = json.loads((BASELINE / "CANONICAL_RECORDS.json").read_text(encoding="utf-8"))
    train_paths = [row["waveform_path"] for row in canonical["records"] if row["record_id"] in train_ids]
    rng = np.random.default_rng(20260924)
    train_choice = rng.choice(len(train_paths), size=256, replace=False)
    heedb_choice = rng.choice(len(frame), size=256, replace=False)
    ningbo = [np.asarray(loadmat(ROOT / "data" / "challenge2021_v103" / "ningbo" / Path(train_paths[i]).relative_to("training/ningbo"))["val"][:, ::5], dtype=np.float32) for i in train_choice]
    cache = np.load(PRIVATE / "signals_10000_float32.npy", mmap_mode="r")
    heedb = [np.asarray(cache[i, :, ::5], dtype=np.float32) for i in heedb_choice]
    waveform = {"ningbo_train_256": waveform_summary(ningbo, train_mean, train_std), "heedb_selected_256_after_matched_decimation": waveform_summary(heedb, train_mean, train_std)}
    waveform["heedb_selected_256_as_v1_input"] = waveform_summary([np.asarray(cache[i], dtype=np.float32) for i in heedb_choice], train_mean, train_std)
    waveform["heedb_v1_model_input_length"] = int(cache.shape[2])
    waveform["ningbo_train_model_input_length"] = int(ningbo[0].shape[1])
    waveform["heedb_corrected_v2_proposed_input_length"] = int(heedb[0].shape[1])
    from src.ecg_physics import derive_display, named_reversal
    from src.heedb_external_v1 import reversal_ra_ll

    algebra_input = np.asarray(cache[:32, :, ::5], dtype=np.float32)
    algebra_independent = np.concatenate((algebra_input[:, :2], algebra_input[:, 6:]), axis=1)
    algebra_reference = derive_display(named_reversal(algebra_independent, "RA_LL_reversal"))
    reversal_difference = abs(reversal_ra_ll(algebra_input) - algebra_reference)
    reversal_max_abs_difference_uv = float(np.max(reversal_difference))
    reversal_by_lead = {
        name: {"max_abs_difference_uv": float(np.max(reversal_difference[:, index, :])), "p99_abs_difference_uv": float(np.quantile(reversal_difference[:, index, :], .99))}
        for index, name in enumerate(proto["preprocessing"]["input_lead_order"])
    }

    unit_checks = []
    for rate in (250, 500):
        sampled = dat[dat.remote_bytes == (60000 if rate == 250 else 120000)].head(20)
        max_unit_error = 0.0
        max_cache_error = 0.0
        gains, baselines, leads, frequencies = set(), set(), set(), set()
        for file_name in sampled.FileName:
            path = PRIVATE / "waveforms" / file_name
            physical = wfdb.rdrecord(str(path), physical=True)
            digital = wfdb.rdrecord(str(path), physical=False)
            gains.update(physical.adc_gain)
            baselines.update(physical.baseline)
            leads.add(tuple(physical.sig_name))
            frequencies.add(float(physical.fs))
            max_unit_error = max(max_unit_error, float(np.max(abs(digital.d_signal - physical.p_signal * 1000))))
            x = physical.p_signal.T.astype(np.float32)
            if rate == 250:
                x = resample_poly(x, 2, 1, axis=1).astype(np.float32)
            x *= np.float32(1000)
            row_index = int(np.flatnonzero(frame.FileName.to_numpy() == file_name)[0])
            max_cache_error = max(max_cache_error, float(np.max(abs(x - cache[row_index]))))
        unit_checks.append({"rate_hz": rate, "records_checked": len(sampled), "adc_gains_per_millivolt": sorted(gains), "adc_baselines": sorted(baselines), "canonical_lead_order": len(leads) == 1 and next(iter(leads)) == tuple(proto["preprocessing"]["input_lead_order"]), "sampling_frequencies": sorted(frequencies), "max_abs_digital_minus_physical_times_1000": max_unit_error, "max_abs_rebuilt_minus_cached_uv": max_cache_error})

    summary = {
        "hash_checks": hashes,
        "recomputed_label_counts": recomputed,
        "primary_reconstruction": reconstructed_contrasts,
        "maximum_absolute_frozen_seed_metric_difference": float(max(check_errors)),
        "maximum_absolute_frozen_primary_contrast_difference": float(max(contrast_errors)),
        "ambiguous_code_overlap": code_overlap,
        "waveform_comparison": waveform,
        "unit_checks": unit_checks,
        "reversal_max_abs_difference_from_frozen_physics_uv": reversal_max_abs_difference_uv,
        "reversal_difference_by_lead": reversal_by_lead,
        "rate_counts": {str(rate): int((rates == rate).sum()) for rate in (250, 500)},
        "model_channel_order": list(LABELS),
        "baseline_checkpoint_snomed_order": expected_snomed,
        "m2_validation_labels_match_baseline_by_seed": {
            str(seed): bool(np.array_equal(np.load(M2 / f"seed_{seed}" / "validation.npz", allow_pickle=False)["y"], np.load(BASELINE / f"seed_{seed}" / "validation.npz", allow_pickle=False)["y"]))
            for seed in SEEDS
        },
        "selection_patients": int(frame.BDSPPatientID.nunique()),
    }
    (NEW / "AGGREGATE_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"hashes_pass": all(hashes.values()), "max_seed_metric_error": max(check_errors), "max_primary_contrast_error": max(contrast_errors), "rate_counts": summary["rate_counts"], "input_lengths": [waveform["ningbo_train_model_input_length"], waveform["heedb_v1_model_input_length"]]}))


if __name__ == "__main__":
    main()
