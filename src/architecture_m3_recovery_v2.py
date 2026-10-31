"""Versioned, cache-bound corrected reanalysis of the six frozen M3 Chapman models.

Historical M3 outputs are never modified. Legacy clean/RA_LL scores are reused
only after a frozen deterministic replay; all five-lead mask predictions are
recomputed with the corrected removal semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/architecture_replication_m3_recovery_v2"
OLD = ROOT / "outputs/architecture_replication_m3_v1/Chapman"
V4 = ROOT / "outputs/phase1_chapman_seed1_v4"
V7 = ROOT / "outputs/phase2_chapman_multiseed_v7"
PROTO = ROOT / "configs/FROZEN_ARCHITECTURE_M3_PROTOCOL_v1.json"
SPLIT = V4 / "SPLIT_MEMBERSHIP_v4.json"
INDICES = V7 / "bootstrap_indices.npy"
SEEDS = (104729, 130363, 155921)
ARMS = ("clean", "physics_aware")
LABELS = ("TWA", "SB", "STach")
LABEL_CODES = ("164934002", "426177001", "427084000")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MASKS = tuple(itertools.combinations(range(12), 5))
MASK_NAMES = tuple("mask5_" + "_".join(map(str, x)) for x in MASKS)
EXACT = (0, 1, 2, 3, 5)
EXACT_NAME = "mask5_0_1_2_3_5"
N_BOOT = 1000
N_PROBE = 512
BATCH = 128
ATOL = 1e-5
RTOL = 1e-4
SEMANTIC = "m3_mask5_removed_train_mean_before_channelwise_normalization_v2"

sys.path.insert(0, str(ROOT))
from src.chapman_phase1 import labels as chapman_labels, raw as chapman_raw
from src.ecg_physics import derive_display, named_reversal
from src.m3_transformer import TimePatchTransformer
from src.mask_semantics_recovery_v2 import (
    apply_mask_raw,
    infer_batches,
    prepare_clean_input,
    prepare_masked_input,
    weighted_ap_resamples,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def unit_dir(arm: str, seed: int) -> Path:
    return OLD / arm / f"seed_{seed}"


def cache_dir(arm: str, seed: int) -> Path:
    return OUT / "Chapman" / arm / f"seed_{seed}"


def parts():
    rows = json.loads(SPLIT.read_text(encoding="utf-8"))["rows"]
    out = {name: [r["record_id"] for r in rows if r["split"] == name]
           for name in ("train", "validation", "test")}
    return out, rows


def labels_for(ids) -> np.ndarray:
    return np.stack([chapman_labels(r) for r in ids]).astype(np.uint8)


def order_binding(test_ids, test_labels, split_rows) -> dict:
    test_rows = [r for r in split_rows if r["split"] == "test"]
    return {
        "test_records": len(test_ids),
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "ordered_test_rows_sha256": hash_obj([[r["record_id"], r.get("waveform_sha256")] for r in test_rows]),
        "test_labels_sha256": hashlib.sha256(np.ascontiguousarray(test_labels).tobytes()).hexdigest(),
        "split_membership_sha256": sha(SPLIT),
    }


def audit_artifacts() -> dict:
    from src.chapman_stage23 import LABELS as SOURCE_LABELS
    split, split_rows = parts()
    if tuple(SOURCE_LABELS) != LABEL_CODES:
        raise RuntimeError(f"SNOMED label code order mismatch: {SOURCE_LABELS}")
    if (len(split["train"]), len(split["validation"]), len(split["test"])) != (7163, 1021, 2050):
        raise RuntimeError("Chapman M3 split counts differ from the frozen split")
    if len(set(split["train"] + split["validation"] + split["test"])) != sum(map(len, split.values())):
        raise RuntimeError("Chapman split contains duplicate or overlapping IDs")
    binding = order_binding(split["test"], labels_for(split["test"]), split_rows)
    protocol = json.loads(PROTO.read_text(encoding="utf-8"))
    original_protocol_sha = sha(PROTO).upper()
    units = []
    for arm in ARMS:
        for seed in SEEDS:
            d = unit_dir(arm, seed)
            required = ("checkpoint.pt", "best.pt", "preprocessing.npz", "validation.npz",
                        "validation_gate.json", "evaluation_freeze.json", "evaluation_complete.json")
            files = {n: d / n for n in required}
            missing = [n for n, p in files.items() if not p.is_file()]
            if missing:
                raise RuntimeError(f"{arm}/{seed} missing: {missing}")
            ckpt = torch.load(files["checkpoint.pt"], map_location="cpu", weights_only=True)
            if (ckpt.get("dataset"), ckpt.get("arm"), int(ckpt.get("seed", -1)),
                    str(ckpt.get("protocol_sha256", "")).upper()) != ("Chapman", arm, seed, original_protocol_sha):
                raise RuntimeError(f"checkpoint identity mismatch: {arm}/{seed}")
            model = TimePatchTransformer()
            model.load_state_dict(ckpt["state_dict"], strict=True)
            del model
            with np.load(files["preprocessing.npz"], allow_pickle=False) as z:
                mean, std = z["mean"], z["std"]
                if mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                    raise RuntimeError(f"invalid preprocessing: {arm}/{seed}")
            with np.load(files["validation.npz"], allow_pickle=False) as z:
                vy, vp, thresholds = z["y"], z["p"], z["thresholds"]
                expected_y = labels_for(split["validation"])
                if vy.shape != (1021, 3) or vp.shape != vy.shape or thresholds.shape != (3,) or not np.array_equal(vy, expected_y):
                    raise RuntimeError(f"validation artifact order/shape mismatch: {arm}/{seed}")
                if not np.isfinite(vp).all() or not np.isfinite(thresholds).all():
                    raise RuntimeError(f"non-finite validation artifact: {arm}/{seed}")
                computed_thresholds = np.asarray([np.quantile(vp[vy[:, i] == 0, i], .90) for i in range(3)])
                if not np.allclose(computed_thresholds, thresholds, rtol=0, atol=1e-7):
                    raise RuntimeError(f"validation thresholds mismatch: {arm}/{seed}")
                val_metrics = {"auprc": float(average_precision_score(vy[:, 0], vp[:, 0])),
                               "auroc": float(roc_auc_score(vy[:, 0], vp[:, 0]))}
            gate = json.loads(files["validation_gate.json"].read_text(encoding="utf-8"))
            freeze = json.loads(files["evaluation_freeze.json"].read_text(encoding="utf-8"))
            complete = json.loads(files["evaluation_complete.json"].read_text(encoding="utf-8"))
            cksha = sha(files["checkpoint.pt"])
            if gate.get("decision") != "PASS" or gate.get("checkpoint_sha256") != cksha:
                raise RuntimeError(f"validation gate/checkpoint binding mismatch: {arm}/{seed}")
            if not np.isclose(gate["validation_twa_auprc"], val_metrics["auprc"], atol=1e-10) or not np.isclose(gate["validation_twa_auroc"], val_metrics["auroc"], atol=1e-10):
                raise RuntimeError(f"validation gate metric mismatch: {arm}/{seed}")
            if freeze.get("checkpoint_sha256") != cksha or freeze.get("protocol_sha256", "").upper() != original_protocol_sha or freeze.get("seed") != seed or freeze.get("arm") != arm:
                raise RuntimeError(f"evaluation freeze provenance mismatch: {arm}/{seed}")
            if complete.get("checkpoint_sha256") != cksha or complete.get("conditions") != 794 or len(complete.get("prediction_sha256", {})) != 794:
                raise RuntimeError(f"historical evaluation manifest incomplete: {arm}/{seed}")
            old_checks = {}
            for condition in ("clean", "RA_LL_reversal"):
                p = d / "test_predictions" / f"{condition}.npz"
                if not p.is_file() or sha(p) != complete["prediction_sha256"].get(condition):
                    raise RuntimeError(f"historical {condition} artifact hash mismatch: {arm}/{seed}")
                with np.load(p, allow_pickle=False) as z:
                    scores = z["p"]
                    if scores.shape != (2050, 3) or not np.isfinite(scores).all():
                        raise RuntimeError(f"invalid historical {condition} score shape: {arm}/{seed}")
                old_checks[condition] = sha(p)
            units.append({"arm": arm, "seed": seed,
                          "artifacts": {n: {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size} for n, p in files.items()},
                          "validation_metrics_recomputed": val_metrics,
                          "validation_thresholds": thresholds.tolist(),
                          "legacy_clean_ra_ll_hashes": old_checks,
                          "legacy_mask_predictions_reused": False,
                          "legacy_complete_records": len(complete["prediction_sha256"]),
                          "checkpoint_strict_state_load": True})
    idx = np.load(INDICES, allow_pickle=False)
    progress = json.loads((V7 / "BOOTSTRAP_PROGRESS.json").read_text(encoding="utf-8"))
    if idx.shape != (N_BOOT, binding["test_records"]) or not np.issubdtype(idx.dtype, np.integer) or idx.min() < 0 or idx.max() >= binding["test_records"]:
        raise RuntimeError(f"Chapman bootstrap indices incompatible: {idx.shape}")
    if progress.get("completed_replicates") != N_BOOT or progress.get("indices_sha256") != sha(INDICES):
        raise RuntimeError("Chapman bootstrap index completion/hash binding failed")
    return {"dataset": "Chapman", "historical_test_access": True,
            "analysis_status": "corrected_reanalysis_of_previously_accessed_test_set",
            "original_protocol": {"path": str(PROTO.relative_to(ROOT)), "sha256": original_protocol_sha,
                                  "stage_a_gate": protocol["stage_a_gate"], "validation_gate": protocol["validation_gate"]},
            "split_counts": {k: len(v) for k, v in split.items()},
            "labels": [{"name": n, "snomed_code": code} for n, code in zip(LABELS, LABEL_CODES)],
            "lead_order": list(LEADS), "record_order_binding": binding,
            "units": units,
            "bootstrap": {"path": str(INDICES.relative_to(ROOT)), "sha256": sha(INDICES),
                          "shape": list(idx.shape), "dtype": str(idx.dtype),
                          "source_protocol_sha256": json.loads((V7 / "FROZEN_PROTOCOL_HASH.json").read_text(encoding="utf-8"))["protocol_sha256"],
                          "same_test_split_binding": sha(SPLIT) == json.loads((V7 / "FROZEN_PROTOCOL_HASH.json").read_text(encoding="utf-8"))["split_sha256"].lower(),
                          "synchronized": True},
            "historical_old_mask_predictions": "not reused; old evaluator semantics inverted"}


def protocol_freeze() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = audit_artifacts()
    if not audit["bootstrap"]["same_test_split_binding"]:
        raise RuntimeError("candidate Chapman bootstrap indices do not bind the M3 split")
    import torch as torch_mod
    obj = {
        "recovery_version": "architecture_replication_m3_recovery_v2",
        "created_utc": now(),
        "status": "FROZEN_BEFORE_CORRECTED_INFERENCE",
        "analysis_status": audit["analysis_status"],
        "historical_test_access": True,
        "source_audit": audit,
        "evaluator": {"source": str(Path(__file__).relative_to(ROOT)), "source_sha256": sha(Path(__file__)),
                      "semantic_version": SEMANTIC, "device": torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else "CPU",
                      "torch": torch_mod.__version__, "amp": "CUDA float16 autocast; float32 input/output sigmoid",
                      "batch_size": BATCH, "clean_replay_tolerance": {"atol": ATOL, "rtol": RTOL}},
        "conditions": {"replay_or_verified_reuse": ["clean", "RA_LL_reversal"],
                       "recompute": list(MASK_NAMES), "count_per_unit": 794,
                       "mask_count": 792, "old_mask_predictions_reused": False},
        "mask": {"meaning": "listed channels are removed", "operation": "replace listed raw displayed channels with matching train-derived lead mean, then normalize", "enumeration": "lexicographic C(12,5)", "exact_mask_indices": list(EXACT), "exact_mask_leads": [LEADS[i] for i in EXACT], "unchanged_leads": [LEADS[i] for i in range(12) if i not in EXACT]},
        "metrics": {"endpoints": list(LABELS), "primary": "TWA", "secondary": ["SB", "STach"],
                    "per_condition": ["AUPRC", "AUROC", "validation_fixed_sensitivity", "validation_fixed_specificity", "ECE_15"],
                    "thresholds": "existing validation-frozen per-seed thresholds",
                    "mask_panel": "calculate metrics independently for every mask; report mean, median, p05, p25, p75, p95; never average prediction vectors",
                    "central_estimates": "metrics on full original test sample; distinguish from bootstrap means"},
        "estimands": {"seed_ra_ll_gain": "physics-aware RA_LL AUPRC minus clean-trained RA_LL AUPRC",
                      "contrast_by_arm": "mean of 792 per-mask AUPRCs minus RA_LL AUPRC",
                      "contrast_reduction": "clean-trained contrast minus physics-aware contrast, averaged across paired seeds",
                      "clean_change": "physics-aware clean AUPRC minus clean-trained clean AUPRC",
                      "mean_mask_change": "physics-aware mean-mask AUPRC minus clean-trained mean-mask AUPRC"},
        "bootstrap": {"indices": str(INDICES.relative_to(ROOT)), "sha256": sha(INDICES), "shape_axes": ["replicate", "canonical_test_record"],
                      "replicates": N_BOOT, "unit": "canonical record", "synchronized_across_models_seeds_conditions_endpoints": True,
                      "calculation": "direct AP for each condition and each mask within each resample, then average 792 AP values; pair conditions and seeds using same replicate; do not average prediction vectors",
                      "interval": "percentile 2.5/97.5"},
        "stage_m3a_gate": json.loads(PROTO.read_text(encoding="utf-8"))["stage_a_gate"],
        "cache_contract": ["protocol binding", "evaluator source hash", "semantic version", "checkpoint and preprocessing hashes", "split/record order and labels hashes", "condition and removed-index set", "lead order", "normalization", "numeric mode"],
        "provenance": "Corrected reanalysis; Chapman test has historical prior access and this is not an untouched confirmation.",
    }
    obj["binding_sha256"] = hash_obj(obj)
    atomic_json(OUT / "RECOVERY_PROTOCOL.json", obj)
    atomic_json(OUT / "INPUT_AUDIT.json", audit)
    return obj


def amend_protocol_for_legacy_replay_mismatch() -> dict:
    """Bind the predeclared full clean/RA re-inference branch after replay failure."""
    p = OUT / "RECOVERY_PROTOCOL.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("binding_sha256") != hash_obj({k: v for k, v in obj.items() if k != "binding_sha256"}):
        raise RuntimeError("cannot amend a recovery protocol with an invalid binding")
    version = "002" if obj.get("amendment") else "001"
    old_source = obj["evaluator"]["source_sha256"]
    new_source = sha(Path(__file__))
    prior = None
    if version == "001":
        mismatch = json.loads((OUT / "LEGACY_REPLAY_MISMATCH_BEFORE_REINFERENCE.json").read_text(encoding="utf-8"))
        if mismatch.get("decision") != "STOP_M3_TECHNICAL_BLOCKER" or not mismatch.get("checks") or any(q.get("allclose") for q in mismatch["checks"]):
            raise RuntimeError("no preserved clean/RA replay failure to amend")
        reason = "Legacy evaluation_complete binds prediction file hashes and checkpoint but not evaluator source or canonical test record order; frozen clean/RA probe exceeded predeclared tolerance in all 12 checks. Re-infer these conditions on the frozen test order into the versioned recovery root."
    else:
        prior_path = OUT / "RECOVERY_PROTOCOL_AMENDMENT_001.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("amendment_binding_sha256") != hash_obj({k: v for k, v in prior.items() if k != "amendment_binding_sha256"}):
            raise RuntimeError("cannot chain from invalid prior amendment")
        reason = "Avoid recomputing the already frozen test labels for every one of 4,752 cache lookups; use the protocol-bound label and split hashes directly. This changes cache lookup cost only, not condition data or model input."
    amendment = {"amendment": version, "prior_protocol_binding_sha256": obj["binding_sha256"],
                 "prior_evaluator_source_sha256": old_source, "runner_source_sha256": new_source,
                 "reason": reason,
                 "method_changes": False, "scientific_thresholds_changed": False,
                 "clean_replay_tolerance_changed": False, "mask_semantics_changed": False,
                 "legacy_mask_cache_reused": False}
    if prior:
        amendment["prior_amendment_binding_sha256"] = prior["amendment_binding_sha256"]
        amendment["prior_amendment_path"] = "RECOVERY_PROTOCOL_AMENDMENT_001.json"
    else:
        amendment["preserved_failed_probe"] = "LEGACY_REPLAY_MISMATCH_BEFORE_REINFERENCE.json"
    amendment["amendment_binding_sha256"] = hash_obj(amendment)
    atomic_json(OUT / f"RECOVERY_PROTOCOL_FREEZE_{version}.json", obj)
    atomic_json(OUT / f"RECOVERY_PROTOCOL_AMENDMENT_{version}.json", amendment)
    obj["evaluator"]["source_sha256"] = new_source
    obj["amendment"] = {"path": f"RECOVERY_PROTOCOL_AMENDMENT_{version}.json",
                        "sha256": amendment["amendment_binding_sha256"]}
    obj["binding_sha256"] = hash_obj({k: v for k, v in obj.items() if k != "binding_sha256"})
    atomic_json(p, obj)
    return obj


def read_protocol() -> dict:
    p = OUT / "RECOVERY_PROTOCOL.json"
    if not p.is_file():
        raise RuntimeError("freeze recovery protocol before pilot/inference")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("binding_sha256") != hash_obj({k: v for k, v in obj.items() if k != "binding_sha256"}):
        raise RuntimeError("recovery protocol binding hash mismatch")
    if obj["evaluator"]["source_sha256"] != sha(Path(__file__)):
        raise RuntimeError("evaluator changed after protocol freeze")
    if obj.get("amendment"):
        a = json.loads((OUT / obj["amendment"]["path"]).read_text(encoding="utf-8"))
        if a.get("amendment_binding_sha256") != hash_obj({k: v for k, v in a.items() if k != "amendment_binding_sha256"}):
            raise RuntimeError("recovery protocol amendment hash mismatch")
        if a.get("runner_source_sha256") != sha(Path(__file__)) or a.get("amendment_binding_sha256") != obj["amendment"]["sha256"]:
            raise RuntimeError("recovery protocol amendment does not bind this evaluator")
        if a.get("prior_amendment_path"):
            prior = json.loads((OUT / a["prior_amendment_path"]).read_text(encoding="utf-8"))
            if prior.get("amendment_binding_sha256") != a.get("prior_amendment_binding_sha256"):
                raise RuntimeError("recovery protocol amendment chain mismatch")
    audit_artifacts()
    return obj


def load_model(arm: str, seed: int):
    d = unit_dir(arm, seed)
    with np.load(d / "preprocessing.npz", allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    model = TimePatchTransformer().cuda()
    ckpt = torch.load(d / "checkpoint.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    with np.load(d / "validation.npz", allow_pickle=False) as z:
        thresholds = z["thresholds"].astype(np.float64)
    return model, mean, std, thresholds


def test_rows_and_data():
    split, all_rows = parts()
    y = labels_for(split["test"])
    return split["test"], y, all_rows


def load_raw(ids) -> np.ndarray:
    x = np.stack([chapman_raw(r) for r in ids]).astype(np.float32, copy=False)
    if x.shape != (len(ids), 12, 1000) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid Chapman raw waveform matrix {x.shape}")
    return x


def reversal(x: np.ndarray) -> np.ndarray:
    independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
    return derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)


def condition_binding(arm: str, seed: int, condition: str, protocol: dict, indices=None) -> str:
    d = unit_dir(arm, seed)
    order = protocol["source_audit"]["record_order_binding"]
    bind = {"protocol": protocol["binding_sha256"], "evaluator_source": protocol["evaluator"]["source_sha256"],
            "semantic_version": SEMANTIC, "arm": arm, "seed": seed,
            "checkpoint_sha256": sha(d / "checkpoint.pt"), "preprocessing_sha256": sha(d / "preprocessing.npz"),
            "split_sha256": order["split_membership_sha256"], "record_order_sha256": order["test_ids_sha256"],
            "labels_sha256": order["test_labels_sha256"],
            "condition": condition, "removed_indices": list(indices) if indices is not None else [],
            "lead_order": list(LEADS), "normalization": "replace before channel-wise train mean/std normalization",
            "numeric_mode": "float32 inputs, CUDA float16 autocast, float32 sigmoid outputs", "batch_size": BATCH}
    return hash_obj(bind)


def cache_path(arm: str, seed: int, condition: str) -> Path:
    return cache_dir(arm, seed) / "predictions" / f"{condition}.npz"


def load_cache(arm: str, seed: int, condition: str, protocol: dict, indices=None):
    p = cache_path(arm, seed, condition)
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        if set(z.files) != {"p", "binding_sha256", "shape_axes"}:
            raise RuntimeError(f"cache schema mismatch: {p}")
        a = z["p"]
        if str(z["binding_sha256"].item()) != condition_binding(arm, seed, condition, protocol, indices):
            raise RuntimeError(f"stale cache binding: {p}")
        if str(z["shape_axes"].item()) != "canonical_test_record,endpoint" or a.shape != (2050, 3) or not np.isfinite(a).all():
            raise RuntimeError(f"invalid cache shape/values: {p}")
    return a.astype(np.float32, copy=False)


def save_cache(arm: str, seed: int, condition: str, scores: np.ndarray, protocol: dict, indices=None):
    if scores.shape != (2050, 3) or not np.isfinite(scores).all():
        raise RuntimeError("invalid scores refusing cache")
    p = cache_path(arm, seed, condition)
    atomic_npz(p, p=scores.astype(np.float32), binding_sha256=np.asarray(condition_binding(arm, seed, condition, protocol, indices)),
               shape_axes=np.asarray("canonical_test_record,endpoint"))
    return p


def pilot(protocol: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no corrected inference started")
    ids, y, _ = test_rows_and_data()
    probe = np.linspace(0, len(ids) - 1, N_PROBE, dtype=np.int64)
    x = load_raw([ids[i] for i in probe])
    checks = []
    times = []
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, std, _ = load_model(arm, seed)
            for cond, rawx in (("clean", x), ("RA_LL_reversal", reversal(x))):
                prepared = prepare_clean_input(rawx, mean, std)
                t0 = time.perf_counter()
                new = infer_batches(model, prepared, "cuda", BATCH, True)
                elapsed = time.perf_counter() - t0
                old_file = unit_dir(arm, seed) / "test_predictions" / f"{cond}.npz"
                with np.load(old_file, allow_pickle=False) as z:
                    old = z["p"][probe]
                diff = np.abs(new - old)
                ok = bool(np.allclose(new, old, rtol=RTOL, atol=ATOL))
                checks.append({"arm": arm, "seed": seed, "condition": cond, "records": len(probe),
                               "allclose": ok, "atol": ATOL, "rtol": RTOL,
                               "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
                               "old_score_sha256": sha(old_file)})
                times.append(elapsed)
            del model
            torch.cuda.empty_cache()
    # Smoke every transformation with the actual preprocessing and model input adapter.
    arm, seed = "clean", SEEDS[0]
    model, mean, std, _ = load_model(arm, seed)
    mask_probe = (EXACT, MASKS[0], MASKS[100], MASKS[396], MASKS[-1])
    mask_times = []
    for m in mask_probe:
        raw_masked = apply_mask_raw(x, mean, m)
        retained = [i for i in range(12) if i not in m]
        if not np.array_equal(raw_masked[:, retained], x[:, retained]):
            raise RuntimeError("pilot failed: unselected channels changed")
        norm = prepare_masked_input(x, mean, std, m)
        if not np.array_equal(norm[:, list(m)], np.zeros_like(norm[:, list(m)])):
            raise RuntimeError("pilot failed: train-mean sentinel is not zero")
        t0 = time.perf_counter()
        p = infer_batches(model, norm, "cuda", BATCH, True)
        mask_times.append(time.perf_counter() - t0)
        if p.shape != (N_PROBE, 3) or not np.isfinite(p).all():
            raise RuntimeError("pilot failed: model received invalid masked input/output")
    del model
    torch.cuda.empty_cache()
    # Validated historical clean/RA_LL scores are copied under the new cache binding.
    # If any deterministic replay differs, re-infer that full condition instead.
    mismatched = {(x["arm"], x["seed"], x["condition"]) for x in checks if not x["allclose"]}
    full_ids = ids if mismatched else None
    estimated_mask_hours = float(np.mean(mask_times) * len(MASKS) * len(ARMS) * len(SEEDS) / 3600)
    if full_ids is not None:
        full_raw = load_raw(full_ids)
    else:
        full_raw = None
    reused = []
    for arm in ARMS:
        for seed in SEEDS:
            model = None
            for cond in ("clean", "RA_LL_reversal"):
                old_path = unit_dir(arm, seed) / "test_predictions" / f"{cond}.npz"
                with np.load(old_path, allow_pickle=False) as z:
                    scores = z["p"].astype(np.float32, copy=False)
                if (arm, seed, cond) in mismatched:
                    if model is None:
                        model, mean, std, _ = load_model(arm, seed)
                    transformed = full_raw if cond == "clean" else reversal(full_raw)
                    scores = infer_batches(model, prepare_clean_input(transformed, mean, std), "cuda", BATCH, True)
                save_cache(arm, seed, cond, scores, protocol)
                reused.append({"arm": arm, "seed": seed, "condition": cond,
                               "source": "full re-inference" if (arm, seed, cond) in mismatched else "legacy prediction after 512-record replay",
                               "cache_sha256": sha(cache_path(arm, seed, cond))})
            if model is not None:
                del model
                torch.cuda.empty_cache()
    report = {"decision": "PASS_WITH_FULL_REINFERENCE" if mismatched else "PASS",
              "completed_utc": now(), "protocol_binding_sha256": protocol["binding_sha256"],
              "probe_records": N_PROBE, "clean_ra_ll_checks": checks, "verified_reuse_or_regeneration": reused,
              "legacy_score_replay_failed": bool(mismatched),
              "legacy_scores_reused": not bool(mismatched),
              "mask_smoke_indices": [list(m) for m in mask_probe], "mask_smoke_inference_seconds": mask_times,
              "estimated_792_mask_inference_hours_all_six_units": estimated_mask_hours,
              "bootstrap_indices_shape_and_sha256": protocol["bootstrap"],
              "historical_test_access": True, "reserved_for_no_new_confirmation": True}
    atomic_json(OUT / "PILOT_REPORT.json", report)
    return report


def status(stage: str, **fields) -> None:
    atomic_json(OUT / "WORKFLOW_STATUS.json", {"stage": stage, "updated_utc": now(), **fields})
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def run_inference(protocol: dict) -> None:
    ids, _, _ = test_rows_and_data()
    x = load_raw(ids)
    status("MASK_INFERENCE", total_units=6, completed_units=0, total_masks=6 * len(MASKS), completed_masks=0)
    complete_masks = 0
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, std, _ = load_model(arm, seed)
            udir = cache_dir(arm, seed)
            udir.mkdir(parents=True, exist_ok=True)
            for i, m in enumerate(MASKS):
                name = MASK_NAMES[i]
                p = load_cache(arm, seed, name, protocol, m)
                if p is None:
                    inp = prepare_masked_input(x, mean, std, m)
                    p = infer_batches(model, inp, "cuda", BATCH, True)
                    save_cache(arm, seed, name, p, protocol, m)
                complete_masks += 1
                if complete_masks % 10 == 0 or complete_masks == 6 * len(MASKS):
                    status("MASK_INFERENCE", arm=arm, seed=seed, condition=name,
                           completed_units=sum(1 for a in ARMS for s in SEEDS if all(cache_path(a, s, n).is_file() for n in ("clean", "RA_LL_reversal", *MASK_NAMES))),
                           total_units=6, completed_masks=complete_masks, total_masks=6 * len(MASKS))
            del model
            torch.cuda.empty_cache()
    status("MASK_INFERENCE_COMPLETE", total_units=6, completed_units=6,
           completed_masks=complete_masks, total_masks=6 * len(MASKS))


def load_all_scores(arm: str, seed: int, protocol: dict) -> dict[str, np.ndarray]:
    out = {}
    for name in ("clean", "RA_LL_reversal", *MASK_NAMES):
        indices = MASKS[MASK_NAMES.index(name)] if name.startswith("mask5_") else None
        a = load_cache(arm, seed, name, protocol, indices)
        if a is None:
            raise RuntimeError(f"missing prediction cache: {arm}/{seed}/{name}")
        out[name] = a
    return out


def ece15(y, p) -> float:
    result = 0.0
    for b in range(15):
        lo, hi = b / 15, (b + 1) / 15
        chosen = (p >= lo) & (p <= hi if b == 14 else p < hi)
        if chosen.any():
            result += float(chosen.mean()) * abs(float(y[chosen].mean()) - float(p[chosen].mean()))
    return result


def metric(y, p, threshold) -> dict:
    pos, neg = y == 1, y == 0
    pred = p >= threshold
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "sensitivity": float(pred[pos].mean()), "specificity": float((~pred[neg]).mean()), "ece_15": ece15(y, p)}


def summarize(protocol: dict) -> tuple[list[dict], list[dict], list[dict]]:
    ids, y, _ = test_rows_and_data()
    rows, distributions, contrasts = [], [], []
    score_bank = {}
    for arm in ARMS:
        for seed in SEEDS:
            score = load_all_scores(arm, seed, protocol)
            score_bank[(arm, seed)] = score
            with np.load(unit_dir(arm, seed) / "validation.npz", allow_pickle=False) as z:
                threshold = z["thresholds"]
            for li, endpoint in enumerate(LABELS):
                mask_aps = []
                for condition, pred in score.items():
                    z = metric(y[:, li], pred[:, li], float(threshold[li]))
                    rows.append({"arm": arm, "seed": seed, "endpoint": endpoint, "condition": condition, **z,
                                 "threshold": float(threshold[li]), "records": len(ids)})
                    if condition.startswith("mask5_"):
                        mask_aps.append(z["auprc"])
                ma = np.asarray(mask_aps, dtype=np.float64)
                find = lambda name: next(q for q in rows if q["arm"] == arm and q["seed"] == seed and q["endpoint"] == endpoint and q["condition"] == name)
                cl, ra, ex = find("clean"), find("RA_LL_reversal"), find(EXACT_NAME)
                distributions.append({"arm": arm, "seed": seed, "endpoint": endpoint,
                                      "mask_mean_auprc": float(ma.mean()), "mask_median_auprc": float(np.median(ma)),
                                      "mask_p05_auprc": float(np.quantile(ma, .05)), "mask_p25_auprc": float(np.quantile(ma, .25)),
                                      "mask_p75_auprc": float(np.quantile(ma, .75)), "mask_p95_auprc": float(np.quantile(ma, .95)),
                                      "exact_mask_auprc": ex["auprc"], "ra_ll_auprc": ra["auprc"], "masks": len(ma)})
                contrasts.append({"arm": arm, "seed": seed, "endpoint": endpoint,
                                  "clean_auprc": cl["auprc"], "ra_ll_auprc": ra["auprc"],
                                  "exact_mask_auprc": ex["auprc"], "mean_mask_auprc": float(ma.mean()),
                                  "exact_mask_minus_ra_ll": ex["auprc"] - ra["auprc"],
                                  "mean_mask_minus_ra_ll": float(ma.mean()) - ra["auprc"],
                                  "ra_ll_loss_from_clean": cl["auprc"] - ra["auprc"]})
    atomic_csv(OUT / "CONDITION_RESULTS_BY_SEED.csv", rows)
    atomic_csv(OUT / "MASK_DISTRIBUTION_SUMMARY.csv", distributions)
    atomic_csv(OUT / "PRIMARY_CONTRASTS.csv", contrasts)
    atomic_json(OUT / "POINT_ESTIMATES.json", {"data": "full original test sample", "test_accessed_historically": True,
                                                "rows": len(rows), "masks_per_unit_endpoint": 792,
                                                "estimates": contrasts})
    return rows, distributions, contrasts


def bootstrap(protocol: dict) -> list[dict]:
    ids, y, _ = test_rows_and_data()
    ix = np.load(INDICES, allow_pickle=False).astype(np.int64)
    if ix.shape != (N_BOOT, len(ids)):
        raise RuntimeError("bootstrap matrix axes or dimensions changed")
    counts = np.stack([np.bincount(q, minlength=len(ids)) for q in ix]).astype(np.uint16)
    tasks = {}
    progress_path = OUT / "BOOTSTRAP_PROGRESS.json"
    for arm in ARMS:
        for seed in SEEDS:
            sc = load_all_scores(arm, seed, protocol)
            for li, endpoint in enumerate(LABELS):
                target = OUT / "bootstrap_tasks" / f"{arm}_seed_{seed}_{endpoint}.npz"
                source_hashes = {n: sha(cache_path(arm, seed, n)) for n in ("clean", "RA_LL_reversal", *MASK_NAMES)}
                binding = hash_obj({"protocol": protocol["binding_sha256"], "indices": sha(INDICES),
                                    "arm": arm, "seed": seed, "endpoint": endpoint, "prediction_hashes": source_hashes})
                if target.is_file():
                    with np.load(target, allow_pickle=False) as z:
                        if str(z["binding_sha256"].item()) != binding:
                            raise RuntimeError(f"bootstrap cache binding mismatch: {target}")
                        ma, cl, ra = z["mask_ap"], z["clean_ap"], z["ra_ll_ap"]
                    if ma.shape != (N_BOOT, len(MASKS)) or cl.shape != (N_BOOT,) or ra.shape != (N_BOOT,):
                        raise RuntimeError(f"bootstrap cache shape mismatch: {target}")
                else:
                    ma = weighted_ap_resamples(y[:, li], np.stack([sc[n][:, li] for n in MASK_NAMES]), counts).T
                    cl = weighted_ap_resamples(y[:, li], sc["clean"][:, li][None, :], counts)[0]
                    ra = weighted_ap_resamples(y[:, li], sc["RA_LL_reversal"][:, li][None, :], counts)[0]
                    atomic_npz(target, mask_ap=ma.astype(np.float32), clean_ap=cl.astype(np.float32),
                               ra_ll_ap=ra.astype(np.float32), binding_sha256=np.asarray(binding),
                               shape_axes=np.asarray("replicate,mask_index"))
                tasks[(arm, seed, endpoint)] = (ma, cl, ra)
                atomic_json(progress_path, {"completed_tasks": len(tasks), "total_tasks": 18,
                                            "indices_sha256": sha(INDICES), "indices_shape_axes": ["replicate", "canonical_test_record"],
                                            "updated_utc": now()})
    # Paired replicate-level estimands across the three original seeds.
    replicate_rows = []
    for li, endpoint in enumerate(LABELS):
        for r in range(N_BOOT):
            fields = {"baseline_ra_ll": [], "physics_ra_ll": [], "baseline_clean": [], "physics_clean": [],
                      "baseline_mean_mask": [], "physics_mean_mask": [], "baseline_contrast": [], "physics_contrast": []}
            for seed in SEEDS:
                bma, bcl, bra = tasks[("clean", seed, endpoint)]
                pma, pcl, pra = tasks[("physics_aware", seed, endpoint)]
                bm, pm = float(bma[r].mean()), float(pma[r].mean())
                fields["baseline_ra_ll"].append(float(bra[r])); fields["physics_ra_ll"].append(float(pra[r]))
                fields["baseline_clean"].append(float(bcl[r])); fields["physics_clean"].append(float(pcl[r]))
                fields["baseline_mean_mask"].append(bm); fields["physics_mean_mask"].append(pm)
                fields["baseline_contrast"].append(bm - float(bra[r])); fields["physics_contrast"].append(pm - float(pra[r]))
            bmcon = float(np.mean(fields["baseline_contrast"])); pmcon = float(np.mean(fields["physics_contrast"]))
            replicate_rows.append({"replicate": r, "endpoint": endpoint,
                                   "physics_minus_clean_ra_ll": float(np.mean(fields["physics_ra_ll"]) - np.mean(fields["baseline_ra_ll"])),
                                   "contrast_reduction": bmcon - pmcon, "physics_mean_mask_minus_ra_ll": pmcon,
                                   "physics_minus_clean_clean": float(np.mean(fields["physics_clean"]) - np.mean(fields["baseline_clean"])),
                                   "physics_minus_clean_mean_mask": float(np.mean(fields["physics_mean_mask"]) - np.mean(fields["baseline_mean_mask"]))})
    atomic_csv(OUT / "BOOTSTRAP_REPLICATES.csv", replicate_rows)
    summary = []
    for endpoint in LABELS:
        keys = [k for k in replicate_rows[0] if k not in ("replicate", "endpoint")]
        for key in keys:
            v = np.asarray([r[key] for r in replicate_rows if r["endpoint"] == endpoint])
            summary.append({"endpoint": endpoint, "estimand": key, "bootstrap_mean": float(v.mean()),
                            "lower95": float(np.quantile(v, .025)), "upper95": float(np.quantile(v, .975)),
                            "replicates": N_BOOT, "unit": "canonical record", "synchronized": True})
    atomic_csv(OUT / "BOOTSTRAP_SUMMARY.csv", summary)
    return summary


def make_decision(protocol: dict, distributions: list[dict], contrasts: list[dict], bootstrap_rows: list[dict]) -> dict:
    g = protocol["stage_m3a_gate"]
    t = lambda arm, seed: next(r for r in contrasts if r["arm"] == arm and r["seed"] == seed and r["endpoint"] == "TWA")
    d = lambda arm, seed: next(r for r in distributions if r["arm"] == arm and r["seed"] == seed and r["endpoint"] == "TWA")
    b = {r["estimand"]: r for r in bootstrap_rows if r["endpoint"] == "TWA"}
    ra_gain = [t("physics_aware", s)["ra_ll_auprc"] - t("clean", s)["ra_ll_auprc"] for s in SEEDS]
    clean_change = [t("physics_aware", s)["clean_auprc"] - t("clean", s)["clean_auprc"] for s in SEEDS]
    mask_change = [d("physics_aware", s)["mask_mean_auprc"] - d("clean", s)["mask_mean_auprc"] for s in SEEDS]
    base_contrast = [t("clean", s)["mean_mask_minus_ra_ll"] for s in SEEDS]
    physics_contrast = [t("physics_aware", s)["mean_mask_minus_ra_ll"] for s in SEEDS]
    reduction_point = float(np.mean(np.asarray(base_contrast) - np.asarray(physics_contrast)))
    all_integrity = all(u["checkpoint_strict_state_load"] and u["legacy_complete_records"] == 794 for u in protocol["source_audit"]["units"])
    checks = {
        "ra_ll_twa_improvement_each_seed_min_0_15": {"pass": all(x >= .15 for x in ra_gain), "values": ra_gain, "threshold": .15},
        "aggregate_contrast_reduction_min_0_15": {"pass": reduction_point >= .15, "value": reduction_point, "threshold": .15},
        "physics_contrast_upper95_max_0_15": {"pass": b["physics_mean_mask_minus_ra_ll"]["upper95"] <= .15,
                                               "upper95": b["physics_mean_mask_minus_ra_ll"]["upper95"], "threshold": .15},
        "mean_clean_decline_max_0_02": {"pass": float(np.mean(clean_change)) >= -.02, "values": clean_change, "threshold": -.02},
        "any_clean_decline_max_0_04": {"pass": all(x >= -.04 for x in clean_change), "values": clean_change, "threshold": -.04},
        "mean_mask_decline_max_0_02": {"pass": float(np.mean(mask_change)) >= -.02, "values": mask_change, "threshold": -.02},
        "any_mask_decline_max_0_04": {"pass": all(x >= -.04 for x in mask_change), "values": mask_change, "threshold": -.04},
        "all_integrity_gates": {"pass": all_integrity, "value": all_integrity},
    }
    gate_pass = all(v["pass"] for v in checks.values())
    decision = "PASS_M3_ARCHITECTURE_REPLICATION" if gate_pass else ("FAIL_M3_ARCHITECTURE_SPECIFIC" if all_integrity else "STOP_M3A_VALIDATION_GATE")
    result = {"decision": decision, "analysis_status": "corrected_reanalysis_of_previously_accessed_Chapman_test; not untouched confirmation",
              "protocol_binding_sha256": protocol["binding_sha256"], "gate_checks": checks,
              "per_seed": {str(s): {"clean_ra_ll_auprc": t("clean", s)["ra_ll_auprc"],
                                     "physics_ra_ll_auprc": t("physics_aware", s)["ra_ll_auprc"],
                                     "ra_ll_gain": ra_gain[i], "clean_change": clean_change[i],
                                     "mean_mask_change": mask_change[i], "clean_contrast": base_contrast[i],
                                     "physics_contrast": physics_contrast[i]} for i, s in enumerate(SEEDS)},
              "mean_contrast_reduction_point_estimate": reduction_point,
              "bootstrap_twa": b, "frozen_thresholds_changed": False,
              "ningbo_stage_b_authorized_by_gate": gate_pass,
              "if_failed": "Do not proceed to Ningbo; retain corrected Chapman results and report exact failed conditions."}
    atomic_json(OUT / "conclusions_m3.json", result)
    return result


def report(protocol: dict, decision: dict) -> None:
    c = decision["per_seed"]
    lines = ["# M3 corrected Chapman recovery", "",
             "This is a corrected reanalysis of an already accessed test set, not an untouched confirmation.",
             f"Protocol binding: `{protocol['binding_sha256']}`.", f"Decision: **{decision['decision']}**.", "",
             "## Stage M3A gates", "", "| Criterion | Result | Threshold |", "|---|---:|---:|"]
    for name, item in decision["gate_checks"].items():
        value = item.get("values", item.get("value", item.get("upper95")))
        lines.append(f"| {name} | {'PASS' if item['pass'] else 'FAIL'} ({value}) | {item.get('threshold', 'integrity')} |")
    lines += ["", "## Per-seed endpoints", "", "| Seed | clean RA–LL TWA AP | physics-aware RA–LL TWA AP | RA–LL gain | clean change | mean-mask change | clean contrast | physics contrast |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        q = c[str(seed)]
        lines.append("| " + str(seed) + " | " + " | ".join(f"{q[k]:.4f}" for k in ("clean_ra_ll_auprc", "physics_ra_ll_auprc", "ra_ll_gain", "clean_change", "mean_mask_change", "clean_contrast", "physics_contrast")) + " |")
    lines += ["", "Corrected mask metrics, all per-condition metrics, per-mask distributions, paired bootstrap replicates and provenance are in the CSV/JSON artifacts in this directory.",
              "Old inverted mask predictions were not reused. No Chapman retraining occurred.",
              "Ningbo is not evaluated unless all frozen Stage M3A gates pass.", ""]
    (OUT / "CHAPMAN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(command: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if command == "freeze":
        p = protocol_freeze()
        print("PROTOCOL_FROZEN", p["binding_sha256"], flush=True)
        return
    protocol = read_protocol()
    if command == "audit":
        print(json.dumps(protocol["source_audit"], indent=2))
    elif command == "pilot":
        pilot(protocol)
    elif command == "run":
        p = json.loads((OUT / "PILOT_REPORT.json").read_text(encoding="utf-8"))
        if p.get("decision") not in {"PASS", "PASS_WITH_FULL_REINFERENCE"}:
            raise RuntimeError("engineering pilot has not passed")
        run_inference(protocol)
        _, distributions, contrasts = summarize(protocol)
        bs = bootstrap(protocol)
        decision = make_decision(protocol, distributions, contrasts, bs)
        report(protocol, decision)
        status("COMPLETE", decision=decision["decision"], ningbo_authorized=decision["ningbo_stage_b_authorized_by_gate"])
    elif command == "summarize":
        _, distributions, contrasts = summarize(protocol)
        bs = bootstrap(protocol)
        decision = make_decision(protocol, distributions, contrasts, bs)
        report(protocol, decision)
        print(decision["decision"])
    else:
        raise ValueError(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "freeze", "amend", "pilot", "run", "summarize"))
    command = parser.parse_args().command
    if command == "amend":
        amended = amend_protocol_for_legacy_replay_mismatch()
        print("PROTOCOL_AMENDED", amended["binding_sha256"], flush=True)
    else:
        run(command)
