"""Versioned L1 recovery of Ningbo five-lead mask evaluation.

This module is intentionally separate from frozen historical workflows.
All masks mean channels replaced by the per-lead train mean before
normalization. The old mask prediction caches are never reused.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/mask_semantics_recovery_v2"
BASE = ROOT / "outputs/ningbo_confirmation_v2"
M2 = ROOT / "outputs/mitigation_m2_ningbo_v1"
BASE_PROTO = ROOT / "configs/FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json"
M2_PROTO = ROOT / "configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json"
SEEDS = (104729, 130363, 155921)
MODES = ("baseline_v2", "physics_aware_m2")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MASKS = tuple(itertools.combinations(range(12), 5))
MASK_NAMES = tuple("mask5_" + "_".join(map(str, q)) for q in MASKS)
EXACT_INDICES = (0, 1, 2, 3, 5)
EXACT_NAME = "mask5_0_1_2_3_5"
LABELS = ("TWA", "SB", "STach")
N_TEST = 6981
N_BOOT = 1000
SEMANTIC_VERSION = "mask5_removed_train_mean_pre_normalization_v2"
INFERENCE_BATCH = 128
PROBE_RECORDS = 512
CLEAN_TOL_ATOL = 1e-5
CLEAN_TOL_RTOL = 1e-4
BOOT_SEED = 20260829  # reused immutable protocol bootstrap index file


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def mode_root(mode: str) -> Path:
    if mode == "baseline_v2":
        return BASE
    if mode == "physics_aware_m2":
        return M2
    raise ValueError(mode)


def unit_dir(mode: str, seed: int) -> Path:
    return mode_root(mode) / f"seed_{seed}"


def recovery_unit_dir(mode: str, seed: int) -> Path:
    return OUT / mode / f"seed_{seed}"


def masks() -> tuple[tuple[int, ...], ...]:
    return MASKS


def apply_mask_raw(x: np.ndarray, mean: np.ndarray, indices) -> np.ndarray:
    """Replace selected channel(s) with their raw-unit training means.

    Accepts a single (12,time) example or a batch (batch,12,time). Returns a
    copy, preserving all unselected values byte-for-byte.
    """
    a = np.asarray(x)
    mu = np.asarray(mean)
    idx = tuple(int(i) for i in indices)
    if a.ndim not in (2, 3) or a.shape[-2] != 12:
        raise ValueError(f"expected (12,time) or (batch,12,time), got {a.shape}")
    if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= 12 for i in idx):
        raise ValueError(f"invalid mask indices: {idx}")
    if mu.shape != (12,) or not np.isfinite(mu).all():
        raise ValueError("mean must be a finite vector of length 12")
    out = np.array(a, copy=True)
    if a.ndim == 2:
        out[np.asarray(idx), :] = mu[np.asarray(idx), None]
    else:
        out[:, np.asarray(idx), :] = mu[np.asarray(idx)][None, :, None]
    return out


def normalize_raw(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    mu = np.asarray(mean, dtype=np.float32)
    sig = np.asarray(std, dtype=np.float32)
    if a.ndim not in (2, 3) or a.shape[-2] != 12 or mu.shape != (12,) or sig.shape != (12,):
        raise ValueError("invalid signal or preprocessing shape")
    if not np.isfinite(mu).all() or not np.isfinite(sig).all() or np.any(sig <= 0):
        raise ValueError("invalid normalization statistics")
    if a.ndim == 2:
        return ((a.astype(np.float32, copy=False) - mu[:, None]) / sig[:, None]).astype(np.float32)
    return ((a.astype(np.float32, copy=False) - mu[None, :, None]) / sig[None, :, None]).astype(np.float32)


def prepare_masked_input(x: np.ndarray, mean: np.ndarray, std: np.ndarray, indices) -> np.ndarray:
    """The exact transformation invoked by recovery inference."""
    return normalize_raw(apply_mask_raw(x, mean, indices), mean, std)


def prepare_clean_input(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return normalize_raw(x, mean, std)


def infer_batches(model, inputs: np.ndarray, device: str, batch_size: int = INFERENCE_BATCH, amp: bool = True) -> np.ndarray:
    """Forward adapter used by new inference and captured by integration tests."""
    import torch

    if inputs.ndim != 3 or inputs.shape[1] != 12 or not np.isfinite(inputs).all():
        raise ValueError("model inputs must be finite (batch,12,time) float arrays")
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(np.ascontiguousarray(inputs[start : start + batch_size], dtype=np.float32)).to(device)
            if amp and str(device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = torch.sigmoid(model(tensor))
            else:
                pred = torch.sigmoid(model(tensor))
            outputs.append(pred.float().cpu().numpy())
    out = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise FloatingPointError("model produced non-finite scores")
    return out


def load_parts():
    """Reconstruct Ningbo v2 test records in frozen split-row order."""
    records = json.loads((BASE / "CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    split = json.loads((BASE / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by_id = {r["record_id"]: r for r in records}
    test_rows = [r for r in split if r["split"] == "test"]
    test = [by_id[r["record_id"]] for r in test_rows]
    if len(test) != N_TEST:
        raise RuntimeError(f"test record count mismatch: {len(test)}")
    for sr, r in zip(test_rows, test):
        if sr["labels"] != r["labels"] or sr["waveform_sha256"] != r["waveform_sha256"]:
            raise RuntimeError("split membership row does not match canonical record")
    ids = [r["record_id"] for r in test]
    labels = np.asarray([r["labels"] for r in test], dtype=np.int8)
    if labels.shape != (N_TEST, 3) or not np.isin(labels, (0, 1)).all():
        raise RuntimeError(f"test label shape/value mismatch: {labels.shape}")
    return test, labels, test_rows


def order_bindings(test, labels, rows) -> dict:
    compact = [[r["record_id"], r["waveform_sha256"], list(r["labels"])] for r in rows]
    return {
        "record_count": len(test),
        "ordered_test_membership_sha256": hash_json(compact),
        "ordered_test_labels_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "split_membership_file_sha256": sha256(BASE / "SPLIT_MEMBERSHIP.json"),
        "canonical_records_file_sha256": sha256(BASE / "CANONICAL_RECORDS.json"),
    }


def artifact_entry(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_original_bindings() -> dict:
    """Verify frozen protocol links and all saved prediction score files."""
    base_cfg = json.loads(BASE_PROTO.read_text(encoding="utf-8"))
    m2_cfg = json.loads(M2_PROTO.read_text(encoding="utf-8"))
    if sha256(BASE_PROTO).upper() != "1FE829E4357B4EFDE0E5F227BAF5BF94E8D22D12AF0959A38CEE12CCE1F40D71":
        raise RuntimeError("Ningbo v2 frozen protocol hash mismatch")
    if sha256(M2_PROTO).upper() != "2D7B605AF175EF6D5B59B223A9EA37BB3F7D675C2C1AC3E27FFF23FCD43E96B8":
        raise RuntimeError("M2 frozen protocol hash mismatch")
    for binding in m2_cfg["immutable_bindings"].values():
        p = ROOT / binding["path"]
        if not p.is_file() or sha256(p).upper() != binding["sha256"]:
            raise RuntimeError(f"M2 immutable protocol binding mismatch: {binding['path']}")
    if sha256(BASE / "bootstrap_indices.npy").upper() != m2_cfg["immutable_bindings"]["bootstrap_indices"]["sha256"]:
        raise RuntimeError("bootstrap index binding mismatch")

    model_artifacts = {}
    for mode in MODES:
        model_artifacts[mode] = {}
        for seed in SEEDS:
            d = unit_dir(mode, seed)
            ckpt = d / "checkpoint.pt"
            prep = d / "preprocessing.npz"
            val = d / "validation.npz"
            gate = d / ("baseline.json" if mode == "baseline_v2" else "validation_gate.json")
            complete = d / "evaluation_complete.json"
            freeze = d / ("EVALUATION_FREEZE.json" if mode == "baseline_v2" else "evaluation_freeze.json")
            for p in (ckpt, prep, val, gate, complete, freeze):
                if not p.is_file():
                    raise RuntimeError(f"missing required saved artifact: {p.relative_to(ROOT)}")
            g = json.loads(gate.read_text(encoding="utf-8"))
            e = json.loads(complete.read_text(encoding="utf-8"))
            f = json.loads(freeze.read_text(encoding="utf-8"))
            if g.get("decision") != "PASS" or g.get("checkpoint_sha256", "").lower() != sha256(ckpt).lower():
                raise RuntimeError(f"validation/checkpoint binding failed for {mode} seed {seed}")
            if e.get("conditions") != 794 or len(e.get("prediction_sha256", {})) != 794:
                raise RuntimeError(f"condition completion manifest incomplete for {mode} seed {seed}")
            if e.get("checkpoint_sha256", "").lower() != sha256(ckpt).lower() or f.get("checkpoint_sha256", "").lower() != sha256(ckpt).lower():
                raise RuntimeError(f"evaluation/checkpoint binding failed for {mode} seed {seed}")
            pred_root = d / "test_predictions"
            old_score_hashes = {}
            for condition in ("clean", "RA_LL_reversal"):
                p = pred_root / f"{condition}.npz"
                digest = e["prediction_sha256"].get(condition)
                if not p.is_file() or not digest or sha256(p).lower() != str(digest).lower():
                    raise RuntimeError(f"saved {condition} scores fail hash check: {p}")
                with np.load(p, allow_pickle=False) as z:
                    if set(z.files) != {"p"} or z["p"].shape != (N_TEST, 3) or not np.isfinite(z["p"]).all():
                        raise RuntimeError(f"invalid saved {condition} scores: {p}")
                old_score_hashes[condition] = sha256(p)
            # Saved historical masks are deliberately excluded from reuse.
            with np.load(val, allow_pickle=False) as z:
                if set(("y", "p", "thresholds")) - set(z.files):
                    raise RuntimeError(f"validation artifact arrays missing: {val}")
                if z["y"].shape != (3488, 3) or z["p"].shape != (3488, 3) or z["thresholds"].shape != (3,):
                    raise RuntimeError(f"validation artifact shape mismatch: {val}")
                if not np.isfinite(z["p"]).all() or not np.isfinite(z["thresholds"]).all():
                    raise RuntimeError(f"validation artifact non-finite: {val}")
            model_artifacts[mode][str(seed)] = {
                "checkpoint": artifact_entry(ckpt), "preprocessing": artifact_entry(prep),
                "validation": artifact_entry(val), "validation_gate": artifact_entry(gate),
                "evaluation_freeze": artifact_entry(freeze), "evaluation_complete": artifact_entry(complete),
                "reused_clean_ra_ll_score_hashes": old_score_hashes,
            }
    # Architecture and physical-reversal implementation are also bound.
    from src.chapman_phase1 import ECGResNetSmall
    if ECGResNetSmall is None:
        raise RuntimeError("ECGResNetSmall import failed")
    return {"model_artifacts": model_artifacts, "baseline_protocol": artifact_entry(BASE_PROTO),
            "m2_protocol": artifact_entry(M2_PROTO),
            "architecture_source": artifact_entry(ROOT / "src/chapman_phase1.py"),
            "physics_source": artifact_entry(ROOT / "src/ecg_physics.py"),
            "bootstrap_indices": artifact_entry(BASE / "bootstrap_indices.npy"),
            "base_protocol_json": base_cfg, "m2_protocol_json": m2_cfg}


def write_recovery_protocol() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    test, labels, rows = load_parts()
    original = verify_original_bindings()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no inference started")
    protocol = {
        "recovery_version": "ningbo_mask_semantics_recovery_v2",
        "semantic_version": SEMANTIC_VERSION,
        "created_utc": now(),
        "analysis_status": "corrected_reanalysis_after_historical_test_access",
        "historical_test_access": True,
        "statement": "This is a corrected reanalysis of an already accessed Ningbo test set; it is not an untouched confirmation.",
        "models": {"families": ["ECGResNet-small baseline v2", "ECGResNet-small physics-aware M2"], "seeds": list(SEEDS), "count": 6,
                   "model_artifacts": original["model_artifacts"], "architecture_source": original["architecture_source"]},
        "source_protocols": {"ningbo_confirmation_v2": original["baseline_protocol"], "mitigation_m2_ningbo_v1": original["m2_protocol"]},
        "data": {"source": "existing outputs/ningbo_confirmation_v2 canonical records/split and existing data/challenge2021_v103 waveforms",
                 "counts": {"canonical": 34899, "train": 24430, "validation": 3488, "test": len(test)},
                 **order_bindings(test, labels, rows), "labels_column_order": list(LABELS),
                 "raw_record_ids_in_protocol": False},
        "lead_order": list(LEADS),
        "mask_definition": {"meaning": "indices are the five channels to remove", "enumeration": "lexicographic C(12,5)",
                            "count": len(MASKS), "exact_ra_ll": {"indices": list(EXACT_INDICES), "leads": [LEADS[i] for i in EXACT_INDICES]},
                            "operation": "replace selected raw/display channels with matching train-derived per-lead mean; preserve other seven; then normalize"},
        "evaluator": {"source": "src/mask_semantics_recovery_v2.py", "source_sha256": sha256(Path(__file__),),
                      "semantic_version": SEMANTIC_VERSION, "normalization": "(x - train_mean[channel]) / train_std[channel] after replacement",
                      "model_input": "float32", "device": torch.cuda.get_device_name(0), "autocast": "float16 on CUDA",
                      "batch_size": INFERENCE_BATCH, "clean_score_tolerance": {"atol": CLEAN_TOL_ATOL, "rtol": CLEAN_TOL_RTOL}},
        "conditions": {"reuse": ["clean", "RA_LL_reversal"], "recompute": list(MASK_NAMES), "full_per_model": 794,
                       "old_mask_prediction_reuse": False, "full_model_units": [f"{m}/seed_{s}" for m in MODES for s in SEEDS]},
        "metrics": {"labels": list(LABELS), "metrics": ["AUPRC", "AUROC", "sensitivity", "specificity", "ECE_15"],
                    "thresholds": "existing per-seed validation-frozen thresholds; no recalibration",
                    "ece": "15 equal-width probability bins [k/15,(k+1)/15), with 1.0 included in final bin; weighted absolute calibration gap",
                    "panel_summary": ["mean", "median", "p05", "p25", "p75", "p95"],
                    "primary_contrasts": ["exact-mask AUPRC - RA_LL AUPRC", "mean of 792 per-mask AUPRCs - RA_LL AUPRC",
                                          "M2 minus baseline change in both contrasts", "M2 minus baseline RA_LL AUPRC", "M2 minus baseline clean AUPRC"]},
        "bootstrap": {"path": original["bootstrap_indices"]["path"], "sha256": original["bootstrap_indices"]["sha256"],
                      "shape": [N_BOOT, N_TEST], "replicates": N_BOOT, "unit": "canonical record",
                      "same_resamples_for_all_seeds_conditions_endpoints_arms": True,
                      "estimand": "per resample and condition calculate average_precision_score, average the 792 mask metrics; do not average prediction vectors",
                      "resume": "atomic per arm/seed/endpoint bootstrap mask-score file; each file bound to prediction hashes and bootstrap index hash"},
        "cache_policy": {"new_root": str(OUT.relative_to(ROOT)), "old_mask_caches_reused": False,
                          "bindings": ["semantic_version", "evaluator source hash", "checkpoint hash", "preprocessing hash", "protocol hash",
                                       "test membership and labels hashes", "condition and removed indices", "lead order", "normalization", "numerical inference mode"],
                          "clean_ra_ll_reuse": "verify original completion-manifest hashes, dimensions, finite values, checkpoint links, then compare new deterministic probe inference on 512 evenly spaced test indices at frozen tolerance"},
        "provenance_note": "Prior Ningbo v2 and M2 test evaluations are already in project history; this recovery does not claim test_accessed_before_freeze=false for project history.",
    }
    protocol["binding_sha256"] = hash_json(protocol)
    atomic_json(OUT / "RECOVERY_PROTOCOL.json", protocol)
    return protocol


def load_model(mode: str, seed: int):
    import torch
    from src.chapman_phase1 import ECGResNetSmall
    d = unit_dir(mode, seed)
    z = np.load(d / "preprocessing.npz", allow_pickle=False)
    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    if mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError(f"invalid preprocessing: {mode} seed {seed}")
    model = ECGResNetSmall().cuda()
    checkpoint = torch.load(d / "checkpoint.pt", map_location="cuda", weights_only=True)
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint missing state_dict: {mode} seed {seed}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with np.load(d / "validation.npz", allow_pickle=False) as v:
        thresholds = v["thresholds"].astype(np.float64)
    return model, mean, std, thresholds


def old_scores(mode: str, seed: int, condition: str) -> np.ndarray:
    p = unit_dir(mode, seed) / "test_predictions" / f"{condition}.npz"
    with np.load(p, allow_pickle=False) as z:
        a = z["p"].astype(np.float32, copy=False)
    if a.shape != (N_TEST, 3) or not np.isfinite(a).all():
        raise RuntimeError(f"invalid saved scores: {p}")
    return a


def condition_binding(mode: str, seed: int, condition: str, protocol: dict,
                      evaluator_source_sha256: str | None = None) -> dict:
    d = unit_dir(mode, seed)
    return {"recovery_protocol_sha256": protocol["binding_sha256"], "semantic_version": SEMANTIC_VERSION,
            "evaluator_source_sha256": evaluator_source_sha256 or sha256(Path(__file__)), "mode": mode, "seed": seed,
            "checkpoint_sha256": sha256(d / "checkpoint.pt"), "preprocessing_sha256": sha256(d / "preprocessing.npz"),
            "protocol_sha256": sha256(BASE_PROTO if mode == "baseline_v2" else M2_PROTO),
            "test_membership_sha256": protocol["data"]["ordered_test_membership_sha256"],
            "test_labels_sha256": protocol["data"]["ordered_test_labels_sha256"],
            "condition": condition, "lead_order": list(LEADS),
            "normalization": "float32 (x - train_mean[channel])/train_std[channel] after condition transformation",
            "inference_numeric_mode": "float32 input, CUDA float16 autocast, float32 sigmoid output",
            "batch_size": INFERENCE_BATCH}


def save_reinferred_condition(mode: str, seed: int, condition: str, p: np.ndarray, protocol: dict) -> Path:
    root = recovery_unit_dir(mode, seed) / "verified_conditions"
    target = root / f"{condition}.npz"
    binding = condition_binding(mode, seed, condition, protocol)
    atomic_npz(target, p=np.asarray(p, np.float32), binding_sha256=np.asarray(hash_json(binding)),
               condition=np.asarray(condition), record_count=np.asarray(len(p), np.int32))
    return target


def verified_or_old_scores(mode: str, seed: int, condition: str, protocol: dict) -> np.ndarray:
    path = recovery_unit_dir(mode, seed) / "verified_conditions" / f"{condition}.npz"
    if not path.is_file():
        return old_scores(mode, seed, condition)
    accepted_sources = {sha256(Path(__file__))}
    amendment_path = OUT / "RECOVERY_PROTOCOL_AMENDMENT_001.json"
    if amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        accepted_sources.add(amendment["runner_source_sha256"])
    bindings = {hash_json(condition_binding(mode, seed, condition, protocol, source))
                for source in accepted_sources}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"p", "binding_sha256", "condition", "record_count"}:
            raise RuntimeError(f"verified condition cache schema mismatch: {path}")
        p = z["p"]
        if str(z["binding_sha256"].item()) not in bindings or str(z["condition"].item()) != condition:
            raise RuntimeError(f"verified condition cache binding mismatch: {path}")
        if int(z["record_count"].item()) != N_TEST or p.shape != (N_TEST, 3) or not np.isfinite(p).all():
            raise RuntimeError(f"verified condition cache invalid: {path}")
    return p.astype(np.float32, copy=False)


def load_all_raw(test: list[dict]) -> np.ndarray:
    from src import ningbo_confirmation_v1 as b
    start = time.monotonic()
    x = np.stack([b.raw(r) for r in test]).astype(np.float32, copy=False)
    if x.shape[0] != N_TEST or x.shape[1] != 12 or not np.isfinite(x).all():
        raise RuntimeError(f"invalid raw test matrix shape/values: {x.shape}")
    atomic_json(OUT / "DATA_LOAD_REPORT.json", {"records": len(x), "shape": list(x.shape), "dtype": str(x.dtype),
                                                 "bytes_in_memory": int(x.nbytes), "seconds": time.monotonic() - start,
                                                 "source": "existing waveforms only", "raw_record_ids_written": False})
    return x


def generate_reversal(x: np.ndarray) -> np.ndarray:
    from src.ecg_physics import derive_display, named_reversal
    independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
    return derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)


def compare_old_score(mode: str, seed: int, condition: str, predicted: np.ndarray, sample_indices: np.ndarray) -> dict:
    old = old_scores(mode, seed, condition)[sample_indices]
    new = predicted
    delta = np.abs(old - new)
    close = np.allclose(old, new, rtol=CLEAN_TOL_RTOL, atol=CLEAN_TOL_ATOL)
    return {"mode": mode, "seed": seed, "condition": condition, "n_compared": len(sample_indices),
            "atol": CLEAN_TOL_ATOL, "rtol": CLEAN_TOL_RTOL, "allclose": bool(close),
            "max_abs_difference": float(delta.max()), "mean_abs_difference": float(delta.mean()),
            "old_scores_sha256": sha256(unit_dir(mode, seed) / "test_predictions" / f"{condition}.npz")}


def run_pilot(protocol: dict) -> dict:
    import torch
    test, labels, rows = load_parts()
    n = len(test)
    sample_indices = np.linspace(0, n - 1, PROBE_RECORDS, dtype=np.int64)
    from src import ningbo_confirmation_v1 as b
    t0 = time.monotonic()
    raw_probe = np.stack([b.raw(test[i]) for i in sample_indices]).astype(np.float32)
    probe_load_seconds = time.monotonic() - t0
    clean_checks = []
    model_info = {}
    for mode in MODES:
        for seed in SEEDS:
            model, mean, std, _ = load_model(mode, seed)
            clean_input = prepare_clean_input(raw_probe, mean, std)
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            clean_p = infer_batches(model, clean_input, "cuda", batch_size=INFERENCE_BATCH, amp=True)
            clean_seconds = time.monotonic() - t
            clean_checks.append(compare_old_score(mode, seed, "clean", clean_p, sample_indices))
            rev_raw = generate_reversal(raw_probe)
            rev_p = infer_batches(model, prepare_clean_input(rev_raw, mean, std), "cuda", INFERENCE_BATCH, True)
            clean_checks.append(compare_old_score(mode, seed, "RA_LL_reversal", rev_p, sample_indices))
            model_info[f"{mode}/seed_{seed}"] = {"clean_probe_seconds": clean_seconds,
                                                  "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                                  "mean": mean, "std": std}
            del model
            torch.cuda.empty_cache()
    failed_units = sorted({(r["mode"], r["seed"]) for r in clean_checks if not r["allclose"]})
    # Load the existing held-out data once. Any score cache that misses the
    # frozen numeric tolerance is regenerated for that model/condition.
    x = load_all_raw(test)
    regenerated = []
    for mode, seed in failed_units:
        model, mean, std, _ = load_model(mode, seed)
        raw_clean = prepare_clean_input(x, mean, std)
        clean_full = infer_batches(model, raw_clean, "cuda", INFERENCE_BATCH, True)
        raw_ra = generate_reversal(x)
        ra_full = infer_batches(model, prepare_clean_input(raw_ra, mean, std), "cuda", INFERENCE_BATCH, True)
        for condition, scores in (("clean", clean_full), ("RA_LL_reversal", ra_full)):
            path = save_reinferred_condition(mode, seed, condition, scores, protocol)
            regenerated.append({"mode": mode, "seed": seed, "condition": condition,
                                "prediction_sha256": sha256(path), "records": len(scores),
                                "source": "full re-inference with existing checkpoint and frozen test order"})
        del model
        torch.cuda.empty_cache()

    # Full-array pilot timing uses the first baseline model, with five fixed masks.
    model, mean, std, _ = load_model("baseline_v2", SEEDS[0])
    pilot_indices = (EXACT_INDICES, MASKS[0], MASKS[100], MASKS[396], MASKS[-1])
    pilot_rows = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for idx in pilot_indices:
        name = "mask5_" + "_".join(map(str, idx))
        inp = prepare_masked_input(x, mean, std, idx)
        t = time.monotonic()
        p = infer_batches(model, inp, "cuda", INFERENCE_BATCH, True)
        elapsed = time.monotonic() - t
        save_prediction("baseline_v2", SEEDS[0], name, p, idx, protocol)
        pilot_rows.append({"condition": name, "removed_indices": list(idx), "inference_seconds": elapsed,
                           "score_shape": list(p.shape), "finite": bool(np.isfinite(p).all())})
    five_mask_seconds = time.monotonic() - start
    # Optional forensic replay: compare current inverted exact-mask behavior to old cache.
    inverted_raw = np.where(np.isin(np.arange(12), np.asarray(EXACT_INDICES))[None, :, None],
                            x, mean[None, :, None])
    inverted_p = infer_batches(model, prepare_clean_input(inverted_raw, mean, std), "cuda", INFERENCE_BATCH, True)
    old_exact = old_scores("baseline_v2", SEEDS[0], EXACT_NAME)
    inverted_diff = np.abs(old_exact - inverted_p)
    inverted_match = bool(np.allclose(old_exact, inverted_p, rtol=CLEAN_TOL_RTOL, atol=CLEAN_TOL_ATOL))
    corrected_exact = load_or_validate_prediction("baseline_v2", SEEDS[0], EXACT_NAME, EXACT_INDICES, protocol)
    if corrected_exact is None:
        raise RuntimeError("corrected exact-mask pilot cache missing")
    atomic_npz(OUT / "pilot_forensic_baseline_seed_104729.npz", historical_exact=old_exact,
              inverted_replay=inverted_p, corrected_exact=corrected_exact)
    full_mask_sec = five_mask_seconds / len(pilot_rows) * len(MASKS)
    full_inference_sec = full_mask_sec * (len(MODES) * len(SEEDS))

    # Benchmark exact bootstrap mask mean on existing saved score panel; this is timing only.
    idx = np.load(BASE / "bootstrap_indices.npy", allow_pickle=False)
    if idx.shape != (N_BOOT, N_TEST):
        raise RuntimeError(f"bootstrap index matrix shape mismatch: {idx.shape}")
    counts = bootstrap_counts(idx[:20])
    historical = {name: old_scores("baseline_v2", SEEDS[0], name)[:, 0] for name in MASK_NAMES}
    y0 = labels[:, 0]
    stat_start = time.monotonic()
    benchmark = weighted_ap_resamples(y0, np.stack([historical[name] for name in MASK_NAMES]), counts)
    stat_seconds = time.monotonic() - stat_start
    estimated_stat_seconds = stat_seconds * (N_BOOT / 20) * len(MODES) * len(SEEDS) * len(LABELS)
    pilot = {"decision": "PASS", "completed_utc": now(), "protocol_binding_sha256": protocol["binding_sha256"],
             "probe_records": PROBE_RECORDS, "clean_ra_ll_score_checks": clean_checks,
             "clean_ra_ll_reuse_verified": True, "regenerated_score_units": regenerated,
             "evaluator_source_sha256": sha256(Path(__file__)), "pilot_model": "baseline_v2 seed 104729",
             "pilot_masks": pilot_rows, "five_mask_inference_seconds": five_mask_seconds,
             "mean_mask_inference_seconds": five_mask_seconds / len(pilot_rows),
             "estimated_full_mask_inference_seconds": full_inference_sec,
             "estimated_full_mask_inference_hours": full_inference_sec / 3600,
             "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
             "test_raw_array_shape": list(x.shape), "test_raw_array_bytes": int(x.nbytes),
             "forensic_inverted_replay": {"mode": "baseline_v2", "seed": 104729, "condition": EXACT_NAME,
                                           "matches_historical_score_cache": inverted_match,
                                           "max_abs_difference": float(inverted_diff.max()),
                                           "historical_semantics_supported_for_this_condition_only": inverted_match},
             "bootstrap_statistics_benchmark": {"resamples": 20, "masks": 792, "endpoint": "TWA",
                                                  "seconds": stat_seconds, "estimated_6_models_3_endpoints_1000_replicates_seconds": estimated_stat_seconds,
                                                  "method": "weighted exact AP with score-tie grouping; see integration tests"},
             "raw_waveform_load_seconds_for_probe": probe_load_seconds,
             "no_training_or_new_data": True}
    atomic_json(OUT / "CLEAN_REUSE_CHECKS.json", {"decision": "PASS_WITH_REGENERATION" if regenerated else "PASS",
                                                   "checks": clean_checks, "regenerated_full_conditions": regenerated,
                                                   "tolerance": {"atol": CLEAN_TOL_ATOL, "rtol": CLEAN_TOL_RTOL}})
    atomic_json(OUT / "PILOT_REPORT.json", pilot)
    report_lines = ["# Ningbo mask recovery pilot", "", "Decision: **PASS**", "",
                    f"Frozen 512-record clean and RA–LL probe used atol={CLEAN_TOL_ATOL:g}, rtol={CLEAN_TOL_RTOL:g}. {len(clean_checks)-sum(not q['allclose'] for q in clean_checks)}/{len(clean_checks)} saved-score checks passed. Any failed model unit had its full clean and RA–LL conditions re-inferred; tolerance was unchanged.",
                    f"Five-mask inference took {five_mask_seconds:.2f} s (mean {five_mask_seconds/5:.2f} s/mask). Estimated 4,752-mask inference: {full_inference_sec/3600:.2f} h.",
                    f"Exact weighted-bootstrap benchmark: {stat_seconds:.2f} s for 20 resamples × 792 masks × one endpoint; projected full stats: {estimated_stat_seconds/3600:.2f} h.",
                    f"Peak CUDA allocation: {pilot['cuda_peak_allocated_bytes']/1024**2:.1f} MiB; raw test matrix: {x.nbytes/1024**2:.1f} MiB.",
                    f"Baseline seed 104729 forensic inverted exact replay matches old cache: **{inverted_match}** (max absolute score difference {inverted_diff.max():.3g}); this applies only to this condition.",
                    "", "No model training or data acquisition occurred. Full recovery proceeds using the new mask semantic version."]
    (OUT / "PILOT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return pilot


def binding_for(mode: str, seed: int, condition: str, indices, protocol: dict,
                evaluator_source_sha256: str | None = None) -> dict:
    d = unit_dir(mode, seed)
    return {"recovery_protocol_sha256": protocol["binding_sha256"], "semantic_version": SEMANTIC_VERSION,
            "evaluator_source_sha256": evaluator_source_sha256 or sha256(Path(__file__)), "mode": mode, "seed": seed,
            "checkpoint_sha256": sha256(d / "checkpoint.pt"),
            "preprocessing_sha256": sha256(d / "preprocessing.npz"),
            "protocol_sha256": sha256(BASE_PROTO if mode == "baseline_v2" else M2_PROTO),
            "test_membership_sha256": protocol["data"]["ordered_test_membership_sha256"],
            "test_labels_sha256": protocol["data"]["ordered_test_labels_sha256"],
            "condition": condition, "removed_indices": list(indices), "lead_order": list(LEADS),
            "normalization": "raw selected channels=train mean; then float32 (x-mean)/std",
            "inference_numeric_mode": "float32 input, CUDA float16 autocast, float32 sigmoid output",
            "batch_size": INFERENCE_BATCH}


def save_prediction(mode: str, seed: int, condition: str, p: np.ndarray, indices, protocol: dict) -> Path:
    target = recovery_unit_dir(mode, seed) / "test_predictions" / f"{condition}.npz"
    bind = binding_for(mode, seed, condition, indices, protocol)
    atomic_npz(target, p=np.asarray(p, np.float32), binding_sha256=np.asarray(hash_json(bind)),
               condition=np.asarray(condition), record_count=np.asarray(len(p), np.int32))
    return target


def load_or_validate_prediction(mode: str, seed: int, condition: str, indices, protocol: dict) -> np.ndarray | None:
    path = recovery_unit_dir(mode, seed) / "test_predictions" / f"{condition}.npz"
    if not path.exists():
        return None
    accepted_sources = {sha256(Path(__file__))}
    amendment_path = OUT / "RECOVERY_PROTOCOL_AMENDMENT_001.json"
    if amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        accepted_sources.add(amendment["runner_source_sha256"])
    expected = {hash_json(binding_for(mode, seed, condition, indices, protocol, source))
                for source in accepted_sources}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"p", "binding_sha256", "condition", "record_count"}:
            raise RuntimeError(f"recovery cache schema mismatch: {path}")
        p = z["p"]
        if str(z["binding_sha256"].item()) not in expected or str(z["condition"].item()) != condition:
            raise RuntimeError(f"recovery cache binding mismatch: {path}")
        if int(z["record_count"].item()) != N_TEST or p.shape != (N_TEST, 3) or not np.isfinite(p).all():
            raise RuntimeError(f"recovery cache dimensions/finite check failed: {path}")
    return p.astype(np.float32, copy=False)


def load_recovery_cache(mode: str, seed: int, protocol: dict, include_reused=True) -> dict[str, np.ndarray]:
    out = {}
    if include_reused:
        out["clean"] = verified_or_old_scores(mode, seed, "clean", protocol)
        out["RA_LL_reversal"] = verified_or_old_scores(mode, seed, "RA_LL_reversal", protocol)
    for indices, name in zip(MASKS, MASK_NAMES):
        p = load_or_validate_prediction(mode, seed, name, indices, protocol)
        if p is None:
            raise RuntimeError(f"missing recovered mask cache: {mode} seed {seed} {name}")
        out[name] = p
    return out


def run_model_unit(mode: str, seed: int, x: np.ndarray, protocol: dict) -> dict:
    import torch
    d = recovery_unit_dir(mode, seed)
    pred_dir = d / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model, mean, std, _ = load_model(mode, seed)
    started = time.monotonic()
    done = 0
    for ordinal, (indices, name) in enumerate(zip(MASKS, MASK_NAMES), start=1):
        old = load_or_validate_prediction(mode, seed, name, indices, protocol)
        if old is not None:
            done += 1
            continue
        inp = prepare_masked_input(x, mean, std, indices)
        p = infer_batches(model, inp, "cuda", INFERENCE_BATCH, True)
        path = save_prediction(mode, seed, name, p, indices, protocol)
        done += 1
        status = {"stage": "MASK_INFERENCE", "mode": mode, "seed": seed,
                  "completed_conditions_this_unit": done, "total_conditions_this_unit": len(MASKS),
                  "global_completed_conditions": count_completed(protocol), "global_total_conditions": 6 * len(MASKS),
                  "last_condition": name, "last_prediction_sha256": sha256(path), "elapsed_seconds": time.monotonic() - started,
                  "updated_utc": now(), "pid": os.getpid(), "protocol_binding_sha256": protocol["binding_sha256"]}
        atomic_json(OUT / "WORKFLOW_STATUS.json", status)
        atomic_json(OUT / "resume_state.json", {"status": "RUNNING", "last_complete_unit": f"{mode}/seed_{seed}",
                                                   "last_complete_condition": name, "protocol_binding_sha256": protocol["binding_sha256"],
                                                   "updated_utc": status["updated_utc"]})
        if done % 10 == 0 or ordinal == len(MASKS):
            print(json.dumps(status, sort_keys=True), flush=True)
    complete = {name: sha256(pred_dir / f"{name}.npz") for name in MASK_NAMES}
    manifest = {"status": "PASS", "mode": mode, "seed": seed, "semantic_version": SEMANTIC_VERSION,
                "protocol_binding_sha256": protocol["binding_sha256"], "evaluator_source_sha256": sha256(Path(__file__)),
                "checkpoint_sha256": sha256(unit_dir(mode, seed) / "checkpoint.pt"),
                "preprocessing_sha256": sha256(unit_dir(mode, seed) / "preprocessing.npz"),
                "test_membership_sha256": protocol["data"]["ordered_test_membership_sha256"],
                "test_labels_sha256": protocol["data"]["ordered_test_labels_sha256"], "conditions": len(complete),
                "prediction_sha256": complete, "completed_utc": now(), "elapsed_seconds": time.monotonic() - started}
    atomic_json(d / "evaluation_complete.json", manifest)
    del model
    torch.cuda.empty_cache()
    return manifest


def count_completed(protocol: dict) -> int:
    count = 0
    for mode in MODES:
        for seed in SEEDS:
            pred_dir = recovery_unit_dir(mode, seed) / "test_predictions"
            if pred_dir.is_dir():
                present = {p.stem for p in pred_dir.glob("mask5_*.npz")}
                count += len(present.intersection(MASK_NAMES))
    return count


def validate_complete_units(protocol: dict) -> dict:
    checked = 0
    for mode in MODES:
        for seed in SEEDS:
            d = recovery_unit_dir(mode, seed)
            manifest_path = d / "evaluation_complete.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"unit incomplete: {mode} seed {seed}")
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if m.get("protocol_binding_sha256") != protocol["binding_sha256"] or m.get("semantic_version") != SEMANTIC_VERSION:
                raise RuntimeError(f"unit manifest binding mismatch: {mode} seed {seed}")
            if len(m.get("prediction_sha256", {})) != 792:
                raise RuntimeError(f"unit condition count mismatch: {mode} seed {seed}")
            for indices, name in zip(MASKS, MASK_NAMES):
                p = d / "test_predictions" / f"{name}.npz"
                if not p.is_file() or sha256(p) != m["prediction_sha256"].get(name):
                    raise RuntimeError(f"unit prediction hash mismatch: {mode} seed {seed} {name}")
                load_or_validate_prediction(mode, seed, name, indices, protocol)
                checked += 1
    return {"decision": "PASS", "units": 6, "mask_predictions_verified": checked}


def ece_15(y: np.ndarray, p: np.ndarray) -> float:
    out = 0.0
    for k in range(15):
        lo, hi = k / 15, (k + 1) / 15
        m = (p >= lo) & (p <= hi if k == 14 else p < hi)
        if np.any(m):
            out += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return out


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    q = p >= threshold
    pos = y == 1
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "sensitivity": float(q[pos].mean()), "specificity": float((~q[~pos]).mean()), "ece_15": ece_15(y, p)}


def summarize_predictions(protocol: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    test, y, _ = load_parts()
    condition_rows, exact_rows, distributions, contrasts = [], [], [], []
    from sklearn.metrics import average_precision_score
    for mode in MODES:
        for seed in SEEDS:
            scores = load_recovery_cache(mode, seed, protocol)
            with np.load(unit_dir(mode, seed) / "validation.npz", allow_pickle=False) as z:
                thresholds = z["thresholds"].astype(np.float64)
            for li, label in enumerate(LABELS):
                aps = []
                for condition, p_all in scores.items():
                    p = p_all[:, li]
                    row = {"mode": mode, "seed": seed, "condition": condition, "endpoint": label,
                           **metric_row(y[:, li], p, thresholds[li]), "threshold": float(thresholds[li]), "n": len(y)}
                    condition_rows.append(row)
                    if condition.startswith("mask5_"):
                        aps.append(row["auprc"])
                mask_ap = np.asarray(aps, dtype=np.float64)
                exact = next(r for r in condition_rows if r["mode"] == mode and r["seed"] == seed and r["condition"] == EXACT_NAME and r["endpoint"] == label)
                ra = next(r for r in condition_rows if r["mode"] == mode and r["seed"] == seed and r["condition"] == "RA_LL_reversal" and r["endpoint"] == label)
                clean = next(r for r in condition_rows if r["mode"] == mode and r["seed"] == seed and r["condition"] == "clean" and r["endpoint"] == label)
                distribution = {"mode": mode, "seed": seed, "endpoint": label,
                                "mask_mean_auprc": float(mask_ap.mean()), "mask_median_auprc": float(np.median(mask_ap)),
                                "mask_p05_auprc": float(np.quantile(mask_ap, .05)), "mask_p25_auprc": float(np.quantile(mask_ap, .25)),
                                "mask_p75_auprc": float(np.quantile(mask_ap, .75)), "mask_p95_auprc": float(np.quantile(mask_ap, .95)),
                                "ra_ll_auprc": ra["auprc"], "ra_ll_percentile_le": float(np.mean(mask_ap <= ra["auprc"])),
                                "exact_mask_auprc": exact["auprc"], "exact_mask_percentile_le": float(np.mean(mask_ap <= exact["auprc"])),
                                "n_masks": int(len(mask_ap))}
                distributions.append(distribution)
                exact_rows.append({"mode": mode, "seed": seed, "endpoint": label, "exact_condition": EXACT_NAME,
                                   "exact_mask_auprc": exact["auprc"], "ra_ll_auprc": ra["auprc"],
                                   "exact_minus_ra_ll": exact["auprc"] - ra["auprc"],
                                   "mask_mean_auprc": distribution["mask_mean_auprc"],
                                   "mean_mask_minus_ra_ll": distribution["mask_mean_auprc"] - ra["auprc"]})
                contrasts.append({"mode": mode, "seed": seed, "endpoint": label, "clean_auprc": clean["auprc"],
                                  "ra_ll_auprc": ra["auprc"], "exact_mask_auprc": exact["auprc"],
                                  "exact_mask_minus_ra_ll": exact["auprc"] - ra["auprc"],
                                  "mean_mask_auprc": distribution["mask_mean_auprc"],
                                  "mean_mask_minus_ra_ll": distribution["mask_mean_auprc"] - ra["auprc"],
                                  "ra_ll_loss_from_clean": clean["auprc"] - ra["auprc"]})
    atomic_csv(OUT / "CONDITION_RESULTS_BY_SEED.csv", condition_rows)
    atomic_csv(OUT / "EXACT_MASK_COMPARISON.csv", exact_rows)
    atomic_csv(OUT / "MASK_DISTRIBUTION_SUMMARY.csv", distributions)
    atomic_csv(OUT / "PRIMARY_CONTRASTS.csv", contrasts)
    return condition_rows, exact_rows, distributions, contrasts


def bootstrap_counts(indices: np.ndarray) -> np.ndarray:
    ix = np.asarray(indices)
    if ix.ndim != 2 or np.any(ix < 0) or np.any(ix >= N_TEST):
        raise ValueError("invalid bootstrap indices")
    counts = np.zeros((len(ix), N_TEST), dtype=np.uint16)
    for i, row in enumerate(ix):
        counts[i] = np.bincount(row, minlength=N_TEST).astype(np.uint16)
    return counts


def weighted_ap_resamples(y: np.ndarray, score_matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact sklearn AP for bootstrap resamples, using integer multiplicity weights.

    score_matrix is (conditions, records); counts is (resamples, records).
    Score ties are grouped exactly as threshold points in sklearn's AP.
    """
    y = np.asarray(y, dtype=np.uint8)
    scores = np.asarray(score_matrix, dtype=np.float32)
    counts = np.asarray(counts)
    if scores.ndim != 2 or counts.ndim != 2 or scores.shape[1] != len(y) or counts.shape[1] != len(y):
        raise ValueError("bootstrap AP shapes do not align")
    out = np.empty((scores.shape[0], counts.shape[0]), dtype=np.float32)
    for ci, score in enumerate(scores):
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
        w = counts[:, order]
        group_total = np.add.reduceat(w, starts, axis=1)
        group_positive = np.add.reduceat(w * y[order][None, :], starts, axis=1)
        cumulative_total = np.cumsum(group_total, axis=1, dtype=np.int64)
        cumulative_positive = np.cumsum(group_positive, axis=1, dtype=np.int64)
        positives = cumulative_positive[:, -1]
        if np.any(positives == 0):
            raise ValueError("bootstrap replicate contains no positive examples")
        precision = cumulative_positive / np.maximum(cumulative_total, 1)
        ap = np.sum(group_positive * precision, axis=1) / positives
        out[ci] = ap.astype(np.float32)
    return out


def bootstrap_task_path(mode: str, seed: int, label: str) -> Path:
    return OUT / "bootstrap_tasks" / f"{mode}_seed_{seed}_{label}.npz"


def run_bootstrap_task(mode: str, seed: int, li: int, labels: np.ndarray, idx: np.ndarray,
                       scores: dict[str, np.ndarray], protocol: dict) -> dict:
    label = LABELS[li]
    task_path = bootstrap_task_path(mode, seed, label)
    prediction_hashes = {}
    for n in ("clean", "RA_LL_reversal", *MASK_NAMES):
        if n in ("clean", "RA_LL_reversal"):
            recovered = recovery_unit_dir(mode, seed) / "verified_conditions" / f"{n}.npz"
            prediction_path = recovered if recovered.exists() else unit_dir(mode, seed) / "test_predictions" / f"{n}.npz"
        else:
            prediction_path = recovery_unit_dir(mode, seed) / "test_predictions" / f"{n}.npz"
        prediction_hashes[n] = sha256(prediction_path)
    bind = {"protocol": protocol["binding_sha256"], "mode": mode, "seed": seed, "endpoint": label,
            "bootstrap_indices_sha256": sha256(BASE / "bootstrap_indices.npy"), "prediction_hashes": prediction_hashes}
    bind_hash = hash_json(bind)
    if task_path.is_file():
        with np.load(task_path, allow_pickle=False) as z:
            if str(z["binding_sha256"].item()) != bind_hash:
                raise RuntimeError(f"bootstrap task stale binding: {task_path}")
            if z["mask_ap"].shape != (N_BOOT, len(MASKS)) or z["clean_ap"].shape != (N_BOOT,) or z["ra_ll_ap"].shape != (N_BOOT,) or z["exact_ap"].shape != (N_BOOT,):
                raise RuntimeError(f"bootstrap task shape mismatch: {task_path}")
            if not all(np.isfinite(z[k]).all() for k in ("mask_ap", "clean_ap", "ra_ll_ap", "exact_ap")):
                raise RuntimeError(f"bootstrap task has non-finite values: {task_path}")
            return {"mode": mode, "seed": seed, "endpoint": label, "status": "REUSED_VERIFIED", "path": str(task_path.relative_to(ROOT))}
    counts = bootstrap_counts(idx)
    mask_matrix = np.stack([scores[n][:, li] for n in MASK_NAMES])
    mask_aps = weighted_ap_resamples(labels[:, li], mask_matrix, counts)
    clean_ap = weighted_ap_resamples(labels[:, li], scores["clean"][:, li][None, :], counts)[0]
    ra_ap = weighted_ap_resamples(labels[:, li], scores["RA_LL_reversal"][:, li][None, :], counts)[0]
    exact_ap = mask_aps[MASKS.index(EXACT_INDICES)]
    atomic_npz(task_path, mask_ap=mask_aps.T, clean_ap=clean_ap, ra_ll_ap=ra_ap, exact_ap=exact_ap,
               binding_sha256=np.asarray(bind_hash))
    return {"mode": mode, "seed": seed, "endpoint": label, "status": "PASS", "path": str(task_path.relative_to(ROOT)),
            "sha256": sha256(task_path)}


def run_bootstrap(protocol: dict) -> dict:
    idx = np.load(BASE / "bootstrap_indices.npy", allow_pickle=False)
    if idx.shape != (N_BOOT, N_TEST) or np.any(idx >= N_TEST):
        raise RuntimeError(f"immutable bootstrap index validation failed: {idx.shape}")
    test, y, _ = load_parts()
    tasks = []
    for mode in MODES:
        for seed in SEEDS:
            scores = load_recovery_cache(mode, seed, protocol)
            for li in range(3):
                tasks.append(run_bootstrap_task(mode, seed, li, y, idx, scores, protocol))
                atomic_json(OUT / "BOOTSTRAP_PROGRESS.json", {"completed_tasks": len(tasks), "total_tasks": 18,
                                                               "last": tasks[-1], "updated_utc": now()})
                print(json.dumps({"stage": "BOOTSTRAP", **tasks[-1], "completed_tasks": len(tasks), "total_tasks": 18}), flush=True)
    summary, replicates = summarize_bootstrap(protocol)
    atomic_csv(OUT / "BOOTSTRAP_SUMMARY.csv", summary)
    atomic_csv(OUT / "BOOTSTRAP_REPLICATES.csv", replicates)
    return {"tasks": tasks, "summary_rows": len(summary)}


def summarize_bootstrap(protocol: dict) -> tuple[list[dict], list[dict]]:
    task = {}
    for mode in MODES:
        for seed in SEEDS:
            for label in LABELS:
                p = bootstrap_task_path(mode, seed, label)
                if not p.is_file():
                    raise RuntimeError(f"missing bootstrap task: {p}")
                with np.load(p, allow_pickle=False) as z:
                    task[(mode, seed, label)] = {k: z[k] for k in ("mask_ap", "clean_ap", "ra_ll_ap", "exact_ap")}
    # For each endpoint/resample, average seed-specific paired metrics; seeds are not an ensemble.
    replicate_rows = []
    summary_rows = []
    field_names = ("baseline_exact_mask_minus_ra_ll", "m2_exact_mask_minus_ra_ll",
                   "baseline_mean_mask_minus_ra_ll", "m2_mean_mask_minus_ra_ll",
                   "m2_minus_baseline_exact_contrast", "m2_minus_baseline_mean_contrast",
                   "m2_minus_baseline_ra_ll_auprc", "m2_minus_baseline_clean_auprc",
                   "m2_minus_baseline_mean_mask_auprc")
    for label in LABELS:
        matrix = {f: [] for f in field_names}
        for r in range(N_BOOT):
            per_seed = {"baseline_exact_mask_minus_ra_ll": [], "m2_exact_mask_minus_ra_ll": [],
                        "baseline_mean_mask_minus_ra_ll": [], "m2_mean_mask_minus_ra_ll": [],
                        "m2_minus_baseline_exact_contrast": [], "m2_minus_baseline_mean_contrast": [],
                        "m2_minus_baseline_ra_ll_auprc": [], "m2_minus_baseline_clean_auprc": [],
                        "m2_minus_baseline_mean_mask_auprc": []}
            for seed in SEEDS:
                b = task[("baseline_v2", seed, label)]
                m = task[("physics_aware_m2", seed, label)]
                b_mean = float(b["mask_ap"][r].mean()); m_mean = float(m["mask_ap"][r].mean())
                be = float(b["exact_ap"][r] - b["ra_ll_ap"][r]); me = float(m["exact_ap"][r] - m["ra_ll_ap"][r])
                bm = float(b_mean - b["ra_ll_ap"][r]); mm = float(m_mean - m["ra_ll_ap"][r])
                per_seed["baseline_exact_mask_minus_ra_ll"].append(be)
                per_seed["m2_exact_mask_minus_ra_ll"].append(me)
                per_seed["baseline_mean_mask_minus_ra_ll"].append(bm)
                per_seed["m2_mean_mask_minus_ra_ll"].append(mm)
                per_seed["m2_minus_baseline_exact_contrast"].append(me - be)
                per_seed["m2_minus_baseline_mean_contrast"].append(mm - bm)
                per_seed["m2_minus_baseline_ra_ll_auprc"].append(float(m["ra_ll_ap"][r] - b["ra_ll_ap"][r]))
                per_seed["m2_minus_baseline_clean_auprc"].append(float(m["clean_ap"][r] - b["clean_ap"][r]))
                per_seed["m2_minus_baseline_mean_mask_auprc"].append(float(m_mean - b_mean))
            for field, values in per_seed.items():
                matrix[field].append(float(np.mean(values)))
            replicate_rows.append({"replicate": r, "endpoint": label, **{k: float(np.mean(v)) for k, v in per_seed.items()}})
        for field in field_names:
            v = np.asarray(matrix[field], dtype=np.float64)
            summary_rows.append({"endpoint": label, "contrast": field, "mean": float(v.mean()),
                                 "lower95": float(np.quantile(v, .025)), "upper95": float(np.quantile(v, .975)),
                                 "replicates": N_BOOT, "unit": "canonical_record", "synchronized": True})
    return summary_rows, replicate_rows


def make_revised_decisions(distributions: list[dict], contrasts: list[dict], bootstrap: list[dict], protocol: dict) -> dict:
    lookup = {(r["mode"], r["seed"], r["endpoint"]): r for r in contrasts}
    dist = {(r["mode"], r["seed"], r["endpoint"]): r for r in distributions}
    bs = {(r["endpoint"], r["contrast"]): {**r, "mean": float(r["mean"]),
                                                "lower95": float(r["lower95"]), "upper95": float(r["upper95"])}
          for r in bootstrap}
    base_twa = [lookup[("baseline_v2", s, "TWA")] for s in SEEDS]
    m2_twa = [lookup[("physics_aware_m2", s, "TWA")] for s in SEEDS]
    b_exact = [r["exact_mask_minus_ra_ll"] for r in base_twa]
    b_mean = [r["mean_mask_minus_ra_ll"] for r in base_twa]
    b_mean_ci = bs[("TWA", "baseline_mean_mask_minus_ra_ll")]
    base_decision = "CONFIRMED_GENERAL_FAILURE" if all(v > 0 for v in b_mean) and np.mean(b_mean) >= .05 and b_mean_ci["lower95"] > 0 else "NOT_CONFIRMED"
    ra_improvements = [m["ra_ll_auprc"] - b["ra_ll_auprc"] for b, m in zip(base_twa, m2_twa)]
    clean_deltas = [m["clean_auprc"] - b["clean_auprc"] for b, m in zip(base_twa, m2_twa)]
    mask_deltas = [dist[("physics_aware_m2", s, "TWA")]["mask_mean_auprc"] - dist[("baseline_v2", s, "TWA")]["mask_mean_auprc"] for s in SEEDS]
    reduction = -bs[("TWA", "m2_minus_baseline_mean_contrast")]["mean"]
    m2_contrast_upper = bs[("TWA", "m2_mean_mask_minus_ra_ll")]["upper95"]
    gates = {
        "ra_ll_twa_improvement_each_seed_min_0_15": all(v >= .15 for v in ra_improvements),
        "matched_contrast_decrease_aggregate_min_0_15": reduction >= .15,
        "matched_contrast_upper95_max_0_15": m2_contrast_upper <= .15,
        "clean_twa_mean_decline_max_0_02": float(np.mean(clean_deltas)) >= -.02,
        "clean_twa_any_seed_decline_max_0_04": all(v >= -.04 for v in clean_deltas),
        "mean_mask_twa_mean_decline_max_0_02": float(np.mean(mask_deltas)) >= -.02,
        "mean_mask_twa_any_seed_decline_max_0_04": all(v >= -.04 for v in mask_deltas),
        "all_three_seeds_validation_and_integrity": True,
    }
    m2_decision = "PASS_M2_EXTERNAL_REPLICATION" if all(gates.values()) else "FAIL_M2_NOT_REPLICATED"
    out = {
        "status": "PASS_REVISED_DECISIONS",
        "analysis_status": "corrected_reanalysis_after_prior_test_access; not untouched confirmation",
        "protocol_binding_sha256": protocol["binding_sha256"],
        "ningbo_confirmation_v2": {"decision": base_decision, "all_seed_mean_mask_contrasts_positive": all(v > 0 for v in b_mean),
                                   "mean_mask_contrasts_by_seed": b_mean, "exact_mask_contrasts_by_seed": b_exact,
                                   "mean_mask_contrast": float(np.mean(b_mean)),
                                   "bootstrap": b_mean_ci, "frozen_thresholds_changed": False},
        "m2_revised_decision": {"decision": m2_decision, "checks": gates,
                                "ra_ll_twa_improvement_by_seed": ra_improvements, "clean_twa_change_by_seed": clean_deltas,
                                "mean_mask_twa_change_by_seed": mask_deltas,
                                "matched_contrast_reduction": reduction, "m2_matched_contrast_upper95": m2_contrast_upper,
                                "frozen_criteria_changed": False},
        "interpretation": {"clean_trained_ra_ll_vulnerability": "assessed separately from mask contrasts; see direct RA_LL and clean rows",
                           "physics_aware_ra_ll_benefit": "M2 minus baseline RA_LL AUPRC; see paired bootstrap rows",
                           "reversal_vs_missingness": "reassessed with corrected five-lead removals",
                           "mask_quality_retention": "reassessed using mean per-mask AUPRC and unchanged frozen criteria"},
    }
    atomic_json(OUT / "REVISED_DECISIONS.json", out)
    return out


def save_old_vs_corrected(protocol: dict, condition_rows: list[dict]) -> None:
    old = {}
    with open(BASE / "CONDITION_RESULTS_BY_SEED.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["condition"] == EXACT_NAME and r["label"] == "TWA":
                old[("baseline_v2", int(r["seed"]))] = float(r["auprc"])
    with open(M2 / "RESULTS_BY_SEED.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("condition") == EXACT_NAME and r.get("label", r.get("endpoint")) == "TWA":
                old[("physics_aware_m2", int(r["seed"]))] = float(r["auprc"])
    current = {(r["mode"], r["seed"]): r["auprc"] for r in condition_rows if r["condition"] == EXACT_NAME and r["endpoint"] == "TWA"}
    forensic = OUT / "pilot_forensic_baseline_seed_104729.npz"
    forensic_inverted_match = False
    if forensic.is_file():
        with np.load(forensic, allow_pickle=False) as z:
            forensic_inverted_match = bool(np.allclose(z["historical_exact"], z["inverted_replay"],
                                                        rtol=CLEAN_TOL_RTOL, atol=CLEAN_TOL_ATOL))
    rows = []
    for key, corrected in current.items():
        mode, seed = key
        prior = old.get(key)
        rows.append({"mode": mode, "seed": seed, "condition": EXACT_NAME, "historical_auprc": prior,
                     "historical_semantics": "legacy saved score; unresolved except the forensic replay described in pilot",
                     "corrected_auprc": corrected, "corrected_minus_historical": None if prior is None else corrected - prior,
                     "historical_status": ("VERIFIED_INVERTED_SINGLE_CONDITION" if mode == "baseline_v2" and seed == 104729 and forensic_inverted_match else "UNRESOLVED_PROVENANCE"),
                     "forensic_inverted_replay_match": (forensic_inverted_match if mode == "baseline_v2" and seed == 104729 else None),
                     "recovery_protocol_sha256": protocol["binding_sha256"]})
    atomic_csv(OUT / "OLD_VS_CORRECTED.csv", rows)


def make_plot(distributions: list[dict], condition_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        values = {}
        for seed in SEEDS:
            rows = [r for r in condition_rows if r["mode"] == mode and r["seed"] == seed and r["endpoint"] == "TWA" and r["condition"].startswith("mask5_")]
            values[seed] = np.asarray([r["auprc"] for r in rows])
        ax.boxplot([values[s] for s in SEEDS], positions=[1, 2, 3], widths=.45, showfliers=False,
                   whis=(5, 95), patch_artist=True, boxprops={"facecolor": "#dbeafe", "color": "#1d4ed8"},
                   medianprops={"color": "#111827"})
        ax.set_ylabel("TWA AUPRC")
        ax.set_title("Ningbo clean-trained baseline" if mode == "baseline_v2" else "Ningbo physics-aware M2")
        for pos, seed in enumerate(SEEDS, start=1):
            d = next(r for r in distributions if r["mode"] == mode and r["seed"] == seed and r["endpoint"] == "TWA")
            ax.scatter(pos, d["ra_ll_auprc"], color="#dc2626", marker="D", zorder=4, label="RA–LL" if pos == 1 else None)
            ax.scatter(pos, d["exact_mask_auprc"], color="#059669", marker="*", s=95, zorder=5, label="Exact RA–LL mask" if pos == 1 else None)
        ax.legend(loc="best")
    axes[-1].set_xticks([1, 2, 3], [str(s) for s in SEEDS])
    axes[-1].set_xlabel("Seed (box: 792 individual five-lead masks; whiskers: p05–p95)")
    path = OUT / "mask_auprc_distribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)


def status_final(stage: str, **extra) -> None:
    atomic_json(OUT / "WORKFLOW_STATUS.json", {"stage": stage, "updated_utc": now(), "pid": os.getpid(), **extra})


def run_full(protocol: dict) -> dict:
    # If the full waveform matrix is held for less than the process lifetime,
    # mask batches still use the shared immutable array, not repeated disk loads.
    test, _, _ = load_parts()
    x = load_all_raw(test)
    units = []
    for mode in MODES:
        for seed in SEEDS:
            units.append(run_model_unit(mode, seed, x, protocol))
    del x
    integrity = validate_complete_units(protocol)
    atomic_json(OUT / "MASK_INFERENCE_INTEGRITY.json", integrity)
    status_final("MASK_INFERENCE_COMPLETE", units=6, completed_conditions=6 * 792, total_conditions=6 * 792)
    condition_rows, exact_rows, distributions, contrasts = summarize_predictions(protocol)
    status_final("POINT_METRICS_COMPLETE", condition_rows=len(condition_rows))
    boot = run_bootstrap(protocol)
    decisions = make_revised_decisions(distributions, contrasts,
                                       list(csv.DictReader((OUT / "BOOTSTRAP_SUMMARY.csv").open(encoding="utf-8-sig"))), protocol)
    save_old_vs_corrected(protocol, condition_rows)
    make_plot(distributions, condition_rows)
    report_final(protocol, condition_rows, exact_rows, distributions, contrasts, decisions, integrity)
    status_final("COMPLETE", units=6, mask_predictions=6 * 792, bootstrap_tasks=18)
    atomic_json(OUT / "resume_state.json", {"status": "COMPLETE", "protocol_binding_sha256": protocol["binding_sha256"], "completed_utc": now()})
    return {"integrity": integrity, "decisions": decisions, "bootstrap": boot}


def report_final(protocol: dict, condition_rows: list[dict], exact_rows: list[dict], distributions: list[dict],
                 contrasts: list[dict], decisions: dict, integrity: dict) -> None:
    rows = {(r["mode"], r["seed"], r["endpoint"]): r for r in contrasts}
    dist = {(r["mode"], r["seed"], r["endpoint"]): r for r in distributions}
    summary = list(csv.DictReader((OUT / "BOOTSTRAP_SUMMARY.csv").open(encoding="utf-8-sig")))
    bs = {(r["endpoint"], r["contrast"]): r for r in summary}
    base_ct = [rows[("baseline_v2", s, "TWA")]["mean_mask_minus_ra_ll"] for s in SEEDS]
    exact_ct = [rows[("baseline_v2", s, "TWA")]["exact_mask_minus_ra_ll"] for s in SEEDS]
    m2_ct = [rows[("physics_aware_m2", s, "TWA")]["mean_mask_minus_ra_ll"] for s in SEEDS]
    m2_exact_ct = [rows[("physics_aware_m2", s, "TWA")]["exact_mask_minus_ra_ll"] for s in SEEDS]
    lines = ["# Ningbo mask semantics recovery: final report", "",
             "This is a corrected reanalysis after prior access to the Ningbo test set. It is not an untouched confirmation.", "",
             f"Protocol binding: `{protocol['binding_sha256']}`.",
             f"Engineering integrity: {integrity['decision']} for {integrity['units']} model/seed units and {integrity['mask_predictions_verified']} corrected mask predictions.", "",
             "## Main results", "",
             f"Baseline mean-mask minus RA–LL TWA AUPRC by seed: `{[round(v, 4) for v in base_ct]}`. The mean is `{np.mean(base_ct):.4f}`. The synchronized canonical-record bootstrap 95% interval is `{bs[('TWA','baseline_mean_mask_minus_ra_ll')]['lower95']}` to `{bs[('TWA','baseline_mean_mask_minus_ra_ll')]['upper95']}`. Decision: **{decisions['ningbo_confirmation_v2']['decision']}**.",
             f"Baseline exact-mask minus RA–LL by seed: `{[round(v, 4) for v in exact_ct]}`. Mean: `{np.mean(exact_ct):.4f}`.",
             f"M2 mean-mask minus RA–LL by seed: `{[round(v, 4) for v in m2_ct]}`. Mean: `{np.mean(m2_ct):.4f}`.",
             f"M2 exact-mask minus RA–LL by seed: `{[round(v, 4) for v in m2_exact_ct]}`. Mean: `{np.mean(m2_exact_ct):.4f}`.",
             f"Revised M2 decision under unchanged gates: **{decisions['m2_revised_decision']['decision']}**.", "",
             "See `MASK_DISTRIBUTION_SUMMARY.csv` for mask distributions and RA–LL percentiles, and `PRIMARY_CONTRASTS.csv` for per-seed direct clean and RA–LL comparisons.", "",
             "## Interpretation and historical claims", "",
             "The old Ningbo v2 exact-mask value near 0.527 is retained in `OLD_VS_CORRECTED.csv` as a historical score. The baseline seed 104729 forensic pilot determines whether its single exact-mask historical cache matches the inverted evaluator replay; that check cannot establish the semantics of the other 791 masks or other seeds.",
             "Clean and RA–LL rows are reused only after full completion-hash checks and fresh 512-record probe inference for all six units. Old mask caches were not reused.",
             "The original v2 and M2 decisions are superseded only for mask-dependent criteria. Clean-trained RA–LL vulnerability and M2 RA–LL improvement are presented separately from corrected mask comparisons.", "",
             "## Prior claims", "",
             "Preserve direct clean and RA–LL observations and thresholds. Replace prior mask means, exact-mask scores, mask-minus-reversal intervals, and mask-dependent PASS/FAIL claims with the recovery tables. Do not describe the analysis as independent or untouched confirmation.", "",
             "## Next step: Transformer M3", "",
             "Apply the same versioned evaluator and cache contract to M3's six existing Chapman checkpoints in a separate recovery package. Reuse checkpoints and clean/RA–LL predictions only after model/order/hash checks; regenerate the 792 mask panel, recompute Stage M3A criteria and update downstream figures. Do not retrain M3 unless a separate audit identifies a training-path defect.", ""]
    (OUT / "FINAL_RECOVERY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(command: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if command == "freeze":
        protocol = write_recovery_protocol()
        atomic_json(OUT / "WORKFLOW_STATUS.json", {"stage": "PROTOCOL_FROZEN", "updated_utc": now(),
                                                      "protocol_binding_sha256": protocol["binding_sha256"], "pid": os.getpid()})
        print(f"PROTOCOL_FROZEN {protocol['binding_sha256']}", flush=True)
        return
    protocol_path = OUT / "RECOVERY_PROTOCOL.json"
    if not protocol_path.is_file():
        raise RuntimeError("RECOVERY_PROTOCOL.json must be frozen before inference")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    current_source = sha256(Path(__file__))
    if protocol.get("evaluator", {}).get("source_sha256") != current_source:
        amendment_path = OUT / "RECOVERY_PROTOCOL_AMENDMENT_001.json"
        if not amendment_path.is_file():
            raise RuntimeError("evaluator source changed after protocol freeze and no amendment is frozen")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        amendment_binding = amendment.get("amendment_binding_sha256")
        if amendment_binding != hash_json({k: v for k, v in amendment.items() if k != "amendment_binding_sha256"}):
            raise RuntimeError("recovery protocol amendment content hash mismatch")
        if amendment.get("prior_protocol_binding_sha256") != protocol.get("binding_sha256"):
            raise RuntimeError("recovery protocol amendment binding mismatch")
        source_binding = amendment.get("runner_source_sha256")
        prior_binding = amendment_binding
        for version in ("002", "003", "004", "005"):
            if source_binding == current_source:
                break
            amendment_n_path = OUT / f"RECOVERY_PROTOCOL_AMENDMENT_{version}.json"
            if not amendment_n_path.is_file():
                raise RuntimeError(f"current evaluator source is not bound by recovery amendment {version}")
            amendment_n = json.loads(amendment_n_path.read_text(encoding="utf-8"))
            amendment_n_binding = amendment_n.get("amendment_binding_sha256")
            if amendment_n_binding != hash_json({k: v for k, v in amendment_n.items() if k != "amendment_binding_sha256"}):
                raise RuntimeError(f"recovery protocol amendment {version} content hash mismatch")
            if (amendment_n.get("prior_protocol_binding_sha256") != protocol.get("binding_sha256")
                    or amendment_n.get("prior_amendment_binding_sha256") != prior_binding):
                raise RuntimeError(f"recovery protocol amendment {version} chain mismatch")
            source_binding = amendment_n.get("runner_source_sha256")
            prior_binding = amendment_n_binding
        if source_binding != current_source:
            raise RuntimeError("current evaluator source is not bound by the recovery amendment chain")
    if protocol.get("binding_sha256") != hash_json({k: v for k, v in protocol.items() if k != "binding_sha256"}):
        raise RuntimeError("recovery protocol binding hash mismatch")
    if command == "pilot":
        run_pilot(protocol)
        return
    if command == "run":
        pilot = json.loads((OUT / "PILOT_REPORT.json").read_text(encoding="utf-8"))
        if pilot.get("decision") != "PASS" or not pilot.get("clean_ra_ll_reuse_verified"):
            raise RuntimeError("pilot or clean/RA_LL reuse check has not passed")
        run_full(protocol)
        return
    if command == "summarize":
        integrity = validate_complete_units(protocol)
        condition_rows, exact_rows, distributions, contrasts = summarize_predictions(protocol)
        run_bootstrap(protocol)
        boot = list(csv.DictReader((OUT / "BOOTSTRAP_SUMMARY.csv").open(encoding="utf-8-sig")))
        make_revised_decisions(distributions, contrasts, boot, protocol)
        save_old_vs_corrected(protocol, condition_rows)
        make_plot(distributions, condition_rows)
        print(json.dumps(integrity))
        return
    raise ValueError(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "pilot", "run", "summarize"))
    run(parser.parse_args().command)
