from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import ecgfounder_baseline_v2 as founder  # noqa: E402
from src.ecg_physics import named_reversal  # noqa: E402
from outputs.heldout_limb_cycles_v1 import cycle_adapter  # noqa: E402

SEEDS = (104729, 130363, 155921)
COHORTS = ("Chapman", "Ningbo")
MODES = ("clean_head", "physics_head")
TRAIN_VIEWS = ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BATCH_INFER = 16
BATCH_TRAIN = 256
TOL = 1e-6
LABELS = ("TWA", "SB", "STach")
ACTIVE_PHASE = "setup"
SESSION_STARTED = time.perf_counter()
PHASE_STARTED = SESSION_STARTED
PREVIOUS_RUNTIME = {}
SESSION_ID = ""
VALIDATION_INPUTS = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def protocol() -> dict:
    global VALIDATION_INPUTS
    p = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    digest = p.pop("protocol_sha256")
    if hash_obj(p) != digest:
        raise RuntimeError("frozen protocol checksum mismatch")
    p["protocol_sha256"] = digest
    if p["status"] != "FROZEN_BEFORE_NEW_TEST_OUTCOMES":
        raise RuntimeError("protocol is not frozen before new test outcomes")
    if sha(founder.CHECKPOINT) != p["source"]["checkpoint_sha256"]:
        raise RuntimeError("Founder checkpoint changed after freeze")
    amendment_path = OUT / "CONTINUATION_AMENDMENT.json"
    if not amendment_path.is_file():
        raise RuntimeError("continuation amendment is required before running this runner")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    frozen_runner = OUT / "run_mitigation.py"
    amendment_digest = amendment.pop("amendment_sha256", None)
    computed_amendment_digest = hash_obj(amendment)
    amendment["amendment_sha256"] = amendment_digest
    if (amendment.get("base_protocol_sha256") != digest or
            amendment.get("original_runner_sha256") != sha(frozen_runner) or
            amendment.get("continuation_runner_sha256") != sha(Path(__file__)) or
            amendment_digest != computed_amendment_digest):
        raise RuntimeError("continuation amendment/code bindings do not match")
    for rel, expected in amendment.get("preserved_artifact_sha256", {}).items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"preserved artifact changed after continuation amendment: {rel}")
    validation_path = OUT / "CONTINUATION_VALIDATION_INPUT_BINDINGS.json"
    if sha(validation_path) != amendment.get("validation_inputs_sha256"):
        raise RuntimeError("validation raw input bindings changed after amendment")
    VALIDATION_INPUTS = json.loads(validation_path.read_text(encoding="utf-8"))
    val_binding_digest = VALIDATION_INPUTS.pop("binding_sha256", None)
    if (VALIDATION_INPUTS.get("base_protocol_sha256") != digest or
            hash_obj(VALIDATION_INPUTS) != val_binding_digest):
        raise RuntimeError("validation input binding fingerprint mismatch")
    VALIDATION_INPUTS["binding_sha256"] = val_binding_digest
    for dataset in COHORTS:
        val_rows = founder.split(dataset, "validation")
        val_ids = [r["record_id"] for r in val_rows]
        if (VALIDATION_INPUTS["record_counts"].get(dataset) != len(val_ids) or
                VALIDATION_INPUTS["record_order_sha256"].get(dataset) != id_digest(val_ids)):
            raise RuntimeError(f"validation input binding order mismatch: {dataset}")
    return p


def record_hash_paths(dataset: str, rid: str):
    if dataset == "Ningbo":
        c = founder.ningbo_canonical()[rid]
        return founder.canonical_path(c["header_path"], ".hea"), founder.canonical_path(c["waveform_path"], ".mat")
    stem = founder.record_path(dataset, rid)
    return stem.with_suffix(".hea"), stem.with_suffix(".mat")


def verify_input(dataset: str, partition: str, rid: str, p: dict) -> None:
    key = f"{dataset}/{partition}/{rid}"
    expected = p["data"]["new_waveform_header_hashes"].get(key)
    hea, mat = record_hash_paths(dataset, rid)
    if expected is None and partition == "validation" and VALIDATION_INPUTS is not None:
        expected = VALIDATION_INPUTS["raw_waveform_header_hashes"].get(key)
        if expected is not None:
            if sha(hea) != expected["header_sha256"] or sha(mat) != expected["waveform_file_sha256"]:
                raise RuntimeError(f"validation raw waveform/header changed after continuation binding: {key}")
            return
    if expected is None:
        raise RuntimeError(f"record was not frozen in protocol or continuation binding: {key}")
    if sha(hea) != expected["header_sha256"] or sha(mat) != expected["waveform_sha256"]:
        raise RuntimeError(f"raw waveform/header changed after protocol freeze: {key}")


def feature_path(dataset: str, partition: str, condition: str) -> Path:
    tag = condition.replace("/", "_")
    return OUT / "features" / f"{dataset.lower()}_{partition}_{tag}.npy"


def feature_meta_path(dataset: str, partition: str, condition: str) -> Path:
    return feature_path(dataset, partition, condition).with_suffix(".meta.json")


def scaler_path(dataset: str) -> Path:
    return OUT / "scaling" / f"{dataset.lower()}_clean_train.npz"


def load_scaler(dataset: str, p: dict) -> tuple[np.ndarray, np.ndarray, str]:
    path = scaler_path(dataset)
    if not path.is_file():
        raise RuntimeError(f"missing clean-train scaler for {dataset}: {path}")
    with np.load(path, allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        ids_hash = str(z["train_ids_sha256"].item())
        features_hash = str(z["feature_sha256"].item())
        protocol_hash = str(z["protocol_sha256"].item())
    ids = [r["record_id"] for r in records(dataset, "train")]
    if (mean.shape != (1024,) or std.shape != (1024,) or not np.isfinite(mean).all() or
            not np.isfinite(std).all() or np.any(std < 1e-6) or ids_hash != id_digest(ids) or
            features_hash != sha(feature_path(dataset, "train", "clean")) or
            protocol_hash != p["protocol_sha256"]):
        raise RuntimeError(f"scaler provenance/shape check failed for {dataset}")
    return mean, std, sha(path)


def records(dataset: str, partition: str):
    return founder.split(dataset, partition)


def id_digest(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def make_transform(raw: np.ndarray, condition: str) -> np.ndarray:
    independent = np.concatenate((raw[:2], raw[6:]), axis=0)
    if condition in TRAIN_VIEWS:
        transformed = founder.derive_display(named_reversal(independent, condition))
    elif condition in ("cycle_slots_120", "cycle_slots_201"):
        transformed = cycle_adapter.transform(raw, condition)
    else:
        raise ValueError(f"unknown signal transform: {condition}")
    # The cycle adapter emits float32 from float64 source arrays, so compare
    # after its declared cast. Float64 algebra can add only roundoff noise.
    chest_unchanged = (
        np.array_equal(transformed[6:], raw[6:].astype(np.float32, copy=False))
        if transformed.dtype == np.float32
        else np.allclose(transformed[6:], raw[6:], rtol=0, atol=1e-10)
    )
    if not chest_unchanged:
        raise RuntimeError(f"chest leads changed under {condition}")
    return transformed


def prepare_condition(raw: np.ndarray, condition: str, mask_indices: tuple[int, ...] | None = None) -> np.ndarray:
    x = raw if condition == "clean" or mask_indices is not None else make_transform(raw, condition)
    z = founder.preprocess(x)
    if mask_indices is not None:
        z = z.copy()
        z[list(mask_indices), :] = 0.0
    if z.shape != (12, 5000) or not np.isfinite(z).all():
        raise RuntimeError("invalid preprocessed encoder input")
    return z


def copy_verified_clean_cache(dataset: str, partition: str, p: dict) -> np.ndarray:
    src, _, meta_path = founder.cache_paths(dataset, partition, "clean")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = [r["record_id"] for r in records(dataset, partition)]
    if meta.get("record_ids_sha256") != id_digest(ids) or meta.get("checkpoint_sha256") != p["source"]["checkpoint_sha256"]:
        raise RuntimeError(f"old clean feature cache ID/checkpoint binding mismatch: {dataset}/{partition}")
    source_rel, meta_rel = src.relative_to(ROOT).as_posix(), meta_path.relative_to(ROOT).as_posix()
    if (meta.get("protocol_sha256") != sha(founder.CFG) or sha(src) != meta.get("features_sha256") or
            sha(src) != p["frozen_inputs_sha256"].get(source_rel) or
            sha(meta_path) != p["frozen_inputs_sha256"].get(meta_rel)):
        raise RuntimeError(f"old clean feature cache source hash mismatch: {dataset}/{partition}")
    target = feature_path(dataset, partition, "clean")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(src):
            raise RuntimeError(f"local copied clean cache mismatch: {target}")
    else:
        shutil.copyfile(src, target)
    outmeta = {"dataset": dataset, "partition": partition, "condition": "clean",
               "source": str(src.relative_to(ROOT)), "source_cache_sha256": sha(src),
               "features_sha256": sha(target), "checkpoint_sha256": p["source"]["checkpoint_sha256"],
               "encoder_protocol_sha256": sha(founder.CFG), "mitigation_protocol_sha256": p["protocol_sha256"],
               "record_ids_sha256": id_digest(ids), "record_count": len(ids), "feature_shape": [len(ids), 1024],
               "verified_reuse": True}
    atomic_json(feature_meta_path(dataset, partition, "clean"), outmeta)
    return np.load(target, mmap_mode="r", allow_pickle=False)


def ensure_features(dataset: str, partition: str, condition: str, p: dict,
                    mask_indices: tuple[int, ...] | None = None,
                    force_recompute: bool = False) -> np.ndarray:
    if condition == "clean" and not force_recompute:
        return copy_verified_clean_cache(dataset, partition, p)
    rows = records(dataset, partition)
    ids = [r["record_id"] for r in rows]
    target = feature_path(dataset, partition, condition)
    meta_path = feature_meta_path(dataset, partition, condition)
    binding = {"dataset": dataset, "partition": partition, "condition": condition,
               "record_ids_sha256": id_digest(ids), "record_count": len(ids),
               "checkpoint_sha256": p["source"]["checkpoint_sha256"],
               "encoder_protocol_sha256": sha(founder.CFG),
               "mitigation_protocol_sha256": p["protocol_sha256"],
               "mask_indices_zeroed": list(mask_indices or []), "batch_size": BATCH_INFER,
               "features_shape": [len(ids), 1024], "dtype": "float32"}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta.get(k) != v for k, v in binding.items()):
            raise RuntimeError(f"feature cache binding changed: {meta_path}")
        if meta.get("status") == "COMPLETE":
            if not target.is_file() or sha(target) != meta.get("features_sha256"):
                raise RuntimeError(f"completed feature cache hash mismatch: {target}")
            return np.load(target, mmap_mode="r", allow_pickle=False)
        completed = int(meta.get("completed_records", 0))
        if not target.is_file() or completed < 0 or completed > len(ids):
            raise RuntimeError(f"invalid resumable feature cache: {target}")
        features = np.load(target, mmap_mode="r+", allow_pickle=False)
    else:
        if target.exists():
            raise RuntimeError(f"unbound feature file exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        features = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(ids), 1024))
        completed = 0
        atomic_json(meta_path, {**binding, "status": "IN_PROGRESS", "completed_records": 0})

    net = founder.model()
    if net.training or any(pp.requires_grad for pp in net.parameters()):
        raise RuntimeError("Founder encoder must be eval with every parameter frozen")
    source_partition = partition
    for start in range(completed, len(rows), BATCH_INFER):
        block = rows[start:start + BATCH_INFER]
        inputs = []
        for row in block:
            rid = row["record_id"]
            verify_input(dataset, source_partition, rid, p)
            raw = founder.physical_signal(dataset, rid)
            inputs.append(prepare_condition(raw, condition, mask_indices))
        f, z = founder.infer(net, np.stack(inputs))
        if f.shape != (len(block), 1024) or not np.isfinite(f).all() or not np.isfinite(z).all():
            raise RuntimeError(f"non-finite or malformed encoder output: {dataset}/{partition}/{condition}")
        end = start + len(block)
        features[start:end] = f.astype(np.float32)
        features.flush()
        atomic_json(meta_path, {**binding, "status": "IN_PROGRESS", "completed_records": end})
        if end % 512 < BATCH_INFER or end == len(rows):
            status("FEATURES", dataset=dataset, partition=partition, condition=condition,
                   completed_records=end, total_records=len(rows))
    atomic_json(meta_path, {**binding, "status": "COMPLETE", "completed_records": len(rows),
                            "features_sha256": sha(target), "completed_utc": now()})
    del net
    torch.cuda.empty_cache()
    return np.load(target, mmap_mode="r", allow_pickle=False)


def status(stage: str, **kw) -> None:
    runtime_path = OUT / "CONTINUATION_RUNTIME.json"
    global PREVIOUS_RUNTIME
    prev = PREVIOUS_RUNTIME or {}
    active = time.perf_counter() - SESSION_STARTED
    phase_active = time.perf_counter() - PHASE_STARTED
    phase_key = f"{ACTIVE_PHASE}_seconds"
    same_session = prev.get("session_id") == SESSION_ID
    pipeline_delta = max(0.0, active - (float(prev.get("last_session_elapsed_seconds", 0.0)) if same_session else 0.0))
    same_phase = same_session and prev.get("current_phase") == ACTIVE_PHASE
    phase_delta = max(0.0, phase_active - (float(prev.get("last_phase_elapsed_seconds", 0.0)) if same_phase else 0.0))
    obj = {"stage": stage, "phase": ACTIVE_PHASE, "updated_utc": now(),
           "session_elapsed_seconds": active,
           "pipeline_elapsed_seconds": float(prev.get("pipeline_elapsed_seconds", 0.0)) + pipeline_delta,
           phase_key: float(prev.get(phase_key, 0.0)) + phase_delta,
           "session_id": SESSION_ID, "current_phase": ACTIVE_PHASE,
           "last_session_elapsed_seconds": active, "last_phase_elapsed_seconds": phase_active, **kw}
    atomic_json(OUT / "WORKFLOW_STATUS.json", obj)
    with (OUT / "CONTINUATION_EVENTS.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    atomic_json(runtime_path, {k: v for k, v in obj.items()
                               if k.endswith("_seconds") or k in ("updated_utc", "session_id", "current_phase")})
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    PREVIOUS_RUNTIME = obj
    if ACTIVE_PHASE == "feature_preparation" and obj.get("feature_preparation_seconds", 0.0) > 12 * 3600:
        raise RuntimeError("12-hour cumulative feature-preparation limit exceeded")


def set_phase(name: str) -> None:
    global ACTIVE_PHASE, PHASE_STARTED
    status("PHASE_FINISHED")
    ACTIVE_PHASE = name
    PHASE_STARTED = time.perf_counter()
    status("PHASE_STARTED", phase=name)


def prepare_training_features(p: dict) -> None:
    audit_partial_chapman_cache(p)
    for dataset in COHORTS:
        for partition in ("train", "validation"):
            copy_verified_clean_cache(dataset, partition, p)
    audit_clean_cache_replay(p)
    for dataset in COHORTS:
        for condition in TRAIN_VIEWS:
            ensure_features(dataset, "train", condition, p)
    status("TRAIN_FEATURES_COMPLETE")


def feature_prefix_sha(features: np.ndarray, n: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(features[:n]).tobytes()).hexdigest()


def audit_partial_chapman_cache(p: dict) -> None:
    target = feature_path("Chapman", "train", "RA_LA_reversal")
    meta_path = feature_meta_path("Chapman", "train", "RA_LA_reversal")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = [r["record_id"] for r in records("Chapman", "train")]
    x = np.load(target, mmap_mode="r", allow_pickle=False)
    completed = int(meta.get("completed_records", 0))
    expected = {"dataset": "Chapman", "partition": "train", "condition": "RA_LA_reversal",
                "record_ids_sha256": id_digest(ids), "record_count": len(ids),
                "checkpoint_sha256": p["source"]["checkpoint_sha256"],
                "encoder_protocol_sha256": sha(founder.CFG),
                "mitigation_protocol_sha256": p["protocol_sha256"],
                "batch_size": BATCH_INFER, "features_shape": [len(ids), 1024], "dtype": "float32"}
    if x.shape != (len(ids), 1024) or x.dtype != np.float32 or any(meta.get(k) != v for k, v in expected.items()):
        raise RuntimeError("partial Chapman RA_LA cache metadata, shape or canonical order mismatch")
    if not 0 < completed <= len(ids) or not np.isfinite(x[:completed]).all():
        raise RuntimeError("partial Chapman RA_LA cache has invalid committed prefix")
    frozen_path = OUT / "audits" / "CACHE_CONTINUATION_PREFLIGHT.json"
    frozen_path.parent.mkdir(exist_ok=True)
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        prefix_count = int(frozen["initial_completed_records"])
        prefix_hash = feature_prefix_sha(x, prefix_count)
        if prefix_hash != frozen["initial_prefix_sha256"]:
            raise RuntimeError("previously committed Chapman RA_LA prefix changed")
    else:
        if completed != 2464:
            raise RuntimeError(f"expected initial committed prefix of 2464 rows, found {completed}")
        frozen = {"dataset": "Chapman", "partition": "train", "condition": "RA_LA_reversal",
                  "initial_completed_records": completed,
                  "initial_prefix_sha256": feature_prefix_sha(x, completed),
                  "cache_file_sha256_at_preflight": sha(target), "metadata_sha256_at_preflight": sha(meta_path),
                  "record_ids_sha256": id_digest(ids), "shape": list(x.shape), "dtype": str(x.dtype),
                  "checkpoint_sha256": p["source"]["checkpoint_sha256"],
                  "protocol_sha256": p["protocol_sha256"], "status": "PREFIX_BOUND_REPLAY_PENDING"}
        atomic_json(frozen_path, frozen)
    if frozen.get("status") == "PASS":
        status("PARTIAL_CACHE_PREFIX_VERIFIED", committed_records=completed,
               initial_prefix_sha256=frozen["initial_prefix_sha256"])
        return
    net = founder.model()
    state_before = state_hash(net.state_dict())
    indices = (0, 16 * 31, completed - BATCH_INFER)
    results = []
    rows = records("Chapman", "train")
    for start in sorted(set(int(i) for i in indices)):
        if start < 0 or start + BATCH_INFER > completed:
            continue
        raw_batch = []
        for row in rows[start:start + BATCH_INFER]:
            verify_input("Chapman", "train", row["record_id"], p)
            raw_batch.append(prepare_condition(founder.physical_signal("Chapman", row["record_id"]), "RA_LA_reversal"))
        replay, _ = founder.infer(net, np.stack(raw_batch))
        diff = float(np.max(np.abs(replay - x[start:start + BATCH_INFER])))
        results.append({"start": start, "rows": BATCH_INFER, "max_abs_feature_difference": diff,
                        "atol": TOL, "passed": bool(np.allclose(replay, x[start:start + BATCH_INFER], rtol=0, atol=TOL))})
    state_after = state_hash(net.state_dict())
    if state_before != state_after or not results or not all(r["passed"] for r in results):
        raise RuntimeError(f"partial-cache batch replay failed: {results}; encoder_unchanged={state_before == state_after}")
    frozen.update({"status": "PASS", "replayed_batches": results, "encoder_state_sha256_before": state_before,
                   "encoder_state_sha256_after": state_after, "verified_utc": now()})
    atomic_json(frozen_path, frozen)
    del net
    torch.cuda.empty_cache()
    status("PARTIAL_CACHE_PREFIX_VERIFIED", committed_records=completed,
           initial_prefix_sha256=frozen["initial_prefix_sha256"], replayed_batches=results)


def audit_clean_cache_replay(p: dict) -> None:
    audit_path = OUT / "audits" / "CLEAN_CACHE_REPLAY.json"
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") == p["protocol_sha256"] and old.get("status") == "PASS":
            return
    results = []
    net = founder.model()
    state_before = state_hash(net.state_dict())
    prior_pilot = json.loads((OUT / "ENGINEERING_PILOT.json").read_text(encoding="utf-8"))
    if prior_pilot.get("status") != "PASS" or prior_pilot.get("checkpoint_sha256") != p["source"]["checkpoint_sha256"]:
        raise RuntimeError("prior clean cache replay pilot is not valid for the frozen checkpoint")
    for dataset in COHORTS:
        for partition in ("train", "validation"):
            cache = np.load(feature_path(dataset, partition, "clean"), mmap_mode="r", allow_pickle=False)
            rows = records(dataset, partition)
            if partition == "validation":
                prior = next((r for r in prior_pilot["old_clean_cache_replay"]
                             if r["dataset"] == dataset and r["partition"] == partition), None)
                expected_ids = id_digest([r["record_id"] for r in rows])
                if (prior is None or prior.get("cache_features_sha256") != sha(feature_path(dataset, partition, "clean")) or
                        prior.get("ordered_ids_sha256") != expected_ids or
                        float(prior.get("sample_feature_max_abs_error", float("inf"))) > TOL or
                        int(prior.get("sample_replay_records", 0)) < 2 * BATCH_INFER):
                    raise RuntimeError(f"prior validation clean cache replay evidence mismatch: {dataset}")
                results.append({"dataset": dataset, "partition": partition,
                                "replay_source": "frozen ENGINEERING_PILOT.json; validation inputs were not raw-hash-bound in base protocol",
                                "max_abs_feature_difference": prior["sample_feature_max_abs_error"],
                                "passed": True})
                continue
            for start in (0, 16):
                if start + BATCH_INFER > len(rows):
                    continue
                batch = []
                for row in rows[start:start + BATCH_INFER]:
                    verify_input(dataset, partition, row["record_id"], p)
                    batch.append(prepare_condition(founder.physical_signal(dataset, row["record_id"]), "clean"))
                replay, _ = founder.infer(net, np.stack(batch))
                diff = float(np.max(np.abs(replay - cache[start:start + BATCH_INFER])))
                results.append({"dataset": dataset, "partition": partition, "start": start,
                                "max_abs_feature_difference": diff, "atol": TOL,
                                "passed": bool(np.allclose(replay, cache[start:start + BATCH_INFER], rtol=0, atol=TOL))})
    state_after = state_hash(net.state_dict())
    if state_before != state_after or not all(r["passed"] for r in results):
        raise RuntimeError(f"clean cache replay failed: {results}; encoder_unchanged={state_before == state_after}")
    atomic_json(audit_path, {"status": "PASS", "protocol_sha256": p["protocol_sha256"],
                             "source_cache_hashes": {f"{d}/{part}": sha(feature_path(d, part, "clean"))
                                                     for d in COHORTS for part in ("train", "validation")},
                             "replayed_batches": results, "encoder_state_sha256_before": state_before,
                             "encoder_state_sha256_after": state_after, "verified_utc": now()})
    del net
    torch.cuda.empty_cache()


def digest_int(data: bytes) -> int:
    return int.from_bytes(data[:8], "big", signed=False)


def train_view(seed: int, epoch: int, record_id: str) -> str:
    d = hashlib.sha256(f"{seed}\0{epoch}\0{record_id}".encode()).digest()
    if d[0] & 1 == 0:
        return "clean"
    return TRAIN_VIEWS[digest_int(d[1:]) % len(TRAIN_VIEWS)]


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict(head, x: np.ndarray, mean: np.ndarray, std: np.ndarray, batch: int = 8192) -> np.ndarray:
    outputs = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            xb = (np.asarray(x[start:start + batch], dtype=np.float32) - mean) / std
            logits = head(torch.from_numpy(np.ascontiguousarray(xb)).cuda())
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def load_checkpoint_head(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    head = torch.nn.Linear(1024, 3).cuda().eval()
    head.load_state_dict(obj["state_dict"])
    for pp in head.parameters():
        pp.requires_grad_(False)
    return obj, head


def verify_completed_head(unit: Path, dataset: str, seed: int, mode: str, p: dict,
                          mean: np.ndarray, std: np.ndarray, scaler_sha: str,
                          val_y: np.ndarray, val_ids: list[str], val_features: np.ndarray) -> dict:
    result_path, checkpoint_path = unit / "training_result.json", unit / "best.pt"
    val_pred_path = OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz"
    if not result_path.is_file() or not checkpoint_path.is_file() or not val_pred_path.is_file():
        raise RuntimeError(f"incomplete head result bundle: {unit}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("protocol_sha256") != p["protocol_sha256"] or
            result.get("dataset") != dataset or result.get("seed") != seed or result.get("mode") != mode or
            result.get("scaler_sha256") != scaler_sha or result.get("best_checkpoint_sha256") != sha(checkpoint_path) or
            result.get("validation_predictions_sha256") != sha(val_pred_path) or
            result.get("epoch_history_sha256") != sha(unit / "epoch_history.json")):
        raise RuntimeError(f"completed head binding mismatch: {unit}")
    checkpoint, head = load_checkpoint_head(checkpoint_path)
    if (checkpoint.get("protocol_sha256") != p["protocol_sha256"] or
            checkpoint.get("encoder_checkpoint_sha256") != p["source"]["checkpoint_sha256"] or
            checkpoint.get("scaler_sha256") != scaler_sha or int(checkpoint.get("epoch", -1)) != int(result["best_epoch"])):
        raise RuntimeError(f"completed checkpoint metadata mismatch: {unit}")
    with np.load(val_pred_path, allow_pickle=False) as z:
        saved_p, saved_y = z["p"], z["y"]
        saved_ids = z["record_ids"].astype(str).tolist()
        saved_thresholds = z["thresholds"].astype(np.float64)
        if (z["protocol_sha256"].item() != p["protocol_sha256"] or
                z["scaler_sha256"].item() != scaler_sha or
                z["head_sha256"].item() != sha(checkpoint_path)):
            raise RuntimeError(f"completed validation prediction binding mismatch: {unit}")
    if saved_ids != val_ids or not np.array_equal(saved_y, val_y):
        raise RuntimeError(f"completed validation order/labels mismatch: {unit}")
    replay = predict(head, val_features, mean, std)
    diff = float(np.max(np.abs(replay - saved_p)))
    if diff > 1e-7:
        raise RuntimeError(f"saved validation probability replay mismatch: {unit}; max_abs={diff}")
    if not np.isclose(float(average_precision_score(val_y[:, 0], replay[:, 0])),
                      float(result["best_validation_twa_auprc"]), rtol=0, atol=1e-12):
        raise RuntimeError(f"completed head validation AP binding mismatch: {unit}")
    expected_thresholds = np.asarray([np.quantile(replay[val_y[:, j] == 0, j], 0.90, method="linear")
                                      for j in range(3)], dtype=np.float64)
    if (not np.allclose(saved_thresholds, expected_thresholds, rtol=0, atol=1e-12) or
            not np.allclose(np.asarray(result["thresholds"]), expected_thresholds, rtol=0, atol=1e-12)):
        raise RuntimeError(f"completed validation threshold binding mismatch: {unit}")
    del head
    return result


def train_unit(dataset: str, seed: int, mode: str, p: dict, mean: np.ndarray, std: np.ndarray,
               train_y: np.ndarray, val_y: np.ndarray, train_ids: list[str], val_ids: list[str],
               train_views: dict[str, np.ndarray], val_features: np.ndarray) -> dict:
    unit = OUT / "heads" / dataset.lower() / f"seed_{seed}" / mode
    unit.mkdir(parents=True, exist_ok=True)
    init_path = OUT / "heads" / dataset.lower() / f"seed_{seed}" / "paired_initial_state.pt"
    if not init_path.exists():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        init = torch.nn.Linear(1024, 3).state_dict()
        atomic_torch(init_path, {"seed": seed, "state_dict": {k: v.cpu() for k, v in init.items()},
                                 "initialized_utc": now()})
    init_obj = torch.load(init_path, map_location="cpu", weights_only=False)
    init_state = init_obj["state_dict"]
    init_sha = state_hash(init_state)
    resume_path = unit / "resume.pt"
    checkpoint_path = unit / "best.pt"
    rows_path = unit / "epoch_history.json"
    scaler_sha = sha(scaler_path(dataset))
    if (unit / "training_result.json").exists():
        result = verify_completed_head(unit, dataset, seed, mode, p, mean, std, scaler_sha,
                                       val_y, val_ids, val_features)
        status("COMPLETED_HEAD_REUSED", dataset=dataset, seed=seed, mode=mode,
               best_epoch=result["best_epoch"], optimizer_steps_added=0,
               best_checkpoint_sha256=result["best_checkpoint_sha256"])
        return result
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (resume.get("protocol_sha256") != p["protocol_sha256"] or
                resume.get("initial_state_sha256") != init_sha or resume.get("dataset") != dataset or
                resume.get("seed") != seed or resume.get("mode") != mode or
                resume.get("scaler_sha256") != scaler_sha):
            raise RuntimeError(f"head resume binding mismatch: {unit}")
        if not rows_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"resume is missing history or best checkpoint: {unit}")
        history = json.loads(rows_path.read_text(encoding="utf-8"))
        completed_epoch = int(resume["completed_epoch"])
        if ([int(row["epoch"]) for row in history] != list(range(1, completed_epoch + 1)) or
                int(history[-1]["epoch"]) != completed_epoch or
                resume.get("epoch_history_sha256") != sha(rows_path) or
                not isinstance(resume.get("optimizer_state"), dict)):
            raise RuntimeError(f"resume history/optimizer consistency mismatch: {unit}")
        head = torch.nn.Linear(1024, 3).cuda()
        head.load_state_dict(resume["current_state"])
        if state_hash(head.state_dict()) != state_hash(resume["current_state"]):
            raise RuntimeError(f"resume current state mismatch: {unit}")
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.load_state_dict(resume["optimizer_state"])
        if len(optimizer.state) != len(list(head.parameters())):
            raise RuntimeError(f"resume optimizer state incomplete: {unit}")
        if any(float(group["lr"]) != 0.001 or float(group["weight_decay"]) != 0.0001
               for group in optimizer.param_groups):
            raise RuntimeError(f"resume optimizer settings mismatch: {unit}")
        start_epoch = completed_epoch + 1
        best_ap = float(resume["best_validation_twa_auprc"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        stale = int(resume["stale_epochs"])
        best_file = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (best_epoch < 1 or best_epoch > completed_epoch or
                state_hash(best_state) != state_hash(best_file["state_dict"]) or
                int(best_file.get("epoch", -1)) != best_epoch or
                best_file.get("protocol_sha256") != p["protocol_sha256"] or
                best_file.get("scaler_sha256") != scaler_sha or
                not np.isclose(float(history[best_epoch - 1]["validation_twa_auprc"]), best_ap, rtol=0, atol=1e-12)):
            raise RuntimeError(f"resume best-state/history consistency mismatch: {unit}")
        prior_elapsed = float(resume.get("elapsed_seconds_total", 0.0))
        stop_reason = str(resume.get("stop_reason", "max_epochs"))
    else:
        head = torch.nn.Linear(1024, 3).cuda()
        head.load_state_dict(init_state)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
        start_epoch, best_ap, best_epoch, best_state, stale, history = 1, -np.inf, 0, None, 0, []
        prior_elapsed, stop_reason = 0.0, "max_epochs"

    positives = train_y.sum(axis=0).astype(np.float64)
    positive_weights = np.minimum((len(train_y) - positives) / positives, 20.0).astype(np.float32)
    pos_weight_t = torch.tensor(positive_weights, dtype=torch.float32, device="cuda")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    started = time.perf_counter()
    loop_start = 51 if stale >= 5 or start_epoch > 50 else start_epoch
    if stale >= 5:
        stop_reason = "early_stopping_patience_5"
    elif start_epoch > 50:
        stop_reason = "max_epochs"
    for epoch in range(loop_start, 51):
        head.train()
        order_seed = digest_int(hashlib.sha256(f"{seed}\0{epoch}\0shuffle".encode()).digest())
        order = np.random.default_rng(order_seed).permutation(len(train_ids))
        loss_sum, seen = 0.0, 0
        for j in range(0, len(order), BATCH_TRAIN):
            idx = order[j:j + BATCH_TRAIN]
            views = ["clean"] * len(idx) if mode == "clean_head" else [train_view(seed, epoch, train_ids[int(i)]) for i in idx]
            xb_rows = [train_views[v][int(i)] for i, v in zip(idx, views)]
            xb = np.stack(xb_rows).astype(np.float32, copy=False)
            xb = np.ascontiguousarray((xb - mean) / std)
            yb = train_y[idx].astype(np.float32, copy=False)
            optimizer.zero_grad(set_to_none=True)
            logits = head(torch.from_numpy(xb).cuda())
            loss = criterion(logits, torch.from_numpy(yb).cuda())
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        vp = predict(head, val_features, mean, std)
        ap = float(average_precision_score(val_y[:, 0], vp[:, 0]))
        row = {"epoch": epoch, "train_weighted_bce": loss_sum / seen, "validation_twa_auprc": ap}
        history.append(row)
        if ap > best_ap:  # strict > preserves the earlier epoch on a tie
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            atomic_torch(checkpoint_path, {"dataset": dataset, "seed": seed, "mode": mode,
                                           "epoch": best_epoch, "validation_twa_auprc": best_ap,
                                           "state_dict": best_state, "initial_state_sha256": init_sha,
                                           "protocol_sha256": p["protocol_sha256"],
                                           "scaler_sha256": scaler_sha,
                                           "encoder_checkpoint_sha256": p["source"]["checkpoint_sha256"]})
        else:
            stale += 1
        if stale >= 5:
            stop_reason = "early_stopping_patience_5"
        rows_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        atomic_torch(resume_path, {"dataset": dataset, "seed": seed, "mode": mode,
                                   "protocol_sha256": p["protocol_sha256"],
                                   "initial_state_sha256": init_sha, "completed_epoch": epoch,
                                   "scaler_sha256": scaler_sha,
                                   "current_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                                   "optimizer_state": optimizer.state_dict(),
                                   "best_validation_twa_auprc": best_ap, "best_epoch": best_epoch,
                                   "best_state": best_state, "stale_epochs": stale,
                                   "stop_reason": stop_reason,
                                   "elapsed_seconds_total": prior_elapsed + (time.perf_counter() - started),
                                   "epoch_history_sha256": sha(rows_path)})
        status("TRAINING_HEAD", dataset=dataset, seed=seed, mode=mode, epoch=epoch,
               best_epoch=best_epoch, best_validation_twa_auprc=best_ap, patience_remaining=max(0, 5-stale))
        if stale >= 5:
            break

    best, head = load_checkpoint_head(checkpoint_path)
    vp = predict(head, val_features, mean, std)
    thresholds = np.asarray([np.quantile(vp[val_y[:, j] == 0, j], 0.90, method="linear") for j in range(3)], dtype=np.float64)
    val_pred_path = OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz"
    val_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(val_pred_path, p=vp, y=val_y.astype(np.uint8), record_ids=np.asarray(val_ids),
                        thresholds=thresholds, endpoints=np.asarray(LABELS), head_sha256=sha(checkpoint_path),
                        encoder_sha256=p["source"]["checkpoint_sha256"], protocol_sha256=p["protocol_sha256"],
                        scaler_sha256=scaler_sha)
    result = {"dataset": dataset, "seed": seed, "mode": mode, "best_epoch": int(best_epoch),
              "last_epoch": int(history[-1]["epoch"]), "best_validation_twa_auprc": float(best_ap),
              "validation_twa_auroc": float(roc_auc_score(val_y[:, 0], vp[:, 0])),
              "thresholds": thresholds.tolist(), "positive_weights": positive_weights.tolist(),
              "initial_state_sha256": init_sha, "best_checkpoint_sha256": sha(checkpoint_path),
              "validation_predictions_sha256": sha(val_pred_path), "stop_reason": stop_reason,
              "scaler_sha256": scaler_sha,
              "epoch_history_sha256": sha(rows_path),
              "elapsed_seconds_total": prior_elapsed + (time.perf_counter() - started),
              "protocol_sha256": p["protocol_sha256"], "encoder_frozen": True}
    atomic_json(unit / "training_result.json", result)
    return result


def train_heads(p: dict) -> list[dict]:
    results = []
    for dataset in COHORTS:
        train = records(dataset, "train")
        val = records(dataset, "validation")
        train_ids = [r["record_id"] for r in train]
        val_ids = [r["record_id"] for r in val]
        train_y = founder.label_matrix(dataset, "train").astype(np.uint8)
        val_y = founder.label_matrix(dataset, "validation").astype(np.uint8)
        clean_train = ensure_features(dataset, "train", "clean", p)
        val_x = ensure_features(dataset, "validation", "clean", p)
        mean = np.asarray(clean_train.mean(axis=0, dtype=np.float64), dtype=np.float32)
        std = np.asarray(clean_train.std(axis=0, dtype=np.float64), dtype=np.float32)
        std = np.maximum(std, 1e-6)
        scaling_path = OUT / "scaling" / f"{dataset.lower()}_clean_train.npz"
        scaling_path.parent.mkdir(parents=True, exist_ok=True)
        if scaling_path.exists():
            with np.load(scaling_path, allow_pickle=False) as z:
                old_mean, old_std = z["mean"], z["std"]
                old_ids_hash = str(z["train_ids_sha256"].item())
                old_features_hash = str(z["feature_sha256"].item())
                old_protocol_hash = str(z["protocol_sha256"].item())
            if (not np.array_equal(mean, old_mean) or not np.array_equal(std, old_std) or
                    old_ids_hash != id_digest(train_ids) or
                    old_features_hash != sha(feature_path(dataset, "train", "clean")) or
                    old_protocol_hash != p["protocol_sha256"]):
                raise RuntimeError("clean-train feature standardization cache changed")
        else:
            np.savez_compressed(scaling_path, mean=mean, std=std,
                                train_ids_sha256=id_digest(train_ids), feature_sha256=sha(feature_path(dataset, "train", "clean")),
                                protocol_sha256=p["protocol_sha256"])
        mean, std, scaler_sha = load_scaler(dataset, p)
        views = {"clean": clean_train}
        for condition in TRAIN_VIEWS:
            views[condition] = ensure_features(dataset, "train", condition, p)
        # Both modes for every seed start from the same saved initialization.
        for seed in SEEDS:
            pair_init = OUT / "heads" / dataset.lower() / f"seed_{seed}" / "paired_initial_state.pt"
            if not pair_init.exists():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                init = torch.nn.Linear(1024, 3).state_dict()
                atomic_torch(pair_init, {"seed": seed, "state_dict": {k: v.cpu() for k, v in init.items()},
                                         "initialized_utc": now()})
            for mode in MODES:
                results.append(train_unit(dataset, seed, mode, p, mean, std, train_y, val_y,
                                          train_ids, val_ids, views, val_x))
        status("COHORT_HEADS_COMPLETE", dataset=dataset, scaler_sha256=scaler_sha)
        del views, clean_train, val_x
        torch.cuda.empty_cache()
    rows = [json.loads(x.read_text(encoding="utf-8")) for x in sorted((OUT / "heads").glob("*/seed_*/*/training_result.json"))]
    if len(rows) != 12:
        raise RuntimeError(f"expected twelve completed head trainings, got {len(rows)}")
    keys = {(r["dataset"], r["seed"], r["mode"]) for r in rows}
    if len(keys) != 12:
        raise RuntimeError("duplicate or missing head training results")
    for dataset in COHORTS:
        train_ids = [r["record_id"] for r in records(dataset, "train")]
        val_ids = [r["record_id"] for r in records(dataset, "validation")]
        train_y = founder.label_matrix(dataset, "train").astype(np.uint8)
        val_y = founder.label_matrix(dataset, "validation").astype(np.uint8)
        mean, std, _ = load_scaler(dataset, p)
        val_x = np.load(feature_path(dataset, "validation", "clean"), mmap_mode="r", allow_pickle=False)
        for seed in SEEDS:
            for mode in MODES:
                unit = OUT / "heads" / dataset.lower() / f"seed_{seed}" / mode
                paths = [unit / "best.pt", unit / "training_result.json", unit / "epoch_history.json",
                         OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz"]
                before_hashes = [sha(path) for path in paths]
                train_unit(dataset, seed, mode, p, mean, std, train_y, val_y, train_ids, val_ids, {}, val_x)
                after_hashes = [sha(path) for path in paths]
                if before_hashes != after_hashes:
                    raise RuntimeError(f"completed-head repeat-run changed artifacts: {dataset}/{seed}/{mode}")
                status("HEAD_IDEMPOTENCY_CHECK", dataset=dataset, seed=seed, mode=mode,
                       unchanged=True, optimizer_steps_added=0)
        del val_x
    status("ALL_HEADS_TRAINED", completed=12)
    return rows


def replay_test_cache(dataset: str, condition: str, net, p: dict) -> tuple[bool, float]:
    old_f, _, old_meta_path = founder.cache_paths(dataset, "test", condition)
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    ids = [r["record_id"] for r in records(dataset, "test")]
    candidate = p["verified_existing_test_cache_reuse_candidates"][f"{dataset}/{condition}"]
    if (old_meta.get("record_ids_sha256") != id_digest(ids) or
            old_meta.get("checkpoint_sha256") != p["source"]["checkpoint_sha256"] or
            old_meta.get("protocol_sha256") != sha(founder.CFG) or
            sha(old_f) != old_meta.get("features_sha256") or
            sha(old_f) != candidate["features_sha256"]):
        return False, float("inf")
    subset = records(dataset, "test")[:2 * BATCH_INFER]
    prepared = []
    for row in subset:
        verify_input(dataset, "test", row["record_id"], p)
        raw = founder.physical_signal(dataset, row["record_id"])
        prepared.append(prepare_condition(raw, condition))
    chunks = []
    for start in range(0, len(prepared), BATCH_INFER):
        f, _ = founder.infer(net, np.stack(prepared[start:start + BATCH_INFER]))
        chunks.append(f)
    replay = np.concatenate(chunks)
    old = np.load(old_f, mmap_mode="r", allow_pickle=False)[:len(replay)]
    diff = float(np.max(np.abs(replay - old)))
    return bool(np.allclose(replay, old, rtol=0, atol=TOL)), diff


def copy_replayed_test_cache(dataset: str, condition: str, p: dict) -> np.ndarray:
    src, _, src_meta_path = founder.cache_paths(dataset, "test", condition)
    src_meta = json.loads(src_meta_path.read_text(encoding="utf-8"))
    ids = [r["record_id"] for r in records(dataset, "test")]
    candidate = p["verified_existing_test_cache_reuse_candidates"][f"{dataset}/{condition}"]
    if (sha(src) != candidate["features_sha256"] or sha(src_meta_path) != candidate["metadata_sha256"] or
            src_meta.get("record_ids_sha256") != id_digest(ids)):
        raise RuntimeError(f"legacy test feature cache changed after its replay gate: {dataset}/{condition}")
    target = feature_path(dataset, "test", condition)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(src):
            raise RuntimeError(f"copied test cache differs from source: {target}")
    else:
        shutil.copyfile(src, target)
    atomic_json(feature_meta_path(dataset, "test", condition), {
        "dataset": dataset, "partition": "test", "condition": condition,
        "source": str(src.relative_to(ROOT)), "source_cache_sha256": sha(src),
        "features_sha256": sha(target), "checkpoint_sha256": p["source"]["checkpoint_sha256"],
        "encoder_protocol_sha256": sha(founder.CFG), "mitigation_protocol_sha256": p["protocol_sha256"],
        "record_ids_sha256": id_digest(ids), "record_count": len(ids), "feature_shape": [len(ids), 1024],
        "verified_reuse": True, "same_context_replay_atol": TOL, "batch_size": BATCH_INFER})
    return np.load(target, mmap_mode="r", allow_pickle=False)


def test_conditions(p: dict):
    mask_rows = p["mask_panel"]
    out = ["clean", *TRAIN_VIEWS, "cycle_slots_120", "cycle_slots_201"]
    out += [r["name"] for r in mask_rows]
    return [(condition, None) if not condition.startswith("mask5_") else
            (condition, tuple(next(r["indices_zeroed"] for r in mask_rows if r["name"] == condition)))
            for condition in out]


def audit_mask_optimization(p: dict) -> None:
    audit_path = OUT / "audits" / "MASK_OPTIMIZATION_TRAIN_VALIDATION.json"
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") == p["protocol_sha256"] and old.get("status") == "PASS":
            return
    results = []
    net = founder.model()
    before = state_hash(net.state_dict())
    for dataset in COHORTS:
        for partition in ("train", "validation"):
            rows = records(dataset, partition)[:BATCH_INFER]
            raw_batch, official_inputs = [], []
            for row in rows:
                verify_input(dataset, partition, row["record_id"], p)
                raw = founder.physical_signal(dataset, row["record_id"])
                raw_batch.append(raw)
                official_inputs.append(founder.preprocess(raw))
            official = np.stack(official_inputs).astype(np.float32)
            for mask in p["mask_panel"]:
                name, indices = mask["name"], tuple(mask["indices_zeroed"])
                old_inputs = np.stack([prepare_condition(raw, "clean", indices) for raw in raw_batch])
                new_inputs = official.copy()
                new_inputs[:, list(indices), :] = 0.0
                input_diff = float(np.max(np.abs(old_inputs - new_inputs)))
                retained = [i for i in range(12) if i not in indices]
                channels_unchanged = bool(np.array_equal(new_inputs[:, retained], official[:, retained]))
                old_features, _ = founder.infer(net, old_inputs)
                new_features, _ = founder.infer(net, new_inputs)
                feature_diff = float(np.max(np.abs(old_features - new_features)))
                results.append({"dataset": dataset, "partition": partition, "condition": name,
                                "encoder_input_max_abs_difference": input_diff,
                                "feature_max_abs_difference": feature_diff,
                                "retained_channels_exactly_unchanged": channels_unchanged,
                                "input_passed": bool(input_diff == 0.0),
                                "feature_passed": bool(np.allclose(old_features, new_features, rtol=0, atol=TOL))})
    after = state_hash(net.state_dict())
    passed = (before == after and len(results) == 4 * 32 and all(x["input_passed"] and x["feature_passed"] and
                                                                 x["retained_channels_exactly_unchanged"] for x in results))
    atomic_json(audit_path, {"status": "PASS" if passed else "FAIL", "protocol_sha256": p["protocol_sha256"],
                             "records_per_batch": BATCH_INFER, "batch_size": BATCH_INFER,
                             "masks_checked": len(p["mask_panel"]), "comparisons": results,
                             "encoder_state_sha256_before": before, "encoder_state_sha256_after": after,
                             "verified_utc": now()})
    del net
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("train/validation mask optimization equivalence failed; use old path")


def audit_scaled_prediction_path(p: dict) -> None:
    audit_path = OUT / "audits" / "SCALER_PREDICTION_PATH.json"
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") == p["protocol_sha256"] and old.get("status") == "PASS":
            return
    rows = []
    for dataset in COHORTS:
        mean, std, scale_sha = load_scaler(dataset, p)
        for seed in SEEDS:
            for mode in MODES:
                unit = OUT / "heads" / dataset.lower() / f"seed_{seed}" / mode
                checkpoint, head = load_checkpoint_head(unit / "best.pt")
                if checkpoint.get("scaler_sha256") != scale_sha:
                    raise RuntimeError(f"head scaler binding mismatch before test: {dataset}/{seed}/{mode}")
                for partition in ("train", "validation"):
                    features = np.load(feature_path(dataset, partition, "clean"), mmap_mode="r", allow_pickle=False)
                    n = min(32, len(features))
                    through_path = predict(head, features[:n], mean, std, batch=16)
                    direct_rows = []
                    with torch.inference_mode():
                        for start in range(0, n, 16):
                            x = torch.from_numpy(np.ascontiguousarray(
                                (np.asarray(features[start:start + 16], dtype=np.float32)-mean)/std)).cuda()
                            direct_rows.append(torch.sigmoid(head(x)).float().cpu().numpy())
                    direct = np.concatenate(direct_rows, axis=0)
                    diff = float(np.max(np.abs(through_path - direct)))
                    rows.append({"dataset": dataset, "seed": seed, "mode": mode, "partition": partition,
                                 "path_vs_direct_scaled_max_abs_difference": diff,
                                 "scaler_sha256": scale_sha, "single_standardization": True,
                                 "passed": bool(np.allclose(through_path, direct, rtol=0, atol=1e-7))})
                # Every saved validation probability is reproduced using best.pt and the same scaler.
                val_x = np.load(feature_path(dataset, "validation", "clean"), mmap_mode="r", allow_pickle=False)
                replay = predict(head, val_x, mean, std)
                val_path = OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz"
                with np.load(val_path, allow_pickle=False) as z:
                    saved = z["p"]
                val_diff = float(np.max(np.abs(replay - saved)))
                if val_diff > 1e-7:
                    raise RuntimeError(f"post-training validation replay failed: {dataset}/{seed}/{mode}; {val_diff}")
                rows.append({"dataset": dataset, "seed": seed, "mode": mode, "partition": "saved_validation_replay",
                             "max_abs_probability_difference": val_diff, "scaler_sha256": scale_sha,
                             "passed": True})
                del head
    passed = bool(rows) and all(r["passed"] for r in rows)
    atomic_json(audit_path, {"status": "PASS" if passed else "FAIL", "protocol_sha256": p["protocol_sha256"],
                             "standardization": "raw 1024-D feature -> (feature-mean)/std exactly once inside predict",
                             "comparison_rows": rows, "verified_utc": now()})
    if not passed:
        raise RuntimeError("scaled prediction path check failed before test inference")


def ensure_mask_panel_features(dataset: str, p: dict, net) -> dict[str, np.ndarray]:
    rows = records(dataset, "test")
    ids = [r["record_id"] for r in rows]
    mask_rows = p["mask_panel"]
    cache_info = []
    arrays = {}
    common_completed = None
    for mask in mask_rows:
        condition, indices = mask["name"], list(mask["indices_zeroed"])
        target, meta_path = feature_path(dataset, "test", condition), feature_meta_path(dataset, "test", condition)
        binding = {"dataset": dataset, "partition": "test", "condition": condition,
                   "record_ids_sha256": id_digest(ids), "record_count": len(ids),
                   "checkpoint_sha256": p["source"]["checkpoint_sha256"],
                   "encoder_protocol_sha256": sha(founder.CFG), "mitigation_protocol_sha256": p["protocol_sha256"],
                   "mask_indices_zeroed": indices, "batch_size": BATCH_INFER,
                   "features_shape": [len(ids), 1024], "dtype": "float32"}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(meta.get(k) != v for k, v in binding.items()):
                raise RuntimeError(f"optimized mask cache binding mismatch: {meta_path}")
            x = np.load(target, mmap_mode="r+" if meta.get("status") != "COMPLETE" else "r", allow_pickle=False)
            if x.shape != (len(ids), 1024) or x.dtype != np.float32:
                raise RuntimeError(f"optimized mask cache shape mismatch: {target}")
            done = len(ids) if meta.get("status") == "COMPLETE" else int(meta.get("completed_records", -1))
            if done < 0 or done > len(ids) or (meta.get("status") == "COMPLETE" and sha(target) != meta.get("features_sha256")):
                raise RuntimeError(f"optimized mask cache progress/hash mismatch: {target}")
        else:
            if target.exists():
                raise RuntimeError(f"unbound optimized mask cache exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            x = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(ids), 1024))
            done = 0
            atomic_json(meta_path, {**binding, "status": "IN_PROGRESS", "completed_records": 0,
                                    "optimization": "one clean preprocess per ECG batch; apply selected mask afterward"})
        if common_completed is None:
            common_completed = done
        elif common_completed != done:
            raise RuntimeError("mask panel caches do not share a common committed prefix")
        arrays[condition] = x
        cache_info.append((condition, x, meta_path, binding))
    if common_completed == len(ids):
        for condition, x, meta_path, binding in cache_info:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_sha = sha(feature_path(dataset, "test", condition))
            if meta.get("status") != "COMPLETE":
                atomic_json(meta_path, {**binding, "status": "COMPLETE", "completed_records": len(rows),
                                        "features_sha256": cache_sha,
                                        "optimization": "one clean preprocess per ECG batch; apply selected mask afterward",
                                        "completed_utc": now()})
            elif meta.get("features_sha256") != cache_sha:
                raise RuntimeError(f"optimized mask cache hash mismatch: {condition}")
        return {name: np.load(feature_path(dataset, "test", name), mmap_mode="r", allow_pickle=False)
                for name, *_ in cache_info}
    state_before = state_hash(net.state_dict())
    for start in range(common_completed, len(rows), BATCH_INFER):
        block = rows[start:start + BATCH_INFER]
        prepared = []
        for row in block:
            verify_input(dataset, "test", row["record_id"], p)
            prepared.append(founder.preprocess(founder.physical_signal(dataset, row["record_id"])))
        clean = np.stack(prepared).astype(np.float32)
        for mask, (condition, x, meta_path, binding) in zip(mask_rows, cache_info):
            masked = clean.copy()
            masked[:, list(mask["indices_zeroed"]), :] = 0.0
            f, _ = founder.infer(net, masked)
            x[start:start + len(block)] = f.astype(np.float32)
        end = start + len(block)
        for _, x, meta_path, binding in cache_info:
            x.flush()
            atomic_json(meta_path, {**binding, "status": "IN_PROGRESS", "completed_records": end,
                                    "optimization": "one clean preprocess per ECG batch; apply selected mask afterward"})
        status("MASK_PANEL_FEATURES", dataset=dataset, completed_records=end, total_records=len(rows),
               mask_count=len(mask_rows))
    state_after = state_hash(net.state_dict())
    if state_before != state_after:
        raise RuntimeError("frozen encoder state changed during optimized mask extraction")
    outputs = {}
    for condition, x, meta_path, binding in cache_info:
        atomic_json(meta_path, {**binding, "status": "COMPLETE", "completed_records": len(rows),
                                "features_sha256": sha(feature_path(dataset, "test", condition)),
                                "optimization": "one clean preprocess per ECG batch; apply selected mask afterward",
                                "completed_utc": now()})
        outputs[condition] = np.load(feature_path(dataset, "test", condition), mmap_mode="r", allow_pickle=False)
    return outputs


def make_test_predictions(p: dict) -> None:
    # All train/validation-only implementation gates must pass before the first test prediction.
    set_phase("feature_preparation")
    audit_mask_optimization(p)
    audit_scaled_prediction_path(p)
    model_order = [(mode, seed) for mode in MODES for seed in SEEDS]
    thresholds_by_dataset, scalers_by_dataset = {}, {}
    heads_by_dataset = {}
    for dataset in COHORTS:
        mean, std, scale_sha = load_scaler(dataset, p)
        scalers_by_dataset[dataset] = (mean, std, scale_sha)
        heads, thresholds = [], []
        for mode, seed in model_order:
            unit = OUT / "heads" / dataset.lower() / f"seed_{seed}" / mode
            train_result = json.loads((unit / "training_result.json").read_text(encoding="utf-8"))
            val_y = founder.label_matrix(dataset, "validation").astype(np.uint8)
            val_ids = [r["record_id"] for r in records(dataset, "validation")]
            val_x = np.load(feature_path(dataset, "validation", "clean"), mmap_mode="r", allow_pickle=False)
            verify_completed_head(unit, dataset, seed, mode, p, mean, std, scale_sha, val_y, val_ids, val_x)
            checkpoint, head = load_checkpoint_head(unit / "best.pt")
            if train_result.get("scaler_sha256") != scale_sha:
                raise RuntimeError(f"training result scaler binding mismatch: {unit}")
            with np.load(OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz", allow_pickle=False) as z:
                thresholds.append(z["thresholds"].astype(np.float64))
            heads.append(head)
        heads_by_dataset[dataset], thresholds_by_dataset[dataset] = heads, np.stack(thresholds)

    for dataset in COHORTS:
        if ACTIVE_PHASE != "feature_preparation":
            set_phase("feature_preparation")
        rows = records(dataset, "test")
        net = founder.model()
        cache_replay = {}
        for cond in ("clean", "RA_LL_reversal"):
            okay, diff = replay_test_cache(dataset, cond, net, p)
            cache_replay[cond] = {"pass": okay, "max_abs_feature_difference": diff,
                                  "atol": TOL, "batch_size": BATCH_INFER}
        (OUT / "audits").mkdir(exist_ok=True)
        atomic_json(OUT / "audits" / f"{dataset.lower()}_old_test_cache_replay.json", cache_replay)
        mask_features = ensure_mask_panel_features(dataset, p, net)
        features_by_condition = {}
        for cond, mask_indices in test_conditions(p):
            if cond.startswith("mask5_"):
                features = mask_features[cond]
            elif cond in ("clean", "RA_LL_reversal") and cache_replay[cond]["pass"]:
                features = copy_replayed_test_cache(dataset, cond, p)
            else:
                features = ensure_features(dataset, "test", cond, p, mask_indices,
                                           force_recompute=(cond == "clean"))
            if len(features) != len(rows):
                raise RuntimeError(f"test feature row mismatch: {dataset}/{cond}")
            features_by_condition[cond] = features
        state_hash_before_release = state_hash(net.state_dict())
        if any(not np.isfinite(features[:min(4, len(features))]).all() for features in features_by_condition.values()):
            raise RuntimeError(f"non-finite test feature cache: {dataset}")
        del net
        torch.cuda.empty_cache()

        # Test labels are read only after the train/validation-only audits passed and features are ready.
        ids = [r["record_id"] for r in rows]
        y = founder.label_matrix(dataset, "test").astype(np.uint8)
        mean, std, scale_sha = scalers_by_dataset[dataset]
        set_phase("analysis")
        for cond, _ in test_conditions(p):
            features = features_by_condition[cond]
            target = OUT / "predictions" / dataset.lower() / f"{cond}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            meta_path = target.with_suffix(".meta.json")
            cache_hash = sha(feature_path(dataset, "test", cond))
            expected_models = [f"{mode}|{seed}" for mode, seed in model_order]
            if target.exists() or meta_path.exists():
                if not target.is_file() or not meta_path.is_file():
                    raise RuntimeError(f"partial prediction bundle exists: {target}")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (meta.get("protocol_sha256") != p["protocol_sha256"] or
                        meta.get("record_ids_sha256") != id_digest(ids) or
                        meta.get("probabilities_sha256") != sha(target) or
                        meta.get("feature_cache_sha256") != cache_hash or
                        meta.get("scaler_sha256") != scale_sha or meta.get("model_order") != expected_models):
                    raise RuntimeError(f"existing prediction binding mismatch: {target}")
                with np.load(target, allow_pickle=False) as z:
                      if (z["probabilities"].shape != (len(rows), len(model_order), 3) or
                            z["protocol_sha256"].item() != p["protocol_sha256"] or
                            z["scaler_sha256"].item() != scale_sha or
                            not np.array_equal(z["record_ids"].astype(str), np.asarray(ids)) or
                            not np.array_equal(z["y"], y) or
                            not np.allclose(z["thresholds"], thresholds_by_dataset[dataset], rtol=0, atol=0) or
                            not np.array_equal(z["model_order"].astype(str), np.asarray(expected_models))):
                        raise RuntimeError(f"existing prediction payload mismatch: {target}")
                status("TEST_CONDITION_REUSED", dataset=dataset, condition=cond,
                       feature_cache_sha256=cache_hash, scaler_sha256=scale_sha)
                continue
            probs = np.empty((len(rows), len(model_order), 3), dtype=np.float32)
            for j, head in enumerate(heads_by_dataset[dataset]):
                probs[:, j] = predict(head, features, mean, std)
            np.savez_compressed(target, probabilities=probs, y=y, record_ids=np.asarray(ids),
                                model_order=np.asarray([f"{mode}|{seed}" for mode, seed in model_order]),
                                thresholds=thresholds_by_dataset[dataset], endpoints=np.asarray(LABELS),
                                feature_cache_sha256=cache_hash, protocol_sha256=p["protocol_sha256"],
                                scaler_sha256=scale_sha)
            atomic_json(meta_path, {
                "dataset": dataset, "condition": cond, "n": len(ids), "record_ids_sha256": id_digest(ids),
                "labels_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                "probabilities_sha256": sha(target), "feature_cache_sha256": cache_hash,
                "encoder_checkpoint_sha256": p["source"]["checkpoint_sha256"],
                "model_order": expected_models, "scaler_sha256": scale_sha,
                "protocol_sha256": p["protocol_sha256"]})
            status("TEST_CONDITION_COMPLETE", dataset=dataset, condition=cond,
                   completed_conditions=sum(1 for _ in (OUT / "predictions" / dataset.lower()).glob("*.npz")),
                   total_conditions=len(test_conditions(p)))
        for head in heads_by_dataset[dataset]:
            del head
        torch.cuda.empty_cache()


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.linspace(0, 1, 16)
    which = np.minimum(np.digitize(p, bins[1:-1], right=False), 14)
    total = 0.0
    for b in range(15):
        m = which == b
        if m.any():
            total += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return total


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "brier": float(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2)),
            "ece15": ece15(y, p), "sensitivity": float(pred[pos].mean()),
            "specificity": float((~pred[neg]).mean()), "predicted_positive_fraction": float(pred.mean())}


def bootstrap_counts(indices: np.ndarray, n: int) -> np.ndarray:
    counts = np.zeros((len(indices), n), dtype=np.uint16)
    for r, ix in enumerate(indices):
        counts[r] = np.bincount(ix, minlength=n).astype(np.uint16)
    return counts


def bootstrap_metric(y: np.ndarray, p: np.ndarray, counts: np.ndarray, metric: str) -> np.ndarray:
    """Vectorized exact bootstrap metrics from the frozen index multiplicities."""
    yy = np.asarray(y, dtype=np.uint8)
    scores = np.asarray(p, dtype=np.float32)
    if metric == "brier":
        losses = (scores.astype(np.float64) - yy.astype(np.float64)) ** 2
        return (counts.astype(np.float64) @ losses) / counts.sum(axis=1)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    weights = counts[:, order]
    group_total = np.add.reduceat(weights, starts, axis=1).astype(np.int64)
    group_positive = np.add.reduceat(weights * yy[order][None, :], starts, axis=1).astype(np.int64)
    positives = group_positive.sum(axis=1)
    negatives = group_total.sum(axis=1) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise RuntimeError("bootstrap replicate has a single class")
    if metric == "auprc":
        cumulative_total = np.cumsum(group_total, axis=1)
        cumulative_positive = np.cumsum(group_positive, axis=1)
        precision = cumulative_positive / np.maximum(cumulative_total, 1)
        return (group_positive * precision).sum(axis=1) / positives
    if metric == "auroc":
        group_negative = group_total - group_positive
        negative_after = negatives[:, None] - np.cumsum(group_negative, axis=1)
        concordant = group_positive * (negative_after + 0.5 * group_negative)
        return concordant.sum(axis=1) / (positives * negatives)
    raise ValueError(metric)


def evaluate(p: dict) -> None:
    model_order = [(mode, seed) for mode in MODES for seed in SEEDS]
    endpoint_rows, condition_lookup = [], {}
    for dataset in COHORTS:
        for cond, _ in test_conditions(p):
            path = OUT / "predictions" / dataset.lower() / f"{cond}.npz"
            meta_path = path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as z:
                probs, y = z["probabilities"], z["y"]
                ids, thresholds = z["record_ids"].astype(str), z["thresholds"]
                expected_ids = [r["record_id"] for r in records(dataset, "test")]
                if probs.shape != (len(ids), 6, 3) or not np.array_equal(ids, expected_ids):
                    raise RuntimeError(f"test prediction binding/order mismatch: {dataset}/{cond}")
                if (z["protocol_sha256"].item() != p["protocol_sha256"] or
                        z["scaler_sha256"].item() != load_scaler(dataset, p)[2] or
                        meta.get("probabilities_sha256") != sha(path) or
                        meta.get("protocol_sha256") != p["protocol_sha256"] or
                        meta.get("record_ids_sha256") != id_digest(expected_ids) or
                        meta.get("feature_cache_sha256") != sha(feature_path(dataset, "test", cond))):
                    raise RuntimeError(f"test prediction provenance mismatch: {dataset}/{cond}")
                condition_lookup[(dataset, cond)] = (y, probs, thresholds)
                for model_i, (mode, seed) in enumerate(model_order):
                    for j, endpoint in enumerate(LABELS):
                        m = metrics(y[:, j], probs[:, model_i, j], float(thresholds[model_i, j]))
                        endpoint_rows.append({"cohort": dataset, "mode": mode, "seed": seed,
                                              "condition": cond, "endpoint": endpoint,
                                              "threshold": float(thresholds[model_i, j]), **m})
    import pandas as pd
    df = pd.DataFrame(endpoint_rows)
    df.to_csv(OUT / "TEST_METRICS_BY_SEED.csv", index=False, float_format="%.12g")
    value_cols = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "threshold"]
    summary = df.groupby(["cohort", "mode", "condition", "endpoint"], as_index=False)[value_cols].agg(["mean", "std"])
    summary.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in summary.columns]
    summary = summary.reset_index(drop=True)
    # Deltas from clean are calculated within the same seed and mode.
    clean = df[df.condition == "clean"].set_index(["cohort", "mode", "seed", "endpoint"])
    for metric in value_cols:
        if metric == "threshold":
            continue
        df[f"delta_{metric}_from_clean"] = [getattr(r, metric) - clean.loc[(r.cohort, r.mode, r.seed, r.endpoint), metric]
                                            for r in df.itertuples(index=False)]
    df.to_csv(OUT / "TEST_METRICS_BY_SEED.csv", index=False, float_format="%.12g")
    delta_cols = [f"delta_{m}_from_clean" for m in value_cols if m != "threshold"]
    delta_summary = df.groupby(["cohort", "mode", "condition", "endpoint"], as_index=False)[delta_cols].agg(["mean", "std"])
    delta_summary.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in delta_summary.columns]
    delta_summary.to_csv(OUT / "TEST_METRIC_DELTAS_FROM_CLEAN.csv", index=False, float_format="%.12g")
    summary.to_csv(OUT / "TEST_METRICS_MEAN_SD.csv", index=False, float_format="%.12g")

    # Fixed conditions include clean, all three training reversals, both held-out cycles,
    # and the exact five-channel mask. The 32-mask panel mean is a separate estimand.
    fixed_conditions = ["clean", *TRAIN_VIEWS, "cycle_slots_120", "cycle_slots_201", "mask5_0_1_2_3_5"]
    bootstrap_rows = []
    for dataset in COHORTS:
        y, _, _ = condition_lookup[(dataset, "clean")]
        bootstrap_path = ROOT / p["test_evaluation"]["bootstrap"]["indices"][dataset]
        boot = np.load(bootstrap_path, mmap_mode="r", allow_pickle=False)
        if boot.shape[1] != len(y) or len(boot) != 1000:
            raise RuntimeError(f"bootstrap indices mismatch: {dataset}")
        counts = bootstrap_counts(boot, len(y))
        for condition in fixed_conditions:
            y, probs, thresholds = condition_lookup[(dataset, condition)]
            for metric in ("auprc", "auroc", "brier"):
                seed_diffs, full_diffs = [], []
                for seed_i, seed in enumerate(SEEDS):
                    clean_i = model_order.index(("clean_head", seed))
                    phys_i = model_order.index(("physics_head", seed))
                    a = probs[:, phys_i, 0]
                    b = probs[:, clean_i, 0]
                    if metric == "brier":
                        fn = lambda pp, yy: float(np.mean((pp.astype(np.float64)-yy.astype(np.float64))**2))
                        full = fn(a, y[:, 0]) - fn(b, y[:, 0])
                    elif metric == "auprc":
                        full = average_precision_score(y[:, 0], a) - average_precision_score(y[:, 0], b)
                    else:
                        full = roc_auc_score(y[:, 0], a) - roc_auc_score(y[:, 0], b)
                    full_diffs.append(float(full))
                    a_boot = bootstrap_metric(y[:, 0], a, counts, metric)
                    b_boot = bootstrap_metric(y[:, 0], b, counts, metric)
                    seed_diffs.append(a_boot - b_boot)
                draws = np.mean(np.stack(seed_diffs), axis=0)
                bootstrap_rows.append({"cohort": dataset, "contrast": "physics_head_minus_clean_head",
                                       "condition": condition, "endpoint": "TWA", "metric": metric,
                                       "full_cohort_point_difference": float(np.mean(full_diffs)),
                                       "bootstrap_mean": float(draws.mean()), "ci95_low": float(np.quantile(draws, .025)),
                                       "ci95_high": float(np.quantile(draws, .975)), "replicates": 1000,
                                       "seeds_averaged_within_replicate": True, "paired_synchronized": True})
        # Mean AUPRC over the preselected 32 masks, averaging seed differences inside each draw.
        panel = [r["name"] for r in p["mask_panel"]]
        per_seed_draws, full_seed = [], []
        for seed in SEEDS:
            clean_i = model_order.index(("clean_head", seed))
            phys_i = model_order.index(("physics_head", seed))
            mask_draw_diffs, full_mask_diffs = [], []
            for condition in panel:
                y0, pp, _ = condition_lookup[(dataset, condition)]
                pa, pb = pp[:, phys_i, 0], pp[:, clean_i, 0]
                full_mask_diffs.append(average_precision_score(y0[:, 0], pa) - average_precision_score(y0[:, 0], pb))
                a_draw = bootstrap_metric(y0[:, 0], pa, counts, "auprc")
                b_draw = bootstrap_metric(y0[:, 0], pb, counts, "auprc")
                mask_draw_diffs.append(a_draw - b_draw)
            per_seed_draws.append(np.mean(mask_draw_diffs, axis=0))
            full_seed.append(float(np.mean(full_mask_diffs)))
        draws = np.mean(np.stack(per_seed_draws), axis=0)
        bootstrap_rows.append({"cohort": dataset, "contrast": "physics_head_minus_clean_head",
                               "condition": "mean_32_mask_panel", "endpoint": "TWA", "metric": "auprc",
                               "full_cohort_point_difference": float(np.mean(full_seed)),
                               "bootstrap_mean": float(draws.mean()), "ci95_low": float(np.quantile(draws, .025)),
                               "ci95_high": float(np.quantile(draws, .975)), "replicates": 1000,
                               "seeds_averaged_within_replicate": True, "paired_synchronized": True})
        status("BOOTSTRAP_COMPLETE", cohort=dataset, groups=len(bootstrap_rows))
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "TWA_PAIRED_BOOTSTRAP.csv", index=False, float_format="%.12g")
    status("EVALUATION_COMPLETE", metric_rows=len(df), bootstrap_groups=len(bootstrap_rows))


def main() -> None:
    global PREVIOUS_RUNTIME, SESSION_ID, SESSION_STARTED, PHASE_STARTED
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train_features", "heads", "test", "evaluate", "all"), default="all")
    args = ap.parse_args()
    runtime_path = OUT / "CONTINUATION_RUNTIME.json"
    if runtime_path.exists():
        PREVIOUS_RUNTIME = json.loads(runtime_path.read_text(encoding="utf-8"))
    SESSION_ID = uuid.uuid4().hex
    SESSION_STARTED = time.perf_counter()
    PHASE_STARTED = SESSION_STARTED
    p = protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.stage in ("train_features", "all"):
        set_phase("feature_preparation")
        prepare_training_features(p)
    if args.stage in ("heads", "all"):
        set_phase("head_training")
        train_heads(p)
    if args.stage in ("test", "all"):
        if len(list((OUT / "heads").glob("*/seed_*/*/training_result.json"))) != 12:
            raise RuntimeError("all twelve heads must be trained before test inference")
        make_test_predictions(p)
    if args.stage in ("evaluate", "all"):
        if not all((OUT / "predictions" / dataset.lower()).is_dir() for dataset in COHORTS):
            raise RuntimeError("test predictions are not complete")
        if any(not all((OUT / "predictions" / dataset.lower() / f"{condition}.npz").is_file()
                       for condition, _ in test_conditions(p)) for dataset in COHORTS):
            raise RuntimeError("not all 38 test conditions per cohort are present")
        set_phase("analysis")
        evaluate(p)
    status("SESSION_FINISHED", requested_stage=args.stage)


if __name__ == "__main__":
    main()
