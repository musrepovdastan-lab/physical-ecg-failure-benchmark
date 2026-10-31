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
    os.replace(tmp, path)


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def protocol() -> dict:
    p = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    digest = p.pop("protocol_sha256")
    if hash_obj(p) != digest:
        raise RuntimeError("frozen protocol checksum mismatch")
    p["protocol_sha256"] = digest
    if p["status"] != "FROZEN_BEFORE_NEW_TEST_OUTCOMES":
        raise RuntimeError("protocol is not frozen before new test outcomes")
    if sha(founder.CHECKPOINT) != p["source"]["checkpoint_sha256"]:
        raise RuntimeError("Founder checkpoint changed after freeze")
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
    if expected is None:
        raise RuntimeError(f"record was not frozen in protocol: {key}")
    hea, mat = record_hash_paths(dataset, rid)
    if sha(hea) != expected["header_sha256"] or sha(mat) != expected["waveform_sha256"]:
        raise RuntimeError(f"raw waveform/header changed after protocol freeze: {key}")


def feature_path(dataset: str, partition: str, condition: str) -> Path:
    tag = condition.replace("/", "_")
    return OUT / "features" / f"{dataset.lower()}_{partition}_{tag}.npy"


def feature_meta_path(dataset: str, partition: str, condition: str) -> Path:
    return feature_path(dataset, partition, condition).with_suffix(".meta.json")


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
    if not np.array_equal(transformed[6:], raw[6:]):
        if not np.allclose(transformed[6:], raw[6:], rtol=0, atol=1e-10):
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
    if meta.get("protocol_sha256") != sha(founder.CFG) or sha(src) != meta.get("features_sha256"):
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
    obj = {"stage": stage, "updated_utc": now(), **kw}
    atomic_json(OUT / "WORKFLOW_STATUS.json", obj)
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def prepare_training_features(p: dict) -> None:
    for dataset in COHORTS:
        ensure_features(dataset, "train", "clean", p)
        ensure_features(dataset, "validation", "clean", p)
        for condition in TRAIN_VIEWS:
            ensure_features(dataset, "train", condition, p)
    status("TRAIN_FEATURES_COMPLETE")


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
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume.get("protocol_sha256") != p["protocol_sha256"] or resume.get("initial_state_sha256") != init_sha:
            raise RuntimeError(f"head resume binding mismatch: {unit}")
        head = torch.nn.Linear(1024, 3).cuda()
        head.load_state_dict(resume["current_state"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["completed_epoch"]) + 1
        best_ap = float(resume["best_validation_twa_auprc"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        stale = int(resume["stale_epochs"])
        history = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
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
    for epoch in range(start_epoch, 51):
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
                                           "encoder_checkpoint_sha256": p["source"]["checkpoint_sha256"]})
        else:
            stale += 1
        if stale >= 5:
            stop_reason = "early_stopping_patience_5"
        rows_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        atomic_torch(resume_path, {"dataset": dataset, "seed": seed, "mode": mode,
                                   "protocol_sha256": p["protocol_sha256"],
                                   "initial_state_sha256": init_sha, "completed_epoch": epoch,
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

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head.load_state_dict(best["state_dict"])
    vp = predict(head, val_features, mean, std)
    thresholds = np.asarray([np.quantile(vp[val_y[:, j] == 0, j], 0.90, method="linear") for j in range(3)], dtype=np.float64)
    val_pred_path = OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz"
    val_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(val_pred_path, p=vp, y=val_y.astype(np.uint8), record_ids=np.asarray(val_ids),
                        thresholds=thresholds, endpoints=np.asarray(LABELS), head_sha256=sha(checkpoint_path),
                        encoder_sha256=p["source"]["checkpoint_sha256"], protocol_sha256=p["protocol_sha256"])
    result = {"dataset": dataset, "seed": seed, "mode": mode, "best_epoch": int(best_epoch),
              "last_epoch": int(history[-1]["epoch"]), "best_validation_twa_auprc": float(best_ap),
              "validation_twa_auroc": float(roc_auc_score(val_y[:, 0], vp[:, 0])),
              "thresholds": thresholds.tolist(), "positive_weights": positive_weights.tolist(),
              "initial_state_sha256": init_sha, "best_checkpoint_sha256": sha(checkpoint_path),
              "validation_predictions_sha256": sha(val_pred_path), "stop_reason": stop_reason,
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
            if not np.array_equal(mean, old_mean) or not np.array_equal(std, old_std):
                raise RuntimeError("clean-train feature standardization cache changed")
        else:
            np.savez_compressed(scaling_path, mean=mean, std=std,
                                train_ids_sha256=id_digest(train_ids), feature_sha256=sha(feature_path(dataset, "train", "clean")),
                                protocol_sha256=p["protocol_sha256"])
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
        del views, clean_train, val_x
        torch.cuda.empty_cache()
    rows = [json.loads(x.read_text(encoding="utf-8")) for x in sorted((OUT / "heads").glob("*/seed_*/*/training_result.json"))]
    if len(rows) != 12:
        raise RuntimeError(f"expected twelve completed head trainings, got {len(rows)}")
    keys = {(r["dataset"], r["seed"], r["mode"]) for r in rows}
    if len(keys) != 12:
        raise RuntimeError("duplicate or missing head training results")
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


def make_test_predictions(p: dict) -> None:
    model_order = [(mode, seed) for mode in MODES for seed in SEEDS]
    thresholds_by_dataset = {}
    heads_by_dataset = {}
    for dataset in COHORTS:
        heads, thresholds = [], []
        for mode, seed in model_order:
            unit = OUT / "heads" / dataset.lower() / f"seed_{seed}" / mode
            checkpoint = torch.load(unit / "best.pt", map_location="cpu", weights_only=False)
            head = torch.nn.Linear(1024, 3).cuda().eval()
            head.load_state_dict(checkpoint["state_dict"])
            for pp in head.parameters():
                pp.requires_grad_(False)
            with np.load(OUT / "validation_predictions" / dataset.lower() / f"seed_{seed}_{mode}.npz", allow_pickle=False) as z:
                thresholds.append(z["thresholds"].astype(np.float64))
            heads.append(head)
        heads_by_dataset[dataset], thresholds_by_dataset[dataset] = heads, np.stack(thresholds)

    for dataset in COHORTS:
        rows = records(dataset, "test")
        ids = [r["record_id"] for r in rows]
        y = founder.label_matrix(dataset, "test").astype(np.uint8)
        net = founder.model()
        cache_replay = {}
        for cond in ("clean", "RA_LL_reversal"):
            okay, diff = replay_test_cache(dataset, cond, net, p)
            cache_replay[cond] = {"pass": okay, "max_abs_feature_difference": diff,
                                  "atol": TOL, "batch_size": BATCH_INFER}
        del net
        torch.cuda.empty_cache()
        (OUT / "audits").mkdir(exist_ok=True)
        atomic_json(OUT / "audits" / f"{dataset.lower()}_old_test_cache_replay.json", cache_replay)

        for cond, mask_indices in test_conditions(p):
            if cond in ("clean", "RA_LL_reversal") and cache_replay[cond]["pass"]:
                features = copy_replayed_test_cache(dataset, cond, p)
            else:
                features = ensure_features(dataset, "test", cond, p, mask_indices,
                                           force_recompute=(cond == "clean"))
            if len(features) != len(rows):
                raise RuntimeError(f"test feature row mismatch: {dataset}/{cond}")
            # Do one frozen encoder cache per condition; the six linear heads share it.
            probs = np.empty((len(rows), len(model_order), 3), dtype=np.float32)
            for j, head in enumerate(heads_by_dataset[dataset]):
                probs[:, j] = predict(head, features, np.zeros(1024, np.float32), np.ones(1024, np.float32))
            target = OUT / "predictions" / dataset.lower() / f"{cond}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, probabilities=probs, y=y, record_ids=np.asarray(ids),
                                model_order=np.asarray([f"{mode}|{seed}" for mode, seed in model_order]),
                                thresholds=thresholds_by_dataset[dataset], endpoints=np.asarray(LABELS),
                                feature_cache_sha256=sha(feature_path(dataset, "test", cond)),
                                protocol_sha256=p["protocol_sha256"])
            atomic_json(target.with_suffix(".meta.json"), {
                "dataset": dataset, "condition": cond, "n": len(ids), "record_ids_sha256": id_digest(ids),
                "labels_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                "probabilities_sha256": sha(target), "feature_cache_sha256": sha(feature_path(dataset, "test", cond)),
                "encoder_checkpoint_sha256": p["source"]["checkpoint_sha256"],
                "model_order": [f"{mode}|{seed}" for mode, seed in model_order],
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
            with np.load(path, allow_pickle=False) as z:
                probs, y = z["probabilities"], z["y"]
                ids, thresholds = z["record_ids"].astype(str), z["thresholds"]
                expected_ids = [r["record_id"] for r in records(dataset, "test")]
                if probs.shape != (len(ids), 6, 3) or not np.array_equal(ids, expected_ids):
                    raise RuntimeError(f"test prediction binding/order mismatch: {dataset}/{cond}")
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
        df[f"delta_{metric}_from_clean"] = [r[metric] - clean.loc[(r.cohort, r.mode, r.seed, r.endpoint), metric]
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train_features", "heads", "test", "evaluate", "all"), default="all")
    args = ap.parse_args()
    p = protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.stage in ("train_features", "all"):
        prepare_training_features(p)
    if args.stage in ("heads", "all"):
        train_heads(p)
    if args.stage in ("test", "all"):
        if len(list((OUT / "heads").glob("*/seed_*/*/training_result.json"))) != 12:
            raise RuntimeError("all twelve heads must be trained before test inference")
        make_test_predictions(p)
    if args.stage in ("evaluate", "all"):
        if not all((OUT / "predictions" / dataset.lower()).is_dir() for dataset in COHORTS):
            raise RuntimeError("test predictions are not complete")
        evaluate(p)


if __name__ == "__main__":
    main()
