"""One disjoint, preregistered HEEDB I0006 confirmation of corrected v2.

Selection never reads outcome columns. Restricted manifests, waveforms, labels,
and predictions stay under data/heedb_restricted. Public progress is aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import heedb_external_v1 as v1
from src import heedb_external_corrected_v2 as v2
from src.ecg_physics import derive_display, named_reversal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_independent_confirmation_v1"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "independent_confirmation_v1"
WAVES = PRIVATE / "waveforms"
V1_PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "external_v1"
READINESS = ROOT / "data" / "heedb_restricted" / "I0006" / "audit_manifests" / "waveform_sample_candidates.csv"
DIAGNOSES = ROOT / "data" / "heedb_restricted" / "I0006" / "12SL_diagnoses" / "diagnoses_acquisition.csv"
PROTOCOL = ROOT / "configs" / "FROZEN_HEEDB_INDEPENDENT_CONFIRMATION_v1.json"
COHORT = PRIVATE / "selection_initial.csv"
CANDIDATES = PRIVATE / "candidate_reserve_order.csv"
OBJECTS = PRIVATE / "waveform_object_manifest.csv"
CAP = 2 * 1024**3
SELECTION_SEED = "heedb_independent_confirmation_v1|2026-09-24|patient_hash"
ARMS = v2.ARMS
SEEDS = v2.SEEDS
LABELS = v2.LABELS
VARIANTS = v2.VARIANTS
BOOT = 1000
BOOT_SEED = 20260924


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def log(stage: str, **counts: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"utc": timestamp(), "stage": stage, **counts}
    with (OUT / "PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def check_upstream() -> dict:
    old = json.loads((ROOT / "outputs" / "heedb_external_v1" / "SELECTION_FREEZE.json").read_text(encoding="utf-8"))
    cache = json.loads((ROOT / "outputs" / "heedb_external_v1" / "WAVEFORM_CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    if digest(V1_PRIVATE / "candidate_patient_order.csv") != old["candidate_manifest_sha256"]:
        raise RuntimeError("v1 candidate source hash changed")
    if digest(V1_PRIVATE / "selection_final.csv") != cache["final_selection_sha256"]:
        raise RuntimeError("v1/v2 evaluation selection hash changed")
    if digest(ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_CORRECTED_V2_PROTOCOL.json") != "c339031084061dddf683f263319fd80c9d87bcec47df781eeeef0a3ed3399838":
        raise RuntimeError("corrected-v2 protocol hash changed")
    return old


def select() -> None:
    old = check_upstream()
    if COHORT.exists() or (OUT / "COHORT_FREEZE.json").exists():
        raise RuntimeError("confirmation cohort already selected; refusing overwrite")
    if not (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines().count("data/heedb_restricted/"):
        raise RuntimeError("restricted HEEDB directory is not git-ignored")
    prior = pd.read_csv(V1_PRIVATE / "selection_final.csv", usecols=["BDSPPatientID"], dtype=str, keep_default_na=False)
    readiness = pd.read_csv(READINESS, usecols=["BDSPPatientID"], dtype=str, keep_default_na=False)
    if len(prior) != 10000 or prior.BDSPPatientID.nunique() != 10000 or readiness.BDSPPatientID.nunique() != 23:
        raise RuntimeError("prior evaluation/readiness exclusion size mismatch")
    # The v1 candidate source already contains one hash-selected recording per
    # physician-code-available patient. Read identifiers/paths ONLY, not labels.
    columns = ["BDSPPatientID", "FileName", "remote_stem"]
    source = pd.read_csv(V1_PRIVATE / "candidate_patient_order.csv", usecols=columns, dtype=str, keep_default_na=False)
    if len(source) != old["eligible_patient_count"] or source.BDSPPatientID.nunique() != len(source):
        raise RuntimeError("source candidate population integrity failed")
    excluded = set(prior.BDSPPatientID) | set(readiness.BDSPPatientID)
    if len(excluded) != 10023:
        raise RuntimeError("prior and readiness patient sets unexpectedly overlap")
    keep = source.loc[~source.BDSPPatientID.isin(excluded)].copy()
    if len(keep) != 339482:
        raise RuntimeError("unexpected disjoint eligible patient count")
    keep["confirmation_hash"] = [hashlib.sha256(f"{SELECTION_SEED}|{patient}".encode()).hexdigest() for patient in keep.BDSPPatientID]
    keep.sort_values(["confirmation_hash", "BDSPPatientID"], kind="mergesort", inplace=True)
    keep.reset_index(drop=True, inplace=True)
    keep.insert(0, "confirmation_rank", np.arange(1, len(keep) + 1, dtype=np.int64))
    selected = keep.iloc[:10000].copy()
    selected.insert(1, "confirmation_slot", np.arange(1, 10001, dtype=np.int64))
    if set(selected.BDSPPatientID) & excluded or selected.BDSPPatientID.nunique() != 10000:
        raise RuntimeError("disjointness/uniqueness gate failed")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    keep.to_csv(CANDIDATES, index=False)
    selected.to_csv(COHORT, index=False)
    save_json(OUT / "COHORT_FREEZE.json", {
        "decision": "FROZEN_BEFORE_OUTCOME_ACCESS",
        "selection_seed": SELECTION_SEED,
        "selection_rule": "v1 label-blind one-record-per-patient candidate; exclude v1/v2 and readiness patients; sort SHA256(seed|patient ID), tie-break patient ID; first 10000; no outcome/rate/quality selection",
        "unreadable_rule": "stop the confirmation; do not replace after the frozen cohort is selected",
        "upstream_joined_recordings": old["joined_recordings"],
        "upstream_missing_patient_id": old["missing_patient_id"],
        "upstream_missing_physician_codes": old["missing_physician_codes_after_patient_filter"],
        "upstream_eligible_patients": old["eligible_patient_count"],
        "excluded_v1_v2_patients": len(prior),
        "excluded_readiness_patients": len(readiness),
        "disjoint_eligible_patients": len(keep),
        "selected_patients": len(selected),
        "v1_candidate_source_sha256": old["candidate_manifest_sha256"],
        "v1_evaluation_selection_sha256": digest(V1_PRIVATE / "selection_final.csv"),
        "readiness_manifest_sha256": digest(READINESS),
        "candidate_reserve_order_sha256": digest(CANDIDATES),
        "selected_manifest_sha256": digest(COHORT),
        "created_utc": timestamp(),
    })
    log("COHORT_FROZEN", eligible=len(keep), selected=len(selected), excluded_prior=len(prior), excluded_readiness=len(readiness))


def frozen() -> tuple[dict, pd.DataFrame]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sidecar = (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip()
    if digest(PROTOCOL) != sidecar or digest(Path(__file__)) != protocol["evaluator_sha256"]:
        raise RuntimeError("confirmation protocol or evaluator hash mismatch")
    for relative, expected in protocol["bound_artifacts_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source artifact changed: {relative}")
    upstream_v2 = json.loads((ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_CORRECTED_V2_PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol["thresholds"] != upstream_v2["thresholds"] or protocol["label_mappings"] != {
        "core": {name: sorted(codes) for name, codes in v1.CORE.items()},
        "strict": {name: sorted(codes) for name, codes in v1.STRICT.items()},
    }:
        raise RuntimeError("frozen threshold or physician-code mapping changed")
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint, preprocessing, validation = v1.model_paths(arm, seed)
            expected = upstream_v2["model_artifact_sha256"][arm][str(seed)]
            if (digest(checkpoint) != expected["checkpoint_sha256"] or
                digest(preprocessing) != expected["preprocessing_sha256"] or
                digest(validation) != expected["validation_threshold_source_sha256"]):
                raise RuntimeError("one of six frozen model/threshold artifacts changed")
    check_upstream()
    freeze = json.loads((OUT / "COHORT_FREEZE.json").read_text(encoding="utf-8"))
    if digest(OUT / "COHORT_FREEZE.json") != protocol["cohort_freeze_sha256"]:
        raise RuntimeError("cohort-freeze record hash mismatch")
    if freeze["selected_manifest_sha256"] != protocol["selected_manifest_sha256"] or digest(COHORT) != freeze["selected_manifest_sha256"]:
        raise RuntimeError("confirmation manifest hash mismatch")
    selected = pd.read_csv(COHORT, dtype=str, keep_default_na=False)
    prior = pd.read_csv(V1_PRIVATE / "selection_final.csv", usecols=["BDSPPatientID"], dtype=str, keep_default_na=False)
    readiness = pd.read_csv(READINESS, usecols=["BDSPPatientID"], dtype=str, keep_default_na=False)
    if len(selected) != 10000 or selected.BDSPPatientID.nunique() != 10000 or set(selected.BDSPPatientID) & (set(prior.BDSPPatientID) | set(readiness.BDSPPatientID)):
        raise RuntimeError("confirmation cohort disjointness failed")
    if protocol["arms"] != list(ARMS) or protocol["seeds"] != list(SEEDS) or protocol["labels"] != list(LABELS):
        raise RuntimeError("arm, seed, or channel order mismatch")
    return protocol, selected


def restricted_path(relative: str) -> Path:
    path = (WAVES / relative).resolve()
    if WAVES.resolve() not in path.parents:
        raise RuntimeError("unsafe waveform relative path")
    return path


def size_check() -> None:
    protocol, selected = frozen()
    if OBJECTS.exists() or (OUT / "TRANSFER_PREFLIGHT.json").exists():
        raise RuntimeError("remote-size preflight already exists; refusing overwrite")
    s3 = v1.require_identity()  # exact non-root account and IAM ARN gate
    rows = []
    for row in selected.itertuples(index=False):
        for suffix in (".hea", ".dat"):
            rows.append({"confirmation_slot": int(row.confirmation_slot), "remote_key": row.remote_stem + suffix,
                         "local_relative_path": row.FileName + suffix})

    def head(index: int, key: str) -> tuple[int, int, str, bool]:
        response = s3.head_object(Bucket=v1.ALIAS, Key=key)
        return index, int(response["ContentLength"]), response.get("StorageClass", "STANDARD"), bool(response.get("RequestCharged"))

    sizes = np.zeros(len(rows), dtype=np.int64)
    classes = set()
    charged = 0
    log("SIZE_CHECK_START", objects=len(rows))
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(head, i, row["remote_key"]) for i, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                index, size, storage_class, request_charged = future.result()
            except Exception as exc:
                raise RuntimeError("remote HEAD failed; no waveform transfer started") from exc
            sizes[index] = size
            classes.add(storage_class)
            charged += int(request_charged)
            if completed % 1000 == 0 or completed == len(rows):
                log("SIZE_CHECK_PROGRESS", completed=completed, total=len(rows))
    if classes != {"STANDARD"} or charged:
        raise RuntimeError("non-Standard storage or requester charge observed; no transfer")
    for row, size in zip(rows, sizes):
        row["remote_bytes"] = int(size)
        local = restricted_path(row["local_relative_path"])
        cached = v1.WAVES / row["local_relative_path"]
        row["local_verified"] = int((local.is_file() and local.stat().st_size == size) or (cached.is_file() and cached.stat().st_size == size))
    table = pd.DataFrame(rows)
    table.to_csv(OBJECTS, index=False)
    remote_total = int(sizes.sum())
    needed = int(table.loc[table.local_verified.eq(0), "remote_bytes"].sum())
    # Public AWS standard-price estimate, excluding any owner discounts/free tier.
    estimated_usd = round(needed / 1e9 * .09 + len(rows) / 1000 * .0004 * 2, 4)
    gate = needed <= CAP and estimated_usd <= .25
    save_json(OUT / "TRANSFER_PREFLIGHT.json", {"objects": len(rows), "patients": len(selected), "exact_remote_bytes": remote_total,
        "exact_new_download_bytes": needed, "reusable_local_objects": int(table.local_verified.sum()), "cap_bytes": CAP,
        "storage_class": "STANDARD", "request_charged_head_responses": charged,
        "charge_assumption": "non-Requester-Pays inferred from successful HEAD without RequestPayer; standard us-east-1 internet egress $0.09/decimal GB plus Standard request charges; bucket owner ordinarily billed",
        "estimated_aws_service_charge_usd_upper_without_discount": estimated_usd, "estimated_charge_ceiling_usd": .25,
        "safe_to_transfer": gate, "restricted_object_manifest_sha256": digest(OBJECTS), "checked_utc": timestamp()})
    log("SIZE_CHECK_COMPLETE", objects=len(rows), remote_bytes=remote_total, new_bytes=needed, within_cap=gate)
    if not gate:
        raise RuntimeError("transfer exceeds the frozen 2 GiB or $0.25 estimate gate; no transfer")


def download_one(s3, row: dict) -> int:
    destination = restricted_path(str(row["local_relative_path"]))
    expected = int(row["remote_bytes"])
    if destination.is_file() and destination.stat().st_size == expected:
        return 0
    cached = v1.WAVES / str(row["local_relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file() and cached.stat().st_size == expected:
        shutil.copy2(cached, destination)
        return 0
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected:
        raise RuntimeError("local partial exceeds remote size")
    if start == expected:
        os.replace(partial, destination)
        return 0
    args = {"Bucket": v1.ALIAS, "Key": str(row["remote_key"])}
    if start:
        args["Range"] = f"bytes={start}-"
    response = s3.get_object(**args)
    if response.get("RequestCharged"):
        raise RuntimeError("unexpected requester charge; stopped")
    received = 0
    with partial.open("ab" if start else "wb") as handle:
        body = response["Body"]
        while True:
            chunk = body.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
    if partial.stat().st_size != expected:
        raise RuntimeError("downloaded size mismatch")
    os.replace(partial, destination)
    return received


def download() -> None:
    _protocol, _selected = frozen()
    preflight = json.loads((OUT / "TRANSFER_PREFLIGHT.json").read_text(encoding="utf-8"))
    if not preflight["safe_to_transfer"] or preflight["exact_new_download_bytes"] > CAP or digest(OBJECTS) != preflight["restricted_object_manifest_sha256"]:
        raise RuntimeError("transfer preflight gate failed")
    if (OUT / "DOWNLOAD_COMPLETE.json").exists():
        raise RuntimeError("download already completed; refusing duplicate job")
    lock = OUT / ".download.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    transferred = 0
    started = time.monotonic()
    try:
        s3 = v1.require_identity()
        table = pd.read_csv(OBJECTS, dtype={"remote_bytes": np.int64})
        rows = table.to_dict("records")
        log("DOWNLOAD_START", objects=len(rows), planned_new_bytes=preflight["exact_new_download_bytes"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(download_one, s3, row) for row in rows]
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    transferred += int(future.result())
                except Exception as exc:
                    raise RuntimeError("bounded transfer failed; partial objects remain resumable") from exc
                if transferred > CAP:
                    raise RuntimeError("2 GiB cumulative transfer cap exceeded")
                if completed % 250 == 0 or completed == len(rows):
                    log("DOWNLOAD_PROGRESS", completed=completed, total=len(rows), downloaded_bytes=transferred)
        if any(not restricted_path(str(row["local_relative_path"])).is_file() or
               restricted_path(str(row["local_relative_path"])).stat().st_size != int(row["remote_bytes"]) for row in rows):
            raise RuntimeError("post-download object-size gate failed")
        save_json(OUT / "DOWNLOAD_COMPLETE.json", {"decision": "PASS", "objects": len(rows), "verified_objects": len(rows),
            "new_download_bytes": transferred, "exact_remote_bytes": preflight["exact_remote_bytes"],
            "restricted_object_manifest_sha256": digest(OBJECTS), "elapsed_seconds": round(time.monotonic() - started, 3), "completed_utc": timestamp()})
        log("DOWNLOAD_COMPLETE", objects=len(rows), downloaded_bytes=transferred)
    finally:
        lock.unlink(missing_ok=True)


def build_cache() -> None:
    _protocol, selected = frozen()
    if (OUT / "WAVEFORM_CACHE_COMPLETE.json").exists():
        raise RuntimeError("waveform cache already built")
    if json.loads((OUT / "DOWNLOAD_COMPLETE.json").read_text(encoding="utf-8"))["decision"] != "PASS":
        raise RuntimeError("download gate failed")
    temp = PRIVATE / "signals_10000_float32.npy.partial"
    final = PRIVATE / "signals_10000_float32.npy"
    if temp.exists() or final.exists():
        raise RuntimeError("cache output already exists; refusing overwrite")
    array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(10000, 12, 5000))
    original_root = v1.WAVES
    try:
        v1.WAVES = WAVES
        for index, row in enumerate(selected.itertuples(index=False), 1):
            try:
                waveform = v1.read_waveform(row.FileName)
            except Exception as exc:
                raise RuntimeError("a frozen selected waveform is unreadable; no outcome-blind replacement was preregistered") from exc
            if waveform.shape != (12, 5000) or not np.isfinite(waveform).all():
                raise RuntimeError("invalid source waveform")
            array[index - 1] = waveform
            if index % 250 == 0 or index == len(selected):
                array.flush()
                log("CACHE_PROGRESS", completed=index, total=len(selected))
    finally:
        v1.WAVES = original_root
    array.flush()
    del array
    os.replace(temp, final)
    save_json(OUT / "WAVEFORM_CACHE_COMPLETE.json", {"decision": "PASS", "patients": len(selected),
        "unreadable_attempts": 0, "replacements": 0, "cache_bytes": final.stat().st_size,
        "cache_sha256": digest(final), "selection_sha256": digest(COHORT), "completed_utc": timestamp()})
    log("CACHE_COMPLETE", patients=len(selected), replacements=0)


def model_inputs(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean, reversed_input = v2.normalized_conditions(raw, mean, std, tuple())
    if clean.shape[1:] != (12, 1000) or reversed_input.shape != clean.shape:
        raise RuntimeError("exact confirmation model input shape failed")
    return clean, reversed_input


def smoke() -> None:
    _protocol, selected = frozen()
    if (OUT / "ENGINEERING_SMOKE.json").exists() or (OUT / "EVALUATION_FREEZE.json").exists():
        raise RuntimeError("smoke/evaluation already started")
    import torch
    readiness = pd.read_csv(READINESS, usecols=["FileName", "BDSPPatientID"], dtype=str, keep_default_na=False)
    if len(readiness) != 23 or set(readiness.BDSPPatientID) & set(selected.BDSPPatientID):
        raise RuntimeError("excluded engineering sample gate failed")
    original_root = v1.WAVES
    try:
        v1.WAVES = ROOT / "data" / "heedb_restricted" / "I0006" / "waveform_sample"
        raw = np.stack([v1.read_waveform(name) for name in readiness.FileName]).astype(np.float32)
    finally:
        v1.WAVES = original_root
    sampled = v2.to_ningbo_time(raw)
    independent = np.concatenate((sampled[:, :2, :], sampled[:, 6:, :]), axis=1)
    expected = derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)
    if sampled.shape != (23, 12, 1000) or not np.array_equal(v2.exact_ra_ll(sampled), expected):
        raise RuntimeError("all-12-lead reversal or exact shape smoke failed")
    checked = 0
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, std, _ = v2.load_frozen_model(arm, seed)
            for array in model_inputs(raw, mean, std):
                p = v1.predict_array(model, array)
                if p.shape != (23, 3) or not np.isfinite(p).all():
                    raise RuntimeError("smoke prediction gate failed")
                checked += 1
            del model
            torch.cuda.empty_cache()
    save_json(OUT / "ENGINEERING_SMOKE.json", {"decision": "PASS", "excluded_records": 23,
        "models": 6, "conditions": 2, "finite_prediction_batches": checked,
        "exact_model_shape": [12, 1000], "all_12_reversed_leads_match_physics_library": True,
        "completed_utc": timestamp()})
    log("ENGINEERING_SMOKE_PASS", records=23, models=6, conditions=2)


def evaluate() -> None:
    protocol, selected = frozen()
    if json.loads((OUT / "ENGINEERING_SMOKE.json").read_text(encoding="utf-8"))["decision"] != "PASS":
        raise RuntimeError("engineering smoke did not pass")
    cache_record = json.loads((OUT / "WAVEFORM_CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    cache_path = PRIVATE / "signals_10000_float32.npy"
    if digest(cache_path) != cache_record["cache_sha256"]:
        raise RuntimeError("confirmation source cache hash mismatch")
    if (OUT / "EVALUATION_FREEZE.json").exists() or (OUT / "INFERENCE_COMPLETE.json").exists():
        raise RuntimeError("evaluation already started; no duplicate run")
    temp = PRIVATE / "predictions_float32.npy.partial"
    final = PRIVATE / "predictions_float32.npy"
    if temp.exists() or final.exists():
        raise RuntimeError("predictions already exist; no duplicate run")
    lock = OUT / ".evaluation.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    started = time.monotonic()
    try:
        save_json(OUT / "EVALUATION_FREEZE.json", {"protocol_sha256": digest(PROTOCOL), "evaluator_sha256": digest(Path(__file__)),
            "selection_sha256": digest(COHORT), "cache_sha256": digest(cache_path), "patients": len(selected),
            "models": 6, "conditions": ["clean", "RA_LL_reversal"], "outcome_blind": True, "started_utc": timestamp()})
        import torch
        cache = np.load(cache_path, mmap_mode="r")
        if cache.shape != (10000, 12, 5000):
            raise RuntimeError("confirmation source cache shape mismatch")
        pred = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(2, 3, 2, 10000, 3))
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                model, mean, std, thresholds = v2.load_frozen_model(arm, seed)
                if not np.array_equal(thresholds, np.asarray(protocol["thresholds"][arm][str(seed)], dtype=np.float32)):
                    raise RuntimeError("frozen threshold changed")
                for batch_no, start in enumerate(range(0, 10000, 128), 1):
                    stop = min(10000, start + 128)
                    for condition_index, array in enumerate(model_inputs(np.asarray(cache[start:stop]), mean, std)):
                        scores = v1.predict_array(model, array)
                        if scores.shape != (stop - start, 3) or not np.isfinite(scores).all():
                            raise RuntimeError("prediction shape or finiteness failed")
                        pred[arm_index, seed_index, condition_index, start:stop] = scores
                    if batch_no % 10 == 0 or stop == 10000:
                        pred.flush()
                        log("INFERENCE_PROGRESS", arm=arm, seed=seed, completed=stop, total=10000)
                del model
                torch.cuda.empty_cache()
        pred.flush()
        del pred
        os.replace(temp, final)
        save_json(OUT / "INFERENCE_COMPLETE.json", {"decision": "PASS", "patients": len(selected),
            "models": 6, "conditions": 2, "prediction_sha256": digest(final),
            "elapsed_seconds": round(time.monotonic() - started, 3), "completed_utc": timestamp()})
        log("INFERENCE_COMPLETE", patients=len(selected))
    finally:
        lock.unlink(missing_ok=True)


def labels_after_freeze(selected: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    diagnoses = pd.read_csv(DIAGNOSES, usecols=["FileName", "codes_physician"], dtype=str, keep_default_na=False)
    frame = selected[["FileName"]].merge(diagnoses, on="FileName", how="left", validate="one_to_one", sort=False)
    if len(frame) != len(selected) or frame.codes_physician.isna().any() or frame.codes_physician.eq("").any():
        raise RuntimeError("selected physician-label linkage failed")
    result = {}
    for mapping_name, mapping in (("core", v1.CORE), ("strict", v1.STRICT)):
        vectors = np.asarray([v1.label_vector(value, mapping) for value in frame.codes_physician], dtype=np.uint8)
        for index, label in enumerate(LABELS):
            result[(mapping_name, label)] = vectors[:, index]
    return result


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    positive = y == 1
    predicted = p >= threshold
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
        "sensitivity": float(predicted[positive].mean()), "specificity": float((~predicted[~positive]).mean()),
        "predicted_positive_fraction": float(predicted.mean()), "mean_score": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)), "ece15": v2.ece15(y, p)}


def analyze() -> None:
    protocol, selected = frozen()
    complete = json.loads((OUT / "INFERENCE_COMPLETE.json").read_text(encoding="utf-8"))
    path = PRIVATE / "predictions_float32.npy"
    if complete["decision"] != "PASS" or digest(path) != complete["prediction_sha256"]:
        raise RuntimeError("prediction artifact integrity failed")
    if (OUT / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("analysis already complete")
    predictions = np.load(path, mmap_mode="r")
    if predictions.shape != (2, 3, 2, 10000, 3) or not np.isfinite(predictions).all():
        raise RuntimeError("prediction array invalid")
    labels = labels_after_freeze(selected)
    thresholds = { (arm, seed): np.asarray(protocol["thresholds"][arm][str(seed)], dtype=np.float32)
        for arm in ARMS for seed in SEEDS }
    rows = []
    for mapping, label in VARIANTS:
        y = labels[(mapping, label)]
        channel = LABELS.index(label)
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                threshold = float(thresholds[(arm, seed)][channel])
                for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                    p = np.asarray(predictions[arm_index, seed_index, condition_index, :, channel])
                    rows.append({"mapping": mapping, "label": label, "arm": arm, "seed": seed, "condition": condition,
                        "patients": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
                        "frozen_threshold": threshold, **metric(y, p, threshold)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "AGGREGATE_RESULTS_BY_SEED.csv", index=False)
    mean_cols = ["auprc", "auroc", "sensitivity", "specificity", "predicted_positive_fraction", "mean_score", "brier", "ece15"]
    result.groupby(["mapping", "label", "arm", "condition"], sort=False)[mean_cols].mean().reset_index().to_csv(OUT / "THREE_SEED_MEAN_METRICS.csv", index=False)
    save_json(OUT / "AGGREGATE_LABEL_COUNTS.json", {"patients": len(selected), "outcomes": {"_".join(key): int(value.sum()) for key, value in labels.items() if key in VARIANTS}})
    log("AGGREGATE_METRICS_COMPLETE", patients=len(selected), rows=len(rows))

    contrasts = []
    bootstrap_values = {(mapping, label, name, contrast): [] for mapping, label in VARIANTS
        for name in ("auprc", "sensitivity", "specificity") for contrast in
        ("physics_aware_minus_baseline_ra_ll", "physics_aware_minus_baseline_clean", "baseline_clean_minus_ra_ll", "physics_aware_clean_minus_ra_ll")}
    for mapping, label in VARIANTS:
        part = result[(result.mapping == mapping) & (result.label == label)]
        for name in ("auprc", "sensitivity", "specificity"):
            values = {(arm, condition): part[(part.arm == arm) & (part.condition == condition)].sort_values("seed")[name].to_numpy()
                for arm in ARMS for condition in ("clean", "RA_LL_reversal")}
            for contrast, left, right in (
                ("physics_aware_minus_baseline_ra_ll", ("physics_aware", "RA_LL_reversal"), ("baseline", "RA_LL_reversal")),
                ("physics_aware_minus_baseline_clean", ("physics_aware", "clean"), ("baseline", "clean")),
                ("baseline_clean_minus_ra_ll", ("baseline", "clean"), ("baseline", "RA_LL_reversal")),
                ("physics_aware_clean_minus_ra_ll", ("physics_aware", "clean"), ("physics_aware", "RA_LL_reversal"))):
                contrasts.append({"mapping": mapping, "label": label, "metric": name, "contrast": contrast,
                    "estimate": float(np.mean(values[left] - values[right]))})
    random = np.random.default_rng(BOOT_SEED)
    for replicate in range(BOOT):
        index = random.integers(0, 10000, 10000)
        for mapping, label in VARIANTS:
            y = labels[(mapping, label)][index]
            channel = LABELS.index(label)
            positive = y == 1
            by = {}
            for arm_index, arm in enumerate(ARMS):
                for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
                    series = {"auprc": [], "sensitivity": [], "specificity": []}
                    for seed_index, seed in enumerate(SEEDS):
                        p = np.asarray(predictions[arm_index, seed_index, condition_index, index, channel])
                        pred = p >= thresholds[(arm, seed)][channel]
                        series["auprc"].append(float(average_precision_score(y, p)))
                        series["sensitivity"].append(float(pred[positive].mean()))
                        series["specificity"].append(float((~pred[~positive]).mean()))
                    for name, values in series.items():
                        by[(arm, condition, name)] = np.asarray(values)
            for name in ("auprc", "sensitivity", "specificity"):
                for contrast, left, right in (
                    ("physics_aware_minus_baseline_ra_ll", ("physics_aware", "RA_LL_reversal"), ("baseline", "RA_LL_reversal")),
                    ("physics_aware_minus_baseline_clean", ("physics_aware", "clean"), ("baseline", "clean")),
                    ("baseline_clean_minus_ra_ll", ("baseline", "clean"), ("baseline", "RA_LL_reversal")),
                    ("physics_aware_clean_minus_ra_ll", ("physics_aware", "clean"), ("physics_aware", "RA_LL_reversal"))):
                    bootstrap_values[(mapping, label, name, contrast)].append(float(np.mean(by[(*left, name)] - by[(*right, name)])))
        if (replicate + 1) % 25 == 0:
            log("PAIRED_BOOTSTRAP", completed=replicate + 1, total=BOOT)
    intervals = []
    for (mapping, label, name, contrast), values in bootstrap_values.items():
        array = np.asarray(values)
        intervals.append({"mapping": mapping, "label": label, "metric": name, "contrast": contrast,
            "bootstrap_mean": float(array.mean()), "low95": float(np.quantile(array, .025)),
            "high95": float(np.quantile(array, .975)), "replicates": BOOT, "paired_patient": True})
    interval_frame = pd.DataFrame(intervals)
    interval_frame.to_csv(OUT / "PAIRED_BOOTSTRAP_INTERVALS.csv", index=False)
    pd.DataFrame(contrasts).merge(interval_frame.drop(columns=["replicates", "paired_patient"]), on=["mapping", "label", "metric", "contrast"]).to_csv(OUT / "AGGREGATE_CONTRASTS.csv", index=False)
    save_json(OUT / "ANALYSIS_COMPLETE.json", {"decision": "PASS", "patients": len(selected), "bootstrap_replicates": BOOT,
        "paired_unit": "patient", "completed_utc": timestamp()})
    log("ANALYSIS_COMPLETE", patients=len(selected), bootstrap_replicates=BOOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "size-check", "download", "build-cache", "smoke", "evaluate", "analyze"))
    command = parser.parse_args().command
    {"select": select, "size-check": size_check, "download": download,
     "build-cache": build_cache, "smoke": smoke, "evaluate": evaluate, "analyze": analyze}[command]()


if __name__ == "__main__":
    main()
