"""Frozen HEEDB I0006 external transportability evaluation.

Patient- and recording-level artifacts are written only below the git-ignored
``data/heedb_restricted`` tree.  Public outputs contain aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_external_v1"
DATA = ROOT / "data" / "heedb_restricted" / "I0006"
PRIVATE = DATA / "external_v1"
WAVES = PRIVATE / "waveforms"
PROTO = ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json"
ALIAS = "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
PROFILE = "bdsp-reader"
EXPECTED_ACCOUNT = "552883519608"
EXPECTED_ARN = "arn:aws:iam::552883519608:user/bdsp-heedb-reader"
CAP_BYTES = 2 * 1024**3
SEEDS = (104729, 130363, 155921)
ARMS = ("baseline", "physics_aware")
LABELS = ("TWA", "SB", "STach")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SELECT_SEED = "heedb_external_v1|2026-09-24|label-blind|one-ecg-per-patient"

CORE = {
    "TWA": frozenset((1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181)),
    "SB": frozenset((21, 24)),
    "STach": frozenset((23,)),
}
STRICT = {"TWA": frozenset((1140,)), "SB": frozenset((21,)), "STach": frozenset((23,))}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def progress(stage: str, **fields: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"updated_utc": now(), "stage": stage, **fields}
    with (OUT / "PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def protocol() -> dict:
    if not PROTO.is_file():
        raise RuntimeError("Frozen protocol is missing")
    value = json.loads(PROTO.read_text(encoding="utf-8"))
    expected = (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip().lower()
    if sha256(PROTO) != expected:
        raise RuntimeError("Frozen protocol hash mismatch")
    if value.get("protocol_sha256") not in (None, expected):
        raise RuntimeError("Embedded protocol hash mismatch")
    if sha256(Path(__file__)) != value["source_hashes"]["evaluation_code"]:
        raise RuntimeError("Frozen evaluation code hash mismatch")
    return value


def mask_panel() -> list[tuple[int, ...]]:
    value = protocol()["mask_panel"]["masks_zero_based"]
    return [tuple(int(index) for index in mask) for mask in value]


def condition_names() -> list[str]:
    return ["clean", "RA_LL_reversal"] + ["mask5_" + "_".join(map(str, mask)) for mask in mask_panel()]


def parse_codes(value: object) -> set[int]:
    import re

    return {int(token) for token in re.findall(r"\d+", str(value))}


def label_vector(value: object, mapping: dict[str, frozenset[int]]) -> tuple[int, int, int]:
    codes = parse_codes(value)
    return tuple(int(bool(codes & mapping[label])) for label in LABELS)


def require_identity():
    # Importing awscli first exposes its bundled botocore package without
    # inspecting, printing, or copying any credential material.
    import awscli  # noqa: F401
    from botocore.config import Config
    from botocore.session import Session

    session = Session(profile=PROFILE)
    config = Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"})
    sts = session.create_client("sts", config=config)
    identity = sts.get_caller_identity()
    if identity.get("Account") != EXPECTED_ACCOUNT or identity.get("Arn") != EXPECTED_ARN:
        raise RuntimeError(
            f"Identity gate failed: expected {EXPECTED_ARN} in {EXPECTED_ACCOUNT}; "
            f"received account={identity.get('Account')} arn={identity.get('Arn')}"
        )
    if ":root" in str(identity.get("Arn", "")):
        raise RuntimeError("Identity gate failed: root credentials are forbidden")
    return session.create_client("s3", config=config)


def selection_key(kind: str, patient_id: str, file_name: str = "") -> str:
    return canonical_hash(f"{SELECT_SEED}|{kind}|{patient_id}|{file_name}")


def prepare_selection() -> None:
    protocol()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    readiness = pd.read_csv(
        DATA / "audit_manifests" / "waveform_sample_candidates.csv",
        usecols=["BDSPPatientID"],
        dtype=str,
        keep_default_na=False,
    )
    excluded_patients = set(readiness["BDSPPatientID"])
    if len(excluded_patients) != 23:
        raise RuntimeError("Expected exactly 23 readiness-sample patients")

    metadata = pd.read_csv(
        DATA / "metadata" / "metadata.csv",
        usecols=["BDSPPatientID", "FileName", "FileID"],
        dtype=str,
        keep_default_na=False,
    )
    diagnoses = pd.read_csv(
        DATA / "12SL_diagnoses" / "diagnoses_acquisition.csv",
        usecols=["FileName", "codes_physician", "codes_software"],
        dtype=str,
        keep_default_na=False,
    )
    frame = metadata.merge(diagnoses, on="FileName", how="inner", validate="one_to_one")
    exclusion_counts = {
        "joined_recordings": int(len(frame)),
        "missing_patient_id": int(frame["BDSPPatientID"].eq("").sum()),
        "missing_physician_codes_after_patient_filter": int(
            (frame["BDSPPatientID"].ne("") & frame["codes_physician"].eq("")).sum()
        ),
        "readiness_sample_patients": 23,
    }
    frame = frame.loc[
        frame["BDSPPatientID"].ne("")
        & frame["codes_physician"].ne("")
        & ~frame["BDSPPatientID"].isin(excluded_patients)
    ].copy()
    frame["record_selection_sha256"] = [
        selection_key("record", patient, record)
        for patient, record in zip(frame["BDSPPatientID"], frame["FileName"])
    ]
    frame.sort_values(
        ["BDSPPatientID", "record_selection_sha256", "FileName"],
        kind="mergesort",
        inplace=True,
    )
    one = frame.drop_duplicates("BDSPPatientID", keep="first").copy()
    one["patient_selection_sha256"] = [
        selection_key("patient", patient) for patient in one["BDSPPatientID"]
    ]
    one.sort_values(["patient_selection_sha256", "BDSPPatientID"], kind="mergesort", inplace=True)
    one.reset_index(drop=True, inplace=True)
    one.insert(0, "patient_rank", np.arange(1, len(one) + 1, dtype=np.int64))
    for name, mapping in (("core", CORE), ("strict", STRICT)):
        vectors = np.asarray([label_vector(value, mapping) for value in one["codes_physician"]], dtype=np.uint8)
        for index, label in enumerate(LABELS):
            one[f"{name}_{label}"] = vectors[:, index]
    one["remote_stem"] = "ECG/I0006/" + one["FileName"]
    one.to_csv(PRIVATE / "candidate_patient_order.csv", index=False)
    selected = one.iloc[:10_000].copy()
    selected.insert(1, "evaluation_slot", np.arange(1, len(selected) + 1, dtype=np.int64))
    selected.to_csv(PRIVATE / "selection_initial.csv", index=False)
    aggregate = {
        **exclusion_counts,
        "eligible_patient_count": int(len(one)),
        "initial_selected_patients": int(len(selected)),
        "selection_seed": SELECT_SEED,
        "candidate_manifest_sha256": sha256(PRIVATE / "candidate_patient_order.csv"),
        "initial_manifest_sha256": sha256(PRIVATE / "selection_initial.csv"),
        "created_utc": now(),
    }
    atomic_json(OUT / "SELECTION_FREEZE.json", aggregate)
    progress("SELECTION_FROZEN", eligible_patients=len(one), selected_patients=len(selected))


def selected_objects(selection_path: Path) -> pd.DataFrame:
    selected = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    rows = []
    for row in selected.itertuples(index=False):
        for suffix in (".hea", ".dat"):
            key = f"{row.remote_stem}{suffix}"
            rows.append(
                {
                    "evaluation_slot": int(row.evaluation_slot),
                    "patient_rank": int(row.patient_rank),
                    "FileName": row.FileName,
                    "remote_key": key,
                    "local_relative_path": str(Path(row.FileName + suffix)),
                    "remote_bytes": "",
                    "local_verified": 0,
                }
            )
    return pd.DataFrame(rows)


def size_check() -> None:
    protocol()
    s3 = require_identity()
    manifest_path = PRIVATE / "waveform_object_manifest.csv"
    objects = selected_objects(PRIVATE / "selection_initial.csv")

    def head(index: int, key: str) -> tuple[int, int]:
        value = s3.head_object(Bucket=ALIAS, Key=key)
        return index, int(value["ContentLength"])

    sizes = np.zeros(len(objects), dtype=np.int64)
    progress("REMOTE_SIZE_CHECK_START", objects=len(objects))
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(head, i, key) for i, key in enumerate(objects["remote_key"])]
        for complete, future in enumerate(as_completed(futures), 1):
            index, size = future.result()
            sizes[index] = size
            if complete % 500 == 0 or complete == len(futures):
                progress("REMOTE_SIZE_CHECK", completed=complete, total=len(futures))
    objects["remote_bytes"] = sizes
    total = int(sizes.sum())
    objects.to_csv(manifest_path, index=False)
    atomic_json(
        OUT / "TRANSFER_PREFLIGHT.json",
        {
            "objects": int(len(objects)),
            "selected_patients": int(len(objects) // 2),
            "exact_remote_bytes": total,
            "cap_bytes": CAP_BYTES,
            "within_cap": total <= CAP_BYTES,
            "restricted_manifest_sha256": sha256(manifest_path),
            "checked_utc": now(),
        },
    )
    progress("REMOTE_SIZE_CHECK_COMPLETE", objects=len(objects), exact_remote_bytes=total, cap_bytes=CAP_BYTES)
    if total > CAP_BYTES:
        raise RuntimeError(f"Frozen selection requires {total} bytes, exceeding cap {CAP_BYTES}")


def download_one(s3, row: dict[str, object]) -> int:
    destination = WAVES / str(row["local_relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(row["remote_bytes"])
    if destination.is_file():
        if destination.stat().st_size != expected:
            raise RuntimeError(f"Existing completed object size mismatch: {destination}")
        return 0
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected:
        raise RuntimeError(f"Partial object exceeds expected size: {partial}")
    if start == expected:
        os.replace(partial, destination)
        return 0
    args = {"Bucket": ALIAS, "Key": str(row["remote_key"])}
    if start:
        args["Range"] = f"bytes={start}-"
    response = s3.get_object(**args)
    mode = "ab" if start else "wb"
    transferred = 0
    with partial.open(mode) as handle:
        body = response["Body"]
        while True:
            block = body.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            transferred += len(block)
    if partial.stat().st_size != expected:
        raise RuntimeError(f"Downloaded object size mismatch: {partial}")
    os.replace(partial, destination)
    return transferred


def download() -> None:
    protocol()
    s3 = require_identity()
    manifest_path = PRIVATE / "waveform_object_manifest.csv"
    objects = pd.read_csv(manifest_path, dtype={"remote_bytes": np.int64})
    if int(objects["remote_bytes"].sum()) > CAP_BYTES:
        raise RuntimeError("Transfer cap gate failed")
    lock = OUT / ".download.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("Download lock exists; refusing to launch a duplicate job") from exc
    new_bytes = 0
    started = time.monotonic()
    try:
        progress("DOWNLOAD_START", objects=len(objects), exact_remote_bytes=int(objects["remote_bytes"].sum()))
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(download_one, s3, row._asdict()) for row in objects.itertuples(index=False)]
            for complete, future in enumerate(as_completed(futures), 1):
                new_bytes += int(future.result())
                if complete % 250 == 0 or complete == len(futures):
                    progress(
                        "DOWNLOAD",
                        completed=complete,
                        total=len(futures),
                        new_bytes=new_bytes,
                        elapsed_seconds=round(time.monotonic() - started, 1),
                    )
        verified = []
        for row in objects.itertuples(index=False):
            destination = WAVES / str(row.local_relative_path)
            ok = destination.is_file() and destination.stat().st_size == int(row.remote_bytes)
            if not ok:
                raise RuntimeError(f"Post-download size verification failed: {destination}")
            verified.append(1)
        objects["local_verified"] = verified
        objects.to_csv(manifest_path, index=False)
        atomic_json(
            OUT / "DOWNLOAD_COMPLETE.json",
            {
                "objects": int(len(objects)),
                "verified_objects": int(sum(verified)),
                "exact_remote_bytes": int(objects["remote_bytes"].sum()),
                "new_bytes_this_invocation": new_bytes,
                "cap_bytes": CAP_BYTES,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "restricted_manifest_sha256": sha256(manifest_path),
                "completed_utc": now(),
            },
        )
        progress("DOWNLOAD_COMPLETE", objects=len(objects), new_bytes=new_bytes)
    finally:
        lock.unlink(missing_ok=True)


def read_waveform(file_name: str) -> np.ndarray:
    import wfdb
    from scipy.signal import resample_poly

    record = wfdb.rdrecord(str(WAVES / file_name), physical=True)
    if tuple(record.sig_name) != LEADS:
        raise ValueError("noncanonical lead order")
    if tuple(record.units) != ("mV",) * 12:
        raise ValueError("unexpected physical units")
    signal = np.asarray(record.p_signal, dtype=np.float32).T
    if signal.shape[0] != 12 or not np.isfinite(signal).all():
        raise ValueError("invalid waveform values")
    if float(record.fs) == 250.0 and signal.shape[1] == 2500:
        signal = resample_poly(signal, up=2, down=1, axis=1).astype(np.float32)
    elif float(record.fs) != 500.0 or signal.shape[1] != 5000:
        raise ValueError(f"unsupported sampling geometry fs={record.fs} shape={signal.shape}")
    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("invalid post-resampling waveform")
    return signal * np.float32(1000.0)  # physical mV to Ningbo/Challenge microvolts


def append_replacement_object_rows(objects: pd.DataFrame, candidate: pd.Series, s3) -> pd.DataFrame:
    rows = []
    for suffix in (".hea", ".dat"):
        key = f"{candidate['remote_stem']}{suffix}"
        size = int(s3.head_object(Bucket=ALIAS, Key=key)["ContentLength"])
        rows.append(
            {
                "evaluation_slot": -1,
                "patient_rank": int(candidate["patient_rank"]),
                "FileName": candidate["FileName"],
                "remote_key": key,
                "local_relative_path": str(Path(candidate["FileName"] + suffix)),
                "remote_bytes": size,
                "local_verified": 0,
            }
        )
    projected = int(objects["remote_bytes"].astype(np.int64).sum()) + sum(int(row["remote_bytes"]) for row in rows)
    if projected > CAP_BYTES:
        raise RuntimeError(f"Replacement would exceed transfer cap: projected={projected} cap={CAP_BYTES}")
    for row in rows:
        download_one(s3, row)
        destination = WAVES / str(row["local_relative_path"])
        if destination.stat().st_size != int(row["remote_bytes"]):
            raise RuntimeError("Replacement object size verification failed")
        row["local_verified"] = 1
    return pd.concat((objects, pd.DataFrame(rows)), ignore_index=True)


def build_cache() -> None:
    protocol()
    candidates = pd.read_csv(PRIVATE / "candidate_patient_order.csv", dtype=str, keep_default_na=False)
    initial = pd.read_csv(PRIVATE / "selection_initial.csv", dtype=str, keep_default_na=False)
    objects_path = PRIVATE / "waveform_object_manifest.csv"
    objects = pd.read_csv(objects_path)
    if int(objects["local_verified"].sum()) != len(objects):
        raise RuntimeError("Not all waveform objects have verified local sizes")
    final_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    reserve_index = 10_000
    s3 = None
    temporary = PRIVATE / "signals_10000_float32.npy.partial"
    if temporary.exists():
        temporary.unlink()
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(initial), 12, 5000))
    for slot, row in enumerate(initial.to_dict("records"), 1):
        candidate = row
        while True:
            try:
                signal = read_waveform(str(candidate["FileName"]))
                break
            except Exception as exc:
                failures.append(
                    {
                        "evaluation_slot": slot,
                        "failed_patient_rank": int(candidate["patient_rank"]),
                        "reason_type": type(exc).__name__,
                    }
                )
                if reserve_index >= len(candidates):
                    raise RuntimeError("No hash-ranked reserve patients remain") from exc
                candidate = candidates.iloc[reserve_index].to_dict()
                reserve_index += 1
                if s3 is None:
                    s3 = require_identity()
                objects = append_replacement_object_rows(objects, pd.Series(candidate), s3)
        cache[slot - 1] = signal
        candidate = dict(candidate)
        candidate["evaluation_slot"] = slot
        candidate["replaced_initial"] = int(int(candidate["patient_rank"]) != int(row["patient_rank"]))
        final_rows.append(candidate)
        if slot % 250 == 0 or slot == len(initial):
            cache.flush()
            progress("WAVEFORM_CACHE", completed=slot, total=len(initial), replacements=len(failures))
    cache.flush()
    del cache
    final_cache = PRIVATE / "signals_10000_float32.npy"
    os.replace(temporary, final_cache)
    final = pd.DataFrame(final_rows)
    final.to_csv(PRIVATE / "selection_final.csv", index=False)
    pd.DataFrame(failures).to_csv(PRIVATE / "unreadable_replacement_audit.csv", index=False)
    objects.to_csv(objects_path, index=False)
    total_bytes = int(objects["remote_bytes"].astype(np.int64).sum())
    atomic_json(
        OUT / "WAVEFORM_CACHE_COMPLETE.json",
        {
            "patients": int(len(final)),
            "unreadable_attempts": int(len(failures)),
            "replacement_patients": int(final["replaced_initial"].astype(int).sum()),
            "total_remote_waveform_bytes": total_bytes,
            "cap_bytes": CAP_BYTES,
            "cache_bytes": final_cache.stat().st_size,
            "cache_sha256": sha256(final_cache),
            "final_selection_sha256": sha256(PRIVATE / "selection_final.csv"),
            "object_manifest_sha256": sha256(objects_path),
            "completed_utc": now(),
        },
    )
    progress("WAVEFORM_CACHE_COMPLETE", patients=len(final), replacements=len(failures))


def reversal_ra_ll(x: np.ndarray) -> np.ndarray:
    # Input x is displayed I, II, III, aVR, aVL, aVF, V1..V6 in microvolts.
    i, ii = x[:, 0], x[:, 1]
    out = x.copy()
    out[:, 0] = i - ii
    out[:, 1] = -ii
    out[:, 2] = -i
    out[:, 3] = (ii - i) / np.float32(2.0)
    out[:, 4] = i - ii / np.float32(2.0)
    out[:, 5] = i / np.float32(2.0) - ii
    return out


def model_paths(arm: str, seed: int) -> tuple[Path, Path, Path]:
    base = ROOT / ("outputs/ningbo_confirmation_v2" if arm == "baseline" else "outputs/mitigation_m2_ningbo_v1")
    unit = base / f"seed_{seed}"
    return unit / "checkpoint.pt", unit / "preprocessing.npz", unit / "validation.npz"


def load_model(arm: str, seed: int):
    import torch
    from src.chapman_phase1 import ECGResNetSmall

    checkpoint, preprocessing, validation = model_paths(arm, seed)
    frozen = protocol()
    expected = frozen["models"][arm][str(seed)]
    if sha256(ROOT / "src" / "chapman_phase1.py") != frozen["source_hashes"]["model_architecture_source"]:
        raise RuntimeError("Frozen ECGResNet architecture source hash mismatch")
    if sha256(checkpoint) != expected["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch for {arm} {seed}")
    if sha256(preprocessing) != expected["preprocessing_sha256"]:
        raise RuntimeError(f"Preprocessing hash mismatch for {arm} {seed}")
    if sha256(validation) != expected["validation_threshold_source_sha256"]:
        raise RuntimeError(f"Validation artifact hash mismatch for {arm} {seed}")
    model = ECGResNetSmall().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cuda")["state_dict"])
    with np.load(preprocessing, allow_pickle=False) as loaded:
        mean, std = loaded["mean"].astype(np.float32), loaded["std"].astype(np.float32)
    with np.load(validation, allow_pickle=False) as loaded:
        thresholds = loaded["thresholds"].astype(np.float32)
    if not np.allclose(thresholds, np.asarray(expected["thresholds"], dtype=np.float32), rtol=0, atol=0):
        raise RuntimeError(f"Threshold mismatch for {arm} {seed}")
    return model, mean, std, thresholds


def validated_thresholds(arm: str, seed: int) -> np.ndarray:
    checkpoint, preprocessing, validation = model_paths(arm, seed)
    expected = protocol()["models"][arm][str(seed)]
    for path, field in (
        (checkpoint, "checkpoint_sha256"),
        (preprocessing, "preprocessing_sha256"),
        (validation, "validation_threshold_source_sha256"),
    ):
        if sha256(path) != expected[field]:
            raise RuntimeError(f"Frozen artifact hash mismatch for {arm} {seed}: {path.name}")
    with np.load(validation, allow_pickle=False) as loaded:
        thresholds = loaded["thresholds"].astype(np.float32)
    if not np.allclose(thresholds, np.asarray(expected["thresholds"], dtype=np.float32), rtol=0, atol=0):
        raise RuntimeError(f"Threshold mismatch for {arm} {seed}")
    return thresholds


def predict_array(model, array: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        value = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(array)).cuda(non_blocking=True)))
    return value.float().cpu().numpy()


def smoke() -> None:
    protocol()
    import torch

    object_manifest = pd.read_csv(DATA / "audit_manifests" / "waveform_object_manifest.csv", dtype=str)
    sample = pd.read_csv(DATA / "audit_manifests" / "waveform_sample_candidates.csv", dtype=str)
    old_waves = DATA / "waveform_sample"
    signals = []
    resampled = 0
    # Temporarily point the reader at the already-downloaded readiness sample.
    global WAVES
    evaluation_waves = WAVES
    WAVES = old_waves
    try:
        for file_name in sample["FileName"]:
            before = next(
                row for row in object_manifest.itertuples(index=False)
                if row.FileName == file_name and str(row.remote_key).endswith(".dat")
            )
            resampled += int(int(before.remote_bytes) == 60_000)
            signals.append(read_waveform(file_name))
    finally:
        WAVES = evaluation_waves
    x = np.stack(signals).astype(np.float32)
    conditions = condition_names()
    checked = 0
    started = time.monotonic()
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, std, _ = load_model(arm, seed)
            normalized = (x - mean[None, :, None]) / std[None, :, None]
            arrays = [normalized, (reversal_ra_ll(x) - mean[None, :, None]) / std[None, :, None]]
            for mask in mask_panel():
                masked = normalized.copy()
                masked[:, list(mask), :] = 0.0
                arrays.append(masked)
            for array in arrays:
                predictions = predict_array(model, array.astype(np.float32, copy=False))
                if predictions.shape != (len(x), 3) or not np.isfinite(predictions).all():
                    raise RuntimeError("Engineering smoke produced invalid predictions")
                checked += 1
            del model
            torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    atomic_json(
        OUT / "ENGINEERING_SMOKE.json",
        {
            "decision": "PASS",
            "excluded_readiness_records": int(len(x)),
            "resampled_250_to_500_records": resampled,
            "models": 6,
            "conditions_per_model": len(conditions),
            "prediction_batches_checked": checked,
            "finite_predictions": True,
            "elapsed_seconds": round(elapsed, 3),
            "cuda_device": torch.cuda.get_device_name(0),
            "completed_utc": now(),
        },
    )
    progress("ENGINEERING_SMOKE_PASS", records=len(x), models=6, conditions=len(conditions), elapsed_seconds=round(elapsed, 1))


def evaluate() -> None:
    protocol()
    import torch

    smoke_gate = json.loads((OUT / "ENGINEERING_SMOKE.json").read_text(encoding="utf-8"))
    if smoke_gate.get("decision") != "PASS":
        raise RuntimeError("Engineering smoke gate has not passed")
    cache_report = json.loads((OUT / "WAVEFORM_CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    cache_path = PRIVATE / "signals_10000_float32.npy"
    if sha256(cache_path) != cache_report["cache_sha256"]:
        raise RuntimeError("Frozen waveform cache hash mismatch")
    selection_path = PRIVATE / "selection_final.csv"
    if sha256(selection_path) != cache_report["final_selection_sha256"]:
        raise RuntimeError("Final selection hash mismatch")
    lock = OUT / ".evaluation.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("Evaluation lock exists; refusing to launch a duplicate job") from exc

    x = np.load(cache_path, mmap_mode="r")
    n = len(x)
    conditions = condition_names()
    temporary = PRIVATE / "predictions_float32.npy.partial"
    if temporary.exists():
        raise RuntimeError("Partial evaluation artifact exists; inspect before a deliberate resume")
    predictions = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(ARMS), len(SEEDS), len(conditions), n, len(LABELS)),
    )
    started = time.monotonic()
    try:
        atomic_json(
            OUT / "EVALUATION_FREEZE.json",
            {
                "protocol_sha256": sha256(PROTO),
                "selection_sha256": sha256(selection_path),
                "waveform_cache_sha256": cache_report["cache_sha256"],
                "patients": n,
                "conditions": conditions,
                "arms": list(ARMS),
                "seeds": list(SEEDS),
                "outcomes_inspected_before_freeze": False,
                "started_utc": now(),
            },
        )
        batch_size = 128
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                model, mean, std, _ = load_model(arm, seed)
                batches = math.ceil(n / batch_size)
                for batch_number, start in enumerate(range(0, n, batch_size), 1):
                    end = min(n, start + batch_size)
                    raw = np.asarray(x[start:end], dtype=np.float32)
                    normalized = (raw - mean[None, :, None]) / std[None, :, None]
                    predictions[arm_index, seed_index, 0, start:end] = predict_array(model, normalized)
                    reversed_normalized = (reversal_ra_ll(raw) - mean[None, :, None]) / std[None, :, None]
                    predictions[arm_index, seed_index, 1, start:end] = predict_array(model, reversed_normalized)
                    for condition_index, mask in enumerate(mask_panel(), 2):
                        masked = normalized.copy()
                        masked[:, list(mask), :] = 0.0
                        predictions[arm_index, seed_index, condition_index, start:end] = predict_array(model, masked)
                    if batch_number % 10 == 0 or batch_number == batches:
                        predictions.flush()
                        progress(
                            "FROZEN_EVALUATION",
                            arm=arm,
                            seed=seed,
                            completed_batches=batch_number,
                            total_batches=batches,
                            elapsed_seconds=round(time.monotonic() - started, 1),
                        )
                del model
                torch.cuda.empty_cache()
        predictions.flush()
        del predictions
        final_predictions = PRIVATE / "predictions_float32.npy"
        os.replace(temporary, final_predictions)
        prediction_hash = sha256(final_predictions)
        atomic_json(
            OUT / "INFERENCE_COMPLETE.json",
            {
                "decision": "PASS",
                "patients": n,
                "models": 6,
                "conditions_per_model": len(conditions),
                "prediction_artifact_sha256": prediction_hash,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_utc": now(),
            },
        )
        progress("INFERENCE_COMPLETE", patients=n, elapsed_seconds=round(time.monotonic() - started, 1))
    finally:
        lock.unlink(missing_ok=True)


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def sensitivity(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    positive = y == 1
    return float((p[positive] >= threshold).mean()) if positive.any() else float("nan")


def percentile(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def analyze() -> None:
    protocol()
    inference = json.loads((OUT / "INFERENCE_COMPLETE.json").read_text(encoding="utf-8"))
    predictions_path = PRIVATE / "predictions_float32.npy"
    if sha256(predictions_path) != inference["prediction_artifact_sha256"]:
        raise RuntimeError("Prediction artifact hash mismatch")
    frame = pd.read_csv(PRIVATE / "selection_final.csv", dtype=str, keep_default_na=False)
    predictions = np.load(predictions_path, mmap_mode="r")
    conditions = condition_names()
    variants = [("core", label) for label in LABELS] + [("strict", "TWA"), ("strict", "SB")]
    thresholds: dict[tuple[str, int], np.ndarray] = {}
    for arm in ARMS:
        for seed in SEEDS:
            thresholds[(arm, seed)] = validated_thresholds(arm, seed)

    rows: list[dict[str, object]] = []
    for mapping, label in variants:
        label_index = LABELS.index(label)
        y = frame[f"{mapping}_{label}"].astype(np.uint8).to_numpy()
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                threshold = float(thresholds[(arm, seed)][label_index])
                for condition_index, condition in enumerate(conditions):
                    p = np.asarray(predictions[arm_index, seed_index, condition_index, :, label_index])
                    rows.append(
                        {
                            "mapping": mapping,
                            "label": label,
                            "arm": arm,
                            "seed": seed,
                            "condition": condition,
                            "positives": int(y.sum()),
                            "negatives": int((1 - y).sum()),
                            "auprc": average_precision(y, p),
                            "sensitivity_at_frozen_threshold": sensitivity(y, p, threshold),
                            "frozen_threshold": threshold,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "AGGREGATE_RESULTS_BY_SEED.csv", index=False)

    label_counts = {
        f"{mapping}_{label}": {"positive": int(frame[f"{mapping}_{label}"].astype(int).sum()), "total": len(frame)}
        for mapping, label in variants
    }
    pattern_counts = (
        frame[["core_TWA", "core_SB", "core_STach"]].astype(str).agg("".join, axis=1).value_counts().sort_index()
    )
    atomic_json(
        OUT / "AGGREGATE_LABEL_COUNTS.json",
        {"patients": len(frame), "outcomes": label_counts, "core_pattern_counts": {k: int(v) for k, v in pattern_counts.items()}},
    )

    contrasts: list[dict[str, object]] = []
    for mapping, label in variants:
        subset = result[(result["mapping"] == mapping) & (result["label"] == label)]
        for metric in ("auprc", "sensitivity_at_frozen_threshold"):
            values: dict[str, list[float]] = {}
            for arm in ARMS:
                clean = subset[(subset["arm"] == arm) & (subset["condition"] == "clean")].sort_values("seed")[metric].to_numpy()
                reversal = subset[(subset["arm"] == arm) & (subset["condition"] == "RA_LL_reversal")].sort_values("seed")[metric].to_numpy()
                values[f"{arm}_clean"] = clean.tolist()
                values[f"{arm}_ra_ll"] = reversal.tolist()
                contrasts.append({"mapping": mapping, "label": label, "metric": metric, "contrast": f"{arm}_clean_minus_ra_ll", "estimate": float(np.mean(clean - reversal))})
            contrasts.append({"mapping": mapping, "label": label, "metric": metric, "contrast": "physics_aware_minus_baseline_ra_ll", "estimate": float(np.mean(np.asarray(values["physics_aware_ra_ll"]) - np.asarray(values["baseline_ra_ll"])))})
            contrasts.append({"mapping": mapping, "label": label, "metric": metric, "contrast": "physics_aware_minus_baseline_clean", "estimate": float(np.mean(np.asarray(values["physics_aware_clean"]) - np.asarray(values["baseline_clean"])))})
        for arm in ARMS:
            masks = subset[(subset["arm"] == arm) & subset["condition"].str.startswith("mask5_")]
            contrasts.append({"mapping": mapping, "label": label, "metric": "auprc", "contrast": f"{arm}_24_mask_subset_mean", "estimate": float(masks.groupby("seed")["auprc"].mean().mean())})
    contrast_frame = pd.DataFrame(contrasts)

    # Synchronized patient resamples.  All clean/RA-LL contrasts for core and
    # strict outcomes receive intervals.  The secondary mask-panel interval is
    # limited, as frozen, to core TWA to bound exact AUPRC computation.
    rng = np.random.default_rng(20260924)
    boot_values: dict[tuple[str, str, str, str], list[float]] = {}
    for mapping, label in variants:
        for metric in ("auprc", "sensitivity_at_frozen_threshold"):
            for contrast in (
                "baseline_clean_minus_ra_ll",
                "physics_aware_clean_minus_ra_ll",
                "physics_aware_minus_baseline_ra_ll",
                "physics_aware_minus_baseline_clean",
            ):
                boot_values[(mapping, label, metric, contrast)] = []
    boot_values[("core", "TWA", "auprc", "physics_aware_minus_baseline_24_mask_subset_mean")] = []

    for replicate in range(1000):
        index = rng.integers(0, len(frame), len(frame))
        for mapping, label in variants:
            label_index = LABELS.index(label)
            y = frame[f"{mapping}_{label}"].astype(np.uint8).to_numpy()[index]
            by_arm: dict[str, dict[str, list[float]]] = {}
            for arm_index, arm in enumerate(ARMS):
                by_arm[arm] = {"clean_ap": [], "ra_ap": [], "clean_sens": [], "ra_sens": []}
                for seed_index, seed in enumerate(SEEDS):
                    threshold = float(thresholds[(arm, seed)][label_index])
                    clean = np.asarray(predictions[arm_index, seed_index, 0, index, label_index])
                    reversal = np.asarray(predictions[arm_index, seed_index, 1, index, label_index])
                    by_arm[arm]["clean_ap"].append(average_precision(y, clean))
                    by_arm[arm]["ra_ap"].append(average_precision(y, reversal))
                    by_arm[arm]["clean_sens"].append(sensitivity(y, clean, threshold))
                    by_arm[arm]["ra_sens"].append(sensitivity(y, reversal, threshold))
            for metric, suffix in (("auprc", "ap"), ("sensitivity_at_frozen_threshold", "sens")):
                for arm in ARMS:
                    value = float(np.mean(np.asarray(by_arm[arm][f"clean_{suffix}"]) - np.asarray(by_arm[arm][f"ra_{suffix}"])))
                    boot_values[(mapping, label, metric, f"{arm}_clean_minus_ra_ll")].append(value)
                for condition in ("ra", "clean"):
                    value = float(np.mean(np.asarray(by_arm["physics_aware"][f"{condition}_{suffix}"]) - np.asarray(by_arm["baseline"][f"{condition}_{suffix}"])))
                    boot_values[(mapping, label, metric, f"physics_aware_minus_baseline_{'ra_ll' if condition == 'ra' else 'clean'}")].append(value)
        y = frame["core_TWA"].astype(np.uint8).to_numpy()[index]
        panel_by_arm = {}
        for arm_index, arm in enumerate(ARMS):
            seed_means = []
            for seed_index, _seed in enumerate(SEEDS):
                aps = [average_precision(y, np.asarray(predictions[arm_index, seed_index, condition_index, index, 0])) for condition_index in range(2, len(conditions))]
                seed_means.append(float(np.mean(aps)))
            panel_by_arm[arm] = float(np.mean(seed_means))
        boot_values[("core", "TWA", "auprc", "physics_aware_minus_baseline_24_mask_subset_mean")].append(panel_by_arm["physics_aware"] - panel_by_arm["baseline"])
        if (replicate + 1) % 25 == 0:
            progress("PAIRED_BOOTSTRAP", completed=replicate + 1, total=1000)

    interval_rows = []
    for (mapping, label, metric, contrast), values in boot_values.items():
        mean, low, high = percentile(values)
        interval_rows.append({"mapping": mapping, "label": label, "metric": metric, "contrast": contrast, "bootstrap_mean": mean, "low95": low, "high95": high, "replicates": 1000, "unit": "patient", "paired": True})
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(OUT / "PAIRED_BOOTSTRAP_INTERVALS.csv", index=False)
    contrast_frame = contrast_frame.merge(intervals.drop(columns=["replicates", "unit", "paired"]), on=["mapping", "label", "metric", "contrast"], how="left")
    contrast_frame.to_csv(OUT / "AGGREGATE_CONTRASTS.csv", index=False)
    atomic_json(
        OUT / "ANALYSIS_COMPLETE.json",
        {
            "decision": "PASS",
            "patients": len(frame),
            "bootstrap_replicates": 1000,
            "bootstrap_unit": "patient",
            "paired_across_arms_seeds_conditions": True,
            "patient_or_record_identifiers_in_outputs": False,
            "completed_utc": now(),
        },
    )
    progress("ANALYSIS_COMPLETE", patients=len(frame), bootstrap_replicates=1000)


def report() -> None:
    protocol_value = protocol()
    if json.loads((OUT / "ANALYSIS_COMPLETE.json").read_text(encoding="utf-8")).get("decision") != "PASS":
        raise RuntimeError("Analysis completion gate has not passed")
    counts = json.loads((OUT / "AGGREGATE_LABEL_COUNTS.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "SELECTION_FREEZE.json").read_text(encoding="utf-8"))
    transfer = json.loads((OUT / "WAVEFORM_CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    inference = json.loads((OUT / "INFERENCE_COMPLETE.json").read_text(encoding="utf-8"))
    smoke_result = json.loads((OUT / "ENGINEERING_SMOKE.json").read_text(encoding="utf-8"))
    contrasts = pd.read_csv(OUT / "AGGREGATE_CONTRASTS.csv")
    results = pd.read_csv(OUT / "AGGREGATE_RESULTS_BY_SEED.csv")

    primary_names = {
        "physics_aware_minus_baseline_ra_ll",
        "physics_aware_minus_baseline_clean",
        "baseline_clean_minus_ra_ll",
        "physics_aware_clean_minus_ra_ll",
    }
    primary = contrasts[
        (contrasts["mapping"] == "core")
        & (contrasts["label"] == "TWA")
        & contrasts["contrast"].isin(primary_names)
    ].copy()

    def fmt(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        return f"{float(value):.4f}"

    lines = [
        "# HEEDB I0006 external transportability test v1",
        "",
        "## Decision",
        "",
        "The frozen external evaluation completed on 10,000 patient-disjoint Emory ECGs. It is a fair, label-blind external transportability test of the frozen Ningbo baseline and physics-aware classifiers under the prespecified HEEDB physician-overread proxy labels. It is **not** an exact SNOMED-label replication, so construct mismatch remains a limitation.",
        "",
        "## Primary core-TWA comparisons",
        "",
        "Positive values for `physics-aware minus baseline` favor the physics-aware arm. Positive within-arm `clean minus RA–LL` values indicate degradation under reversal.",
        "",
        "| Metric | Contrast | Estimate | Paired patient bootstrap 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in primary.sort_values(["metric", "contrast"]).itertuples(index=False):
        interval = "—" if pd.isna(row.low95) else f"[{fmt(row.low95)}, {fmt(row.high95)}]"
        lines.append(f"| {row.metric} | {row.contrast} | {fmt(row.estimate)} | {interval} |")

    lines += ["", "## Aggregate label counts", "", "All counts use `codes_physician` only.", "", "| Mapping/outcome | Positive | Total |", "|---|---:|---:|"]
    for name, values in counts["outcomes"].items():
        lines.append(f"| {name} | {values['positive']:,} | {values['total']:,} |")

    lines += [
        "",
        "## Secondary results",
        "",
        "Core SB/STach and strict TWA/SB clean and RA–LL results are in `AGGREGATE_RESULTS_BY_SEED.csv` and `AGGREGATE_CONTRASTS.csv`. The mask comparison is the prespecified deterministic **24-mask subset**, not the 792-mask mean; each displayed lead occurs in exactly 10 masks. Only its core-TWA arm contrast receives the prespecified paired-bootstrap interval.",
        "",
        "## Cohort, exclusions, and transfers",
        "",
        f"- Joined table rows: {selection['joined_recordings']:,}.",
        f"- Excluded for missing patient ID: {selection['missing_patient_id']:,}.",
        f"- Excluded after patient linkage for empty physician codes: {selection['missing_physician_codes_after_patient_filter']:,}.",
        f"- Excluded readiness-sample patients: {selection['readiness_sample_patients']:,}.",
        f"- Eligible patients after one deterministic ECG per patient: {selection['eligible_patient_count']:,}.",
        f"- Evaluated patients/recordings: {counts['patients']:,} / {counts['patients']:,}.",
        f"- Unreadable attempts: {transfer['unreadable_attempts']:,}; replacements: {transfer['replacement_patients']:,}.",
        f"- Exact waveform bytes downloaded/retained for this test: {transfer['total_remote_waveform_bytes']:,} ({transfer['total_remote_waveform_bytes'] / 1024**3:.3f} GiB), against the 2 GiB cap.",
        "- Patient- and recording-level manifests, waveforms, cached tensors, and predictions remain under `data/heedb_restricted/I0006/external_v1/`, which is git-ignored.",
        "",
        "## Label interpretation",
        "",
        "Primary labels are physician-overread `codes_physician`: core TWA 1140/1145/1150/1151/1160/1161/1170/1171/1172/1180/1181, SB 21/24, and sinus tachycardia 23. Strict sensitivity mappings are TWA 1140 and SB 21. Software codes and v24 labels were never merged. Mixed ST–T, suggestive/fragment, serial-change T-wave codes and probable SB code 265 were excluded from the positive definitions. These clinical constructs overlap with, but are not identical to, the Chapman/Ningbo SNOMED outcomes.",
        "",
        "## Frozen implementation and provenance",
        "",
        f"- Protocol SHA-256: `{sha256(PROTO)}`.",
        f"- Evaluation code SHA-256: `{sha256(Path(__file__))}`.",
        f"- Engineering smoke: PASS on {smoke_result['excluded_readiness_records']} excluded readiness records, all six models and {smoke_result['conditions_per_model']} conditions.",
        f"- Frozen inference elapsed: {inference['elapsed_seconds'] / 60:.1f} minutes.",
        "- Three matched seeds were aggregated as means of seed-level metrics/contrasts; prediction vectors were not averaged across seeds.",
        "- Uncertainty uses 1,000 synchronized patient-level bootstrap resamples.",
        "",
        "## Reproducibility artifacts",
        "",
        "Aggregate results and all hashes are in this output directory. Restricted manifests and predictions are intentionally excluded. The exact checkpoint, validation-threshold, preprocessing, source, and upstream protocol hashes are recorded in the frozen protocol and `RUN_PROVENANCE.json`.",
    ]
    report_path = OUT / "HEEDB_EXTERNAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_files = [
        OUT / "AGGREGATE_LABEL_COUNTS.json",
        OUT / "AGGREGATE_RESULTS_BY_SEED.csv",
        OUT / "AGGREGATE_CONTRASTS.csv",
        OUT / "PAIRED_BOOTSTRAP_INTERVALS.csv",
        OUT / "ENGINEERING_SMOKE.json",
        OUT / "INFERENCE_COMPLETE.json",
        report_path,
    ]
    provenance = {
        "protocol_sha256": sha256(PROTO),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "frozen_source_hashes": protocol_value["source_hashes"],
        "models": protocol_value["models"],
        "selection_sha256": transfer["final_selection_sha256"],
        "waveform_cache_sha256": transfer["cache_sha256"],
        "prediction_artifact_sha256": inference["prediction_artifact_sha256"],
        "aggregate_output_sha256": {path.name: sha256(path) for path in output_files},
        "downloaded_waveform_bytes": transfer["total_remote_waveform_bytes"],
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "completed_utc": now(),
    }
    atomic_json(OUT / "RUN_PROVENANCE.json", provenance)
    progress("REPORT_COMPLETE", disk_free_bytes=provenance["disk_free_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-selection", "size-check", "download", "build-cache", "smoke", "evaluate", "analyze", "report"))
    command = parser.parse_args().command
    {
        "prepare-selection": prepare_selection,
        "size-check": size_check,
        "download": download,
        "build-cache": build_cache,
        "smoke": smoke,
        "evaluate": evaluate,
        "analyze": analyze,
        "report": report,
    }[command]()


if __name__ == "__main__":
    main()
