"""Additive, transfer-only amendment for frozen HEEDB confirmation v1.

The original cohort/protocol/evaluator are immutable. This module only validates
restricted paths, resumes missing objects, and certifies the 20,000-object cache.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import heedb_external_v1 as v1
from src import heedb_independent_confirmation_v1 as original

AMENDMENT = original.ROOT / "configs" / "FROZEN_HEEDB_CONFIRMATION_TRANSFER_AMENDMENT_v1a.json"
SIDE_CAR = original.OUT / "FROZEN_TRANSFER_AMENDMENT_SHA256.txt"


def validated_path(relative: str) -> Path:
    """Contain a frozen POSIX S3 relative key without filesystem-dependent resolve.

    Existing symlink/junction components are rejected. The caller repeats this
    check after parent creation and just before opening the output file.
    """
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\x00" in relative:
        raise ValueError("unsafe waveform relative path syntax")
    if relative.startswith("/") or relative.endswith("/"):
        raise ValueError("absolute or directory waveform path")
    parts = relative.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(not part or part in (".", "..") or part.endswith((" ", ".")) or part.upper().split(".")[0] in reserved for part in parts):
        raise ValueError("unsafe waveform path component")
    if Path(parts[-1]).suffix not in (".hea", ".dat"):
        raise ValueError("unexpected waveform suffix")
    root = original.WAVES.absolute()
    target = root.joinpath(*parts)
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise ValueError("waveform path escapes restricted root")
    current = root
    for component in parts[:-1]:
        if current.exists() and (current.is_symlink() or os.path.isjunction(current)):
            raise ValueError("symlink or junction in restricted waveform path")
        current = current / component
    if current.exists() and (current.is_symlink() or os.path.isjunction(current)):
        raise ValueError("symlink or junction in restricted waveform path")
    if target.exists() and (target.is_symlink() or os.path.isjunction(target)):
        raise ValueError("symlink or junction at waveform target")
    return target


def frozen() -> tuple[dict, pd.DataFrame, dict]:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if original.digest(AMENDMENT) != SIDE_CAR.read_text(encoding="utf-8").strip():
        raise RuntimeError("transfer amendment protocol hash mismatch")
    if original.digest(Path(__file__)) != amendment["source_sha256"]:
        raise RuntimeError("transfer amendment source hash mismatch")
    for relative, expected in amendment["bound_sha256"].items():
        if original.digest(original.ROOT / relative) != expected:
            raise RuntimeError("bound frozen artifact changed")
    _scientific, selected = original.frozen()
    preflight = json.loads((original.OUT / "TRANSFER_PREFLIGHT.json").read_text(encoding="utf-8"))
    if not preflight["safe_to_transfer"] or preflight["exact_new_download_bytes"] > original.CAP or preflight["estimated_aws_service_charge_usd_upper_without_discount"] > .25:
        raise RuntimeError("frozen size/charge gate failed")
    if original.digest(original.OBJECTS) != preflight["restricted_object_manifest_sha256"]:
        raise RuntimeError("object manifest hash changed")
    objects = pd.read_csv(original.OBJECTS, dtype={"remote_bytes": np.int64})
    if len(objects) != 20000 or len(selected) != 10000 or int(objects.remote_bytes.sum()) != preflight["exact_remote_bytes"]:
        raise RuntimeError("frozen object count or exact bytes changed")
    return amendment, objects, preflight


def verify_local(objects: pd.DataFrame) -> tuple[list[dict], int, int]:
    pending = []
    valid = 0
    valid_bytes = 0
    for row in objects.to_dict("records"):
        path = validated_path(str(row["local_relative_path"]))
        expected = int(row["remote_bytes"])
        if path.is_file() and path.stat().st_size == expected:
            valid += 1
            valid_bytes += expected
        else:
            pending.append(row)
    return pending, valid, valid_bytes


def resume_one(s3, row: dict) -> int:
    destination = validated_path(str(row["local_relative_path"]))
    expected = int(row["remote_bytes"])
    head = s3.head_object(Bucket=v1.ALIAS, Key=str(row["remote_key"]))
    if int(head["ContentLength"]) != expected or head.get("StorageClass", "STANDARD") != "STANDARD" or head.get("RequestCharged"):
        raise RuntimeError("remote size, storage, or charge changed; no transfer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = validated_path(str(row["local_relative_path"]))
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists() and (partial.is_symlink() or os.path.isjunction(partial)):
        raise RuntimeError("unsafe partial-file link")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected:
        raise RuntimeError("partial file exceeds remote size")
    if start == expected:
        os.replace(partial, destination)
        return 0
    arguments = {"Bucket": v1.ALIAS, "Key": str(row["remote_key"])}
    if start:
        arguments["Range"] = f"bytes={start}-"
    response = s3.get_object(**arguments)
    if response.get("RequestCharged"):
        raise RuntimeError("unexpected requester charge")
    received = 0
    with partial.open("ab" if start else "wb") as handle:
        while True:
            block = response["Body"].read(1 << 20)
            if not block:
                break
            handle.write(block)
            received += len(block)
    if partial.stat().st_size != expected:
        raise RuntimeError("resumed waveform size mismatch")
    os.replace(partial, destination)
    return received


def resume() -> None:
    amendment, objects, preflight = frozen()
    if (original.OUT / "DOWNLOAD_COMPLETE.json").exists() or (original.OUT / "EVALUATION_FREEZE.json").exists():
        raise RuntimeError("download/evaluation already complete or started")
    if (original.OUT / ".download.lock").exists():
        raise RuntimeError("original downloader lock exists")
    lock = original.OUT / ".download-amendment.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    start_time = time.monotonic()
    try:
        pending, valid, valid_bytes = verify_local(objects)
        pending_bytes = sum(int(row["remote_bytes"]) for row in pending)
        original.log("TRANSFER_AMENDMENT_PREFLIGHT", retained_verified=valid, retained_bytes=valid_bytes,
                     pending_objects=len(pending), pending_bytes=pending_bytes)
        if valid_bytes + pending_bytes != preflight["exact_remote_bytes"] or pending_bytes > original.CAP - valid_bytes:
            raise RuntimeError("retained+pending bytes exceed frozen transfer total or cap")
        # This amendment is deliberately scoped to the one-object stopped run.
        if valid != 19999 or len(pending) != 1 or pending_bytes != 120000:
            raise RuntimeError("retained-object state differs from stopped 19,999 + one 120,000-byte object")
        s3 = v1.require_identity()  # exact non-root IAM user and account gate
        received = 0
        for completed, row in enumerate(pending, 1):
            received += resume_one(s3, row)
            original.log("TRANSFER_AMENDMENT_PROGRESS", completed=completed, total=len(pending), received_bytes=received)
        remaining, verified, verified_bytes = verify_local(objects)
        if remaining or verified != 20000 or verified_bytes != preflight["exact_remote_bytes"]:
            raise RuntimeError("final 20,000-object exact-size verification failed")
        original.save_json(original.OUT / "DOWNLOAD_COMPLETE.json", {
            "decision": "PASS", "objects": verified, "verified_objects": verified,
            "exact_remote_bytes": verified_bytes, "new_download_bytes_total": verified_bytes,
            "resumed_bytes_this_amendment": received, "cap_bytes": original.CAP,
            "estimated_aws_service_charge_usd_upper_without_discount": preflight["estimated_aws_service_charge_usd_upper_without_discount"],
            "restricted_object_manifest_sha256": original.digest(original.OBJECTS),
            "scientific_protocol_sha256": amendment["scientific_protocol_sha256"],
            "transfer_amendment_sha256": original.digest(AMENDMENT),
            "elapsed_seconds": round(time.monotonic() - start_time, 3), "completed_utc": original.timestamp(),
        })
        original.log("DOWNLOAD_COMPLETE", verified_objects=verified, exact_bytes=verified_bytes, resumed_bytes=received)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    resume()
