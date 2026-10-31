"""Official, resumable acquisition of Challenge 2021 v1.0.3 Ningbo files only.

This entry point intentionally has no header parsing, label inspection, training,
evaluation, or bootstrap functionality.  It is limited to durable source-file
acquisition and release-manifest verification.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import ctypes
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import socketserver
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v1.json"
OUT = ROOT / "outputs" / "ningbo_confirmation_v1"
DATA = ROOT / "data" / "challenge2021_v103" / "ningbo"
OFFICIAL_BASE = "https://physionet.org/files/challenge-2021/1.0.3"
OFFICIAL_MANIFEST = f"{OFFICIAL_BASE}/SHA256SUMS.txt"
EXPECTED_PROTOCOL_SHA256 = "B6EEB79C0CD214FE7F41FDC526AEC64DF2E9D982F59A45E7F21D0108B5A3698E"
EXPECTED_RECORDS = 34_905
EXPECTED_FILES = 69_810
EXPECTED_OFFICIAL_MANIFEST_SHA256 = "f29c2609343c38734f3a3d62be4a9052ca56a9e560e7ad2efc2118563fca2473"
DEFAULT_WORKERS = 4
MAX_WORKERS = 12
MAX_DOWNLOAD_ATTEMPTS = 5
CHECKSUM_EXCEPTION_KEY = "training/ningbo/g3/JS13118.mat"
CHECKSUM_EXCEPTION_STALE_EXPECTED = "de32699cbd677b507177b60fa06125e55a935c8ef69daa0f8e8171b06d1b6f5a"
CHECKSUM_EXCEPTION_OBSERVED = "bbd5fe831ad2ce60dc62588a9f8eeccc2ea37564cf0bfb4d7a8d49c6a9eafdfd"
CHECKSUM_EXCEPTION_BYTES = 120_024
RECORD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AcquisitionError(RuntimeError):
    """A fail-closed data-acquisition error."""


class IntegrityError(AcquisitionError):
    """Raised when an official source cannot prove its integrity."""


class TransientDownloadError(AcquisitionError):
    """A transfer failure that may safely be retried against the immutable manifest."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_exception_structure_valid(path: Path) -> bool:
    """Validate only the documented JS13118 MATLAB v4 physical structure."""
    if path.stat().st_size != CHECKSUM_EXCEPTION_BYTES:
        return False
    with path.open("rb") as handle:
        header = handle.read(24)
        payload = handle.read()
    if len(header) != 24 or len(payload) != 120_000:
        return False
    try:
        mopt, mrows, ncols, imagf, namelen = struct.unpack("<5i", header[:20])
    except struct.error:
        return False
    return (mopt, mrows, ncols, imagf, namelen, header[20:24]) == (30, 12, 5000, 0, 4, b"val\x00")


def checksum_exception_applies(key: str, manifest_expected: str, observed: str,
                               path: Path) -> bool:
    return (key == CHECKSUM_EXCEPTION_KEY and
            manifest_expected == CHECKSUM_EXCEPTION_STALE_EXPECTED and
            observed == CHECKSUM_EXCEPTION_OBSERVED and
            checksum_exception_structure_valid(path))


def write_checksum_exception(out_dir: Path, manifest_sha256: str,
                             checksums: dict[str, str]) -> Path:
    """Record the sole source checksum amendment before any exceptional promotion."""
    if checksums.get(CHECKSUM_EXCEPTION_KEY) != CHECKSUM_EXCEPTION_STALE_EXPECTED:
        raise IntegrityError("JS13118 checksum exception requires the exact documented stale manifest hash")
    artifact = out_dir / "ACQUISITION_CHECKSUM_EXCEPTION_v1.json"
    evidence = {
        "version": "ningbo_acquisition_checksum_exception_v1",
        "created_utc": utc_now(),
        "scope": "one official-source checksum discrepancy only",
        "official_manifest_sha256": manifest_sha256,
        "path": CHECKSUM_EXCEPTION_KEY,
        "official_manifest_expected_sha256": CHECKSUM_EXCEPTION_STALE_EXPECTED,
        "repeated_observed_sha256": CHECKSUM_EXCEPTION_OBSERVED,
        "independent_official_download_attempts": 6,
        "observed_length_bytes": CHECKSUM_EXCEPTION_BYTES,
        "matlab_v4_structure": {
            "mopt": 30, "mrows": 12, "ncols": 5000, "imagf": 0,
            "namelen": 4, "name": "val", "payload_bytes": 120000,
            "payload_interpretation": "60000 int16 = 12x5000",
        },
        "acceptance_rule": {
            "exact_path_only": CHECKSUM_EXCEPTION_KEY,
            "manifest_hash_must_equal": CHECKSUM_EXCEPTION_STALE_EXPECTED,
            "download_hash_must_equal": CHECKSUM_EXCEPTION_OBSERVED,
            "exact_size_bytes": CHECKSUM_EXCEPTION_BYTES,
            "exact_matlab_v4_structure_required": True,
        },
        "scientific_definitions_unchanged": [
            "endpoints", "splits", "models", "faults", "metrics", "bootstrap", "decision gates",
        ],
    }
    atomic_json(artifact, evidence)
    atomic_text(out_dir / "ACQUISITION_CHECKSUM_EXCEPTION_v1.md", "\n".join([
        "# Ningbo acquisition checksum exception v1", "",
        "This is a single, exact source checksum amendment for `training/ningbo/g3/JS13118.mat`.",
        f"- Manifest SHA-256: `{manifest_sha256}`",
        f"- Stale manifest SHA-256: `{CHECKSUM_EXCEPTION_STALE_EXPECTED}`",
        f"- Repeated observed SHA-256 (six independent official downloads): `{CHECKSUM_EXCEPTION_OBSERVED}`",
        "- Exact length: 120,024 bytes; MATLAB v4 `mopt=30`, `mrows=12`, `ncols=5000`, `imagf=0`, `namelen=4`, `name=val`; 120,000-byte payload.",
        "- No endpoints, splits, models, faults, metrics, bootstrap settings, or decision gates change.", "",
    ]))
    return artifact


def transient_replace_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32}


def append_telemetry_warning(path: Path, error: OSError) -> None:
    warning = {"timestamp_utc": utc_now(), "target": str(path), "error": repr(error)}
    try:
        with (path.parent / "telemetry_warnings.log").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(warning, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as warning_error:
        print(f"telemetry warning write failed: {warning_error!r}; original={warning!r}", file=sys.stderr, flush=True)


def atomic_json(path: Path, value: dict[str, Any], *, telemetry: bool = False) -> bool:
    """Replace JSON atomically; only live telemetry may survive a locked target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for attempt in range(1, 7):
            try:
                os.replace(temporary, path)
                return True
            except OSError as error:
                if not transient_replace_error(error):
                    raise
                last_error = error
                if attempt < 6:
                    # Total bounded wait is below four seconds, including <=0.1 s jitter per retry.
                    time.sleep(min(1.0, 0.05 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.1))
        assert last_error is not None
        if telemetry:
            append_telemetry_warning(path, last_error)
            return False
        raise last_error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Do not use os.kill(pid, 0) on Windows: unlike POSIX, it may signal or
        # terminate the target.  Query a read-only process handle instead.
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_uint32()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        # Windows reports ERROR_INVALID_PARAMETER (WinError 87) when the
        # recorded PID no longer exists. Treat that lock owner as stale.
        if getattr(error, "winerror", None) == 87:
            return False
        raise
    return True


class ProcessLock:
    """Exclusive PID lock; only a demonstrably dead owner is treated as stale."""

    def __init__(self, path: Path, details: dict[str, Any]) -> None:
        self.path = path
        self.details = details
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.details, sort_keys=True, indent=2) + "\n"
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                prior = json.loads(self.path.read_text(encoding="utf-8"))
                prior_pid = int(prior["pid"])
            except Exception as error:
                raise AcquisitionError(f"acquisition lock is malformed: {self.path}") from error
            if pid_is_alive(prior_pid):
                raise AcquisitionError(
                    f"duplicate acquisition rejected: live PID {prior_pid} owns {self.path}"
                )
            # No process can acquire an O_EXCL lock until this stale path is gone.
            self.path.unlink()
            return self.acquire()
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self.owned = True

    def release(self) -> None:
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False


@dataclass(frozen=True)
class AcquisitionConfig:
    base_url: str
    manifest_url: str
    data_dir: Path
    out_dir: Path
    groups: tuple[int, ...]
    expected_records: int
    workers: int = DEFAULT_WORKERS
    protocol_path: Path | None = PROTOCOL
    official: bool = True

    @property
    def expected_files(self) -> int:
        return self.expected_records * 2


def get_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ningbo-acquisition-v1/1.0"})
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except (URLError, TimeoutError, OSError) as error:
        raise AcquisitionError(f"source request failed for {url}: {error}") from error


def parse_manifest(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise IntegrityError("official SHA256 manifest is not UTF-8") from error
    parsed: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*([0-9A-Fa-f]{64})\s+\*?(.+?)\s*", line)
        if match is None:
            raise IntegrityError(f"malformed SHA256 manifest line {number}")
        checksum, name = match.group(1).lower(), match.group(2).replace("\\", "/")
        if name.startswith("./"):
            name = name[2:]
        if not name or name in parsed:
            raise IntegrityError(f"malformed or duplicate SHA256 manifest path at line {number}")
        parsed[name] = checksum
    if not parsed:
        raise IntegrityError("official SHA256 manifest parsed to zero files")
    return parsed


def validate_official_manifest(raw: bytes) -> dict[str, str]:
    """Accept only the one previously verified Challenge 2021 release manifest."""
    if hashlib.sha256(raw).hexdigest() != EXPECTED_OFFICIAL_MANIFEST_SHA256:
        raise IntegrityError("official SHA256 manifest hash does not match the frozen release")
    checksums = parse_manifest(raw)
    ningbo = {
        key: digest for key, digest in checksums.items()
        if key.startswith("training/ningbo/") and key.endswith((".hea", ".mat"))
    }
    if len(ningbo) != EXPECTED_FILES:
        raise IntegrityError(
            f"official manifest contains {len(ningbo)} Ningbo waveform/header files, expected {EXPECTED_FILES}"
        )
    return checksums


def enumerate_records(config: AcquisitionConfig) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    seen: set[str] = set()
    for group in config.groups:
        location = f"{config.base_url}/training/ningbo/g{group}/RECORDS"
        try:
            text = get_bytes(location).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise IntegrityError(f"RECORDS is not UTF-8 for g{group}") from error
        group_records: list[str] = []
        for number, raw in enumerate(text.splitlines(), start=1):
            record = raw.strip()
            if not record:
                # Blank lines make an index ambiguous, except for the standard final newline.
                if raw:
                    raise IntegrityError(f"malformed blank RECORDS line g{group}:{number}")
                continue
            if record != raw or not RECORD_NAME.fullmatch(record):
                raise IntegrityError(f"malformed RECORDS entry g{group}:{number}: {raw!r}")
            full_id = f"g{group}/{record}"
            if full_id in seen:
                raise IntegrityError(f"duplicate RECORDS entry: {full_id}")
            seen.add(full_id)
            group_records.append(record)
        if not group_records:
            raise IntegrityError(f"official RECORDS index g{group} is empty")
        records.extend((group, record) for record in group_records)
    if len(records) != config.expected_records or len(seen) != config.expected_records:
        raise IntegrityError(
            f"official RECORDS count is {len(records)}, expected {config.expected_records}"
        )
    return records


def resolve_source_files(records: list[tuple[int, str]], checksums: dict[str, str],
                         groups: tuple[int, ...]) -> list[dict[str, Any]]:
    """Resolve imperfect RECORDS indices to checksum-manifest physical paths."""
    by_name_extension: dict[tuple[str, str], list[str]] = {}
    allowed_groups = set(groups)
    for key in checksums:
        parts = key.split("/")
        if len(parts) != 4 or parts[:2] != ["training", "ningbo"]:
            continue
        if not parts[2].startswith("g") or not parts[2][1:].isdigit():
            continue
        physical_group = int(parts[2][1:])
        if physical_group not in allowed_groups or "." not in parts[3]:
            continue
        basename, extension = parts[3].rsplit(".", 1)
        if extension not in {"hea", "mat"}:
            continue
        by_name_extension.setdefault((basename, extension), []).append(key)

    resolved: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for listed_group, logical_record in records:
        header_basename = "S23074" if logical_record == "JS23074" else logical_record
        physical: dict[str, str] = {}
        for extension, basename in (("hea", header_basename), ("mat", logical_record)):
            candidates = sorted(by_name_extension.get((basename, extension), []))
            if len(candidates) != 1:
                raise IntegrityError(
                    f"logical record g{listed_group}/{logical_record} resolves to "
                    f"{len(candidates)} {extension} candidates: {candidates}"
                )
            physical[extension] = candidates[0]
        if physical["hea"] in used_paths or physical["mat"] in used_paths:
            raise IntegrityError(f"physical source path reused by g{listed_group}/{logical_record}")
        used_paths.update(physical.values())
        resolved.append({
            "listed_group": listed_group,
            "logical_record": logical_record,
            "header_path": physical["hea"],
            "waveform_path": physical["mat"],
        })
    expected_record_count = len(records)
    expected_file_count = expected_record_count * 2
    if len(resolved) != expected_record_count or len(used_paths) != expected_file_count:
        raise IntegrityError(
            f"resolved population is {len(resolved)} records/{len(used_paths)} files; "
            f"expected {expected_record_count}/{expected_file_count}"
        )
    return resolved


def write_source_amendment(out_dir: Path, records: list[dict[str, Any]],
                           manifest_sha256: str) -> None:
    relocations: list[dict[str, Any]] = []
    for row in records:
        header_group = int(row["header_path"].split("/")[2][1:])
        waveform_group = int(row["waveform_path"].split("/")[2][1:])
        if header_group != waveform_group:
            raise IntegrityError(f"header/waveform group mismatch for {row['logical_record']}")
        if header_group != row["listed_group"]:
            relocations.append({
                "logical_record": row["logical_record"],
                "records_group": row["listed_group"],
                "physical_group": header_group,
            })
    if len(relocations) != 34:
        raise IntegrityError(f"expected 34 RECORDS group relocations, observed {len(relocations)}")
    alias = next((row for row in records if row["logical_record"] == "JS23074"), None)
    if alias is None or not alias["header_path"].endswith("/S23074.hea") or not alias["waveform_path"].endswith("/JS23074.mat"):
        raise IntegrityError("the documented S23074/JS23074 source alias was not resolved exactly")
    amendment = {
        "version": "ningbo_acquisition_source_amendment_v1",
        "created_utc": utc_now(),
        "scope": "source enumeration and physical filename resolution only",
        "official_manifest_sha256": manifest_sha256,
        "logical_records": len(records),
        "physical_files": len(records) * 2,
        "records_group_relocations": relocations,
        "filename_alias": {
            "logical_record": "JS23074",
            "header_path": alias["header_path"],
            "waveform_path": alias["waveform_path"],
            "reason": "official header basename differs and references JS23074.mat",
        },
        "scientific_definitions_unchanged": [
            "endpoints", "split seed and fractions", "model", "faults", "metrics",
            "bootstrap", "decision gates",
        ],
    }
    atomic_json(out_dir / "ACQUISITION_SOURCE_AMENDMENT_v1.json", amendment)
    lines = [
        "# Ningbo acquisition source amendment v1", "",
        "This amendment changes only source enumeration and physical path resolution.",
        "All scientific endpoints, splits, models, faults, metrics, bootstrap settings, and decision gates remain unchanged.", "",
        f"- Official SHA256SUMS SHA-256: `{manifest_sha256}`",
        f"- Logical records: {len(records)}", f"- Checksum-verifiable physical files: {len(records) * 2}",
        "- RECORDS boundary entries physically stored in the following group: 34.",
        "- Filename exception: logical `JS23074` uses official `g13/S23074.hea` and `g13/JS23074.mat`.", "",
        "## Group relocations", "",
    ]
    lines.extend(
        f"- `{item['logical_record']}`: RECORDS g{item['records_group']} -> physical g{item['physical_group']}"
        for item in relocations
    )
    atomic_text(out_dir / "ACQUISITION_SOURCE_AMENDMENT_v1.md", "\n".join(lines) + "\n")
    atomic_json(out_dir / "SOURCE_RECORD_MAPPING.json", {
        "version": "ningbo_logical_to_physical_mapping_v1",
        "logical_record_count": len(records),
        "physical_file_count": len(records) * 2,
        "records": records,
    })


def validate_official_source_mapping(raw: bytes, checksums: dict[str, str]) -> list[dict[str, Any]]:
    """Fail closed unless a cached mapping is a complete deterministic manifest resolution."""
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("cached SOURCE_RECORD_MAPPING.json is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or payload.get("version") != "ningbo_logical_to_physical_mapping_v1":
        raise IntegrityError("cached source mapping has an unexpected version")
    if payload.get("logical_record_count") != EXPECTED_RECORDS or payload.get("physical_file_count") != EXPECTED_FILES:
        raise IntegrityError("cached source mapping has incorrect frozen population counts")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_RECORDS:
        raise IntegrityError("cached source mapping does not contain the complete logical record population")
    listed: list[tuple[int, str]] = []
    physical: set[str] = set()
    logical: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise IntegrityError("cached source mapping contains a non-object row")
        group, record = row.get("listed_group"), row.get("logical_record")
        header, waveform = row.get("header_path"), row.get("waveform_path")
        if (not isinstance(group, int) or group not in range(1, 36) or
                not isinstance(record, str) or not RECORD_NAME.fullmatch(record) or
                not isinstance(header, str) or not isinstance(waveform, str)):
            raise IntegrityError("cached source mapping contains an invalid row")
        if (group, record) in logical:
            raise IntegrityError("cached source mapping duplicates a logical record")
        logical.add((group, record))
        if header not in checksums or waveform not in checksums:
            raise IntegrityError("cached source mapping references a path absent from the verified manifest")
        if not header.endswith(".hea") or not waveform.endswith(".mat"):
            raise IntegrityError("cached source mapping has an invalid header/waveform extension")
        if header in physical or waveform in physical:
            raise IntegrityError("cached source mapping reuses a physical source path")
        physical.update((header, waveform))
        listed.append((group, record))
    if len(physical) != EXPECTED_FILES:
        raise IntegrityError("cached source mapping has an incorrect unique physical-file count")
    # Re-resolve from the immutable manifest: this rejects any altered, ambiguous,
    # or merely self-consistent mapping rather than trusting the cache's claims.
    expected = resolve_source_files(listed, checksums, tuple(range(1, 36)))
    if rows != expected:
        raise IntegrityError("cached source mapping differs from deterministic manifest resolution")
    return rows


class Acquirer:
    def __init__(self, config: AcquisitionConfig) -> None:
        if not 1 <= config.workers <= MAX_WORKERS:
            raise AcquisitionError(f"workers must be between 1 and {MAX_WORKERS}")
        if config.official and (config.groups != tuple(range(1, 36)) or
                                config.expected_records != EXPECTED_RECORDS or
                                config.base_url != OFFICIAL_BASE or
                                config.manifest_url != OFFICIAL_MANIFEST):
            raise AcquisitionError("official acquisition configuration was modified")
        self.config = config
        self.completed_files = 0
        self.verified_bytes_total = 0
        self.downloaded_bytes_run = 0
        self.states: Counter[str] = Counter()
        self.retry_attempts_total = 0
        self.retry_files: dict[str, dict[str, Any]] = {}
        self.current_group: int | None = None
        self.started_utc = utc_now()
        self.manifest_sha256: str | None = None
        self.source_mapping_sha256: str | None = None
        self.source_index_mode: str | None = None
        self.lock = ProcessLock(
            config.out_dir / "acquisition.lock",
            {"pid": os.getpid(), "started_utc": self.started_utc,
             "command": "official_ningbo_acquisition_v1", "workers": config.workers},
        )

    def status(self, stage: str, **extra: Any) -> None:
        value = {
            "stage": stage, "pid": os.getpid(), "updated_utc": utc_now(),
            "release": "PhysioNet Challenge 2021 v1.0.3",
            "current_group": None if self.current_group is None else f"g{self.current_group}",
            "completed_files": self.completed_files, "expected_files": self.config.expected_files,
            "verified_bytes_total": self.verified_bytes_total,
            "downloaded_bytes_run": self.downloaded_bytes_run,
            "retry_attempts_total": self.retry_attempts_total,
            "workers": self.config.workers,
            "official_manifest_sha256": self.manifest_sha256,
            "source_mapping_sha256": self.source_mapping_sha256,
            "source_index_mode": self.source_index_mode, **extra,
        }
        atomic_json(self.config.out_dir / "ACQUISITION_STATUS.json", value, telemetry=True)
        print(json.dumps(value, sort_keys=True), flush=True)

    def progress(self, force: bool = False) -> None:
        if not force and self.completed_files % 25:
            return
        value = {
            "stage": "official_ningbo_acquisition", "pid": os.getpid(),
            "updated_utc": utc_now(),
            "current_group": None if self.current_group is None else f"g{self.current_group}",
            "completed_files": self.completed_files, "expected_files": self.config.expected_files,
            "completed_records_equivalent": self.completed_files // 2,
            "verified_bytes_total": self.verified_bytes_total,
            "downloaded_bytes_run": self.downloaded_bytes_run,
            "states": dict(sorted(self.states.items())),
            "retry_attempts_total": self.retry_attempts_total,
            "workers": self.config.workers,
            "official_manifest_sha256": self.manifest_sha256,
            "source_mapping_sha256": self.source_mapping_sha256,
            "source_index_mode": self.source_index_mode,
            "per_file_retry_information": dict(sorted(self.retry_files.items())),
            "progress_flush_interval_files": 25,
        }
        atomic_json(self.config.out_dir / "DOWNLOAD_PROGRESS.json", value, telemetry=True)

    def freeze(self, protocol_hash: str | None) -> None:
        implementation_hash = sha256_file(Path(__file__).resolve())
        exception_path = self.config.out_dir / "ACQUISITION_CHECKSUM_EXCEPTION_v1.json"
        atomic_json(self.config.out_dir / "ACQUISITION_IMPLEMENTATION_FREEZE.json", {
            "stage": "official_ningbo_acquisition_implementation_freeze",
            "created_utc": utc_now(), "acquisition_code": str(Path(__file__).resolve()),
            "acquisition_code_sha256": implementation_hash,
            "frozen_protocol": None if self.config.protocol_path is None else str(self.config.protocol_path),
            "frozen_protocol_sha256": protocol_hash,
            "checksum_exception_artifact": str(exception_path) if exception_path.exists() else None,
            "checksum_exception_artifact_sha256": sha256_file(exception_path) if exception_path.exists() else None,
            "release": "PhysioNet Challenge 2021 v1.0.3",
            "expected_records": self.config.expected_records,
            "expected_files": self.config.expected_files, "workers": self.config.workers,
            "official_manifest_sha256": self.manifest_sha256,
            "source_mapping_sha256": self.source_mapping_sha256,
            "source_index_mode": self.source_index_mode,
        })

    def load_official_source_indices(self) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Use fully verified local source indices before making any index-network request."""
        manifest_path = self.config.out_dir / "OFFICIAL_SHA256SUMS.txt"
        mapping_path = self.config.out_dir / "SOURCE_RECORD_MAPPING.json"
        if manifest_path.exists() and mapping_path.exists():
            try:
                manifest_raw = manifest_path.read_bytes()
                checksums = validate_official_manifest(manifest_raw)
                mapping_raw = mapping_path.read_bytes()
                resolved = validate_official_source_mapping(mapping_raw, checksums)
            except (OSError, IntegrityError):
                # Never use a partially valid cache. The immutable release source
                # will be refetched and revalidated below.
                pass
            else:
                self.manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
                self.source_mapping_sha256 = hashlib.sha256(mapping_raw).hexdigest()
                self.source_index_mode = "verified_local_cache"
                return checksums, resolved

        manifest_raw = get_bytes(self.config.manifest_url)
        checksums = validate_official_manifest(manifest_raw)
        records = enumerate_records(self.config)
        resolved = resolve_source_files(records, checksums, self.config.groups)
        atomic_text(manifest_path, manifest_raw.decode("utf-8", errors="strict"))
        write_checksum_exception(self.config.out_dir, hashlib.sha256(manifest_raw).hexdigest(), checksums)
        write_source_amendment(self.config.out_dir, resolved, hashlib.sha256(manifest_raw).hexdigest())
        mapping_raw = mapping_path.read_bytes()
        # Validate the generated artifact too, before it becomes a trusted cache.
        resolved = validate_official_source_mapping(mapping_raw, checksums)
        self.manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        self.source_mapping_sha256 = hashlib.sha256(mapping_raw).hexdigest()
        self.source_index_mode = "official_network_refetch"
        return checksums, resolved

    @staticmethod
    def failure_label(error: BaseException) -> str:
        if "checksum mismatch" in str(error):
            return "checksum_mismatch"
        if "truncated download" in str(error):
            return "truncated_content_length"
        if isinstance(error, (TimeoutError, ConnectionResetError, http.client.IncompleteRead)):
            return "transport"
        if isinstance(error, (URLError, HTTPError, OSError)):
            return "http_or_transport"
        return "transfer_failure"

    @staticmethod
    def preserve_failed_partial(temporary: Path, destination: Path, attempt: int,
                                label: str) -> str | None:
        if not temporary.exists():
            return None
        retained = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}."
            f"attempt-{attempt}.failed-{label}.part"
        )
        os.replace(temporary, retained)
        return retained.name

    def download_attempt(self, key: str, expected: str, temporary: Path) -> tuple[int, str]:
        """Download one immutable source object to a unique unpromoted partial."""
        digest = hashlib.sha256()
        byte_count = 0
        try:
            request = Request(f"{self.config.base_url}/{key}",
                              headers={"User-Agent": "ningbo-acquisition-v1/1.0"})
            with urlopen(request, timeout=120) as response, temporary.open("xb") as handle:
                length_header = response.headers.get("Content-Length")
                try:
                    expected_length = int(length_header) if length_header is not None else None
                except ValueError as error:
                    raise IntegrityError(f"malformed Content-Length for {key}") from error
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
                    digest.update(block)
                    byte_count += len(block)
                handle.flush()
                os.fsync(handle.fileno())
        except HTTPError as error:
            if error.code in {408, 429} or 500 <= error.code <= 599:
                raise TransientDownloadError(f"transient HTTP {error.code} for {key}") from error
            raise AcquisitionError(f"non-retryable HTTP {error.code} for {key}") from error
        except (URLError, TimeoutError, ConnectionResetError, http.client.IncompleteRead, OSError) as error:
            raise TransientDownloadError(f"transient transfer failure for {key}: {error}") from error
        if expected_length is not None and byte_count != expected_length:
            raise TransientDownloadError(
                f"truncated download for {key}: {byte_count} != {expected_length}"
            )
        observed = digest.hexdigest()
        if observed != expected:
            if checksum_exception_applies(key, expected, observed, temporary):
                return byte_count, "checksum_exception_validated"
            raise TransientDownloadError(f"download checksum mismatch for {key}")
        return byte_count, "downloaded"

    def download_one(self, key: str, checksums: dict[str, str]) -> dict[str, Any]:
        expected = checksums.get(key)
        if expected is None or SHA256.fullmatch(expected) is None:
            raise IntegrityError(f"missing official checksum for {key}")
        prefix = "training/ningbo/"
        if not key.startswith(prefix):
            raise IntegrityError(f"non-Ningbo source path rejected: {key}")
        relative = key.removeprefix(prefix)
        destination = self.config.data_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            observed = sha256_file(destination)
            if observed != expected and not checksum_exception_applies(key, expected, observed, destination):
                raise IntegrityError(f"existing unverified file will not be overwritten: {destination}")
            return {"state": "verified_existing", "bytes": destination.stat().st_size,
                    "file": relative, "retry_attempts": 0}
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.attempt-{attempt}.part"
            )
            try:
                byte_count, state = self.download_attempt(key, expected, temporary)
                # A competing/nonconforming process must never cause a replacement.
                if destination.exists():
                    existing_observed = sha256_file(destination)
                    if existing_observed == expected or checksum_exception_applies(key, expected, existing_observed, destination):
                        temporary.unlink(missing_ok=True)
                        return {"state": "verified_existing", "bytes": destination.stat().st_size,
                                "file": relative, "retry_attempts": attempt - 1}
                    raise IntegrityError(f"refusing to overwrite unverified destination: {destination}")
                os.replace(temporary, destination)
                if attempts:
                    self.retry_files[relative] = {"attempts": attempt, "retries": attempt - 1,
                                                  "attempt_errors": attempts, "outcome": "verified"}
                return {"state": state, "bytes": byte_count, "file": relative,
                        "retry_attempts": attempt - 1}
            except TransientDownloadError as error:
                label = self.failure_label(error)
                retained = self.preserve_failed_partial(temporary, destination, attempt, label)
                detail = {"attempt": attempt, "error": str(error), "retained_partial": retained}
                attempts.append(detail)
                self.retry_attempts_total += 1
                self.retry_files[relative] = {"attempts": attempt, "retries": attempt,
                                              "attempt_errors": attempts, "outcome": "retrying"}
                if attempt == MAX_DOWNLOAD_ATTEMPTS:
                    self.retry_files[relative]["outcome"] = "exhausted"
                    raise AcquisitionError(
                        f"download retries exhausted for {key}; attempt errors: "
                        f"{json.dumps(attempts, sort_keys=True)}"
                    ) from error
                # Bounded exponential retry delay: 0.25, 0.5, 1, 2 seconds plus <=0.1 jitter.
                time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.1))
            except Exception:
                # Non-retryable integrity and destination-safety failures remain fail-closed.
                raise
        raise AssertionError("unreachable retry loop")

    def consume(self, result: dict[str, Any]) -> None:
        self.completed_files += 1
        self.verified_bytes_total += int(result["bytes"])
        self.states[str(result["state"])] += 1
        if result["state"] in {"downloaded", "checksum_exception_validated"}:
            self.downloaded_bytes_run += int(result["bytes"])
        self.progress()

    def run(self) -> None:
        protocol_hash: str | None = None
        if self.config.protocol_path is not None:
            protocol_hash = sha256_file(self.config.protocol_path).upper()
            if self.config.official and protocol_hash != EXPECTED_PROTOCOL_SHA256:
                raise IntegrityError("frozen protocol SHA-256 mismatch; refusing network acquisition")
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        self.freeze(protocol_hash)
        self.lock.acquire()
        try:
            self.status("ENUMERATING_OFFICIAL_RECORDS")
            if self.config.official:
                checksums, resolved = self.load_official_source_indices()
            else:
                manifest_bytes = get_bytes(self.config.manifest_url)
                checksums = parse_manifest(manifest_bytes)
                records = enumerate_records(self.config)
                resolved = resolve_source_files(records, checksums, self.config.groups)
                atomic_json(self.config.out_dir / "SOURCE_RECORD_MAPPING.json", {
                    "version": "synthetic_logical_to_physical_mapping_v1",
                    "logical_record_count": len(resolved),
                    "physical_file_count": len(resolved) * 2,
                    "records": resolved,
                })
            self.freeze(protocol_hash)
            expected_keys = {
                path for row in resolved for path in (row["header_path"], row["waveform_path"])
            }
            for group in self.config.groups:
                self.current_group = group
                selected = sorted(
                    key for key in expected_keys
                    if key.startswith(f"training/ningbo/g{group}/")
                )
                if not selected:
                    raise IntegrityError(f"source enumeration omitted g{group}")
                self.status("DOWNLOADING")
                with futures.ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="ningbo-download") as pool:
                    jobs = [pool.submit(self.download_one, key, checksums) for key in selected]
                    for job in futures.as_completed(jobs):
                        self.consume(job.result())
                self.progress(force=True)
                self.status("GROUP_VERIFIED", group_completed_files=len(selected))
            if self.completed_files != self.config.expected_files:
                raise IntegrityError(f"completed {self.completed_files} files, expected {self.config.expected_files}")
            remaining_final = [key for key in expected_keys if not (self.config.data_dir / key.removeprefix("training/ningbo/")).exists()]
            if remaining_final:
                raise IntegrityError(f"completion has {len(remaining_final)} missing final file(s)")
            self.progress(force=True)
            self.status("ACQUISITION_COMPLETE", checksum_verified_files=self.completed_files,
                        part_files_forbidden_at_completion=False,
                        note="unique partial files from interrupted attempts are retained safely")
        except Exception as error:
            self.progress(force=True)
            self.status("FAILED", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self.lock.release()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def run_self_test() -> None:
    """Exercise success, checksum rejection, reuse, interrupted resumption, and lock exclusion locally."""
    global EXPECTED_RECORDS, EXPECTED_FILES, EXPECTED_OFFICIAL_MANIFEST_SHA256
    root = Path(tempfile.mkdtemp(prefix="ningbo-acquisition-selftest-"))
    server: socketserver.TCPServer | None = None
    try:
        # Windows readers can briefly lock live JSON telemetry. Two lock failures
        # must retry to success, while an exhausted telemetry-only failure warns
        # and returns without aborting its caller. Scientific JSON stays strict.
        telemetry_dir = root / "telemetry"
        telemetry_target = telemetry_dir / "DOWNLOAD_PROGRESS.json"
        original_replace = os.replace
        replace_calls = 0
        def fail_twice(source: Any, target: Any) -> None:
            nonlocal replace_calls
            if Path(target) == telemetry_target and replace_calls < 2:
                replace_calls += 1
                raise PermissionError(5, "simulated Windows sharing violation")
            original_replace(source, target)
        try:
            os.replace = fail_twice  # type: ignore[assignment]
            assert atomic_json(telemetry_target, {"telemetry": "retry"}, telemetry=True)
            assert replace_calls == 2
        finally:
            os.replace = original_replace  # type: ignore[assignment]
        persistent_target = telemetry_dir / "ACQUISITION_STATUS.json"
        strict_target = telemetry_dir / "ACQUISITION_IMPLEMENTATION_FREEZE.json"
        def always_fail(source: Any, target: Any) -> None:
            if Path(target) in {persistent_target, strict_target}:
                raise PermissionError(32, "simulated Windows sharing violation")
            original_replace(source, target)
        try:
            os.replace = always_fail  # type: ignore[assignment]
            assert not atomic_json(persistent_target, {"telemetry": "nonfatal"}, telemetry=True)
            try:
                atomic_json(strict_target, {"strict": True})
                raise AssertionError("strict scientific artifact replace was not tested")
            except PermissionError:
                pass
        finally:
            os.replace = original_replace  # type: ignore[assignment]
        assert (telemetry_dir / "telemetry_warnings.log").exists()
        assert not list(telemetry_dir.glob(".*.tmp"))
        # The official-source exception has no wildcard behavior: all five
        # identifiers (path, manifest hash, observed hash, size, structure) bind it.
        exception_file = root / "exception" / "JS13118.mat"
        exception_file.parent.mkdir(parents=True, exist_ok=True)
        exception_file.write_bytes(struct.pack("<5i", 30, 12, 5000, 0, 4) + b"val\x00" + b"\x00" * 120000)
        assert checksum_exception_applies(CHECKSUM_EXCEPTION_KEY, CHECKSUM_EXCEPTION_STALE_EXPECTED,
                                          CHECKSUM_EXCEPTION_OBSERVED, exception_file)
        assert not checksum_exception_applies("training/ningbo/g3/other.mat", CHECKSUM_EXCEPTION_STALE_EXPECTED,
                                              CHECKSUM_EXCEPTION_OBSERVED, exception_file)
        assert not checksum_exception_applies(CHECKSUM_EXCEPTION_KEY, "0" * 64,
                                              CHECKSUM_EXCEPTION_OBSERVED, exception_file)
        assert not checksum_exception_applies(CHECKSUM_EXCEPTION_KEY, CHECKSUM_EXCEPTION_STALE_EXPECTED,
                                              "1" * 64, exception_file)
        exception_file.write_bytes(exception_file.read_bytes()[:-1])
        assert not checksum_exception_applies(CHECKSUM_EXCEPTION_KEY, CHECKSUM_EXCEPTION_STALE_EXPECTED,
                                              CHECKSUM_EXCEPTION_OBSERVED, exception_file)
        exception_file.write_bytes(struct.pack("<5i", 30, 12, 5000, 0, 4) + b"bad\x00" + b"\x00" * 120000)
        assert not checksum_exception_applies(CHECKSUM_EXCEPTION_KEY, CHECKSUM_EXCEPTION_STALE_EXPECTED,
                                              CHECKSUM_EXCEPTION_OBSERVED, exception_file)
        release = root / "release"
        files = {"g1/a.hea": b"a header\n", "g1/a.mat": b"a waveform\x00",
                 "g2/b.hea": b"b header\n", "g2/b.mat": b"b waveform\x01"}
        for relative, content in files.items():
            path = release / "training" / "ningbo" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (release / "training" / "ningbo" / "g1" / "RECORDS").write_text("a\n", encoding="utf-8")
        (release / "training" / "ningbo" / "g2" / "RECORDS").write_text("b\n", encoding="utf-8")
        manifest_lines = [f"{hashlib.sha256(content).hexdigest()}  training/ningbo/{relative}"
                          for relative, content in sorted(files.items())]
        (release / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(release), **kwargs)
        server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        def config(name: str) -> AcquisitionConfig:
            return AcquisitionConfig(base, f"{base}/SHA256SUMS.txt", root / name / "data", root / name / "out",
                                     (1, 2), 2, protocol_path=None, official=False)
        assert config("default_workers").workers == DEFAULT_WORKERS == 4
        for worker_count in (1, 4, 12):
            Acquirer(AcquisitionConfig(base, f"{base}/SHA256SUMS.txt", root / "workers" / str(worker_count) / "data",
                                        root / "workers" / str(worker_count) / "out", (1, 2), 2,
                                        workers=worker_count, protocol_path=None, official=False))
        for worker_count in (0, 13):
            try:
                Acquirer(AcquisitionConfig(base, f"{base}/SHA256SUMS.txt", root / "invalid" / "data",
                                            root / "invalid" / "out", (1, 2), 2, workers=worker_count,
                                            protocol_path=None, official=False))
                raise AssertionError("invalid worker bound was accepted")
            except AcquisitionError:
                pass
        first = Acquirer(config("success")); first.run()
        assert first.completed_files == 4 and first.states["downloaded"] == 4
        assert json.loads((first.config.out_dir / "ACQUISITION_IMPLEMENTATION_FREEZE.json").read_text())["workers"] == 4
        assert json.loads((first.config.out_dir / "DOWNLOAD_PROGRESS.json").read_text())["workers"] == 4
        reused = Acquirer(config("success")); reused.run()
        assert reused.states["verified_existing"] == 4 and reused.downloaded_bytes_run == 0
        # A process-unique partial left by interruption cannot affect a restarted acquisition.
        interrupted_config = config("interrupted")
        partial = interrupted_config.data_dir / "g1" / ".a.hea.interrupted.part"
        partial.parent.mkdir(parents=True, exist_ok=True); partial.write_bytes(b"incomplete")
        resumed = Acquirer(interrupted_config); resumed.run()
        assert (interrupted_config.data_dir / "g1" / "a.hea").exists() and partial.exists()
        # Serve content that no longer matches the immutable manifest, and ensure no final is committed.
        rejected_config = config("rejected")
        corrupt = release / "training" / "ningbo" / "g1" / "a.mat"
        original = corrupt.read_bytes(); corrupt.write_bytes(b"corrupt")
        try:
            Acquirer(rejected_config).run()
            raise AssertionError("checksum rejection did not occur")
        except AcquisitionError:
            assert not (rejected_config.data_dir / "g1" / "a.mat").exists()
            retained = list((rejected_config.data_dir / "g1").glob("*.failed-checksum_mismatch.part"))
            assert len(retained) == MAX_DOWNLOAD_ATTEMPTS
        finally:
            corrupt.write_bytes(original)
        locked_config = config("locked")
        lock = ProcessLock(locked_config.out_dir / "acquisition.lock", {"pid": os.getpid(), "test": True})
        lock.acquire()
        try:
            try:
                Acquirer(locked_config).run()
                raise AssertionError("duplicate live process was not rejected")
            except AcquisitionError as error:
                assert "duplicate acquisition rejected" in str(error)
        finally:
            lock.release()
        # A flaky endpoint delivers a bad body once and the manifest-correct body
        # on its second request. It must complete via the bounded retry path.
        retry_release = root / "retry_release"
        retry_files = {"g1/retry.hea": b"retry header\n", "g1/retry.mat": b"retry waveform\x02"}
        for relative, content in retry_files.items():
            path = retry_release / "training" / "ningbo" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (retry_release / "training" / "ningbo" / "g1" / "RECORDS").write_text("retry\n", encoding="utf-8")
        (retry_release / "SHA256SUMS.txt").write_text(
            "\n".join(f"{hashlib.sha256(content).hexdigest()}  training/ningbo/{relative}"
                      for relative, content in sorted(retry_files.items())) + "\n", encoding="utf-8"
        )
        class CorruptOnceHandler(QuietHandler):
            corrupt_requests = 0

            def do_GET(self) -> None:
                if self.path == "/training/ningbo/g1/retry.mat" and type(self).corrupt_requests == 0:
                    type(self).corrupt_requests += 1
                    body = b"corrupt-first-response"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()
        retry_server = socketserver.TCPServer(("127.0.0.1", 0),
                                              lambda *a, **kw: CorruptOnceHandler(*a, directory=str(retry_release), **kw))
        try:
            threading.Thread(target=retry_server.serve_forever, daemon=True).start()
            retry_base = f"http://127.0.0.1:{retry_server.server_address[1]}"
            retry_config = AcquisitionConfig(retry_base, f"{retry_base}/SHA256SUMS.txt",
                                            root / "retry" / "data", root / "retry" / "out", (1,), 1,
                                            protocol_path=None, official=False)
            retried = Acquirer(retry_config); retried.run()
            assert retried.completed_files == 2 and retried.retry_attempts_total == 1
            retry_info = retried.retry_files["g1/retry.mat"]
            assert retry_info["retries"] == 1 and retry_info["outcome"] == "verified"
        finally:
            retry_server.shutdown(); retry_server.server_close()
        # RECORDS may list a boundary record in the preceding group, while the
        # immutable manifest carries the canonical physical group. The known
        # JS23074 source also has a different official header basename.
        mapping_checks = {
            "training/ningbo/g2/move.hea": "0" * 64,
            "training/ningbo/g2/move.mat": "1" * 64,
            "training/ningbo/g13/S23074.hea": "2" * 64,
            "training/ningbo/g13/JS23074.mat": "3" * 64,
        }
        mapping = resolve_source_files(
            [(1, "move"), (13, "JS23074")], mapping_checks, (1, 2, 13)
        )
        assert mapping[0]["header_path"].startswith("training/ningbo/g2/")
        assert mapping[1]["header_path"].endswith("/S23074.hea")
        ambiguous = dict(mapping_checks)
        ambiguous["training/ningbo/g1/move.hea"] = "4" * 64
        try:
            resolve_source_files([(1, "move"), (13, "JS23074")], ambiguous, (1, 2, 13))
            raise AssertionError("ambiguous physical mapping was not rejected")
        except IntegrityError as error:
            assert "2 hea candidates" in str(error)
        # The official-index cache is intentionally independent from the download
        # server.  A complete synthetic release proves cache reuse makes no index
        # HTTP call, while one-byte manifest or mapping tampering is rejected.
        old_records, old_files, old_manifest_hash = (
            EXPECTED_RECORDS, EXPECTED_FILES, EXPECTED_OFFICIAL_MANIFEST_SHA256
        )
        try:
            cache_count = 34_905
            EXPECTED_RECORDS, EXPECTED_FILES = cache_count, cache_count * 2
            cache_checksums: dict[str, str] = {}
            cache_rows: list[dict[str, Any]] = []
            for number in range(cache_count):
                record = f"cache{number:05d}"
                header = f"training/ningbo/g1/{record}.hea"
                waveform = f"training/ningbo/g1/{record}.mat"
                cache_checksums[header] = hashlib.sha256(header.encode()).hexdigest()
                cache_checksums[waveform] = hashlib.sha256(waveform.encode()).hexdigest()
                cache_rows.append({"listed_group": 1, "logical_record": record,
                                   "header_path": header, "waveform_path": waveform})
            cache_manifest = "".join(
                f"{digest}  {key}\n" for key, digest in sorted(cache_checksums.items())
            ).encode("utf-8")
            EXPECTED_OFFICIAL_MANIFEST_SHA256 = hashlib.sha256(cache_manifest).hexdigest()
            cache_root = root / "verified_index_cache"
            cache_out = cache_root / "out"; cache_data = cache_root / "data"
            cache_out.mkdir(parents=True)
            (cache_out / "OFFICIAL_SHA256SUMS.txt").write_bytes(cache_manifest)
            cache_mapping = {"version": "ningbo_logical_to_physical_mapping_v1",
                             "logical_record_count": cache_count,
                             "physical_file_count": cache_count * 2, "records": cache_rows}
            atomic_json(cache_out / "SOURCE_RECORD_MAPPING.json", cache_mapping)
            cache_acquirer = Acquirer(AcquisitionConfig("https://cache.invalid", "https://cache.invalid/SHA256SUMS.txt",
                                                         cache_data, cache_out, (1,), cache_count,
                                                         protocol_path=None, official=False))
            original_get_bytes = get_bytes
            def forbidden_index_http(url: str) -> bytes:
                raise AssertionError(f"verified local cache unexpectedly requested {url}")
            globals()["get_bytes"] = forbidden_index_http
            try:
                cache_acquirer.load_official_source_indices()
            finally:
                globals()["get_bytes"] = original_get_bytes
            assert cache_acquirer.source_index_mode == "verified_local_cache"
            assert cache_acquirer.source_mapping_sha256 == sha256_file(cache_out / "SOURCE_RECORD_MAPPING.json")
            tampered_manifest = bytearray(cache_manifest); tampered_manifest[-2] ^= 1
            try:
                validate_official_manifest(bytes(tampered_manifest))
                raise AssertionError("tampered cached manifest was accepted")
            except IntegrityError:
                pass
            tampered_mapping = json.loads((cache_out / "SOURCE_RECORD_MAPPING.json").read_text())
            tampered_mapping["records"][0]["header_path"] = "training/ningbo/g1/not-in-manifest.hea"
            try:
                validate_official_source_mapping(json.dumps(tampered_mapping).encode(), cache_checksums)
                raise AssertionError("tampered cached mapping was accepted")
            except IntegrityError:
                pass
        finally:
            EXPECTED_RECORDS, EXPECTED_FILES, EXPECTED_OFFICIAL_MANIFEST_SHA256 = (
                old_records, old_files, old_manifest_hash
            )
        print(json.dumps({"self_test": "PASS", "success_download": "PASS", "checksum_rejection": "PASS",
                          "verified_file_reuse": "PASS", "interrupted_download_resume": "PASS",
                          "duplicate_live_process_rejection": "PASS",
                          "corrupt_once_retry": "PASS",
                          "telemetry_replace_retry": "PASS", "telemetry_nonfatal_warning": "PASS",
                          "scientific_artifact_replace_strict": "PASS",
                          "checksum_exception_exact_acceptance_and_rejections": "PASS",
                          "worker_default_and_bounds": "PASS", "worker_artifact_recording": "PASS",
                          "cross_group_resolution": "PASS", "basename_alias_resolution": "PASS",
                          "ambiguous_mapping_rejection": "PASS", "verified_index_cache_no_http": "PASS",
                          "tampered_index_cache_rejection": "PASS"}, sort_keys=True), flush=True)
    finally:
        if server is not None:
            server.shutdown(); server.server_close()
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run local fake-source durability tests only")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"bounded public-server download workers (1..{MAX_WORKERS}; default: {DEFAULT_WORKERS})")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    config = AcquisitionConfig(OFFICIAL_BASE, OFFICIAL_MANIFEST, DATA, OUT, tuple(range(1, 36)), EXPECTED_RECORDS,
                               workers=args.workers)
    Acquirer(config).run()


if __name__ == "__main__":
    main()
