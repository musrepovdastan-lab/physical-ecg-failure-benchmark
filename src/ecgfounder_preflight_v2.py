"""Pre-outcome v2 path, waveform-hash, and missing-sample audit."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import wfdb
from scipy.io import loadmat

from src.ecgfounder_baseline_v2 import LEADS, ROOT, OUT, physical_signal, reverse_ra_ll, preprocess, model, infer, record_path

SPLITS = ("train", "validation", "test")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def schema(header: Path, waveform_name: str):
    lines = header.read_text(encoding="utf-8").splitlines()
    first = lines[0].split()
    if len(first) < 4 or first[:4] != [header.stem, "12", "500", "5000"]:
        raise RuntimeError(f"header schema mismatch: {header}")
    leads = tuple(line.split()[-1] for line in lines[1:13])
    if leads != LEADS:
        raise RuntimeError(f"header lead order mismatch: {header}")
    if any(line.split()[0] != waveform_name for line in lines[1:13]):
        raise RuntimeError(f"header references unexpected waveform: {header}")
    return lines


def canonical_file(relative: str, suffix: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or "\\" in relative or ".." in path.parts or path.parts[:2] != ("training", "ningbo") or path.suffix != suffix:
        raise RuntimeError(f"invalid canonical path: {relative}")
    absolute = ROOT / "data/challenge2021_v103/ningbo" / Path(*path.parts[2:])
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return absolute


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json").exists():
        raise RuntimeError("v2 protocol already frozen; do not re-run preflight")
    ningbo_split_path = ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"
    ningbo_canon_path = ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json"
    chapman_split_path = ROOT / "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json"
    ningbo_rows = json.loads(ningbo_split_path.read_text(encoding="utf-8"))["rows"]
    canonical_rows = json.loads(ningbo_canon_path.read_text(encoding="utf-8"))["records"]
    chapman_rows = json.loads(chapman_split_path.read_text(encoding="utf-8"))["rows"]
    canonical = {r["record_id"]: r for r in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise RuntimeError("duplicate canonical ID")
    if {r["record_id"] for r in ningbo_rows} != set(canonical):
        raise RuntimeError("canonical and split ID sets differ")
    if len({r["record_id"] for r in chapman_rows}) != len(chapman_rows):
        raise RuntimeError("duplicate Chapman split ID")
    counts = {d: {s: {"records": 0, "affected_records": 0, "missing_samples": 0,
                     "path_group_mismatches": 0, "header_waveform_stem_mismatches": 0} for s in SPLITS}
              for d in ("Chapman", "Ningbo")}
    examples = {d: {s: [] for s in SPLITS} for d in ("Chapman", "Ningbo")}
    total = len(chapman_rows) + len(ningbo_rows)
    done = 0
    for dataset, records in (("Chapman", chapman_rows), ("Ningbo", ningbo_rows)):
        for r in records:
            record_id, split = r["record_id"], r["split"]
            if split not in SPLITS:
                raise RuntimeError(f"unknown split: {record_id}")
            if dataset == "Ningbo":
                c = canonical[record_id]
                if c["record_id"] != record_id or c["waveform_sha256"] != r["waveform_sha256"]:
                    raise RuntimeError(f"canonical/split waveform-hash mismatch: {record_id}")
                header = canonical_file(c["header_path"], ".hea")
                waveform = canonical_file(c["waveform_path"], ".mat")
                if waveform.stem != record_id.split("/")[-1]:
                    raise RuntimeError(f"header/waveform ID mismatch: {record_id}")
                if header.stem != waveform.stem:
                    counts[dataset][split]["header_waveform_stem_mismatches"] += 1
                if waveform.parent.name != record_id.split("/")[0]:
                    counts[dataset][split]["path_group_mismatches"] += 1
                if not set(r["labels"]) <= {0, 1} or r["labels"] != c["labels"]:
                    raise RuntimeError(f"canonical/split label mismatch: {record_id}")
            else:
                header = (ROOT / "data/challenge2021_v103/chapman_shaoxing" / record_id).with_suffix(".hea")
                waveform = header.with_suffix(".mat")
                if not header.is_file() or not waveform.is_file():
                    raise FileNotFoundError(f"Chapman files missing: {record_id}")
            schema(header, waveform.name)
            raw = np.asarray(loadmat(waveform)["val"])
            if raw.shape != (12, 5000) or not np.isfinite(raw).all():
                raise RuntimeError(f"digital waveform schema or finiteness mismatch: {record_id}")
            if dataset == "Ningbo":
                actual = hashlib.sha256(np.ascontiguousarray(raw.astype(np.float32)).tobytes()).hexdigest()
                if actual != r["waveform_sha256"]:
                    raise RuntimeError(f"source waveform-hash mismatch: {record_id}")
            missing = int(np.count_nonzero(raw == -32768))
            counts[dataset][split]["records"] += 1
            if missing:
                counts[dataset][split]["affected_records"] += 1
                counts[dataset][split]["missing_samples"] += missing
                if len(examples[dataset][split]) < 8:
                    examples[dataset][split].append({"record_id": record_id, "missing_samples": missing,
                                                     "waveform_path": str(waveform.relative_to(ROOT)).replace("\\", "/")})
                # Verify that WFDB maps exactly these digital sentinels to NaN.
                physical, metadata = wfdb.rdsamp(str(header.with_suffix("")))
                if physical.shape != (5000, 12) or metadata["fs"] != 500 or tuple(metadata["sig_name"]) != LEADS:
                    raise RuntimeError(f"WFDB schema mismatch: {record_id}")
                if np.isinf(physical).any() or int(np.isnan(physical).sum()) != missing:
                    raise RuntimeError(f"WFDB missing-value mismatch: {record_id}")
            done += 1
            if done % 500 == 0 or done == total:
                save(OUT / "PREFLIGHT_PROGRESS.json", {"stage": "SCANNING_INPUTS", "dataset": dataset,
                                                         "records_completed": done, "total_records": total,
                                                         "updated_utc": utc()})
                print(f"preflight {done}/{total} {dataset}", flush=True)
    # The known failures are training records and must resolve without a fallback path.
    known = []
    for record_id in ("g1/JS10767", "g11/JS21646", "g13/JS23074"):
        src = record_path("Ningbo", record_id)
        x = physical_signal("Ningbo", record_id)
        rev = reverse_ra_ll(x)
        if not np.allclose(rev[6:], x[6:], rtol=0, atol=1e-12):
            raise RuntimeError(f"chest lead changed: {record_id}")
        known.append({"record_id": record_id, "resolved_stem": str(src.relative_to(ROOT)).replace("\\", "/"),
                      "clean_finite": bool(np.isfinite(preprocess(x)).all()),
                      "reversal_finite": bool(np.isfinite(preprocess(rev)).all())})
    audit = {"status": "PASS", "version": "ecgfounder_baseline_v2", "completed_utc": utc(),
             "total_records": total, "counts_by_dataset_split": counts, "examples": examples,
             "known_failure_cases": known,
             "waveform_hash_verification": "every Ningbo .mat converted to float32 and SHA-256 checked against frozen split and canonical mapping",
             "wfdb_missing_check": "every sentinel-affected recording read through WFDB; NaN count required to equal digital -32768 count",
             "bindings": {
                 "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json": sha(ningbo_split_path),
                 "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json": sha(ningbo_canon_path),
                 "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json": sha(chapman_split_path),
             }}
    save(OUT / "PRE_RUN_AUDIT.json", audit)
    save(OUT / "MISSING_PATH_AUDIT.json", {"counts_by_dataset_split": counts, "examples": examples,
                                           "known_failure_cases": known})
    print(json.dumps({"status": "PASS", "counts": counts, "known": known}), flush=True)


if __name__ == "__main__":
    main()
