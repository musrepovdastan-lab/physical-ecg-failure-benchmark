"""Bounded, Chapman-only acquisition, integrity audit, and deterministic split.

This program never constructs a Ningbo URL.  It downloads only PhysioNet's public
Challenge 2021 v1.0.3 Chapman-Shaoxing files, validates each file against the
release SHA-256 manifest, and refuses to turn malformed data into exclusions.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "challenge2021_v103" / "chapman_shaoxing"
OUT = ROOT / "outputs" / "phase1_chapman_seed1_v3"
BASE = "https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing"
RECORDS_URL = "https://physionet.org/files/challenge-2021/1.0.3/RECORDS"
CHECKSUM_URL = "https://physionet.org/files/challenge-2021/1.0.3/SHA256SUMS.txt"
EXPECTED_RECORDS = 10247
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["164934002", "426177001", "427084000"]
SPLIT_SEED = "20260828"
LOCK = OUT / "chapman_download.lock"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temp, path)


def get_bytes(url: str, timeout: int = 90) -> bytes:
    request = Request(url, headers={"User-Agent": "chapman-phase1-v3/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def release_manifest() -> dict[str, str]:
    cache = OUT / "official_SHA256SUMS.txt"
    if not cache.exists():
        raw = get_bytes(CHECKSUM_URL)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line.strip())
        if match:
            entries[match.group(2).replace("\\", "/")] = match.group(1).lower()
    if not entries:
        raise RuntimeError("official SHA256SUMS parsed to zero entries")
    return entries


def remote_records() -> list[str]:
    # Challenge 2021's release-level RECORDS lists source directories rather than
    # individual records.  Each whitelisted Chapman group has a public index whose
    # .hea links are the authoritative record list used here.
    records: list[str] = []
    for group in range(1, 12):
        listing = get_bytes(f"{BASE}/g{group}/").decode("utf-8", errors="strict")
        headers = re.findall(r'href="([A-Za-z0-9_-]+\.hea)"', listing)
        if not headers or len(headers) != len(set(headers)):
            raise RuntimeError(f"could not uniquely parse Chapman g{group} header index")
        records.extend(f"g{group}/{Path(header).stem}" for header in headers)
    if len(records) != EXPECTED_RECORDS or len(set(records)) != EXPECTED_RECORDS:
        raise RuntimeError(f"official Chapman RECORDS expected {EXPECTED_RECORDS}; got {len(records)}")
    if any(not re.fullmatch(r"g(?:[1-9]|1[01])/[A-Za-z0-9_-]+", item) for item in records):
        raise RuntimeError("unexpected record path in official Chapman RECORDS")
    return records


def relative_checksum_key(record: str, extension: str) -> str:
    return f"training/chapman_shaoxing/{record}{extension}"


def download_one(record: str, extension: str, sums: dict[str, str]) -> dict[str, object]:
    key = relative_checksum_key(record, extension)
    expected = sums.get(key)
    if expected is None:
        raise RuntimeError(f"checksum missing for {key}")
    destination = DATA / f"{record}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual == expected:
            return {"file": f"{record}{extension}", "bytes": destination.stat().st_size, "state": "verified_existing"}
        raise RuntimeError(f"existing file checksum mismatch: {destination}")
    raw = get_bytes(f"{BASE}/{record}{extension}")
    actual = sha256_bytes(raw)
    if actual != expected:
        raise RuntimeError(f"download checksum mismatch: {record}{extension}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return {"file": f"{record}{extension}", "bytes": len(raw), "state": "downloaded"}


def download_record(record: str, sums: dict[str, str]) -> list[dict[str, object]]:
    """A record is durable only after its header and waveform both verify."""
    return [download_one(record, extension, sums) for extension in (".hea", ".mat")]


def download_group(group: int, workers: int, max_files: int | None,
                   sums: dict[str, str] | None = None, all_records: list[str] | None = None) -> dict[str, object]:
    sums = release_manifest() if sums is None else sums
    all_records = remote_records() if all_records is None else all_records
    records = [record for record in all_records if record.startswith(f"g{group}/")]
    if not records:
        raise RuntimeError(f"group g{group} absent from official RECORDS")
    selected = records
    complete_group = True
    if max_files is not None:
        # A bounded foreground batch is useful where a supervisor has a short
        # command lifetime.  Existing files are checksum-verified before reuse.
        pending = [record for record in records if not (DATA / f"{record}.hea").exists() or not (DATA / f"{record}.mat").exists()]
        selected = pending[:max_files]
        complete_group = not pending or len(pending) <= max_files
    if not selected:
        print(f"g{group}: no files selected", flush=True)
        return {"group": f"g{group}", "complete_group": complete_group, "records": len(records), "selected_records": 0}
    completed: list[dict[str, object]] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(download_record, record, sums) for record in selected]
        for number, job in enumerate(futures.as_completed(jobs), start=1):
            completed.extend(job.result())
            if number % 100 == 0 or number == len(selected):
                progress = {"stage": "foreground_chapman_download", "group": f"g{group}",
                            "records_completed_in_current_group": number, "records_selected_in_current_group": len(selected),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "ningbo_accessed": False}
                write_json(OUT / "DOWNLOAD_PROGRESS.json", progress)
                print(f"g{group}: {number}/{len(selected)} records verified", flush=True)
    byte_total = sum(int(item["bytes"]) for item in completed)
    audit = {
        "stage": "official_chapman_download",
        "release": "challenge-2021/1.0.3",
        "base_url": BASE + "/",
        "checksum_url": CHECKSUM_URL,
        "group": f"g{group}",
        "records": len(records),
        "files": len(completed), "selected_records": len(selected),
        "bytes": byte_total,
        "states": dict(Counter(str(item["state"]) for item in completed)),
        "all_selected_files_sha256_verified": True,
        "complete_group": complete_group,
        "ningbo_accessed": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / f"DOWNLOAD_AUDIT_g{group}.json", audit)
    print(canonical_json(audit))
    return audit


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"acquisition lock exists: {LOCK}; do not launch a duplicate downloader") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "command": "download-all", "source": "Chapman-Shaoxing only"}, handle)


def download_all() -> None:
    acquire_lock()
    started = datetime.now(timezone.utc)
    try:
        sums, records = release_manifest(), remote_records()
        summaries = []
        for group in range(1, 12):
            summary = download_group(group, workers=8, max_files=None, sums=sums, all_records=records)
            if not summary.get("complete_group"):
                raise RuntimeError(f"group g{group} did not complete")
            summaries.append(summary)
        remaining_parts = list(DATA.rglob("*.part"))
        expected_files = [(record, extension) for record in records for extension in (".hea", ".mat")]
        missing = [f"{record}{extension}" for record, extension in expected_files if not (DATA / f"{record}{extension}").exists()]
        if missing or remaining_parts:
            raise RuntimeError(f"download completeness failure: {len(missing)} missing, {len(remaining_parts)} .part files")
        result = {"stage": "official_chapman_download_complete", "release": "challenge-2021/1.0.3",
                  "records": len(records), "header_files": EXPECTED_RECORDS, "waveform_files": EXPECTED_RECORDS,
                  "groups": summaries, "part_files": 0, "ningbo_accessed": False,
                  "started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(OUT / "DOWNLOAD_COMPLETE.json", result)
        print(canonical_json(result))
    except Exception as error:
        write_json(OUT / "DOWNLOAD_FAILURE.json", {"stage": "official_chapman_download_failure", "error": str(error),
                                                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                                     "ningbo_accessed": False})
        raise
    finally:
        if LOCK.exists():
            LOCK.unlink()


def parse_header(path: Path) -> tuple[list[str], set[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 13:
        raise ValueError("header has fewer than 13 lines")
    fields = lines[0].split()
    if len(fields) < 4 or int(fields[1]) != 12 or float(fields[2]) != 500 or int(fields[3]) != 5000:
        raise ValueError(f"expected 12 leads, 500 Hz, 5000 samples; first line was: {lines[0]}")
    leads = [line.split()[-1] for line in lines[1:13]]
    if leads != LEADS:
        raise ValueError(f"noncanonical lead order: {leads}")
    comments: dict[str, str] = {}
    for line in lines[13:]:
        if line.startswith("#") and ":" in line:
            name, value = line[1:].split(":", 1)
            comments[name.strip()] = value.strip()
    dx = {item.strip() for item in comments.get("Dx", "").split(",") if item.strip()}
    if not dx:
        raise ValueError("missing Dx comment")
    return leads, dx, comments


def audit_data() -> None:
    records = remote_records()
    sums = release_manifest()
    malformed: list[dict[str, str]] = []
    checksum_mismatches: list[str] = []
    duplicate_waveforms: list[dict[str, str]] = []
    seen_waveforms: dict[str, str] = {}
    nonfinite: list[str] = []
    labels: Counter[str] = Counter()
    stratum: Counter[str] = Counter()
    missing_age = 0
    missing_sex = 0
    multiplicity: Counter[int] = Counter()
    total_verified_bytes = 0
    verified_files = 0
    for position, record in enumerate(records, start=1):
        hea, mat = DATA / f"{record}.hea", DATA / f"{record}.mat"
        try:
            _, dx, comments = parse_header(hea)
            if not mat.exists():
                raise ValueError("missing .mat after download")
            for path, extension in ((hea, ".hea"), (mat, ".mat")):
                expected = sums.get(relative_checksum_key(record, extension))
                if expected is None or sha256_file(path) != expected:
                    checksum_mismatches.append(str(path.relative_to(DATA).as_posix()))
                else:
                    total_verified_bytes += path.stat().st_size
                    verified_files += 1
            waveform = loadmat(mat).get("val")
            if waveform is None or waveform.shape != (12, 5000):
                raise ValueError(f"expected val (12,5000), got {None if waveform is None else waveform.shape}")
            if not np.isfinite(waveform).all():
                nonfinite.append(record)
            wave_hash = sha256_file(mat)
            if wave_hash in seen_waveforms:
                duplicate_waveforms.append({"record": record, "matches": seen_waveforms[wave_hash], "sha256": wave_hash})
            else:
                seen_waveforms[wave_hash] = record
            labels.update(dx)
            multiplicity[len(dx)] += 1
            bits = "".join("1" if code in dx else "0" for code in LABELS)
            stratum[bits] += 1
            age = comments.get("Age", "").strip().lower()
            sex = comments.get("Sex", "").strip().lower()
            missing_age += int(age in {"", "nan", "unknown"})
            missing_sex += int(sex in {"", "nan", "unknown"})
        except Exception as error:  # report every malformed object; never repair it
            malformed.append({"record": record, "error": str(error)})
        if position % 1000 == 0 or position == len(records):
            print(f"audited {position}/{len(records)} records", flush=True)
    report = {
        "stage": "strict_header_and_waveform_audit",
        "release": "challenge-2021/1.0.3",
        "source": "Chapman-Shaoxing only",
        "records_expected": EXPECTED_RECORDS,
        "records_seen": len(records),
        "record_id_unique": len(set(records)) == len(records),
        "duplicate_record_ids": [],
        "canonical_12_leads_500hz_5000_samples": len(malformed) == 0,
        "malformed_records": malformed,
        "checksum_mismatches": checksum_mismatches,
        "verified_files": verified_files,
        "total_verified_bytes": total_verified_bytes,
        "nonfinite_records": nonfinite,
        "duplicate_waveforms": duplicate_waveforms,
        "missing_age_count": missing_age,
        "missing_sex_count": missing_sex,
        "label_counts_all_dx": dict(sorted(labels.items())),
        "diagnosis_multiplicity_records": dict(sorted(multiplicity.items())),
        "primary_and_control_strata": dict(sorted(stratum.items())),
        "exclusions": [],
        "decision": "PASS" if not malformed and not checksum_mismatches and verified_files == 2 * EXPECTED_RECORDS and not nonfinite and not duplicate_waveforms else "STOP_DATA_INTEGRITY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "DATA_INGESTION_AUDIT.json", report)
    markdown = ["# Chapman ingestion audit", "", "- Official source: Challenge 2021 v1.0.3 / Chapman-Shaoxing only.",
                f"- Records: {len(records)} expected and observed; unique: {report['record_id_unique']}.",
                f"- Verified files / bytes: {verified_files} / {total_verified_bytes}; checksum mismatches: {len(checksum_mismatches)}.",
                f"- Malformed headers/waveforms: {len(malformed)}; nonfinite: {len(nonfinite)}; exact duplicate waveforms: {len(duplicate_waveforms)}.",
                f"- Missing age: {missing_age}; missing sex: {missing_sex}; exclusions: none.",
                f"- Integrity decision: **{report['decision']}**.", "", "## Three-label strata", ""]
    markdown.extend(f"- `{key}`: {value}" for key, value in sorted(stratum.items()))
    markdown.extend(["", "## Diagnosis multiplicity", ""])
    markdown.extend(f"- `{key}` diagnoses: {value} records" for key, value in sorted(multiplicity.items()))
    (OUT / "DATA_INGESTION_AUDIT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if report["decision"] != "PASS":
        raise RuntimeError("data integrity gate failed; no split or training is allowed")


def audited_records() -> list[tuple[str, set[str]]]:
    audit = json.loads((OUT / "DATA_INGESTION_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("decision") != "PASS":
        raise RuntimeError("strict ingestion audit has not passed")
    output = []
    for record in remote_records():
        _, dx, _ = parse_header(DATA / f"{record}.hea")
        output.append((record, dx))
    return output


def deterministic_split() -> None:
    buckets: dict[str, list[str]] = {format(value, "03b"): [] for value in range(8)}
    lookup: dict[str, set[str]] = {}
    for record, dx in audited_records():
        bits = "".join("1" if label in dx else "0" for label in LABELS)
        buckets[bits].append(record)
        lookup[record] = dx
    membership: dict[str, str] = {}
    allocation: dict[str, dict[str, int]] = {}
    for bits, identifiers in buckets.items():
        ordered = sorted(identifiers, key=lambda value: hashlib.sha256((SPLIT_SEED + value).encode()).hexdigest())
        n = len(ordered)
        n_train, n_val = math.floor(.7 * n), math.floor(.1 * n)
        allocation[bits] = {"train": n_train, "validation": n_val, "test": n - n_train - n_val}
        membership.update({item: "train" for item in ordered[:n_train]})
        membership.update({item: "validation" for item in ordered[n_train:n_train + n_val]})
        membership.update({item: "test" for item in ordered[n_train + n_val:]})
    if len(membership) != EXPECTED_RECORDS or set(membership.values()) != {"train", "validation", "test"}:
        raise RuntimeError("split did not create complete disjoint membership")
    rows = [{"record_id": record, "split": membership[record], "labels": sorted(lookup[record]),
             "bits_twa_sb_stach": "".join("1" if label in lookup[record] else "0" for label in LABELS)}
            for record in sorted(membership)]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=["record_id", "split", "labels", "bits_twa_sb_stach"], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "labels": ";".join(row["labels"])})
    csv_text = csv_buffer.getvalue()
    (OUT / "split_membership.csv").write_text(csv_text, encoding="utf-8", newline="")
    membership_hash = sha256_bytes(csv_text.encode("utf-8"))
    manifest = {"seed": 20260828, "proportions": [0.7, 0.1, 0.2], "labels": LABELS,
                "membership_sha256": membership_hash, "allocation_by_3bit_stratum": allocation,
                "split_counts": dict(Counter(membership.values())), "rows": rows}
    write_json(OUT / "SPLIT_MEMBERSHIP.json", manifest)
    write_json(OUT / "SPLIT_MANIFEST.json", {key: value for key, value in manifest.items() if key != "rows"})
    print(f"split membership SHA-256: {membership_hash}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-group")
    download.add_argument("group", type=int, choices=range(1, 12))
    download.add_argument("--workers", type=int, default=16)
    download.add_argument("--max-files", type=int, default=None)
    sub.add_parser("download-all")
    sub.add_parser("audit")
    sub.add_parser("split")
    args = parser.parse_args()
    if args.command == "download-group":
        download_group(args.group, args.workers, args.max_files)
    elif args.command == "download-all":
        download_all()
    elif args.command == "audit":
        audit_data()
    else:
        deterministic_split()


if __name__ == "__main__":
    main()
