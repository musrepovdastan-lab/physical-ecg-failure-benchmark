"""Foreground, headers-only metadata audit for Challenge 2021 Chapman/Ningbo.

Never requests .mat waveform files. It is intentionally opt-in: Phase 0 must run it
only after authorising the ~45k small official WFDB header requests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

ROOT = "https://physionet.org/files/challenge-2021/1.0.3/training"
SOURCES = {"chapman": ("chapman_shaoxing", 11), "ningbo": ("ningbo", 35)}


def get(url: str) -> bytes:
    if ".mat" in url:
        raise RuntimeError("waveform download forbidden")
    with urlopen(url, timeout=60) as response:
        return response.read()


def header_urls(source: str) -> list[str]:
    folder, groups = SOURCES[source]
    records: list[str] = []
    for group in range(1, groups + 1):
        base = f"{ROOT}/{folder}/g{group}"
        for item in get(f"{base}/RECORDS").decode("utf-8").splitlines():
            record = item.strip()
            if record:
                records.append(f"{base}/{record}.hea")
    return records


def parse_header(url: str) -> dict:
    text = get(url).decode("utf-8")
    lines = text.splitlines()
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("malformed first header line")
    n_sig, fs, n_samples = int(first[1]), float(first[2]), int(first[3])
    lead_order = tuple(line.split()[-1] for line in lines[1 : 1 + n_sig])
    fields = {}
    for line in lines[1 + n_sig :]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            fields[key.strip()] = value.strip()
    return {
        "record_id": first[0], "n_signals": n_sig, "sampling_hz": fs,
        "samples": n_samples, "duration_seconds": n_samples / fs,
        "lead_order": lead_order, "age": fields.get("Age"), "sex": fields.get("Sex"),
        "diagnoses": tuple(x.strip() for x in fields.get("Dx", "").split(",") if x.strip()),
        "header_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def audit(source: str, workers: int) -> dict:
    urls = header_urls(source)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_header, url): url for url in urls}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # retain each failure rather than silently dropping it
                failures.append({"url": futures[future], "error": repr(exc)})
    label_counts = Counter(label for row in rows for label in row["diagnoses"])
    return {"source": source, "attempted_headers": len(urls), "valid_headers": len(rows),
            "malformed_or_unreadable_headers": failures, "label_counts": dict(sorted(label_counts.items())),
            "age_missing": sum(not row["age"] or row["age"] == "NaN" for row in rows),
            "sex_missing": sum(not row["sex"] or row["sex"] == "Unknown" for row in rows),
            "multiple_diagnoses": sum(len(row["diagnoses"]) > 1 for row in rows),
            "lead_orders": Counter(row["lead_order"] for row in rows),
            "sampling_hz": Counter(row["sampling_hz"] for row in rows),
            "durations_seconds": Counter(row["duration_seconds"] for row in rows),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        result = audit(source, args.workers)
        rows = result.pop("rows")
        (args.output / f"{source}_header_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (args.output / f"{source}_records.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["record_id", "age", "sex", "diagnoses", "n_signals", "sampling_hz", "samples", "duration_seconds", "lead_order", "header_sha256"])
            writer.writeheader()
            for row in sorted(rows, key=lambda item: item["record_id"]):
                writer.writerow({**row, "diagnoses": "|".join(row["diagnoses"]), "lead_order": "|".join(row["lead_order"])})


if __name__ == "__main__":
    main()
