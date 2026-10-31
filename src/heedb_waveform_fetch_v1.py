"""Size-check and optionally download a bounded HEEDB WFDB readability sample."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data" / "heedb_restricted" / "I0006"
SAMPLE = RESTRICTED / "audit_manifests" / "waveform_sample_candidates.csv"
OBJECT_MANIFEST = RESTRICTED / "audit_manifests" / "waveform_object_manifest.csv"
WAVEFORM_ROOT = RESTRICTED / "waveform_sample"
ALIAS = "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
PROFILE = "bdsp-reader"
CAP_BYTES = 2 * 1024**3
TABLE_BYTES = 331_748_784


def aws(*args: str) -> str:
    command = [sys.executable, "-m", "awscli", *args, "--profile", PROFILE]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(message)
    return completed.stdout.strip()


def collect_sizes() -> list[dict[str, str | int]]:
    with SAMPLE.open(newline="", encoding="utf-8") as handle:
        sample = list(csv.DictReader(handle))
    objects: list[dict[str, str | int]] = []
    for row in sample:
        for suffix in (".hea", ".dat"):
            key = f"{row['remote_stem']}{suffix}"
            size_text = aws(
                "s3api",
                "head-object",
                "--bucket",
                ALIAS,
                "--key",
                key,
                "--query",
                "ContentLength",
                "--output",
                "text",
            )
            size = int(size_text)
            relative = Path(row["FileName"] + suffix)
            destination = WAVEFORM_ROOT / relative
            objects.append(
                {
                    "FileName": row["FileName"],
                    "label_pattern_TWA_SB_STach": row["label_pattern_TWA_SB_STach"],
                    "remote_key": key,
                    "remote_bytes": size,
                    "local_relative_path": str(destination.relative_to(RESTRICTED)),
                    "local_verified": 0,
                    "newly_downloaded": 0,
                }
            )
    return objects


def write_manifest(objects: list[dict[str, str | int]]) -> None:
    OBJECT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OBJECT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    objects = collect_sizes()
    waveform_bytes = sum(int(row["remote_bytes"]) for row in objects)
    if TABLE_BYTES + waveform_bytes > CAP_BYTES:
        write_manifest(objects)
        raise RuntimeError(
            f"Cap exceeded: tables={TABLE_BYTES}, waveforms={waveform_bytes}, cap={CAP_BYTES}"
        )

    new_bytes = 0
    if args.download:
        for row in objects:
            destination = RESTRICTED / str(row["local_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = int(row["remote_bytes"])
            if destination.is_file():
                if destination.stat().st_size != expected:
                    raise RuntimeError(f"Existing file size mismatch; refusing overwrite: {destination}")
            else:
                aws(
                    "s3",
                    "cp",
                    f"s3://{ALIAS}/{row['remote_key']}",
                    str(destination),
                    "--only-show-errors",
                )
                new_bytes += expected
                row["newly_downloaded"] = 1
            if destination.stat().st_size != expected:
                raise RuntimeError(f"Downloaded file size mismatch: {destination}")
            row["local_verified"] = 1

    write_manifest(objects)
    print(f"records={len(objects) // 2}")
    print(f"objects={len(objects)}")
    print(f"waveform_remote_bytes={waveform_bytes}")
    print(f"table_plus_waveform_bytes={TABLE_BYTES + waveform_bytes}")
    print(f"new_waveform_bytes={new_bytes}")
    print(f"cap_bytes={CAP_BYTES}")
    print(f"manifest={OBJECT_MANIFEST}")


if __name__ == "__main__":
    main()
