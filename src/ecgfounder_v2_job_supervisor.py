"""Operational retry wrapper for transient Windows metadata-rename denials.

This never changes scientific inputs, model code, cache bindings, or the frozen
protocol. It starts exactly one frozen runner process at a time and retries only
WinError 5 from atomic_json after that process exits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def log(path, item):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("Chapman", "Ningbo"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    progress = OUT / f"{args.dataset.lower()}_progress.log"
    events = OUT / f"{args.dataset.lower()}_supervisor_events.jsonl"
    for attempt in range(1, 101):
        status_path = OUT / "WORKFLOW_STATUS.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        log(events, {"utc": now(), "event": "start", "attempt": attempt, "dataset": args.dataset,
                     "previous_stage": previous.get("stage"),
                     "previous_completed": previous.get("records_completed")})
        err_path = OUT / f"{args.dataset.lower()}_attempt_{attempt:03d}_stderr.log"
        with progress.open("a", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
            child = subprocess.run([sys.executable, "-u", "-m", "src.ecgfounder_baseline_v2",
                                    "run", "--dataset", args.dataset], cwd=ROOT,
                                   stdout=stdout, stderr=stderr, check=False)
        if child.returncode == 0:
            log(events, {"utc": now(), "event": "complete", "attempt": attempt, "dataset": args.dataset})
            return
        error = err_path.read_text(encoding="utf-8", errors="replace")
        transient = "PermissionError: [WinError 5]" in error and "atomic_json" in error
        last = json.loads(status_path.read_text()) if status_path.exists() else {}
        log(events, {"utc": now(), "event": "retry" if transient else "stop", "attempt": attempt,
                     "dataset": args.dataset, "returncode": child.returncode,
                     "stage": last.get("stage"), "error": last.get("error"),
                     "records_completed": last.get("records_completed")})
        if not transient:
            raise RuntimeError(f"non-retryable ECGFounder error; see {err_path}")
        time.sleep(3)
    raise RuntimeError("100 transient metadata-rename retries exhausted")


if __name__ == "__main__":
    main()
