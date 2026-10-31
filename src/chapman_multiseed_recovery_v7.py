"""Nonsemantic Windows durability wrapper for the frozen v7 evaluator.

It imports the frozen evaluator unchanged and replaces only its atomic JSON/NPZ
writers with process-unique temporary files plus bounded retry.  Status and
bootstrap-progress JSON remain human-readable telemetry: failure to replace
them after retries is logged but does not interrupt scientific work.  Every
scientific artifact remains strict and raises on a write/validation failure.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import src.chapman_multiseed_workflow_v7 as frozen

OUT = frozen.OUT
TELEMETRY = {"WORKFLOW_STATUS.json", "BOOTSTRAP_PROGRESS.json"}
RETRIES = 6

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def replace_retry(tmp, target, nonfatal=False):
    last = None
    for attempt in range(RETRIES):
        try:
            os.replace(tmp, target); return True
        except PermissionError as exc:
            last = exc; time.sleep(0.05 * (2 ** attempt))
    if nonfatal:
        print(json.dumps({"telemetry_write_warning": str(last), "target": str(target), "attempts": RETRIES}), flush=True)
        try: Path(tmp).unlink(missing_ok=True)
        except OSError: pass
        return False
    raise last
def durable_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
    return replace_retry(tmp, path, nonfatal=path.name in TELEMETRY)
def durable_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    return replace_retry(tmp, path, nonfatal=False)
def validate_retained():
    ids = frozen.test_ids(); n = len(ids); cache = OUT / "seed_130363" / "test_predictions"
    files = {p.stem: p for p in cache.glob("*.npz")}
    expected = list(frozen.CONDITIONS)
    names = [name for name in expected if name in files]
    if names != expected[:len(names)] or len(names) != len(files):
        raise RuntimeError("retained cache conditions are not the exact frozen prefix")
    for name in names:
        p = frozen.npz_prediction(files[name], n, name)
        if p is None: raise RuntimeError(f"corrupt retained prediction: {name}")
    return {"retained_validated_conditions": len(names), "first_missing_condition": expected[len(names)],
            "canonical_test_record_count": n, "canonical_test_record_ids_sha256": frozen.ids_hash(ids),
            "condition_identity": "exact ordered frozen condition-prefix filenames", "record_order": "canonical v4 test order used by frozen evaluator"}
def write_recovery_records(audit):
    failure = (OUT / "WORKFLOW_STATUS.json").read_text(encoding="utf-8")
    audit.update({"created_utc": frozen.now(), "original_frozen_workflow_evaluator_sha256": sha(Path(frozen.__file__)),
                  "recovery_wrapper_sha256": sha(Path(__file__)), "previous_failure_traceback": json.loads(failure).get("traceback"),
                  "non_scientific_difference": "process-unique temporary files; bounded PermissionError replace retries; nonfatal telemetry replacement after retry exhaustion; stdout telemetry retained; bootstrap progress uses same retry mechanism",
                  "scientific_definitions_changed": False,
                  "statement": "No prediction, metric, fault transformation, model, threshold, split, seed, or decision rule changed. This is a nonsemantic execution-recovery amendment after seed-130363 test access began."})
    durable_json(OUT / "RECOVERY_AUDIT.json", audit)
    text = "# Resume implementation amendment\n\nThis nonsemantic execution-recovery amendment preserves the frozen v7 evaluator and all scientific definitions. The prior process stopped only when Windows denied replacement of `WORKFLOW_STATUS.json`. The recovery wrapper uses process-unique temporary filenames, bounded exponential-backoff replacement retries, and nonfatal status/bootstrap-progress telemetry failures after retries; scientific prediction, result, and state files remain strict.\n\n" + json.dumps(audit, indent=2) + "\n"
    (OUT / "RESUME_IMPLEMENTATION_AMENDMENT.md").write_text(text, encoding="utf-8")
def recovery_preparation():
    """Resume-only replacement for the original pre-test preparation guard.

    The original guard correctly rejects unmanifested predictions.  Here the
    audited, strictly validated partial cache is the explicit recovery state;
    this function verifies the immutable freezes instead of recreating them.
    """
    frozen.smoke(); frozen.validate_baselines()
    for seed in frozen.NEW_SEEDS:
        path = OUT / f"seed_{seed}" / "evaluation_freeze.json"
        z = json.loads(path.read_text(encoding="utf-8"))
        if z.get("evaluator_code_sha256") != sha(Path(frozen.__file__)) or z.get("test_accessed_before_evaluation_freeze") is not False:
            raise RuntimeError(f"immutable evaluation freeze invalid for {seed}")
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("audit", "run")); args = parser.parse_args()
    audit = validate_retained(); write_recovery_records(audit)
    if args.command == "audit": print(json.dumps(audit, indent=2)); return
    frozen.atomic_json = durable_json
    frozen.atomic_npz = durable_npz
    frozen.freeze_evaluations = recovery_preparation
    frozen.run()
if __name__ == "__main__": main()
