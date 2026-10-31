"""Audit and resume test evaluation after an evaluator-only correction."""
from __future__ import annotations

import json
from pathlib import Path

from src import chest_only_baseline_v1 as run


def main():
    protocol = json.loads(run.PROTO.read_text(encoding="utf-8"))
    protocol_hash = run.sha(run.PROTO)
    original_source_hash = protocol["source_sha256"]["src/chest_only_baseline_v1.py"]
    current_source_hash = run.sha(Path(run.__file__))
    if original_source_hash == current_source_hash:
        raise RuntimeError("expected the evaluation-only source correction to be distinct from the frozen runner")

    units = []
    for dataset in run.DATASETS:
        for seed in run.SEEDS:
            d = run.unit_dir(dataset, seed)
            complete = json.loads((d / "training_complete.json").read_text(encoding="utf-8"))
            if complete.get("protocol_sha256") != protocol_hash or complete.get("checkpoint_sha256") != run.sha(d / "checkpoint.pt"):
                raise RuntimeError(f"training checkpoint/protocol binding mismatch: {dataset}/{seed}")
            units.append({"dataset": dataset, "seed": seed, "best_epoch": complete["best_epoch"],
                          "checkpoint_sha256": complete["checkpoint_sha256"], "status": "TRAINED_AND_FROZEN"})
    if len(units) != 6:
        raise RuntimeError("all six frozen model units are required before test evaluation")
    comparator_audit = [run.comparator_audit(d, m, seed) for d in run.DATASETS for m in run.MODELS[:2] for seed in run.SEEDS]
    if comparator_audit != protocol["comparators"]:
        raise RuntimeError("a comparator artifact differs from its frozen binding")

    amendment = {
        "amendment": "EVALUATION_CORRECTION_001",
        "created_utc": run.utc(),
        "scope": "evaluation plumbing only, after all six training units completed",
        "training_protocol_sha256_unchanged": protocol_hash,
        "training_source_sha256": original_source_hash,
        "corrected_evaluator_source_sha256": current_source_hash,
        "error": "Initial evaluator passed a condition argument to the clean-only predict function; TypeError occurred before any prediction cache was written.",
        "first_attempt_test_label_access": "test labels were loaded before evaluation_freeze.json was written; no test predictions were generated in that attempt",
        "corrected_order": "evaluation_freeze.json is atomically written before test labels or waveforms are loaded",
        "method_changes": False,
        "training_repeated": False,
        "training_units": units,
        "comparator_artifact_audit_units": len(comparator_audit),
    }
    run.atomic_json(run.OUT / "EVALUATION_CORRECTION_AMENDMENT_001.json", amendment)
    incident = {
        "incident": "INITIAL_EVALUATION_ATTEMPT_STOPPED_BEFORE_PREDICTION",
        "created_utc": amendment["created_utc"],
        "exception": "TypeError: predict() takes 5 positional arguments but 6 were given",
        "failing_function": "evaluate_unit",
        "test_labels_loaded_before_evaluation_freeze": True,
        "evaluation_freeze_written": True,
        "prediction_cache_written": False,
        "training_units_complete": 6,
        "training_repeated": False,
        "original_frozen_source_snapshot": "outputs/chest_only_baseline_v1/frozen_source/chest_only_baseline_v1_pre_eval.py",
        "original_frozen_source_sha256": original_source_hash,
    }
    run.atomic_json(run.OUT / "INITIAL_EVALUATION_INCIDENT.json", incident)

    # run_all rechecks real train-fragment invariance, sees six matching completion records,
    # skips every fit, and continues only with the corrected test inference/finalization.
    run.prepare = lambda: None
    run.run_all()

    manifest_path = run.OUT / "REPORTING_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("EVALUATION_CORRECTION_AMENDMENT_001.json", "INITIAL_EVALUATION_INCIDENT.json",
                 "frozen_source/chest_only_baseline_v1_pre_eval.py"):
        path = run.OUT / name
        manifest.setdefault("artifacts", {})[name] = {"sha256": run.sha(path), "bytes": path.stat().st_size}
    manifest["corrected_evaluator_source_sha256"] = current_source_hash
    run.atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
