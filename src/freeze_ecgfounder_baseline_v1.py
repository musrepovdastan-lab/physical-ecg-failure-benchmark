"""Create the one-time ECGFounder protocol before new held-out evaluation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v1"
DEST = ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v1.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise RuntimeError("protocol already exists; v1 is immutable")
    audit_path = OUT / "PRE_RUN_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for relative, digest in audit["bindings"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"pre-run audit binding changed: {relative}")
    code_commit = json.loads((OUT / "official/github_commit.json").read_text())["sha"]
    if code_commit != "04edac702b61c91face519774ddcc0cd712fef23":
        raise RuntimeError("official code commit changed")
    cfg = {
        "version": "ecgfounder_baseline_v1",
        "status": "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory published-pretrained-model baseline; previously examined held-out sets",
        "scientific_question": "Does exact RA-LL cable reversal lower ECGFounder diagnostic performance on the same Chapman and Ningbo held-out records as the saved ECGResNet-small predictions?",
        "source": {
            "code_repo": "https://github.com/PKUDigitalHealth/ECGFounder",
            "code_commit": code_commit,
            "checkpoint_repo": "https://huggingface.co/PKUDigitalHealth/ECGFounder",
            "checkpoint_revision": "47a965e",
            "checkpoint_file": "12_lead_ECGFounder.pth",
            "checkpoint_sha256": audit["bindings"]["outputs/ecgfounder_baseline_v1/official/12_lead_ECGFounder.pth"],
            "model": "official Net1D, 12 channels, 150 classes, use_bn=False, use_do=False, return_features=True, strict state-dict load",
            "feature": "1024-dimensional global temporal average before official dense layer",
            "backbone_frozen": True,
        },
        "data": {
            "datasets": ["Chapman", "Ningbo"],
            "same_record_ids_and_existing_train_validation_test_splits": True,
            "labels_in_order": ["TWA", "SB", "STach"],
            "snomed_codes_in_order": ["164934002", "426177001", "427084000"],
            "patient_unit": "existing canonical one-record-per-patient split/cluster assumption; no new split or exclusion",
            "training_sizes": {"Chapman": 7163, "Ningbo": 24430},
            "validation_sizes": {"Chapman": 1021, "Ningbo": 3488},
            "test_sizes": {"Chapman": 2050, "Ningbo": 6981},
        },
        "label_mapping": {
            "TWA": "no exact published output; NONSPECIFIC T WAVE ABNORMALITY is excluded from exact-label and zero-shot comparison",
            "SB": "official tasks.txt SINUS BRADYCARDIA, index 4 (name-level match)",
            "STach": "official tasks.txt SINUS TACHYCARDIA, index 6 (name-level match)",
            "zero_shot_scope": "SB and STach only, reported separately with validation-fixed thresholds",
        },
        "preprocessing": {
            "input": "WFDB physical values in mV, 12 x 5000, 500 Hz, 10 seconds",
            "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            "pre_fault": "RA-LL terminal permutation on raw physical I, II, V1..V6; derive 12 displayed leads; chest leads invariant",
            "order": ["50 Hz iirnotch Q=30 with filtfilt per lead", "4th-order Butterworth 0.67-40 Hz bandpass with filtfilt per lead", "subtract 201-sample median-filter baseline per lead", "global z-score over all 12 x 5000 values using population std and epsilon 1e-8", "cast float32"],
            "provenance": "official util.py::filter_bandpass and dataset.py::z_score_normalization; 500 Hz data need no resampling",
            "nonfinite_or_schema_mismatch": "stop",
        },
        "frozen_head": {
            "family": "three independent L2-regularized linear logistic heads on the 1024-D frozen features, trained clean only",
            "feature_standardization": "per-dimension mean and population std from each dataset's existing training split, floor std at 1e-6",
            "solver": "sklearn LogisticRegression liblinear, max_iter=300, tol=1e-4, class_weight=None, random_state=20260920",
            "C_grid": [0.01, 0.1, 1.0],
            "selection": "highest clean validation AUPRC for each label; tie selects smaller C",
            "threshold": "90th percentile of clean validation negative probabilities per label, fixed for both test conditions",
            "no_backbone_updates": True,
        },
        "evaluation": {
            "conditions": ["clean", "RA_LL_reversal"],
            "metrics": ["AUPRC", "AUROC", "sensitivity_at_validation_fixed_threshold"],
            "primary": "TWA clean minus RA-LL AUPRC within each dataset",
            "secondary": ["SB", "STach"],
            "comparison": "paired with each of the three existing clean-trained ECGResNet-small seeds on identical sorted held-out record IDs; report per-seed and mean ResNet metrics, and clean-to-reversal drops",
            "zero_shot": "published 150-class logits for name-matched SB and STach separately; no TWA zero-shot score",
            "bootstrap": "1000 synchronized resamples of canonical patient/record units with replacement, seed 20260921, paired across conditions and model predictions; percentile 95% intervals for metrics, drops, and model drop differences",
            "interpretation": "exploratory; not fresh blinded confirmation, architecture or SOTA superiority claim",
        },
        "runtime": {
            "smoke_first": True, "batch_size": 16, "inference": "CUDA float16 autocast, frozen eval model",
            "cache": "resumable feature and logit arrays bound to protocol/checkpoint/ordered IDs, completed hashes required before reuse",
            "single_job_lock": True,
            "progress": "WORKFLOW_STATUS.json plus append-only run logs",
        },
        "excluded": ["HEEDB download", "ECGFounder backbone retraining", "all 792 masks", "M3 resume or overwrite", "new robustness method"],
        "runner_sha256": sha(ROOT / "src/ecgfounder_baseline_v1.py"),
        "tests_sha256": sha(ROOT / "tests/test_ecgfounder_baseline_v1.py"),
        "pre_run_audit_sha256": sha(audit_path),
        "file_bindings": audit["bindings"],
    }
    DEST.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "FROZEN_PROTOCOL_HASH.json").write_text(json.dumps({
        "protocol_path": str(DEST.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha(DEST), "frozen_utc": cfg["frozen_utc"],
        "test_evaluation_started": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(sha(DEST))


if __name__ == "__main__":
    main()
