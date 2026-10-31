"""Create the immutable HEEDB external-validation protocol and hash sidecar."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_external_v1"
TARGET = ROOT / "configs" / "FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json"
SEEDS = (104729, 130363, 155921)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ranked_masks() -> list[list[int]]:
    remaining = set(itertools.combinations(range(12), 5))
    counts = np.zeros(12, dtype=np.int64)
    chosen: list[tuple[int, ...]] = []

    def digest(mask: tuple[int, ...]) -> str:
        value = "heedb_external_v1|mask-panel-v1|" + ",".join(map(str, mask))
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    for step in range(24):
        target = (step + 1) * 5 / 12
        mask = min(
            remaining,
            key=lambda candidate: (
                sum(
                    (counts[index] + int(index in candidate) - target) ** 2
                    for index in range(12)
                ),
                digest(candidate),
            ),
        )
        chosen.append(mask)
        remaining.remove(mask)
        counts[list(mask)] += 1
    if counts.tolist() != [10] * 12:
        raise RuntimeError("Deterministic mask panel is not lead-balanced")
    return [list(mask) for mask in chosen]


def model_units(base: Path) -> dict[str, object]:
    result = {}
    for seed in SEEDS:
        unit = base / f"seed_{seed}"
        with np.load(unit / "validation.npz", allow_pickle=False) as loaded:
            thresholds = loaded["thresholds"].astype(float).tolist()
        result[str(seed)] = {
            "checkpoint": str((unit / "checkpoint.pt").relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256(unit / "checkpoint.pt"),
            "preprocessing": str((unit / "preprocessing.npz").relative_to(ROOT)).replace("\\", "/"),
            "preprocessing_sha256": sha256(unit / "preprocessing.npz"),
            "validation_threshold_source": str((unit / "validation.npz").relative_to(ROOT)).replace("\\", "/"),
            "validation_threshold_source_sha256": sha256(unit / "validation.npz"),
            "thresholds": thresholds,
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sidecar = OUT / "FROZEN_PROTOCOL_SHA256.txt"
    if TARGET.exists() or sidecar.exists():
        raise RuntimeError("Frozen protocol or sidecar already exists; refusing overwrite")

    masks = ranked_masks()
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    value = {
        "protocol": "HEEDB external transportability v1",
        "frozen_date": "2026-09-24",
        "scope": "Previously frozen external test on HEEDB I0006 (Emory); not an exact SNOMED-label replication",
        "prohibitions": [
            "No HEEDB training, fine-tuning, threshold selection, or calibration",
            "No mixing physician-overread, software-generated, or diagnoses_v24 label fields",
            "No outcome-driven changes after evaluation begins",
            "No root AWS profile and no S3 permission broadening",
        ],
        "data": {
            "repository_alias": "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias",
            "aws_profile": "bdsp-reader",
            "required_identity": {
                "account": "552883519608",
                "arn": "arn:aws:iam::552883519608:user/bdsp-heedb-reader",
                "root_forbidden": True,
            },
            "dataset": "ECG/I0006 (Emory)",
            "reused_tables": [
                "ECG/I0006/12SL_diagnoses/diagnoses_dictionary.csv",
                "ECG/I0006/12SL_diagnoses/diagnoses_acquisition.csv",
                "ECG/I0006/12SL_diagnoses/diagnoses_v24.csv",
                "ECG/I0006/metadata/metadata.csv",
            ],
            "waveform_transfer_cap_bytes": 2 * 1024**3,
            "download_only_selected_pairs": True,
            "resumable_partial_suffix": ".partial",
            "verify_remote_and_local_exact_bytes": True,
        },
        "labels": {
            "source": "diagnoses_acquisition.csv codes_physician only",
            "primary_core": {
                "TWA": [1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181],
                "SB": [21, 24],
                "STach": [23],
            },
            "sensitivity_strict": {"TWA": [1140], "SB": [21]},
            "software_codes_role": "audit-only; never merged with physician labels",
            "v24_codes_role": "audit-only; never merged with physician labels",
            "ambiguity": {
                "TWA_excluded": {
                    "codes": [1141, 1142, 1182, 1200, 1201, 1203, 1207, 1208] + list(range(1210, 1225)),
                    "reason": "mixed ST-T, suggestive/lead-fragment, or serial-change language rather than the frozen core construct",
                },
                "SB_excluded": {"codes": [265], "reason": "probable sinus bradycardia with verification language"},
                "cross_dataset_caveat": "These 12SL physician-overread constructs are clinically related proxies, not SNOMED-equivalent Chapman/Ningbo labels.",
            },
        },
        "sampling": {
            "eligibility": [
                "nonempty BDSPPatientID",
                "nonempty codes_physician",
                "not one of the 23 patients used by heedb_readiness_v1 waveform smoke",
            ],
            "selection_seed": "heedb_external_v1|2026-09-24|label-blind|one-ecg-per-patient",
            "record_within_patient_rank": "ascending SHA256(seed|record|BDSPPatientID|FileName), then FileName",
            "patient_rank": "ascending SHA256(seed|patient|BDSPPatientID), then BDSPPatientID",
            "sample": "first 10,000 eligible patients after selecting one record per patient",
            "label_blind": True,
            "unreadable_replacement": "For each failed evaluation slot, take the next unused patient in the frozen patient-rank order; use that patient's already frozen one-record choice. Repeat without consulting labels or predictions.",
            "patient_manifest_location": "data/heedb_restricted/I0006/external_v1/ (git-ignored)",
        },
        "models": {
            "label_order": ["TWA", "SB", "STach"],
            "architecture": "src.chapman_phase1.ECGResNetSmall",
            "baseline": model_units(ROOT / "outputs" / "ningbo_confirmation_v2"),
            "physics_aware": model_units(ROOT / "outputs" / "mitigation_m2_ningbo_v1"),
        },
        "preprocessing": {
            "input_lead_order": lead_names,
            "HEEDB_read": "WFDB physical samples; require all 12 units mV and canonical lead order",
            "amplitude_conversion": "multiply mV by 1000 to match the Ningbo Challenge val/gain representation in microvolts",
            "sampling": "500 Hz and 5000 samples unchanged; 250 Hz and 2500 samples resampled to 500 Hz/5000 with scipy.signal.resample_poly(up=2, down=1, default Kaiser window)",
            "normalization": "per-lead (x - frozen Ningbo training mean) / frozen Ningbo training std",
            "fault_order": "apply electrical reversal in raw microvolts before normalization",
            "mask_order": "replace each selected displayed lead by its frozen Ningbo training mean before normalization (equivalently normalized zero)",
        },
        "reversal": {
            "name": "RA_LL_reversal",
            "independent_gauge": "RA=0, LA=I, LL=II; swap RA and LL; Wilson central terminal is invariant",
            "equations": {
                "I_prime": "I-II",
                "II_prime": "-II",
                "III_prime": "-I",
                "aVR_prime": "(II-I)/2",
                "aVL_prime": "I-II/2",
                "aVF_prime": "I/2-II",
                "V1_to_V6_prime": "unchanged",
            },
        },
        "metrics": {
            "AUPRC": "sklearn.metrics.average_precision_score",
            "sensitivity": "fraction of positives with probability >= the arm/seed's frozen Ningbo validation threshold",
            "aggregation": "compute each metric per matched seed, then arithmetic mean of the three seed-level metrics or contrasts; never average prediction vectors across seeds",
            "clean_performance_change": "physics-aware minus baseline under clean ECG",
            "within_arm_fault_change": "clean minus RA-LL metric",
            "uncertainty": "1000 synchronized patient-level resamples with replacement, seed 20260924, percentile 95% intervals",
        },
        "primary_comparisons": [
            "Core TWA RA-LL AUPRC: physics-aware minus baseline",
            "Core TWA RA-LL sensitivity at frozen thresholds: physics-aware minus baseline",
            "Core TWA clean AUPRC and sensitivity: physics-aware minus baseline",
            "Within-arm core TWA clean-minus-RA-LL changes",
        ],
        "separate_results": [
            "Core SB and STach clean and RA-LL results",
            "Strict TWA and strict SB clean and RA-LL sensitivity analyses",
        ],
        "mask_panel": {
            "all_possible_five_displayed_lead_masks": 792,
            "panel_name": "deterministic 24-mask subset (not the all-mask mean)",
            "selection": "24-step deterministic greedy balance: at each step choose the remaining mask minimizing squared deviation from uniform cumulative lead counts; break ties by SHA256(heedb_external_v1|mask-panel-v1|comma-separated zero-based indices)",
            "balance": "each of the 12 displayed leads appears in exactly 10 of the 24 masks",
            "lead_order": lead_names,
            "masks_zero_based": masks,
            "masks_named": [[lead_names[index] for index in mask] for mask in masks],
            "summary": "mean of per-mask AUPRC values within seed, then mean across seeds",
            "bootstrap_scope": "paired patient bootstrap interval for the core-TWA physics-aware-minus-baseline subset-mean contrast; other mask outcomes are point estimates",
            "compute_estimate": {
                "historical_evidence": "M2 all-mask evaluation: 2382 evaluations (3 seeds x 794 conditions) on 6981 records in about 2290 seconds",
                "all_792_masks_two_arms_10000_patients_seconds": 6560,
                "all_792_masks_two_arms_10000_patients_minutes": 109.3,
                "frozen_24_mask_subset_two_arms_10000_patients_seconds": 215,
                "frozen_24_mask_subset_two_arms_10000_patients_minutes": 3.6,
                "scope": "forward-pass scaling estimate before HEEDB inference; excludes waveform preparation and bootstrap",
            },
        },
        "gates": {
            "engineering_smoke": "all six frozen models; all frozen conditions; the 23 already-downloaded and final-evaluation-excluded readiness ECGs; require readable/resampled signals and finite 3-output probabilities",
            "evaluation_lock": "exclusive lock prevents duplicate jobs",
            "no_interim_adaptation": "after EVALUATION_FREEZE.json is written, do not inspect outcomes to alter sample, mappings, models, thresholds, preprocessing, reversal, or masks",
        },
        "source_hashes": {
            "evaluation_code": sha256(ROOT / "src" / "heedb_external_v1.py"),
            "freeze_code": sha256(Path(__file__)),
            "model_architecture_source": sha256(ROOT / "src" / "chapman_phase1.py"),
            "physics_source": sha256(ROOT / "src" / "ecg_physics.py"),
            "readiness_report": sha256(ROOT / "outputs" / "heedb_readiness_v1" / "HEEDB_READINESS_REPORT.md"),
            "baseline_protocol": sha256(ROOT / "configs" / "FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json"),
            "m2_protocol": sha256(ROOT / "configs" / "FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json"),
        },
    }
    TARGET.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(TARGET)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    (OUT / "IMMUTABLE_PROTOCOL.md").write_text(
        "# Frozen HEEDB external transportability protocol v1\n\n"
        f"Protocol: `configs/FROZEN_HEEDB_EXTERNAL_PROTOCOL_v1.json`  \nSHA-256: `{digest}`\n\n"
        "This protocol was frozen before any new HEEDB waveform download or inference. It compares the three matched clean-trained and physics-aware Ningbo ECGResNet seeds on HEEDB I0006 under clean input, electrically exact RA-LL reversal, and a deterministic 24-of-792 mask subset. Physician-overread codes are the only labels; HEEDB labels are proxy constructs and are not represented as exact SNOMED replication.\n",
        encoding="utf-8",
    )
    print(digest)


if __name__ == "__main__":
    main()
