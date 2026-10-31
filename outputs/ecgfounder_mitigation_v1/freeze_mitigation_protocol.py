from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import ecgfounder_baseline_v2 as founder  # noqa: E402

SEEDS = (104729, 130363, 155921)
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXACT = (0, 1, 2, 3, 5)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def hash_obj(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    if (OUT / "FROZEN_PROTOCOL.json").exists():
        raise RuntimeError("refusing to overwrite an existing frozen protocol")
    pilot = json.loads((OUT / "ENGINEERING_PILOT.json").read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS" or not pilot["resource_estimate"]["within_limit"]:
        raise RuntimeError("engineering pilot did not pass within the six-hour GPU limit")
    expected_checkpoint = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"
    if sha(founder.CHECKPOINT) != expected_checkpoint:
        raise RuntimeError("frozen Founder checkpoint hash changed")

    rng = np.random.default_rng(20261018)
    all_masks = list(itertools.combinations(range(12), 5))
    remaining = [m for m in all_masks if m != EXACT]
    selected_positions = rng.choice(len(remaining), size=31, replace=False)
    mask_panel = [EXACT] + [remaining[int(i)] for i in selected_positions]
    if len(mask_panel) != 32 or len(set(mask_panel)) != 32 or EXACT not in mask_panel:
        raise RuntimeError("invalid frozen 32-mask panel")
    mask_rows = []
    for panel_index, indices in enumerate(mask_panel):
        mask_rows.append({"panel_index": panel_index, "name": "mask5_" + "_".join(map(str, indices)),
                          "indices_zeroed": list(indices), "leads_zeroed": [LEADS[i] for i in indices],
                          "retained_indices": [i for i in range(12) if i not in indices],
                          "retained_leads": [LEADS[i] for i in range(12) if i not in indices],
                          "is_exact_affected_lead_mask": indices == EXACT})
    (OUT / "MASK_PANEL_32.csv").write_text(
        "panel_index,name,indices_zeroed,leads_zeroed,retained_indices,retained_leads,is_exact_affected_lead_mask\n" +
        "".join(f'{r["panel_index"]},{r["name"]},"{r["indices_zeroed"]}","{r["leads_zeroed"]}","{r["retained_indices"]}","{r["retained_leads"]}",{r["is_exact_affected_lead_mask"]}\n' for r in mask_rows),
        encoding="utf-8")

    source_paths = [
        "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json",
        "src/ecgfounder_baseline_v2.py", "src/ecg_physics.py",
        "outputs/heldout_limb_cycles_v1/cycle_adapter.py",
        "src/mask_semantics_recovery_v2.py",
        "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json",
        "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json",
        "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json",
        "outputs/ecgfounder_baseline_v2/MISSING_PATH_AUDIT.json",
        "src/ecgfounder_preflight_v2.py",
        "outputs/ecgfounder_baseline_v2/preflight_final_stdout.log",
        "outputs/ecgfounder_baseline_v2/ningbo_progress.log",
        "outputs/ecgfounder_baseline_v2/PREFREEZE_SMOKE.json",
        "outputs/ecgfounder_baseline_v2/FINAL_BASELINE_REPORT.md",
        "outputs/architecture_replication_m3_recovery_v2/RECOVERY_PROTOCOL.json",
        "outputs/architecture_replication_m3_recovery_v2/INPUT_AUDIT.json",
        "outputs/architecture_replication_m3_recovery_v2/conclusions_m3.json",
        "outputs/architecture_replication_m3_recovery_v2/FINAL_M3_DECISION.md",
        "outputs/heldout_limb_cycles_v1/ENGINEERING_CHECKS.json",
        "outputs/heldout_limb_cycles_v1/FROZEN_PROTOCOL.json",
        "outputs/heldout_limb_cycles_v1/REVIEW_NOTES_20261018.md",
        "outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy",
        "outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
        "outputs/ecgfounder_baseline_v2/features/chapman_train_clean_features.npy",
        "outputs/ecgfounder_baseline_v2/features/chapman_train_clean_meta.json",
        "outputs/ecgfounder_baseline_v2/features/chapman_validation_clean_features.npy",
        "outputs/ecgfounder_baseline_v2/features/chapman_validation_clean_meta.json",
        "outputs/ecgfounder_baseline_v2/features/ningbo_train_clean_features.npy",
        "outputs/ecgfounder_baseline_v2/features/ningbo_train_clean_meta.json",
        "outputs/ecgfounder_baseline_v2/features/ningbo_validation_clean_features.npy",
        "outputs/ecgfounder_baseline_v2/features/ningbo_validation_clean_meta.json",
        "outputs/ecgfounder_baseline_v1/official/12_lead_ECGFounder.pth",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/dataset.py",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/net1d.py",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/util.py",
        "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/tasks.txt",
        "outputs/ecgfounder_mitigation_v1/engineering_pilot.py",
        "outputs/ecgfounder_mitigation_v1/freeze_mitigation_protocol.py",
        "outputs/ecgfounder_mitigation_v1/run_mitigation.py",
        "outputs/ecgfounder_mitigation_v1/ENGINEERING_PILOT.json",
        "outputs/ecgfounder_mitigation_v1/IMPLEMENTATION_PREFLIGHT.json",
        "outputs/ecgfounder_mitigation_v1/MASK_PANEL_32.csv",
    ]
    input_hashes = {p: sha(ROOT / p) for p in source_paths}

    split_id_hashes = {}
    split_counts = {}
    label_hashes = {}
    raw_input_hashes = {}
    canonical_hash_binding_mismatches = []
    # Hash every waveform/header used for training views or test evaluation before
    # any new head test outcome exists. Validation comes only from verified clean caches.
    for dataset in ("Chapman", "Ningbo"):
        for partition in ("train", "validation", "test"):
            records = founder.split(dataset, partition)
            ids = [r["record_id"] for r in records]
            split_id_hashes[f"{dataset}/{partition}"] = hash_ids(ids)
            split_counts[f"{dataset}/{partition}"] = len(ids)
            y = founder.label_matrix(dataset, partition).astype(np.uint8)
            label_hashes[f"{dataset}/{partition}"] = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
            if partition not in ("train", "test"):
                continue
            for record in records:
                rid = record["record_id"]
                if dataset == "Ningbo":
                    canonical = founder.ningbo_canonical()[rid]
                    hea = founder.canonical_path(canonical["header_path"], ".hea")
                    mat = founder.canonical_path(canonical["waveform_path"], ".mat")
                else:
                    stem = founder.record_path(dataset, rid)
                    hea, mat = stem.with_suffix(".hea"), stem.with_suffix(".mat")
                for path in (hea, mat):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                mat_sha = sha(mat)
                raw_input_hashes[f"{dataset}/{partition}/{rid}"] = {
                    "header_sha256": sha(hea), "waveform_sha256": mat_sha,
                    "canonical_signal_sha256": record.get("waveform_sha256") if dataset == "Ningbo" else None,
                    "header_bytes": hea.stat().st_size, "waveform_bytes": mat.stat().st_size}
                if dataset == "Ningbo" and record.get("waveform_sha256") != founder.ningbo_canonical()[rid].get("waveform_sha256"):
                    canonical_hash_binding_mismatches.append(f"{dataset}/{partition}/{rid}")
    if canonical_hash_binding_mismatches:
        raise RuntimeError(f"Ningbo split/canonical signal hash bindings differ: {len(canonical_hash_binding_mismatches)}")

    # Existing clean test caches may be reused only after same-context feature replay.
    # Freeze their exact artifact hashes here; the post-freeze replay is a gate.
    reuse_candidates = {}
    for dataset in ("Chapman", "Ningbo"):
        ids = [r["record_id"] for r in founder.split(dataset, "test")]
        for condition in ("clean", "RA_LL_reversal"):
            fp, lp, mp = founder.cache_paths(dataset, "test", condition)
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if meta.get("record_ids_sha256") != hash_ids(ids):
                raise RuntimeError(f"test cache order mismatch before protocol freeze: {dataset}/{condition}")
            reuse_candidates[f"{dataset}/{condition}"] = {
                "features_sha256": sha(fp), "logits_sha256": sha(lp), "metadata_sha256": sha(mp),
                "metadata_record_ids_sha256": meta["record_ids_sha256"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "protocol_sha256": meta["protocol_sha256"],
                "reuse_gate": "same-order, same-batch-size 16, fixed subset feature replay atol=1e-6 after this protocol freezes"}

    protocol = {
        "package": "ecgfounder_mitigation_v1",
        "status": "FROZEN_BEFORE_NEW_TEST_OUTCOMES",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Transfer of physical ECG augmentation to the head trained on one fixed, frozen ECGFounder encoder; this is not architecture search or full Founder fine-tuning.",
        "temporal_and_access_note": "Existing project test sets and prior clean-head outcomes have been examined in earlier packages. This package's new head test predictions/metrics have not been computed. Test waveform/header hashes and labels/splits are bound here before those new outcomes.",
        "engineering_gate": {"pilot_status": pilot["status"], "pilot_path": "outputs/ecgfounder_mitigation_v1/ENGINEERING_PILOT.json",
                             "pilot_model_only_estimate_gpu_hours": pilot["resource_estimate"]["estimated_gpu_hours"],
                             "six_gpu_hour_limit": 6.0,
                             "measured_historical_end_to_end_rate_records_per_second": 41880 / (27 * 60 + 52),
                             "historical_rate_source": "outputs/ecgfounder_baseline_v2/ningbo_progress.log: 41,880 clean/reversal train+validation+test feature rows over 27m52s; this includes waveform loading, filtering, batch-16 frozen inference and cache writes.",
                             "conservative_estimated_gpu_hours": ((3 * (7163 + 24430) + 38 * (2050 + 6981)) / (41880 / (27 * 60 + 52)) * 1.2) / 3600,
                             "resource_estimate_includes": "three transformed train views per record; up to all 38 test conditions per record if the old clean/RA_LL test-cache replay fails; 1.2x allowance over measured prior end-to-end extraction throughput. Estimated worst case remains under six GPU-hours."},
        "source": {"official_repository": "https://github.com/PKUDigitalHealth/ECGFounder",
                   "pinned_code_commit": "04edac702b61c91face519774ddcc0cd712fef23",
                   "model_card": "https://huggingface.co/PKUDigitalHealth/ECGFounder",
                   "checkpoint_path": "outputs/ecgfounder_baseline_v1/official/12_lead_ECGFounder.pth",
                   "checkpoint_sha256": "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997",
                   "encoder": "official Net1D, 12 inputs, 150 published outputs, return_features=True; 1024-D global temporal average before official dense layer; eval() and all parameters requires_grad=False.",
                   "pretraining_documentation": "The authors report supervised training on 10,771,552 expert-annotated ECGs from Harvard-Emory ECG Database (HEEDB). Direct patient/record overlap with Chapman or Ningbo is not disclosed or independently ruled out; this is a provenance limitation, not a reason to indefinitely defer this exploratory test.",
                   "data_overlap": {"Chapman": "No record-level crosswalk or explicit overlap statement found in the available model card/paper. HEEDB is described as Harvard-Emory source data while Chapman is a separate public PhysioNet cohort; apparent source distinction does not prove zero patient overlap.",
                                    "Ningbo": "No record-level crosswalk or explicit overlap statement found. The checkpoint sources paper identifies HEEDB; Ningbo is a separate PhysioNet Challenge cohort. Patient identity overlap remains unknown.",
                                    "HEEDB_included_in_this_package": False}},
        "data": {"cohorts_order": ["Chapman", "Ningbo"], "split_counts": split_counts,
                 "record_order_sha256": split_id_hashes, "label_order_sha256": label_hashes,
                 "labels": ["TWA", "SB", "STach"], "label_codes": ["164934002", "426177001", "427084000"],
                 "lead_order": list(LEADS), "new_waveform_header_hashes": raw_input_hashes,
                 "all_Ningbo_signal_hash_bindings_match_frozen_split_and_canonical_records": True,
                 "all_Ningbo_actual_signal_hashes_previously_verified_by_preflight": "source code loads digital MAT arrays as float32 and checks per-record hashes; full preflight PASS scans 45,133 split rows before the earlier Founder baseline."},
        "head_training": {"arms": {"clean_head": "100% clean ECG views", "physics_head": "50% clean; 50% one of RA_LA_reversal, RA_LL_reversal, LA_LL_reversal equally likely"},
                          "no_training_masks_or_cycles": True,
                          "architecture": "one linear 1024 -> 3 logits head",
                          "paired_initialization": "identical initial head state within each cohort/seed clean/physics pair; one fixed encoder shared across all seeds and arms",
                          "feature_scaling": "mean/std estimated only from clean train embeddings; population std floor 1e-6; shared by both arms in cohort/seed pair",
                          "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
                          "batch_size": 256,
                          "loss": "BCEWithLogitsLoss with per-label positive weights from train labels only, negative/positive capped at 20",
                          "max_epochs": 50, "patience": 5,
                          "selection": "maximum clean validation TWA AUPRC; ties keep earlier epoch",
                          "training_view_selector": "SHA256(seed || epoch || canonical_record_id); first digest bit selects clean (0) vs augmented (1); for augmented views next 64 bits modulo 3 select equally likely one of the three named physical cable reversals; selection is deterministic by seed, epoch and canonical record ID.",
                          "shuffle": "same deterministic per-epoch order within each clean/physics pair, derived from seed and epoch",
                          "seeds": list(SEEDS), "seed_interpretation": "head initialization/training replicates only; not independently pretrained Founder encoders"},
        "preprocessing": {"input": "raw physical waveform in mV; 12 x 5000; 500 Hz; 10 seconds",
                          "nan": "WFDB NaN physical values -> zero before any physical permutation or filter; reject infinities and schema errors",
                          "permutations": "apply to I, II, V1-V6 independent physical signals before official filtering/preprocessing; derive displayed 12-lead channels; verify chest leads unchanged",
                          "official_order": ["WFDB NaN->0", "50Hz iirnotch Q=30, filtfilt per lead", "4th order 0.67-40Hz Butterworth bandpass, filtfilt per lead", "201-sample median baseline subtraction per lead", "global population z-score over 12x5000 with epsilon 1e-8", "float32"],
                          "mask_semantics": {"selected_indices": list(EXACT), "selected_leads": [LEADS[i] for i in EXACT],
                                             "operation": "set exactly the selected five channels to zero after the complete official Founder filtering and global z-score; preserve all other normalized channels exactly",
                                             "same_for_both_heads": True,
                                             "resnet_difference": "The M3 ResNet evaluator replaces selected raw displayed channels with train-lead means before its channelwise normalization; this produces zero at its normalized model input. Founder instead uses official global normalization, then explicitly zeros selected normalized input channels. The two models receive an absent-channel zero sentinel but mask insertion occurs at different pipeline stages."}},
        "cache_policy": {"old_clean_train_validation": "reuse only after SHA256, canonical record order, checkpoint/protocol binding, exact shape and same batch context 16 feature replay within abs tolerance 1e-6; pilot passed",
                         "old_test_clean_ra_ll": "reuse only after frozen same-batch test feature replay; otherwise compute complete cache anew",
                         "new_features": "feature vectors are produced by rerunning each transformed ECG through the frozen encoder, never by altering embedding coordinates; stream waveforms in batches and write resumable, hash-bound memory-mapped caches",
                         "test_embedding_inference": "one per condition and cohort; six heads consume the same 1024-D test feature cache"},
        "numerical_replay_tolerance": {"absolute": 1e-6, "relative": 0.0, "batch_size": 16,
                                        "why": "established on train/validation with same original batch context; the earlier 32-row replay pilot was invalid because it changed AMP batch shape"},
        "test_evaluation": {"conditions": ["clean", *TRANSFORM_ORDER(), "cycle_slots_120", "cycle_slots_201", *[r["name"] for r in mask_rows]],
                            "mask_panel_seed": 20261018, "mask_count": 32, "exact_mask_first": list(EXACT),
                            "metrics": ["AUPRC", "AUROC", "raw Brier", "ECE15", "sensitivity", "specificity", "predicted-positive fraction"],
                            "threshold": "linear 90th percentile of negative examples from that head's clean validation probabilities; no test tuning",
                            "primary_endpoint": "TWA", "secondary_endpoints": ["SB", "STach"],
                            "primary_contrast": "physics_head minus clean_head", "show_absolute_and_change_from_clean": True,
                            "bootstrap": {"replicates": 1000, "indices": {"Chapman": "outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy", "Ningbo": "outputs/ningbo_confirmation_v2/bootstrap_indices.npy"},
                                          "bootstrap_seed_rule": "use existing synchronized cohort index arrays unchanged", "resample_unit": "canonical record under existing one-record-per-patient assumption",
                                          "estimands": "paired physics_head minus clean_head differences for TWA AUPRC/AUROC/raw Brier at every fixed condition; plus mean TWA AUPRC across the 32-mask panel; average three seed-specific differences within each replicate",
                                          "report_full_cohort_point_and_bootstrap_mean_separately": True,
                                          "ci": "percentile 95% interval; an interval containing zero is not evidence of equality"}},
        "historical_results_context": {"clean_founder_logistic_head": "preserve as historical reference; primary contrast uses two newly trained heads with identical procedure",
                                       "transformer": "preserve existing mixed result: corrected Chapman M3 gate failed because seed 155921 mean-mask TWA AUPRC fell by 0.1126; RA-LL gains occurred but did not rescue that gate, and Ningbo M3 stage was not run.",
                                       "analysis_scope": "existing cohorts and one fixed encoder; exploratory transfer test only"},
        "excluded": ["HEEDB data", "new encoder pretraining", "full ECGFounder fine-tuning", "all 792 masks", "new architecture search", "hyperparameter tuning", "paper work"],
        "frozen_inputs_sha256": input_hashes,
        "verified_existing_test_cache_reuse_candidates": reuse_candidates,
        "mask_panel": mask_rows,
        "primary_sources": [
            {"type": "official model card", "url": "https://huggingface.co/PKUDigitalHealth/ECGFounder", "supports": "repository identity, checkpoint access, paper link"},
            {"type": "primary paper", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/", "supports": "HEEDB source and 10,771,552 expert-annotated ECG pretraining cohort"},
            {"type": "official code repository", "url": "https://github.com/PKUDigitalHealth/ECGFounder", "supports": "official implementation; local checkout pinned to commit above"}
        ],
    }
    protocol["protocol_sha256"] = hash_obj(protocol)
    (OUT / "FROZEN_PROTOCOL.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol["protocol_sha256"],
                      "mask_panel_size": len(mask_panel), "waveform_header_pairs_hashed": len(raw_input_hashes),
                      "input_files_hashed": len(input_hashes)}, ensure_ascii=False))


def TRANSFORM_ORDER() -> tuple[str, ...]:
    return ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")


if __name__ == "__main__":
    main()
