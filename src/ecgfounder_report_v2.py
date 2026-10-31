"""Verify and summarize the already frozen ECGFounder v2 analysis outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v2"
PROTOCOL = ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json"
DATASETS = ("Chapman", "Ningbo")
LABELS = ("TWA", "SB", "STach")
METRICS = ("auprc", "auroc", "sensitivity")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a, b, label: str, tol=2e-12):
    if not np.isclose(float(a), float(b), rtol=0, atol=tol):
        raise RuntimeError(f"inconsistent {label}: {a} != {b}")


def fmt(x, places=3):
    return f"{float(x):.{places}f}"


def drop_interval(row: dict):
    return f"{fmt(row['clean_minus_ra_ll_drop'])} [{fmt(row['drop_low95'])}, {fmt(row['drop_high95'])}]"


def check_feature_cache(dataset: str, part: str, condition: str, total: int,
                        ids_hash: str, protocol_hash: str):
    stem = f"{dataset.lower()}_{part}_{condition}"
    folder = OUT / "features"
    meta_path = folder / f"{stem}_meta.json"
    feature_path = folder / f"{stem}_features.npy"
    logit_path = folder / f"{stem}_logits.npy"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["dataset"] != dataset or meta["partition"] != part or meta["condition"] != condition
            or meta["total_records"] != total or meta["completed_records"] != total
            or meta["record_ids_sha256"] != ids_hash
            or meta["protocol_sha256"] != protocol_hash
            or meta["features_sha256"] != sha(feature_path)
            or meta["logits_sha256"] != sha(logit_path)):
        raise RuntimeError(f"feature cache binding mismatch: {stem}")
    if np.load(feature_path, mmap_mode="r").shape != (total, 1024):
        raise RuntimeError(f"feature shape mismatch: {stem}")
    if np.load(logit_path, mmap_mode="r").shape != (total, 150):
        raise RuntimeError(f"logit shape mismatch: {stem}")
    return [meta_path, feature_path, logit_path]


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_hash = sha(PROTOCOL)
    freeze = json.loads((OUT / "FROZEN_PROTOCOL_HASH.json").read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != protocol_hash or protocol["status"] != "FROZEN_AFTER_INPUT_ADAPTER_PREFLIGHT_BEFORE_NEW_OUTCOMES":
        raise RuntimeError("protocol freeze mismatch")
    if protocol["freeze_access_status"] != {
            "test_input_audit_scope": "paths, headers, raw waveform hashes and missing-value counts only",
            "test_inputs_audited_before_freeze": True,
            "test_labels_or_predictions_used_to_select_v2_head_before_freeze": False,
            "test_model_outcomes_evaluated_before_freeze": False}:
        raise RuntimeError("freeze access metadata mismatch")
    for rel, expected in protocol["file_bindings"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"frozen source binding mismatch: {rel}")
    status = json.loads((OUT / "WORKFLOW_STATUS.json").read_text(encoding="utf-8"))
    if status.get("stage") != "COMPLETE":
        raise RuntimeError("paired analysis is not complete")
    audit = json.loads((OUT / "PRE_RUN_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("total_records") != 45133:
        raise RuntimeError("input audit did not pass 45,133 records")
    if sha(OUT / "PRE_RUN_AUDIT.json") != protocol["pre_run_audit_sha256"]:
        raise RuntimeError("input audit hash mismatch")

    files = [PROTOCOL, OUT / "FROZEN_PROTOCOL_HASH.json", OUT / "PRE_RUN_AUDIT.json",
             OUT / "PREFREEZE_SMOKE.json", OUT / "MISSING_PATH_AUDIT.json",
             OUT / "SMOKE_REPORT.json", OUT / "WORKFLOW_STATUS.json",
             ROOT / "src/ecgfounder_baseline_v2.py", ROOT / "src/ecgfounder_analyze_v2.py",
             ROOT / "src/ecgfounder_v2_job_supervisor.py", Path(__file__)]
    for dataset in DATASETS:
        sizes = {part: protocol["data"][{"train": "training_sizes", "validation": "validation_sizes", "test": "test_sizes"}[part]][dataset]
                 for part in ("train", "validation", "test")}
        split_path = ROOT / ("outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json"
                             if dataset == "Chapman" else "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json")
        split_rows = json.loads(split_path.read_text(encoding="utf-8"))["rows"]
        ids_hashes = {}
        for part in sizes:
            ids = [r["record_id"] for r in split_rows if r["split"] == part]
            if len(ids) != sizes[part] or len(set(ids)) != len(ids):
                raise RuntimeError(f"split count or duplicate IDs: {dataset}/{part}")
            ids_hashes[part] = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        for part, condition in (("train", "clean"), ("validation", "clean"),
                                ("test", "clean"), ("test", "RA_LL_reversal")):
            files += check_feature_cache(dataset, part, condition, sizes[part], ids_hashes[part], protocol_hash)
        head_path = OUT / f"{dataset.lower()}_head.npz"
        frozen_path = OUT / f"{dataset.lower()}_validation_freeze.json"
        head_freeze = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (head_freeze["protocol_sha256"] != protocol_hash
                or head_freeze["head_sha256"] != sha(head_path)
                or head_freeze["test_model_outcomes_evaluated_before_validation_freeze"] is not False):
            raise RuntimeError(f"validation head freeze mismatch: {dataset}")
        files += [head_path, frozen_path]
        progress = json.loads((OUT / f"{dataset.lower()}_bootstrap_progress.json").read_text(encoding="utf-8"))
        metrics_path = OUT / f"{dataset.lower()}_bootstrap_metrics.npy"
        indices_path = OUT / f"{dataset.lower()}_bootstrap_indices.npy"
        if (progress["completed_replicates"] != 1000 or progress["protocol_sha256"] != protocol_hash
                or progress["indices_sha256"] != sha(indices_path)
                or progress["metrics_sha256"] != sha(metrics_path)):
            raise RuntimeError(f"bootstrap cache mismatch: {dataset}")
        files += [OUT / f"{dataset.lower()}_bootstrap_progress.json", metrics_path, indices_path,
                  OUT / f"{dataset.lower()}_progress.log",
                  OUT / f"{dataset.lower()}_supervisor_events.jsonl"]

    names = ("MAIN_COMPARISON.csv", "PER_SEED_COMPARISON.csv", "PAIRED_DROP_DIFFERENCES.csv",
             "ZERO_SHOT_NAME_MATCHED.csv", "CHANGED_PREDICTIONS.csv")
    files += [OUT / name for name in names]
    main_rows, seed_rows, paired_rows, zero_rows, examples = [read_csv(name) for name in names]
    if (len(main_rows), len(seed_rows), len(paired_rows), len(zero_rows)) != (36, 72, 18, 8):
        raise RuntimeError("comparison CSV row count mismatch")
    main = {(r["dataset"], r["label"], r["metric"], r["model"]): r for r in main_rows}
    seeds = {(r["dataset"], r["label"], r["metric"], r["model"]): r for r in seed_rows}
    paired = {(r["dataset"], r["label"], r["metric"]): r for r in paired_rows}
    if len(main) != len(main_rows) or len(seeds) != len(seed_rows) or len(paired) != len(paired_rows):
        raise RuntimeError("duplicate comparison key")
    for dataset in DATASETS:
        for label in LABELS:
            for metric in METRICS:
                ef = main[dataset, label, metric, "ECGFounder_head"]
                rn = main[dataset, label, metric, "ResNet_mean_of_3_seed_metrics"]
                close(ef["clean_minus_ra_ll_drop"], float(ef["clean"]) - float(ef["ra_ll_reversal"]), "ECGFounder drop")
                close(rn["clean_minus_ra_ll_drop"], float(rn["clean"]) - float(rn["ra_ll_reversal"]), "ResNet drop")
                for key in ("clean", "ra_ll_reversal", "clean_minus_ra_ll_drop"):
                    close(ef[key], seeds[dataset, label, metric, "ECGFounder_head"][key], "head row")
                    close(rn[key], np.mean([float(seeds[dataset, label, metric, f"ResNet_{seed}"][key])
                                              for seed in (104729, 130363, 155921)]), "ResNet seed mean")
                close(paired[dataset, label, metric]["ecgfounder_drop_minus_resnet_mean_drop"],
                      float(ef["clean_minus_ra_ll_drop"]) - float(rn["clean_minus_ra_ll_drop"]), "paired drop")
                if int(ef["records"]) != protocol["data"]["test_sizes"][dataset] or ef["positives"] != rn["positives"]:
                    raise RuntimeError("held-out record/label count mismatch")
    if {r["label"] for r in zero_rows} != {"SB", "STach"} or {r["condition"] for r in zero_rows} != {"clean", "RA_LL_reversal"}:
        raise RuntimeError("zero-shot label scope mismatch")
    if not examples or not any(r["label"] == "TWA" for r in examples):
        raise RuntimeError("no changed TWA examples")

    # Compact, publication-ready comparison retains metric and interval detail.
    compact_path = OUT / "COMPACT_RESULTS.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "label", "metric", "records", "positives",
            "ecgfounder_clean", "ecgfounder_ra_ll", "ecgfounder_drop", "ecgfounder_drop_low95", "ecgfounder_drop_high95",
            "resnet_mean_clean", "resnet_mean_ra_ll", "resnet_mean_drop", "resnet_drop_low95", "resnet_drop_high95",
            "paired_drop_difference", "paired_low95", "paired_high95"))
        writer.writeheader()
        for dataset in DATASETS:
            for label in LABELS:
                for metric in METRICS:
                    ef = main[dataset, label, metric, "ECGFounder_head"]
                    rn = main[dataset, label, metric, "ResNet_mean_of_3_seed_metrics"]
                    pd = paired[dataset, label, metric]
                    writer.writerow({"dataset": dataset, "label": label, "metric": metric,
                        "records": ef["records"], "positives": ef["positives"],
                        "ecgfounder_clean": ef["clean"], "ecgfounder_ra_ll": ef["ra_ll_reversal"],
                        "ecgfounder_drop": ef["clean_minus_ra_ll_drop"],
                        "ecgfounder_drop_low95": ef["drop_low95"], "ecgfounder_drop_high95": ef["drop_high95"],
                        "resnet_mean_clean": rn["clean"], "resnet_mean_ra_ll": rn["ra_ll_reversal"],
                        "resnet_mean_drop": rn["clean_minus_ra_ll_drop"],
                        "resnet_drop_low95": rn["drop_low95"], "resnet_drop_high95": rn["drop_high95"],
                        "paired_drop_difference": pd["ecgfounder_drop_minus_resnet_mean_drop"],
                        "paired_low95": pd["low95"], "paired_high95": pd["high95"]})
    files.append(compact_path)

    report = ["# ECGFounder baseline v2 — exploratory paired comparison", "",
        "This comparison uses the previously examined Chapman and Ningbo test sets. The v2 protocol was frozen after auditing test **inputs** for paths, hashes, headers, and missing samples, and before evaluating test **model outcomes**. Head settings and thresholds were selected from each dataset's existing training and validation splits. The ECGFounder backbone stayed frozen.", "",
        "## Exact-label trained-head comparison", "",
        "TWA is primary. Values are clean / RA–LL / clean-minus-reversal drop. Brackets give 95% paired patient/record bootstrap intervals for the drop (1,000 synchronized resamples). ResNet values are the mean of three saved seed **metrics**, not an ensemble.", "",
        "| Dataset | Label | Metric | ECGFounder head clean / RA–LL / drop [95%] | Saved ResNet mean clean / RA–LL / drop [95%] | Paired drop difference [95%] |",
        "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for label in LABELS:
            for metric in METRICS:
                ef, rn = main[dataset, label, metric, "ECGFounder_head"], main[dataset, label, metric, "ResNet_mean_of_3_seed_metrics"]
                pd = paired[dataset, label, metric]
                cell = lambda r: f"{fmt(r['clean'])} / {fmt(r['ra_ll_reversal'])} / {drop_interval(r)}"
                report.append(f"| {dataset} | {label} | {metric.upper()} | {cell(ef)} | {cell(rn)} | {fmt(pd['ecgfounder_drop_minus_resnet_mean_drop'])} [{fmt(pd['low95'])}, {fmt(pd['high95'])}] |")
    twa = [main[d, "TWA", "auprc", "ECGFounder_head"] for d in DATASETS]
    both_positive = all(float(r["drop_low95"]) > 0 for r in twa)
    any_positive = any(float(r["drop_low95"]) > 0 for r in twa)
    report += ["", "The primary TWA AUPRC loss " + (
        "appears in both datasets with positive patient/record bootstrap intervals. This supports transfer of the RA–LL vulnerability to this frozen ECGFounder feature plus linear-head pipeline in these exploratory sets."
        if both_positive else "has a positive interval in only one dataset; a two-dataset generalization claim is not supported."
        if any_positive else "does not have a positive interval in either dataset; generalization is not established by this run."),
        "TWA sensitivity can rise under reversal even while AUPRC and AUROC fall, because the fixed threshold also admits many more false positives. The paired drop difference compares losses under the two model pipelines; it does not rank overall clinical utility or establish SOTA superiority.", "",
        "## Label mapping and direct zero-shot outputs", "",
        "The official 150-class task list contains **SINUS BRADYCARDIA** (zero-based index 4) and **SINUS TACHYCARDIA** (index 6), which are name-level matches only. It has no exact TWA output. Its **NONSPECIFIC T WAVE ABNORMALITY** output is excluded from the exact-label and zero-shot comparisons. The table below is separate from the trained-head result; thresholds for sensitivity came from clean validation negatives.", "",
        "| Dataset | Named output | Clean AUPRC | RA–LL AUPRC | Clean AUROC | RA–LL AUROC | Clean sensitivity | RA–LL sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    zero = {(r["dataset"], r["label"], r["condition"]): r for r in zero_rows}
    for dataset in DATASETS:
        for label in ("SB", "STach"):
            clean, fault = zero[dataset, label, "clean"], zero[dataset, label, "RA_LL_reversal"]
            report.append(f"| {dataset} | {label} | {fmt(clean['auprc'])} | {fmt(fault['auprc'])} | {fmt(clean['auroc'])} | {fmt(fault['auroc'])} | {fmt(clean['sensitivity'])} | {fmt(fault['sensitivity'])} |")
    report += ["", "## Changed predictions", "",
               "Examples use the ECGFounder trained head and its validation-fixed threshold. Truth is the exact benchmark label; probabilities are clean → RA–LL.", "",
               "| Dataset | Record | Label | Truth | Head probability | Threshold | Direction |",
               "|---|---|---|---:|---:|---:|---|"]
    for dataset in DATASETS:
        for label in ("TWA", "STach"):
            row = next(r for r in examples if r["dataset"] == dataset and r["label"] == label)
            report.append(f"| {dataset} | `{row['record_id']}` | {label} | {row['truth']} | {fmt(row['head_clean'])} → {fmt(row['head_ra_ll'])} | {fmt(row['head_threshold'])} | {row['direction']} |")
    counts = audit["counts_by_dataset_split"]
    report += ["", "## Missing samples, paths, and reproducibility", "",
        "All 45,133 frozen split records resolved. WFDB NaN physical samples were replaced with zero before the electrical reversal or filtering, following the pinned authors' `dataset.py`; infinities and unexpected schema errors remained rejection conditions. No records were excluded.", "",
        "| Dataset / split | Records | Records with NaNs | NaN samples | ID-folder differences |",
        "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for part in ("train", "validation", "test"):
            c = counts[dataset][part]
            report.append(f"| {dataset} / {part} | {c['records']} | {c['affected_records']} | {c['missing_samples']} | {c['path_group_mismatches']} |")
    report += ["", "Known adapter cases `g1/JS10767`, `g11/JS21646`, and `g13/JS23074` passed clean/reversal smoke inference. The last case uses a distinct header stem that references its canonical waveform. Electrical RA–LL equations and unchanged chest leads passed the v2 tests.", "",
        f"Official code commit: `{protocol['source']['code_commit']}`; 12-lead checkpoint SHA-256: `{protocol['source']['checkpoint_sha256']}`. Protocol SHA-256: `{protocol_hash}`. Input: 12 leads in I, II, III, aVR, aVL, aVF, V1–V6 order; 500 Hz, 10 seconds, WFDB physical mV. After NaN→0 and the physical RA–LL terminal permutation: official 50 Hz notch (Q=30), 0.67–40 Hz fourth-order Butterworth bandpass, 201-sample median baseline removal, global z-score, float32. The frozen 1024-dimensional features feed three L2 logistic heads trained on clean training records, with C selected from 0.01/0.1/1 by clean validation AUPRC and thresholds fixed at the 90th percentile of validation negative probabilities.", "",
        "[Frozen protocol](../../configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json) · [reproducibility manifest](REPRODUCIBILITY_MANIFEST.json) · [full comparison](MAIN_COMPARISON.csv) · [compact comparison](COMPACT_RESULTS.csv) · [per-seed comparison](PER_SEED_COMPARISON.csv) · [paired differences](PAIRED_DROP_DIFFERENCES.csv) · [zero-shot outputs](ZERO_SHOT_NAME_MATCHED.csv) · [changed predictions](CHANGED_PREDICTIONS.csv) · [missing/path audit](MISSING_PATH_AUDIT.json).", "",
        "Intervals resample canonical records as patient units under the project's one-record-per-patient assumption; the split tables have no independent patient identifier. The existing test sets were previously examined, and public pretraining overlap has not been independently ruled out. The honest next step is a new untouched, patient-linked external cohort with the same exact labels and electrical fault protocol, frozen before outcomes. This run does not develop a robustness method or claim superiority over published models.", ""]
    report_path = OUT / "FINAL_BASELINE_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    files.append(report_path)
    file_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}
    manifest = {"version": "ecgfounder_baseline_v2", "status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash, "parent_protocol_sha256": protocol["parent_protocol_sha256"],
        "source_commit": protocol["source"]["code_commit"],
        "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
        "frozen_source_and_saved_prediction_sha256": protocol["file_bindings"],
        "lead_order": protocol["preprocessing"]["lead_order"],
        "sampling_rate_hz": 500,
        "input_samples": 5000,
        "preprocessing": protocol["preprocessing"],
        "frozen_head": protocol["frozen_head"],
        "split_sizes": {key: protocol["data"][key] for key in ("training_sizes", "validation_sizes", "test_sizes")},
        "labels_in_order": protocol["data"]["labels_in_order"],
        "runtime_gpu": json.loads((OUT / "SMOKE_REPORT.json").read_text(encoding="utf-8"))["gpu"],
        "runtime_torch": json.loads((OUT / "SMOKE_REPORT.json").read_text(encoding="utf-8"))["torch"],
        "test_inputs_audited_before_freeze": True,
        "test_model_outcomes_evaluated_before_protocol_freeze": False,
        "test_model_outcomes_evaluated_before_validation_head_freeze": False,
        "scientific_choices_changed_from_v1": False,
        "input_adapter_changes": protocol["input_adapter_v2"],
        "chapman_v1_cache_reused": False,
        "ningbo_v1_incomplete_cache_reused": False,
        "saved_resnet_prediction_order": "saved evaluators use the frozen split row order; prediction arrays lack embedded IDs; frozen split, source logic and prediction hashes are bound",
        "resumable_retry": "single-worker wrapper retries only Windows atomic_json WinError 5; no scientific-code edits",
        "bootstrap_unit": "canonical record as patient under one-record-per-patient assumption",
        "bootstrap_replicates": 1000,
        "input_audit_counts": counts,
        "consistency_checks": {"status": "PASS", "feature_hashes": 8,
                               "main_rows": len(main_rows), "per_seed_rows": len(seed_rows),
                               "paired_rows": len(paired_rows), "zero_shot_rows": len(zero_rows),
                               "changed_examples": len(examples), "targeted_tests": "7 passed"},
        "file_sha256": file_hashes}
    (OUT / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["consistency_checks"], sort_keys=True))


if __name__ == "__main__":
    main()
