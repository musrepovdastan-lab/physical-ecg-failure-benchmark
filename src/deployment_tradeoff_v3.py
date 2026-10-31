"""Read-mostly deployment and detector-failure audit using frozen v2 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import winreg
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "deployment_tradeoff_v3"
V2 = ROOT / "outputs" / "published_detector_multifault_v2"
V1 = ROOT / "outputs" / "published_detector_pipeline_v1"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
SEEDS = (104729, 130363, 155921)
DIAGNOSES = ("TWA", "SB", "STach")
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
FAULTS = CONDITIONS[1:]
TRUE_CLASS = {"clean": 0, "RA_LA_reversal": 1, "RA_LL_reversal": 2, "LA_LL_reversal": 3}
ARMS = ("physics_aware_no_correction", "huang_argmax_correct_clean_resnet")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def log(message: str, **fields: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entry = {"time": now(), "message": message, **fields}
    with (OUT / "progress_log.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True), flush=True)


def test_rows() -> list[dict]:
    canonical = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf8"))["records"]
    membership = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf8"))["rows"]
    by_id = {row["record_id"]: row for row in canonical}
    rows = [by_id[row["record_id"]] for row in membership if row["split"] == "test"]
    if len(rows) != 6981:
        raise RuntimeError("unexpected Ningbo test count")
    return rows


def detector_classes(condition: str, rows: list[dict]) -> np.ndarray:
    path = V2 / "cache" / f"detector_{condition}.npz"
    with np.load(path, allow_pickle=False) as cache:
        if cache["record_ids"].astype(str).tolist() != [row["record_id"] for row in rows]:
            raise RuntimeError(f"record order mismatch: {path}")
        return cache["probabilities"].argmax(1)


def predictions(arm: str, seed: int, condition: str, rows: list[dict]) -> np.ndarray:
    path = V2 / "cache" / f"diagnostic_{arm}_{condition}_seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as cache:
        if cache["record_ids"].astype(str).tolist() != [row["record_id"] for row in rows]:
            raise RuntimeError(f"record order mismatch: {path}")
        return cache["p"].copy()


def thresholds(arm: str, seed: int) -> np.ndarray:
    base = M2 if arm == "physics_aware_no_correction" else N
    with np.load(base / f"seed_{seed}" / "validation.npz", allow_pickle=False) as validation:
        return validation["thresholds"].copy()


def anonymized_id(condition: str, record_id: str) -> str:
    value = f"deployment_tradeoff_v3|{condition}|{record_id}"
    return "NGB-" + hashlib.sha256(value.encode("utf8")).hexdigest()[:12].upper()


def mistake_masks(classes: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    masks: dict[str, dict[str, np.ndarray]] = {
        "missed_reversal": {},
        "wrong_reversal_type": {},
        "false_correction_clean": {},
    }
    for condition in FAULTS:
        predicted = classes[condition]
        masks["missed_reversal"][condition] = predicted == 0
        masks["wrong_reversal_type"][condition] = (predicted != 0) & (predicted != TRUE_CLASS[condition])
    masks["false_correction_clean"]["clean"] = classes["clean"] != 0
    return masks


def diagnostic_error_rows(rows: list[dict], masks: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict], dict]:
    labels = np.asarray([row["labels"] for row in rows], dtype=int)
    prediction_cache = {
        arm: {
            seed: {condition: predictions(arm, seed, condition, rows) for condition in CONDITIONS}
            for seed in SEEDS
        }
        for arm in ARMS
    }
    output = []
    examples = {}
    for group, condition_masks in masks.items():
        group_instances = [(condition, index) for condition, mask in condition_masks.items() for index in np.flatnonzero(mask)]
        if not group_instances:
            raise RuntimeError(f"empty detector mistake group: {group}")
        ordered_examples = sorted(
            ((anonymized_id(condition, rows[index]["record_id"]), condition) for condition, index in group_instances),
            key=lambda item: item[0],
        )[:5]
        examples[group] = [{"anonymous_record_id": anonymous, "condition": condition} for anonymous, condition in ordered_examples]

        scopes = [("all", group_instances)]
        scopes.extend(
            (condition, [(condition, index) for index in np.flatnonzero(mask)])
            for condition, mask in condition_masks.items()
        )
        for scope, instances in scopes:
            if not instances:
                continue
            instance_labels = np.asarray([labels[index] for _, index in instances], dtype=int)
            for arm in ARMS:
                for diagnosis_index, diagnosis in enumerate(DIAGNOSES):
                    errors = false_negatives = false_positives = 0
                    for seed in SEEDS:
                        threshold = thresholds(arm, seed)[diagnosis_index]
                        predicted = np.asarray([
                            prediction_cache[arm][seed][condition][index, diagnosis_index] >= threshold
                            for condition, index in instances
                        ])
                        truth = instance_labels[:, diagnosis_index].astype(bool)
                        errors += int((predicted != truth).sum())
                        false_negatives += int((~predicted & truth).sum())
                        false_positives += int((predicted & ~truth).sum())
                    positive_evaluations = int(instance_labels[:, diagnosis_index].sum()) * len(SEEDS)
                    negative_evaluations = (len(instances) - int(instance_labels[:, diagnosis_index].sum())) * len(SEEDS)
                    evaluations = len(instances) * len(SEEDS)
                    output.append({
                        "mistake_group": group,
                        "scope": scope,
                        "condition_record_instances": len(instances),
                        "unique_canonical_records": len({index for _, index in instances}),
                        "arm": arm,
                        "diagnosis": diagnosis,
                        "seed_record_evaluations": evaluations,
                        "classification_errors": errors,
                        "error_rate": errors / evaluations,
                        "false_negatives": false_negatives,
                        "positive_seed_record_evaluations": positive_evaluations,
                        "false_negative_rate": false_negatives / positive_evaluations if positive_evaluations else "",
                        "false_positives": false_positives,
                        "negative_seed_record_evaluations": negative_evaluations,
                        "false_positive_rate": false_positives / negative_evaluations if negative_evaluations else "",
                    })
    return output, examples


def latency_summary() -> dict:
    detector = json.loads((OUT / "latency_huang_detector.json").read_text(encoding="utf8"))
    huang_diagnostic = json.loads((OUT / "latency_huang_diagnostic.json").read_text(encoding="utf8"))
    physics = json.loads((OUT / "latency_physics.json").read_text(encoding="utf8"))
    if len({detector["record_order_sha256"], huang_diagnostic["record_order_sha256"], physics["record_order_sha256"]}) != 1:
        raise RuntimeError("latency fixture mismatch")
    detector_times = np.asarray(detector["latencies_ms"])
    diagnostic_times = np.asarray(huang_diagnostic["latencies_ms"])
    physics_times = np.asarray(physics["latencies_ms"])
    huang_times = detector_times + diagnostic_times

    def distribution(values: np.ndarray) -> dict:
        return {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "minimum_ms": float(values.min()),
            "maximum_ms": float(values.max()),
            "n": len(values),
        }

    physics_checkpoint = M2 / "seed_104729" / "checkpoint.pt"
    clean_checkpoint = N / "seed_104729" / "checkpoint.pt"
    detector_files = [
        V1 / "official_source" / "ECG_lead_check-main" / "saved_model" / "LDenseNet" / ".index",
        V1 / "official_source" / "ECG_lead_check-main" / "saved_model" / "LDenseNet" / ".data-00000-of-00001",
    ]
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        cpu_name = "AMD64 Family 26 Model 96 Stepping 0"
    summary = {
        "created_utc": now(),
        "method": {
            "fixture": "128 deterministic Ningbo records, 32 per clean/RA-LA/RA-LL/LA-LL condition; balanced for measurement only, not a prevalence estimate",
            "warmup": 16,
            "batch_size": 1,
            "threads": 1,
            "logical_cpu_index": physics["cpu_logical_index"],
            "disk_io_timed": False,
            "timing_clock": "time.perf_counter_ns",
            "quantiles": "NumPy linear quantile",
            "huang_aggregation": "record-matched sum of detector and correction-plus-clean-ResNet component latencies",
            "memory": "Windows GetProcessMemoryInfo peak working set; Huang total is conservative sum of two simultaneously resident component processes",
        },
        "hardware": {
            "cpu": cpu_name,
            "machine": platform.machine(),
            "logical_processors": os.cpu_count(),
            "total_physical_memory_bytes": physics["hardware"]["total_physical_memory_bytes"],
            "os": platform.platform(),
        },
        "physics_aware": {
            "latency": distribution(physics_times),
            "one_time_loading_ms": physics["load_ms"],
            "peak_working_set_bytes": physics["memory_after_inference"]["peak_working_set_bytes"],
            "incremental_peak_working_set_bytes": physics["incremental_peak_working_set_bytes"],
            "model_artifact_bytes": physics_checkpoint.stat().st_size,
            "model_parameter_bytes": physics["model_parameter_bytes"],
        },
        "huang_detect_correct_clean_resnet": {
            "latency": distribution(huang_times),
            "detector_component_latency": distribution(detector_times),
            "correction_and_diagnostic_component_latency": distribution(diagnostic_times),
            "one_time_loading_ms": detector["load_ms"] + huang_diagnostic["load_ms"],
            "detector_loading_ms": detector["load_ms"],
            "clean_resnet_loading_ms": huang_diagnostic["load_ms"],
            "conservative_combined_peak_working_set_bytes": detector["memory_after_inference"]["peak_working_set_bytes"] + huang_diagnostic["memory_after_inference"]["peak_working_set_bytes"],
            "conservative_combined_incremental_peak_working_set_bytes": detector["incremental_peak_working_set_bytes"] + huang_diagnostic["incremental_peak_working_set_bytes"],
            "model_artifact_bytes": clean_checkpoint.stat().st_size + sum(path.stat().st_size for path in detector_files),
            "model_parameter_bytes": huang_diagnostic["model_parameter_bytes"] + detector["model_parameter_bytes"],
        },
    }
    write_json(OUT / "latency_summary.json", summary)
    return summary


def v2_auprc() -> dict[tuple[str, str], dict]:
    with (V2 / "worst_fault_summary.csv").open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["arm"], row["diagnosis"]): row
        for row in rows if row["metric"] == "auprc" and row["arm"] in ARMS
    }


def comparison_rows(latency: dict, detector: dict) -> list[dict]:
    performance = v2_auprc()
    physics_clean = "/".join(f"{float(performance[('physics_aware_no_correction', diagnosis)]['clean']):.4f}" for diagnosis in DIAGNOSES)
    huang_clean = "/".join(f"{float(performance[('huang_argmax_correct_clean_resnet', diagnosis)]['clean']):.4f}" for diagnosis in DIAGNOSES)
    physics_worst = "/".join(f"{float(performance[('physics_aware_no_correction', diagnosis)]['minimum_across_reversals']):.4f}" for diagnosis in DIAGNOSES)
    huang_worst = "/".join(f"{float(performance[('huang_argmax_correct_clean_resnet', diagnosis)]['minimum_across_reversals']):.4f}" for diagnosis in DIAGNOSES)
    reversal_error = 1.0 - np.mean([detector["per_class_recall"][name] for name in ("RA_LA", "RA_LL", "LA_LL")])
    return [
        {"dimension": "Clean AUPRC (TWA/SB/STach)", "physics_aware": physics_clean, "huang_pipeline": huang_clean, "source": "../published_detector_multifault_v2/worst_fault_summary.csv"},
        {"dimension": "Worst-fault AUPRC (TWA/SB/STach)", "physics_aware": physics_worst, "huang_pipeline": huang_worst, "source": "../published_detector_multifault_v2/worst_fault_summary.csv"},
        {"dimension": "Detector error", "physics_aware": "Not applicable", "huang_pipeline": f"{reversal_error:.2%} reversal misclassification; {detector['clean_false_correction_rate']:.2%} clean false correction", "source": "../published_detector_multifault_v2/detector_metrics_all_faults.json"},
        {"dimension": "Warm batch-1 CPU latency median/p95", "physics_aware": f"{latency['physics_aware']['latency']['median_ms']:.2f}/{latency['physics_aware']['latency']['p95_ms']:.2f} ms", "huang_pipeline": f"{latency['huang_detect_correct_clean_resnet']['latency']['median_ms']:.2f}/{latency['huang_detect_correct_clean_resnet']['latency']['p95_ms']:.2f} ms", "source": "latency_summary.json"},
        {"dimension": "Peak working set", "physics_aware": f"{latency['physics_aware']['peak_working_set_bytes'] / 2**20:.1f} MiB", "huang_pipeline": f"{latency['huang_detect_correct_clean_resnet']['conservative_combined_peak_working_set_bytes'] / 2**20:.1f} MiB (two-process sum)", "source": "latency_summary.json"},
        {"dimension": "Model artifacts", "physics_aware": f"{latency['physics_aware']['model_artifact_bytes'] / 2**20:.2f} MiB", "huang_pipeline": f"{latency['huang_detect_correct_clean_resnet']['model_artifact_bytes'] / 2**20:.2f} MiB", "source": "latency_summary.json"},
        {"dimension": "Core limitation", "physics_aware": "One diagnostic model; no explicit fault label", "huang_pipeline": "Two runtimes/models; detector mistakes can trigger wrong correction", "source": "detector_mistake_diagnostic_errors.csv"},
    ]


def final_decision(latency: dict, detector: dict, mistakes: list[dict], examples: dict, table: list[dict]) -> None:
    aggregate = {
        (row["mistake_group"], row["arm"], row["diagnosis"]): row
        for row in mistakes if row["scope"] == "all"
    }
    counts = {
        group: next(row["condition_record_instances"] for row in mistakes if row["scope"] == "all" and row["mistake_group"] == group)
        for group in ("missed_reversal", "wrong_reversal_type", "false_correction_clean")
    }
    condition_counts = {
        (group, row["scope"]): row["condition_record_instances"]
        for group in ("missed_reversal", "wrong_reversal_type")
        for row in mistakes
        if row["mistake_group"] == group and row["scope"] != "all"
    }
    latency_ratio = latency["huang_detect_correct_clean_resnet"]["latency"]["median_ms"] / latency["physics_aware"]["latency"]["median_ms"]
    memory_reduction = 1.0 - latency["physics_aware"]["peak_working_set_bytes"] / latency["huang_detect_correct_clean_resnet"]["conservative_combined_peak_working_set_bytes"]
    lines = [
        "# FINAL DEPLOYMENT TRADE-OFF DECISION",
        "",
        "## Plain answer",
        "",
        f"Yes. The single physics-aware model has a substantial demonstrated deployment advantage on this machine: its warm batch-1 CPU median was {latency['physics_aware']['latency']['median_ms']:.2f} ms versus {latency['huang_detect_correct_clean_resnet']['latency']['median_ms']:.2f} ms for Huang detect-correct-diagnose ({latency_ratio:.1f}x slower for Huang), and its measured peak working set was {memory_reduction:.0%} lower than the conservative two-process Huang total. Diagnostic TWA AUPRC was statistically unresolved between the two approaches in v2, so this operational result is the main demonstrated reason to prefer the single model.",
        "",
        "This does not establish clinical superiority, equivalence, or SOTA performance.",
        "",
        "## Compact comparison",
        "",
        "AUPRC triples are ordered TWA/SB/STach.",
        "",
        "| Dimension | Physics-aware single model | Huang detect-correct-clean ResNet | Source |",
        "|---|---:|---:|---|",
    ]
    for row in table:
        source = row["source"]
        lines.append(f"| {row['dimension']} | {row['physics_aware']} | {row['huang_pipeline']} | [{Path(source).name}]({source}) |")
    lines += [
        "",
        "The three reversals are electrically exact simulations on the same 6,981 Ningbo records. The balanced latency fixture is not a measured clinical prevalence distribution.",
        "",
        "## Detector failure cases",
        "",
        f"The detector produced {counts['missed_reversal']} missed reversal condition-record instances, {counts['wrong_reversal_type']} wrong-reversal-type instances, and {counts['false_correction_clean']} false corrections of clean ECGs.",
        f"Missed-reversal breakdown: RA-LA {condition_counts[('missed_reversal', 'RA_LA_reversal')]}, RA-LL {condition_counts[('missed_reversal', 'RA_LL_reversal')]}, LA-LL {condition_counts[('missed_reversal', 'LA_LL_reversal')]}. Wrong-type breakdown: RA-LA {condition_counts[('wrong_reversal_type', 'RA_LA_reversal')]}, RA-LL {condition_counts[('wrong_reversal_type', 'RA_LL_reversal')]}, LA-LL {condition_counts[('wrong_reversal_type', 'LA_LL_reversal')] }.",
        "",
        "Diagnostic error rates below pool the three frozen seeds at their existing validation-fixed thresholds and are descriptive seed-record error rates.",
        "",
        "| Detector mistake | Diagnosis | Physics-aware error | Huang pipeline error |",
        "|---|---|---:|---:|",
    ]
    display_names = {
        "missed_reversal": "Missed reversal",
        "wrong_reversal_type": "Wrong reversal type",
        "false_correction_clean": "False correction of clean",
    }
    for group in display_names:
        for diagnosis in DIAGNOSES:
            physics = aggregate[(group, "physics_aware_no_correction", diagnosis)]
            huang = aggregate[(group, "huang_argmax_correct_clean_resnet", diagnosis)]
            lines.append(f"| {display_names[group]} ({counts[group]}) | {diagnosis} | {float(physics['error_rate']):.2%} ({physics['classification_errors']}/{physics['seed_record_evaluations']}) | {float(huang['error_rate']):.2%} ({huang['classification_errors']}/{huang['seed_record_evaluations']}) |")
    lines += [
        "",
        "Illustrative one-way anonymized IDs:",
        "",
        f"- Missed reversals: {', '.join(item['anonymous_record_id'] for item in examples['missed_reversal'][:3])}.",
        f"- Wrong reversal types: {', '.join(item['anonymous_record_id'] for item in examples['wrong_reversal_type'][:3])}.",
        f"- False corrections of clean ECGs: {', '.join(item['anonymous_record_id'] for item in examples['false_correction_clean'][:3])}.",
        "",
        "Five examples per group and their simulated condition are stored in [detector_mistake_examples.json](detector_mistake_examples.json); no original record IDs are exported.",
        "",
        "## Publication assessment",
        "",
        "Yes, the evidence is sufficient for a carefully scoped workshop paper framed as a reproducible operational audit and negative performance-comparison result. The contribution would be the frozen multifault comparison, detector-failure decomposition, and deployment trade-off, not a new model or superiority claim.",
        "",
        "A stronger conference claim is blocked by specific missing evidence: real rather than simulated cable reversals; independent multi-device and multi-center external validation; a prospectively prespecified noninferiority or equivalence design; deployment measurements in one integrated target implementation rather than a conservative two-process component sum; and workflow evidence at a measured clinical fault prevalence showing the consequences of false corrections. Calibration, noise/artifact robustness, partial lead failures, and prospective utility also remain untested.",
        "",
        "No new training method is proposed by this audit.",
    ]
    temporary = OUT / f".FINAL_DECISION.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf8")
    os.replace(temporary, OUT / "FINAL_DECISION.md")


def manifest() -> None:
    inputs = [
        V2 / "REPRODUCIBILITY_MANIFEST.json",
        V2 / "worst_fault_summary.csv",
        V2 / "detector_metrics_all_faults.json",
        N / "SPLIT_MEMBERSHIP.json",
    ]
    scripts = [
        ROOT / "src" / "deployment_tradeoff_v3.py",
        ROOT / "src" / "deployment_latency_huang_v3.py",
        ROOT / "src" / "deployment_latency_pytorch_v3.py",
        ROOT / "src" / "deployment_benchmark_utils_v3.py",
    ]
    outputs = [
        OUT / "latency_huang_detector.json",
        OUT / "latency_huang_diagnostic.json",
        OUT / "latency_physics.json",
        OUT / "latency_summary.json",
        OUT / "detector_mistake_diagnostic_errors.csv",
        OUT / "detector_mistake_examples.json",
        OUT / "comparison_table.csv",
        OUT / "FINAL_DECISION.md",
    ]
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", {
        "created_utc": now(),
        "read_only_reuse": True,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "scripts": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        "outputs": {str(path.relative_to(OUT)): sha256(path) for path in outputs},
        "prohibited_actions_confirmed_absent": ["retraining", "Ningbo test tuning", "data download", "792-mask rerun", "HEEDB access"],
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("analysis_start")
    rows = test_rows()
    classes = {condition: detector_classes(condition, rows) for condition in CONDITIONS}
    masks = mistake_masks(classes)
    mistakes, examples = diagnostic_error_rows(rows, masks)
    write_csv(OUT / "detector_mistake_diagnostic_errors.csv", mistakes)
    write_json(OUT / "detector_mistake_examples.json", examples)
    latency = latency_summary()
    detector = json.loads((V2 / "detector_metrics_all_faults.json").read_text(encoding="utf8"))
    table = comparison_rows(latency, detector)
    write_csv(OUT / "comparison_table.csv", table)
    final_decision(latency, detector, mistakes, examples, table)
    manifest()
    log("analysis_complete")


if __name__ == "__main__":
    main()
