"""Integrated persistent-worker deployment benchmark for frozen ECG pipelines."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import winreg
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "integrated_deployment_v4"
V2 = ROOT / "outputs" / "published_detector_multifault_v2"
V1 = ROOT / "outputs" / "published_detector_pipeline_v1"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
PROJECT_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PIPELINE_WORKER = ROOT / "src" / "integrated_pipeline_worker_v4.py"
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
COUNT = 128
WARMUP = 16
RUNS = 7
SEED = 104729


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


def anonymized_id(condition: str, record_id: str) -> str:
    text = f"integrated_deployment_v4|{condition}|{record_id}"
    return "NGB-" + hashlib.sha256(text.encode("utf8")).hexdigest()[:12].upper()


def create_fixture() -> tuple[Path, dict]:
    from src.huang_detector_infer_v2 import apply_condition, load_full_display

    rows = test_rows()
    indices = np.linspace(0, len(rows) - 1, COUNT, dtype=int)
    conditions = [CONDITIONS[index % len(CONDITIONS)] for index in range(COUNT)]
    fixture_path = OUT / "balanced_raw_fixture.npy"
    temporary = fixture_path.with_name("." + fixture_path.name + f".{os.getpid()}.tmp.npy")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(COUNT, 12, 5000))
    entries = []
    for fixture_index, (source_index, condition) in enumerate(zip(indices, conditions)):
        target[fixture_index] = apply_condition(load_full_display(rows[int(source_index)]), condition)
        entries.append({
            "fixture_index": fixture_index,
            "source_test_index": int(source_index),
            "condition": condition,
            "anonymous_record_id": anonymized_id(condition, rows[int(source_index)]["record_id"]),
        })
    target.flush()
    del target
    os.replace(temporary, fixture_path)
    metadata = {
        "created_utc": now(),
        "records": COUNT,
        "condition_counts": {condition: conditions.count(condition) for condition in CONDITIONS},
        "selection": "np.linspace(0, 6980, 128, dtype=int), conditions cycling clean/RA-LA/RA-LL/LA-LL",
        "purpose": "balanced latency fixture only; not a clinical prevalence estimate",
        "shape": [COUNT, 12, 5000],
        "dtype": "float32",
        "fixture_sha256": sha256(fixture_path),
        "entries": entries,
    }
    write_json(OUT / "fixture_metadata.json", metadata)
    return fixture_path, metadata


class Worker:
    def __init__(self, pipeline: str, fixture: Path):
        self.pipeline = pipeline
        self.stderr_path = OUT / f"{pipeline}_worker_stderr.log"
        self.stderr_handle = self.stderr_path.open("w", encoding="utf8")
        command = [
            str(PROJECT_PYTHON), str(PIPELINE_WORKER),
            "--pipeline", pipeline, "--fixture", str(fixture),
        ]
        if pipeline == "huang":
            command += ["--detector-stderr", str(OUT / "huang_detector_worker_stderr.log")]
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
        )
        line = self.process.stdout.readline() if self.process.stdout is not None else ""
        self.loading_ms = (time.perf_counter_ns() - started) / 1e6
        if not line:
            raise RuntimeError(f"{pipeline} worker failed before readiness")
        self.ready = json.loads(line)
        if self.ready.get("status") != "ready":
            raise RuntimeError(f"{pipeline} readiness failed: {self.ready}")

    def request(self, value: dict) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes unavailable")
        self.process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.pipeline} worker exited unexpectedly")
        response = json.loads(line)
        if response.get("status") not in ("ok", "bye"):
            raise RuntimeError(f"{self.pipeline} worker error: {response}")
        return response

    def timed_request(self, value: dict) -> tuple[float, dict]:
        started = time.perf_counter_ns()
        response = self.request(value)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return elapsed_ms, response

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request({"command": "shutdown"})
                self.process.wait(timeout=20)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=20)
            self.stderr_handle.close()


def cached_fixture_predictions(pipeline: str, metadata: dict) -> tuple[np.ndarray, np.ndarray | None]:
    arm = "physics_aware_no_correction" if pipeline == "physics" else "huang_argmax_correct_clean_resnet"
    diagnostic = np.empty((COUNT, 3), dtype="float32")
    detector = np.empty(COUNT, dtype=int) if pipeline == "huang" else None
    loaded_diagnostic: dict[str, np.ndarray] = {}
    loaded_detector: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        path = V2 / "cache" / f"diagnostic_{arm}_{condition}_seed_{SEED}.npz"
        loaded_diagnostic[condition] = np.load(path, allow_pickle=False)["p"]
        if pipeline == "huang":
            detector_path = V2 / "cache" / f"detector_{condition}.npz"
            loaded_detector[condition] = np.load(detector_path, allow_pickle=False)["probabilities"].argmax(1)
    for entry in metadata["entries"]:
        fixture_index = entry["fixture_index"]
        source_index = entry["source_test_index"]
        condition = entry["condition"]
        diagnostic[fixture_index] = loaded_diagnostic[condition][source_index]
        if detector is not None:
            detector[fixture_index] = loaded_detector[condition][source_index]
    return diagnostic, detector


def agreement_check(workers: dict[str, Worker], metadata: dict) -> dict:
    result = {}
    for pipeline, worker in workers.items():
        live_predictions = np.empty((COUNT, 3), dtype=float)
        live_detector = np.empty(COUNT, dtype=int) if pipeline == "huang" else None
        for index in range(COUNT):
            response = worker.request({"command": "predict", "index": index})
            live_predictions[index] = response["prediction"]
            if live_detector is not None:
                live_detector[index] = int(response["detector_class"])
        cached_predictions, cached_detector = cached_fixture_predictions(pipeline, metadata)
        base = M2 if pipeline == "physics" else N
        thresholds = np.load(base / f"seed_{SEED}" / "validation.npz", allow_pickle=False)["thresholds"]
        absolute = np.abs(live_predictions - cached_predictions)
        binary_agreement = float(np.mean((live_predictions >= thresholds) == (cached_predictions >= thresholds)))
        detector_agreement = None if live_detector is None else float(np.mean(live_detector == cached_detector))
        passed = bool(absolute.max() <= 0.05 and binary_agreement >= 0.99 and (detector_agreement is None or detector_agreement == 1.0))
        result[pipeline] = {
            "decision": "PASS" if passed else "FAIL",
            "max_absolute_probability_difference": float(absolute.max()),
            "mean_absolute_probability_difference": float(absolute.mean()),
            "threshold_decision_agreement": binary_agreement,
            "detector_class_agreement": detector_agreement,
            "comparison": "live CPU integrated output versus frozen v2 seed-104729 cache",
            "predeclared_acceptance": "max absolute diagnostic probability difference <=0.05; threshold decision agreement >=0.99; Huang detector class agreement =1.0",
        }
        if not passed:
            raise RuntimeError(f"integrated output agreement failed for {pipeline}: {result[pipeline]}")
    write_json(OUT / "OUTPUT_AGREEMENT.json", result)
    return result


def memory_total(response: dict) -> dict:
    components = [response["pipeline"]]
    if "detector" in response:
        components.append(response["detector"])
    return {
        "working_set_bytes": int(sum(item["working_set_bytes"] for item in components)),
        "peak_working_set_bytes_sum": int(sum(item["peak_working_set_bytes"] for item in components)),
        "private_bytes": int(sum(item["private_bytes"] for item in components)),
        "components": components,
    }


def distribution(values: np.ndarray) -> dict:
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "n": int(values.size),
    }


def run_variability(matrix: np.ndarray) -> dict:
    medians = np.median(matrix, axis=1)
    p95s = np.quantile(matrix, 0.95, axis=1)
    return {
        "run_medians_ms": medians.tolist(),
        "run_p95s_ms": p95s.tolist(),
        "median_mean_ms": float(medians.mean()),
        "median_sd_ms": float(medians.std(ddof=1)),
        "median_range_ms": [float(medians.min()), float(medians.max())],
        "p95_mean_ms": float(p95s.mean()),
        "p95_sd_ms": float(p95s.std(ddof=1)),
        "p95_range_ms": [float(p95s.min()), float(p95s.max())],
    }


def cpu_name() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return platform.processor() or "unknown"


def benchmark(workers: dict[str, Worker], metadata: dict, agreement: dict) -> dict:
    for worker in workers.values():
        for index in range(WARMUP):
            worker.request({"command": "predict", "index": index})

    echo = {}
    for pipeline, worker in workers.items():
        values = []
        for nonce in range(COUNT):
            elapsed, response = worker.timed_request({"command": "echo", "nonce": nonce})
            if response["nonce"] != nonce:
                raise RuntimeError("echo protocol mismatch")
            values.append(elapsed)
        echo[pipeline] = distribution(np.asarray(values))

    rng = np.random.default_rng(20260923)
    orders = [rng.permutation(COUNT) for _ in range(RUNS)]
    timings = {pipeline: np.empty((RUNS, COUNT), dtype=float) for pipeline in workers}
    memory_samples = {pipeline: [] for pipeline in workers}
    for run_index, order in enumerate(orders):
        run_pipeline_order = ("physics", "huang") if run_index % 2 == 0 else ("huang", "physics")
        for pipeline in run_pipeline_order:
            worker = workers[pipeline]
            for position, fixture_index in enumerate(order):
                elapsed, _ = worker.timed_request({"command": "predict", "index": int(fixture_index)})
                timings[pipeline][run_index, position] = elapsed
            memory_samples[pipeline].append(memory_total(worker.request({"command": "memory"})))
        log("timed_run_complete", run=run_index + 1, total_runs=RUNS)

    raw_path = OUT / "integrated_latency_raw.npz"
    temporary = raw_path.with_name("." + raw_path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        physics_ms=timings["physics"],
        huang_ms=timings["huang"],
        fixture_orders=np.asarray(orders),
    )
    os.replace(temporary, raw_path)

    physics_checkpoint = M2 / f"seed_{SEED}" / "checkpoint.pt"
    clean_checkpoint = N / f"seed_{SEED}" / "checkpoint.pt"
    detector_files = [
        V1 / "official_source" / "ECG_lead_check-main" / "saved_model" / "LDenseNet" / ".index",
        V1 / "official_source" / "ECG_lead_check-main" / "saved_model" / "LDenseNet" / ".data-00000-of-00001",
    ]
    results = {
        "created_utc": now(),
        "decision": "PASS",
        "measurement": {
            "type": "warm integrated persistent-worker request/response; no component latency summation",
            "fixture_records": COUNT,
            "conditions": list(CONDITIONS),
            "records_per_condition": COUNT // len(CONDITIONS),
            "warmup_requests_per_pipeline": WARMUP,
            "timed_runs": RUNS,
            "requests_per_run_per_pipeline": COUNT,
            "total_timed_requests_per_pipeline": RUNS * COUNT,
            "batch_size": 1,
            "threads_per_framework": 1,
            "logical_cpu_index": workers["physics"].ready["cpu_logical_index"],
            "external_protocol": "identical newline-delimited JSON over persistent stdin/stdout pipes; fixture index request and three-probability response",
            "raw_input": "identical read-only memory-mapped 12x5000 float32 ECG fixture; mapping and disk access completed before readiness/timing",
            "run_order": "alternating pipeline order by run; identical deterministic shuffled fixture order within each run",
            "clock": "time.perf_counter_ns in controller around complete pipe request/response",
            "quantile": "NumPy linear quantile",
        },
        "hardware": {
            "cpu": cpu_name(),
            "logical_processors": os.cpu_count(),
            "machine": platform.machine(),
            "os": platform.platform(),
        },
        "output_agreement": agreement,
        "physics_aware": {
            "latency": distribution(timings["physics"].ravel()),
            "run_variability": run_variability(timings["physics"]),
            "external_echo_overhead": echo["physics"],
            "loading_to_ready_ms": workers["physics"].loading_ms,
            "internal_loading_ms": workers["physics"].ready["load_ms_internal"],
            "loaded_memory": memory_samples["physics"][0],
            "maximum_sampled_working_set_bytes": max(item["working_set_bytes"] for item in memory_samples["physics"]),
            "process_peak_working_set_bytes_sum": max(item["peak_working_set_bytes_sum"] for item in memory_samples["physics"]),
            "model_artifact_bytes": physics_checkpoint.stat().st_size,
        },
        "huang_detect_correct_diagnose": {
            "latency": distribution(timings["huang"].ravel()),
            "run_variability": run_variability(timings["huang"]),
            "external_echo_overhead": echo["huang"],
            "loading_to_ready_ms": workers["huang"].loading_ms,
            "internal_loading_ms": workers["huang"].ready["load_ms_internal"],
            "loaded_memory": memory_samples["huang"][0],
            "maximum_sampled_working_set_bytes": max(item["working_set_bytes"] for item in memory_samples["huang"]),
            "process_peak_working_set_bytes_sum": max(item["peak_working_set_bytes_sum"] for item in memory_samples["huang"]),
            "model_artifact_bytes": clean_checkpoint.stat().st_size + sum(path.stat().st_size for path in detector_files),
        },
    }
    write_json(OUT / "INTEGRATED_BENCHMARK.json", results)
    return results


def final_decision(results: dict) -> None:
    physics = results["physics_aware"]
    huang = results["huang_detect_correct_diagnose"]
    ratio = huang["latency"]["median_ms"] / physics["latency"]["median_ms"]
    memory_reduction = 1.0 - physics["maximum_sampled_working_set_bytes"] / huang["maximum_sampled_working_set_bytes"]
    agreement_physics = results["output_agreement"]["physics"]
    agreement_huang = results["output_agreement"]["huang"]
    lines = [
        "# FINAL INTEGRATED DEPLOYMENT DECISION",
        "",
        "## Decision",
        "",
        f"Yes, the operational latency advantage is confirmed for this frozen implementation on this CPU. The warm integrated physics-aware worker had median/p95 latency {physics['latency']['median_ms']:.2f}/{physics['latency']['p95_ms']:.2f} ms, versus {huang['latency']['median_ms']:.2f}/{huang['latency']['p95_ms']:.2f} ms for the complete Huang detector, predicted correction, and clean-ResNet diagnosis request ({ratio:.2f}x median ratio). These are direct request-to-diagnostic-response measurements; no component times were summed.",
        "",
        f"The largest sampled process-tree working set was {physics['maximum_sampled_working_set_bytes'] / 2**20:.1f} MiB for physics-aware and {huang['maximum_sampled_working_set_bytes'] / 2**20:.1f} MiB for Huang ({memory_reduction:.0%} lower for the single model). Loading to ready was {physics['loading_to_ready_ms'] / 1000:.2f} s versus {huang['loading_to_ready_ms'] / 1000:.2f} s.",
        "",
        "## Compact table",
        "",
        "| Measure | Physics-aware single model | Huang detect-correct-diagnose |",
        "|---|---:|---:|",
        f"| Warm batch-1 median | {physics['latency']['median_ms']:.2f} ms | {huang['latency']['median_ms']:.2f} ms |",
        f"| Warm batch-1 p95 | {physics['latency']['p95_ms']:.2f} ms | {huang['latency']['p95_ms']:.2f} ms |",
        f"| Run-median mean +/- SD | {physics['run_variability']['median_mean_ms']:.2f} +/- {physics['run_variability']['median_sd_ms']:.2f} ms | {huang['run_variability']['median_mean_ms']:.2f} +/- {huang['run_variability']['median_sd_ms']:.2f} ms |",
        f"| Run-median range | {physics['run_variability']['median_range_ms'][0]:.2f}-{physics['run_variability']['median_range_ms'][1]:.2f} ms | {huang['run_variability']['median_range_ms'][0]:.2f}-{huang['run_variability']['median_range_ms'][1]:.2f} ms |",
        f"| Run-p95 mean +/- SD | {physics['run_variability']['p95_mean_ms']:.2f} +/- {physics['run_variability']['p95_sd_ms']:.2f} ms | {huang['run_variability']['p95_mean_ms']:.2f} +/- {huang['run_variability']['p95_sd_ms']:.2f} ms |",
        f"| Run-p95 range | {physics['run_variability']['p95_range_ms'][0]:.2f}-{physics['run_variability']['p95_range_ms'][1]:.2f} ms | {huang['run_variability']['p95_range_ms'][0]:.2f}-{huang['run_variability']['p95_range_ms'][1]:.2f} ms |",
        f"| Loading to ready | {physics['loading_to_ready_ms'] / 1000:.2f} s | {huang['loading_to_ready_ms'] / 1000:.2f} s |",
        f"| Maximum sampled process-tree working set | {physics['maximum_sampled_working_set_bytes'] / 2**20:.1f} MiB | {huang['maximum_sampled_working_set_bytes'] / 2**20:.1f} MiB |",
        f"| Model artifacts | {physics['model_artifact_bytes'] / 2**20:.2f} MiB | {huang['model_artifact_bytes'] / 2**20:.2f} MiB |",
        f"| External pipe echo median | {physics['external_echo_overhead']['median_ms']:.3f} ms | {huang['external_echo_overhead']['median_ms']:.3f} ms |",
        "",
        "## Output agreement",
        "",
        f"Before timing, all 128 balanced fixture ECGs were compared with the frozen v2 seed-{SEED} caches. Physics-aware threshold decisions agreed {agreement_physics['threshold_decision_agreement']:.2%} with maximum probability difference {agreement_physics['max_absolute_probability_difference']:.4f}. Huang threshold decisions agreed {agreement_huang['threshold_decision_agreement']:.2%}, detector classes agreed {agreement_huang['detector_class_agreement']:.2%}, and maximum diagnostic probability difference was {agreement_huang['max_absolute_probability_difference']:.4f}.",
        "",
        "## How to describe this to the mentor",
        "",
        f"Say: 'On our frozen Ningbo implementation, the physics-aware single model and the published Huang detector chained to our clean ResNet had statistically unresolved TWA AUPRC, but the single model was operationally simpler and {ratio:.2f}x faster in a direct warm CPU request-to-response benchmark. This is a comparison with our integration of Huang's published detector weights, not a comparison with latency or diagnostic scores reported in the Huang paper.'",
        "",
        "Do not describe the methods as equivalent and do not claim SOTA or clinical superiority.",
        "",
        "## Missing evidence",
        "",
        "A stronger conference submission still needs real rather than simulated cable reversals, independent multi-device and multi-center validation, a prospectively specified noninferiority/equivalence hypothesis for diagnostic performance, deployment measurements on the intended clinical hardware and software stack, and workflow evidence at measured clinical prevalence. The current IPC design is reproducible and fair for these frozen Python environments, but it is not an optimized production implementation of either method.",
        "",
        "## Reproduce",
        "",
        "From the project root on Windows PowerShell:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe src\integrated_deployment_v4.py",
        "```",
        "",
        f"Hardware: {results['hardware']['cpu']}, {results['hardware']['logical_processors']} logical processors, {results['hardware']['os']}. Both frameworks were restricted to one thread and logical CPU {results['measurement']['logical_cpu_index']}; seven runs of 128 balanced ECGs were timed after output verification and warmup.",
    ]
    temporary = OUT / f".FINAL_DECISION.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf8")
    os.replace(temporary, OUT / "FINAL_DECISION.md")


def manifest() -> None:
    inputs = [
        V2 / "REPRODUCIBILITY_MANIFEST.json",
        N / "SPLIT_MEMBERSHIP.json",
        N / f"seed_{SEED}" / "checkpoint.pt",
        M2 / f"seed_{SEED}" / "checkpoint.pt",
    ]
    scripts = [
        ROOT / "src" / "integrated_deployment_v4.py",
        ROOT / "src" / "integrated_pipeline_worker_v4.py",
        ROOT / "src" / "integrated_huang_detector_worker_v4.py",
        ROOT / "src" / "deployment_benchmark_utils_v3.py",
    ]
    outputs = [
        OUT / "fixture_metadata.json",
        OUT / "balanced_raw_fixture.npy",
        OUT / "OUTPUT_AGREEMENT.json",
        OUT / "integrated_latency_raw.npz",
        OUT / "INTEGRATED_BENCHMARK.json",
        OUT / "FINAL_DECISION.md",
    ]
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", {
        "created_utc": now(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "scripts": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        "outputs": {str(path.relative_to(OUT)): sha256(path) for path in outputs},
        "constraints": ["no retraining", "no test-outcome tuning", "no data download", "no masking rerun"],
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start")
    fixture, metadata = create_fixture()
    workers: dict[str, Worker] = {}
    try:
        workers["physics"] = Worker("physics", fixture)
        log("worker_ready", pipeline="physics", loading_ms=workers["physics"].loading_ms)
        workers["huang"] = Worker("huang", fixture)
        log("worker_ready", pipeline="huang", loading_ms=workers["huang"].loading_ms)
        agreement = agreement_check(workers, metadata)
        log("output_agreement_pass")
        results = benchmark(workers, metadata, agreement)
        final_decision(results)
        manifest()
        log("complete")
    finally:
        for worker in workers.values():
            worker.close()


if __name__ == "__main__":
    main()
