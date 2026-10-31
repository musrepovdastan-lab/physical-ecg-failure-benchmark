"""CPU batch-1 latency for the frozen Huang detector component."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np

from deployment_benchmark_utils_v3 import pin_to_first_available_logical_cpu, process_memory, total_physical_memory_bytes
from huang_detector_infer_v2 import apply_condition, load_full_display, load_model, preprocess_one, records


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")


def fixture(count: int) -> tuple[list[dict], list[str], np.ndarray]:
    all_rows = records(None)
    indices = np.linspace(0, len(all_rows) - 1, count, dtype=int)
    selected = [all_rows[index] for index in indices]
    conditions = [CONDITIONS[index % len(CONDITIONS)] for index in range(count)]
    waveforms = np.asarray(
        [apply_condition(load_full_display(row), condition) for row, condition in zip(selected, conditions)],
        dtype="float32",
    )
    return selected, conditions, waveforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    args = parser.parse_args()

    cpu_index = pin_to_first_available_logical_cpu()
    rows, conditions, waveforms = fixture(args.count)
    baseline_memory = process_memory()
    load_started = time.perf_counter_ns()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model, checkpoint, tensorflow_version = load_model()
    load_ms = (time.perf_counter_ns() - load_started) / 1e6

    for index in range(args.warmup):
        detector_input = preprocess_one(waveforms[index % len(waveforms)])[None]
        _ = model(detector_input, training=False).numpy().argmax(1)

    latencies_ms = np.empty(args.count, dtype="float64")
    predicted_classes = np.empty(args.count, dtype="int64")
    for index, waveform in enumerate(waveforms):
        started = time.perf_counter_ns()
        detector_input = preprocess_one(waveform)[None]
        probabilities = model(detector_input, training=False).numpy()
        predicted_classes[index] = int(probabilities.argmax(1)[0])
        latencies_ms[index] = (time.perf_counter_ns() - started) / 1e6

    final_memory = process_memory()
    parameter_bytes = int(sum(np.asarray(variable.numpy()).nbytes for variable in model.variables))
    record_text = "\n".join(row["record_id"] for row in rows) + "\n"
    result = {
        "component": "published_huang_detector",
        "count": args.count,
        "warmup": args.warmup,
        "batch_size": 1,
        "cpu_logical_index": cpu_index,
        "cpu_threads": 1,
        "disk_io_timed": False,
        "timed_operations": "Butterworth 0.5-35 Hz filtering, 500-to-120 Hz resampling, per-lead z-score, III/aVR/V6 selection, LDenseNet inference, argmax",
        "load_ms": load_ms,
        "latencies_ms": latencies_ms.tolist(),
        "predicted_classes": predicted_classes.tolist(),
        "conditions": conditions,
        "record_order_sha256": hashlib.sha256(record_text.encode("utf8")).hexdigest().upper(),
        "checkpoint": checkpoint,
        "tensorflow_version": tensorflow_version,
        "model_parameter_bytes": parameter_bytes,
        "memory_before_framework_and_model": baseline_memory,
        "memory_after_inference": final_memory,
        "incremental_peak_working_set_bytes": final_memory["peak_working_set_bytes"] - baseline_memory["working_set_bytes"],
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_processors": os.cpu_count(),
            "total_physical_memory_bytes": total_physical_memory_bytes(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
