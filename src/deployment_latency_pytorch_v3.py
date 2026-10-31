"""CPU batch-1 latency for frozen ResNet deployment components."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from deployment_benchmark_utils_v3 import pin_to_first_available_logical_cpu, process_memory, total_physical_memory_bytes
from huang_detector_infer_v2 import apply_condition, load_full_display, records


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
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


def correct_display(x: np.ndarray, predicted_class: int) -> np.ndarray:
    if predicted_class == 0:
        return x
    condition = {1: "RA_LA_reversal", 2: "RA_LL_reversal", 3: "LA_LL_reversal"}[int(predicted_class)]
    return apply_condition(x, condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("physics", "huang_diagnostic"), required=True)
    parser.add_argument("--detector-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    args = parser.parse_args()
    if args.pipeline == "huang_diagnostic" and args.detector_json is None:
        parser.error("--detector-json is required for huang_diagnostic")

    cpu_index = pin_to_first_available_logical_cpu()
    rows, conditions, waveforms = fixture(args.count)
    detector_classes = np.zeros(args.count, dtype=int)
    if args.detector_json:
        detector_result = json.loads(args.detector_json.read_text(encoding="utf8"))
        if detector_result["conditions"] != conditions or detector_result["count"] != args.count:
            raise RuntimeError("detector latency fixture mismatch")
        detector_classes = np.asarray(detector_result["predicted_classes"], dtype=int)

    baseline_memory = process_memory()
    load_started = time.perf_counter_ns()
    import torch
    from src.chapman_phase1 import ECGResNetSmall

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = M2 if args.pipeline == "physics" else N
    seed = 104729
    model = ECGResNetSmall().cpu().eval()
    checkpoint_path = base / f"seed_{seed}" / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location="cpu")["state_dict"])
    with np.load(base / f"seed_{seed}" / "preprocessing.npz", allow_pickle=False) as preprocessing:
        mean, std = preprocessing["mean"].copy(), preprocessing["std"].copy()
    load_ms = (time.perf_counter_ns() - load_started) / 1e6

    def infer(index: int) -> None:
        x = waveforms[index]
        if args.pipeline == "huang_diagnostic":
            x = correct_display(x, int(detector_classes[index]))
        x = x[:, ::5].astype("float32", copy=False)
        x = ((x - mean[:, None]) / std[:, None]).astype("float32")
        with torch.inference_mode():
            _ = torch.sigmoid(model(torch.from_numpy(x[None]))).numpy()

    for index in range(args.warmup):
        infer(index % len(waveforms))

    latencies_ms = np.empty(args.count, dtype="float64")
    for index in range(args.count):
        started = time.perf_counter_ns()
        infer(index)
        latencies_ms[index] = (time.perf_counter_ns() - started) / 1e6

    final_memory = process_memory()
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    record_text = "\n".join(row["record_id"] for row in rows) + "\n"
    result = {
        "component": args.pipeline,
        "count": args.count,
        "warmup": args.warmup,
        "batch_size": 1,
        "seed": seed,
        "cpu_logical_index": cpu_index,
        "cpu_threads": 1,
        "disk_io_timed": False,
        "timed_operations": (
            "500-to-100 Hz deterministic downsampling, frozen normalization, physics-aware ResNet inference"
            if args.pipeline == "physics" else
            "predicted inverse terminal correction on raw 500 Hz display, 500-to-100 Hz deterministic downsampling, frozen normalization, clean ResNet inference"
        ),
        "load_ms": load_ms,
        "latencies_ms": latencies_ms.tolist(),
        "conditions": conditions,
        "record_order_sha256": hashlib.sha256(record_text.encode("utf8")).hexdigest().upper(),
        "checkpoint": str(checkpoint_path),
        "model_parameter_bytes": parameter_bytes,
        "memory_before_framework_and_model": baseline_memory,
        "memory_after_inference": final_memory,
        "incremental_peak_working_set_bytes": final_memory["peak_working_set_bytes"] - baseline_memory["working_set_bytes"],
        "torch_version": torch.__version__,
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
