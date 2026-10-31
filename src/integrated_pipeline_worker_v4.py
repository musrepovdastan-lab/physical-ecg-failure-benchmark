"""Persistent end-to-end CPU worker for a frozen deployable ECG pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from deployment_benchmark_utils_v3 import pin_to_first_available_logical_cpu, process_memory
from huang_detector_infer_v2 import apply_condition


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
N = ROOT / "outputs" / "ningbo_confirmation_v2"
M2 = ROOT / "outputs" / "mitigation_m2_ningbo_v1"
DETECTOR_PYTHON = ROOT / "outputs" / "published_detector_pipeline_v1" / "detector_env" / "Scripts" / "python.exe"
DETECTOR_WORKER = ROOT / "src" / "integrated_huang_detector_worker_v4.py"


def send(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def child_request(process: subprocess.Popen, value: dict) -> dict:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("detector worker pipes unavailable")
    process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("detector worker exited unexpectedly")
    response = json.loads(line)
    if response.get("status") not in ("ok", "ready", "bye"):
        raise RuntimeError(f"detector worker error: {response}")
    return response


def correct_display(x: np.ndarray, predicted_class: int) -> np.ndarray:
    if predicted_class == 0:
        return x
    condition = {1: "RA_LA_reversal", 2: "RA_LL_reversal", 3: "LA_LL_reversal"}[int(predicted_class)]
    return apply_condition(x, condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("physics", "huang"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--detector-stderr", type=Path)
    args = parser.parse_args()

    cpu_index = pin_to_first_available_logical_cpu()
    fixture = np.load(args.fixture, mmap_mode="r", allow_pickle=False)
    load_started = time.perf_counter_ns()
    import torch
    from src.chapman_phase1 import ECGResNetSmall

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = M2 if args.pipeline == "physics" else N
    checkpoint_path = base / "seed_104729" / "checkpoint.pt"
    model = ECGResNetSmall().cpu().eval()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location="cpu")["state_dict"])
    with np.load(base / "seed_104729" / "preprocessing.npz", allow_pickle=False) as preprocessing:
        mean, std = preprocessing["mean"].copy(), preprocessing["std"].copy()

    detector = None
    detector_ready = None
    detector_stderr_handle = None
    if args.pipeline == "huang":
        if args.detector_stderr is None:
            raise RuntimeError("--detector-stderr is required for Huang pipeline")
        args.detector_stderr.parent.mkdir(parents=True, exist_ok=True)
        detector_stderr_handle = args.detector_stderr.open("w", encoding="utf8")
        detector = subprocess.Popen(
            [str(DETECTOR_PYTHON), str(DETECTOR_WORKER), "--fixture", str(args.fixture)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=detector_stderr_handle,
            text=True,
            bufsize=1,
        )
        line = detector.stdout.readline() if detector.stdout is not None else ""
        if not line:
            raise RuntimeError("detector worker did not become ready")
        detector_ready = json.loads(line)
        if detector_ready.get("status") != "ready":
            raise RuntimeError(f"detector worker readiness failed: {detector_ready}")

    load_ms_internal = (time.perf_counter_ns() - load_started) / 1e6
    send({
        "status": "ready",
        "pipeline": args.pipeline,
        "pid": os.getpid(),
        "detector_pid": detector_ready["pid"] if detector_ready else None,
        "cpu_logical_index": cpu_index,
        "fixture_shape": list(fixture.shape),
        "load_ms_internal": load_ms_internal,
        "checkpoint": str(checkpoint_path),
        "torch_version": torch.__version__,
        "detector_ready": detector_ready,
        "memory": process_memory(),
    })

    def predict(index: int) -> dict:
        x = np.asarray(fixture[index])
        detector_class = None
        if args.pipeline == "huang":
            detector_result = child_request(detector, {"command": "predict", "index": index})
            detector_class = int(detector_result["class"])
            x = correct_display(x, detector_class)
        x = x[:, ::5].astype("float32", copy=False)
        x = ((x - mean[:, None]) / std[:, None]).astype("float32")
        with torch.inference_mode():
            prediction = torch.sigmoid(model(torch.from_numpy(x[None]))).numpy()[0]
        return {
            "status": "ok",
            "prediction": prediction.astype(float).tolist(),
            "detector_class": detector_class,
        }

    try:
        for line in sys.stdin:
            request = json.loads(line)
            command = request.get("command")
            if command == "predict":
                send(predict(int(request["index"])))
            elif command == "echo":
                send({"status": "ok", "nonce": request.get("nonce")})
            elif command == "memory":
                result = {"status": "ok", "pipeline": process_memory()}
                if detector is not None:
                    result["detector"] = child_request(detector, {"command": "memory"})["memory"]
                send(result)
            elif command == "shutdown":
                if detector is not None:
                    child_request(detector, {"command": "shutdown"})
                send({"status": "bye"})
                return
            else:
                send({"status": "error", "message": f"unknown command: {command}"})
    finally:
        if detector is not None and detector.poll() is None:
            detector.terminate()
            detector.wait(timeout=10)
        if detector_stderr_handle is not None:
            detector_stderr_handle.close()


if __name__ == "__main__":
    main()
