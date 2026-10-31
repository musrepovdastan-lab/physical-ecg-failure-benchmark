"""Persistent TensorFlow worker for the published Huang detector."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np

from deployment_benchmark_utils_v3 import pin_to_first_available_logical_cpu, process_memory
from huang_detector_infer_v2 import load_model, preprocess_one


def send(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    cpu_index = pin_to_first_available_logical_cpu()
    fixture = np.load(args.fixture, mmap_mode="r", allow_pickle=False)
    started = time.perf_counter_ns()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model, checkpoint, tensorflow_version = load_model()
    load_ms = (time.perf_counter_ns() - started) / 1e6
    send({
        "status": "ready",
        "pid": os.getpid(),
        "cpu_logical_index": cpu_index,
        "fixture_shape": list(fixture.shape),
        "load_ms_internal": load_ms,
        "checkpoint": checkpoint,
        "tensorflow_version": tensorflow_version,
        "memory": process_memory(),
    })

    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        if command == "predict":
            index = int(request["index"])
            detector_input = preprocess_one(np.asarray(fixture[index]))[None]
            probabilities = model(detector_input, training=False).numpy()[0]
            send({
                "status": "ok",
                "class": int(probabilities.argmax()),
                "probabilities": probabilities.astype(float).tolist(),
            })
        elif command == "memory":
            send({"status": "ok", "memory": process_memory()})
        elif command == "shutdown":
            send({"status": "bye"})
            return
        else:
            send({"status": "error", "message": f"unknown command: {command}"})


if __name__ == "__main__":
    main()
