"""Exploratory, resumable frozen-backbone ECGFounder comparison, v2 adapter.

The authors' Net1D implementation is imported unchanged from the pinned source archive.
The preprocessing below copies the operations and order in official util.py and
dataset.py, avoiding util.py's unrelated plotting/HDF5 imports.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v2"
CFG = ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json"
OFFICIAL = ROOT / "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23"
CHECKPOINT = ROOT / "outputs/ecgfounder_baseline_v1/official/12_lead_ECGFounder.pth"
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(ROOT))
from net1d import Net1D  # noqa: E402
from src.ecg_physics import derive_display, named_reversal  # noqa: E402

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LABELS = ("TWA", "SB", "STach")
CONDITIONS = ("clean", "RA_LL_reversal")
BATCH = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def status(stage: str, **kw):
    obj = {"stage": stage, "pid": os.getpid(), "updated_utc": now(), **kw}
    atomic_json(OUT / "WORKFLOW_STATUS.json", obj)
    print(json.dumps(obj, sort_keys=True), flush=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        code = ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


class SingleRun:
    def __enter__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "ecgfounder_baseline_v2.lock"
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            if pid_alive(int(obj["pid"])):
                raise RuntimeError(f"duplicate ECGFounder job already running (PID {obj['pid']})")
            self.path.unlink()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "protocol_sha256": sha(CFG), "started_utc": now()}, f)
        return self

    def __exit__(self, *_):
        if self.path.exists():
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            if obj.get("pid") == os.getpid():
                self.path.unlink()


def config() -> dict:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    if cfg["status"] != "FROZEN_AFTER_INPUT_ADAPTER_PREFLIGHT_BEFORE_NEW_OUTCOMES":
        raise RuntimeError("protocol is not frozen")
    if sha(Path(__file__)) != cfg["runner_sha256"]:
        raise RuntimeError("runner changed after protocol freeze")
    audit = OUT / "PRE_RUN_AUDIT.json"
    if sha(audit) != cfg["pre_run_audit_sha256"]:
        raise RuntimeError("pre-run audit changed")
    for rel, expected in cfg["file_bindings"].items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"source binding changed: {rel}")
    return cfg


def split(dataset: str, partition: str):
    if dataset == "Chapman":
        obj = json.loads((ROOT / "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json").read_text())
        return [r for r in obj["rows"] if r["split"] == partition]
    obj = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text())
    return [r for r in obj["rows"] if r["split"] == partition]


def label_matrix(dataset: str, partition: str):
    records = split(dataset, partition)
    if dataset == "Chapman":
        return np.asarray([[int(c) for c in r["bits_twa_sb_stach"]] for r in records], np.int8)
    return np.asarray([r["labels"] for r in records], np.int8)


@lru_cache(maxsize=1)
def ningbo_canonical() -> dict:
    path = ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    result = {r["record_id"]: r for r in records}
    if len(result) != len(records):
        raise RuntimeError("duplicate Ningbo canonical record ID")
    return result


@lru_cache(maxsize=1)
def ningbo_split_rows() -> dict:
    path = ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"
    records = json.loads(path.read_text(encoding="utf-8"))["rows"]
    result = {r["record_id"]: r for r in records}
    if len(result) != len(records):
        raise RuntimeError("duplicate Ningbo split record ID")
    return result


def canonical_path(relative: str, suffix: str) -> Path:
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or "\\" in relative or ".." in rel.parts or
            rel.parts[:2] != ("training", "ningbo") or rel.suffix != suffix):
        raise RuntimeError(f"invalid canonical waveform path: {relative}")
    path = ROOT / "data/challenge2021_v103/ningbo" / Path(*rel.parts[2:])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record_path(dataset: str, record_id: str) -> Path:
    if dataset == "Chapman":
        path = ROOT / "data/challenge2021_v103/chapman_shaoxing" / record_id
        if not path.with_suffix(".hea").is_file() or not path.with_suffix(".mat").is_file():
            raise FileNotFoundError(path)
        return path
    canonical = ningbo_canonical().get(record_id)
    split_row = ningbo_split_rows().get(record_id)
    if canonical is None or split_row is None or canonical["record_id"] != record_id:
        raise RuntimeError(f"Ningbo canonical/split ID mismatch: {record_id}")
    if canonical["waveform_sha256"] != split_row["waveform_sha256"]:
        raise RuntimeError(f"Ningbo split waveform hash mismatch: {record_id}")
    header = canonical_path(canonical["header_path"], ".hea")
    waveform = canonical_path(canonical["waveform_path"], ".mat")
    if waveform.stem != record_id.split("/")[-1]:
        raise RuntimeError(f"Ningbo header/waveform identity mismatch: {record_id}")
    lines = header.read_text(encoding="utf-8").splitlines()
    if lines[0].split()[0] != header.stem or any(line.split()[0] != waveform.name for line in lines[1:13]):
        raise RuntimeError(f"Ningbo header does not reference canonical waveform: {record_id}")
    return header.with_suffix("")


def physical_signal(dataset: str, record_id: str) -> np.ndarray:
    x, metadata = wfdb.rdsamp(str(record_path(dataset, record_id)))
    if metadata["fs"] != 500 or x.shape != (5000, 12) or tuple(metadata["sig_name"]) != LEADS:
        raise RuntimeError(f"waveform schema mismatch: {dataset}/{record_id}")
    if np.isinf(x).any():
        raise RuntimeError(f"infinite waveform: {dataset}/{record_id}")
    # Official dataset.py applies nan_to_num before filtering. Do so on the
    # physical 12-lead signal before either the filter or RA-LL permutation.
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    if not np.isfinite(x).all():
        raise RuntimeError(f"unexpected nonfinite waveform: {dataset}/{record_id}")
    return np.ascontiguousarray(x.T, dtype=np.float64)


def reverse_ra_ll(x: np.ndarray) -> np.ndarray:
    independent = np.concatenate((x[:2], x[6:]), axis=0)
    return derive_display(named_reversal(independent, "RA_LL_reversal"))


def official_filter(x: np.ndarray) -> np.ndarray:
    # Exact sequence and coefficients from pinned official util.py::filter_bandpass.
    b, a = iirnotch(50, 30, 500)
    filtered = np.zeros_like(x)
    for c in range(x.shape[0]):
        filtered[c] = filtfilt(b, a, x[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=500)
    for c in range(x.shape[0]):
        filtered[c] = filtfilt(b, a, filtered[c])
    baseline = np.zeros_like(filtered)
    for c in range(filtered.shape[0]):
        baseline[c] = medfilt(filtered[c], kernel_size=201)
    return filtered - baseline


def preprocess(x: np.ndarray) -> np.ndarray:
    filtered = official_filter(x)
    # Official dataset.py applies one scalar mean and std to the full 12 x 5000 array.
    z = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    if z.shape != (12, 5000) or not np.isfinite(z).all():
        raise RuntimeError("preprocessing produced an invalid input")
    return z.astype(np.float32)


def model():
    net = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
        stride=2, groups_width=16, verbose=False, use_bn=False,
        use_do=False, n_classes=150, return_features=True,
    )
    # The official checkpoint includes optimizer data with NumPy scalars and requires
    # weights_only=False; its exact SHA-256 is checked against the frozen protocol.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    keys = net.load_state_dict(checkpoint["state_dict"], strict=True)
    if keys.missing_keys or keys.unexpected_keys:
        raise RuntimeError("checkpoint architecture mismatch")
    for p in net.parameters():
        p.requires_grad_(False)
    return net.cuda().eval()


def infer(net, batch: np.ndarray):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits, features = net(torch.from_numpy(batch).cuda(non_blocking=True))
    return features.float().cpu().numpy(), logits.float().cpu().numpy()


def smoke():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    record = split("Chapman", "train")[0]
    raw = physical_signal("Chapman", record["record_id"])
    rev = reverse_ra_ll(raw)
    if not np.allclose(rev[6:], raw[6:], atol=1e-10):
        raise RuntimeError("precordial leads changed under RA–LL permutation")
    clean = preprocess(raw)
    altered = preprocess(rev)
    net = model()
    features, logits = infer(net, np.stack((clean, altered)))
    if features.shape != (2, 1024) or logits.shape != (2, 150) or not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise RuntimeError("inference smoke failed")
    for known in ("g1/JS10767", "g11/JS21646", "g13/JS23074"):
        raw_known = physical_signal("Ningbo", known)
        changed_known = reverse_ra_ll(raw_known)
        if not np.allclose(raw_known[6:], changed_known[6:], rtol=0, atol=1e-12):
            raise RuntimeError(f"Ningbo chest lead change: {known}")
        f_known, z_known = infer(net, np.stack((preprocess(raw_known), preprocess(changed_known))))
        if f_known.shape != (2, 1024) or z_known.shape != (2, 150) or not np.isfinite(z_known).all():
            raise RuntimeError(f"Ningbo smoke failed: {known}")
    report = {
        "status": "PASS", "record_id": record["record_id"], "partition": "train",
        "preprocessing_shape": list(clean.shape), "embedding_shape": list(features.shape),
        "logit_shape": list(logits.shape), "input_finite": True,
        "known_ningbo_records": ["g1/JS10767", "g11/JS21646", "g13/JS23074"],
        "source_commit": "04edac702b61c91face519774ddcc0cd712fef23",
        "checkpoint_sha256": sha(CHECKPOINT), "protocol_sha256": sha(CFG),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "completed_utc": now(),
    }
    atomic_json(OUT / "SMOKE_REPORT.json", report)
    status("SMOKE_PASS", records_completed=4, total_records=4)


def cache_paths(dataset: str, partition: str, condition: str):
    root = OUT / "features"
    stem = f"{dataset.lower()}_{partition}_{condition}"
    return root / f"{stem}_features.npy", root / f"{stem}_logits.npy", root / f"{stem}_meta.json"


def extract(net, dataset: str, partition: str, condition: str):
    records = split(dataset, partition)
    fp, lp, mp = cache_paths(dataset, partition, condition)
    fp.parent.mkdir(parents=True, exist_ok=True)
    ids = [r["record_id"] for r in records]
    ids_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    binding = {"dataset": dataset, "partition": partition, "condition": condition,
               "record_ids_sha256": ids_sha, "protocol_sha256": sha(CFG),
               "checkpoint_sha256": sha(CHECKPOINT), "total_records": len(records)}
    if mp.exists():
        metadata = json.loads(mp.read_text(encoding="utf-8"))
        if any(metadata.get(k) != v for k, v in binding.items()):
            raise RuntimeError(f"stale feature cache binding: {mp}")
        if not fp.is_file() or not lp.is_file():
            raise RuntimeError(f"incomplete cache pair: {mp}")
        features = np.load(fp, mmap_mode="r+")
        logits = np.load(lp, mmap_mode="r+")
        if features.shape != (len(records), 1024) or logits.shape != (len(records), 150):
            raise RuntimeError(f"feature cache shape mismatch: {mp}")
        start = int(metadata["completed_records"])
        if start < 0 or start > len(records):
            raise RuntimeError(f"feature cache count mismatch: {mp}")
        if start == len(records):
            if metadata.get("features_sha256") != sha(fp) or metadata.get("logits_sha256") != sha(lp):
                raise RuntimeError(f"completed feature cache hash mismatch: {mp}")
            return np.load(fp, mmap_mode="r"), np.load(lp, mmap_mode="r")
    else:
        if fp.exists() or lp.exists():
            raise RuntimeError(f"unbound feature cache exists: {fp}")
        features = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float32, shape=(len(records), 1024))
        logits = np.lib.format.open_memmap(lp, mode="w+", dtype=np.float32, shape=(len(records), 150))
        start = 0
        atomic_json(mp, {**binding, "completed_records": 0})
    for index in range(start, len(records), BATCH):
        block = records[index:index + BATCH]
        inputs = []
        for r in block:
            raw = physical_signal(dataset, r["record_id"])
            if condition == "RA_LL_reversal":
                raw = reverse_ra_ll(raw)
            inputs.append(preprocess(raw))
        f, z = infer(net, np.stack(inputs))
        if not np.isfinite(f).all() or not np.isfinite(z).all():
            raise RuntimeError("nonfinite model output")
        end = index + len(block)
        features[index:end] = f
        logits[index:end] = z
        features.flush()
        logits.flush()
        atomic_json(mp, {**binding, "completed_records": end})
        if end % 160 < BATCH or end == len(records):
            status("EXTRACTING", dataset=dataset, partition=partition, condition=condition,
                   records_completed=end, total_records=len(records))
    atomic_json(mp, {**binding, "completed_records": len(records),
                     "features_sha256": sha(fp), "logits_sha256": sha(lp)})
    return np.load(fp, mmap_mode="r"), np.load(lp, mmap_mode="r")


def train_head(dataset: str):
    head_path = OUT / f"{dataset.lower()}_head.npz"
    freeze_path = OUT / f"{dataset.lower()}_validation_freeze.json"
    train_features = np.asarray(extract_result(dataset, "train", "clean")[0])
    val_features = np.asarray(extract_result(dataset, "validation", "clean")[0])
    ytrain, yval = label_matrix(dataset, "train"), label_matrix(dataset, "validation")
    mean = train_features.mean(axis=0, dtype=np.float64)
    std = train_features.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    xtrain = ((train_features - mean) / std).astype(np.float32)
    xval = ((val_features - mean) / std).astype(np.float32)
    del train_features, val_features
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != sha(CFG) or frozen.get("head_sha256") != sha(head_path):
            raise RuntimeError("head freeze binding mismatch")
        head = np.load(head_path, allow_pickle=False)
        return head, frozen
    if head_path.exists():
        raise RuntimeError("unfrozen head cache exists")
    coefficients, intercepts, choices, thresholds, val_ap = [], [], [], [], []
    for j, label in enumerate(LABELS):
        candidates = []
        for c in (0.01, 0.1, 1.0):
            lr = LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=300,
                                    tol=1e-4, class_weight=None, random_state=20260920)
            lr.fit(xtrain, ytrain[:, j])
            p = lr.predict_proba(xval)[:, 1]
            score = float(average_precision_score(yval[:, j], p))
            candidates.append((score, c, lr, p))
            status("VALIDATION_SELECTION", dataset=dataset, label=label, C=c, validation_auprc=score)
        score, c, lr, p = max(candidates, key=lambda v: (v[0], -v[1]))
        coefficients.append(lr.coef_[0].astype(np.float64))
        intercepts.append(float(lr.intercept_[0]))
        choices.append(c)
        thresholds.append(float(np.quantile(p[yval[:, j] == 0], 0.9)))
        val_ap.append(score)
    np.savez_compressed(head_path, mean=mean, std=std, coef=np.stack(coefficients),
                        intercept=np.asarray(intercepts), C=np.asarray(choices),
                        thresholds=np.asarray(thresholds), validation_auprc=np.asarray(val_ap))
    frozen = {"dataset": dataset, "protocol_sha256": sha(CFG), "head_sha256": sha(head_path),
              "selected_C": dict(zip(LABELS, choices)), "validation_auprc": dict(zip(LABELS, val_ap)),
              "validation_threshold": dict(zip(LABELS, thresholds)),
              "test_inputs_audited_before_protocol_freeze": True,
              "test_model_outcomes_evaluated_before_validation_freeze": False,
              "completed_utc": now()}
    atomic_json(freeze_path, frozen)
    status("VALIDATION_FROZEN", dataset=dataset, records_completed=len(ytrain) + len(yval),
           total_records=len(ytrain) + len(yval))
    return np.load(head_path, allow_pickle=False), frozen


def extract_result(dataset: str, partition: str, condition: str):
    fp, lp, mp = cache_paths(dataset, partition, condition)
    if not mp.exists():
        raise RuntimeError(f"missing feature cache: {mp}")
    metadata = json.loads(mp.read_text(encoding="utf-8"))
    if metadata.get("completed_records") != metadata.get("total_records"):
        raise RuntimeError(f"incomplete feature cache: {mp}")
    if metadata.get("features_sha256") != sha(fp) or metadata.get("logits_sha256") != sha(lp):
        raise RuntimeError(f"feature cache hash mismatch: {mp}")
    return np.load(fp, mmap_mode="r"), np.load(lp, mmap_mode="r")


def run(dataset: str):
    config()
    if not (OUT / "SMOKE_REPORT.json").exists():
        smoke()
    else:
        smoke_report = json.loads((OUT / "SMOKE_REPORT.json").read_text(encoding="utf-8"))
        if smoke_report.get("status") != "PASS" or smoke_report.get("protocol_sha256") != sha(CFG):
            raise RuntimeError("smoke report binding mismatch")
    net = model()
    for partition in ("train", "validation"):
        extract(net, dataset, partition, "clean")
    train_head(dataset)
    for condition in CONDITIONS:
        extract(net, dataset, "test", condition)
    status("FEATURES_COMPLETE", dataset=dataset, records_completed=len(split(dataset, "test")) * 2,
           total_records=len(split(dataset, "test")) * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "run"))
    parser.add_argument("--dataset", choices=("Chapman", "Ningbo"))
    args = parser.parse_args()
    config()
    with SingleRun():
        if args.command == "smoke":
            smoke()
        else:
            if not args.dataset:
                parser.error("--dataset required for run")
            run(args.dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("STOP_TECHNICAL_BLOCKER", error=repr(exc))
        raise
