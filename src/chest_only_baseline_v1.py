"""Frozen six-run chest-only ResNet baseline and paired comparator analysis."""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, os, random, shutil, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/chest_only_baseline_v1"
PROTO = OUT / "FROZEN_CHEST_ONLY_PROTOCOL.json"
SEEDS = (104729, 130363, 155921)
DATASETS = ("Chapman", "Ningbo")
MODELS = ("clean_12lead", "physics_aware_12lead", "chest_only")
LABELS = ("TWA", "SB", "STach")
CHEST = np.arange(6, 12)
CONDITIONS = ("clean", "RA_LL_reversal", "RA_LA_reversal", "LA_LL_reversal")
DATA12 = ROOT / "data/challenge2021_v103/chapman_shaoxing"
V4 = ROOT / "outputs/phase1_chapman_seed1_v4"
V7 = ROOT / "outputs/phase2_chapman_multiseed_v7"
M1 = ROOT / "outputs/mitigation_m1_v1/physics_aware"
N = ROOT / "outputs/ningbo_confirmation_v2"
NBASE = ROOT / "outputs/mask_semantics_recovery_v2/baseline_v2"
NM2 = ROOT / "outputs/mask_semantics_recovery_v2/physics_aware_m2"
NM2TRAIN = ROOT / "outputs/mitigation_m2_ningbo_v1"
sys.path.insert(0, str(ROOT))
from src.chapman_stage23 import parse_header
from src.chapman_phase1 import ECGResNetSmall as ResNet12, raw as chapman_raw
from src.ecg_physics import derive_display, named_reversal
from src import ningbo_confirmation_v1 as ningbo


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def canonical_hash(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".pt")
    torch.save(value, tmp); os.replace(tmp, path)
def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


class Block(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(a, b, 7, stride, 3, bias=False), nn.GroupNorm(8, b), nn.GELU(),
                               nn.Conv1d(b, b, 7, 1, 3, bias=False), nn.GroupNorm(8, b))
        self.z = nn.Identity() if a == b and stride == 1 else nn.Sequential(nn.Conv1d(a, b, 1, stride, bias=False), nn.GroupNorm(8, b))
        self.g = nn.GELU()
    def forward(self, x): return self.g(self.a(x) + self.z(x))


class ChestOnlyResNetSmall(nn.Module):
    """ECGResNet-small with only the input stem changed from 12 to 6 channels."""
    def __init__(self):
        super().__init__()
        widths = [32, 64, 128, 256]
        self.s = nn.Sequential(nn.Conv1d(6, widths[0], 7, 1, 3, bias=False), nn.GroupNorm(8, widths[0]), nn.GELU())
        self.b = nn.ModuleList([Block(widths[i - 1] if i else widths[0], width,
                                      1 if i == 0 else 2) for i, width in enumerate(widths)])
        self.h = nn.Sequential(nn.Dropout(.2), nn.Linear(widths[-1], 3))
    def forward(self, x):
        x = self.s(x)
        for block in self.b: x = block(x)
        return self.h(x.mean(-1))


def split_records(dataset):
    if dataset == "Chapman":
        rows = json.loads((V4 / "SPLIT_MEMBERSHIP_v4.json").read_text(encoding="utf-8"))["rows"]
        return {s: [r["record_id"] for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    rows = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    canonical = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    byid = {r["record_id"]: r for r in canonical}
    return {s: [byid[r["record_id"]] for r in rows if r["split"] == s] for s in ("train", "validation", "test")}


def record_id(dataset, record): return record if dataset == "Chapman" else record["record_id"]
def record_labels(dataset, record):
    if dataset == "Chapman":
        codes = parse_header(DATA12 / f"{record}.hea")[1]
        return np.asarray([int(c in codes) for c in ("164934002", "426177001", "427084000")], dtype=np.float32)
    return np.asarray(record["labels"], dtype=np.float32)
def raw12(dataset, record):
    return chapman_raw(record) if dataset == "Chapman" else ningbo.raw(record).astype(np.float32)
def physical_reversal_chest(x, name):
    independent = np.concatenate((x[:2], x[6:]), axis=0)
    return derive_display(named_reversal(independent, name)).astype(np.float32)[6:12]
def normalized_chest(dataset, record, mean, std, condition="clean"):
    x = raw12(dataset, record)
    chest = x[CHEST] if condition == "clean" else physical_reversal_chest(x, condition)
    return ((chest - mean[:, None]) / std[:, None]).astype(np.float32)


class RecordDataset(Dataset):
    def __init__(self, dataset, records, mean, std): self.dataset, self.records, self.mean, self.std = dataset, records, mean, std
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        return normalized_chest(self.dataset, r, self.mean, self.std), record_labels(self.dataset, r)


def loader(dataset, records, mean, std, train=False):
    return DataLoader(RecordDataset(dataset, records, mean, std), batch_size=64 if train else 128,
                      shuffle=train, num_workers=2, pin_memory=True, persistent_workers=True)
def stats(dataset, records):
    total = np.zeros(6, np.float64); squares = total.copy(); n = 0
    for i, rec in enumerate(records, 1):
        x = raw12(dataset, rec)[CHEST].astype(np.float64)
        total += x.sum(1); squares += (x * x).sum(1); n += x.shape[1]
        if i % 1000 == 0: print(f"{dataset}: train statistics {i}/{len(records)}", flush=True)
    mean = total / n
    return mean.astype(np.float32), np.sqrt(np.maximum(squares / n - mean * mean, 1e-8)).astype(np.float32)
def predict(model, dataset, records, mean, std):
    ys, ps = [], []; model.eval()
    with torch.no_grad():
        for x, y in loader(dataset, records, mean, std):
            with torch.autocast("cuda", dtype=torch.float16): p = torch.sigmoid(model(x.cuda(non_blocking=True))).float().cpu().numpy()
            ys.append(y.numpy()); ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


def ece15(y, p):
    value = 0.
    for k in range(15):
        lo, hi = k / 15, (k + 1) / 15
        take = (p >= lo) & ((p < hi) if k < 14 else (p <= hi))
        if take.any(): value += take.mean() * abs(float(y[take].mean()) - float(p[take].mean()))
    return float(value)
def metrics(y, p, thresholds):
    rows = []
    for i, label in enumerate(LABELS):
        call = p[:, i] >= thresholds[i]; pos = y[:, i] == 1; neg = ~pos
        rows.append({"endpoint": label, "n": len(y), "auprc": float(average_precision_score(y[:, i], p[:, i])),
                     "auroc": float(roc_auc_score(y[:, i], p[:, i])), "brier": float(brier_score_loss(y[:, i], p[:, i])),
                     "ece_15": ece15(y[:, i], p[:, i]), "threshold": float(thresholds[i]),
                     "sensitivity": float(call[pos].mean()), "specificity": float((~call[neg]).mean()),
                     "predicted_positive_fraction": float(call.mean())})
    return rows
def thresholds(y, p): return np.asarray([np.quantile(p[y[:, i] == 0, i], .90) for i in range(3)], dtype=np.float32)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
def unit_dir(dataset, seed): return OUT / "models" / dataset / f"seed_{seed}"
def config_hash(): return hashlib.sha256(b"chest_only_v1|input=V1-V6[6:12]|AdamW=0.001,0.0001|weighted_BCE_pos_cap20|batch64|max30|AMP_FP16|select_validation_TWA_AUPRC|patience5").hexdigest()


def comparator_spec(dataset, model, seed):
    if dataset == "Chapman" and model == "clean_12lead":
        if seed == 104729:
            d = V4; ck = d / "ecgresnet_small_seed104729_v4.pt"; pre = d / "preprocessing_v4.npz"
            val = d / "validation_predictions_v4.npz"; gate = json.loads((d / "BASELINE_ADEQUACY_v4.json").read_text())
            freeze = json.loads((d / "FAILURE_ANALYSIS_FREEZE_v4.json").read_text())
            if freeze.get("ecgresnet_small_seed104729_v4.pt") != sha(ck): raise RuntimeError("Chapman v4 checkpoint hash mismatch")
            gate["checkpoint_sha256"] = sha(ck)
            th = np.asarray(gate["thresholds_at_90pct_specificity"], np.float32); split_path = d / "SPLIT_MEMBERSHIP_v4.json"
        else:
            d = V7 / f"seed_{seed}"; ck = d / "checkpoint.pt"; pre = V4 / "preprocessing_v4.npz"
            val = d / "validation.npz"; gate = json.loads((d / "baseline.json").read_text())
            th = np.asarray(gate["thresholds"], np.float32); split_path = V4 / "SPLIT_MEMBERSHIP_v4.json"
        return {"checkpoint": ck, "preprocessing": pre, "validation": val, "gate": gate,
                "thresholds": th, "split": split_path, "condition": "clean_12lead"}
    if dataset == "Chapman":
        d = M1 / f"seed_{seed}"; gate = json.loads((d / "validation_gate.json").read_text())
        return {"checkpoint": d / "checkpoint.pt", "preprocessing": V4 / "preprocessing_v4.npz",
                "validation": d / "validation.npz", "gate": gate, "thresholds": np.asarray(gate["thresholds"], np.float32),
                "split": V4 / "SPLIT_MEMBERSHIP_v4.json", "condition": "physics_aware_12lead"}
    if model == "clean_12lead":
        d = N / f"seed_{seed}"; gate = json.loads((d / "baseline.json").read_text())
        return {"checkpoint": d / "checkpoint.pt", "preprocessing": d / "preprocessing.npz", "validation": d / "validation.npz",
                "gate": gate, "thresholds": np.asarray(np.load(d / "validation.npz", allow_pickle=False)["thresholds"], np.float32),
                "split": N / "SPLIT_MEMBERSHIP.json", "condition": "clean_12lead"}
    d = NM2TRAIN / f"seed_{seed}"; gate = json.loads((d / "validation_gate.json").read_text())
    return {"checkpoint": d / "checkpoint.pt", "preprocessing": d / "preprocessing.npz", "validation": d / "validation.npz",
            "gate": gate, "thresholds": np.asarray(gate["thresholds"], np.float32), "split": N / "SPLIT_MEMBERSHIP.json",
            "condition": "physics_aware_12lead"}


def comparator_audit(dataset, model, seed):
    spec = comparator_spec(dataset, model, seed)
    ck, pre, val, gate = spec["checkpoint"], spec["preprocessing"], spec["validation"], spec["gate"]
    if not all(p.is_file() for p in (ck, pre, val)): raise RuntimeError(f"missing comparator artifacts {dataset}/{model}/{seed}")
    decision = gate.get("decision")
    if decision != "PASS": raise RuntimeError(f"comparator validation gate failed {dataset}/{model}/{seed}: {decision}")
    if dataset == "Chapman":
        if gate.get("test_accessed") is True or gate.get("split_sha256") not in (None, sha(spec["split"])):
            raise RuntimeError(f"comparator provenance mismatch {dataset}/{model}/{seed}")
    elif gate.get("test_accessed") is True:
        raise RuntimeError(f"test access flag mismatch {dataset}/{model}/{seed}")
    z = np.load(val, allow_pickle=False)
    vy = z["y"] if "y" in z.files else None
    vp = z["p"] if "p" in z.files else None
    if vp is None or not np.isfinite(vp).all() or vp.ndim != 2 or vp.shape[1] != 3 or spec["thresholds"].shape != (3,):
        raise RuntimeError(f"comparator validation prediction invalid {dataset}/{model}/{seed}")
    if model == "clean_12lead" and dataset == "Chapman" and seed == 104729:
        if vy is None or not np.array_equal(thresholds(vy, vp), spec["thresholds"]):
            raise RuntimeError("Chapman v4 validation threshold replay mismatch")
    expected_ck = gate.get("checkpoint_sha256")
    if not expected_ck:
        if dataset == "Chapman" and model == "clean_12lead" and seed == 104729:
            expected_ck = gate["checkpoint_sha256"]
        elif dataset == "Chapman" and model == "clean_12lead":
            expected_ck = json.loads((V7 / f"seed_{seed}" / "evaluation_complete.json").read_text())["checkpoint_sha256"]
        elif dataset == "Ningbo" and model == "clean_12lead":
            expected_ck = json.loads((N / f"seed_{seed}" / "evaluation_complete.json").read_text())["checkpoint_sha256"]
    if sha(ck).lower() != str(expected_ck or "").lower():
        raise RuntimeError(f"comparator checkpoint hash mismatch {dataset}/{model}/{seed}")
    if np.load(pre, allow_pickle=False)["mean"].shape != (12,) and dataset == "Chapman":
        raise RuntimeError("Chapman comparator preprocessing shape mismatch")
    if dataset == "Ningbo" and np.load(pre, allow_pickle=False)["mean"].shape != (12,):
        raise RuntimeError("Ningbo comparator preprocessing shape mismatch")
    return {"dataset": dataset, "model": model, "seed": seed, "decision": decision,
            "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha(ck),
            "preprocessing": str(pre.relative_to(ROOT)), "preprocessing_sha256": sha(pre),
            "validation": str(val.relative_to(ROOT)), "validation_sha256": sha(val),
            "validation_twa_auprc": gate.get("validation_twa_auprc", gate.get("auprc")),
            "validation_thresholds": spec["thresholds"].tolist(), "split_sha256": sha(spec["split"])}


def protocol_template():
    split_hashes = {"Chapman": sha(V4 / "SPLIT_MEMBERSHIP_v4.json"), "Ningbo": sha(N / "SPLIT_MEMBERSHIP.json")}
    counts = {"Chapman": {"train": 7163, "validation": 1021, "test": 2050},
              "Ningbo": {"train": 24430, "validation": 3488, "test": 6981}}
    comparators = [comparator_audit(d, m, s) for d in DATASETS for m in MODELS[:2] for s in SEEDS]
    model6 = ChestOnlyResNetSmall(); model12 = ResNet12()
    return {"version": "chest_only_baseline_v1", "status": "FROZEN_BEFORE_TRAINING",
            "created_utc": utc(), "scope": "Chapman and Ningbo chest-only ECGResNet-small; six new fits; no tuning",
            "question": "Does physics-aware training of a 12-lead ResNet improve performance over a ResNet trained only on unchanged chest leads V1-V6?",
            "datasets": {d: {"split_membership_sha256": split_hashes[d], "counts": counts[d],
                             "endpoints": {"TWA": "164934002", "SB": "426177001", "STach": "427084000"}} for d in DATASETS},
            "seeds": list(SEEDS), "new_models": 6,
            "chest_only_input": {"lead_order": ["V1", "V2", "V3", "V4", "V5", "V6"], "canonical_indices": [6, 7, 8, 9, 10, 11],
                                 "augmentation": "none", "physical_reversal_or_masking_training": False,
                                 "normalization": "per-channel mean/std computed over training split only using V1-V6"},
            "architecture": {"base": "ECGResNet-small from src/chapman_phase1.py", "change": "stem input channels 12 to 6 only",
                             "remaining_blocks_pooling_head_unchanged": True,
                             "parameter_count_chest": sum(p.numel() for p in model6.parameters()),
                             "parameter_count_12lead": sum(p.numel() for p in model12.parameters()),
                             "parameter_delta": sum(p.numel() for p in model6.parameters()) - sum(p.numel() for p in model12.parameters())},
            "training": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                         "loss": "weighted multilabel BCEWithLogitsLoss, per-label positive weight min(negative/positive,20)",
                         "batch_size": 64, "max_epochs": 30, "mixed_precision": "CUDA autocast float16 + GradScaler",
                         "early_stopping": "validation TWA AUPRC only, patience 5, strict improvement", "initialization": "seeded PyTorch initialization",
                         "data_loader_workers": 2, "validation_gate": "record all fits, including weak; no test-based model selection"},
            "evaluation": {"conditions": list(CONDITIONS), "primary": ["clean", "RA_LL_reversal"],
                           "secondary": ["RA_LA_reversal", "LA_LL_reversal"],
                           "metrics": ["AUPRC", "AUROC", "Brier", "ECE_15", "validation_fixed_sensitivity", "validation_fixed_specificity", "predicted_positive_fraction"],
                           "validation_threshold": "90th percentile of negative validation scores per endpoint, frozen per model/seed",
                           "expected_chest_condition_invariance": "scores equal across clean and physical limb reversals because V1-V6 are unchanged"},
            "engineering_tests": ["six-channel tensor and parameter-count check", "synthetic and real training-record reversal invariance",
                                  "six-channel train-only normalization", "actual eval-mode inference invariance", "labels/order and split binding",
                                  "finite CUDA forward/backward step", "comparator validation/checkpoint/preprocessing audit"],
            "comparators": comparators,
            "related_frozen_protocols": {
                "M1_Chapman": {"path": "configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json", "sha256": sha(ROOT / "configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json")},
                "M2_Ningbo": {"path": "configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json", "sha256": sha(ROOT / "configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json")}},
            "data_manifest_sha256": sha(ROOT / "DATA_MANIFEST.json"),
            "bootstrap": {d: {"indices_path": ("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if d == "Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy"),
                               "indices_sha256": sha((V7 if d == "Chapman" else N) / "bootstrap_indices.npy"),
                               "shape": list(np.load((V7 if d == "Chapman" else N) / "bootstrap_indices.npy", mmap_mode="r").shape),
                               "replicates": 1000, "unit": "canonical test record", "synchronized_across_models_seeds_conditions": True,
                               "summary": "paired metric differences per seed, averaged across seeds; prediction vectors are never averaged"} for d in DATASETS},
            "test_access": "Only after all six chest-only training units have finished and validation artifacts are frozen; test labels never select checkpoints/parameters",
            "prohibited": ["Transformer/Ningbo retraining", "Founder", "HEEDB", "other architectures", "combined/masking training", "hyperparameter selection", "test-based tuning"],
            "source_sha256": {p: sha(ROOT / p) for p in ["src/chest_only_baseline_v1.py", "tests/test_chest_only_baseline_v1.py",
                           "src/chapman_phase1.py", "src/chapman_stage23.py", "src/mitigation_m1_v1.py",
                           "src/mitigation_m2_ningbo_v1.py", "src/ningbo_confirmation_v1.py", "src/ecg_physics.py"]}}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol_template()
    if PROTO.exists():
        old = json.loads(PROTO.read_text(encoding="utf-8"))
        if old != p: raise RuntimeError("frozen protocol already exists and differs; refuse to overwrite")
    else: atomic_json(PROTO, p)
    atomic_json(OUT / "FROZEN_PROTOCOL_HASH.json", {"path": str(PROTO.relative_to(ROOT)), "sha256": sha(PROTO), "frozen_utc": utc()})
    print(json.dumps({"decision": "FROZEN", "sha256": sha(PROTO), "models": 6,
                      "parameters_chest": p["architecture"]["parameter_count_chest"],
                      "parameters_12lead": p["architecture"]["parameter_count_12lead"],
                      "comparator_units_audited": len(p["comparators"])}, indent=2))


def prepare():
    if not PROTO.is_file(): raise RuntimeError("frozen protocol missing; run freeze after source is final")
    frozen = json.loads(PROTO.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_TRAINING": raise RuntimeError("invalid protocol status")
    for path, expected in frozen["source_sha256"].items():
        if sha(ROOT / path) != expected: raise RuntimeError(f"frozen source changed: {path}")
    if sha(ROOT / "DATA_MANIFEST.json") != frozen["data_manifest_sha256"]: raise RuntimeError("frozen data manifest changed")
    for item in frozen["related_frozen_protocols"].values():
        if sha(ROOT / item["path"]) != item["sha256"]: raise RuntimeError(f"related protocol changed: {item['path']}")
    for dataset in DATASETS:
        p = split_records(dataset)
        counts = {k: len(v) for k, v in p.items()}
        if counts != frozen["datasets"][dataset]["counts"]: raise RuntimeError(f"split counts changed: {dataset}: {counts}")
        if dataset == "Chapman": split_path = V4 / "SPLIT_MEMBERSHIP_v4.json"
        else: split_path = N / "SPLIT_MEMBERSHIP.json"
        if sha(split_path) != frozen["datasets"][dataset]["split_membership_sha256"]: raise RuntimeError(f"split binding changed: {dataset}")
        trids = {record_id(dataset, r) for r in p["train"]}; valid = {record_id(dataset, r) for r in p["validation"]}; test = {record_id(dataset, r) for r in p["test"]}
        if (trids & valid) or (trids & test) or (valid & test): raise RuntimeError(f"split leakage: {dataset}")
    # Comparator hashes and all source artifacts are rechecked before any fit.
    audited = [comparator_audit(d, m, s) for d in DATASETS for m in MODELS[:2] for s in SEEDS]
    if audited != frozen["comparators"]: raise RuntimeError("frozen comparator artifact bindings changed")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.manual_seed(4); model = ChestOnlyResNetSmall().cuda(); x = torch.randn(2, 6, 1000, device="cuda")
    target = torch.randn(2, 3, device="cuda"); y = model(x); loss = nn.BCEWithLogitsLoss()(y, target); loss.backward()
    if y.shape != (2, 3) or not torch.isfinite(y).all() or not torch.isfinite(loss): raise RuntimeError("CUDA chest-only model smoke failed")
    paramdiff = frozen["architecture"]["parameter_delta"]
    if paramdiff != -1344: raise RuntimeError(f"unexpected architecture parameter delta {paramdiff}")
    report = {"decision": "PASS", "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "protocol_sha256": sha(PROTO), "six_comparator_units_reverified": len(audited), "splits_disjoint": True,
              "parameter_delta": paramdiff, "smoke_shape": list(y.shape), "finite": bool(torch.isfinite(y).all() and torch.isfinite(loss)),
              "test_waveforms_or_labels_loaded": False, "completed_utc": utc()}
    atomic_json(OUT / "ENGINEERING_PREFLIGHT.json", report)
    print(json.dumps(report, indent=2))


def real_fragment_check(dataset, records):
    # Only a training record is read. Check signal and full normalization path across all physical reversals.
    r = records[0]; x = raw12(dataset, r).astype(np.float32)
    m = np.mean(x[CHEST], axis=1).astype(np.float32); s = np.std(x[CHEST], axis=1).clip(1e-4).astype(np.float32)
    clean = ((x[CHEST] - m[:, None]) / s[:, None]).astype(np.float32)
    result = {}
    for condition in CONDITIONS[1:]:
        changed = physical_reversal_chest(x, condition)
        norm = ((changed - m[:, None]) / s[:, None]).astype(np.float32)
        result[condition] = {"max_abs_signal_delta": float(np.max(np.abs(x[CHEST] - changed))),
                             "max_abs_normalized_delta": float(np.max(np.abs(clean - norm))),
                             "pass": bool(np.array_equal(x[CHEST], changed) and np.array_equal(clean, norm))}
    # Exercise the actual model.eval() inference path on the same real training fragment.
    model = ChestOnlyResNetSmall().cuda().eval()
    sample = torch.from_numpy(clean[None, :, :]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): clean_score = model(sample).float().cpu().numpy()
    for condition in CONDITIONS[1:]:
        changed = physical_reversal_chest(x, condition)
        norm = ((changed - m[:, None]) / s[:, None]).astype(np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): score = model(torch.from_numpy(norm[None, :, :]).cuda()).float().cpu().numpy()
        result[condition]["max_abs_eval_score_delta"] = float(np.max(np.abs(clean_score - score)))
        result[condition]["pass"] = bool(result[condition]["pass"] and np.array_equal(clean_score, score))
    return {"dataset": dataset, "record_id": record_id(dataset, r), "conditions": result,
            "pass": all(x["pass"] for x in result.values()), "split": "train"}


def train_unit(dataset, seed, records):
    d = unit_dir(dataset, seed); d.mkdir(parents=True, exist_ok=True)
    gate_path = d / "training_complete.json"
    protocol_hash = sha(PROTO)
    if gate_path.exists():
        old = json.loads(gate_path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != protocol_hash or old.get("checkpoint_sha256") != sha(d / "checkpoint.pt"):
            raise RuntimeError(f"completed run provenance mismatch: {dataset}/{seed}")
        return old
    seed_all(seed); model = ChestOnlyResNetSmall().cuda()
    resume_path, best_path = d / "resume.pt", d / "best.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda")
    m, s = stats(dataset, records["train"])
    atomic_npz(d / "preprocessing.npz", mean=m, std=s, channels=np.asarray(["V1", "V2", "V3", "V4", "V5", "V6"]))
    ytrain = np.stack([record_labels(dataset, r) for r in records["train"]])
    posweight = np.minimum((len(ytrain) - ytrain.sum(0)) / np.maximum(ytrain.sum(0), 1), 20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posweight, device="cuda"))
    best, wait, hist, start_epoch = -1., 0, [], 1
    if resume_path.exists():
        q = torch.load(resume_path, weights_only=False, map_location="cuda")
        if q.get("protocol_sha256") != protocol_hash or q.get("source_sha256") != frozen_source_hashes():
            raise RuntimeError(f"resume provenance mismatch: {dataset}/{seed}")
        model.load_state_dict(q["model"]); opt.load_state_dict(q["optimizer"]); scaler.load_state_dict(q["scaler"])
        best, wait, hist, start_epoch = q["best"], q["wait"], q["history"], int(q["epoch"]) + 1
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, 31):
        model.train(); losses = []
        for xb, yb in loader(dataset, records["train"], m, s, True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16): loss = loss_fn(model(xb.cuda(non_blocking=True)), yb.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss: {dataset}/{seed}/{epoch}")
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        vy, vp = predict(model, dataset, records["validation"], m, s)
        val_ap = float(average_precision_score(vy[:, 0], vp[:, 0]))
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_twa_auprc": val_ap})
        if val_ap > best:
            best, wait = val_ap, 0
            atomic_torch(best_path, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                     "epoch": epoch, "validation_twa_auprc": best, "seed": seed, "protocol_sha256": protocol_hash})
        else: wait += 1
        atomic_json(d / "history.json", hist)
        atomic_torch(resume_path, {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                                   "best": best, "wait": wait, "history": hist, "epoch": epoch,
                                   "protocol_sha256": protocol_hash, "source_sha256": frozen_source_hashes(),
                                   "mean": m, "std": s, "posweight": posweight})
        status = {"dataset": dataset, "seed": seed, "stage": "TRAINING", "epoch": epoch}
        status.update({"best_epoch": torch.load(best_path, weights_only=True, map_location="cpu")["epoch"],
                       "best_validation_twa_auprc": best, "wait": wait, "updated_utc": utc()})
        atomic_json(OUT / "WORKFLOW_STATUS.json", status)
        print(json.dumps(status), flush=True)
        if wait >= 5: break
    best_state = torch.load(best_path, weights_only=True, map_location="cpu")
    model.load_state_dict(best_state["state_dict"])
    vy, vp = predict(model, dataset, records["validation"], m, s); th = thresholds(vy, vp)
    atomic_torch(d / "checkpoint.pt", {"state_dict": model.state_dict(), "seed": seed,
                                       "dataset": dataset, "input_channels": ["V1", "V2", "V3", "V4", "V5", "V6"],
                                       "protocol_sha256": protocol_hash, "best_epoch": best_state["epoch"]})
    atomic_npz(d / "validation.npz", y=vy, p=vp, thresholds=th)
    elapsed = time.monotonic() - started
    result = {"dataset": dataset, "seed": seed, "decision": "TRAINED", "best_epoch": int(best_state["epoch"]),
              "best_validation_twa_auprc": float(best_state["validation_twa_auprc"]),
              "validation_twa_auroc": float(roc_auc_score(vy[:, 0], vp[:, 0])), "validation_thresholds": th.tolist(),
              "checkpoint_sha256": sha(d / "checkpoint.pt"), "preprocessing_sha256": sha(d / "preprocessing.npz"),
              "history_sha256": sha(d / "history.json"), "protocol_sha256": protocol_hash,
              "source_sha256": frozen_source_hashes(), "split_sha256": json.loads(PROTO.read_text(encoding="utf-8"))["datasets"][dataset]["split_membership_sha256"],
              "training_seconds": elapsed, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "training_loss_finite": bool(np.isfinite([q["train_loss"] for q in hist]).all()),
              "validation_predictions_finite": bool(np.isfinite(vp).all()), "stopping_reason": "patience_5" if wait >= 5 else "max_epochs_30",
              "history": hist, "checkpoint_path": str((d / "checkpoint.pt").relative_to(ROOT))}
    atomic_json(gate_path, result)
    return result


def frozen_source_hashes():
    p = json.loads(PROTO.read_text(encoding="utf-8"))
    return p["source_sha256"]


def load12_model(spec, device="cuda"):
    model = ResNet12().to(device)
    obj = torch.load(spec["checkpoint"], weights_only=True, map_location="cpu")
    state = obj.get("state_dict", obj)
    model.load_state_dict(state, strict=True); model.eval()
    z = np.load(spec["preprocessing"], allow_pickle=False)
    return model, z["mean"].astype(np.float32), z["std"].astype(np.float32), spec["thresholds"]
def predict12(model, dataset, records, mean, std, condition):
    probs = []
    for start in range(0, len(records), 128):
        xs = []
        for r in records[start:start + 128]:
            x = raw12(dataset, r).astype(np.float32)
            if condition != "clean":
                independent = np.concatenate((x[:2], x[6:]), axis=0)
                x = derive_display(named_reversal(independent, condition)).astype(np.float32)
            xs.append(((x - mean[:, None]) / std[:, None]).astype(np.float32))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            probs.append(torch.sigmoid(model(torch.from_numpy(np.stack(xs)).cuda())).float().cpu().numpy())
    return np.concatenate(probs)


def predict_chest_condition(model, dataset, records, mean, std, condition):
    """Fresh condition-bound chest inference; physical reversals occur before lead selection."""
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), 128):
            xs, ys = [], []
            for record in records[start:start + 128]:
                raw = raw12(dataset, record).astype(np.float32)
                chest = raw[CHEST] if condition == "clean" else physical_reversal_chest(raw, condition)
                xs.append(((chest - mean[:, None]) / std[:, None]).astype(np.float32))
                ys.append(record_labels(dataset, record))
            batch = torch.from_numpy(np.stack(xs)).cuda()
            with torch.autocast("cuda", dtype=torch.float16): p = torch.sigmoid(model(batch)).float().cpu().numpy()
            probs.append(p); labels.extend(ys)
    return np.stack(labels), np.concatenate(probs)


def evaluate_unit(dataset, seed, records, training):
    d = unit_dir(dataset, seed); pred_dir = d / "test_predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    obj = torch.load(d / "checkpoint.pt", weights_only=True, map_location="cpu")
    if obj.get("protocol_sha256") != sha(PROTO): raise RuntimeError("chest checkpoint protocol hash mismatch")
    z = np.load(d / "preprocessing.npz", allow_pickle=False); mean, std = z["mean"], z["std"]
    model = ChestOnlyResNetSmall().cuda(); model.load_state_dict(obj["state_dict"], strict=True); model.eval()
    test_ids = [record_id(dataset, r) for r in records["test"]]
    order_hash = hashlib.sha256(("\n".join(test_ids) + "\n").encode()).hexdigest()
    frozen_ds = json.loads(PROTO.read_text(encoding="utf-8"))["datasets"][dataset]
    atomic_json(d / "evaluation_freeze.json", {"protocol_sha256": sha(PROTO), "checkpoint_sha256": sha(d / "checkpoint.pt"),
                 "split_sha256": frozen_ds["split_membership_sha256"], "test_record_order_sha256": order_hash,
                 "test_accessed_before_freeze": False, "test_accessed_at": utc(), "conditions": list(CONDITIONS),
                 "evaluator_source_sha256": sha(Path(__file__))})
    y = np.stack([record_labels(dataset, r) for r in records["test"]])
    metric_rows = []
    preds = {}
    for condition in CONDITIONS:
        path = pred_dir / f"{condition}.npz"
        binding = {"protocol_sha256": sha(PROTO), "checkpoint_sha256": sha(d / "checkpoint.pt"), "test_record_order_sha256": order_hash,
                   "condition": condition, "input_channels": ["V1", "V2", "V3", "V4", "V5", "V6"]}
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if old["p"].shape != (len(y), 3) or not np.isfinite(old["p"]).all() or json.loads(str(old["binding"].item())) != binding:
                    raise RuntimeError(f"invalid cached chest prediction: {path}")
                p = old["p"]
        else:
            _, p = predict_chest_condition(model, dataset, records["test"], mean, std, condition)
            atomic_npz(path, y=y, p=p, binding=np.asarray(json.dumps(binding, sort_keys=True)))
        preds[condition] = p
        for row in metrics(y, p, np.load(d / "validation.npz", allow_pickle=False)["thresholds"]):
            row.update({"dataset": dataset, "model": "chest_only", "seed": seed, "condition": condition,
                        "test_record_order_sha256": order_hash})
            metric_rows.append(row)
    # Invariant proof is measured on complete test inference, not asserted from lead names alone.
    invariant = {c: float(np.max(np.abs(preds["clean"] - preds[c]))) for c in CONDITIONS[1:]}
    if any(v != 0. for v in invariant.values()): raise RuntimeError(f"chest-only test scores changed under limb reversal: {invariant}")
    complete = {"decision": "PASS", "dataset": dataset, "seed": seed, "conditions": len(CONDITIONS),
                "checkpoint_sha256": sha(d / "checkpoint.pt"), "preprocessing_sha256": sha(d / "preprocessing.npz"),
                "prediction_sha256": {c: sha(pred_dir / f"{c}.npz") for c in CONDITIONS},
                "test_record_order_sha256": order_hash, "test_labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "max_abs_score_delta_vs_clean_by_reversal": invariant, "completed_utc": utc()}
    atomic_json(d / "evaluation_complete.json", complete)
    return y, preds, metric_rows, complete


def evaluate_comparators(dataset, seed, records, order_hash, y):
    outputs = {}; audit = []
    for model_name in MODELS[:2]:
        spec = comparator_spec(dataset, model_name, seed); audit_row = comparator_audit(dataset, model_name, seed); audit.append(audit_row)
        cache = OUT / "comparators" / dataset / model_name / f"seed_{seed}"; cache.mkdir(parents=True, exist_ok=True)
        model, mean, std, th = load12_model(spec)
        outputs[model_name] = {}
        for condition in CONDITIONS:
            path = cache / f"{condition}.npz"
            binding = {"dataset": dataset, "model": model_name, "seed": seed, "condition": condition,
                       "source_checkpoint_sha256": audit_row["checkpoint_sha256"], "source_preprocessing_sha256": audit_row["preprocessing_sha256"],
                       "test_record_order_sha256": order_hash, "protocol_sha256": sha(PROTO)}
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if z["p"].shape != y.shape or json.loads(str(z["binding"].item())) != binding:
                        raise RuntimeError(f"invalid comparator cache {path}")
                    p = z["p"]
            else:
                p = predict12(model, dataset, records["test"], mean, std, condition)
                atomic_npz(path, y=y, p=p, binding=np.asarray(json.dumps(binding, sort_keys=True)))
            outputs[model_name][condition] = p
        atomic_json(cache / "evaluation_complete.json", {"decision": "PASS", "source_artifacts": audit_row,
                     "test_record_order_sha256": order_hash, "prediction_sha256": {c: sha(cache / f"{c}.npz") for c in CONDITIONS}})
    return outputs, audit


def bootstrap_metrics(y, p, idx):
    out = np.empty((len(idx), 3, 3), np.float64)
    for j, ids in enumerate(idx):
        yy = y[ids]; pp = p[ids]
        for e in range(3):
            out[j, e, 0] = average_precision_score(yy[:, e], pp[:, e])
            out[j, e, 1] = roc_auc_score(yy[:, e], pp[:, e])
            out[j, e, 2] = brier_score_loss(yy[:, e], pp[:, e])
    return out


def finalize(all_metrics, per_run, comparator_audits):
    csv_write(OUT / "RESULTS_BY_DATASET_MODEL_SEED.csv", all_metrics)
    training_rows = []
    for run in per_run:
        training_rows.append({k: json.dumps(run[k], sort_keys=True) if isinstance(run.get(k), (dict, list)) else run.get(k)
                              for k in ("dataset", "seed", "decision", "best_epoch", "best_validation_twa_auprc", "validation_twa_auroc",
                                        "validation_thresholds", "stopping_reason", "training_seconds", "gpu_peak_allocated_bytes",
                                        "gpu_peak_reserved_bytes", "training_loss_finite", "validation_predictions_finite",
                                        "checkpoint_path", "checkpoint_sha256", "preprocessing_sha256", "history_sha256")})
    csv_write(OUT / "TRAINING_STATUS_BY_DATASET_SEED.csv", training_rows)
    atomic_json(OUT / "COMPARATOR_AUDIT.json", {"reused_training": False, "units": comparator_audits,
                  "source_models_reinferred_on_bound_test_order": True, "completed_utc": utc()})
    contrasts = []; bootstrap_rows = []
    pair_defs = [("physics_aware_12lead", "clean", "chest_only", "clean", "physics_aware_12lead_minus_chest_only_clean"),
                 ("physics_aware_12lead", "RA_LL_reversal", "chest_only", "RA_LL_reversal", "physics_aware_12lead_minus_chest_only_RA_LL"),
                 ("clean_12lead", "clean", "chest_only", "clean", "clean_12lead_minus_chest_only_clean"),
                 ("clean_12lead", "RA_LL_reversal", "chest_only", "RA_LL_reversal", "clean_12lead_minus_chest_only_RA_LL")]
    pred_lookup = {(r["dataset"], r["seed"], r["model"], r["condition"]): r["p"] for r in []}
    # prediction arrays are materialized in a compact resumable bundle by run_all.
    for dataset in DATASETS:
        bundle = np.load(OUT / "statistics" / f"{dataset}_predictions.npz", allow_pickle=False)
        # y is identical across seeds/models, stored once; each score array keyed by model/seed/condition.
        idx_path = V7 / "bootstrap_indices.npy" if dataset == "Chapman" else N / "bootstrap_indices.npy"
        indices = np.load(idx_path, allow_pickle=False)
        if indices.shape != (1000, len(bundle["y"])): raise RuntimeError(f"bootstrap index shape mismatch: {dataset}")
        metric_cache = {}
        for model in MODELS:
            for seed in SEEDS:
                for condition in ("clean", "RA_LL_reversal"):
                    metric_cache[(model, seed, condition)] = bootstrap_metrics(
                        bundle["y"], bundle[f"{model}|{seed}|{condition}"], indices)
        for endpoint_i, endpoint in enumerate(LABELS):
            for left, lc, right, rc, name in pair_defs:
                seed_diffs = []
                for seed_i, seed in enumerate(SEEDS):
                    a = bundle[f"{left}|{seed}|{lc}"]; b = bundle[f"{right}|{seed}|{rc}"]
                    y = bundle["y"]
                    full = [average_precision_score(y[:, endpoint_i], a[:, endpoint_i]) - average_precision_score(y[:, endpoint_i], b[:, endpoint_i]),
                            roc_auc_score(y[:, endpoint_i], a[:, endpoint_i]) - roc_auc_score(y[:, endpoint_i], b[:, endpoint_i]),
                            brier_score_loss(y[:, endpoint_i], a[:, endpoint_i]) - brier_score_loss(y[:, endpoint_i], b[:, endpoint_i])]
                    seed_diffs.append(full)
                    contrasts.append({"dataset": dataset, "endpoint": endpoint, "seed": seed, "comparison": name,
                                      "metric": "AUPRC", "left_model": left, "left_condition": lc,
                                      "right_model": right, "right_condition": rc, "left_minus_right": full[0]})
                # 1000 paired bootstrap resamples; compute each seed's paired metric first, then mean seed deltas.
                reps = np.mean(np.stack([metric_cache[(left, seed, lc)][:, endpoint_i, :] -
                                         metric_cache[(right, seed, rc)][:, endpoint_i, :] for seed in SEEDS]), axis=0)
                for mi, metric_name in enumerate(("AUPRC", "AUROC", "Brier")):
                    estimates = reps[:, mi]; bootmean = float(np.mean(estimates)); lo, hi = np.quantile(estimates, [.025, .975])
                    point = float(np.mean(np.asarray(seed_diffs)[:, mi]))
                    bootstrap_rows.append({"dataset": dataset, "endpoint": endpoint, "comparison": name, "metric": metric_name,
                                           "point_mean_of_seed_differences": point, "bootstrap_mean": bootmean,
                                           "lower95": float(lo), "upper95": float(hi), "replicates": len(indices),
                                           "unit": "canonical_test_record", "synchronized": True})
    csv_write(OUT / "PAIRED_CONTRASTS.csv", contrasts)
    csv_write(OUT / "BOOTSTRAP_SUMMARY.csv", bootstrap_rows)
    report(all_metrics, per_run, bootstrap_rows)


def report(rows, runs, boot):
    # Compact primary TWA table and explicitly bounded interpretation.
    by = {(r["dataset"], r["seed"], r["model"], r["condition"], r["endpoint"]): r for r in rows}
    primary = []
    for ds in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                for cond in ("clean", "RA_LL_reversal"):
                    primary.append(by[(ds, seed, model, cond, "TWA")])
    boot_twa = [r for r in boot if r["endpoint"] == "TWA"]
    lines = ["# Chest-only ResNet baseline: Chapman and Ningbo", "",
             "This report compares six newly trained ECGResNet-small models using only V1-V6 with existing clean-trained and physics-aware 12-lead ResNet comparators. Splits, labels, and training hyperparameters were frozen before fitting. No test-set result selected a checkpoint or parameter.", "",
             f"Frozen protocol SHA-256: `{sha(PROTO)}`. The six fits are in `TRAINING_STATUS_BY_DATASET_SEED.csv`; trained weights and resume states are under `models/`.", "",
             "Chest-only does not receive limb leads, masking, or reversal augmentation. It uses train-derived per-channel V1-V6 normalization. The paired physical reversal conditions leave chest inputs unchanged; exact score invariance was checked in the real inference path.", "",
             "## TWA primary results", "", "| Dataset | Seed | Model | Condition | AUPRC | AUROC | Brier | ECE-15 | Sensitivity | Specificity | Predicted positive |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['model']} | {r['condition']} | {float(r['auprc']):.4f} | {float(r['auroc']):.4f} | {float(r['brier']):.4f} | {float(r['ece_15']):.4f} | {float(r['sensitivity']):.3f} | {float(r['specificity']):.3f} | {float(r['predicted_positive_fraction']):.3f} |")
    lines += ["", "Validation thresholds are model- and seed-specific 90th percentiles of negative validation scores. Sensitivity/specificity comparisons therefore use each model's own frozen validation operating point; they do not establish clinical superiority.", "",
              "## Paired comparisons", "", "Positive AUPRC/AUROC deltas favor the left-hand 12-lead comparator; positive Brier deltas mean its squared probability error is higher. Intervals resample canonical test records synchronously and average the paired seed-level differences.", "",
              "| Dataset | Endpoint | Comparison | Metric | Mean paired difference | Bootstrap 95% interval |", "|---|---|---|---|---:|---:|"]
    for r in boot_twa:
        lines.append(f"| {r['dataset']} | {r['endpoint']} | {r['comparison']} | {r['metric']} | {float(r['point_mean_of_seed_differences']):+.4f} | [{float(r['lower95']):+.4f}, {float(r['upper95']):+.4f}] |")
    lines += ["", "## Interpretation limits", "",
              "If physics-aware 12-lead performance exceeds chest-only, the result supports practical robustness beyond removal of limb leads in these datasets; it does not establish which leads or internal features the network uses. If chest-only is similar or better, it is a strong simpler baseline and changes the practical interpretation of augmentation.",
              "Ningbo is a corrected reanalysis after historical test access, not an untouched independent confirmation. The chest-only models are newly fit but evaluated on that previously accessed test split. Results do not establish clinical utility, equivalence, or general architecture superiority. Calibration and validation-fixed operating points are reported descriptively; no extra calibration or mostly-clean mixture analysis was performed.", "",
              "## Reproducibility", "",
              "The protocol, six model manifests, comparator audit, inference manifests, source hashes, training histories, validation outputs, checkpoint hashes, synchronized bootstrap indices, and paired tables are stored in this directory. Existing source checkpoints, frozen protocols, and historical outputs were not modified.", ""]
    (OUT / "FINAL_CHEST_ONLY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    # A compact plot of primary TWA AUPRC by dataset, seed, model and primary condition.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
    colors = {"clean_12lead": "#2563eb", "physics_aware_12lead": "#059669", "chest_only": "#d97706"}
    for ax, ds in zip(axes, DATASETS):
        xs = np.arange(len(SEEDS)); width = .24
        for j, (model, cond) in enumerate((m, c) for m in MODELS for c in ("clean", "RA_LL_reversal")):
            vals = [float(by[(ds, s, model, cond, "TWA")]["auprc"]) for s in SEEDS]
            ax.bar(xs + (j - 2.5) * width, vals, width, label=f"{model.replace('_12lead','')} {cond}", color=colors[model], alpha=.95 if cond == "clean" else .58)
        ax.set_title(ds); ax.set_xticks(xs, [str(s) for s in SEEDS]); ax.set_xlabel("Seed"); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("TWA AUPRC"); axes[1].legend(fontsize=8, frameon=False, loc="best")
    fig.savefig(OUT / "twa_primary_comparisons.png", dpi=180); plt.close(fig)
    artifact_names = ["FROZEN_CHEST_ONLY_PROTOCOL.json", "FROZEN_PROTOCOL_HASH.json", "ENGINEERING_TEST_REPORT.md", "ENGINEERING_PREFLIGHT.json",
                      "TRAINING_STATUS_BY_DATASET_SEED.csv", "RESULTS_BY_DATASET_MODEL_SEED.csv", "PAIRED_CONTRASTS.csv", "BOOTSTRAP_SUMMARY.csv",
                      "COMPARATOR_AUDIT.json", "FINAL_CHEST_ONLY_REPORT.md", "twa_primary_comparisons.png"]
    manifest = {"protocol_sha256": sha(PROTO), "artifacts": {n: {"sha256": sha(OUT / n), "bytes": (OUT / n).stat().st_size} for n in artifact_names if (OUT / n).is_file()},
                "completed_utc": utc()}
    atomic_json(OUT / "REPORTING_MANIFEST.json", manifest)


def run_all():
    if sha(PROTO) != json.loads((OUT / "FROZEN_PROTOCOL_HASH.json").read_text())["sha256"]: raise RuntimeError("frozen protocol hash mismatch")
    prepare(); records = {d: split_records(d) for d in DATASETS}
    # Engineering invariance checks use training rows only and occur before the first model fit.
    checks = [real_fragment_check(d, records[d]["train"]) for d in DATASETS]
    if not all(q["pass"] for q in checks): raise RuntimeError("real-fragment chest invariance check failed")
    atomic_json(OUT / "ENGINEERING_TEST_REPORT.json", {"decision": "PASS", "real_train_fragment_checks": checks, "test_data_accessed": False, "completed_utc": utc()})
    (OUT / "ENGINEERING_TEST_REPORT.md").write_text(
        "# Chest-only engineering test report\n\n"
        f"Decision: PASS. Protocol SHA-256: `{sha(PROTO)}`. Test waveforms and labels were not accessed.\n\n" +
        "\n".join(f"- {q['dataset']} train record `{q['record_id']}`: " +
                   ", ".join(f"{c}: signal delta={v['max_abs_signal_delta']:.3g}, normalized delta={v['max_abs_normalized_delta']:.3g}, eval score delta={v['max_abs_eval_score_delta']:.3g}" for c, v in q['conditions'].items())
                   for q in checks) +
        "\n\nCUDA forward/backward, six-channel output shape, parameter-count delta, split separation, train-only normalization, comparator provenance, and synthetic unit tests passed.\n",
        encoding="utf-8")
    runs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"START chest-only {dataset} seed={seed}", flush=True)
            runs.append(train_unit(dataset, seed, records[dataset]))
    atomic_json(OUT / "TRAINING_STATUS_BY_DATASET_SEED.json", {"decision": "ALL_SIX_TRAINED", "runs": runs, "completed_utc": utc()})
    all_metrics = []; comparator_audits = []; bundles = {}
    for dataset in DATASETS:
        # Open the test records only after all six training units completed.
        y_ref = None; bundle_arrays = {}; test_ids = None
        for seed in SEEDS:
            y, chest_pred, chest_rows, chest_done = evaluate_unit(dataset, seed, records[dataset], runs[DATASETS.index(dataset) * 3 + SEEDS.index(seed)])
            this_ids = [record_id(dataset, r) for r in records[dataset]["test"]]
            order_hash = hashlib.sha256(("\n".join(this_ids) + "\n").encode()).hexdigest()
            if test_ids is None: test_ids = this_ids; y_ref = y
            elif test_ids != this_ids or not np.array_equal(y_ref, y): raise RuntimeError(f"test order/labels differ: {dataset}")
            comp, audits = evaluate_comparators(dataset, seed, records[dataset], order_hash, y)
            comparator_audits.extend(audits)
            for row in chest_rows: all_metrics.append(row)
            th = np.load(unit_dir(dataset, seed) / "validation.npz", allow_pickle=False)["thresholds"]
            for model_name in MODELS[:2]:
                spec = comparator_spec(dataset, model_name, seed)
                for condition in CONDITIONS:
                    for row in metrics(y, comp[model_name][condition], spec["thresholds"]):
                        row.update({"dataset": dataset, "model": model_name, "seed": seed, "condition": condition,
                                    "test_record_order_sha256": order_hash})
                        all_metrics.append(row)
                    bundle_arrays[f"{model_name}|{seed}|{condition}"] = comp[model_name][condition]
            for condition in CONDITIONS: bundle_arrays[f"chest_only|{seed}|{condition}"] = chest_pred[condition]
        bundle_arrays["y"] = y_ref
        atomic_npz(OUT / "statistics" / f"{dataset}_predictions.npz", **bundle_arrays)
    finalize(all_metrics, runs, comparator_audits)
    atomic_json(OUT / "WORKFLOW_STATUS.json", {"stage": "COMPLETE", "decision": "SIX_RUNS_TRAINED_AND_EVALUATED", "protocol_sha256": sha(PROTO), "updated_utc": utc()})
    print(json.dumps({"stage": "COMPLETE", "runs": len(runs), "metrics": len(all_metrics), "protocol_sha256": sha(PROTO)}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("freeze", "prepare", "run")); args = parser.parse_args()
    if args.command == "freeze": freeze()
    elif args.command == "prepare": prepare()
    else: run_all()


if __name__ == "__main__": main()
