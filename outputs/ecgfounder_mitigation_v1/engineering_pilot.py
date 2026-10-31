from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import ecgfounder_baseline_v2 as founder  # noqa: E402
from src.ecg_physics import named_reversal  # noqa: E402
from outputs.heldout_limb_cycles_v1 import cycle_adapter  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20261018
PILOT_PER_COHORT = 64
BATCH = 16
TOL = 1e-6
TRANSFORMS = ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
MASK = (0, 1, 2, 3, 5)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transform(raw: np.ndarray, condition: str) -> np.ndarray:
    if condition in TRANSFORMS:
        return founder.derive_display(named_reversal(np.concatenate((raw[:2], raw[6:]), axis=0), condition))
    return cycle_adapter.transform(raw, condition)


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def pilot_cache_replay(net) -> list[dict]:
    checks = []
    # Existing clean train/validation feature caches are eligible only after exact
    # ordered-ID, source-protocol, checkpoint, file-hash and same-batch replay checks.
    for dataset in ("Chapman", "Ningbo"):
        for partition in ("train", "validation"):
            records = founder.split(dataset, partition)
            ids = [r["record_id"] for r in records]
            fp, lp, mp = founder.cache_paths(dataset, partition, "clean")
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if meta.get("record_ids_sha256") != ids_hash(ids):
                raise RuntimeError(f"canonical ID order mismatch in old cache: {dataset}/{partition}")
            for key, value in (("checkpoint_sha256", sha(founder.CHECKPOINT)),
                               ("protocol_sha256", sha(founder.CFG)),
                               ("dataset", dataset), ("partition", partition),
                               ("condition", "clean"), ("completed_records", len(records)),
                               ("total_records", len(records))):
                if meta.get(key) != value:
                    raise RuntimeError(f"old feature cache binding mismatch: {dataset}/{partition}/{key}")
            if sha(fp) != meta.get("features_sha256") or sha(lp) != meta.get("logits_sha256"):
                raise RuntimeError(f"old clean feature cache file hash mismatch: {dataset}/{partition}")
            feats = np.load(fp, mmap_mode="r", allow_pickle=False)
            logits = np.load(lp, mmap_mode="r", allow_pickle=False)
            if feats.shape != (len(records), 1024) or logits.shape != (len(records), 150):
                raise RuntimeError(f"old feature cache shape mismatch: {dataset}/{partition}")
            n = min(BATCH * 2, len(records))
            raw_batch = np.stack([founder.physical_signal(dataset, rid) for rid in ids[:n]])
            prepared = np.stack([founder.preprocess(x) for x in raw_batch])
            before = {k: v.detach().clone() for k, v in net.state_dict().items()}
            f_parts, z_parts = [], []
            for start in range(0, len(prepared), BATCH):
                f0, z0 = founder.infer(net, prepared[start:start + BATCH])
                f_parts.append(f0)
                z_parts.append(z0)
            f, z = np.concatenate(f_parts), np.concatenate(z_parts)
            d = np.abs(f - feats[:n])
            if not np.allclose(f, feats[:n], rtol=0, atol=TOL):
                raise RuntimeError(f"old clean cache replay failed: {dataset}/{partition}, max={d.max()}")
            if not all(torch.equal(before[k], v) for k, v in net.state_dict().items()):
                raise RuntimeError("backbone state changed during inference")
            checks.append({"dataset": dataset, "partition": partition,
                           "records": len(records), "sample_replay_records": n,
                           "sample_feature_max_abs_error": float(d.max()),
                           "cache_features_sha256": meta["features_sha256"],
                           "ordered_ids_sha256": meta["record_ids_sha256"],
                           "cache_reused_eligible": True})
    return checks


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if sha(founder.CHECKPOINT) != "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997":
        raise RuntimeError("official checkpoint SHA256 mismatch")
    net = founder.model()
    if net.training or any(p.requires_grad for p in net.parameters()):
        raise RuntimeError("backbone is not frozen/eval")
    state_before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    cache_checks = pilot_cache_replay(net)

    timings = []
    transform_checks = []
    raw_reads = 0
    torch.cuda.reset_peak_memory_stats()
    wall0 = time.perf_counter()
    for dataset in ("Chapman", "Ningbo"):
        records = founder.split(dataset, "train")[:PILOT_PER_COHORT]
        ids = [r["record_id"] for r in records]
        # This pilot only touches train records. Each transformed waveform is
        # reprocessed by the official waveform pipeline, then sent through Founder.
        raw_signals = []
        for rid in ids:
            raw_signals.append(founder.physical_signal(dataset, rid))
            raw_reads += 1
        for condition in TRANSFORMS:
            start = time.perf_counter()
            prepared = []
            chest_error = 0.0
            for raw in raw_signals:
                changed = transform(raw, condition)
                chest_error = max(chest_error, float(np.max(np.abs(changed[6:] - raw[6:]))))
                prepared.append(founder.preprocess(changed))
            for i in range(0, len(prepared), BATCH):
                f, z = founder.infer(net, np.stack(prepared[i:i + BATCH]))
                if f.shape[1] != 1024 or z.shape[1] != 150 or not np.isfinite(f).all() or not np.isfinite(z).all():
                    raise RuntimeError(f"invalid transformed inference output: {dataset}/{condition}")
            elapsed = time.perf_counter() - start
            timings.append({"dataset": dataset, "condition": condition, "records": len(ids),
                            "seconds_transform_preprocess_infer": elapsed,
                            "records_per_second": len(ids) / elapsed})
            transform_checks.append({"dataset": dataset, "condition": condition,
                                     "preprocessing_before_encoder": True,
                                     "feature_dimension": 1024,
                                     "chest_lead_max_abs_change_raw_mv": chest_error,
                                     "finite_features_and_logits": True})

    # Mask control is on the normalized tensor delivered to Founder: exactly
    # listed channels are zero; all other channels remain unchanged.
    raw = raw_signals[0]
    normalized = founder.preprocess(raw)
    masked = normalized.copy()
    masked[list(MASK), :] = 0.0
    retained = [i for i in range(12) if i not in MASK]
    if not np.array_equal(masked[list(MASK)], np.zeros_like(masked[list(MASK)])):
        raise RuntimeError("mask channels were not zero at encoder input")
    if not np.array_equal(masked[retained], normalized[retained]):
        raise RuntimeError("unlisted mask channels changed")
    f_mask, z_mask = founder.infer(net, np.stack((normalized, masked)))
    if f_mask.shape != (2, 1024) or z_mask.shape != (2, 150):
        raise RuntimeError("mask engineering smoke output shape mismatch")
    if not all(torch.equal(state_before[k], v) for k, v in net.state_dict().items()):
        raise RuntimeError("frozen backbone state changed")
    wall = time.perf_counter() - wall0
    total_records = sum(t["records"] for t in timings)
    total_seconds = sum(t["seconds_transform_preprocess_infer"] for t in timings)

    # 3 train permutations for every train record plus test views not covered
    # by the verified old clean/RA-LL test caches: RA-LA, LA-LL, 2 cycles, 32 masks.
    train_new = 3 * sum(len(founder.split(d, "train")) for d in ("Chapman", "Ningbo"))
    test_records = sum(len(founder.split(d, "test")) for d in ("Chapman", "Ningbo"))
    test_new = 36 * test_records
    observed_rate = total_records / total_seconds
    # Apply a conservative 1.75x allowance for later runs, transforms, CPU data I/O,
    # and batch tail overhead relative to the train-only pilot.
    estimate_seconds = (train_new + test_new) / observed_rate * 1.75
    report = {
        "status": "PASS",
        "completed_utc": founder.now(),
        "scope": "engineering only; train/validation inputs only; no new test outcomes",
        "checkpoint_sha256": sha(founder.CHECKPOINT),
        "official_code_commit": "04edac702b61c91face519774ddcc0cd712fef23",
        "source_code_sha256": {"runner": sha(Path(founder.__file__)),
                               "physics": sha(ROOT / "src/ecg_physics.py"),
                               "cycle_adapter": sha(ROOT / "outputs/heldout_limb_cycles_v1/cycle_adapter.py"),
                               "mask_semantics": sha(ROOT / "src/mask_semantics_recovery_v2.py")},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "backbone_eval": not net.training,
        "all_parameters_frozen": all(not p.requires_grad for p in net.parameters()),
        "state_dict_unchanged": True,
        "feature_dimension": 1024,
        "numeric_tolerance_atol_rtol": [TOL, 0.0],
        "old_clean_cache_replay": cache_checks,
        "transformed_train_engineering_checks": transform_checks,
        "timings": timings,
        "pilot_total_wall_seconds": wall,
        "pilot_inference_pipeline_records_per_second": observed_rate,
        "input_reads_train_only": raw_reads,
        "mask_control": {"indices_zeroed_at_encoder_input_after_official_preprocessing": list(MASK),
                          "other_channels_unchanged": True,
                          "same_rule_for_clean_head_and_physics_head": True,
                          "resnet_difference": "ResNet masks in raw displayed-lead space by replacing selected channels with train lead means before channel-wise normalization; this Founder control zeros those channels after its official filtering and global z-score, preserving all other normalized channels."},
        "resource_estimate": {"new_train_feature_rows": train_new,
                              "new_test_feature_rows": test_new,
                              "new_test_views_per_record": 36,
                              "estimated_total_new_feature_rows": train_new + test_new,
                              "conservative_factor_over_pilot": 1.75,
                              "estimated_gpu_hours": estimate_seconds / 3600,
                              "limit_gpu_hours": 6.0,
                              "within_limit": estimate_seconds < 6 * 3600},
    }
    if not report["resource_estimate"]["within_limit"]:
        report["status"] = "RESOURCE_LIMIT_EXCEEDED"
    (OUT / "ENGINEERING_PILOT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records_per_second": observed_rate,
                      "estimated_gpu_hours": report["resource_estimate"]["estimated_gpu_hours"],
                      "peak_allocated_gb": report["gpu_peak_allocated_bytes"] / 1e9,
                      "peak_reserved_gb": report["gpu_peak_reserved_bytes"] / 1e9,
                      "clean_cache_replays": len(cache_checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
