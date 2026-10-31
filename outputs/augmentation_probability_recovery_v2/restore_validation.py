from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
sys.path.insert(0, str(ROOT))
from outputs.augmentation_probability_study_v1 import run_probability_study as runner

COHORTS = ("Chapman", "Ningbo")
ARMS = ("physics_p010", "physics_p025")
SEEDS = (104729, 130363, 155921)
ENDPOINTS = ("TWA", "SB", "STach")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_state_match(a: dict, b: dict) -> tuple[bool, str]:
    sa, sb = a["state_dict"], b["state_dict"]
    if sa.keys() != sb.keys():
        return False, "state_dict keys differ"
    for name in sa:
        if not torch.equal(sa[name].detach().cpu(), sb[name].detach().cpu()):
            return False, f"state tensor differs: {name}"
    return True, "all state tensors bitwise equal"


def canonical_ids(cohort: str, records: list) -> list[str]:
    if cohort == "Chapman":
        return [str(x) for x in records]
    return [str(x["record_id"]) for x in records]


def normalize_stored_ids(cohort: str, values: np.ndarray) -> list[str]:
    out = []
    for value in values.tolist():
        text = str(value)
        if cohort == "Ningbo":
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict) or "record_id" not in parsed:
                raise RuntimeError(f"Ningbo stored record identity is not a dict with record_id: {text[:100]}")
            text = str(parsed["record_id"])
        out.append(text)
    return out


def thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.asarray([np.quantile(p[y[:, j] == 0, j], .9, method="linear") for j in range(3)], dtype=np.float64)


def main() -> None:
    protocol_path = OUT / "RECOVERY_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    recorded_hash = protocol.pop("recovery_protocol_sha256")
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    protocol["recovery_protocol_sha256"] = recorded_hash
    if hashlib.sha256(canonical).hexdigest() != recorded_hash:
        raise RuntimeError("Frozen recovery protocol checksum mismatch")
    inputs = protocol["original_inputs_sha256"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for inference-only recovery")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tolerance = float(protocol["validation_recovery"]["predeclared_absolute_tolerance"])
    test_tolerance = float(protocol["validation_recovery"]["test_probability_replay"]["predeclared_max_absolute_tolerance"])
    sample_idx = np.asarray(protocol["validation_recovery"]["test_probability_replay"]["indices"], dtype=np.int64)
    rows = []
    state_audits = []
    replay_audits = []
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in SEEDS:
                unit = STUDY / "training" / cohort / arm / f"seed_{seed}"
                pred_dir = STUDY / "predictions" / cohort / arm / f"seed_{seed}"
                gate = json.loads((unit / "validation_gate.json").read_text(encoding="utf-8"))
                if gate.get("decision") != "PASS":
                    raise RuntimeError(f"Unit validation gate is not PASS: {cohort}/{arm}/{seed}")
                best = torch.load(unit / "best.pt", weights_only=True, map_location="cpu")
                infer_ckpt = torch.load(unit / "checkpoint.pt", weights_only=True, map_location="cpu")
                same, detail = exact_state_match(best, infer_ckpt)
                state_audits.append({"cohort": cohort, "arm": arm, "seed": seed, "match": same, "detail": detail, "best_epoch": int(best["epoch"]), "checkpoint_best_epoch": int(infer_ckpt["best_epoch"]), "best_sha256": sha(unit / "best.pt"), "inference_checkpoint_sha256": sha(unit / "checkpoint.pt")})
                if not same or int(best["epoch"]) != int(infer_ckpt["best_epoch"]):
                    raise RuntimeError(f"best.pt does not match test inference checkpoint: {cohort}/{arm}/{seed}: {detail}")

                rec, mean, std, split_file, prep_file = runner.source_data(cohort, seed)
                val_records = rec["validation"]
                yv = runner.labels_for(cohort, val_records).astype(np.uint8)
                val_ids = canonical_ids(cohort, val_records)
                if len(set(val_ids)) != len(val_ids):
                    raise RuntimeError(f"duplicate canonical validation IDs: {cohort}/{seed}")

                # The original test-label-order artifact is not trusted for ID semantics;
                # match its row order to the frozen canonical test split instead.
                test_records = rec["test"]
                test_ids = canonical_ids(cohort, test_records)
                test_y = runner.labels_for(cohort, test_records).astype(np.uint8)
                labels_order_path = pred_dir / "test_labels_order.npz"
                with np.load(labels_order_path, allow_pickle=False) as z:
                    stored_ids = normalize_stored_ids(cohort, z["record_ids"])
                    stored_y = z["y"].astype(np.uint8)
                if stored_ids != test_ids:
                    raise RuntimeError(f"test record order does not match canonical split for {cohort}/{arm}/{seed}")
                if not np.array_equal(stored_y, test_y):
                    raise RuntimeError(f"test labels do not match canonical test split for {cohort}/{arm}/{seed}")

                model = runner.ECGResNetSmall().cuda()
                model.load_state_dict(infer_ckpt["state_dict"])
                model.eval()
                pv = runner.inference(model, runner.raw_for(cohort, val_records), mean, std).astype(np.float64)
                if not np.isfinite(pv).all() or pv.shape != yv.shape:
                    raise RuntimeError(f"nonfinite/wrong shape best-checkpoint validation: {cohort}/{arm}/{seed}")
                old_path = unit / "validation.npz"
                with np.load(old_path, allow_pickle=False) as z:
                    old_y, old_p = z["y"].astype(np.uint8), z["p"].astype(np.float64)
                    old_thresholds = z["thresholds"].astype(np.float64)
                if not np.array_equal(old_y, yv):
                    raise RuntimeError(f"v1 validation labels/order do not match canonical validation split: {cohort}/{arm}/{seed}")
                old_ap = float(average_precision_score(yv[:, 0], old_p[:, 0]))
                restored_ap = float(average_precision_score(yv[:, 0], pv[:, 0]))
                restored_auc = float(roc_auc_score(yv[:, 0], pv[:, 0]))
                gate_ap = float(gate["validation_twa_auprc"])
                gate_auc = float(gate["validation_twa_auroc"])
                ap_error, auc_error = abs(restored_ap - gate_ap), abs(restored_auc - gate_auc)
                if ap_error > tolerance or auc_error > tolerance:
                    raise RuntimeError(f"best-checkpoint validation does not reproduce gate within frozen tolerance ({ap_error=}, {auc_error=}): {cohort}/{arm}/{seed}")
                new_thresholds = thresholds(yv, pv)
                validation_output = OUT / "validation" / cohort / arm / f"seed_{seed}.npz"
                validation_output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    validation_output,
                    y=yv, p=pv, record_ids=np.asarray(val_ids, dtype=np.str_), thresholds=new_thresholds,
                )
                metadata = {
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "best_epoch": int(best["epoch"]), "last_epoch": int(gate["history"][-1]["epoch"]),
                    "old_saved_validation_twa_auprc": old_ap,
                    "last_history_twa_auprc": float(gate["history"][-1]["validation_twa_auprc"]),
                    "restored_best_checkpoint_twa_auprc": restored_ap,
                    "validation_gate_twa_auprc": gate_ap,
                    "restored_best_checkpoint_twa_auroc": restored_auc,
                    "validation_gate_twa_auroc": gate_auc,
                    "absolute_ap_gate_difference": ap_error,
                    "absolute_auroc_gate_difference": auc_error,
                    "old_thresholds_last_epoch": old_thresholds.tolist(),
                    "restored_thresholds": new_thresholds.tolist(),
                    "state_dicts_bitwise_equal": same,
                    "checkpoint_sha256": sha(unit / "checkpoint.pt"),
                    "best_sha256": sha(unit / "best.pt"),
                    "preprocessing_sha256": sha(prep_file),
                    "split_sha256": sha(split_file),
                    "validation_prediction_sha256": hashlib.sha256(pv.tobytes()).hexdigest(),
                    "canonical_validation_record_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
                    "canonical_test_record_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
                    "test_order_binding": "old row order exactly matched canonical frozen split; original labels/order files untouched",
                }
                (OUT / "validation" / cohort / arm / f"seed_{seed}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                rows.append(metadata)

                if sample_idx.max() >= len(test_ids):
                    raise RuntimeError(f"fixed replay indices outside test split: {cohort}/{seed}")
                x = runner.raw_for(cohort, test_records)
                for condition in ("clean", "RA_LL_reversal"):
                    with np.load(pred_dir / f"{condition}.npz", allow_pickle=False) as z:
                        saved_p = z["p"][sample_idx].astype(np.float64)
                        saved_y = z["y"][sample_idx].astype(np.uint8)
                    # Use the original 128-record batch context. Isolated records
                    # differ by up to one FP16 probability step on this GPU even
                    # with identical weights and raw inputs; same-context replay is exact.
                    replay_by_index = {}
                    for batch_start in sorted(set((sample_idx // 128 * 128).tolist())):
                        batch_end = min(batch_start + 128, len(test_ids))
                        raw_batch = x[batch_start:batch_end]
                        if condition == "RA_LL_reversal":
                            raw_batch = np.stack([
                                runner.derive_display(runner.named_reversal(np.concatenate((a[:2], a[6:]), axis=0), "RA_LL_reversal"))
                                for a in raw_batch
                            ]).astype(np.float32)
                        batch_p = runner.inference(model, raw_batch, mean, std).astype(np.float64)
                        for index in sample_idx[(sample_idx >= batch_start) & (sample_idx < batch_end)]:
                            replay_by_index[int(index)] = batch_p[int(index) - batch_start]
                    replay_p = np.stack([replay_by_index[int(i)] for i in sample_idx])
                    err = float(np.max(np.abs(replay_p - saved_p)))
                    if not np.array_equal(saved_y, test_y[sample_idx]):
                        raise RuntimeError(f"saved test sample labels/order mismatch: {cohort}/{arm}/{seed}/{condition}")
                    replay_audits.append({"cohort": cohort, "arm": arm, "seed": seed, "condition": condition, "sample_indices": sample_idx.tolist(), "max_abs_probability_difference": err, "tolerance": test_tolerance, "match": err <= test_tolerance})
                    if err > test_tolerance:
                        raise RuntimeError(f"saved test probability replay exceeded tolerance: {cohort}/{arm}/{seed}/{condition} max={err}")
                del model
                torch.cuda.empty_cache()

    (OUT / "VALIDATION_RESTORATION_AUDIT.json").write_text(json.dumps({"validation_gate_abs_tolerance": tolerance, "state_dict_audits": state_audits, "test_probability_replay": replay_audits, "units": rows, "all_units_gate_reproduced": True, "all_test_subsamples_reproduced": True}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    import csv
    fields = ["cohort", "arm", "seed", "best_epoch", "last_epoch", "old_saved_validation_twa_auprc", "restored_best_checkpoint_twa_auprc", "validation_gate_twa_auprc", "restored_best_checkpoint_twa_auroc", "validation_gate_twa_auroc", "absolute_ap_gate_difference"]
    with (OUT / "VALIDATION_RESTORATION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows([{k: row[k] for k in fields} for row in rows])
    print(json.dumps({"units": len(rows), "state_dict_matches": sum(x["match"] for x in state_audits), "test_replay_checks": len(replay_audits), "test_replay_pass": sum(x["match"] for x in replay_audits), "gate_abs_tolerance": tolerance}, ensure_ascii=False))


if __name__ == "__main__":
    main()
