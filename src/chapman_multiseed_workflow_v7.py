"""Durable frozen v7 Chapman evaluator and aggregate bootstrap workflow.

This is deliberately separate from the historical v4/v6 scripts.  It neither
trains models nor accesses Ningbo.  `prepare` is the pre-test-access gate;
`run` resumes only validated prediction and bootstrap artifacts.
"""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, os, re, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.chapman_phase1 import ECGResNetSmall, raw, labels
from src.ecg_physics import derive_display, named_reversal

OUT = ROOT / "outputs" / "phase2_chapman_multiseed_v7"
V4 = ROOT / "outputs" / "phase1_chapman_seed1_v4"
V6 = ROOT / "outputs" / "phase1b_matched_reversal_development_v6"
PROTO = ROOT / "configs" / "FROZEN_CHAPMAN_MULTISEED_PROTOCOL_v7.json"
SEEDS = (104729, 130363, 155921)
NEW_SEEDS = (130363, 155921)
LABELS = ("TWA", "SB", "STach")
MASKS = tuple(itertools.combinations(range(12), 5))
MASK_NAMES = tuple("mask5_" + "_".join(map(str, q)) for q in MASKS)
CONDITIONS = ("clean", "RA_LL_reversal", *MASK_NAMES)
N_BOOT = 1000
BOOT_SEED = 20260828


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    x = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): x.update(b)
    return x.hexdigest()
def shatext(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
def status(stage, **extra):
    payload = {"pid": os.getpid(), "stage": stage, "updated_utc": now(), **extra}
    atomic_json(OUT / "WORKFLOW_STATUS.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
def test_ids():
    rows = json.loads((V4 / "SPLIT_MEMBERSHIP_v4.json").read_text(encoding="utf-8"))["rows"]
    return [r["record_id"] for r in rows if r["split"] == "test"]
def split_hash():
    return sha(V4 / "SPLIT_MEMBERSHIP_v4.json")
def ids_hash(ids): return shatext("\n".join(ids) + "\n")
def condition_manifest():
    return {"conditions": list(CONDITIONS), "mask_count": len(MASKS),
            "condition_manifest_sha256": shatext(json.dumps(list(CONDITIONS), separators=(",", ":")))}
def npz_prediction(path, n, identity):
    if not Path(path).is_file(): return None
    try:
        z = np.load(path, allow_pickle=False); p = z["p"]
        if p.shape != (n, 3) or not np.isfinite(p).all(): return None
        return p.astype(np.float32, copy=False)
    except Exception:
        return None
def write_csv(path, rows, fields):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)
def ece(y, p, bins=15):
    total = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        take = (p >= lo) & ((p < hi) if k < bins - 1 else (p <= hi))
        if take.any(): total += take.mean() * abs(float(y[take].mean()) - float(p[take].mean()))
    return float(total)
def metric_rows(seed, name, y, p, thresholds):
    rows = []
    for i, label in enumerate(LABELS):
        q = p[:, i]; z = y[:, i]; t = float(thresholds[i]); call = q >= t
        pos, neg = z == 1, z == 0
        rows.append({"seed": seed, "condition": name, "label": label,
                     "auprc": float(average_precision_score(z, q)), "auroc": float(roc_auc_score(z, q)),
                     "sensitivity": float(call[pos].mean()), "specificity": float((~call[neg]).mean()),
                     "ece_15": ece(z, q), "threshold": t, "n": len(z)})
    return rows


def smoke():
    if len(MASKS) != 792 or len(set(MASK_NAMES)) != 792: raise RuntimeError("five-mask enumeration failure")
    rng = np.random.default_rng(17); independent = rng.normal(size=(4, 8, 19))
    clean = derive_display(independent)
    expected = {"RA_LA_reversal": 5, "RA_LL_reversal": 4, "LA_LL_reversal": 3}
    for fault, invariant in expected.items():
        changed = derive_display(named_reversal(independent, fault))
        if not np.allclose(clean[:, invariant], changed[:, invariant], atol=1e-12):
            raise RuntimeError(f"algebra invariant failed: {fault}")
        limb_changed = [k for k in range(6) if not np.allclose(clean[:, k], changed[:, k], atol=1e-12)]
        if limb_changed != [k for k in range(6) if k != invariant] or not np.allclose(clean[:, 6:], changed[:, 6:], atol=1e-12):
            raise RuntimeError(f"algebra consequence count failed: {fault}")
    y = np.array([0, 1, 0, 1, 1]); a = np.array([.1, .9, .2, .7, .8]); b = np.array([.2, .8, .3, .6, .7])
    if not np.isfinite(average_precision_score(y, a) - average_precision_score(y, b)):
        raise RuntimeError("bootstrap metric smoke failure")
    print("synthetic smoke PASS", flush=True)


def validate_baselines():
    protocol_hash = sha(PROTO).upper()
    if protocol_hash != "312DABB9A0400675097D492F66DF58C9DA5EAD69623FFF0563F4DAECE0944CB7":
        raise RuntimeError("frozen protocol hash mismatch")
    for seed in NEW_SEEDS:
        d = OUT / f"seed_{seed}"; baseline = json.loads((d / "baseline.json").read_text())
        if baseline.get("decision") != "PASS" or baseline.get("test_accessed") is not False:
            raise RuntimeError(f"invalid baseline state for {seed}")
        for name in ("checkpoint.pt", "validation.npz", "pretest_freeze.json"):
            if not (d / name).is_file(): raise RuntimeError(f"missing {seed}/{name}")
        z = np.load(d / "validation.npz", allow_pickle=False)
        if z["p"].shape != z["y"].shape or not np.isfinite(z["p"]).all() or not np.isfinite(z["thresholds"]).all():
            raise RuntimeError(f"invalid validation predictions for {seed}")
    return protocol_hash
def freeze_evaluations():
    """This function intentionally performs no waveform/prediction reads."""
    OUT.mkdir(parents=True, exist_ok=True); smoke(); protocol_hash = validate_baselines()
    for seed in NEW_SEEDS:
        d = OUT / f"seed_{seed}"; pred_dir = d / "test_predictions"
        if pred_dir.exists() and any(pred_dir.glob("*.npz")) and not (d / "evaluation_complete.json").is_file():
            raise RuntimeError(f"unmanifested existing test predictions for {seed}")
        target = d / "evaluation_freeze.json"
        payload = {"version": "v7_pre_test_evaluation_freeze", "created_utc": now(), "seed": seed,
                   "protocol_sha256": protocol_hash, "canonical_split_sha256": split_hash(),
                   "checkpoint_sha256": sha(d / "checkpoint.pt"), "validation_sha256": sha(d / "validation.npz"),
                   "physics_library_sha256": sha(ROOT / "src" / "ecg_physics.py"),
                   "evaluator_code_sha256": sha(Path(__file__)), "bootstrap_code_sha256": sha(Path(__file__)),
                   "test_accessed_before_evaluation_freeze": False, **condition_manifest()}
        if target.exists():
            old = json.loads(target.read_text())
            for key in ("protocol_sha256", "canonical_split_sha256", "checkpoint_sha256", "validation_sha256",
                        "physics_library_sha256", "evaluator_code_sha256", "bootstrap_code_sha256",
                        "condition_manifest_sha256", "test_accessed_before_evaluation_freeze"):
                if old.get(key) != payload[key]: raise RuntimeError(f"existing evaluation freeze mismatch for {seed}: {key}")
        else: atomic_json(target, payload)
    atomic_json(OUT / "PRETEST_PREPARATION_v7.json", {"status": "PASS", "prepared_utc": now(),
                                                        "protocol_sha256": protocol_hash, "synthetic_smoke": "PASS"})


def inference(model, a):
    chunks = []; model.eval()
    with torch.no_grad():
        for k in range(0, len(a), 128):
            with torch.autocast("cuda", dtype=torch.float16):
                chunks.append(torch.sigmoid(model(torch.from_numpy(a[k:k + 128]).cuda())).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)
def seed_paths(seed):
    return (OUT / f"seed_{seed}") if seed != 104729 else V6
def load_seed104(ids):
    base = V4 / "test_predictions_v4"; cache = V6 / "predictions"; n = len(ids)
    out = {"clean": npz_prediction(base / "clean.npz", n, "clean"),
           "RA_LL_reversal": npz_prediction(base / "RA_LL_reversal.npz", n, "RA_LL_reversal")}
    for name in MASK_NAMES: out[name] = npz_prediction(cache / f"{name}.npz", n, name)
    if any(v is None for v in out.values()): raise RuntimeError("invalid/incomplete seed-104729 v6 prediction cache")
    z = np.load(V4 / "validation_predictions_v4.npz", allow_pickle=False)
    return out, z["thresholds"].astype(float), {"source": "v4/v6 immutable cache", "record_ids_sha256": ids_hash(ids)}
def eval_new_seed(seed, x, ids):
    d = OUT / f"seed_{seed}"; cache = d / "test_predictions"; cache.mkdir(exist_ok=True); n = len(ids)
    frozen = json.loads((d / "evaluation_freeze.json").read_text())
    if frozen.get("test_accessed_before_evaluation_freeze") is not False or frozen.get("evaluator_code_sha256") != sha(Path(__file__)):
        raise RuntimeError(f"evaluation freeze invalid for {seed}")
    z = np.load(d / "validation.npz", allow_pickle=False); thresholds = z["thresholds"].astype(float)
    done = d / "evaluation_complete.json"
    if done.exists():
        manifest = json.loads(done.read_text()); got = {}
        for name in CONDITIONS: got[name] = npz_prediction(cache / f"{name}.npz", n, name)
        if all(v is not None for v in got.values()) and manifest.get("record_ids_sha256") == ids_hash(ids): return got, thresholds, manifest
        raise RuntimeError(f"completed evaluation artifact invalid for {seed}")
    model = ECGResNetSmall().cuda()
    state = torch.load(d / "checkpoint.pt", weights_only=True, map_location="cuda")["state_dict"]; model.load_state_dict(state)
    norm = lambda a: ((a - m[None, :, None]) / s[None, :, None]).astype(np.float32)
    specs = {"clean": norm(x)}
    independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
    specs["RA_LL_reversal"] = norm(derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32))
    result = {}
    for ordinal, name in enumerate(CONDITIONS):
        target = cache / f"{name}.npz"; p = npz_prediction(target, n, name)
        if p is None:
            status(f"SEED_{seed}_INFERENCE", seed=seed, completed_conditions=ordinal, total_conditions=len(CONDITIONS), condition=name)
            a = specs[name] if name in specs else norm(np.where(np.isin(np.arange(12), MASKS[ordinal - 2])[None, :, None], m[None, :, None], x))
            p = inference(model, a); atomic_npz(target, p=p)
        result[name] = p
    manifest = {"version": "v7_seed_evaluation_complete", "seed": seed, "completed_utc": now(), "test_accessed": True,
                "record_count": n, "record_ids_sha256": ids_hash(ids), "condition_count": len(CONDITIONS),
                "condition_manifest_sha256": condition_manifest()["condition_manifest_sha256"],
                "checkpoint_sha256": sha(d / "checkpoint.pt"), "validation_sha256": sha(d / "validation.npz"),
                "evaluation_freeze_sha256": sha(d / "evaluation_freeze.json"),
                "prediction_files": {name: sha(cache / f"{name}.npz") for name in CONDITIONS}}
    atomic_json(done, manifest); del model; torch.cuda.empty_cache()
    return result, thresholds, manifest


def aggregate(y, per_seed, thresholds):
    status("AGGREGATION")
    base_rows, condition_rows, primary_rows = [], [], []
    for seed in SEEDS:
        if seed == 104729:
            b = json.loads((V4 / "BASELINE_ADEQUACY_v4.json").read_text())
            base_rows.append({"seed": seed, "decision": "PASS", "twa_auroc": b.get("twa_auroc"), "twa_auprc": b.get("twa_auprc"), "source": "v4 reused"})
        else:
            b = json.loads((OUT / f"seed_{seed}" / "baseline.json").read_text())
            base_rows.append({"seed": seed, "decision": b["decision"], "twa_auroc": b["auroc"], "twa_auprc": b["auprc"], "source": "v7"})
        for name in CONDITIONS: condition_rows.extend(metric_rows(seed, name, y, per_seed[seed][name], thresholds[seed]))
        for i, label in enumerate(LABELS):
            clean = average_precision_score(y[:, i], per_seed[seed]["clean"][:, i])
            ra = average_precision_score(y[:, i], per_seed[seed]["RA_LL_reversal"][:, i])
            masks = np.array([average_precision_score(y[:, i], per_seed[seed][name][:, i]) for name in MASK_NAMES])
            primary_rows.append({"seed": seed, "label": label, "clean_auprc": float(clean), "ra_ll_auprc": float(ra),
                                 "mean_mask_auprc": float(masks.mean()), "median_mask_auprc": float(np.median(masks)),
                                 "mask_p5_auprc": float(np.quantile(masks, .05)), "mask_p95_auprc": float(np.quantile(masks, .95)),
                                 "ra_ll_loss": float(clean-ra), "matched_contrast": float(masks.mean()-ra),
                                 "ra_ll_percentile_among_masks": float((masks <= ra).mean())})
    write_csv(OUT / "BASELINE_RESULTS_BY_SEED.csv", base_rows, list(base_rows[0]))
    write_csv(OUT / "CONDITION_RESULTS_BY_SEED.csv", condition_rows, list(condition_rows[0]))
    write_csv(OUT / "PRIMARY_CONTRASTS_BY_SEED.csv", primary_rows, list(primary_rows[0]))
    return primary_rows
def bootstrap(y, per_seed):
    n = len(y); idir = OUT / "bootstrap_indices.npy"; statefile = OUT / "bootstrap_state.npz"; progress = OUT / "BOOTSTRAP_PROGRESS.json"
    if idir.exists(): indices = np.load(idir, allow_pickle=False)
    else:
        indices = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
        tmp = idir.with_name(idir.name + ".tmp.npy"); np.save(tmp, indices); os.replace(tmp, idir)
    if indices.shape != (N_BOOT, n): raise RuntimeError("bootstrap index state mismatch")
    if statefile.exists():
        z = np.load(statefile, allow_pickle=False); done, contrasts = z["done"].astype(bool), z["contrasts"]
        if contrasts.shape != (N_BOOT, len(SEEDS), len(LABELS)): raise RuntimeError("bootstrap state shape mismatch")
    else: done, contrasts = np.zeros(N_BOOT, bool), np.full((N_BOOT, len(SEEDS), len(LABELS)), np.nan, np.float32)
    for b in range(N_BOOT):
        if done[b]: continue
        ix = indices[b]
        for si, seed in enumerate(SEEDS):
            for li in range(3):
                ra = average_precision_score(y[ix, li], per_seed[seed]["RA_LL_reversal"][ix, li])
                mask_mean = np.mean([average_precision_score(y[ix, li], per_seed[seed][name][ix, li]) for name in MASK_NAMES])
                contrasts[b, si, li] = mask_mean - ra
        done[b] = True
        if done.sum() % 25 == 0 or done.all():
            atomic_npz(statefile, done=done, contrasts=contrasts)
            atomic_json(progress, {"completed_replicates": int(done.sum()), "total_replicates": N_BOOT,
                                   "updated_utc": now(), "indices_sha256": sha(idir), "state_sha256": sha(statefile)})
            status(f"BOOTSTRAP_{int(done.sum())}_OF_{N_BOOT}", completed_replicates=int(done.sum()), total_replicates=N_BOOT)
    agg = contrasts.mean(axis=1)
    rows = []
    for li, label in enumerate(LABELS):
        a = agg[:, li]; rows.append({"label": label, "mean_contrast": float(a.mean()), "low95": float(np.quantile(a, .025)),
                                     "high95": float(np.quantile(a, .975)), "replicates": N_BOOT,
                                     "bootstrap_unit": "canonical_record", "aggregate": "mean_of_three_seed_contrasts"})
    write_csv(OUT / "MULTISEED_BOOTSTRAP_SUMMARY.csv", rows, list(rows[0])); return rows
def finalize(primary, summary):
    tw = {r["seed"]: r for r in primary if r["label"] == "TWA"}; bs = {r["label"]: r for r in summary}
    all_positive = all(r["matched_contrast"] > 0 for r in tw.values())
    base_ok = all(json.loads((OUT / f"seed_{s}" / "baseline.json").read_text())["decision"] == "PASS" for s in NEW_SEEDS)
    first4 = base_ok and all_positive and bs["TWA"]["mean_contrast"] >= .05 and bs["TWA"]["low95"] > 0
    morphology = first4 and bs["TWA"]["mean_contrast"] >= bs["SB"]["mean_contrast"] + .05 and bs["TWA"]["mean_contrast"] >= bs["STach"]["mean_contrast"] + .05
    decision = "FREEZE_NINGBO_MORPHOLOGY_CONFIRMATION" if morphology else ("FREEZE_NINGBO_GENERAL_FAILURE_CONFIRMATION" if first4 else "STOP_SEED_UNSTABLE")
    conclusion = {"decision": decision, "completed_utc": now(), "exploratory": True, "ningbo_accessed": False,
                  "baseline_all_pass": base_ok, "all_seed_twa_contrasts_positive": all_positive,
                  "seed_twa_contrasts": {str(k): v["matched_contrast"] for k, v in tw.items()}, "bootstrap": bs,
                  "frozen_protocol_sha256": sha(PROTO).upper()}
    atomic_json(OUT / "conclusions_v7.json", conclusion)
    (OUT / "SEED_STABILITY_ANALYSIS.md").write_text("# v7 seed stability analysis\n\n" + json.dumps(conclusion, indent=2) + "\n", encoding="utf-8")
    (OUT / "FINAL_CHAPMAN_MULTISEED_DECISION.md").write_text("# " + decision + "\n\n" + json.dumps(conclusion, indent=2) + "\n", encoding="utf-8")
    if decision.startswith("FREEZE_NINGBO"):
        draft = {"status": "DRAFT_ONLY_NOT_EXECUTED", "scope": "untouched Ningbo confirmation; no access performed by Chapman v7",
                 "seeds": [104729, 130363, 155921], "model_family": "ECGResNet-small", "endpoints": ["164934002", "426177001", "427084000"],
                 "primary_fault": "RA_LL_reversal", "comparator": "all C(12,5)=792 five-displayed-lead masks", "bootstrap": {"replicates": 1000, "level": "scenario", "synchronized": True},
                 "replication": "same-direction result on untouched Ningbo", "no_tuning_after_access": True,
                 "estimate": {"compute": "three CUDA seed trainings plus 3 x 794 inference conditions and 1,000 bootstrap replicates", "storage": "approximately 3-6 GB prediction/checkpoint/log workspace"}}
        atomic_json(ROOT / "configs" / "FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v1_DRAFT.json", draft)
    readme = ROOT / "README.md"; text = readme.read_text(encoding="utf-8")
    block = "<!-- V7_STATUS_START -->\n## Chapman multiseed v7\n\nFinal frozen decision: `" + decision + "`. Chapman results remain exploratory; Ningbo was not accessed. See `outputs/phase2_chapman_multiseed_v7/`.\n<!-- V7_STATUS_END -->"
    text = re.sub(r"<!-- V7_STATUS_START -->.*?<!-- V7_STATUS_END -->", block, text, flags=re.S) if "<!-- V7_STATUS_START -->" in text else text + "\n\n" + block + "\n"
    readme.write_text(text, encoding="utf-8")
    return decision


def acquire_lock():
    lock = OUT / "v7_workflow.lock"; OUT.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try: pid = int(json.loads(lock.read_text()).get("pid", -1)); os.kill(pid, 0)
        except ProcessLookupError: lock.unlink()
        except Exception: raise RuntimeError("existing workflow lock cannot be safely cleared")
        else: raise RuntimeError(f"v7 workflow already active with PID {pid}")
    atomic_json(lock, {"pid": os.getpid(), "started_utc": now(), "entry": str(Path(__file__).resolve())}); return lock
def run():
    lock = acquire_lock(); started = time.monotonic()
    try:
        status("PREPARATION"); freeze_evaluations()
        global m, s
        z = np.load(V4 / "preprocessing_v4.npz", allow_pickle=False); m, s = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        ids = test_ids(); y = np.stack([labels(r) for r in ids]).astype(np.int8); x = np.stack([raw(r) for r in ids]).astype(np.float32)
        if x.shape != (len(ids), 12, 1000) or not np.isfinite(x).all(): raise RuntimeError("test waveform integrity failure")
        per_seed, thresholds = {}, {}
        for seed in NEW_SEEDS:
            per_seed[seed], thresholds[seed], _ = eval_new_seed(seed, x, ids)
        status("SEED_104729_CACHE_VALIDATION"); per_seed[104729], thresholds[104729], _ = load_seed104(ids)
        primary = aggregate(y, per_seed, thresholds); summary = bootstrap(y, per_seed); decision = finalize(primary, summary)
        status("COMPLETE", decision=decision, elapsed_hours=(time.monotonic()-started)/3600)
    except Exception as exc:
        status("FAILED", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if lock.exists(): lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("smoke", "prepare", "run")); args = parser.parse_args()
    if args.command == "smoke": smoke()
    elif args.command == "prepare": freeze_evaluations()
    else: run()
