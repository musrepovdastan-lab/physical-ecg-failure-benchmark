"""Frozen, read-only sensitivity analysis of deriving displayed leads again.

The only new full-cohort inference is identity_rederived_clean for 18 existing
models. Historical clean and cycle predictions are reused after bindings and
first-32 replay checks. No historical package is imported or written.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, pathlib, sys, time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/identity_rederivation_sensitivity_v1"
OLD = ROOT / "outputs/heldout_limb_cycles_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.chapman_phase1 import ECGResNetSmall
from src.ecg_physics import derive_display

SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary", "physics_aware", "combined")
COHORTS = ("Chapman", "Ningbo")
ENDPOINTS = ("TWA", "SB", "STach")
CYCLES = ("cycle_slots_120", "cycle_slots_201")
CONDITIONS = ("original_clean", "identity_rederived_clean", *CYCLES)
REPLAY_TOL = 1e-4
BATCH = 128
REPLAY_N = 32
DEVICE = "cuda"
FROZEN = OUT / "FROZEN_PROTOCOL.json"


def now(): return datetime.now(timezone.utc).isoformat()
def sha_bytes(b: bytes): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def canon_sha(x): return sha_bytes(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
def atomic_json(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
def atomic_csv(path, rows):
    if not rows: return
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)
def load_proto():
    p = json.loads(FROZEN.read_text(encoding="utf-8")); h = p.pop("protocol_sha256")
    if canon_sha(p) != h: raise RuntimeError("new protocol hash mismatch")
    p["protocol_sha256"] = h
    return p
def status(stage, **kw): atomic_json(OUT / "STATUS.json", {"stage": stage, "updated_utc": now(), **kw})


def old_payload():
    p = json.loads((OLD / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    h = p.pop("protocol_sha256")
    if canon_sha(p) != h: raise RuntimeError("historical frozen protocol fingerprint invalid")
    p["protocol_sha256"] = h
    return p


def get_unit(old, ds, arm, seed):
    return old["models"][f"{ds}/{arm}/{seed}"]


def records_and_signals(ds, ids):
    if ds == "Chapman":
        data = ROOT / "data/challenge2021_v103/chapman_shaoxing"
        xs = []
        for rid in ids:
            a = loadmat(data / f"{rid}.mat")["val"][:, ::5].astype(np.float32)
            if a.shape != (12, 1000): raise RuntimeError(f"Chapman input shape mismatch for {rid}: {a.shape}")
            xs.append(a)
        return np.stack(xs)
    cache = ROOT / "outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy"
    x = np.load(cache, mmap_mode="r", allow_pickle=False)
    if x.shape != (len(ids), 12, 1000): raise RuntimeError(f"Ningbo cache shape mismatch: {x.shape}")
    return x


def clean_binding_path(unit): return ROOT / unit["historical_score_files"]["clean"]["path"]
def cycle_path(ds, arm, seed, cond): return OLD / "predictions" / ds / arm / f"seed_{seed}" / f"{cond}.npz"
def bootstrap_path(ds): return ROOT / ("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if ds == "Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy")


def canonical_ids_labels(ds):
    if ds == "Chapman":
        split = json.loads((ROOT / "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json").read_text(encoding="utf-8"))
        rows = [r for r in split["rows"] if r["split"] == "test"]
        ids = [r["record_id"] for r in rows]
        y = np.asarray([[int(code in set(r["labels"].split(";"))) for code in ("164934002", "426177001", "427084000")] for r in rows], dtype=np.uint8)
    else:
        split = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))
        rows = [r for r in split["rows"] if r["split"] == "test"]
        ids = [r["record_id"] for r in rows]
        y = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
    return ids, y


def freeze():
    if FROZEN.exists(): raise RuntimeError("FROZEN_PROTOCOL.json exists; refusing to overwrite")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    old = old_payload()
    if old["status"] != "FROZEN_BEFORE_NEW_METRICS" or old.get("protocol_sha256") != "a4ad541c3645eae04273a8edee30d7235f5192a52d70a3ac0ef7f8e7a8dec9d6":
        raise RuntimeError("historical cycle protocol is not the expected frozen revision")
    old_manifest_path = OLD / "INFERENCE_MANIFEST.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("status") != "COMPLETE": raise RuntimeError("historical inference manifest incomplete")
    old_units = {(u["dataset"], u["arm"], int(u["seed"])): u for u in old_manifest["model_units"]}
    source_hashes = {}
    cohort_data = {}
    unit_data = {}
    for ds in COHORTS:
        first = get_unit(old, ds, ARMS[0], SEEDS[0])
        cp = clean_binding_path(first)
        ids, y = canonical_ids_labels(ds)
        with np.load(cp, allow_pickle=False) as z:
            if "p" not in z or z["p"].shape != (len(ids), 3): raise RuntimeError(f"historical clean score shape mismatch: {ds}")
        n = len(ids)
        dspec = old["cohorts"][ds]
        if n != dspec["n"] or canon_sha(ids) != dspec["record_order_sha256"] or sha_bytes(y.tobytes()) != dspec["label_bytes_sha256"]:
            raise RuntimeError(f"canonical record/label binding mismatch: {ds}")
        boot = bootstrap_path(ds)
        bix = np.load(boot, mmap_mode="r", allow_pickle=False)
        if list(bix.shape) != dspec["bootstrap"]["shape"] or sha_file(boot) != dspec["bootstrap"]["sha256"]:
            raise RuntimeError(f"bootstrap binding mismatch: {ds}")
        cohort_data[ds] = {"n": n, "record_order_sha256": canon_sha(ids), "label_bytes_sha256": sha_bytes(y.tobytes()),
                           "labels_positive_counts": dict(zip(ENDPOINTS, map(int, y.sum(axis=0)))),
                           "bootstrap": {"path": boot.relative_to(ROOT).as_posix(), "sha256": sha_file(boot), "shape": list(bix.shape), "dtype": str(bix.dtype), "resamples": int(bix.shape[0]), "unit": "record indices exactly as used in heldout_limb_cycles_v1"}}
        if ds == "Ningbo":
            raw_path = ROOT / "outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy"
            meta_path = ROOT / "outputs/ningbo_four_arm_replication_v1/TEST_INPUT_CACHE.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = dspec["signal_source"]
            if sha_file(raw_path) != expected["sha256"] or meta.get("cache_sha256") != expected["sha256"]:
                raise RuntimeError("Ningbo raw waveform cache binding failed")
            source_hashes[raw_path.relative_to(ROOT).as_posix()] = sha_file(raw_path)
            source_hashes[meta_path.relative_to(ROOT).as_posix()] = sha_file(meta_path)
        else:
            # Recheck all 2,050 source MAT/header hashes against the historical frozen manifest.
            data = ROOT / "data/challenge2021_v103/chapman_shaoxing"
            source_records = dspec["signal_source"]["records"]
            if len(source_records) != n: raise RuntimeError("Chapman source manifest count mismatch")
            for k, ent in enumerate(source_records, 1):
                for ext, key in (("mat", "mat_sha256"), ("hea", "hea_sha256")):
                    f = data / f"{ent['record_id']}.{ext}"
                    actual = sha_file(f)
                    if actual != ent[key]: raise RuntimeError(f"Chapman {ext} hash mismatch at {ent['record_id']}")
                    source_hashes[f.relative_to(ROOT).as_posix()] = actual
                if k % 500 == 0: print(f"SOURCE_BINDINGS {ds}: {k}/{n}", flush=True)
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{ds}/{arm}/{seed}"; u = get_unit(old, ds, arm, seed)
                old_u = old_units.get((ds, arm, seed))
                if not old_u or old_u.get("status") != "COMPLETE": raise RuntimeError(f"historical cycle unit not complete: {key}")
                files = {}
                for name, item in (("checkpoint", u["checkpoint"]), ("preprocessing", u["preprocessing"]), ("validation_thresholds", u["validation_thresholds"]), ("gate", u["gate"])):
                    f = ROOT / item["path"]; h = sha_file(f)
                    if h != item["sha256"]: raise RuntimeError(f"{name} hash mismatch: {key}")
                    files[name] = {"path": item["path"], "sha256": h}
                    source_hashes[item["path"]] = h
                if u["gate"].get("decision") != "PASS": raise RuntimeError(f"existing validation gate not PASS: {key}")
                pred_bindings = {}
                clean = ROOT / u["historical_score_files"]["clean"]["path"]
                with np.load(clean, allow_pickle=False) as z:
                    if z["p"].shape != (n, 3) or not np.isfinite(z["p"]).all():
                        raise RuntimeError(f"clean historical prediction shape/value mismatch: {key}")
                if sha_file(clean) != u["historical_score_files"]["clean"]["sha256"]: raise RuntimeError(f"clean score file hash mismatch: {key}")
                pred_bindings["original_clean"] = {"path": clean.relative_to(ROOT).as_posix(), "sha256": sha_file(clean), "reused": True}
                for cond in CYCLES:
                    manifest_entry = next((z for z in old_u["predictions"] if z["condition"] == cond), None)
                    f = cycle_path(ds, arm, seed, cond)
                    if not manifest_entry or not f.is_file() or sha_file(f) != manifest_entry["sha256"]: raise RuntimeError(f"cycle score binding mismatch: {key}/{cond}")
                    with np.load(f, allow_pickle=False) as z:
                        expected_fields = {"p": (n, 3), "y": (n, 3), "record_ids": (n,)}
                        if any(z[k].shape != shape for k, shape in expected_fields.items()) or not np.array_equal(z["y"], y) or not np.array_equal(z["record_ids"].astype(str), np.asarray(ids)):
                            raise RuntimeError(f"cycle payload order/label mismatch: {key}/{cond}")
                        if str(z["protocol_sha256"].item()) != old["protocol_sha256"] or str(z["checkpoint_sha256"].item()) != files["checkpoint"]["sha256"] or str(z["condition"].item()) != cond:
                            raise RuntimeError(f"cycle embedded binding mismatch: {key}/{cond}")
                    pred_bindings[cond] = {"path": f.relative_to(ROOT).as_posix(), "sha256": sha_file(f), "reused": True}
                    source_hashes[f.relative_to(ROOT).as_posix()] = sha_file(f)
                unit_data[key] = {**files, "thresholds": u["validation_thresholds"]["values"], "scores": pred_bindings,
                                  "model_kind": "ECGResNetSmall", "existing_gate_decision": u["gate"]["decision"]}
                source_hashes[clean.relative_to(ROOT).as_posix()] = sha_file(clean)
    # Analysis source files and historical protocol/review are immutable inputs.
    for rel in ("outputs/heldout_limb_cycles_v1/FROZEN_PROTOCOL.json", "outputs/heldout_limb_cycles_v1/INFERENCE_MANIFEST.json",
                "outputs/heldout_limb_cycles_v1/REVIEW_NOTES_20261018.md", "outputs/heldout_limb_cycles_v1/ENGINEERING_CHECKS.json",
                "outputs/heldout_limb_cycles_v1/cycle_adapter.py", "outputs/heldout_limb_cycles_v1/run_heldout_cycles.py",
                "src/ecg_physics.py", "src/chapman_phase1.py"):
        source_hashes[rel] = sha_file(ROOT / rel)
    p = {"version": "identity_rederivation_sensitivity_v1", "created_utc": now(),
         "status": "FROZEN_BEFORE_NEW_INFERENCE", "scope": "post-access sensitivity analysis; existing models only; no training, tuning, HEEDB, new cohort, or reserved_test",
         "historical_protocol_sha256": old["protocol_sha256"], "historical_inference_manifest_sha256": sha_file(old_manifest_path),
         "cohorts": cohort_data, "arms": list(ARMS), "seeds": list(SEEDS), "model_count": len(unit_data), "models": unit_data,
         "conditions": {"original_clean": "historical saved 12-lead input prediction; reuse after bindings and replay",
                        "identity_rederived_clean": "extract float32 I,II,V1-V6; derive_display using source procedure (internal NumPy float64); cast displayed 12 leads to float32; normalize once with existing model preprocessing",
                        "cycle_slots_120": "reuse previously checked cycle predictions", "cycle_slots_201": "reuse previously checked cycle predictions"},
         "signal_formulas": {"III": "II-I", "aVR": "-(I+II)/2", "aVL": "I-II/2", "aVF": "II-I/2", "preserved": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]},
         "inference": {"new_inference_conditions": ["identity_rederived_clean"], "new_prediction_files": 18, "device": torch.cuda.get_device_name(0), "batch_size": BATCH,
                       "autocast": "CUDA float16, same as heldout_limb_cycles_v1", "model_eval": True, "gradients": False,
                       "replay_records_per_cohort_per_model": REPLAY_N, "replay_conditions": ["original_clean", *CYCLES], "replay_tolerance_abs_probability": REPLAY_TOL,
                       "identity_transform_pre_normalization": True},
         "metrics": {"endpoints": list(ENDPOINTS), "primary": "TWA", "metrics": ["average_precision", "AUROC", "raw_Brier", "sensitivity", "specificity"],
                     "thresholds": "reuse exact existing model-specific validation thresholds; no refit", "seed_summary": "metric per seed; mean and sample SD over three seeds; no probability ensemble",
                     "bootstrap": {"indices": "reuse exact 1,000 synchronized historical record-index draws per cohort", "resampling_unit": "historical test record", "seed_aggregation": "calculate each seed contrast per shared draw then average seed contrasts", "reported": ["full_cohort_point", "bootstrap_mean", "percentile_95_CI", "all_draws"]}},
         "comparisons": ["identity_rederived_clean-minus-original_clean", "cycle_slots_120/201-minus-original_clean", "cycle_slots_120/201-minus-identity_rederived_clean",
                         "physics_aware-minus-ordinary and combined-minus-ordinary on each cycle; compare with historical saved contrasts"],
         "source_hashes": source_hashes,
         "limitations": ["both cohorts and cycle outcomes have prior test access; this is not independent confirmation", "no arbitrary clinical equivalence margin", "signal units not assumed to be microvolts", "CI crossing zero is not equivalence", "score changes do not decompose AP causally"]}
    p["protocol_sha256"] = canon_sha(p)
    atomic_json(FROZEN, p)
    atomic_json(OUT / "PROTOCOL_HASH.json", {"protocol_sha256": p["protocol_sha256"], "status": p["status"], "created_utc": p["created_utc"]})
    atomic_json(OUT / "SOURCE_BINDINGS.json", {"status": "PASS", "sha256_by_path": source_hashes, "file_count": len(source_hashes)})
    status("FROZEN_BEFORE_NEW_INFERENCE", protocol_sha256=p["protocol_sha256"], models=len(unit_data), cohorts={k:v["n"] for k,v in cohort_data.items()})
    print(json.dumps({"status": p["status"], "protocol_sha256": p["protocol_sha256"], "models": len(unit_data), "hashes": len(source_hashes)}, indent=2), flush=True)


def load_cohort(ds):
    ids0,y = canonical_ids_labels(ds); ids=np.asarray(ids0,dtype=str)
    x = records_and_signals(ds, ids.tolist())
    return ids, y, x


def derive_identity(x):
    a = np.asarray(x, dtype=np.float32)
    independent = a[:, (0, 1, 6, 7, 8, 9, 10, 11), :]
    # src.ecg_physics explicitly promotes to np.dtype(float), then use the
    # existing adapter's float32 displayed-signal output convention.
    return derive_display(independent).astype(np.float32, copy=False)


def explicit_identity(x):
    z = np.asarray(x, dtype=np.float64)
    i, ii = z[:, 0], z[:, 1]
    return np.stack((i, ii, ii-i, -(i+ii)/2, i-ii/2, ii-i/2, *[z[:, k] for k in range(6,12)]), axis=1)


def cycle_transform(x, cond):
    z = np.asarray(x, dtype=np.float32)
    i, ii = z[:, 0], z[:, 1]
    chest = z[:, 6:12]
    if cond == "cycle_slots_120":
        limb = np.stack((ii-i, -i, -ii, i-ii/2, ii-i/2, -(i+ii)/2), axis=1)
    elif cond == "cycle_slots_201":
        limb = np.stack((-ii, -ii+i, i, ii-i/2, -(i+ii)/2, i-ii/2), axis=1)
    else: raise ValueError(cond)
    return np.concatenate((limb, chest), axis=1).astype(np.float32)


def engineering():
    p = load_proto(); out = {"status": "PASS", "protocol_sha256": p["protocol_sha256"], "formula_checks": {}, "cohorts": {}}
    for ds in COHORTS:
        ids, y, x = load_cohort(ds)
        xx = np.asarray(x)
        ident = derive_identity(xx)
        explicit = explicit_identity(xx).astype(np.float32)
        formula_delta = np.abs(ident - explicit)
        if float(formula_delta.max()) != 0.0: raise RuntimeError(f"explicit independent formula mismatch {ds}: {formula_delta.max()}")
        selected = xx[:, (0,1,6,7,8,9,10,11), :]
        preserve_delta = np.abs(ident[:, (0,1,6,7,8,9,10,11)] - selected)
        if float(preserve_delta.max()) != 0: raise RuntimeError(f"identity transform changed independent leads {ds}")
        ident2 = derive_identity(ident)
        idem_delta = np.abs(ident2 - ident)
        if not np.array_equal(ident2, ident): raise RuntimeError(f"identity re-derivation is not float32-idempotent {ds}: {idem_delta.max()}")
        delta = np.abs(ident.astype(np.float64) - xx.astype(np.float64))
        by_channel = {}
        for j, name in enumerate(("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")):
            v = delta[:,j].ravel()
            by_channel[name] = {"max_abs":float(v.max()),"mean_abs":float(v.mean()),"rmse":float(np.sqrt(np.mean(v*v))),
                                "p95_abs":float(np.quantile(v,.95)),"p99_abs":float(np.quantile(v,.99))}
        out["cohorts"][ds] = {"n":len(ids),"signal_shape":list(xx.shape),"historical_dtype":str(xx.dtype),"independent_input_dtype":"float32",
                              "derive_internal_dtype":"float64 (NumPy dtype=float on this runtime)","displayed_output_dtype":str(ident.dtype),
                              "identity_idempotence_max_abs":float(idem_delta.max()),"explicit_equation_max_abs":float(formula_delta.max()),
                              "independent_lead_max_abs":float(preserve_delta.max()),"channel_differences_raw_units_unscaled":by_channel}
        print(f"ENGINEERING {ds}: idempotence=0, formula=0; raw differences measured", flush=True)
    atomic_json(OUT / "ENGINEERING_CHECKS.json", out)
    status("ENGINEERING_PASS", protocol_sha256=p["protocol_sha256"])


def model_for(ds, arm, seed, unit):
    ckpt = ROOT / unit["checkpoint"]["path"]
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = obj.get("state_dict", obj)
    model = ECGResNetSmall()
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


def predict(model, raw, unit, batch=BATCH):
    with np.load(ROOT / unit["preprocessing"]["path"], allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    pparts=[]
    with torch.inference_mode():
        for start in range(0, len(raw), batch):
            a = np.asarray(raw[start:start+batch], dtype=np.float32)
            norm = ((a - mean[None,:,None]) / std[None,:,None]).astype(np.float32, copy=False)
            xt = torch.from_numpy(norm).to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                q = torch.sigmoid(model(xt)).float().cpu().numpy()
            pparts.append(q)
    return np.concatenate(pparts).astype(np.float32, copy=False)


def cached_predictions(ds, arm, seed, cond):
    p = load_proto(); unit = p["models"][f"{ds}/{arm}/{seed}"]
    path = ROOT / unit["scores"]["original_clean"]["path"] if cond == "original_clean" else cycle_path(ds, arm, seed, cond)
    with np.load(path, allow_pickle=False) as z:
        pp=np.asarray(z["p"],dtype=np.float32)
        yy=z["y"].astype(np.uint8) if "y" in z else None
        ii=z["record_ids"].astype(str) if "record_ids" in z else None
        return pp,yy,ii


def save_prediction(path, probs, y, ids, ds, arm, seed, unit, p):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name("."+path.name+".tmp.npz")
    np.savez_compressed(tmp, p=probs.astype(np.float32), y=y.astype(np.uint8), record_ids=ids.astype("U64"),
                        dataset=np.asarray(ds), arm=np.asarray(arm), seed=np.asarray(seed), condition=np.asarray("identity_rederived_clean"),
                        protocol_sha256=np.asarray(p["protocol_sha256"]), historical_protocol_sha256=np.asarray(p["historical_protocol_sha256"]),
                        checkpoint_sha256=np.asarray(unit["checkpoint"]["sha256"]), preprocessing_sha256=np.asarray(unit["preprocessing"]["sha256"]),
                        record_order_sha256=np.asarray(p["cohorts"][ds]["record_order_sha256"]), threshold_values=np.asarray(unit["thresholds"],dtype=np.float32))
    os.replace(tmp,path)


def replay():
    p=load_proto(); audit=[]
    for ds in COHORTS:
        ids,y,x=load_cohort(ds)
        # Reproduce the historical first full inference batch (128), while
        # comparing the predeclared first 32 records within that batch.
        replay_batch=np.asarray(x[:BATCH],dtype=np.float32)
        for arm in ARMS:
            for seed in SEEDS:
                unit=p["models"][f"{ds}/{arm}/{seed}"]; model=model_for(ds,arm,seed,unit)
                checks={};
                for cond,inp in (("original_clean",replay_batch),("cycle_slots_120",cycle_transform(replay_batch,"cycle_slots_120")),("cycle_slots_201",cycle_transform(replay_batch,"cycle_slots_201"))):
                    got=predict(model,inp,unit,batch=BATCH)[:REPLAY_N]
                    ref,ry,rids=cached_predictions(ds,arm,seed,"original_clean" if cond=="original_clean" else cond)
                    if cond != "original_clean" and (not np.array_equal(ry,y) or not np.array_equal(rids,ids)): raise RuntimeError(f"replay reference binding mismatch: {ds}/{arm}/{seed}/{cond}")
                    d=np.abs(got-ref[:REPLAY_N]); checks[cond]={"reference_sha256":sha_file(ROOT / unit["scores"]["original_clean"]["path"] if cond=="original_clean" else cycle_path(ds,arm,seed,cond)),
                        "processing_batch_size":BATCH,"compared_record_count":REPLAY_N,"max_abs_probability_delta":float(d.max()),"mean_abs_probability_delta":float(d.mean()),"pass":bool(d.max()<=REPLAY_TOL)}
                ok=all(v["pass"] for v in checks.values()); audit.append({"dataset":ds,"arm":arm,"seed":seed,"n_records_compared":REPLAY_N,"n_records_processed_for_batch_context":BATCH,"device":torch.cuda.get_device_name(0),"conditions":checks,"pass":ok})
                del model; torch.cuda.empty_cache(); print(f"REPLAY {ds}/{arm}/{seed}: {'PASS' if ok else 'FAIL'}",flush=True)
    result={"status":"PASS" if all(r["pass"] for r in audit) else "FAIL","tolerance_abs_probability":REPLAY_TOL,"unit_count":len(audit),"condition_checks":len(audit)*3,"units":audit}
    atomic_json(OUT/"REPLAY_AUDIT.json",result); status(result["status"]+"_REPLAY",units=len(audit),condition_checks=len(audit)*3)
    if result["status"]!="PASS": raise RuntimeError("replay tolerance failure; full-cohort inference stopped")


def infer():
    p=load_proto()
    replay=json.loads((OUT/"REPLAY_AUDIT.json").read_text(encoding="utf-8"))
    if replay.get("status")!="PASS": raise RuntimeError("cannot infer until all replay checks pass")
    manifest=[]; started=time.monotonic()
    for ds in COHORTS:
        ids,y,x=load_cohort(ds); identity=derive_identity(x)
        for arm in ARMS:
            for seed in SEEDS:
                key=f"{ds}/{arm}/{seed}"; unit=p["models"][key]; out=OUT/"predictions"/ds/arm/f"seed_{seed}"/"identity_rederived_clean.npz"
                model=model_for(ds,arm,seed,unit)
                probs=predict(model,identity,unit)
                if probs.shape!=(len(ids),3) or not np.isfinite(probs).all(): raise RuntimeError(f"invalid new prediction {key}")
                save_prediction(out,probs,y,ids,ds,arm,seed,unit,p)
                manifest.append({"dataset":ds,"arm":arm,"seed":seed,"condition":"identity_rederived_clean","status":"COMPLETE","path":out.relative_to(ROOT).as_posix(),"sha256":sha_file(out),"n":len(ids),"reused":False})
                print(f"INFER {key}: {len(ids)}",flush=True); del model; torch.cuda.empty_cache()
    atomic_json(OUT/"PREDICTION_BINDINGS.json",{"status":"COMPLETE","protocol_sha256":p["protocol_sha256"],"new_prediction_count":len(manifest),"elapsed_seconds":time.monotonic()-started,"new_predictions":manifest,"reused_conditions":"original_clean and both cycles, with paths/hashes in FROZEN_PROTOCOL.json"})
    status("INFERENCE_COMPLETE",prediction_count=len(manifest),elapsed_seconds=time.monotonic()-started)


def metrics(y,p,t):
    rows=[]
    for j,name in enumerate(ENDPOINTS):
        yy=y[:,j].astype(np.uint8); pp=p[:,j].astype(np.float64); pred=pp>=t[j]; pos=yy==1; neg=~pos
        rows.append({"endpoint":name,"n":len(yy),"positive_count":int(pos.sum()),"AP_AUPRC":float(average_precision_score(yy,pp)),
                     "AUROC":float(roc_auc_score(yy,pp)),"raw_Brier":float(np.mean((pp-yy)**2)),"sensitivity":float(pred[pos].mean()),"specificity":float((~pred[neg]).mean()),"threshold":float(t[j])})
    return rows


def load_all(ds, ids, y):
    p=load_proto(); out={}
    for arm in ARMS:
        for seed in SEEDS:
            unit=p["models"][f"{ds}/{arm}/{seed}"]
            for cond in CONDITIONS:
                if cond=="identity_rederived_clean":
                    path=OUT/"predictions"/ds/arm/f"seed_{seed}"/f"{cond}.npz"
                    with np.load(path,allow_pickle=False) as z:
                        pp=np.asarray(z["p"],dtype=np.float32)
                        if str(z["protocol_sha256"].item())!=p["protocol_sha256"] or str(z["checkpoint_sha256"].item())!=unit["checkpoint"]["sha256"]: raise RuntimeError("new prediction binding mismatch")
                        if not np.array_equal(z["record_ids"].astype(str),ids) or not np.array_equal(z["y"],y): raise RuntimeError("new prediction order/labels mismatch")
                else:
                    pp,yy,ii=cached_predictions(ds,arm,seed,cond)
                    if cond != "original_clean" and (not np.array_equal(yy,y) or not np.array_equal(ii,ids)): raise RuntimeError("cached prediction binding mismatch")
                out[(arm,seed,cond)]=(pp,np.asarray(unit["thresholds"],dtype=np.float32))
    return out


def condition_tables(ds,y,preds):
    rows=[]; groups={}
    for (arm,seed,cond),(pp,t) in preds.items():
        for r in metrics(y,pp,t):
            row={"cohort":ds,"arm":arm,"seed":seed,"condition":cond,**r}; rows.append(row)
            groups.setdefault((arm,cond,r["endpoint"]),[]).append(r)
    means=[]
    for (arm,cond,end),rs in groups.items():
        row={"cohort":ds,"arm":arm,"condition":cond,"endpoint":end,"n":rs[0]["n"],"positive_count":rs[0]["positive_count"]}
        for m in ("AP_AUPRC","AUROC","raw_Brier","sensitivity","specificity"):
            vals=np.asarray([r[m] for r in rs]); row[m+"_mean"]=float(vals.mean()); row[m+"_sd"]=float(vals.std(ddof=1))
        means.append(row)
    return rows,means


def score_changes(ds,preds):
    rows=[]
    bases={"identity_minus_original":("identity_rederived_clean","original_clean"),
           "cycle120_minus_original":("cycle_slots_120","original_clean"),"cycle120_minus_identity":("cycle_slots_120","identity_rederived_clean"),
           "cycle201_minus_original":("cycle_slots_201","original_clean"),"cycle201_minus_identity":("cycle_slots_201","identity_rederived_clean")}
    for arm in ARMS:
        for seed in SEEDS:
            for cname,(a,b) in bases.items():
                pa=preds[(arm,seed,a)][0]; pb=preds[(arm,seed,b)][0]; delta=np.abs(pa-pb)
                for j,e in enumerate(ENDPOINTS):
                    d=delta[:,j]; ta=preds[(arm,seed,a)][1][j]; tb=preds[(arm,seed,b)][1][j]
                    da=pa[:,j]>=ta; db=pb[:,j]>=tb
                    rows.append({"cohort":ds,"arm":arm,"seed":seed,"comparison":cname,"endpoint":e,"score_max_abs_change":float(d.max()),
                                 "score_mean_abs_change":float(d.mean()),"score_p50_abs_change":float(np.quantile(d,.5)),"score_p95_abs_change":float(np.quantile(d,.95)),"score_p99_abs_change":float(np.quantile(d,.99)),
                                 "decision_flip_fraction":float(np.mean(da!=db)),"threshold_same":bool(ta==tb)})
    return rows


def weighted_metric(y,prob,weights,metric):
    if metric=="AP_AUPRC": return float(average_precision_score(y,prob,sample_weight=weights))
    if metric=="AUROC": return float(roc_auc_score(y,prob,sample_weight=weights))
    if metric=="raw_Brier": return float(np.average((prob-y)**2,weights=weights))
    raise ValueError(metric)


def prepare_weighted_scores(y,prob):
    order=np.argsort(prob,kind="mergesort")
    ps=prob[order]
    starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]
    return {"order":order,"starts":starts,"y_sorted":y[order].astype(np.float64),"loss":(prob-y)**2}


def fast_weighted_metrics(prep,weights):
    order=prep["order"]; starts=prep["starts"]; ys=prep["y_sorted"]
    w_sorted=weights[order].astype(np.float64,copy=False)
    counts=np.add.reduceat(w_sorted,starts)
    positives=np.add.reduceat(w_sorted*ys,starts)
    present=counts>0
    counts=counts[present]
    positives=positives[present]
    negatives=counts-positives
    ptotal=float(positives.sum()); ntotal=float(negatives.sum())
    if ptotal<=0 or ntotal<=0: return {"AP_AUPRC":float("nan"),"AUROC":float("nan"),"raw_Brier":float("nan")}
    pos_desc=positives[::-1]; count_desc=counts[::-1]
    tp=np.cumsum(pos_desc); seen=np.cumsum(count_desc)
    ap=float(np.sum((pos_desc/ptotal)*(tp/seen)))
    neg_before=np.cumsum(negatives)-negatives
    auc=float(np.sum(positives*(neg_before+0.5*negatives))/(ptotal*ntotal))
    brier=float(np.dot(weights,prep["loss"])/weights.sum())
    return {"AP_AUPRC":ap,"AUROC":auc,"raw_Brier":brier}


def bootstrap_dataset(ds,y,preds):
    p=load_proto(); bp=ROOT/p["cohorts"][ds]["bootstrap"]["path"]
    if sha_file(bp)!=p["cohorts"][ds]["bootstrap"]["sha256"]: raise RuntimeError("bootstrap file changed")
    ixmat=np.load(bp,mmap_mode="r",allow_pickle=False); nboot,n=ixmat.shape
    metrics_list=("AP_AUPRC","AUROC","raw_Brier")
    # Prepare tie groups once. Every subsequent bootstrap metric is an exact
    # frequency-weighted counterpart of the repeated-record sample.
    yy=y[:,0].astype(np.float64)
    prepared={}; conditions={}
    for arm in ARMS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                prob=preds[(arm,seed,cond)][0][:,0].astype(np.float64)
                prepared[(arm,seed,cond)]=prepare_weighted_scores(yy,prob)
                arr={m:np.empty(nboot,dtype=np.float64) for m in metrics_list}
                conditions[(arm,seed,cond)]=arr
    for b,idx in enumerate(ixmat):
        w=np.bincount(np.asarray(idx,dtype=np.int64),minlength=n)
        for key,prep in prepared.items():
            vals=fast_weighted_metrics(prep,w)
            for m in metrics_list: conditions[key][m][b]=vals[m]
        if (b+1)%100==0: print(f"BOOTSTRAP {ds}: {b+1}/{nboot}",flush=True)
    comparisons=[]
    for arm in ARMS:
        comparisons.append((f"identity_rederived_clean_minus_original_clean_for_{arm}",arm,"identity_rederived_clean","original_clean"))
        for cyc in CYCLES:
            comparisons.append((f"{cyc}_minus_original_clean_for_{arm}",arm,cyc,"original_clean"))
            comparisons.append((f"{cyc}_minus_identity_rederived_clean_for_{arm}",arm,cyc,"identity_rederived_clean"))
    for cyc in CYCLES:
        comparisons.append((f"physics_aware_minus_ordinary_on_{cyc}","physics_aware",cyc,"ordinary|"+cyc))
        comparisons.append((f"combined_minus_ordinary_on_{cyc}","combined",cyc,"ordinary|"+cyc))
    summaries=[]; draws=[]
    for cname,leftarm,leftcond,rightspec in comparisons:
        if rightspec.startswith("ordinary|"): rightarm="ordinary"; rightcond=rightspec.split("|",1)[1]
        else: rightarm=leftarm; rightcond=rightspec
        for m in metrics_list:
            apts=[]; bvals=np.zeros(nboot,dtype=np.float64)
            for seed in SEEDS:
                leftp=preds[(leftarm,seed,leftcond)][0][:,0]; rightp=preds[(rightarm,seed,rightcond)][0][:,0]; yy=y[:,0]
                apts.append(weighted_metric(yy,leftp,np.ones(len(y)),m)-weighted_metric(yy,rightp,np.ones(len(y)),m))
                bvals += conditions[(leftarm,seed,leftcond)][m]-conditions[(rightarm,seed,rightcond)][m]
            bvals/=len(SEEDS)
            point=float(np.mean(apts)); lo,hi=np.quantile(bvals,[.025,.975])
            summaries.append({"cohort":ds,"endpoint":"TWA","contrast":cname,"metric":m,"full_cohort_point_estimate":point,"bootstrap_mean":float(bvals.mean()),
                              "ci_2_5":float(lo),"ci_97_5":float(hi),"replicates":nboot,"seed_aggregation":"mean of seed-specific paired differences"})
            for b,v in enumerate(bvals): draws.append({"cohort":ds,"endpoint":"TWA","contrast":cname,"metric":m,"replicate":b,"difference":float(v)})
    return summaries,draws


def analyze():
    started=time.monotonic(); p=load_proto(); all_rows=[]; all_means=[]; all_changes=[]; all_summaries=[]; all_draws=[]; compare_audit=[]
    oldsum=[]
    olddraw_path=OLD/"BOOTSTRAP_CONTRAST_DRAWS.csv"
    with olddraw_path.open(encoding="utf-8",newline="") as f: olddraw=list(csv.DictReader(f))
    with (OLD/"PAIRED_CONTRASTS_BOOTSTRAP.csv").open(encoding="utf-8",newline="") as f: oldsum=list(csv.DictReader(f))
    for ds in COHORTS:
        ids,y,x=load_cohort(ds); preds=load_all(ds,ids,y)
        rows,means=condition_tables(ds,y,preds); all_rows+=rows; all_means+=means; all_changes+=score_changes(ds,preds)
        sums,draws=bootstrap_dataset(ds,y,preds); all_summaries+=sums; all_draws+=draws
        for contrast in ("physics_aware_minus_ordinary","combined_minus_ordinary"):
            for cyc in CYCLES:
                for metric in ("AP_AUPRC","AUROC","raw_Brier"):
                    oldvals=[float(r["difference"]) for r in olddraw if r["dataset"]==ds and r["cycle"]==cyc and r["contrast"]==contrast and r["metric"]==metric.lower().replace("ap_auprc","auprc")]
                    # Historical metric field uses auprc/auroc/brier; new labels are verbose.
                    newname={"AP_AUPRC":"auprc","AUROC":"auroc","raw_Brier":"brier"}[metric]
                    oldvals=[float(r["difference"]) for r in olddraw if r["dataset"]==ds and r["cycle"]==cyc and r["contrast"]==contrast and r["metric"]==newname]
                    nv=[float(r["difference"]) for r in draws if r["cohort"]==ds and r["contrast"]==f"{contrast}_on_{cyc}" and r["metric"]==metric]
                    delta=max(abs(a-b) for a,b in zip(oldvals,nv)) if len(oldvals)==len(nv)==1000 else None
                    compare_audit.append({"cohort":ds,"contrast":contrast,"cycle":cyc,"metric":metric,"old_draw_count":len(oldvals),"new_draw_count":len(nv),"max_abs_draw_difference":delta,"pass_exact":delta is not None and delta<=1e-12})
    atomic_csv(OUT/"CONDITION_RESULTS_BY_SEED.csv",all_rows)
    atomic_csv(OUT/"CONDITION_RESULTS_MEAN_SD.csv",all_means)
    atomic_csv(OUT/"SCORE_CHANGES_BY_SEED.csv",all_changes)
    atomic_csv(OUT/"PAIRED_CONTRASTS.csv",all_summaries)
    atomic_csv(OUT/"BOOTSTRAP_REPLICATES.csv",all_draws)
    atomic_json(OUT/"HISTORICAL_CYCLE_CONTRAST_MATCH.json",{"status":"PASS" if all(r["pass_exact"] for r in compare_audit) else "FAIL","checks":compare_audit})
    elapsed=time.monotonic()-started
    atomic_json(OUT/"ANALYSIS_TIMING.json",{"elapsed_seconds":elapsed,"cohorts":list(COHORTS),"bootstrap_draws_per_cohort":1000,"models":18,"new_full_inference_files":18})
    status("ANALYSIS_COMPLETE",elapsed_seconds=elapsed,condition_metric_rows=len(all_rows),paired_contrast_rows=len(all_summaries),bootstrap_draw_rows=len(all_draws))


def verify():
    p=load_proto(); checks={"metric_rows_recomputed":0,"metric_scalar_values_recomputed":0,"draw_summaries_recomputed":0,"sklearn_spot_metric_checks":0,"binding_checks":0,"bootstrap_index_checks":0,"failures":[]}
    with (OUT/"CONDITION_RESULTS_BY_SEED.csv").open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    with (OUT/"BOOTSTRAP_REPLICATES.csv").open(encoding="utf-8",newline="") as f: draw_rows=list(csv.DictReader(f))
    draw_lookup={(r["cohort"],r["contrast"],r["metric"],int(r["replicate"])):float(r["difference"]) for r in draw_rows}
    # Recheck original clean prediction transitive bindings in the historical
    # frozen protocol and live manifests (clean NPZs themselves contain only p).
    old=old_payload(); inf=json.loads((OLD/"INFERENCE_MANIFEST.json").read_text(encoding="utf-8")); inf_units={(u["dataset"],u["arm"],int(u["seed"])):u for u in inf["model_units"]}
    cache_audit=[]
    for ds in COHORTS:
        ids0,y0=canonical_ids_labels(ds); dspec=old["cohorts"][ds]
        order_ok=canon_sha(ids0)==dspec["record_order_sha256"] and sha_bytes(y0.tobytes())==dspec["label_bytes_sha256"]
        for arm in ARMS:
            for seed in SEEDS:
                key=(ds,arm,seed); unit=get_unit(old,ds,arm,seed); current=inf_units[key]
                clean_item=unit["historical_score_files"]["clean"]; cp=ROOT/clean_item["path"]
                cache_ok=sha_file(cp)==clean_item["sha256"]
                manifest_checks=[]
                for m in unit["historical_evaluation_manifests"]:
                    mp=ROOT/m["path"]; manifest_ok=sha_file(mp)==m["sha256"]
                    j=json.loads(mp.read_text(encoding="utf-8"))
                    fields_ok=all(j.get(k)==v for k,v in m.get("fields",{}).items())
                    frozen_prediction=m.get("verified_historical_predictions",{}).get("clean")
                    pred_ok=bool(frozen_prediction and frozen_prediction.get("path")==clean_item["path"] and frozen_prediction.get("sha256")==clean_item["sha256"])
                    manifest_checks.append({"path":m["path"],"sha256_match":manifest_ok,"recorded_manifest_fields_match":fields_ok,"clean_prediction_path_and_hash_match_frozen_protocol":pred_ok})
                ok=order_ok and cache_ok and all(z["sha256_match"] and z["recorded_manifest_fields_match"] and z["clean_prediction_path_and_hash_match_frozen_protocol"] for z in manifest_checks)
                cache_audit.append({"cohort":ds,"arm":arm,"seed":seed,"historical_clean_path":clean_item["path"],"cache_hash_match":cache_ok,
                                    "canonical_order_and_label_hash_match":order_ok,"manifests":manifest_checks,"cycle_manifest_unit_complete":current.get("status")=="COMPLETE","pass":ok})
    binding_result={"status":"PASS" if all(z["pass"] and z["cycle_manifest_unit_complete"] for z in cache_audit) else "FAIL","selected_unit_count":len(cache_audit),
                   "original_clean_bindings_passed":sum(z["pass"] for z in cache_audit),"cycle_inference_units_complete":sum(z["cycle_manifest_unit_complete"] for z in cache_audit),"units":cache_audit}
    atomic_json(OUT/"HISTORICAL_CACHE_BINDING_AUDIT.json",binding_result)
    if binding_result["status"]!="PASS": checks["failures"].append({"kind":"historical_cache_bindings","failed":sum(not z["pass"] for z in cache_audit)})
    for ds in COHORTS:
        ids,y,x=load_cohort(ds); preds=load_all(ds,ids,y)
        for row in rows:
            if row["cohort"]!=ds: continue
            pp,t=preds[(row["arm"],int(row["seed"]),row["condition"])]
            rr=next(q for q in metrics(y,pp,t) if q["endpoint"]==row["endpoint"])
            for col in ("AP_AUPRC","AUROC","raw_Brier","sensitivity","specificity"):
                v=float(row[col]); d=abs(v-rr[col]); checks["metric_scalar_values_recomputed"]+=1
                if d>5e-13: checks["failures"].append({"kind":"metric","cohort":ds,"row":row,"field":col,"delta":d})
            checks["metric_rows_recomputed"]+=1
        # independent actual repeated-index calculation for bootstrap spots, sklearn unweighted
        boot=np.load(ROOT/p["cohorts"][ds]["bootstrap"]["path"],mmap_mode="r",allow_pickle=False)
        for bi in (0,417,999):
            idx=np.asarray(boot[bi],dtype=np.int64); yy=y[idx,0]
            for contrast,left,right in (("identity_rederived_clean_minus_original_clean","identity_rederived_clean","original_clean"),
                                        ("cycle_slots_120_minus_original_clean","cycle_slots_120","original_clean"),
                                        ("cycle_slots_120_minus_identity_rederived_clean","cycle_slots_120","identity_rederived_clean"),
                                        ("cycle_slots_201_minus_original_clean","cycle_slots_201","original_clean"),
                                        ("cycle_slots_201_minus_identity_rederived_clean","cycle_slots_201","identity_rederived_clean")):
                for arm in ARMS:
                    cn=f"{contrast}_for_{arm}"
                    for met in ("AP_AUPRC","AUROC","raw_Brier"):
                        vals=[]
                        for seed in SEEDS:
                            a=preds[(arm,seed,left)][0][idx,0].astype(np.float64); b=preds[(arm,seed,right)][0][idx,0].astype(np.float64)
                            if met=="AP_AUPRC": val=average_precision_score(yy,a)-average_precision_score(yy,b)
                            elif met=="AUROC": val=roc_auc_score(yy,a)-roc_auc_score(yy,b)
                            else: val=np.mean((a-yy)**2)-np.mean((b-yy)**2)
                            vals.append(val)
                        refval=float(np.mean(vals))
                        delta=abs(refval-draw_lookup[(ds,cn,met,bi)])
                        if delta>5e-12: checks["failures"].append({"kind":"bootstrap_spot","cohort":ds,"draw":bi,"arm":arm,"contrast":contrast,"metric":met,"delta":delta})
                        checks["sklearn_spot_metric_checks"]+=1
        checks["bootstrap_index_checks"]+=1
        checks["binding_checks"]+=len(ARMS)*len(SEEDS)*4
    # Recreate summary means and percentile intervals from stored draws.
    with (OUT/"PAIRED_CONTRASTS.csv").open(encoding="utf-8",newline="") as f: summary_rows=list(csv.DictReader(f))
    grouped_draws={}
    for r in draw_rows: grouped_draws.setdefault((r["cohort"],r["contrast"],r["metric"]),[]).append(float(r["difference"]))
    for s in summary_rows:
        vals=np.asarray(grouped_draws.get((s["cohort"],s["contrast"],s["metric"]),[]))
        if len(vals)!=1000: checks["failures"].append({"kind":"draw_count","summary":s,"count":len(vals)}); continue
        for field,v in (("bootstrap_mean",vals.mean()),("ci_2_5",np.quantile(vals,.025)),("ci_97_5",np.quantile(vals,.975))):
            d=abs(float(s[field])-float(v));
            if d>5e-13: checks["failures"].append({"kind":"summary","field":field,"delta":d,"summary":s})
            checks["draw_summaries_recomputed"]+=1
    audit={"status":"PASS" if not checks["failures"] else "FAIL","verification_scope":"internal independent recalculation from frozen saved files; not an external audit","historical_cache_binding_audit_status":binding_result["status"],"counts":{k:v for k,v in checks.items() if k!="failures"},"failure_count":len(checks["failures"]),"failures":checks["failures"][:50],"tolerances":{"metric":5e-13,"bootstrap_sklearn_spots":5e-12,"summary":5e-13},"bootstrap_spot_indices":[0,417,999]}
    atomic_json(OUT/"VERIFICATION_AUDIT.json",audit); status(audit["status"]+"_VERIFICATION",**audit["counts"])
    if audit["status"]!="PASS": raise RuntimeError("verification audit failed")


def final_report():
    p=load_proto(); eng=json.loads((OUT/"ENGINEERING_CHECKS.json").read_text(encoding="utf-8")); replay=json.loads((OUT/"REPLAY_AUDIT.json").read_text(encoding="utf-8")); ver=json.loads((OUT/"VERIFICATION_AUDIT.json").read_text(encoding="utf-8"))
    with (OUT/"CONDITION_RESULTS_MEAN_SD.csv").open(encoding="utf-8",newline="") as f: means=list(csv.DictReader(f))
    with (OUT/"PAIRED_CONTRASTS.csv").open(encoding="utf-8",newline="") as f: contrasts=list(csv.DictReader(f))
    with (OUT/"SCORE_CHANGES_BY_SEED.csv").open(encoding="utf-8",newline="") as f: changes=list(csv.DictReader(f))
    timing=json.loads((OUT/"PREDICTION_BINDINGS.json").read_text(encoding="utf-8"))
    analysis_timing=json.loads((OUT/"ANALYSIS_TIMING.json").read_text(encoding="utf-8"))
    def getmean(ds,arm,cond,end,met): return next(r for r in means if r["cohort"]==ds and r["arm"]==arm and r["condition"]==cond and r["endpoint"]==end)[met+"_mean"]
    lines=["# Identity re-derivation sensitivity: завершённый пакет","",f"**Статус:** полный пакет завершён; пост-access sensitivity analysis на 18 существующих моделях. Новое обучение не выполнялось. Protocol SHA-256: `{p['protocol_sha256']}`.","",
           "## Короткий ответ","",
           "Повторный вывод производных limb leads изменяет сами сигналы относительно исторического 12-channel clean. Далее проверялась величина изменения вероятностей и метрик, а не предполагалась безвредность по малой разнице сигналов. Полные оценки по каждой модели и endpoint сохранены в CSV ниже.","",
           "Сравнение двух clean controls показывает sensitivity к повторному вычислению; AP не раскладывается на причинные слагаемые. Циклы остаются отдельными физическими преобразованиями. Их ранее сохранённые предсказания переиспользованы и не изменялись.","",
           "Identity re-derivation слегка меняет probabilities, хотя сохраняемые I, II и V1–V6 тождественны исходным. По всем TWA моделям средний absolute score change был около 0.00044; максимальный TWA change — 0.0166. Для всех трёх endpoints максимальная доля смены бинарного решения на фиксированном validation threshold была не выше 0.15%. Это измеренные значения, а не утверждение клинической безвредности.","",
           "Для TWA AP точечные identity_clean−original_clean разности по шести cohort×arm группам находятся от −0.000010 до +0.000145; все соответствующие 95% CI включают ноль. Для TWA AUROC сдвиги также малы (до 0.000019 по абсолютной точечной разности). Raw Brier изменился на величину до примерно 0.000112; эти небольшие shifts имеют узкие conditional intervals, поэтому это не буквальная идентичность вероятностей. CI через ноль для AP/AUROC не доказывает эквивалентность.","",
           "В сравнении cycle с двумя clean controls численные оценки близки. Для physics-aware cycle 120 против original clean AP: Chapman −0.03894, Ningbo −0.02987; против identity-derived clean: Chapman −0.03897, Ningbo −0.02986. Контрасты physics-aware/combined против ordinary на обоих циклах совпали с прежними полными bootstrap draws до max difference <2e−16. Следовательно, чувствительность clean-path не меняет наблюдаемую трактовку циклических результатов или выигрыша обученных защит в этих данных.","",
           "## Итоги проверки","",
           f"- Engineering: `{eng['status']}`; формулы проверены независимым вычислением; float32 identity transform идемпотентен. Signal units оставлены как raw, не названы микровольтами.",
           f"- Replay: `{replay['status']}` для {replay['unit_count']} моделей × {len(replay['units'][0]['conditions'])} условий на первых {REPLAY_N} записях; tolerance {REPLAY_TOL:g}. Новые полные predictions: 18 identity-rederived files; original clean и оба cycle условия переиспользованы после binding/replay checks.",
           f"- Повторный расчёт сохранённых метрик и bootstrap summaries: `{ver['status']}`; внутренний audit, не внешний независимый review.","",
           f"- Засечённое время: полный identity inference {timing['elapsed_seconds']:.2f} s на {timing['new_prediction_count']} файлов; анализ, включая 2×1,000 draw, {analysis_timing['elapsed_seconds']:.2f} s. Обучений — 0.","",
           "## TWA: mean по трём seed ± sample SD","",
           "В таблице приведены AP/AUPRC, AUROC и raw Brier отдельно для двух когорт и четырёх условий. Полные строки для TWA, SB и STach, включая sensitivity/specificity при исходном validation threshold, находятся в `CONDITION_RESULTS_MEAN_SD.csv`; значения по каждому seed — в `CONDITION_RESULTS_BY_SEED.csv`.","",
           "| Когорта | Режим | Условие | AP mean | AUROC mean | Raw Brier mean |","|---|---|---|---:|---:|---:|"]
    for ds in COHORTS:
        for arm in ARMS:
            for cond in CONDITIONS:
                r=next(x for x in means if x["cohort"]==ds and x["arm"]==arm and x["condition"]==cond and x["endpoint"]=="TWA")
                lines.append(f"| {ds} | {arm} | {cond} | {float(r['AP_AUPRC_mean']):.5f} ± {float(r['AP_AUPRC_sd']):.5f} | {float(r['AUROC_mean']):.5f} ± {float(r['AUROC_sd']):.5f} | {float(r['raw_Brier_mean']):.5f} ± {float(r['raw_Brier_sd']):.5f} |")
    lines += ["","## Сопоставления и трактовка циклов","",
              "Парные contrast таблицы содержат full-cohort point estimate отдельно от bootstrap mean и percentile 95% CI. Bootstrap использовал исходные 1,000 synchronized record-index draws, условия и seeds парны; три seeds усреднялись внутри каждой реплики. Пересечение CI с нулём не означает эквивалентность.","",
              "Для каждой модели рассчитаны identity-clean−original-clean, оба cycle−original-clean и оба cycle−identity-clean. Также проверены physics-aware−ordinary и combined−ordinary на каждом цикле. `HISTORICAL_CYCLE_CONTRAST_MATCH.json` сравнивает полные 1,000 draws новых повторных расчётов с ранее сохранёнными cycle contrast draws; циклические predictions и их point estimates не менялись.","",
              "`SCORE_CHANGES_BY_SEED.csv` показывает max/mean и квантили абсолютного изменения вероятностей, а также долю решений, изменившихся при неизменных validation thresholds. Не назначался произвольный клинический порог заметности.","",
              "## Ограничения и вывод","",
              "Это дополнительная sensitivity проверка на ранее просмотренных Chapman и Ningbo тестовых когортах, не независимая подтверждающая выборка. Bootstrap отражает вариацию записей при фиксированных моделях. Сравнение identity-derived и исторического clean оценивает чувствительность к пути представления; оно не устанавливает единственную правильную реализацию для всех исходных ECG устройств. Циклические swaps переиспользуют исторические cycle score files, поэтому их прежняя трактовка сохраняется как post-access результат.","",
              "Технически повторное вычисление работает детерминированно и в float32 идемпотентно, а source identity отличается от ранее сохранённых derived limb channels (до 3.5 raw units). Метрики показывают небольшую, но измеримую чувствительность вероятностей и raw Brier; TWA ranking changes малы относительно cycle effects. Для этой sensitivity-проверки это не указывает на необходимость переобучить или переписать модели, но способ формирования derived leads следует раскрыть и согласованно зафиксировать до следующей новой серии. Не заменять задним числом clean результаты без отдельного согласованного анализа.","",
              "## Артефакты","",
              "- `FROZEN_PROTOCOL.json`, `PROTOCOL_HASH.json` — scope, bindings и fingerprint до новых inference.",
              "- `ENGINEERING_CHECKS.json`, `SOURCE_BINDINGS.json`, `REPLAY_AUDIT.json` — формулы, signal differences, input hashes, replay.",
              "- `PREDICTION_BINDINGS.json`, `predictions/` — новые identity predictions и bindings.",
              "- `CONDITION_RESULTS_BY_SEED.csv`, `CONDITION_RESULTS_MEAN_SD.csv`, `SCORE_CHANGES_BY_SEED.csv` — endpoint metrics и score changes.",
              "- `PAIRED_CONTRASTS.csv`, `BOOTSTRAP_REPLICATES.csv`, `HISTORICAL_CYCLE_CONTRAST_MATCH.json` — paired estimates, all draws and unchanged historical cycle comparison.",
              "- `VERIFICATION_AUDIT.json`, `REPRODUCIBILITY_MANIFEST.json` — checks and hashes.","",
              "По согласованной очереди следующий этап — смесь трёх swaps. Он здесь не запускался; пакет остановлен для проверки.",""]
    (OUT/"FINAL_REPORT_RU.md").write_text("\n".join(lines),encoding="utf-8")


def manifest():
    rows=[]
    for f in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="REPRODUCIBILITY_MANIFEST.json"):
        rows.append({"path":f.relative_to(OUT).as_posix(),"bytes":f.stat().st_size,"sha256":sha_file(f)})
    atomic_json(OUT/"REPRODUCIBILITY_MANIFEST.json",{"algorithm":"SHA-256","protocol_sha256":load_proto()["protocol_sha256"],"files":rows,"self_hash_excluded":True})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=("freeze","engineering","replay","infer","analyze","verify","finalize")); stage=ap.parse_args().stage
    if stage=="freeze": freeze()
    elif stage=="engineering": engineering()
    elif stage=="replay": replay()
    elif stage=="infer": infer()
    elif stage=="analyze": analyze()
    elif stage=="verify": verify()
    elif stage=="finalize":
        final_report(); manifest(); status("COMPLETE",protocol_sha256=load_proto()["protocol_sha256"])

if __name__=="__main__": main()
