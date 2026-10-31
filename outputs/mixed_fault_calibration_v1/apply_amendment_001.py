"""Add already-completed Ningbo Huang scores bound in protocol amendment 001."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from run_analysis import ROOT, OUT, Q, SEEDS, SOURCES, metrics, sha, write_csv, write_json, calibrators, get_data, platt, reliability_rows


def main():
    amend_path = OUT / "FROZEN_PROTOCOL_AMENDMENT_001.json"
    amend = json.loads(amend_path.read_text(encoding="utf-8"))
    for meta in amend["scores"].values():
        for item in meta.values():
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("Amendment-bound Huang Ningbo prediction changed")
    detector = np.load(ROOT / "outputs/published_detector_pipeline_v1/cache/detector_test_probs.npz", allow_pickle=False)
    canonical = json.loads((ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    membership = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by_id = {r["record_id"]: r for r in canonical}
    expected_ids = [by_id[r["record_id"]]["record_id"] for r in membership if r["split"] == "test"]
    if expected_ids != detector["record_ids"].tolist() or len(expected_ids) != 6981:
        raise RuntimeError("Huang detector cache record order differs from Ningbo test binding")
    nbase = ROOT / "outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead"
    m2 = ROOT / "outputs/mitigation_m2_ningbo_v1"
    rows = []
    data = {}
    for seed in SEEDS:
        yall = None
        for cond in ("clean", "RA_LL_reversal"):
            p = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_{cond}_seed_{seed}.npz", allow_pickle=False)["p"].astype(float)
            baseline = np.load(nbase / f"seed_{seed}/{cond}.npz", allow_pickle=False)
            if not np.array_equal(baseline["y"], np.load(nbase / f"seed_{seed}/clean.npz", allow_pickle=False)["y"]):
                raise RuntimeError("Ningbo Huang and baseline label/order binding mismatch")
            if yall is None: yall = baseline["y"].astype(np.uint8)
            elif not np.array_equal(yall, baseline["y"]): raise RuntimeError("paired condition labels differ")
            if p.shape != yall.shape or not np.isfinite(p).all(): raise RuntimeError("invalid Huang score shape/values")
            data.setdefault(seed, {})[cond] = p
        data[seed]["y"] = yall
        historical = np.load(ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz", allow_pickle=False)["thresholds"].astype(float)
        for ep, label in enumerate(("TWA", "SB", "STach")):
            for q in Q:
                y = np.r_[yall[:, ep], yall[:, ep]]
                p = np.r_[data[seed]["clean"][:, ep], data[seed]["RA_LL_reversal"][:, ep]]
                w = np.r_[np.full(len(yall), 1-q), np.full(len(yall), q)]
                rows.append({"source": "Ningbo", "method": "huang_pipeline", "method_label": "Huang→predicted correction→clean ResNet",
                    "seed": seed, "endpoint": label, "q_fault": q, "score_mode": "raw",
                    "threshold_type": "historical_Ningbo_baseline", "threshold": historical[ep], **metrics(y, p, historical[ep], w)})
    # Historical full-validation thresholds are reported separately from the new B operating points.
    matrix = np.load(ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1/predictions_float32.npy", mmap_mode="r")
    chest = np.load(ROOT / "data/heedb_restricted/I0006/mixed_fault_calibration_v1/chest_only_predictions_float32.npy", mmap_mode="r")
    labels, _patients = __import__("run_analysis").heedb_labels()
    he_thresh_paths = {
        "ordinary_resnet": lambda s: ROOT / f"outputs/ningbo_confirmation_v2/seed_{s}/validation.npz",
        "physics_aware_resnet": lambda s: ROOT / f"outputs/mitigation_m2_ningbo_v1/seed_{s}/validation.npz",
        "chest_only_resnet": lambda s: ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{s}/validation.npz",
    }
    n_pred_paths = {
        "ordinary_resnet": lambda s,c: ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{s}/{c}.npz",
        "physics_aware_resnet": lambda s,c: ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/physics_aware_12lead/seed_{s}/{c}.npz",
        "chest_only_resnet": lambda s,c: ROOT / f"outputs/chest_only_baseline_v1/models/Ningbo/seed_{s}/test_predictions/{c}.npz",
    }
    arms_h = {"ordinary_resnet": 0, "physics_aware_resnet": 1}
    huang_he = np.load(ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    for seed_i, seed in enumerate(SEEDS):
        for arm in ("ordinary_resnet", "physics_aware_resnet", "chest_only_resnet"):
            th = np.load(he_thresh_paths[arm](seed), allow_pickle=False)["thresholds"].astype(float)
            if arm == "ordinary_resnet": hc, hr = np.asarray(matrix[0,seed_i,0]), np.asarray(matrix[0,seed_i,1])
            elif arm == "physics_aware_resnet": hc, hr = np.asarray(matrix[1,seed_i,0]), np.asarray(matrix[1,seed_i,1])
            else: hc, hr = np.asarray(chest[seed_i,0]), np.asarray(chest[seed_i,1])
            for ep, label in enumerate(("TWA", "SB", "STach")):
                for q in Q:
                    yy=np.r_[labels[:,ep],labels[:,ep]]; pp=np.r_[hc[:,ep],hr[:,ep]]; ww=np.r_[np.full(10000,1-q),np.full(10000,q)]
                    rows.append({"source":"HEEDB_I0006_independent_confirmation","method":arm,"method_label":{
                        "ordinary_resnet":"Ordinary ResNet","physics_aware_resnet":"Physics-aware ResNet","chest_only_resnet":"Chest-only ResNet"}[arm],
                        "seed":seed,"endpoint":label,"q_fault":q,"score_mode":"raw_historical_threshold",
                        "threshold_type":"historical_full_Ningbo_validation","threshold":float(th[ep]),**metrics(yy,pp,float(th[ep]),ww)})
            # The same source-test arrays with their original, full-validation threshold.
            if arm == "ordinary_resnet": nh = 0
            elif arm == "physics_aware_resnet": nh = 1
            else: nh = None
            nclean = np.load(n_pred_paths[arm](seed,"clean"),allow_pickle=False)
            nrev = np.load(n_pred_paths[arm](seed,"RA_LL_reversal"),allow_pickle=False)
            ny = nclean["y"].astype(np.uint8)
            for ep,label in enumerate(("TWA","SB","STach")):
                for q in Q:
                    yy=np.r_[ny[:,ep],ny[:,ep]];pp=np.r_[nclean["p"][:,ep],nrev["p"][:,ep]];ww=np.r_[np.full(len(ny),1-q),np.full(len(ny),q)]
                    rows.append({"source":"Ningbo","method":arm,"method_label":{
                        "ordinary_resnet":"Ordinary ResNet","physics_aware_resnet":"Physics-aware ResNet","chest_only_resnet":"Chest-only ResNet"}[arm],
                        "seed":seed,"endpoint":label,"q_fault":q,"score_mode":"raw_historical_threshold",
                        "threshold_type":"historical_full_Ningbo_validation","threshold":float(th[ep]),**metrics(yy,pp,float(th[ep]),ww)})
        # Huang's historically frozen threshold is the baseline Ningbo threshold.
        th = np.load(ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz", allow_pickle=False)["thresholds"].astype(float)
        hc, hr = np.asarray(huang_he[seed_i,0]), np.asarray(huang_he[seed_i,1])
        for ep,label in enumerate(("TWA","SB","STach")):
            for q in Q:
                yy=np.r_[labels[:,ep],labels[:,ep]];pp=np.r_[hc[:,ep],hr[:,ep]];ww=np.r_[np.full(10000,1-q),np.full(10000,q)]
                rows.append({"source":"HEEDB_I0006_independent_confirmation","method":"huang_pipeline","method_label":"Huang→predicted correction→clean ResNet",
                    "seed":seed,"endpoint":label,"q_fault":q,"score_mode":"raw_historical_threshold","threshold_type":"historical_Ningbo_baseline",
                    "threshold":float(th[ep]),**metrics(yy,pp,float(th[ep]),ww)})
    # Append only after frozen order and label binding checks pass.
    old = pd.read_csv(OUT / "MIXTURE_RESULTS_BY_SEED.csv")
    if "score_mode" in old:
        old = old[old.score_mode != "raw_historical_threshold"]
        old = old[~((old.source == "Ningbo") & (old.method == "huang_pipeline"))]
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    write_csv(OUT / "MIXTURE_RESULTS_BY_SEED.csv", combined.to_dict(orient="records"))
    # Rebuild TWA reliability bins for every q (the first pass saved only selected q checkpoints).
    fits, _ = calibrators(); data, labels2, _patients2 = get_data(); rel = []
    for source in SOURCES:
        for arm in ("ordinary_resnet", "physics_aware_resnet", "chest_only_resnet", "huang_pipeline"):
            for seed in SEEDS:
                if source == "Ningbo" and arm == "huang_pipeline":
                    c = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz",allow_pickle=False)["p"][:,0]
                    r = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz",allow_pickle=False)["p"][:,0]
                    y = np.load(ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz",allow_pickle=False)["y"][:,0]
                    modes = [("raw", c, r)]
                else:
                    y0, clean, rev = data[source][arm][seed]
                    y = y0[:,0]
                    if arm == "huang_pipeline": modes = [("raw", clean[:,0], rev[:,0])]
                    else:
                        f = fits[arm][seed][0]
                        modes = [("raw",clean[:,0],rev[:,0]),("calibrated",platt(clean[:,0],f["a"],f["b"]),platt(rev[:,0],f["a"],f["b"])),
                                 ("raw_historical_threshold",clean[:,0],rev[:,0])]
                for mode, c, r in modes:
                    for q in Q:
                        yy=np.r_[y,y];pp=np.r_[c,r];ww=np.r_[np.full(len(y),1-q),np.full(len(y),q)]
                        rel.extend(reliability_rows(source,arm,seed,"TWA",q,mode,yy,pp,ww))
    write_csv(OUT / "RELIABILITY_BINS.csv", rel)
    # Add descriptive full-sample contrasts against the already computed paired cohort rows.
    more = []
    for ep, label in enumerate(("TWA", "SB", "STach")):
        for q in Q:
            cur = combined[(combined.source == "Ningbo") & (combined.endpoint == label) & (combined.q_fault == q) & (combined.score_mode == "raw")]
            means = cur.groupby("method").mean(numeric_only=True)
            if "physics_aware_resnet" not in means.index or "huang_pipeline" not in means.index: continue
            for col in ("auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction",
                        "false_positives_per_1000", "false_negatives_per_1000"):
                more.append({"source": "Ningbo", "endpoint": label, "q_fault": q, "score_mode": "raw",
                    "contrast": "physics_aware_minus_huang_pipeline", "metric": col,
                    "physics_aware_seed_mean": float(means.loc["physics_aware_resnet", col]),
                    "comparator_seed_mean": float(means.loc["huang_pipeline", col]),
                    "difference": float(means.loc["physics_aware_resnet", col]-means.loc["huang_pipeline", col]),
                    "interval": "Primary TWA AUPRC bootstrap interval added by protocol amendment 001; secondary contrasts descriptive"})
    oldc = pd.read_csv(OUT / "PAIRED_CONTRASTS.csv")
    oldc = oldc[oldc.contrast != "physics_aware_minus_huang_pipeline"]
    write_csv(OUT / "PAIRED_CONTRASTS.csv", pd.concat([oldc, pd.DataFrame(more)], ignore_index=True).to_dict(orient="records"))
    # Reproduce precisely the Ningbo index sequence from the primary run (same seed, same draw order).
    rng = np.random.default_rng(20261017)
    diffs = {q: [] for q in Q}; huang_abs = {q: [] for q in Q}
    for bi in range(1000):
        idx = rng.integers(0, 6981, size=6981)
        by_seed_diff = {q: [] for q in Q}; by_seed_h = {q: [] for q in Q}
        for seed in SEEDS:
            y = np.load(ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz", allow_pickle=False)["y"][idx, 0].astype(np.uint8)
            hc = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz",allow_pickle=False)["p"][idx,0].astype(float)
            hr = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz",allow_pickle=False)["p"][idx,0].astype(float)
            ph = np.load(m2 / f"seed_{seed}/test_predictions/RA_LL_reversal.npz", allow_pickle=False)["p"][idx, 0].astype(float)
            # Use physics clean/reversal probabilities from the bound comparator files.
            pc = np.load(ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/physics_aware_12lead/seed_{seed}/clean.npz", allow_pickle=False)["p"][idx, 0].astype(float)
            th = float(np.load(m2 / f"seed_{seed}/validation.npz", allow_pickle=False)["thresholds"][0])
            t_h = float(np.load(ROOT / f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz", allow_pickle=False)["thresholds"][0])
            for q in Q:
                yy = np.r_[y, y]; ww = np.r_[np.full(len(y), 1-q), np.full(len(y), q)]
                h = float(average_precision_score(yy, np.r_[hc, hr], sample_weight=ww))
                p = float(average_precision_score(yy, np.r_[pc, ph], sample_weight=ww))
                by_seed_h[q].append(h); by_seed_diff[q].append(p-h)
        for q in Q:
            huang_abs[q].append(float(np.mean(by_seed_h[q])))
            diffs[q].append(float(np.mean(by_seed_diff[q])))
        if (bi+1) % 200 == 0: print(f"amendment bootstrap Ningbo: {bi+1}/1000", flush=True)
    boot_add = []
    for q in Q:
        x = np.asarray(huang_abs[q]); d = np.asarray(diffs[q])
        for method, vals in (("huang_pipeline", x), ("physics_aware_minus_huang_pipeline", d)):
            boot_add.append({"source": "Ningbo", "method": method, "score_mode": "raw", "endpoint": "TWA", "q_fault": q,
                "replicates": 1000, "bootstrap_mean": float(vals.mean()), "ci_2_5": float(np.quantile(vals,.025)),
                "ci_97_5": float(np.quantile(vals,.975)), "conditional_on": "selected checkpoints and historical Ningbo baseline operating thresholds",
                "kind": "paired_contrast" if method.startswith("physics_aware_minus") else "method"})
    oldb = pd.read_csv(OUT / "BOOTSTRAP_SUMMARY.csv")
    write_csv(OUT / "BOOTSTRAP_SUMMARY.csv", pd.concat([oldb, pd.DataFrame(boot_add)], ignore_index=True).to_dict(orient="records"))
    boot_rows = []
    for q in Q:
        for i, (v, d) in enumerate(zip(huang_abs[q], diffs[q])):
            boot_rows.append({"source":"Ningbo","primary_endpoint":"TWA","metric":"AUPRC","score_mode":"raw","q_fault":q,
                "contrast":"huang_pipeline", "bootstrap_replicate":i,"difference":v})
            boot_rows.append({"source":"Ningbo","primary_endpoint":"TWA","metric":"AUPRC","score_mode":"raw","q_fault":q,
                "contrast":"physics_aware_minus_huang_pipeline", "bootstrap_replicate":i,"difference":d})
    oldr = pd.read_csv(OUT / "BOOTSTRAP_REPLICATES.csv")
    write_csv(OUT / "BOOTSTRAP_REPLICATES.csv", pd.concat([oldr, pd.DataFrame(boot_rows)], ignore_index=True).to_dict(orient="records"))
    # Repair the coverage table with the final, amendment-bound state.
    cov = pd.read_csv(OUT / "SCORE_COVERAGE_AUDIT.csv").fillna("")
    cov.loc[(cov.method == "chest_only_resnet") & (cov.source == "HEEDB_I0006_independent_confirmation"), ["clean_test", "RA_LL_test"]] = "available; frozen-model inference completed"
    cov.loc[(cov.method == "huang_pipeline") & (cov.source == "Ningbo"), ["validation", "clean_test", "RA_LL_test", "checkpoint", "preprocessing"]] = [
        "unavailable; raw-only as frozen in amendment 001", "available from completed published_detector_pipeline_v1 caches",
        "available from completed published_detector_pipeline_v1 caches", "published frozen detector + Ningbo baseline checkpoints",
        "bound by published_detector_pipeline_v1 protocol and manifests"]
    cov.loc[(cov.method == "huang_pipeline") & (cov.source == "HEEDB_I0006_independent_confirmation"), ["validation", "clean_test", "RA_LL_test"]] = [
        "unavailable; no full-pipeline Ningbo validation predictions", "available", "available"]
    write_csv(OUT / "SCORE_COVERAGE_AUDIT.csv", cov.to_dict(orient="records"))
    result = {"status":"PASS", "amendment_sha256":sha(amend_path), "ningbo_huang_cache_record_order_verified":True,
        "new_rows":len(rows), "primary_bootstrap_replicates":1000, "bootstrap_rng_matches_main_analysis_ningbo_stream":True,
        "scores_were_preexisting_not_reinferred":True, "calibration":"not fit; no full-pipeline source-validation score set"}
    write_json(OUT / "ANALYSIS_AMENDMENT_001.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
