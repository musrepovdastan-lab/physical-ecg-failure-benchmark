from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys_paths = [str(ROOT), str(ROOT / "outputs/mixed_fault_calibration_v1")]
import sys
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from outputs.mixed_fault_calibration_v1 import run_analysis as base
from src import published_detector_multifault_v2 as mf

SEEDS = (104729, 130363, 155921)
METHODS = ("ordinary_resnet", "physics_aware_p050", "combined", "Huang_full_pipeline")
CONDITIONS = ("clean", "RA_LL_reversal", "RA_LA_reversal", "LA_LL_reversal")
Q = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
ENDPOINTS = ("TWA", "SB", "STach")
SCENARIOS = ("equal_three_swap", "RA_LL_only")
CONTRASTS = (("physics_aware_p050", "ordinary_resnet"), ("combined", "ordinary_resnet"),
             ("Huang_full_pipeline", "ordinary_resnet"), ("physics_aware_p050", "Huang_full_pipeline"),
             ("combined", "Huang_full_pipeline"))
BOOT_METRICS = ("raw_TWA_AP", "raw_TWA_Brier", "calibrated_TWA_Brier", "sensitivity", "FP_per_1000")
BOOT_PATHS={"Ningbo":ROOT/"outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
            "HEEDB":ROOT/"outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_fits() -> dict:
    mixed = json.loads((ROOT / "outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))["calibration_fits"]
    n4 = json.loads((ROOT / "outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))["fits"]
    huang = json.loads((ROOT / "outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))["fits_by_seed"]
    fits = {m: {s: mixed[m][str(s)] for s in SEEDS} for m in ("ordinary_resnet", "physics_aware_resnet")}
    fits["ordinary_resnet"] = fits["ordinary_resnet"]
    fits["physics_aware_p050"] = fits.pop("physics_aware_resnet")
    fits["combined"] = {s: n4[f"combined/{s}"] for s in SEEDS}
    fits["Huang_full_pipeline"] = {int(s): vals for s, vals in huang.items()}
    return fits


def threshold(fit: dict, mode: str) -> float:
    if mode == "raw":
        return float(fit.get("threshold_raw_validation_B_90pct", fit.get("threshold_raw")))
    return float(fit.get("threshold_calibrated_validation_B_90pct", fit.get("threshold_calibrated")))


def probability(fit: dict, p: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.asarray(p, dtype=np.float64)
    return base.platt(np.asarray(p, dtype=np.float64), float(fit["a"]), float(fit["b"]))


def load_predictions() -> tuple[dict, dict, dict, dict]:
    """Return source-specific y, ids and [method][seed][condition] probabilities."""
    data, ids, labels = {}, {}, {}
    ningbo_rows = mf.records()
    ningbo_y = np.asarray([r["labels"] for r in ningbo_rows], dtype=np.uint8)
    ningbo_ids = np.asarray([r["record_id"] for r in ningbo_rows], dtype=str)
    data["Ningbo"] = {m: {} for m in METHODS}
    ids["Ningbo"], labels["Ningbo"] = ningbo_ids, ningbo_y
    for method in METHODS:
        for seed in SEEDS:
            data["Ningbo"][method][seed] = {}
            for cond in CONDITIONS:
                if method == "ordinary_resnet":
                    path = ROOT / f"outputs/published_detector_multifault_v2/cache/diagnostic_clean_resnet_no_correction_{cond}_seed_{seed}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        if not np.array_equal(z["record_ids"].astype(str), ningbo_ids):
                            raise RuntimeError(f"Ningbo ordinary record order mismatch: {path}")
                        p = z["p"]
                elif method == "physics_aware_p050":
                    path = ROOT / f"outputs/published_detector_multifault_v2/cache/diagnostic_physics_aware_no_correction_{cond}_seed_{seed}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        if not np.array_equal(z["record_ids"].astype(str), ningbo_ids):
                            raise RuntimeError(f"Ningbo p050 record order mismatch: {path}")
                        p = z["p"]
                elif method == "Huang_full_pipeline":
                    path = ROOT / f"outputs/published_detector_multifault_v2/cache/diagnostic_huang_argmax_correct_clean_resnet_{cond}_seed_{seed}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        if not np.array_equal(z["record_ids"].astype(str), ningbo_ids):
                            raise RuntimeError(f"Ningbo Huang record order mismatch: {path}")
                        p = z["p"]
                else:
                    path = ROOT / f"outputs/ningbo_four_arm_replication_v1/combined/seed_{seed}/test_predictions/{cond}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        p = z["p"]
                        if not np.array_equal(z["y"], ningbo_y):
                            raise RuntimeError(f"Ningbo combined labels/order mismatch: {path}")
                if p.shape != (6981, 3) or not np.isfinite(p).all():
                    raise RuntimeError(f"invalid Ningbo scores: {method}/{seed}/{cond}")
                data["Ningbo"][method][seed][cond] = np.asarray(p, dtype=np.float64)

    from outputs.mixed_fault_calibration_v1 import run_analysis as old
    heedb_y, heedb_patients = old.heedb_labels()
    selection = pd.read_csv(ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv", dtype=str, keep_default_na=False)
    heedb_ids = selection["FileName"].astype(str).to_numpy()
    if len(heedb_y) != 10000 or len(heedb_patients) != 10000 or len(heedb_ids) != 10000 or len(set(heedb_patients.astype(str))) != 10000:
        raise RuntimeError("HEEDB cohort units or order mismatch")
    data["HEEDB"] = {m: {} for m in METHODS}
    ids["HEEDB"], labels["HEEDB"] = heedb_ids, np.asarray(heedb_y, dtype=np.uint8)
    for method in METHODS:
        for seed in SEEDS:
            data["HEEDB"][method][seed] = {}
            for cond in CONDITIONS:
                path = OUT / f"predictions/HEEDB/{method}/seed_{seed}/{cond}.npz"
                with np.load(path, allow_pickle=False) as z:
                    bad = (not np.array_equal(z["labels"], heedb_y) or
                           not np.array_equal(z["patient_ids"].astype(str), heedb_patients.astype(str)))
                    if bad:
                        raise RuntimeError(f"HEEDB prediction binding mismatch: {path}")
                    p = z["p"]
                if p.shape != (10000, 3) or not np.isfinite(p).all():
                    raise RuntimeError(f"invalid HEEDB scores: {method}/{seed}/{cond}")
                data["HEEDB"][method][seed][cond] = np.asarray(p, dtype=np.float64)
    return data, ids, labels, {"HEEDB_patients": heedb_patients.astype(str)}


def scenario_weights(scenario: str, q: float) -> np.ndarray:
    if scenario == "equal_three_swap":
        return np.asarray((1-q, q/3, q/3, q/3), dtype=np.float64)
    return np.asarray((1-q, q, 0., 0.), dtype=np.float64)


def weighted_metrics(y: np.ndarray, p: np.ndarray, th: float, w: np.ndarray) -> dict:
    keep = w > 0
    y = y[keep].astype(np.uint8); p = p[keep].astype(np.float64); w = w[keep].astype(np.float64)
    pos = y == 1; neg = ~pos; pred = p >= th
    tp = float(w[pos & pred].sum()); fn = float(w[pos & ~pred].sum())
    fp = float(w[neg & pred].sum()); tn = float(w[neg & ~pred].sum())
    total = float(w.sum())
    ece = 0.
    bins = np.minimum((p * 15).astype(np.int64), 14)
    for b in range(15):
        ix = bins == b
        wb = float(w[ix].sum())
        if wb:
            ece += wb / total * abs(float(np.average(y[ix], weights=w[ix])) - float(np.average(p[ix], weights=w[ix])))
    return {"AP": float(average_precision_score(y, p, sample_weight=w)),
            "AUROC": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, p, sample_weight=w)),
            "Brier": float(np.average((p-y)**2, weights=w)), "ECE15": float(ece),
            "sensitivity": tp/(tp+fn) if tp+fn else float("nan"),
            "specificity": tn/(tn+fp) if tn+fp else float("nan"),
            "predicted_positive_fraction": (tp+fp)/total,
            "FP_per_1000": 1000*fp/total, "FN_per_1000": 1000*fn/total,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def create_tables(data: dict, labels: dict, fits: dict) -> tuple[list[dict], list[dict], list[dict]]:
    by_condition, mixtures, calibration_audit = [], [], []
    for source in ("Ningbo", "HEEDB"):
        y_all = labels[source]
        for method in METHODS:
            for seed in SEEDS:
                for endpoint_i, endpoint in enumerate(ENDPOINTS):
                    fitkey = {"ordinary_resnet":"ordinary_resnet", "physics_aware_p050":"physics_aware_p050", "combined":"combined", "Huang_full_pipeline":"Huang_full_pipeline"}[method]
                    fit = fits[fitkey][seed][endpoint_i]
                    tr, tc = threshold(fit,"raw"), threshold(fit,"calibrated")
                    for mode in ("raw","calibrated"):
                        cond_scores = {c: probability(fit,data[source][method][seed][c][:,endpoint_i],mode) for c in CONDITIONS}
                        th = tr if mode == "raw" else tc
                        # Calibration monotonicity/tie and decision audit per condition.
                        if mode == "raw":
                            for c in CONDITIONS:
                                raw = cond_scores[c]
                                cal = probability(fit,data[source][method][seed][c][:,endpoint_i],"calibrated")
                                ord_ix = np.argsort(raw, kind="stable")
                                calibration_audit.append({"source":source,"method":method,"seed":seed,"endpoint":endpoint,"condition":c,
                                    "raw_unique_scores":int(np.unique(raw).size),"calibrated_unique_scores":int(np.unique(cal).size),
                                    "added_ties":int(np.unique(raw).size-np.unique(cal).size),
                                    "max_monotonicity_violation":float(max(0.,-np.diff(cal[ord_ix]).min())) if len(raw)>1 else 0.,
                                    "raw_calibrated_decision_mismatches":int(np.sum((raw>=tr)!=(cal>=tc)))})
                        for ci,c in enumerate(CONDITIONS):
                            yy=y_all[:,endpoint_i]; pp=cond_scores[c]; ww=np.ones(len(yy),dtype=np.float64)
                            met=weighted_metrics(yy,pp,th,ww)
                            by_condition.append({"source":source,"method":method,"seed":seed,"endpoint":endpoint,"condition":c,
                                "score_mode":mode,"threshold_type":"validation_B_90pct","threshold":th,**met})
                        for scenario in SCENARIOS:
                            for q in Q:
                                coeff=scenario_weights(scenario,q)
                                yy=np.concatenate([y_all[:,endpoint_i]]*4)
                                pp=np.concatenate([cond_scores[c] for c in CONDITIONS])
                                ww=np.concatenate([np.full(len(y_all),v,dtype=np.float64) for v in coeff])
                                met=weighted_metrics(yy,pp,th,ww)
                                mixtures.append({"source":source,"method":method,"seed":seed,"endpoint":endpoint,
                                    "scenario":scenario,"q_fault":q,"score_mode":mode,
                                    "threshold_type":"validation_B_90pct","threshold":th,**met})
    return by_condition, mixtures, calibration_audit


def make_group_cache(y: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> dict:
    n=len(y);flat=scores.reshape(-1)
    order=np.argsort(-flat,kind="mergesort");ordered=flat[order]
    starts=np.r_[True,ordered[1:]!=ordered[:-1]]
    group_sorted=np.cumsum(starts,dtype=np.int32)-1
    inverse=np.empty(len(order),dtype=np.int32);inverse[order]=group_sorted
    groups=int(group_sorted[-1])+1
    ids=inverse.reshape(4,n)
    return {"ids":ids,"groups":groups,"scores":scores,"decisions":decisions,"y":y.astype(np.float64)}


def grouped_ap(cache: dict, multiplicity: np.ndarray, coeff: np.ndarray) -> float:
    y=cache["y"];g=cache["groups"]
    pos=np.zeros(g,dtype=np.float64);neg=np.zeros(g,dtype=np.float64)
    for ci in range(4):
        w=multiplicity*coeff[ci]
        pos += np.bincount(cache["ids"][ci],weights=w*y,minlength=g)
        neg += np.bincount(cache["ids"][ci],weights=w*(1-y),minlength=g)
    P=float(pos.sum())
    if P==0:return float("nan")
    ctp=np.cumsum(pos);cfp=np.cumsum(neg);den=ctp+cfp
    prec=np.divide(ctp,den,out=np.zeros_like(ctp),where=den>0)
    return float(np.sum(pos/P*prec))


def bootstrap(data: dict, labels: dict, fits: dict, mixture_rows: list[dict]) -> tuple[list[dict], dict]:
    boot_paths=BOOT_PATHS
    indices={s:np.load(p,allow_pickle=False) for s,p in boot_paths.items()}
    for source,ix in indices.items():
        if ix.shape!=(1000,len(labels[source])):raise RuntimeError(f"bootstrap shape mismatch {source}: {ix.shape}")
    # Prepare per-method/seed TWA arrays and caches. Raw-score AP is evaluated by grouped score ranks.
    raw_cache={}; raw_scores={}; cal_scores={}; decisions_raw={}; decisions_cal={}; ys={}
    for source in ("Ningbo","HEEDB"):
        y=labels[source][:,0].astype(np.uint8);ys[source]=y
        for method in METHODS:
            for seed in SEEDS:
                fitkey=method
                fit=fits[fitkey][seed][0]
                raw=np.stack([data[source][method][seed][c][:,0] for c in CONDITIONS])
                cal=np.stack([probability(fit,data[source][method][seed][c][:,0],"calibrated") for c in CONDITIONS])
                tr,tc=threshold(fit,"raw"),threshold(fit,"calibrated")
                raw_cache[source,method,seed]=make_group_cache(y,raw,raw>=tr)
                raw_scores[source,method,seed]=raw
                cal_scores[source,method,seed]=cal
                decisions_raw[source,method,seed]=raw>=tr
                decisions_cal[source,method,seed]=cal>=tc
    paired_draws={}; rows=[]
    for source in ("Ningbo","HEEDB"):
        ix=indices[source]
        multiplicities=np.stack([np.bincount(row,minlength=len(ys[source])) for row in ix]).astype(np.int32)
        for scenario in SCENARIOS:
            for q in Q:
                coeff=scenario_weights(scenario,q)
                # result[method][metric][draw]
                values={m:{metric:np.empty(1000,dtype=np.float64) for metric in BOOT_METRICS} for m in METHODS}
                for bi,mult in enumerate(multiplicities):
                    if bi%200==0:print(f"BOOT {source} {scenario} q={q:g} {bi}/1000",flush=True)
                    for method in METHODS:
                        perseed={metric:[] for metric in BOOT_METRICS}
                        for seed in SEEDS:
                            cache=raw_cache[source,method,seed];y=cache["y"]
                            perseed["raw_TWA_AP"].append(grouped_ap(cache,mult,coeff))
                            for mode,sarr in (("raw",raw_scores),("calibrated",cal_scores)):
                                scores=sarr[source,method,seed]
                                bs=sum(float(np.dot(mult, (scores[ci]-y)**2))*coeff[ci] for ci in range(4))/float(mult.sum())
                                if mode=="raw":perseed["raw_TWA_Brier"].append(bs)
                                else:perseed["calibrated_TWA_Brier"].append(bs)
                            # Use calibrated validation-B decision after verifying raw/cal equivalence separately.
                            dec=decisions_cal[source,method,seed]
                            pos=y==1;neg=~pos
                            tp=sum(float(np.dot(mult*pos,dec[ci]))*coeff[ci] for ci in range(4))
                            fn=sum(float(np.dot(mult*pos,~dec[ci]))*coeff[ci] for ci in range(4))
                            fp=sum(float(np.dot(mult*neg,dec[ci]))*coeff[ci] for ci in range(4))
                            perseed["sensitivity"].append(tp/(tp+fn) if tp+fn else np.nan)
                            perseed["FP_per_1000"].append(1000*fp/float(mult.sum()))
                        for metric in BOOT_METRICS:values[method][metric][bi]=float(np.mean(perseed[metric]))
                for left,right in CONTRASTS:
                    for metric in BOOT_METRICS:
                        draws=values[left][metric]-values[right][metric]
                        key=f"{source}|{scenario}|q={q:g}|{left}-{right}|{metric}"
                        paired_draws[key]=draws
                        full_left=np.mean([r[metric.replace("raw_TWA_AP","AP").replace("raw_TWA_Brier","Brier").replace("calibrated_TWA_Brier","Brier").replace("FP_per_1000","FP_per_1000")]
                            for r in mixture_rows if r["source"]==source and r["method"]==left and r["scenario"]==scenario and r["q_fault"]==q and r["endpoint"]=="TWA" and r["score_mode"]==("calibrated" if metric in ("calibrated_TWA_Brier","sensitivity","FP_per_1000") else "raw")])
                        full_right=np.mean([r[metric.replace("raw_TWA_AP","AP").replace("raw_TWA_Brier","Brier").replace("calibrated_TWA_Brier","Brier").replace("FP_per_1000","FP_per_1000")]
                            for r in mixture_rows if r["source"]==source and r["method"]==right and r["scenario"]==scenario and r["q_fault"]==q and r["endpoint"]=="TWA" and r["score_mode"]==("calibrated" if metric in ("calibrated_TWA_Brier","sensitivity","FP_per_1000") else "raw")])
                        rows.append({"source":source,"scenario":scenario,"q_fault":q,"contrast":f"{left}-{right}","metric":metric,
                            "full_sample_difference":float(full_left-full_right),"bootstrap_mean_difference":float(draws.mean()),
                            "ci_2_5":float(np.quantile(draws,.025)),"ci_97_5":float(np.quantile(draws,.975)),"replicates":1000,
                            "conditional_on":"fixed selected checkpoints, fitted source calibrators and source validation-B thresholds"})
    np.savez_compressed(OUT/"PAIRED_BOOTSTRAP_DRAWS.npz",draw_names=np.asarray(list(paired_draws)),draws=np.stack(list(paired_draws.values())),
                        Ningbo_indices_sha256=np.asarray(sha(boot_paths["Ningbo"])),HEEDB_indices_sha256=np.asarray(sha(boot_paths["HEEDB"])))
    return rows, paired_draws


def main() -> None:
    for name in ("RECOVERY_PROTOCOL.json","INPUT_HANDOFF_ENGINEERING_AUDIT.json","DETECTOR_BLOCK_REPLAY_AUDIT.json","REPLAY_AUDIT.json","RECOVERY_INFERENCE_COMPLETE.json"):
        p=OUT/name
        if not p.exists():raise RuntimeError(f"pre-analysis gate missing {name}")
    for name in ("INPUT_HANDOFF_ENGINEERING_AUDIT.json","DETECTOR_BLOCK_REPLAY_AUDIT.json","REPLAY_AUDIT.json","RECOVERY_INFERENCE_COMPLETE.json"):
        d=json.loads((OUT/name).read_text(encoding="utf-8"))
        if d.get("status")!="PASS":raise RuntimeError(f"pre-analysis gate failed {name}")
    fits=load_fits()
    data,ids,labels,aux=load_predictions()
    # all source records/patients, labels and conditions are bound in a compact audit.
    order_rows=[]
    for source in ("Ningbo","HEEDB"):
        for i,(rid,y) in enumerate(zip(ids[source],labels[source])):
            order_rows.append({"source":source,"row":i,"record_or_file_id":rid,**{ENDPOINTS[j]:int(y[j]) for j in range(3)}})
    write_csv(OUT/"COHORT_ORDER_AND_LABELS.csv",order_rows)
    by_condition,mixtures,cal_audit=create_tables(data,labels,fits)
    write_csv(OUT/"CONDITION_METRICS_BY_SEED.csv",by_condition)
    write_csv(OUT/"MIXTURE_METRICS_BY_SEED.csv",mixtures)
    write_csv(OUT/"CALIBRATION_RANKING_AND_DECISION_AUDIT.csv",cal_audit)
    df=pd.DataFrame(mixtures)
    metcols=["AP","AUROC","Brier","ECE15","sensitivity","specificity","predicted_positive_fraction","FP_per_1000","FN_per_1000"]
    means=df.groupby(["source","method","endpoint","scenario","q_fault","score_mode","threshold_type"],dropna=False)[metcols].agg(["mean","std"]).reset_index()
    means.columns=["_".join(str(x) for x in c if x!="").rstrip("_") if isinstance(c,tuple) else c for c in means.columns]
    means.to_csv(OUT/"MIXTURE_METRICS_MEAN_SD.csv",index=False)
    # Exact anchor comparisons and weighted-view invariants.
    anchor=[]
    for row in mixtures:
        if row["q_fault"]==0:
            ref=next(x for x in by_condition if x["source"]==row["source"] and x["method"]==row["method"] and x["seed"]==row["seed"] and x["endpoint"]==row["endpoint"] and x["condition"]=="clean" and x["score_mode"]==row["score_mode"])
            err=max(abs(float(row[k])-float(ref[k])) for k in metcols)
            anchor.append({"source":row["source"],"method":row["method"],"seed":row["seed"],"endpoint":row["endpoint"],"score_mode":row["score_mode"],"anchor":"q0_equals_clean","max_abs_metric_delta":err,"status":"PASS" if err<=1e-12 else "FAIL"})
    # The two prescribed scenarios must be identical at q=0, and B equals the RA-LL-only mixture.
    # Reproduce each historical clean/RA-LL comparator row from its immutable report table.
    history=[]
    legacy=pd.read_csv(ROOT/"outputs/mixed_fault_calibration_v2/MIXTURE_RESULTS_BY_SEED.csv")
    n4=pd.read_csv(ROOT/"outputs/ningbo_four_arm_replication_v1/MIXTURE_RESULTS_BY_SEED.csv")
    heedb4=pd.read_csv(ROOT/"outputs/heedb_four_arm_extension_v1/MIXTURE_RESULTS_BY_SEED.csv")
    metric_map={"AP":"auprc","AUROC":"auroc","Brier":"brier","ECE15":"ece15","sensitivity":"sensitivity",
                "specificity":"specificity","predicted_positive_fraction":"predicted_positive_fraction",
                "FP_per_1000":"false_positives_per_1000","FN_per_1000":"false_negatives_per_1000"}
    for current in mixtures:
        if current["scenario"]!="RA_LL_only":continue
        for old_method,old_source in (("ordinary_resnet","legacy"),("physics_aware_p050","legacy"),("Huang_full_pipeline","legacy"),("combined","combined")):
            if current["method"]!=old_method:continue
            if old_source=="legacy":
                oldname={"ordinary_resnet":"ordinary_resnet","physics_aware_p050":"physics_aware_resnet","Huang_full_pipeline":"huang_pipeline"}[old_method]
                srcname="Ningbo" if current["source"]=="Ningbo" else "HEEDB_I0006_independent_confirmation"
                oldmode=current["score_mode"]
                if old_method=="Huang_full_pipeline":
                    oldmode="raw_validation_B_90pct" if current["score_mode"]=="raw" else "calibrated_validation_B_90pct"
                subset=legacy[(legacy.source==srcname)&(legacy.method==oldname)&(legacy.seed==current["seed"])&
                              (legacy.endpoint==current["endpoint"])&(legacy.q_fault==current["q_fault"])&
                              (legacy.score_mode==oldmode)]
            elif current["source"]=="Ningbo":
                subset=n4[(n4.arm=="combined")&(n4.seed==current["seed"])&(n4.endpoint==current["endpoint"])&
                          (n4.q==current["q_fault"])&(n4.score_mode==current["score_mode"])]
            else:
                subset=heedb4[(heedb4.method=="combined")&(heedb4.seed==current["seed"])&(heedb4.endpoint==current["endpoint"])&
                              (heedb4.q_fault==current["q_fault"])&(heedb4.score_mode==current["score_mode"])]
            if len(subset)!=1:raise RuntimeError(f"historical RA-LL anchor missing: {current}")
            oldrow=subset.iloc[0]
            diffs={}
            prevalence=float(labels[current["source"]][:,ENDPOINTS.index(current["endpoint"])].mean())
            for k,v in metric_map.items():
                if v in oldrow.index:
                    historical=float(oldrow[v])
                elif v=="false_positives_per_1000":
                    historical=1000*(1-float(oldrow["specificity"]))*(1-prevalence)
                elif v=="false_negatives_per_1000":
                    historical=1000*(1-float(oldrow["sensitivity"]))*prevalence
                else:
                    raise RuntimeError(f"historical anchor field unavailable: {v}")
                diffs[k]=abs(float(current[k])-historical)
            err=max(diffs.values())
            history.append({"source":current["source"],"method":old_method,"seed":current["seed"],"endpoint":current["endpoint"],
                            "scenario":"RA_LL_only","q_fault":current["q_fault"],"score_mode":current["score_mode"],
                            "max_abs_metric_delta":err,"max_direct_metric_delta":max(diffs[k] for k in diffs if k not in ("FP_per_1000","FN_per_1000")),
                            "max_derived_count_delta":max(diffs[k] for k in ("FP_per_1000","FN_per_1000")),
                            "deltas":json.dumps(diffs,sort_keys=True),"status":"PASS" if max(diffs[k] for k in diffs if k not in ("FP_per_1000","FN_per_1000"))<=1e-6 and max(diffs[k] for k in ("FP_per_1000","FN_per_1000"))<=0.001 else "FAIL"})
    write_csv(OUT/"RA_LL_ANCHOR_AUDIT.csv",history)
    if any(x["status"]!="PASS" for x in history):raise RuntimeError("historical RA-LL-only anchor reproduction failed")
    write_csv(OUT/"ANCHOR_WEIGHT_AUDIT.csv",anchor)
    if any(a["status"]!="PASS" for a in anchor):raise RuntimeError("anchor weight audit failed")
    # For the frozen all-swap scenario, keep the per-fault confusion and detector class confusion visible.
    detector_confusion=[]
    z=np.load(OUT/"huang_detector_probabilities.npz",allow_pickle=False)
    det=np.argmax(z["probabilities"],axis=-1)
    truth_codes=(0,2,1,3)  # detector class map is clean, RA_LA, RA_LL, LA_LL
    for ci,cond in enumerate(CONDITIONS):
        for pred_class in range(4):
            detector_confusion.append({"source":"HEEDB","true_condition":cond,"true_detector_class":truth_codes[ci],"predicted_class":int(pred_class),
                                       "count":int(np.sum(det[ci]==pred_class)),"fraction":float(np.mean(det[ci]==pred_class)),
                                       "is_correct_class":bool(pred_class==truth_codes[ci])})
    write_csv(OUT/"HUANG_DETECTOR_CONFUSION_BY_CONDITION.csv",detector_confusion)
    # Per-fault corrections and clean false corrections, descriptive and outcome blind in deployed branch.
    falsecorr=[]
    for ci,cond in enumerate(CONDITIONS):
        pc=det[ci]
        true_code=(0,2,1,3)[ci]
        falsecorr.append({"source":"HEEDB","condition":cond,"true_detector_class":true_code,"n":len(pc),"detector_argmax_correct":int(np.sum(pc==true_code)),
                          "detector_argmax_incorrect":int(np.sum(pc!=true_code)),"correction_applied_fraction":float(np.mean(pc!=0)),
                          "clean_false_correction_count":int(np.sum(pc!=0)) if cond=="clean" else ""})
    write_csv(OUT/"HUANG_FALSE_CORRECTION_AUDIT.csv",falsecorr)
    paired,draws=bootstrap(data,labels,fits,mixtures)
    write_csv(OUT/"PAIRED_CONTRASTS_POINT_AND_CI.csv",paired)
    # Actual draws 0/417/999: grouped AP vs sklearn's sample-weighted repeated-view calculation.
    from sklearn.metrics import average_precision_score as sklearn_ap
    repeat_audit=[]
    for source in ("Ningbo","HEEDB"):
        ix=np.load(BOOT_PATHS[source],allow_pickle=False)
        for draw_i in (0,417,999):
            mult=np.bincount(ix[draw_i],minlength=len(labels[source]))
            selected=np.repeat(np.arange(len(mult)),mult)
            for method in METHODS:
                seed=SEEDS[0];cache=raw_cache[source,method,seed] if False else None
                raw=np.stack([data[source][method][seed][c][:,0] for c in CONDITIONS])
                y=labels[source][:,0].astype(np.uint8)
                fit=fits[method][seed][0]
                for scenario in SCENARIOS:
                    coeff=scenario_weights(scenario,.05)
                    groups=make_group_cache(y,raw,np.zeros_like(raw,dtype=bool))
                    got=grouped_ap(groups,mult,coeff)
                    yy=np.tile(y[selected],4)
                    pp=np.concatenate([raw[ci,selected] for ci in range(4)])
                    ww=np.concatenate([np.full(len(selected),coeff[ci]) for ci in range(4)])
                    ref=float(sklearn_ap(yy,pp,sample_weight=ww))
                    # Weighted equivalent and physically repeated-record expansion are checked separately.
                    yfull=np.tile(y,4);pfull=raw.reshape(-1)
                    sample_w=np.concatenate([mult.astype(float)*coeff[ci] for ci in range(4)])
                    ref_weight=float(sklearn_ap(yfull,pfull,sample_weight=sample_w))
                    repeat_audit.append({"source":source,"method":method,"seed":seed,"draw":draw_i,"scenario":scenario,"q":.05,
                                         "grouped_AP":got,"sklearn_repeated_AP":ref,"sklearn_weighted_AP":ref_weight,
                                         "max_abs_error":max(abs(got-ref),abs(got-ref_weight)),
                                         "status":"PASS" if max(abs(got-ref),abs(got-ref_weight))<1e-11 else "FAIL"})
    write_csv(OUT/"BOOTSTRAP_DRAW_RECALC_AUDIT.csv",repeat_audit)
    if any(x["status"]!="PASS" for x in repeat_audit):raise RuntimeError("weighted AP bootstrap verifier failed")
    audit={"status":"PASS","protocol_file_sha256":sha(OUT/"RECOVERY_PROTOCOL.json"),"Ningbo_records":len(labels["Ningbo"]),"HEEDB_unique_patients":len(labels["HEEDB"]),
           "condition_metric_rows":len(by_condition),"mixture_metric_rows":len(mixtures),"paired_contrast_rows":len(paired),"paired_draw_count":len(draws),
           "bootstrap_indices_sha256":{s:sha(p) for s,p in BOOT_PATHS.items()},"repeated_sample_sklearn_rows":len(repeat_audit),
           "max_repeated_sample_AP_error":max(x["max_abs_error"] for x in repeat_audit),"reserved_test_evaluated":False,"training_or_calibration_fit":False}
    (OUT/"VERIFICATION_AUDIT.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
