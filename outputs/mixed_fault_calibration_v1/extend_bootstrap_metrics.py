"""Complete paired uncertainty for all prespecified primary TWA metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_analysis import (ROOT, OUT, SEEDS, SOURCES, ARMS, ENDPOINTS, Q, METHODS, metrics,
                          calibrators, get_data, platt, sha, write_csv, write_json)

METRICS=("auprc","auroc","brier","ece15","sensitivity","specificity",
         "predicted_positive_fraction","false_positives_per_1000","false_negatives_per_1000")
MODES=("raw","calibrated")
COMPS=("ordinary_resnet","chest_only_resnet","huang_pipeline")

def make_cache(y,c,r,threshold):
    y=np.asarray(y,dtype=np.uint8); c=np.asarray(c,dtype=np.float64); r=np.asarray(r,dtype=np.float64); n=len(y)
    scores=np.r_[c,r];order=np.argsort(-scores,kind="mergesort");ordered=scores[order]
    starts=np.r_[True,ordered[1:]!=ordered[:-1]];group_sorted=np.cumsum(starts,dtype=np.int32)-1
    inverse=np.empty(len(order),dtype=np.int32);inverse[order]=group_sorted
    return {"y":y,"c":c,"r":r,"gc":inverse[:n],"gr":inverse[n:],"groups":int(group_sorted[-1])+1,
        "bc":np.minimum((c*15).astype(np.int32),14),"br":np.minimum((r*15).astype(np.int32),14),
        "cc":c>=threshold,"rc":r>=threshold}

def fast_metrics(o,mult,q):
    y=o["y"];n=len(y);m=np.asarray(mult,dtype=np.float64);yc=y.astype(float);nc=1.0-yc;g=o["groups"]
    pc=np.bincount(o["gc"],weights=m*yc,minlength=g);pr=np.bincount(o["gr"],weights=m*yc,minlength=g)
    fc=np.bincount(o["gc"],weights=m*nc,minlength=g);fr=np.bincount(o["gr"],weights=m*nc,minlength=g)
    tp=(1-q)*pc+q*pr;fp=(1-q)*fc+q*fr;P=float(tp.sum());N=float(fp.sum())
    ctp=np.cumsum(tp);cfp=np.cumsum(fp);denom=ctp+cfp
    precision=np.divide(ctp,denom,out=np.zeros_like(ctp),where=denom>0)
    ap=float(np.sum((tp/P)*precision))
    auc=float(np.sum(tp*(N-cfp+.5*fp))/(P*N))
    total=float(m.sum())
    brier=float(np.dot(m,(1-q)*(o["c"]-y)**2+q*(o["r"]-y)**2)/total)
    def parts(bins,p):return (np.bincount(bins,weights=m,minlength=15),np.bincount(bins,weights=m*p,minlength=15),np.bincount(bins,weights=m*yc,minlength=15))
    wc,pcbin,ycbin=parts(o["bc"],o["c"]);wr,prbin,yrbin=parts(o["br"],o["r"])
    wb=(1-q)*wc+q*wr;pb=(1-q)*pcbin+q*prbin;yb=(1-q)*ycbin+q*yrbin;active=wb>0
    ece=float(np.sum(wb[active]/total*np.abs(yb[active]/wb[active]-pb[active]/wb[active])))
    callc=o["cc"];callr=o["rc"]
    tp_call=float(np.dot(m*yc,(1-q)*callc+q*callr));fn=float(np.dot(m*yc,(1-q)*(~callc)+q*(~callr)))
    fp_call=float(np.dot(m*nc,(1-q)*callc+q*callr));tn=float(np.dot(m*nc,(1-q)*(~callc)+q*(~callr)))
    return {"auprc":ap,"auroc":auc,"brier":brier,"ece15":ece,"sensitivity":tp_call/(tp_call+fn),
        "specificity":tn/(tn+fp_call),"predicted_positive_fraction":(tp_call+fp_call)/total,
        "false_positives_per_1000":1000*fp_call/total,"false_negatives_per_1000":1000*fn/total}

def validate_fast_metrics():
    rng=np.random.default_rng(809);y=rng.integers(0,2,121,dtype=np.uint8);c=rng.random(121);r=rng.random(121)
    c[:4]=[.5,.5,0,1];r[:4]=[.5,.2,0,1];m=rng.integers(0,3,121);threshold=.4
    from run_analysis import metrics as reference
    for q in (0.,.001,.2,1.):
        got=fast_metrics(make_cache(y,c,r,threshold),m,q);ww=np.r_[m*(1-q),m*q];yy=np.r_[y,y];pp=np.r_[c,r]
        ref=reference(yy,pp,threshold,ww)
        for name in METRICS:
            if not np.isclose(got[name],ref[name],rtol=1e-11,atol=1e-12):raise RuntimeError(f"grouped metric reference mismatch {name} q={q}: {got[name]} vs {ref[name]}")

def main():
    amendment=OUT/"FROZEN_PROTOCOL_AMENDMENT_002.json"
    a=json.loads(amendment.read_text(encoding="utf-8"))
    if a["status"]!="FROZEN_BEFORE_EXTENDED_METRIC_BOOTSTRAP" or a["protocol_sha256"]!=sha(OUT/"FROZEN_ANALYSIS_PROTOCOL.json"):
        raise RuntimeError("Extended bootstrap amendment is not bound to the frozen protocol")
    fits,_=calibrators(); data,_,_=get_data()
    # Add the already-existing Ningbo Huang pipeline predictions; amendment 001 binds their source files/order.
    for seed in SEEDS:
        y=np.load(ROOT/f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz",allow_pickle=False)["y"].astype(np.uint8)
        c=np.load(ROOT/f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz",allow_pickle=False)["p"].astype(float)
        r=np.load(ROOT/f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz",allow_pickle=False)["p"].astype(float)
        data["Ningbo"]["huang_pipeline"][seed]=(y,c,r)
    abs_arr=np.full((len(SOURCES),len(ARMS),len(MODES),len(Q),len(METRICS),1000),np.nan,dtype=np.float64)
    contrasts=np.full((len(SOURCES),len(COMPS),len(MODES),len(Q),len(METRICS),1000),np.nan,dtype=np.float64)
    validate_fast_metrics()
    # Pre-sort fixed scores once. Bootstrap resampling changes multiplicities, not score order.
    cache={}
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                y,c,r=data[source][arm][seed]
                for mode in MODES:
                    if arm=="huang_pipeline" and mode=="calibrated":continue
                    if arm=="huang_pipeline":
                        if source=="Ningbo":th=float(np.load(ROOT/f"outputs/ningbo_confirmation_v2/seed_{seed}/validation.npz",allow_pickle=False)["thresholds"][0])
                        else:
                            hp=json.loads((ROOT/"outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
                            th=float(hp["thresholds"]["baseline"][str(seed)][0])
                        cuse,ruse=c[:,0],r[:,0]
                    else:
                        fit=fits[arm][seed][0]
                        if mode=="raw":cuse,ruse=c[:,0],r[:,0];th=fit["threshold_raw"]
                        else:cuse,ruse=platt(c[:,0],fit["a"],fit["b"]),platt(r[:,0],fit["a"],fit["b"]);th=fit["threshold_calibrated"]
                    cache[(source,arm,seed,mode)]=make_cache(y[:,0],cuse,ruse,th)
    rng=np.random.default_rng(20261017)
    for source_i,source in enumerate(SOURCES):
        n=len(data[source]["ordinary_resnet"][SEEDS[0]][0])
        for bi in range(1000):
            if bi%100==0: print(f"all-TWA metric bootstrap {source}: {bi}/1000",flush=True)
            ix=rng.integers(0,n,size=n)
            multiplicity=np.bincount(ix,minlength=n)
            seed_means={arm:{mode:{q:{} for q in Q} for mode in MODES} for arm in ARMS}
            for arm in ARMS:
                for mode in MODES:
                    if arm=="huang_pipeline" and mode=="calibrated":continue
                    for q in Q:
                        per_seed=[fast_metrics(cache[(source,arm,seed,mode)],multiplicity,q) for seed in SEEDS]
                        for m in METRICS:
                            val=float(np.mean([row[m] for row in per_seed]))
                            seed_means[arm][mode][q][m]=val
                            abs_arr[source_i,ARMS.index(arm),MODES.index(mode),Q.index(q),METRICS.index(m),bi]=val
            for comp_i,comp in enumerate(COMPS):
                for mode in MODES:
                    if comp=="huang_pipeline" and source=="Ningbo": continue
                    if mode=="calibrated" and comp=="huang_pipeline": continue
                    for q in Q:
                        for mi,m in enumerate(METRICS):
                            pval=seed_means["physics_aware_resnet"][mode][q].get(m)
                            cval=seed_means[comp][mode][q].get(m)
                            if pval is not None and cval is not None:
                                contrasts[source_i,comp_i,MODES.index(mode),Q.index(q),mi,bi]=pval-cval
    artifact=OUT/"BOOTSTRAP_METRICS_REPLICATES.npz"
    np.savez_compressed(artifact,absolute=abs_arr,paired_contrasts=contrasts,sources=np.asarray(SOURCES),arms=np.asarray(ARMS),
        modes=np.asarray(MODES),q_grid=np.asarray(Q,dtype=float),metrics=np.asarray(METRICS),competitors=np.asarray(COMPS),
        rng_seed=np.asarray(20261017),replicates=np.asarray(1000))
    summary=[]
    for si,source in enumerate(SOURCES):
        for ai,arm in enumerate(ARMS):
            for mi,mode in enumerate(MODES):
                for qi,q in enumerate(Q):
                    for xi,m in enumerate(METRICS):
                        v=abs_arr[si,ai,mi,qi,xi,:]
                        if not np.isfinite(v).all(): continue
                        summary.append({"source":source,"method":arm,"score_mode":mode,"q_fault":q,"endpoint":"TWA","metric":m,
                            "bootstrap_mean":float(v.mean()),"ci_2_5":float(np.quantile(v,.025)),"ci_97_5":float(np.quantile(v,.975)),
                            "replicates":1000,"conditional_on":"selected checkpoints, fixed source-validation calibrators and thresholds"})
        for ci,comp in enumerate(COMPS):
            for mi,mode in enumerate(MODES):
                for qi,q in enumerate(Q):
                    for xi,m in enumerate(METRICS):
                        v=contrasts[si,ci,mi,qi,xi,:]
                        if not np.isfinite(v).all():continue
                        summary.append({"source":source,"contrast":f"physics_aware_minus_{comp}","score_mode":mode,"q_fault":q,
                            "endpoint":"TWA","metric":m,"bootstrap_mean":float(v.mean()),"ci_2_5":float(np.quantile(v,.025)),
                            "ci_97_5":float(np.quantile(v,.975)),"replicates":1000,
                            "conditional_on":"selected checkpoints, fixed source-validation calibrators and thresholds"})
    outcsv=OUT/"BOOTSTRAP_METRICS_SUMMARY.csv";write_csv(outcsv,summary)
    # Match the expanded AUPRC estimates to the original primary bootstrap. A mismatch means the same
    # record resampling stream or seed-then-average calculation was not reproduced.
    old=pd.read_csv(OUT/"BOOTSTRAP_REPLICATES.csv")
    for row in summary:
        if row["metric"]!="auprc" or "contrast" not in row:continue
        comp=row["contrast"]
        z=old[(old.source==row["source"])&(old.score_mode==row["score_mode"])&(old.q_fault==row["q_fault"])&(old.contrast==comp)]
        if len(z)!=1000:raise RuntimeError(f"Original bootstrap draw binding missing for {row['source']} {comp} q={row['q_fault']} {row['score_mode']}")
        vv=z.difference.to_numpy(float)
        if abs(vv.mean()-row["bootstrap_mean"])>1e-12 or abs(np.quantile(vv,.025)-row["ci_2_5"])>1e-12 or abs(np.quantile(vv,.975)-row["ci_97_5"])>1e-12:
            raise RuntimeError(f"Extended TWA AUPRC bootstrap does not reproduce original intervals: {row}")
    write_json(OUT/"BOOTSTRAP_METRICS_COMPLETE.json",{"status":"PASS","amendment_002_sha256":sha(amendment),
        "replicates":1000,"sources":list(SOURCES),"primary_endpoint":"TWA","metrics":list(METRICS),
        "q_scenarios":list(Q),"absolute_metric_summaries":sum(1 for x in summary if "method" in x),
        "paired_contrast_summaries":sum(1 for x in summary if "contrast" in x),
        "AUPRC_reproduces_original_bootstrap_exactly":True,"replicate_artifact_sha256":sha(artifact),
        "summary_sha256":sha(outcsv),"not_retrained":True})
    print(f"ALL_TWA_BOOTSTRAP_COMPLETE summaries={len(summary)} npz={artifact}",flush=True)

if __name__=="__main__":main()
