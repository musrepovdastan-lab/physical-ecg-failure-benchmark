"""Synchronized paired TWA AUPRC bootstrap over existing cohort resampling indices."""
from __future__ import annotations
import csv, hashlib, itertools, json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
SEEDS=(104729,130363,155921);NEW=("physics_p010","physics_p025");HIST=("ordinary","physics_aware","combined")
MASKS=tuple(itertools.combinations(range(12),5));MASK_NAMES=tuple("mask5_"+"_".join(map(str,x)) for x in MASKS)
SCALAR=("clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201","mask5_0_1_2_3_5")
def atomic_npz(p,**arr):
 p=Path(p);tmp=p.with_name("."+p.stem+".tmp.npz");np.savez_compressed(tmp,**arr);tmp.replace(p)
def path_for(cohort,arm,seed,condition):
 if arm.startswith("physics_p0"):return OUT/"predictions"/cohort/arm/f"seed_{seed}"/(condition+".npz")
 if condition.startswith("cycle_slots_"):return ROOT/"outputs/heldout_limb_cycles_v1/predictions"/cohort/arm/f"seed_{seed}"/(condition+".npz")
 if cohort=="Chapman":
  if arm=="ordinary":
   if seed==104729:
    if condition.startswith("mask5_"):return ROOT/"outputs/phase1b_matched_reversal_development_v6/predictions"/(condition+".npz")
    return ROOT/"outputs/phase1_chapman_seed1_v4/test_predictions_v4"/(condition+".npz")
   return ROOT/"outputs/phase2_chapman_multiseed_v7"/f"seed_{seed}"/"test_predictions"/(condition+".npz")
  return ROOT/"outputs/mitigation_m1_v1"/arm/f"seed_{seed}"/"test_predictions"/(condition+".npz")
 if condition.startswith("mask5_"):
  recarm="baseline_v2" if arm=="ordinary" else "physics_aware_m2"
  if arm in ("ordinary","physics_aware"):return ROOT/"outputs/mask_semantics_recovery_v2"/recarm/f"seed_{seed}"/"test_predictions"/(condition+".npz")
  return ROOT/"outputs/ningbo_four_arm_replication_v1"/arm/f"seed_{seed}"/"test_predictions"/(condition+".npz")
 if arm=="ordinary":return ROOT/"outputs/ningbo_confirmation_v2"/f"seed_{seed}"/"test_predictions"/(condition+".npz")
 if arm=="physics_aware":return ROOT/"outputs/mitigation_m2_ningbo_v1"/f"seed_{seed}"/"test_predictions"/(condition+".npz")
 return ROOT/"outputs/ningbo_four_arm_replication_v1"/arm/f"seed_{seed}"/"test_predictions"/(condition+".npz")
@lru_cache(maxsize=2)
def cohort_labels(cohort):
 if cohort=="Chapman":
  from src.chapman_phase1 import labels as label_fn
  p=ROOT/"outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json";rows=json.loads(p.read_text(encoding="utf-8"))["rows"]
  ids=[r["record_id"] for r in rows if r["split"]=="test"]
  return np.asarray([label_fn(r) for r in ids],dtype=np.uint8)[:,0]
 n=ROOT/"outputs/ningbo_confirmation_v2";records=json.loads((n/"CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"];by={r["record_id"]:r for r in records};split=json.loads((n/"SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
 ids=[r["record_id"] for r in split if r["split"]=="test"]
 return np.asarray([by[r]["labels"][0] for r in ids],dtype=np.uint8)
def arrays(cohort,arm,seed,condition):
 f=path_for(cohort,arm,seed,condition)
 if not f.is_file():raise FileNotFoundError(f"required comparator prediction missing: {f}")
 with np.load(f,allow_pickle=False) as z:
  p=z["p"][:,0].astype(np.float64)
  y=cohort_labels(cohort)
  if "y" in z.files and not np.array_equal(z["y"][:,0].astype(np.uint8),y):raise RuntimeError(f"prediction labels/order mismatch: {f}")
 return y,p
def counts_from_indices(indices,n):return np.stack([np.bincount(row,minlength=n) for row in indices]).astype(np.uint8)
def optimized_mask_bootstrap(y,ps,counts):
 """Mean mask AP for 1000 record bootstraps; tie-aware exact AP via sorted groups."""
 n=len(y);total_pos=np.sum(counts*y[None,:],axis=1).astype(np.float64);acc=np.zeros(counts.shape[0],dtype=np.float64);point=[]
 for j,condition in enumerate(MASK_NAMES):
  p=ps(condition);point.append(float(average_precision_score(y,p)))
  order=np.argsort(-p,kind="mergesort");sp=p[order];ends=np.flatnonzero(np.r_[sp[1:]!=sp[:-1],True]);yp=y[order].astype(np.int32)
  for lo in range(0,len(counts),16):
   c=counts[lo:lo+16][:,order].astype(np.int32,copy=False)
   cumall=np.cumsum(c,axis=1,dtype=np.int32);cumpos=np.cumsum(c*yp[None,:],axis=1,dtype=np.int32)
   tp=cumpos[:,ends].astype(np.float64);tot=cumall[:,ends].astype(np.float64)
   newpos=np.diff(np.concatenate((np.zeros((len(c),1)),tp),axis=1),axis=1)
   recall=np.divide(newpos,total_pos[lo:lo+len(c),None],out=np.zeros_like(newpos,dtype=np.float64),where=total_pos[lo:lo+len(c),None]>0)
   precision=np.divide(tp,tot,out=np.zeros_like(tp,dtype=np.float64),where=tot>0)
   ap=np.sum(recall*precision,axis=1)
   acc[lo:lo+len(c)]+=ap
  if (j+1)%100==0:print(json.dumps({"stage":"MASK_BOOTSTRAP","masks_completed":j+1,"masks_total":792}),flush=True)
 return np.asarray(point).mean(),acc/792,np.asarray(point)
def exact_mask_equivalence(y,p,idx):
 # Audit optimized rank/cumulative formula against sklearn on real predictions and boot indices.
 counts=counts_from_indices(idx[:20],len(y));order=np.argsort(-p,kind="mergesort");sp=p[order];ends=np.flatnonzero(np.r_[sp[1:]!=sp[:-1],True]);yp=y[order].astype(np.int32);total_pos=np.sum(counts*y[None,:],axis=1).astype(float)
 c=counts[:,order].astype(np.int32);ta=np.cumsum(c,axis=1,dtype=np.int32);tp=np.cumsum(c*yp[None,:],axis=1,dtype=np.int32);tpg=tp[:,ends].astype(float);tog=ta[:,ends].astype(float);ng=np.diff(np.concatenate((np.zeros((len(c),1)),tpg),axis=1),axis=1);recall=np.divide(ng,total_pos[:,None],out=np.zeros_like(ng),where=total_pos[:,None]>0);precision=np.divide(tpg,tog,out=np.zeros_like(tpg),where=tog>0);fast=np.sum(recall*precision,axis=1)
 slow=np.asarray([average_precision_score(y[row],p[row]) for row in idx[:20]])
 return float(np.max(np.abs(fast-slow))),fast,slow
def main():
 results=[];cache=OUT/"bootstrap_draws";cache.mkdir(exist_ok=True)
 for cohort in ("Chapman","Ningbo"):
  bootfile=ROOT/("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if cohort=="Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy")
  idx=np.load(bootfile,allow_pickle=False).astype(np.int32)
  if idx.shape[0]!=1000:raise RuntimeError("cohort bootstrap must contain 1000 pre-existing replicates")
  # Validate cached mask artifacts and new prediction labels/order against the same frozen cohort y.
  y0,_=arrays(cohort,"ordinary",SEEDS[0],"clean")
  audit=[]
  for a in ("physics_p010","physics_aware"):
   yy,p=arrays(cohort,a,SEEDS[0],"clean");err,_,_=exact_mask_equivalence(yy,p,idx)
   if not np.array_equal(yy,y0) or err>1e-10:raise RuntimeError(f"bootstrap metric implementation audit failed: {cohort} {a} maxdiff={err}")
   audit.append({"arm":a,"max_abs_difference_fast_vs_sklearn":err,"real_data":True,"replicates_compared":20})
  counts=counts_from_indices(idx,len(y0))
  model_draws={};model_points={}
  models=NEW+HIST
  for arm in models:
   for seed in SEEDS:
    for cond in SCALAR:
     cachefile=cache/f"{cohort}_{arm}_{seed}_{cond}.npz"
     y,p=arrays(cohort,arm,seed,cond)
     if not np.array_equal(y,y0):raise RuntimeError(f"record order/labels mismatch: {cohort} {arm} {seed} {cond}")
     model_points[(arm,seed,cond)]=float(average_precision_score(y,p))
     if cachefile.exists():
      with np.load(cachefile,allow_pickle=False) as z:draw=z["draws"]
     else:
      draw=np.asarray([average_precision_score(y[row],p[row]) for row in idx],dtype=np.float32);atomic_npz(cachefile,draws=draw)
     model_draws[(arm,seed,cond)]=draw
    cfile=cache/f"{cohort}_{arm}_{seed}_mean_792_masks.npz"
    if cfile.exists():
     with np.load(cfile,allow_pickle=False) as z:draw=z["draws"];point=float(z["point"])
    else:
     y,_=arrays(cohort,arm,seed,"clean")
     if not np.array_equal(y,y0):raise RuntimeError("mask bootstrap label order mismatch")
     def loadmask(name):
      yy,p=arrays(cohort,arm,seed,name)
      if not np.array_equal(yy,y0):raise RuntimeError("mask record order mismatch: "+name)
      return p
     point,draw,permask=optimized_mask_bootstrap(y,loadmask,counts);atomic_npz(cfile,draws=draw,point=np.asarray(point),per_mask_point=permask)
    model_draws[(arm,seed,"mean_792_masks")]=draw;model_points[(arm,seed,"mean_792_masks")]=float(point)
    results.append({"cohort":cohort,"arm":arm,"seed":seed,"condition":"mean_792_masks","full_cohort_mean_mask_auprc":point,"bootstrap_draws":len(draw)})
   print(json.dumps({"stage":"MODEL_BOOTSTRAP_COMPLETE","cohort":cohort,"arm":arm,"models_done":models.index(arm)+1,"total_models":len(models)}),flush=True)
  for a,b in (("physics_p025","combined"),("physics_p025","physics_aware"),("physics_p010","ordinary"),("physics_p010","physics_aware")):
   tier="primary" if a=="physics_p025" else "secondary"
   for cond in SCALAR+("mean_792_masks",):
    diffs=[]
    for rep in range(1000):
     vals=[]
     for seed in SEEDS:vals.append(float(model_draws[(a,seed,cond)][rep])-float(model_draws[(b,seed,cond)][rep]))
     diffs.append(float(np.mean(vals)))
    point=np.mean([model_points[(a,s,cond)]-model_points[(b,s,cond)] for s in SEEDS])
    results.append({"cohort":cohort,"tier":tier,"contrast":f"{a}_minus_{b}","condition":cond,"endpoint":"TWA","metric":"AUPRC","full_cohort_point_difference":point,"bootstrap_mean":float(np.mean(diffs)),"ci95_low":float(np.quantile(diffs,.025)),"ci95_high":float(np.quantile(diffs,.975)),"replicates":1000,"paired":True,"seeds_averaged_within_replicate":True})
  (OUT/f"BOOTSTRAP_AUDIT_{cohort.upper()}.json").write_text(json.dumps({"bootstrap_indices":str(bootfile.relative_to(ROOT)),"shape":list(idx.shape),"mask_metric_validation":audit,"implementation":"paired_bootstrap.py; exact tie-aware vectorized cumulative AP checked against sklearn on 20 existing resamples using real scores"},indent=2,sort_keys=True),encoding="utf-8")
 out=[r for r in results if r.get("replicates")==1000]
 with open(OUT/"PAIRED_BOOTSTRAP_TWA_AUPRC.csv","w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(json.dumps({"bootstrap_contrasts":len(out),"cached_seed_metric_draws":len(list(cache.glob("*.npz")))},sort_keys=True))
if __name__=="__main__":main()
