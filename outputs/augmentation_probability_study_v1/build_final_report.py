"""Join frozen new-model metrics to audited historical comparator summaries."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SEEDS=(104729,130363,155921)
LABELS=("TWA","SB","STach")
def rows(path):
 with open(path,encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def readnum(row,key):
 try:return float(row[key])
 except (ValueError,TypeError,KeyError):return float("nan")
def main():
 allrows=[]
 # New-model endpoint metrics and all 792 mask rows.
 for cohort in ("Chapman","Ningbo"):
  for r in rows(OUT/f"METRICS_{cohort.upper()}_NEW_BY_SEED.csv"):
   allrows.append({"cohort":cohort,"arm":r["arm"],"seed":r["seed"],"condition":r["condition"],"endpoint":r["endpoint"],**{k:r.get(k,"") for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","accuracy","threshold")},"threshold_source":"own clean validation negatives, 90th percentile","metric_source":"new frozen probability study"})
 # Audited Chapman M1 table contains ordinary (cached clean baseline), PA, combined, masking.
 m1=rows(ROOT/"outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv")
 for r in m1:
  arm={"cached_clean_baseline":"ordinary"}.get(r["arm"],r["arm"])
  if arm not in ("ordinary","physics_aware","combined"):continue
  allrows.append({"cohort":"Chapman","arm":arm,"seed":r["seed"],"condition":r["condition"],"endpoint":r["endpoint"],"auprc":r["auprc"],"auroc":r["auroc"],"brier":"","ece15":r["ece_15"],"sensitivity":r["sensitivity"],"specificity":r["specificity"],"predicted_positive_fraction":"","accuracy":"","threshold":r["threshold"],"threshold_source":"historical native validation threshold; M1 results table","metric_source":"outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv"})
 # The held-out cycle package is the frozen source for cycle-condition comparators.
 for r in rows(ROOT/"outputs/heldout_limb_cycles_v1/METRICS_BY_MODEL_SEED.csv"):
  if r["dataset"] not in ("Chapman","Ningbo") or r["arm"] not in ("ordinary","physics_aware","combined"):continue
  allrows.append({"cohort":r["dataset"],"arm":r["arm"],"seed":r["seed"],"condition":r["condition"],"endpoint":r["endpoint"],**{k:r.get(k,"") for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction")},"accuracy":"","threshold":r["threshold"],"threshold_source":"frozen historical cycle-package threshold","metric_source":"outputs/heldout_limb_cycles_v1/METRICS_BY_MODEL_SEED.csv"})
 # Corrected Ningbo masks and four-arm metrics are already joined in this final table.
 for r in rows(ROOT/"outputs/ningbo_four_arm_replication_v1/RESULTS_BY_ARM_SEED.csv"):
  if r["arm"] not in ("clean","physics_aware","combined"):continue
  arm="ordinary" if r["arm"]=="clean" else r["arm"]
  allrows.append({"cohort":"Ningbo","arm":arm,"seed":r["seed"],"condition":r["condition"],"endpoint":r["endpoint"],**{k:r.get(k,"") for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","threshold")},"accuracy":"","threshold_source":"historical native validation threshold; four-arm result table","metric_source":"outputs/ningbo_four_arm_replication_v1/RESULTS_BY_ARM_SEED.csv"})
 # Remove duplicate historic comparator rows where the cycle package overlaps clean/RA_LL.
 # Prefer cycle-package values for cycles and four-arm/M1 values for common conditions.
 unique={}
 for r in allrows:
  k=(r["cohort"],str(r["arm"]),str(r["seed"]),r["condition"],r["endpoint"])
  old=unique.get(k)
  if old is None:unique[k]=r;continue
  if r["condition"].startswith("cycle_slots_"):unique[k]=r
  elif r["cohort"]=="Ningbo" and "four_arm" in r["metric_source"]:unique[k]=r
  elif r["cohort"]=="Chapman" and "mitigation_m1" in r["metric_source"]:unique[k]=r
 allrows=list(unique.values())
 keys={(r["cohort"],str(r["arm"]),str(r["seed"]),r["condition"],r["endpoint"]):r for r in allrows}
 def write(path,data):
  if not data:return
  with open(path,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 write(OUT/"METRICS_ALL_ARMS_BY_SEED.csv",allrows)
 contrasts=[]
 pairs={"primary":[("physics_p025","combined"),("physics_p025","physics_aware")],"secondary":[("physics_p010","ordinary"),("physics_p010","physics_aware")]}
 conditions=("clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201","mask5_0_1_2_3_5")
 masknames=["mask5_"+"_".join(map(str,ix)) for ix in __import__("itertools").combinations(range(12),5)]
 for cohort in ("Chapman","Ningbo"):
  newmask={}
  for arm_a,arm_b in pairs["primary"]+pairs["secondary"]:
   tier="primary" if (arm_a,arm_b) in pairs["primary"] else "secondary"
   for condition in conditions:
    for endpoint in ("TWA","SB","STach"):
     diffs=[]
     for seed in SEEDS:
      a=keys.get((cohort,arm_a,str(seed),condition,endpoint));b=keys.get((cohort,arm_b,str(seed),condition,endpoint))
      if a is None or b is None:continue
      for metric in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","accuracy"):
       av,bv=readnum(a,metric),readnum(b,metric)
       contrasts.append({"cohort":cohort,"tier":tier,"contrast":f"{arm_a}_minus_{arm_b}","seed":seed,"condition":condition,"endpoint":endpoint,"metric":metric,"arm_a":av,"arm_b":bv,"difference":av-bv,"metric_source_a":a["metric_source"],"metric_source_b":b["metric_source"]})
       if metric=="auprc":diffs.append(av-bv)
     if endpoint=="TWA" and condition in ("clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201","mask5_0_1_2_3_5"):
      pass
  # Average-mask point contrasts use matched seed-level means across all 792 conditions.
  for arm_a,arm_b in pairs["primary"]+pairs["secondary"]:
   tier="primary" if (arm_a,arm_b) in pairs["primary"] else "secondary"
   for seed in SEEDS:
    a=[readnum(keys[(cohort,arm_a,str(seed),name,"TWA")],"auprc") for name in masknames if (cohort,arm_a,str(seed),name,"TWA") in keys]
    b=[readnum(keys[(cohort,arm_b,str(seed),name,"TWA")],"auprc") for name in masknames if (cohort,arm_b,str(seed),name,"TWA") in keys]
    if len(a)==len(b)==792:
     contrasts.append({"cohort":cohort,"tier":tier,"contrast":f"{arm_a}_minus_{arm_b}","seed":seed,"condition":"mean_792_masks","endpoint":"TWA","metric":"auprc","arm_a":float(np.mean(a)),"arm_b":float(np.mean(b)),"difference":float(np.mean(a)-np.mean(b)),"metric_source_a":"792 per-mask values; within-seed mean","metric_source_b":"792 per-mask values; within-seed mean"})
 write(OUT/"PAIRED_CONTRASTS_POINT_BY_SEED.csv",contrasts)
 summaries=[]
 for cohort in ("Chapman","Ningbo"):
  for arm in ("physics_p010","physics_p025","ordinary","physics_aware","combined"):
   for condition in conditions+("mean_792_masks",):
    for endpoint in LABELS:
     metric="auprc"
     if condition=="mean_792_masks":
      vals=[]
      for seed in SEEDS:
       vals.extend([readnum(r,"auprc") for r in allrows if r["cohort"]==cohort and str(r["arm"])==arm and str(r["seed"])==str(seed) and r["endpoint"]==endpoint and r["condition"].startswith("mask5_")])
     else: vals=[readnum(r,metric) for r in allrows if r["cohort"]==cohort and str(r["arm"])==arm and r["condition"]==condition and r["endpoint"]==endpoint]
     if vals:summaries.append({"cohort":cohort,"arm":arm,"condition":condition,"endpoint":endpoint,"metric":metric,"n":len(vals),"mean":float(np.mean(vals)),"sd":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0})
 write(OUT/"METRICS_MEAN_SD_BY_ARM.csv",summaries)
 print(json.dumps({"metric_rows":len(allrows),"contrast_rows":len(contrasts),"summary_rows":len(summaries)},sort_keys=True))
if __name__=="__main__":main()
