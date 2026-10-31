"""Frozen Ningbo clean/RA-LL weighted mixture analysis and validation-only calibration."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SEEDS=(104729,130363,155921);LABELS=("TWA","SB","STach")
Q=(0,.001,.005,.01,.05,.1,1.)
def loadcsv(p):
 with open(p,encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def atomic_csv(p,rs):
 if not rs:return
 with open(p,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def fit_platt(p,y):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6);y=np.asarray(y,dtype=np.float64);z=np.log(p/(1-p))
 def objective(theta):
  a=np.exp(theta[0]);b=theta[1];eta=a*z+b
  loss=np.mean(np.logaddexp(0,eta)-y*eta)
  residual=expit(eta)-y
  grad=np.array([np.mean(residual*a*z),np.mean(residual)])
  return loss,grad
 result=minimize(objective,np.array([0.,0.]),method="L-BFGS-B",jac=True,bounds=((-12,12),(-30,30)),options={"maxiter":2000,"ftol":1e-13,"gtol":1e-8,"maxls":50})
 a=float(np.exp(result.x[0]));b=float(result.x[1])
 if not result.success or not np.isfinite(a+b):raise RuntimeError("Platt fit failed: "+str(result.message))
 return a,b,{"success":bool(result.success),"iterations":int(result.nit),"loss":float(result.fun),"message":str(result.message)}
def weighted_ece(y,p,w):
 total=float(np.sum(w));out=0.
 for j in range(15):
  lo,hi=j/15,(j+1)/15;z=(p>=lo)&((p<hi) if j<14 else (p<=hi));mass=float(w[z].sum())
  if mass>0:out+=mass/total*abs(float(np.average(y[z],weights=w[z]))-float(np.average(p[z],weights=w[z])))
 return out
def metrics(y,p,w,t):
 pred=p>=t;pos=y==1;neg=~pos;sumw=float(w.sum())
 return {"auprc":float(average_precision_score(y,p,sample_weight=w)),"auroc":float(roc_auc_score(y,p,sample_weight=w)),"brier":float(np.average((p-y)**2,weights=w)),"ece15":weighted_ece(y,p,w),"sensitivity":float(w[pred&pos].sum()/w[pos].sum()),"specificity":float(w[(~pred)&neg].sum()/w[neg].sum()),"predicted_positive_fraction":float(w[pred].sum()/sumw),"threshold":float(t)}
def main():
 split=np.load(ROOT/"outputs/mixed_fault_calibration_v1/calibration_split_indices.npz",allow_pickle=False);A=split["A"].astype(int);B=split["B"].astype(int)
 old=loadcsv(ROOT/"outputs/ningbo_four_arm_replication_v1/MIXTURE_RESULTS_BY_SEED.csv")
 rows=[];provenance={"calibration_split_sha256":hashlib.sha256((ROOT/"outputs/mixed_fault_calibration_v1/calibration_split_indices.npz").read_bytes()).hexdigest(),"new_fits":{},"new_thresholds_validation_B_only":True,"test_data_used_for_fit_or_threshold":False,"q_fault":list(Q)}
 for seed in SEEDS:
  for arm in ("physics_p010","physics_p025"):
   f=OUT/"training"/"Ningbo"/arm/f"seed_{seed}"/"validation.npz"
   with np.load(f,allow_pickle=False) as z:yv=z["y"].astype(np.uint8);pv=z["p"].astype(np.float64)
   cal=[];ab=[];tb=[]
   for j,endpoint in enumerate(LABELS):
    a,b,opt=fit_platt(pv[A,j],yv[A,j]);cal.append(expit(a*np.log(np.clip(pv[:,j],1e-6,1-1e-6)/(1-np.clip(pv[:,j],1e-6,1-1e-6)))+b))
    ab.append({"endpoint":endpoint,"a":a,"b":b,"fit_rows":len(A),"optimizer":opt})
    tb.append(float(np.quantile(pv[B][yv[B,j]==0,j],.9,method="linear")))
   cal=np.stack(cal,axis=1);tc=np.asarray([np.quantile(cal[B,j][yv[B,j]==0],.9,method="linear") for j in range(3)])
   provenance["new_fits"][f"{arm}/{seed}"]={"fit_split":"existing Ningbo validation A","fit_rows":len(A),"fits":ab,"threshold_validation_B_raw":tb,"threshold_validation_B_calibrated":tc.tolist()}
   # Read already frozen test probabilities; score calibration uses validation fit only.
   pp=OUT/"predictions"/"Ningbo"/arm/f"seed_{seed}"
   with np.load(pp/"clean.npz",allow_pickle=False) as z:yc=z["y"].astype(np.uint8);pc=z["p"].astype(np.float64)
   with np.load(pp/"RA_LL_reversal.npz",allow_pickle=False) as z:yr=z["y"].astype(np.uint8);pr=z["p"].astype(np.float64)
   if not np.array_equal(yc,yr):raise RuntimeError("clean/RA_LL labels or order mismatch")
   calc=np.stack([expit(ab[j]["a"]*np.log(np.clip(pc[:,j],1e-6,1-1e-6)/(1-np.clip(pc[:,j],1e-6,1-1e-6)))+ab[j]["b"]) for j in range(3)],axis=1)
   calr=np.stack([expit(ab[j]["a"]*np.log(np.clip(pr[:,j],1e-6,1-1e-6)/(1-np.clip(pr[:,j],1e-6,1-1e-6)))+ab[j]["b"]) for j in range(3)],axis=1)
   for q in Q:
    w=np.r_[np.full(len(yc),1-q,dtype=float),np.full(len(yc),q,dtype=float)]
    for mode,cscore,rscore,thresholds in (("raw",pc,pr,np.asarray(tb)),("calibrated",calc,calr,tc)):
     for j,endpoint in enumerate(LABELS):
      y=np.r_[yc[:,j],yr[:,j]];p=np.r_[cscore[:,j],rscore[:,j]];res=metrics(y,p,w,float(thresholds[j]))
      rows.append({"arm":arm,"seed":seed,"endpoint":endpoint,"q":q,"score_mode":mode,**res,"calibration_a":ab[j]["a"],"calibration_b":ab[j]["b"],"threshold_source":"Ningbo validation B clean negatives, 90th percentile","mixture_source":"weighted pooled clean/RA_LL observations"})
 atomic_csv(OUT/"MIXTURE_RESULTS_NEW_BY_SEED.csv",rows)
 # Preserve precomputed comparator mixture curves without refitting them.
 for r in old:
  arm={"clean":"ordinary","physics_aware":"physics_p050"}.get(r["arm"],r["arm"])
  if arm not in ("ordinary","physics_p050","combined"):continue
  rows.append({"arm":arm,"seed":int(r["seed"]),"endpoint":r["endpoint"],"q":float(r["q"]),"score_mode":r["score_mode"],**{k:r.get(k,"") for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","threshold","calibration_a","calibration_b")},"threshold_source":"historical validation-derived threshold; four-arm provenance","mixture_source":"outputs/ningbo_four_arm_replication_v1/MIXTURE_RESULTS_BY_SEED.csv"})
 atomic_csv(OUT/"MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv",rows)
 (OUT/"CALIBRATION_PROVENANCE_NEW.json").write_text(json.dumps(provenance,indent=2,sort_keys=True),encoding="utf-8")
 print(json.dumps({"new_rows":len([r for r in rows if str(r.get('arm','')).startswith('physics_p')]),"all_rows":len(rows),"new_calibrators":len(provenance['new_fits'])},sort_keys=True))
if __name__=="__main__":main()
