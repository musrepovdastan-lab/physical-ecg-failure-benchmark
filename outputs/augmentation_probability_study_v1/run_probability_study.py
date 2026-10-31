"""Frozen p=0.10/0.25 physical augmentation study; all artifacts stay in this package."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, os, random, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEEDS=(104729,130363,155921)
ARMS=("physics_p010","physics_p025")
FAULTS=("RA_LA_reversal","RA_LL_reversal","LA_LL_reversal")
LABELS=("TWA","SB","STach")
MASKS=tuple(itertools.combinations(range(12),5))
MASK_NAMES=tuple("mask5_"+"_".join(map(str,x)) for x in MASKS)
CONDITIONS=("clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201",*MASK_NAMES)
EXACT=(0,1,2,3,5)
PROTO=OUT/"FROZEN_PROTOCOL.json"
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.chapman_phase1 import ECGResNetSmall, raw as chapman_raw, labels as chapman_labels
from src.ecg_physics import derive_display, named_reversal
from src import ningbo_confirmation_v1 as ningbo
from src import mask_semantics_recovery_v2 as mask_recovery

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name("."+p.name+"."+uuid.uuid4().hex+".tmp")
 q.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");os.replace(q,p)
def atomic_npz(path,**x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=path.with_name("."+path.stem+"."+uuid.uuid4().hex+".npz")
 np.savez_compressed(q,**x);os.replace(q,path)
def atomic_torch(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name("."+p.stem+"."+uuid.uuid4().hex+".pt")
 torch.save(x,q);os.replace(q,p)
def atomic_csv(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name("."+p.name+"."+uuid.uuid4().hex+".tmp")
 with open(q,"w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(q,p)
def progress(stage,**kw):
 z={"stage":stage,"utc":now(),**kw};atomic_json(OUT/"WORKFLOW_STATUS.json",z);print(json.dumps(z,ensure_ascii=False),flush=True)
def deterministic_u(arm,seed,epoch,index,tag):
 return int.from_bytes(hashlib.sha256(f"{arm}|{seed}|{epoch}|{index}|{tag}".encode()).digest()[:8],"little")/2**64
def augment_raw(x,arm,seed,epoch,index):
 u=deterministic_u(arm,seed,epoch,index,"choice")
 p=.10 if arm=="physics_p010" else .25
 if u>=p:return x,"clean"
 j=min(2,int(deterministic_u(arm,seed,epoch,index,"fault")*3));fault=FAULTS[j]
 return derive_display(named_reversal(np.concatenate((x[:2],x[6:]),axis=0),fault)).astype(np.float32),fault

class TrainDataset(Dataset):
 def __init__(self,records,cohort,mean,std,arm,seed): self.records,self.cohort,self.mean,self.std,self.arm,self.seed,self.epoch=records,cohort,mean,std,arm,seed,1
 def __len__(self): return len(self.records)
 def __getitem__(self,i):
  r=self.records[i]
  if self.cohort=="Chapman": x=chapman_raw(r).astype(np.float32);y=chapman_labels(r)
  else: x=ningbo.raw(r).astype(np.float32);y=r["labels"]
  x,_=augment_raw(x,self.arm,self.seed,self.epoch,i)
  return ((x-self.mean[:,None])/self.std[:,None]).astype(np.float32),np.asarray(y,dtype=np.float32)

def source_data(cohort,seed):
 if cohort=="Chapman":
  sp=ROOT/"outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json"
  rows=json.loads(sp.read_text(encoding="utf-8"))["rows"]
  ids={s:[z["record_id"] for z in rows if z["split"]==s] for s in ("train","validation","test")}
  prep=ROOT/"outputs/phase1_chapman_seed1_v4/preprocessing_v4.npz"
  with np.load(prep,allow_pickle=False) as z: mean,std=z["mean"].astype(np.float32),z["std"].astype(np.float32)
  rec=ids
 else:
  nd=ROOT/"outputs/ningbo_confirmation_v2"; by=json.loads((nd/"CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
  splits=json.loads((nd/"SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"];mp={r["record_id"]:r for r in by}
  rec={s:[mp[r["record_id"]] for r in splits if r["split"]==s] for s in ("train","validation","test")}
  prep=ROOT/f"outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz"
  with np.load(prep,allow_pickle=False) as z: mean,std=z["mean"].astype(np.float32),z["std"].astype(np.float32)
 return rec,mean,std,sp if cohort=="Chapman" else nd/"SPLIT_MEMBERSHIP.json",prep
def labels_for(cohort,records):
 return np.asarray([chapman_labels(r) if cohort=="Chapman" else r["labels"] for r in records],dtype=np.uint8)
def raw_for(cohort,records):
 f=chapman_raw if cohort=="Chapman" else ningbo.raw
 return np.stack([f(r) for r in records]).astype(np.float32)
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def prng_state(): return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def restore_rng(q): random.setstate(q["python"]);np.random.set_state(q["numpy"]);torch.set_rng_state(q["torch"]);torch.cuda.set_rng_state_all(q["cuda"])
def inference(model,x,mean,std):
 model.eval();out=[]
 with torch.no_grad():
  for i in range(0,len(x),128):
   a=torch.from_numpy(((x[i:i+128]-mean[None,:,None])/std[None,:,None]).astype(np.float32)).cuda(non_blocking=True)
   with torch.autocast("cuda",dtype=torch.float16):out.append(torch.sigmoid(model(a)).float().cpu().numpy())
 return np.concatenate(out)
def val_gate(model,cohort,records,mean,std):
 x=raw_for(cohort,records);y=labels_for(cohort,records);p=inference(model,x,mean,std)
 thresholds=np.asarray([np.quantile(p[y[:,i]==0,i],.9,method="linear") for i in range(3)])
 return y,p,thresholds,bool(np.isfinite(p).all() and roc_auc_score(y[:,0],p[:,0])>=.75 and average_precision_score(y[:,0],p[:,0])>=.45)
def fit_unit(cohort,arm,seed):
 d=OUT/"training"/cohort/arm/f"seed_{seed}";d.mkdir(parents=True,exist_ok=True)
 gatefile=d/"validation_gate.json"
 if gatefile.exists() and json.loads(gatefile.read_text(encoding="utf-8")).get("decision")=="PASS":return
 rec,mean,std,split,prep=source_data(cohort,seed);train=rec["train"];val=rec["validation"]
 ytrain=labels_for(cohort,train);pw=np.minimum((len(ytrain)-ytrain.sum(0))/np.maximum(ytrain.sum(0),1),20).astype(np.float32)
 seed_all(seed);model=ECGResNetSmall().cuda();lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device="cuda"));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scaler=torch.amp.GradScaler("cuda")
 resume=d/"resume.pt";hist=[];best=-1.;wait=0;start_epoch=1;best_state=None;started=time.monotonic()
 if resume.exists():
  q=torch.load(resume,weights_only=False,map_location="cuda")
  if q["cohort"]!=cohort or q["arm"]!=arm or q["seed"]!=seed or q["split_sha256"]!=sha(split) or q["protocol_sha256"]!=json.loads(PROTO.read_text())["protocol_sha256"]: raise RuntimeError("resume binding mismatch")
  model.load_state_dict(q["model"]);opt.load_state_dict(q["optimizer"]);scaler.load_state_dict(q["scaler"]);restore_rng(q["rng"]);hist=q["history"];best=q["best"];wait=q["wait"];start_epoch=q["epoch"]+1
  if (d/"best.pt").exists(): best_state=torch.load(d/"best.pt",weights_only=True,map_location="cpu")["state_dict"]
  elif start_epoch>1: raise RuntimeError("resume missing best checkpoint")
 for epoch in range(start_epoch,31):
  ds=TrainDataset(train,cohort,mean,std,arm,seed);ds.epoch=epoch
  # Fresh workers per epoch make the epoch key visible on Windows and POSIX.
  loader=DataLoader(ds,batch_size=64,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=False)
  model.train();losses=[];torch.cuda.reset_peak_memory_stats()
  for a,b in loader:
   opt.zero_grad(set_to_none=True)
   with torch.autocast("cuda",dtype=torch.float16): loss=lossfn(model(a.cuda(non_blocking=True)),b.cuda(non_blocking=True))
   if not torch.isfinite(loss):raise RuntimeError("nonfinite training loss")
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
  yv,pv,t,ok=val_gate(model,cohort,val,mean,std);ap=float(average_precision_score(yv[:,0],pv[:,0]));hist.append({"epoch":epoch,"mean_train_loss":float(np.mean(losses)),"validation_twa_auprc":ap,"validation_twa_auroc":float(roc_auc_score(yv[:,0],pv[:,0]))})
  if ap>best:
   best=ap;wait=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};atomic_torch(d/"best.pt",{"state_dict":best_state,"epoch":epoch,"validation_twa_auprc":best})
  else:wait+=1
  atomic_npz(d/"validation.npz",y=yv,p=pv,thresholds=t)
  payload={"cohort":cohort,"arm":arm,"seed":seed,"split_sha256":sha(split),"protocol_sha256":json.loads(PROTO.read_text())["protocol_sha256"],"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"history":hist,"best":best,"wait":wait,"epoch":epoch,"rng":prng_state()};atomic_torch(resume,payload)
  progress("TRAINING",cohort=cohort,arm=arm,seed=seed,epoch=epoch,best_validation_twa_auprc=best,elapsed_hours=(time.monotonic()-started)/3600,gpu_peak_bytes=torch.cuda.max_memory_allocated())
  if wait>=5:break
 if best_state is None:best_state=torch.load(d/"best.pt",weights_only=True,map_location="cpu")["state_dict"]
 model.load_state_dict(best_state);yv,pv,t,ok=val_gate(model,cohort,val,mean,std);best_epoch=int(torch.load(d/"best.pt",weights_only=True,map_location="cpu")["epoch"])
 atomic_torch(d/"checkpoint.pt",{"state_dict":best_state,"cohort":cohort,"arm":arm,"seed":seed,"best_epoch":best_epoch})
 atomic_json(gatefile,{"cohort":cohort,"arm":arm,"seed":seed,"decision":"PASS" if ok else "STOP_VALIDATION_GATE","best_epoch":best_epoch,"validation_twa_auprc":float(average_precision_score(yv[:,0],pv[:,0])),"validation_twa_auroc":float(roc_auc_score(yv[:,0],pv[:,0])),"finite_predictions":bool(np.isfinite(pv).all()),"checkpoint_sha256":sha(d/"checkpoint.pt"),"preprocessing_sha256":sha(prep),"split_sha256":sha(split),"stop_reason":"patience_5" if wait>=5 else "max_epochs_30","elapsed_hours":(time.monotonic()-started)/3600,"gpu_peak_bytes":int(torch.cuda.max_memory_allocated()),"history":hist,"test_accessed":False})
def cycle_views(x,slot):
 from outputs.heldout_limb_cycles_v1.cycle_adapter import transform
 return transform(x,slot).astype(np.float32)
def infer_unit(cohort,arm,seed):
 d=OUT/"training"/cohort/arm/f"seed_{seed}";g=json.loads((d/"validation_gate.json").read_text(encoding="utf-8"))
 if g["decision"]!="PASS":return
 rec,mean,std,split,prep=source_data(cohort,seed);test=rec["test"];x=raw_for(cohort,test);y=labels_for(cohort,test)
 model=ECGResNetSmall().cuda();model.load_state_dict(torch.load(d/"checkpoint.pt",weights_only=True,map_location="cuda")["state_dict"]);model.eval()
 root=OUT/"predictions"/cohort/arm/f"seed_{seed}";root.mkdir(parents=True,exist_ok=True)
 atomic_npz(root/"test_labels_order.npz",y=y,record_ids=np.asarray([str(r) for r in test]))
 conditions=[("clean",x)];conditions.append(("RA_LL_reversal",np.stack([derive_display(named_reversal(np.concatenate((a[:2],a[6:]),axis=0),"RA_LL_reversal")) for a in x]).astype(np.float32)))
 conditions.extend([("cycle_slots_120",cycle_views(x,"cycle_slots_120")),("cycle_slots_201",cycle_views(x,"cycle_slots_201"))])
 done=0
 for name,a in conditions:
  f=root/(name+".npz")
  if not f.exists():atomic_npz(f,p=inference(model,a,mean,std),y=y)
  done+=1
 for j,idx in enumerate(MASKS):
  name=MASK_NAMES[j];f=root/(name+".npz")
  if f.exists():done+=1;continue
  a=x.copy();a[:,np.asarray(idx),:]=mean[np.asarray(idx)][None,:,None]
  atomic_npz(f,p=inference(model,a,mean,std),y=y);done+=1
  if done%50==0:progress("INFERENCE",cohort=cohort,arm=arm,seed=seed,completed_conditions=done,total_conditions=len(CONDITIONS))
 atomic_json(d/"evaluation_complete.json",{"conditions":len(CONDITIONS),"mask_count":792,"checkpoint_sha256":sha(d/"checkpoint.pt"),"prediction_hashes":{f.name:sha(f) for f in root.glob("*.npz")},"protocol_sha256":json.loads(PROTO.read_text())["protocol_sha256"]})

def ece15(y,p):
 out=0.0
 for b in range(15):
  lo,hi=b/15,(b+1)/15;mask=(p>=lo)&((p<hi) if b<14 else (p<=hi))
  if mask.any():out+=float(mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean())))
 return out
def analyze_new_cohort(cohort):
 rows=[];mask_rows=[]
 for arm in ARMS:
  for seed in SEEDS:
   base=OUT/"predictions"/cohort/arm/f"seed_{seed}";val=np.load(OUT/"training"/cohort/arm/f"seed_{seed}"/"validation.npz",allow_pickle=False);thresholds=val["thresholds"]
   for condition in CONDITIONS:
    with np.load(base/(condition+".npz"),allow_pickle=False) as z:y=z["y"].astype(np.uint8);p=z["p"].astype(np.float64)
    for j,label in enumerate(LABELS):
     yt=y[:,j];pt=p[:,j];pred=pt>=thresholds[j]
     row={"cohort":cohort,"arm":arm,"seed":seed,"condition":condition,"endpoint":label,"n":len(yt),"auprc":float(average_precision_score(yt,pt)),"auroc":float(roc_auc_score(yt,pt)),"brier":float(np.mean((pt-yt)**2)),"ece15":ece15(yt,pt),"threshold":float(thresholds[j]),"threshold_source":"own clean validation negatives, 90th percentile","sensitivity":float(pred[yt==1].mean()),"specificity":float((~pred[yt==0]).mean()),"predicted_positive_fraction":float(pred.mean()),"accuracy":float((pred==yt).mean())}
     rows.append(row)
     if label=="TWA" and condition.startswith("mask5_"):mask_rows.append(row)
 atomic_csv(OUT/f"METRICS_{cohort.upper()}_NEW_BY_SEED.csv",rows)
 summary=[]
 for arm in ARMS:
  for cond in ("clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201","mask5_0_1_2_3_5"):
   for endpoint in LABELS:
    cur=[r for r in rows if r["arm"]==arm and r["condition"]==cond and r["endpoint"]==endpoint]
    summary.append({"cohort":cohort,"arm":arm,"condition":cond,"endpoint":endpoint,"n_seeds":len(cur),**{k+"_mean":float(np.mean([x[k] for x in cur])) for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","accuracy")},**{k+"_sd":float(np.std([x[k] for x in cur],ddof=1)) for k in ("auprc","auroc","brier","ece15","sensitivity","specificity","predicted_positive_fraction","accuracy")}})
 atomic_csv(OUT/f"METRICS_{cohort.upper()}_SUMMARY.csv",summary)
 masks=[]
 for arm in ARMS:
  for seed in SEEDS:
   vals=np.asarray([r["auprc"] for r in mask_rows if r["arm"]==arm and r["seed"]==seed])
   masks.append({"cohort":cohort,"arm":arm,"seed":seed,"n_masks":len(vals),"mean_auprc":float(vals.mean()),"median_auprc":float(np.median(vals)),"p05_auprc":float(np.quantile(vals,.05)),"p25_auprc":float(np.quantile(vals,.25)),"p75_auprc":float(np.quantile(vals,.75)),"p95_auprc":float(np.quantile(vals,.95)),"exact_mask_auprc":next(r["auprc"] for r in rows if r["arm"]==arm and r["seed"]==seed and r["condition"]=="mask5_0_1_2_3_5" and r["endpoint"]=="TWA")})
 atomic_csv(OUT/f"MASK_DISTRIBUTION_{cohort.upper()}.csv",masks)
 progress("COHORT_ANALYZED",cohort=cohort,metrics_rows=len(rows),mask_distributions=len(masks))

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 sources=["REMAINING_WORK_PLAN_20261018.md","configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json","src/mitigation_m1_v1.py","src/mitigation_m2_ningbo_v1.py","outputs/ningbo_four_arm_replication_v1/workflow.py","outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json","outputs/heldout_limb_cycles_v1/FROZEN_PROTOCOL.json","outputs/heldout_limb_cycles_v1/REVIEW_NOTES_20261018.md","outputs/heldout_limb_cycles_v1/cycle_adapter.py","outputs/mask_semantics_recovery_v2/RECOVERY_PROTOCOL.json","outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json","outputs/mixed_fault_calibration_v1/calibration_split_indices.npz","outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json","outputs/phase1_chapman_seed1_v4/preprocessing_v4.npz","outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json","outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json","outputs/ningbo_confirmation_v2/bootstrap_indices.npy","outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_001.json","outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_002.json","outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_003.json","outputs/augmentation_probability_study_v1/PROTOCOL_AMENDMENT_004.json","outputs/augmentation_probability_study_v1/engineering_checks.py"]
 bindings={p:sha(ROOT/p) for p in sources}
 spec={"version":"augmentation_probability_study_v1","status":"FROZEN_BEFORE_NEW_TEST_INFERENCE","created_utc":now(),"research_scope":"Exploratory augmentation-probability extension on previously accessed Chapman and Ningbo cohorts; not independent confirmation.","arms":{ "physics_p010":{"clean":0.90,"one_uniform_reversal":0.10},"physics_p025":{"clean":0.75,"one_uniform_reversal":0.25}},"reversal_family":list(FAULTS),"seeds":list(SEEDS),"dataset_order":["Chapman","Ningbo"],"model":"existing ECGResNet-small","training":{"optimizer":"AdamW lr=0.001 weight_decay=0.0001","loss":"weighted multilabel BCE; positive weights estimated on training labels only and capped at 20","batch_size":64,"max_epochs":30,"AMP":"CUDA float16 autocast with GradScaler","checkpoint":"maximum clean validation TWA AUPRC, ties retain earlier epoch","early_stopping":"patience 5","gate":"TWA validation AUROC >=0.75 and AUPRC >=0.45; finite scores and artifact integrity","normalization":"existing train-derived per-lead mean/std; raw transformation before normalization","masks_or_cycles_in_training":False,"data_loader":"two workers, recreated for each epoch so epoch-dependent deterministic transforms reach workers","resume":"epoch-boundary model, optimizer, scaler, Python/NumPy/Torch/CUDA RNG; committed epoch is the resume point"},"evaluation":{"conditions":["clean","RA_LL_reversal","cycle_slots_120","cycle_slots_201","all lexicographic C(12,5)=792 mean-imputed masks"],"exact_mask":[0,1,2,3,5],"primary_endpoint":"TWA","metrics":["AUPRC","AUROC","Brier","ECE15","sensitivity","specificity","predicted_positive_fraction"],"threshold":"per seed/endpoint 90th percentile of clean validation negative raw scores","comparisons":["p025 vs combined (25% reversal; 25% masks replace clean exposure)","p025 vs physics-aware 50%","p010 vs ordinary 0%","p010 vs physics-aware 50%"],"bootstrap":{"replicates":1000,"method":"existing cohort bootstrap indices; paired and synchronized","aggregation":"seed-specific metrics averaged within replicate before contrasts"},"Ningbo_mixtures":{"q_fault":[0,0.001,0.005,0.01,0.05,0.1,1],"fit_new_Platt_only_on_existing_validation_A":True,"new_thresholds_from_existing_validation_B":True,"no_test_tuning":True}},"exclusions":["Reserved_test not separately evaluated","HEEDB","new probabilities","hyperparameter search","manuscript edits"],"bindings":bindings,"runner_source_sha256":sha(Path(__file__))}
 spec["protocol_sha256"]=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
 atomic_json(PROTO,spec);(OUT/"PROTOCOL.md").write_text("# Frozen probability study protocol\n\nThis exploratory extension uses only p=0.10 and p=0.25, three fixed seeds, existing splits and train-derived normalization. The machine-readable protocol binds historical sources and this runner before new test inference. See `FROZEN_PROTOCOL.json` for the complete recipe and hashes. No manuscript, historical protocol, source or result is modified.\n",encoding="utf-8")
 print(spec["protocol_sha256"])
def train_all():
 if not PROTO.exists():raise RuntimeError("freeze protocol before training")
 p=json.loads(PROTO.read_text(encoding="utf-8"))
 if sha(Path(__file__))!=p["runner_source_sha256"]:raise RuntimeError("runner changed after protocol freeze")
 for cohort in ("Chapman","Ningbo"):
  for arm in ARMS:
   for seed in SEEDS:
    progress("UNIT_START",cohort=cohort,arm=arm,seed=seed);fit_unit(cohort,arm,seed)
  for arm in ARMS:
   for seed in SEEDS:infer_unit(cohort,arm,seed)
  analyze_new_cohort(cohort)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("freeze","run"));a=ap.parse_args()
 if a.command=="freeze":freeze();return
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 train_all()
if __name__=="__main__":main()
