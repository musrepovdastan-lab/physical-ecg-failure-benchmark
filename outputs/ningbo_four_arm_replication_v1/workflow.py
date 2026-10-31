"""Four-arm Ningbo replication: frozen recipes, training, inference and metrics."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, os, random, shutil, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ningbo_four_arm_replication_v1"
PROTO = OUT / "FROZEN_NINGBO_FOUR_ARM_PROTOCOL.json"
N = ROOT / "outputs/ningbo_confirmation_v2"
REC = ROOT / "outputs/mask_semantics_recovery_v2"
M2 = ROOT / "outputs/mitigation_m2_ningbo_v1"
M1_PROTO = ROOT / "configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json"
M2_PROTO = ROOT / "configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json"
SEEDS = (104729, 130363, 155921)
ARMS = ("masking", "combined")
ENDPOINTS = ("TWA", "SB", "STach")
FAULTS = ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
MASKS = tuple(itertools.combinations(range(12), 5))
MASK_NAMES = tuple("mask5_" + "_".join(map(str, x)) for x in MASKS)
CONDITIONS = ("clean", *FAULTS, *MASK_NAMES)
EXACT = (0, 1, 2, 3, 5)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import ningbo_confirmation_v1 as data
from src import mask_semantics_recovery_v2 as recovery
from src.chapman_phase1 import ECGResNetSmall
from src.ecg_physics import derive_display, named_reversal


def now(): return datetime.now(timezone.utc).isoformat()

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def hash_obj(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def atomic_torch(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".pt")
    torch.save(value, tmp); os.replace(tmp, path)

def atomic_csv(path: Path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

def status(stage: str, **values):
    z = {"stage": stage, "pid": os.getpid(), "updated_utc": now(), **values}
    atomic_json(OUT / "progress_resume.json", z)
    print(json.dumps(z, ensure_ascii=False, sort_keys=True), flush=True)

def deterministic_u(arm: str, seed: int, epoch: int, record_index: int, kind: str) -> float:
    key = f"{arm}|{seed}|{epoch}|{record_index}|{kind}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") / 2**64

def reverse_raw(x: np.ndarray, fault: str) -> np.ndarray:
    limb = np.concatenate((x[:2], x[6:]), axis=0)
    return derive_display(named_reversal(limb, fault)).astype(np.float32)

def augment_raw(x: np.ndarray, mean: np.ndarray, arm: str, seed: int, epoch: int, record_index: int) -> tuple[np.ndarray, str]:
    if arm not in ARMS: raise ValueError(f"unknown training arm: {arm}")
    u = deterministic_u(arm, seed, epoch, record_index, "choice")
    if u < .5: return x, "clean"
    if arm == "masking" or u < .75:
        j = min(791, int(deterministic_u(arm, seed, epoch, record_index, "mask") * 792))
        idx = MASKS[j]
        return recovery.apply_mask_raw(x, mean, idx).astype(np.float32), "mask5_" + "_".join(map(str, idx))
    j = min(2, int(deterministic_u(arm, seed, epoch, record_index, "fault") * 3))
    fault = FAULTS[j]
    return reverse_raw(x, fault), fault

class TrainingSet(Dataset):
    def __init__(self, records, mean, std, arm, seed):
        self.records, self.mean, self.std, self.arm, self.seed, self.epoch = records, mean, std, arm, seed, 1
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        x, _ = augment_raw(data.raw(r).astype(np.float32), self.mean, self.arm, self.seed, self.epoch, i)
        z = recovery.normalize_raw(x, self.mean, self.std)
        return torch.from_numpy(z), torch.tensor(r["labels"], dtype=torch.float32)

class PlainSet(Dataset):
    def __init__(self, x, y, mean, std): self.x,self.y,self.mean,self.std=x,y,mean,std
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return recovery.normalize_raw(self.x[i],self.mean,self.std), self.y[i]

def make_loader(ds, train=False):
    # Worker processes are recreated for each epoch so the updated deterministic epoch key is visible.
    return DataLoader(ds, batch_size=64 if train else 128, shuffle=train, num_workers=2,
                      pin_memory=True, persistent_workers=False)

def records_by_split():
    can=json.loads((N/"CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    split=json.loads((N/"SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by={r["record_id"]:r for r in can}
    out={s:[by[x["record_id"]] for x in split if x["split"]==s] for s in ("train","validation","test")}
    if tuple(map(len,(can,out["train"],out["validation"],out["test"]))) != (34899,24430,3488,6981):
        raise RuntimeError("canonical split counts mismatch")
    proof=json.loads((N/"SPLIT_LEAKAGE_PROOF.json").read_text(encoding="utf-8"))
    if proof.get("record_id_overlap") != 0 or proof.get("waveform_hash_overlap") != 0:
        raise RuntimeError("split leakage proof does not establish zero overlap")
    return out

def train_stats(train): return data.stats(train)

def load_frozen_train_statistics():
    """Reuse byte-identical statistics already derived from this exact frozen train split."""
    files=[N/f"seed_{s}"/"preprocessing.npz" for s in SEEDS]+[M2/f"seed_{s}"/"preprocessing.npz" for s in SEEDS]
    hashes={sha(f) for f in files}
    if len(hashes)!=1: raise RuntimeError("existing train-only preprocessing bindings disagree")
    with np.load(files[0],allow_pickle=False) as z: mean,std=z["mean"].astype(np.float32),z["std"].astype(np.float32)
    if mean.shape!=(12,) or std.shape!=(12,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise RuntimeError("frozen training statistics are malformed")
    return files[0],mean,std,next(iter(hashes))

def positive_weights(train_labels):
    y=np.asarray(train_labels,dtype=np.float32)
    if y.ndim!=2 or y.shape[1]!=3: raise ValueError("expected train-only (n,3) labels")
    return np.minimum((len(y)-y.sum(0))/np.maximum(y.sum(0),1),20).astype(np.float32)

def predict(model, x, y, mean, std):
    model.eval(); outs=[]
    with torch.no_grad():
        for a,_ in make_loader(PlainSet(x,y,mean,std)):
            with torch.autocast("cuda", dtype=torch.float16):
                outs.append(torch.sigmoid(model(a.cuda(non_blocking=True))).float().cpu().numpy())
    return np.concatenate(outs)

def validation_arrays(records):
    xs=np.stack([data.raw(r) for r in records]).astype(np.float32)
    ys=np.asarray([r["labels"] for r in records],dtype=np.uint8)
    return xs,ys

def unit_dir(arm,seed): return OUT/arm/f"seed_{seed}"

def protocol_check():
    if not PROTO.is_file(): raise RuntimeError("frozen protocol missing; run freeze before training")
    p=json.loads(PROTO.read_text(encoding="utf-8"))
    if p.get("status")!="FROZEN_BEFORE_TRAINING" or p.get("new_training_arms")!=list(ARMS) or p.get("seeds")!=list(SEEDS):
        raise RuntimeError("protocol identity mismatch")
    for path,digest in p["bindings"].items():
        if not (ROOT/path).is_file() or sha(ROOT/path)!=digest: raise RuntimeError(f"protocol-bound input changed: {path}")
    if sha(Path(__file__))!=p["workflow_source_sha256"]: raise RuntimeError("workflow source changed after freeze")
    return p

def validate_resume_payload(q, arm, seed, protocol, best_exists=True):
    if (q.get("protocol_sha256") != protocol["protocol_sha256"] or q.get("arm") != arm or
        q.get("seed") != seed or q.get("train_split_sha256") != protocol["bindings"]["outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"] or not best_exists):
        raise RuntimeError("resume binding mismatch; cross-arm/protocol resume rejected")
    return True

def restore_best(model, state_dict):
    model.load_state_dict(state_dict)
    return model

def freeze_protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    files=["RESEARCH_PLAN_AFTER_MENTOR_20261016.md","configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json",
           "configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json","src/mitigation_m1_v1.py","src/mitigation_m2_ningbo_v1.py",
           "src/chapman_phase1.py","src/ecg_physics.py","src/ningbo_confirmation_v1.py","src/mask_semantics_recovery_v2.py",
           "outputs/mask_semantics_recovery_v2/RECOVERY_PROTOCOL.json","outputs/mask_semantics_recovery_v2/FINAL_RECOVERY_REPORT.md",
           "outputs/architecture_replication_m3_recovery_v2/CHAPMAN_REPORT.md","outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_PROTOCOL.json",
           "outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json","outputs/mixed_fault_calibration_v2/FINAL_FOUR_METHOD_COMPARISON.md",
           "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json","outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json",
           "outputs/ningbo_confirmation_v2/SPLIT_LEAKAGE_PROOF.json","outputs/ningbo_confirmation_v2/bootstrap_indices.npy",
           "outputs/mixed_fault_calibration_v1/calibration_split_indices.npz"]
    bindings={f:sha(ROOT/f) for f in files}
    sources={"mask_recovery":json.loads((REC/"RECOVERY_PROTOCOL.json").read_text(encoding="utf-8")),
             "m1":json.loads(M1_PROTO.read_text(encoding="utf-8")),"m2":json.loads(M2_PROTO.read_text(encoding="utf-8"))}
    models={}
    for arm in ("clean","physics_aware"):
        for seed in SEEDS:
            d=(N/f"seed_{seed}") if arm=="clean" else (M2/f"seed_{seed}")
            recroot=REC/("baseline_v2" if arm=="clean" else "physics_aware_m2")/f"seed_{seed}"
            models[f"{arm}/{seed}"]={"checkpoint":sha(d/"checkpoint.pt"),"preprocessing":sha(d/"preprocessing.npz"),
                                      "validation":sha(d/"validation.npz"),"recovery_evaluation":sha(recroot/"evaluation_complete.json")}
    boot=np.load(N/"bootstrap_indices.npy",allow_pickle=False)
    if boot.shape!=(1000,6981): raise RuntimeError(f"unexpected bootstrap shape {boot.shape}")
    cfg={"version":"ningbo_four_arm_replication_v1","status":"FROZEN_BEFORE_TRAINING","created_utc":now(),
      "research_question":"Does masking and combined training replicate on Ningbo, and how do they change clean-performance, lead-reversal robustness, and missing-lead robustness tradeoffs?",
      "scope":"Ningbo follow-up on an already accessed test cohort; exploratory reanalysis, not untouched confirmation.",
      "arms":["clean","masking","physics_aware","combined"],"new_training_arms":list(ARMS),"seeds":list(SEEDS),
      "arms_recipe":{"clean":{"clean":1.0,"existing_checkpoint_only":True},"masking":{"clean":0.5,"five_lead_mask":0.5},
       "physics_aware":{"clean":0.5,"one_uniform_physical_reversal":0.5,"existing_checkpoint_only":True},
       "combined":{"clean":0.5,"five_lead_mask":0.25,"one_uniform_physical_reversal":0.25,"never_combine":True}},
      "masking":{"sample_space":"uniform over lexicographic C(12,5)=792; indices are removed channels","sentinel":"training split raw per-lead mean","replacement_before_normalization":True},
      "model_training":{"architecture":"ECGResNet-small","optimizer":"AdamW lr=0.001 weight_decay=0.0001","loss":"weighted multilabel BCE; train-only positive weights capped at 20","batch_size":64,"inference_batch_size":128,"max_epochs":30,"mixed_precision":"CUDA float16 autocast and GradScaler","early_stop":"validation TWA AUPRC only; patience 5","best_checkpoint":"highest validation TWA AUPRC; ties retain earlier epoch","validation_gate":{"TWA AUROC minimum":0.75,"TWA AUPRC minimum":0.45,"finite loss/predictions":True,"checkpoint/hash/split/protocol integrity":True,"test inference blocked on failure":True}},
      "preprocessing":"training split per displayed lead mean/std; augmentation in raw units then mean/std normalization",
      "conditions":["clean",*FAULTS,"all_792_removed_five_lead_masks"],"exact_mask":{"name":"I,II,III,aVR,aVF","indices":list(EXACT)},
      "metrics":["AUPRC","AUROC","Brier","ECE15","sensitivity","specificity","predicted_positive_fraction"],"threshold":"per seed/endpoint 90th percentile of negative validation raw scores (linear quantile)",
      "primary_contrasts":["masking-clean on clean TWA, RA_LL TWA, mean-mask TWA","combined-physics_aware on clean TWA, RA_LL TWA, mean-mask TWA"],
      "bootstrap":{"source":"outputs/ningbo_confirmation_v2/bootstrap_indices.npy","replicates":1000,"unit":"canonical record","synchronization":"same record indices across arms/seeds/conditions/endpoints","point_estimates":"full cohort; report bootstrap mean separately","mask_panel":"AP per mask then mean; no prediction vector averaging"},
      "mixtures":{"fault":"RA_LL_reversal","q":[0,0.001,0.005,0.01,0.05,0.1,1.0],"calibration":"same frozen validation A/B positions and v1 Platt recipe; A clean validation fit, B negative clean validation 90th percentile threshold","test_fitting":False},
      "reuse_policy":"corrected recovery v2 only; require source completion manifest and checkpoint/protocol/preprocessing/order/label/condition bindings; old inverted mask caches prohibited",
      "conditions_source":"src/mask_semantics_recovery_v2.py; corrected cache contract",
      "bindings":bindings,"existing_model_bindings":models,"source_protocols":{k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in sources.items()},
      "bootstrap_indices_shape":list(boot.shape),"bootstrap_indices_sha256":bindings["outputs/ningbo_confirmation_v2/bootstrap_indices.npy"],
      "workflow_source_sha256":sha(Path(__file__)),"evaluator_source_sha256":bindings["src/mask_semantics_recovery_v2.py"],
      "recovery_binding_sha256":sources["mask_recovery"]["binding_sha256"],
      "limitations":["Ningbo test has prior access","combined has half the reversal exposure of physics-aware; contrast does not isolate the effect of adding masks","M3 Chapman frozen gate FAIL retained","do not infer channel-use mechanism"]}
    cfg["protocol_sha256"]=hash_obj(cfg)
    atomic_json(PROTO,cfg)
    atomic_json(OUT/"RUN_MANIFEST.json",{"protocol_sha256":cfg["protocol_sha256"],"workflow_source_sha256":cfg["workflow_source_sha256"],"frozen_utc":cfg["created_utc"],"old_results_preserved":True,"new_output_root":OUT.relative_to(ROOT).as_posix()})
    return cfg

def ece15(y,p):
    edges=np.linspace(0,1,16);v=0.0
    for i in range(15):
        m=(p>=edges[i])&((p<edges[i+1]) if i<14 else (p<=edges[i+1]))
        if m.any(): v+=m.mean()*abs(float(y[m].mean()-p[m].mean()))
    return float(v)

def metric(y,p,t):
    c=p>=t;pos=y==1;neg=~pos
    return {"auprc":float(average_precision_score(y,p)),"auroc":float(roc_auc_score(y,p)),"brier":float(np.mean((p-y)**2)),
       "ece15":ece15(y,p),"sensitivity":float(c[pos].mean()) if pos.any() else float("nan"),"specificity":float((~c[neg]).mean()) if neg.any() else float("nan"),
       "predicted_positive_fraction":float(c.mean()),"threshold":float(t),"n":int(len(y))}

def train_one(arm,seed,parts):
    protocol=protocol_check(); d=unit_dir(arm,seed);d.mkdir(parents=True,exist_ok=True)
    gate=d/"validation_gate.json"
    if gate.exists() and json.loads(gate.read_text(encoding="utf-8")).get("decision") in ("PASS","STOP_VALIDATION_GATE"):
        return json.loads(gate.read_text(encoding="utf-8"))
    rng_seed=seed;random.seed(rng_seed);np.random.seed(rng_seed);torch.manual_seed(rng_seed);torch.cuda.manual_seed_all(rng_seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    stats_source,mean,std,stats_hash=load_frozen_train_statistics()
    shutil.copy2(stats_source,d/"preprocessing.npz")
    atomic_json(d/"preprocessing_provenance.json",{"source":stats_source.relative_to(ROOT).as_posix(),"source_sha256":stats_hash,
      "training_split_sha256":protocol["bindings"]["outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"],
      "derivation":"existing clean/physics checkpoints share byte-identical train-only per-displayed-lead statistics; no validation/test data used"})
    yt=np.asarray([r["labels"] for r in parts["train"]],np.float32)
    w=positive_weights(yt)
    model=ECGResNetSmall().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device="cuda"));scaler=torch.amp.GradScaler("cuda")
    resume=d/"resume.pt";bestfile=d/"best.pt"; hist=[];best=-1.;wait=0;start=1
    if resume.exists():
        q=torch.load(resume,weights_only=False,map_location="cuda")
        validate_resume_payload(q,arm,seed,protocol,bestfile.exists())
        model.load_state_dict(q["model"]);opt.load_state_dict(q["optimizer"]);scaler.load_state_dict(q["scaler"])
        hist,best,wait,start=q["history"],q["best"],q["wait"],q["epoch"]+1
        if wait>=5:
            best_state=torch.load(bestfile,weights_only=True,map_location="cpu")["state_dict"]
            return finish_validation(arm,seed,parts,mean,std,hist,best_state)
    ds=TrainingSet(parts["train"],mean,std,arm,seed);starttime=time.monotonic()
    for epoch in range(start,31):
        ds.epoch=epoch;model.train();losses=[]
        for a,y in make_loader(ds,True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): loss=lossfn(model(a.cuda(non_blocking=True)),y.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
        xv,yv=validation_arrays(parts["validation"]);pval=predict(model,xv,yv,mean,std)
        ap=float(average_precision_score(yv[:,0],pval[:,0]));auc=float(roc_auc_score(yv[:,0],pval[:,0]))
        hist.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_twa_auprc":ap,"validation_twa_auroc":auc})
        if ap>best:
            best=ap;wait=0
            atomic_torch(bestfile,{"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"epoch":epoch,"validation_twa_auprc":ap,"arm":arm,"seed":seed,"protocol_sha256":protocol["protocol_sha256"]})
        else: wait+=1
        atomic_torch(resume,{"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"best":best,"wait":wait,"history":hist,"epoch":epoch,
           "protocol_sha256":protocol["protocol_sha256"],"arm":arm,"seed":seed,"train_split_sha256":protocol["bindings"]["outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"]})
        status("TRAINING",arm=arm,seed=seed,epoch=epoch,best_validation_twa_auprc=best,elapsed_seconds=time.monotonic()-starttime,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if wait>=5:break
    best_state=torch.load(bestfile,weights_only=True,map_location="cpu")["state_dict"]
    return finish_validation(arm,seed,parts,mean,std,hist,best_state)

def finish_validation(arm,seed,parts,mean,std,hist,best_state):
    d=unit_dir(arm,seed);model=ECGResNetSmall().cuda();model.load_state_dict(best_state);x,y=validation_arrays(parts["validation"]);p=predict(model,x,y,mean,std)
    thresholds=np.asarray([np.quantile(p[y[:,j]==0,j],.9,method="linear") for j in range(3)],np.float32)
    auc=float(roc_auc_score(y[:,0],p[:,0]));ap=float(average_precision_score(y[:,0],p[:,0]));ok=bool(np.isfinite(p).all() and auc>=.75 and ap>=.45 and all(np.isfinite(h["train_loss"]) for h in hist))
    best=torch.load(d/"best.pt",weights_only=True,map_location="cpu")
    atomic_torch(d/"checkpoint.pt",{"state_dict":best_state,"arm":arm,"seed":seed,"protocol_sha256":json.loads(PROTO.read_text())["protocol_sha256"],"best_epoch":best["epoch"]})
    atomic_npz(d/"validation.npz",y=y,p=p,thresholds=thresholds)
    atomic_json(d/"history.json",{"history":hist,"best_epoch":best["epoch"],"best_validation_twa_auprc":best["validation_twa_auprc"]})
    val={"decision":"PASS" if ok else "STOP_VALIDATION_GATE","arm":arm,"seed":seed,"best_epoch":best["epoch"],"validation_twa_auprc":ap,"validation_twa_auroc":auc,
      "finite_losses_predictions":bool(np.isfinite(p).all() and all(np.isfinite(h["train_loss"]) for h in hist)),"thresholds":thresholds.tolist(),"checkpoint_sha256":sha(d/"checkpoint.pt"),
      "protocol_sha256":json.loads(PROTO.read_text())["protocol_sha256"],"split_sha256":sha(N/"SPLIT_MEMBERSHIP.json"),"test_accessed":False,"history_length":len(hist),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated()}
    atomic_json(d/"validation_gate.json",val);return val

def infer_unit(arm,seed,parts):
    d=unit_dir(arm,seed);g=json.loads((d/"validation_gate.json").read_text(encoding="utf-8"))
    if g["decision"]!="PASS": return None
    done=d/"evaluation_complete.json";pd=d/"test_predictions";pd.mkdir(exist_ok=True)
    if done.exists():
        e=json.loads(done.read_text(encoding="utf-8"))
        if e.get("protocol_sha256")==g["protocol_sha256"] and len(e.get("predictions",{}))==len(CONDITIONS) and all((pd/f"{c}.npz").is_file() and sha(pd/f"{c}.npz")==e["predictions"][c] for c in CONDITIONS):return e
        raise RuntimeError(f"completed predictions fail integrity for {arm}/{seed}")
    mean,std=np.load(d/"preprocessing.npz")["mean"],np.load(d/"preprocessing.npz")["std"]
    model=ECGResNetSmall().cuda();ck=torch.load(d/"checkpoint.pt",weights_only=True,map_location="cuda")
    if ck["arm"]!=arm or ck["seed"]!=seed or ck["protocol_sha256"]!=g["protocol_sha256"]:raise RuntimeError("checkpoint arm/protocol mismatch")
    model.load_state_dict(ck["state_dict"]);xt,yt=validation_arrays(parts["test"])
    atomic_json(d/"evaluation_freeze.json",{"protocol_sha256":g["protocol_sha256"],"checkpoint_sha256":sha(d/"checkpoint.pt"),"test_membership_sha256":sha(N/"SPLIT_MEMBERSHIP.json"),"conditions":len(CONDITIONS),"test_accessed_before_freeze":False})
    manifest={}
    for i,c in enumerate(CONDITIONS):
        f=pd/f"{c}.npz"
        if c=="clean": a=xt
        elif c in FAULTS: a=np.stack([reverse_raw(x,c) for x in xt])
        else: a=np.stack([recovery.apply_mask_raw(x,mean,MASKS[i-4]) for x in xt])
        p=predict(model,a,yt,mean,std);atomic_npz(f,p=p,y=yt)
        manifest[c]=sha(f);status("INFERENCE",arm=arm,seed=seed,condition=c,completed_conditions=i+1,total_conditions=len(CONDITIONS))
    e={"arm":arm,"seed":seed,"protocol_sha256":g["protocol_sha256"],"checkpoint_sha256":sha(d/"checkpoint.pt"),"conditions":len(CONDITIONS),"predictions":manifest}
    atomic_json(done,e);return e

def existing_prediction(arm,seed,condition):
    mode="baseline_v2" if arm=="clean" else "physics_aware_m2"
    return REC/mode/f"seed_{seed}"/"test_predictions"/f"{condition}.npz"

def verify_existing_reuse():
    p=protocol_check();out=[]
    for arm in ("clean","physics_aware"):
      for seed in SEEDS:
        d=N/f"seed_{seed}" if arm=="clean" else M2/f"seed_{seed}"
        recroot=REC/("baseline_v2" if arm=="clean" else "physics_aware_m2")/f"seed_{seed}"
        e=json.loads((recroot/"evaluation_complete.json").read_text(encoding="utf-8"))
        v=np.load(d/"validation.npz",allow_pickle=False);m=np.load(d/"preprocessing.npz",allow_pickle=False)
        exp=p["existing_model_bindings"][f"{arm}/{seed}"]
        if sha(d/"checkpoint.pt")!=exp["checkpoint"] or sha(d/"validation.npz")!=exp["validation"] or sha(d/"preprocessing.npz")!=exp["preprocessing"]:raise RuntimeError("existing model binding changed")
        if e.get("decision")!="PASS" or len(e.get("prediction_sha256",{}))!=794:raise RuntimeError("recovery completion manifest incomplete")
        for cond,digest in e["prediction_sha256"].items():
            f=recroot/"test_predictions"/f"{cond}.npz"
            if not f.is_file() or sha(f).lower()!=digest.lower():raise RuntimeError(f"corrected cache hash mismatch {f}")
        # Corrected recovery manifest is tied to the immutable Ningbo split, source checkpoint and normalization.
        out.append({"arm":arm,"seed":seed,"checkpoint_sha256":sha(d/"checkpoint.pt"),"recovery_manifest_sha256":sha(recroot/"evaluation_complete.json"),"reused_conditions":794})
    atomic_json(OUT/"EXISTING_COMPARATOR_REUSE_AUDIT.json",{"decision":"PASS","models":out,"old_inverted_mask_cache_reused":False})
    return out

def condition_metric_rows(parts):
    ys=np.asarray([r["labels"] for r in parts["test"]],np.uint8);rows=[]
    refs={}
    for arm in ("clean","masking","physics_aware","combined"):
      for seed in SEEDS:
        if arm in ARMS:
            d=unit_dir(arm,seed);g=json.loads((d/"validation_gate.json").read_text(encoding="utf-8"))
            if g["decision"]!="PASS": continue
            predroot=d/"test_predictions";val=np.load(d/"validation.npz",allow_pickle=False);th=val["thresholds"]
        else:
            source=N/f"seed_{seed}" if arm=="clean" else M2/f"seed_{seed}"
            val=np.load(source/"validation.npz",allow_pickle=False);th=val["thresholds"]
            predroot=REC/("baseline_v2" if arm=="clean" else "physics_aware_m2")/f"seed_{seed}"/"test_predictions"
        for cond in CONDITIONS:
            f=predroot/f"{cond}.npz"
            if not f.exists(): continue
            with np.load(f,allow_pickle=False) as z: p=z["p"].astype(np.float32)
            for j,endpoint in enumerate(ENDPOINTS):
                met=metric(ys[:,j],p[:,j],float(th[j]));rows.append({"arm":arm,"seed":seed,"condition":cond,"endpoint":endpoint,**met})
            refs[arm,seed,cond]=p
    atomic_csv(OUT/"RESULTS_BY_ARM_SEED.csv",rows)
    return rows,refs,ys

def mean_mask_ap(y, pred, label_index=0):
    return float(np.mean([average_precision_score(y[:,label_index],pred[n][:,label_index]) for n in MASK_NAMES]))

def analyze(parts):
    rows,refs,ys=condition_metric_rows(parts);atomic_json(OUT/"_analysis_sources.json",{"created_utc":now(),"result_rows":len(rows)})
    # Exact-mask and descriptive 792-mask distribution from per-mask metric calculations.
    exact=[];dist=[]
    for arm in ("clean","masking","physics_aware","combined"):
      for seed in SEEDS:
        if (arm,seed,"mask5_0_1_2_3_5") not in refs: continue
        p=refs[arm,seed,"mask5_0_1_2_3_5"]
        for j,ep in enumerate(ENDPOINTS): exact.append({"arm":arm,"seed":seed,"endpoint":ep,"mask":"I,II,III,aVR,aVF","removed_indices":"0,1,2,3,5","auprc":average_precision_score(ys[:,j],p[:,j])})
        vals=np.asarray([average_precision_score(ys[:,0],refs[arm,seed,n][:,0]) for n in MASK_NAMES])
        dist.append({"arm":arm,"seed":seed,"endpoint":"TWA","mask_count":len(vals),"mean_auprc":float(vals.mean()),"median_auprc":float(np.median(vals)),"sd_auprc":float(vals.std(ddof=1)),"p05":float(np.quantile(vals,.05)),"p25":float(np.quantile(vals,.25)),"p75":float(np.quantile(vals,.75)),"p95":float(np.quantile(vals,.95)),"min":float(vals.min()),"max":float(vals.max()),"exact_mask_auprc":float(vals[MASKS.index(EXACT)])})
    atomic_csv(OUT/"EXACT_MASK_COMPARISON.csv",exact);atomic_csv(OUT/"MASK_DISTRIBUTION_SUMMARY.csv",dist)
    # Full-cohort primary AUPRC contrasts and synchronized bootstrap.
    idx=np.load(N/"bootstrap_indices.npy",allow_pickle=False);boot_rows=[];contrast_rows=[]
    for ep_idx,ep in enumerate(ENDPOINTS):
      for metric_name in ("clean","RA_LL_reversal","mean_mask"):
       pairs=(("masking","clean"),("combined","physics_aware"))
       for arm_a,arm_b in pairs:
        diffs=[]
        for seed in SEEDS:
            if metric_name=="mean_mask":
                if (arm_a,seed,MASK_NAMES[0]) not in refs or (arm_b,seed,MASK_NAMES[0]) not in refs: continue
                a=np.mean([average_precision_score(ys[:,ep_idx],refs[arm_a,seed,n][:,ep_idx]) for n in MASK_NAMES]);b=np.mean([average_precision_score(ys[:,ep_idx],refs[arm_b,seed,n][:,ep_idx]) for n in MASK_NAMES])
            else:
                cond="clean" if metric_name=="clean" else "RA_LL_reversal"
                if (arm_a,seed,cond) not in refs or (arm_b,seed,cond) not in refs: continue
                a=average_precision_score(ys[:,ep_idx],refs[arm_a,seed,cond][:,ep_idx]);b=average_precision_score(ys[:,ep_idx],refs[arm_b,seed,cond][:,ep_idx])
            diffs.append(float(a-b))
        point=float(np.mean(diffs)) if diffs else float("nan")
        # Draw each seed's metric on the shared canonical indices; for masks AP is computed per mask before panel averaging.
        draws=[]
        if len(diffs)==3:
         for r,ix in enumerate(idx):
          seed_d=[]
          for seed in SEEDS:
            if metric_name=="mean_mask":
                aa=np.mean([average_precision_score(ys[ix,ep_idx],refs[arm_a,seed,n][ix,ep_idx]) for n in MASK_NAMES]);bb=np.mean([average_precision_score(ys[ix,ep_idx],refs[arm_b,seed,n][ix,ep_idx]) for n in MASK_NAMES])
            else:
                cond="clean" if metric_name=="clean" else "RA_LL_reversal"
                aa=average_precision_score(ys[ix,ep_idx],refs[arm_a,seed,cond][ix,ep_idx]);bb=average_precision_score(ys[ix,ep_idx],refs[arm_b,seed,cond][ix,ep_idx])
            seed_d.append(aa-bb)
          draws.append(float(np.mean(seed_d)))
        if draws:
            boot_rows.append({"contrast":f"{arm_a}_minus_{arm_b}","endpoint":ep,"condition_estimand":metric_name,"point_estimate_full_cohort":point,"bootstrap_mean":float(np.mean(draws),),"lower95":float(np.quantile(draws,.025)),"upper95":float(np.quantile(draws,.975)),"replicates":len(draws),"resampling":"shared Ningbo canonical-record indices"})
        contrast_rows.append({"contrast":f"{arm_a}_minus_{arm_b}","endpoint":ep,"condition_estimand":metric_name,"point_estimate_full_cohort":point,"seed_differences":json.dumps(diffs)})
    atomic_csv(OUT/"PRIMARY_CONTRASTS.csv",contrast_rows);atomic_csv(OUT/"BOOTSTRAP_SUMMARY.csv",boot_rows)
    return rows,refs,ys,dist,boot_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("freeze","check","train","infer","analyze","all"));args=ap.parse_args()
    if args.command=="freeze": print(json.dumps(freeze_protocol(),indent=2));return
    protocol_check()
    if args.command=="check": print("PROTOCOL_BINDINGS_PASS");return
    parts=records_by_split()
    if args.command in ("train","all"):
        for arm in ARMS:
          for seed in SEEDS:
            status("UNIT_START",arm=arm,seed=seed);g=train_one(arm,seed,parts)
            print(json.dumps(g,sort_keys=True),flush=True)
    if args.command in ("infer","all"):
        for arm in ARMS:
          for seed in SEEDS: infer_unit(arm,seed,parts)
    if args.command in ("analyze","all"): analyze(parts)

if __name__=="__main__": main()
