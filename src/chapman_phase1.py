"""Frozen one-seed Chapman Phase-1 training and physical-fault evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, random, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.ecg_physics import derive_display, named_reversal
from src.chapman_stage23 import DATA, OUT, LABELS, canonical_json, parse_header, sha256_file

ROOT = Path(__file__).resolve().parents[1]; DEVICE = torch.device("cuda")
SEED = 104729; LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def dump(name, x): (OUT/name).write_text(json.dumps(x, indent=2, sort_keys=True), encoding="utf-8")
def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def labels(record): return np.array([int(x in parse_header(DATA/f"{record}.hea")[1]) for x in LABELS], np.float32)
def raw(record): return loadmat(DATA/f"{record}.mat")["val"].astype("float32")[:,::5]
def hash_path(path): return sha256_file(path)

class Block(nn.Module):
    def __init__(self,a,b,s=1):
        super().__init__(); self.a=nn.Sequential(nn.Conv1d(a,b,7,s,3,bias=False),nn.GroupNorm(8,b),nn.GELU(),nn.Conv1d(b,b,7,1,3,bias=False),nn.GroupNorm(8,b)); self.z=nn.Identity() if a==b and s==1 else nn.Sequential(nn.Conv1d(a,b,1,s,bias=False),nn.GroupNorm(8,b)); self.g=nn.GELU()
    def forward(self,x): return self.g(self.a(x)+self.z(x))
class ECGResNetSmall(nn.Module):
    def __init__(self):
        super().__init__(); w=[32,64,128,256]; self.s=nn.Sequential(nn.Conv1d(12,w[0],7,1,3,bias=False),nn.GroupNorm(8,w[0]),nn.GELU()); self.b=nn.ModuleList([Block(w[i-1] if i else w[0],q,1 if i==0 else 2) for i,q in enumerate(w)]); self.h=nn.Sequential(nn.Dropout(.2),nn.Linear(w[-1],3))
    def forward(self,x):
        x=self.s(x)
        for b in self.b: x=b(x)
        return self.h(x.mean(-1))

class ECG(Dataset):
    def __init__(self, ids, mean, std): self.ids,self.mean,self.std=ids,mean,std
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        r=self.ids[i]; return ((raw(r)-self.mean[:,None])/self.std[:,None]).astype("float32"), labels(r), r
def split():
    x=json.loads((OUT/"SPLIT_MEMBERSHIP.json").read_text()); return {s:[r["record_id"] for r in x["rows"] if r["split"]==s] for s in ("train","validation","test")}
def stats(ids):
    a=np.zeros(12,np.float64); b=a.copy(); n=0
    for k,r in enumerate(ids,1):
        x=raw(r); a+=x.sum(1); b+=(x*x).sum(1); n+=x.shape[1]
        if k%500==0: print(f"statistics {k}/{len(ids)}",flush=True)
    m=a/n; return m.astype("float32"),np.sqrt(np.maximum(b/n-m*m,1e-8)).astype("float32")
def loader(ids,m,s,shuffle=False): return DataLoader(ECG(ids,m,s),batch_size=64 if shuffle else 128,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True)
def pred(model, ids, m, s, transform=None):
    ps=[]; ys=[]; names=[]; model.eval()
    with torch.no_grad():
        for x,y,r in loader(ids,m,s):
            if transform: x=torch.from_numpy(transform((x.numpy()*s[None,:,None]+m[None,:,None]).astype("float32"),m,s))
            with torch.autocast("cuda",dtype=torch.float16): p=torch.sigmoid(model(x.cuda(non_blocking=True))).float().cpu().numpy()
            ps.append(p); ys.append(y.numpy()); names.extend(r)
    return np.concatenate(ys),np.concatenate(ps),names
def ap(y,p): return np.array([average_precision_score(y[:,i],p[:,i]) for i in range(3)],float)
def au(y,p): return np.array([roc_auc_score(y[:,i],p[:,i]) for i in range(3)],float)
def thresholds(y,p): return np.array([np.quantile(p[y[:,i]==0,i],.90) for i in range(3)])
def rows(name,y,p,t):
    out=[]
    for i,label in enumerate(("TWA","SB","STach")):
        z=p[:,i]>=t[i]; neg=y[:,i]==0; out.append({"condition":name,"label":label,"auprc":ap(y,p)[i],"auroc":au(y,p)[i],"sensitivity_at_frozen_90pct_specificity":float(z[y[:,i]==1].mean()),"specificity":float((~z[neg]).mean())})
    return out
def write_csv(name,rs):
    if not rs: return
    with (OUT/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rs[0]); w.writeheader(); w.writerows(rs)

def train():
    audit=json.loads((OUT/"DATA_INGESTION_AUDIT.json").read_text())
    if audit.get("decision")!="PASS": raise RuntimeError("integrity gate has not passed")
    parts=split(); assert not set(parts["train"])&set(parts["validation"])
    seed(); m,s=stats(parts["train"]); np.savez(OUT/"preprocessing.npz",mean=m,std=s)
    model=ECGResNetSmall().cuda(); yy=np.stack([labels(r) for r in parts["train"]]); pw=np.minimum((len(yy)-yy.sum(0))/np.maximum(yy.sum(0),1),20)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=DEVICE)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda")
    best=-1.; wait=0; state=None; history=[]; started=time.monotonic()
    for epoch in range(1,31):
        model.train(); losses=[]
        for x,y,_ in loader(parts["train"],m,s,True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): loss=lossfn(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        y,p,_=pred(model,parts["validation"],m,s); score=ap(y,p)[0]; history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_twa_auprc":float(score)})
        print(f"epoch {epoch}: TWA val AUPRC={score:.4f}",flush=True)
        if score>best: best=score; wait=0; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait>=5: break
    model.load_state_dict(state); y,p,_=pred(model,parts["validation"],m,s); t=thresholds(y,p); elapsed=(time.monotonic()-started)/3600
    ckpt={"state_dict":model.state_dict(),"seed":SEED,"epochs":len(history),"labels":LABELS}; torch.save(ckpt,OUT/"ecgresnet_small_seed104729.pt")
    adequate=bool(np.isfinite(p).all() and np.isfinite([x["train_loss"] for x in history]).all() and au(y,p)[0]>=.75 and ap(y,p)[0]>=.45 and elapsed<=8)
    result={"decision":"PASS" if adequate else "STOP_BASELINE_INADEQUATE","validation_twa_auroc":float(au(y,p)[0]),"validation_twa_auprc":float(ap(y,p)[0]),"thresholds_at_90pct_specificity":t.tolist(),"training_gpu_hours":elapsed,"history":history,"train_validation_disjoint":True,"test_accessed":False}
    dump("BASELINE_ADEQUACY.json",result); np.savez_compressed(OUT/"validation_predictions.npz",y=y,p=p,thresholds=t)
    if not adequate: dump("conclusions_v3.json",result); (OUT/"FINAL_PHASE1_PILOT_DECISION.md").write_text("# STOP_BASELINE_INADEQUATE\n",encoding="utf-8"); return
    dump("FAILURE_ANALYSIS_FREEZE.json",{"checkpoint_sha256":hash_path(OUT/"ecgresnet_small_seed104729.pt"),"preprocessing_sha256":hash_path(OUT/"preprocessing.npz"),"fault_library_sha256":hash_path(ROOT/"src/ecg_physics.py"),"thresholds":t.tolist(),"split_sha256":json.loads((OUT/"SPLIT_MANIFEST.json").read_text())["membership_sha256"]})

def availability(x,m,s,unavailable=(),reversal=None,display_mask=()):
    ind=np.concatenate((x[:,0:2],x[:,6:]),axis=1)
    if reversal: ind=named_reversal(ind,reversal)
    d=derive_display(ind).astype("float32")
    unavailable=set(unavailable)
    if unavailable & {0,1}: d[:,:6]=m[:6,None]
    for q in unavailable-{0,1}: d[:,6+q-2]=m[6+q-2,None]
    for q in display_mask: d[:,q]=m[q,None]
    return ((d-m[None,:,None])/s[None,:,None]).astype("float32")
def bootstrap(y, allp, primary, others, reps=1000):
    rng=np.random.default_rng(SEED); out=[]; n=len(y)
    for b in range(reps):
        ix=rng.integers(0,n,n); base=ap(y[ix],allp["clean"][ix]); ref=np.mean([ap(y[ix],allp[k][ix]) for k in others],0); v=ap(y[ix],allp[primary][ix]);
        for j,label in enumerate(("TWA","SB","STach")): out.append({"replicate":b,"label":label,"v1_v4_minus_mean_other69_auprc_degradation":float((base[j]-v[j])-(base[j]-ref[j]))})
    return out
def evaluate():
    gate=json.loads((OUT/"BASELINE_ADEQUACY.json").read_text())
    if gate.get("decision")!="PASS": raise RuntimeError("STOP_BASELINE_INADEQUATE; test remains sealed")
    parts=split(); z=np.load(OUT/"preprocessing.npz"); m,s=z["mean"],z["std"]; model=ECGResNetSmall().cuda(); model.load_state_dict(torch.load(OUT/"ecgresnet_small_seed104729.pt",weights_only=True)["state_dict"]); t=np.load(OUT/"validation_predictions.npz")["thresholds"]
    y,clean,_=pred(model,parts["test"],m,s); allp={"clean":clean}; result=rows("clean",y,clean,t)
    named={f"V{i}":(i+1,) for i in range(1,7)}; named["V1_V4"]=(2,3,4,5)
    for name,mask in named.items(): _,p,_=pred(model,parts["test"],m,s,lambda x,a,b,mask=mask:availability(x,a,b,mask)); allp[name]=p; result+=rows(name,y,p,t)
    subsets=list(itertools.combinations(range(8),4)); names=[]
    for mask in subsets:
        name="subset_"+"_".join(map(str,mask)); _,p,_=pred(model,parts["test"],m,s,lambda x,a,b,mask=mask:availability(x,a,b,mask)); allp[name]=p; names.append(name); result+=rows(name,y,p,t)
    display=[]
    for mask in itertools.combinations(range(12),4):
        name="displaymask_"+"_".join(map(str,mask)); _,p,_=pred(model,parts["test"],m,s,lambda x,a,b,mask=mask:availability(x,a,b,display_mask=mask)); display+=rows(name,y,p,t)
    rev=[]
    for name in ("RA_LA_reversal","RA_LL_reversal","LA_LL_reversal"):
        _,p,_=pred(model,parts["test"],m,s,lambda x,a,b,name=name:availability(x,a,b,reversal=name)); rev+=rows(name,y,p,t)
    write_csv("clean_metrics.csv",[r for r in result if r["condition"]=="clean"]); write_csv("availability_fault_results.csv",[r for r in result if r["condition"]!="clean"]); write_csv("terminal_reversal_results.csv",rev)
    write_csv("display_mask_secondary_results.csv",display)
    write_csv("negative_control_results.csv",[r for r in result+rev if r["label"] in ("SB","STach")])
    boot=bootstrap(y,allp,"V1_V4",[x for x in names if x!="subset_2_3_4_5"]); write_csv("bootstrap_intervals.csv",boot)
    effect=np.array([r["v1_v4_minus_mean_other69_auprc_degradation"] for r in boot if r["label"]=="TWA"]); lo,hi=np.quantile(effect,[.025,.975]); twa=float(np.mean(effect)); neg={q:float(np.mean([r["v1_v4_minus_mean_other69_auprc_degradation"] for r in boot if r["label"]==q])) for q in ("SB","STach")}
    signal=abs(twa)>=.05 and not(lo<=0<=hi) and not (abs(neg["SB"])>=abs(twa) and abs(neg["STach"])>=abs(twa)); conclusion={"decision":"PROCEED_MULTISEED" if signal else "STOP_ONE_SEED_NO_MEANINGFUL_SIGNAL","test_accessed":True,"primary_twa_effect":twa,"bootstrap_95_ci":[float(lo),float(hi)],"negative_controls":neg,"ningbo_accessed":False}
    dump("conclusions_v3.json",conclusion); (OUT/"FINAL_PHASE1_PILOT_DECISION.md").write_text("# "+conclusion["decision"]+"\n\n"+json.dumps(conclusion,indent=2),encoding="utf-8")
def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("train","evaluate")); a=p.parse_args();
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    train() if a.command=="train" else evaluate()
if __name__=="__main__": main()
