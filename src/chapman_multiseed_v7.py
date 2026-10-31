"""Frozen v7 seed runner; historical v4/v6 artifacts are read-only inputs."""
import argparse,csv,hashlib,itertools,json,random,time
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import Dataset,DataLoader
from src.chapman_phase1 import ECGResNetSmall,raw,labels
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase2_chapman_multiseed_v7';V4=ROOT/'outputs/phase1_chapman_seed1_v4';V6=ROOT/'outputs/phase1b_matched_reversal_development_v6'
def h(p):
 x=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):x.update(b)
 return x.hexdigest()
def parts():
 x=json.loads((V4/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];return {s:[q['record_id'] for q in x if q['split']==s] for s in ('train','validation','test')}
class D(Dataset):
 def __init__(self,x,m,s):self.x=x;self.m=m;self.s=s
 def __len__(self):return len(self.x)
 def __getitem__(self,i):r=self.x[i];return ((raw(r)-self.m[:,None])/self.s[:,None]).astype('float32'),labels(r)
def load(x,m,s,sh=False):return DataLoader(D(x,m,s),batch_size=64 if sh else 128,shuffle=sh,num_workers=2,pin_memory=True,persistent_workers=True)
def pred(model,x,m,s):
 P=[];Y=[];model.eval()
 with torch.no_grad():
  for a,b in load(x,m,s):
   with torch.autocast('cuda',dtype=torch.float16):P.append(torch.sigmoid(model(a.cuda())).float().cpu().numpy())
   Y.append(b.numpy())
 return np.concatenate(Y),np.concatenate(P)
def seedall(z):random.seed(z);np.random.seed(z);torch.manual_seed(z);torch.cuda.manual_seed_all(z);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def train(seed):
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 d=OUT/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True); gate=d/'baseline.json'
 if gate.exists():return
 seedall(seed);p=parts();z=np.load(V4/'preprocessing_v4.npz');m,s=z['mean'],z['std'];model=ECGResNetSmall().cuda();y=np.stack([labels(r) for r in p['train']]);w=np.minimum((len(y)-y.sum(0))/np.maximum(y.sum(0),1),20);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sc=torch.amp.GradScaler('cuda');best=-1;wait=0;state=None;hist=[];start=time.monotonic()
 for e in range(1,31):
  model.train();ls=[]
  for a,b in load(p['train'],m,s,True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):q=lossfn(model(a.cuda()),b.cuda())
   if not torch.isfinite(q):raise RuntimeError('nonfinite loss')
   sc.scale(q).backward();sc.step(opt);sc.update();ls.append(float(q.detach().cpu()))
  yy,pp=pred(model,p['validation'],m,s);ap=average_precision_score(yy[:,0],pp[:,0]);hist.append({'epoch':e,'loss':float(np.mean(ls)),'twa_auprc':float(ap)});print(seed,hist[-1],flush=True)
  if ap>best:best=ap;wait=0;state={k:v.detach().cpu() for k,v in model.state_dict().items()}
  else:wait+=1
  if wait>=5:break
 model.load_state_dict(state);yy,pp=pred(model,p['validation'],m,s);t=np.array([np.quantile(pp[yy[:,i]==0,i],.9) for i in range(3)]);ok=np.isfinite(pp).all() and roc_auc_score(yy[:,0],pp[:,0])>=.75 and average_precision_score(yy[:,0],pp[:,0])>=.45
 torch.save({'state_dict':model.state_dict(),'seed':seed},d/'checkpoint.pt');np.savez_compressed(d/'validation.npz',y=yy,p=pp,thresholds=t);r={'seed':seed,'decision':'PASS' if ok else 'STOP_BASELINE_INADEQUATE','auroc':float(roc_auc_score(yy[:,0],pp[:,0])),'auprc':float(average_precision_score(yy[:,0],pp[:,0])),'thresholds':t.tolist(),'gpu_hours':(time.monotonic()-start)/3600,'history':hist,'test_accessed':False};gate.write_text(json.dumps(r,indent=2));(d/'pretest_freeze.json').write_text(json.dumps({'checkpoint':h(d/'checkpoint.pt'),'validation':h(d/'validation.npz'),'evaluator':h(Path(__file__)),'physics':h(ROOT/'src/ecg_physics.py'),'test_accessed_before_freeze':False},indent=2))
def main():
 a=argparse.ArgumentParser();a.add_argument('seed',type=int,choices=(130363,155921));x=a.parse_args();train(x.seed)
if __name__=='__main__':main()
