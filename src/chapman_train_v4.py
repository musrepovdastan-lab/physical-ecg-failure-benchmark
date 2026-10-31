"""v4 canonical-record baseline; architecture and hyperparameters remain frozen."""
import json,random,time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import Dataset,DataLoader
from src.chapman_phase1 import ECGResNetSmall,raw,labels
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/phase1_chapman_seed1_v4'; SEED=104729
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def ids():
 x=json.loads((OUT/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];return {s:[q['record_id'] for q in x if q['split']==s] for s in ('train','validation','test')}
def stat(x):
 a=np.zeros(12);b=np.zeros(12);n=0
 for r in x:
  z=raw(r);a+=z.sum(1);b+=(z*z).sum(1);n+=z.shape[1]
 m=a/n;return m.astype('float32'),np.sqrt(np.maximum(b/n-m*m,1e-8)).astype('float32')
class D(Dataset):
 def __init__(self,x,m,s):self.x=x;self.m=m;self.s=s
 def __len__(self):return len(self.x)
 def __getitem__(self,i):r=self.x[i];return ((raw(r)-self.m[:,None])/self.s[:,None]).astype('float32'),labels(r)
def load(x,m,s,shuffle):return DataLoader(D(x,m,s),batch_size=64 if shuffle else 128,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True)
def predict(model,x,m,s):
 ps=[];ys=[];model.eval()
 with torch.no_grad():
  for a,b in load(x,m,s,False):
   with torch.autocast('cuda',dtype=torch.float16):ps.append(torch.sigmoid(model(a.cuda())).float().cpu().numpy())
   ys.append(b.numpy())
 return np.concatenate(ys),np.concatenate(ps)
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 seed();part=ids();m,s=stat(part['train']);np.savez(OUT/'preprocessing_v4.npz',mean=m,std=s)
 model=ECGResNetSmall().cuda();y=np.stack([labels(r) for r in part['train']]);w=np.minimum((len(y)-y.sum(0))/np.maximum(y.sum(0),1),20);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');best=-1;wait=0;state=None;hist=[];start=time.monotonic()
 for epoch in range(1,31):
  model.train();losses=[]
  for a,b in load(part['train'],m,s,True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(model(a.cuda()),b.cuda())
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
  yy,p=predict(model,part['validation'],m,s);ap=average_precision_score(yy[:,0],p[:,0]);hist.append({'epoch':epoch,'loss':float(np.mean(losses)),'twa_auprc':float(ap)});print(hist[-1],flush=True)
  if ap>best:best=ap;wait=0;state={k:v.detach().cpu() for k,v in model.state_dict().items()}
  else:wait+=1
  if wait>=5:break
 model.load_state_dict(state);yy,p=predict(model,part['validation'],m,s);gpu=(time.monotonic()-start)/3600;t=np.array([np.quantile(p[yy[:,i]==0,i],.9) for i in range(3)]);ok=np.isfinite(p).all() and all(np.isfinite(q['loss']) for q in hist) and roc_auc_score(yy[:,0],p[:,0])>=.75 and average_precision_score(yy[:,0],p[:,0])>=.45 and gpu<=8
 torch.save({'state_dict':model.state_dict(),'seed':SEED,'labels':['164934002','426177001','427084000']},OUT/'ecgresnet_small_seed104729_v4.pt')
 r={'decision':'PASS' if ok else 'STOP_BASELINE_INADEQUATE','validation_twa_auroc':float(roc_auc_score(yy[:,0],p[:,0])),'validation_twa_auprc':float(average_precision_score(yy[:,0],p[:,0])),'thresholds_at_90pct_specificity':t.tolist(),'training_gpu_hours':gpu,'history':hist,'finite_predictions':bool(np.isfinite(p).all()),'test_accessed':False};(OUT/'BASELINE_ADEQUACY_v4.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'validation_predictions_v4.npz',y=yy,p=p,thresholds=t)
if __name__=='__main__':main()
