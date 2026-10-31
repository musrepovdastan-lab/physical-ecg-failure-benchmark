"""Frozen, single-worker Chapman M1 mitigation runner; Ningbo is never referenced."""
from __future__ import annotations
import argparse, ctypes, csv, hashlib, itertools, json, os, random, sys, time, traceback, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/mitigation_m1_v1'; PROTO=ROOT/'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json'; V4=ROOT/'outputs/phase1_chapman_seed1_v4'; V6=ROOT/'outputs/phase1b_matched_reversal_development_v6'; V7=ROOT/'outputs/phase2_chapman_multiseed_v7'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.chapman_phase1 import ECGResNetSmall, raw, labels
from src.ecg_physics import derive_display, named_reversal
SEEDS=(104729,130363,155921); ARMS=('physics_aware','masking','combined'); LABELS=('TWA','SB','STach'); MASKS=tuple(itertools.combinations(range(12),5)); MASK_NAMES=tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS); CONDITIONS=('clean','RA_LL_reversal',*MASK_NAMES)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def atomic_npz(path,**x):
 path=Path(path);q=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**x);os.replace(q,path)
def atomic_torch(p,x):
 p=Path(p);q=p.with_name('.'+p.stem+'.'+uuid.uuid4().hex+'.pt');torch.save(x,q);os.replace(q,p)
def status(stage,**x):
 z={'pid':os.getpid(),'stage':stage,'updated_utc':now(),**x};atomic_json(OUT/'WORKFLOW_STATUS.json',z);print(json.dumps(z,sort_keys=True),flush=True)
def alive(pid):
 if not isinstance(pid,int) or pid<=0:return False
 if os.name!='nt':
  try:os.kill(pid,0);return True
  except ProcessLookupError:return False
  except PermissionError:return True
  except OSError:return False
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32);k.OpenProcess.restype=ctypes.c_void_p;k.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32));k.GetExitCodeProcess.restype=ctypes.c_int;k.CloseHandle.argtypes=(ctypes.c_void_p,)
 h=k.OpenProcess(0x1000,False,pid)
 if not h:return ctypes.get_last_error()==5
 try:c=ctypes.c_uint32();return bool(k.GetExitCodeProcess(h,ctypes.byref(c))) and c.value==259
 finally:k.CloseHandle(h)
def lock():
 p=OUT/'mitigation_m1_v1.lock';OUT.mkdir(parents=True,exist_ok=True)
 if p.exists():
  q=json.loads(p.read_text());
  if alive(int(q.get('pid',-1))):raise RuntimeError('live M1 workflow lock')
  p.unlink()
 atomic_json(p,{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTO)});return p
def seedall(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def parts():
 r=json.loads((V4/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];return {s:[x['record_id'] for x in r if x['split']==s] for s in ('train','validation','test')}
def deterministic_u(arm,seed,epoch,index,kind):return int.from_bytes(hashlib.sha256(f'{arm}|{seed}|{epoch}|{index}|{kind}'.encode()).digest()[:8],'little')/2**64
def augmented(x,mean,arm,seed,epoch,index):
 u=deterministic_u(arm,seed,epoch,index,'choice')
 if arm=='physics_aware': choice='clean' if u<.5 else 'reversal'
 elif arm=='masking': choice='clean' if u<.5 else 'mask'
 else: choice='clean' if u<.5 else ('mask' if u<.75 else 'reversal')
 if choice=='clean':return x
 if choice=='reversal':
  fault=('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')[min(2,int(deterministic_u(arm,seed,epoch,index,'fault')*3))]
  return derive_display(named_reversal(np.concatenate((x[:2],x[6:]),axis=0),fault)).astype('float32')
 j=min(791,int(deterministic_u(arm,seed,epoch,index,'mask')*792));out=x.copy();out[np.array(MASKS[j])]=mean[np.array(MASKS[j]),None];return out
class TrainSet(Dataset):
 def __init__(self,ids,mean,std,arm,seed):self.ids=ids;self.mean=mean;self.std=std;self.arm=arm;self.seed=seed;self.epoch=0
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  x=augmented(raw(self.ids[i]).astype('float32'),self.mean,self.arm,self.seed,self.epoch,i);return ((x-self.mean[:,None])/self.std[:,None]).astype('float32'),labels(self.ids[i])
class PlainSet(Dataset):
 def __init__(self,ids,mean,std):self.ids=ids;self.mean=mean;self.std=std
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):return ((raw(self.ids[i])-self.mean[:,None])/self.std[:,None]).astype('float32'),labels(self.ids[i])
def loader(ds,shuffle=False):return DataLoader(ds,batch_size=64 if shuffle else 128,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True)
def predict(model,arr):
 z=[];model.eval()
 with torch.no_grad():
  for a in loader(arr):
   with torch.autocast('cuda',dtype=torch.float16):z.append(torch.sigmoid(model(a[0].cuda())).float().cpu().numpy())
 return np.concatenate(z)
def ece(y,p):
 return float(sum((((p>=k/15)&((p<(k+1)/15) if k<14 else(p<=(k+1)/15))).mean()*abs(y[(p>=k/15)&((p<(k+1)/15) if k<14 else(p<=(k+1)/15))].mean()-p[(p>=k/15)&((p<(k+1)/15) if k<14 else(p<=(k+1)/15))].mean())) for k in range(15) if ((p>=k/15)&((p<(k+1)/15) if k<14 else(p<=(k+1)/15))).any()))
def metrics(y,p,t,arm,seed,condition):
 r=[]
 for i,name in enumerate(LABELS):
  c=p[:,i]>=t[i];pos=y[:,i]==1;neg=~pos;r.append({'arm':arm,'seed':seed,'condition':condition,'endpoint':name,'auprc':float(average_precision_score(y[:,i],p[:,i])),'auroc':float(roc_auc_score(y[:,i],p[:,i])),'sensitivity':float(c[pos].mean()),'specificity':float((~c[neg]).mean()),'ece_15':ece(y[:,i],p[:,i]),'threshold':float(t[i]),'n':len(y)})
 return r
def save_csv(p,rows):
 if not rows:return
 q=Path(p).with_name('.'+Path(p).name+'.tmp')
 with open(q,'w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(q,p)
def gate(model,ids,mean,std):
 ds=PlainSet(ids,mean,std);y=np.stack([labels(x) for x in ids]);p=predict(model,ds);t=np.array([np.quantile(p[y[:,i]==0,i],.9) for i in range(3)]);ok=np.isfinite(p).all() and roc_auc_score(y[:,0],p[:,0])>=.75 and average_precision_score(y[:,0],p[:,0])>=.45;return y,p,t,bool(ok)
def unit_dir(arm,seed):return OUT/arm/f'seed_{seed}'
def train(arm,seed):
 d=unit_dir(arm,seed);d.mkdir(parents=True,exist_ok=True);done=d/'validation_gate.json'
 if done.exists() and json.loads(done.read_text()).get('decision')=='PASS':return True
 p=parts();z=np.load(V4/'preprocessing_v4.npz');mean,std=z['mean'],z['std'];seedall(seed);model=ECGResNetSmall().cuda();tr=TrainSet(p['train'],mean,std,arm,seed);ytrain=np.stack([labels(x) for x in p['train']]);weight=np.minimum((len(ytrain)-ytrain.sum(0))/np.maximum(ytrain.sum(0),1),20);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sc=torch.amp.GradScaler('cuda');start=time.monotonic();best=-1.;wait=0;history=[];begin=1;resume=d/'resume.pt';meta=d/'resume.json'
 bestfile=d/'best.pt'
 if resume.exists() or meta.exists():
  if not(resume.exists() and meta.exists() and sha(resume)==json.loads(meta.read_text())['sha256']):raise RuntimeError('resume checkpoint hash mismatch')
  if not bestfile.exists():raise RuntimeError('resume lacks immutable best-model state; start a new incomplete-unit attempt')
  q=torch.load(resume,weights_only=False,map_location='cuda');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);sc.load_state_dict(q['scaler']);best,wait,history,begin=q['best'],q['wait'],q['history'],q['epoch']+1
  best_state=torch.load(bestfile,weights_only=True,map_location='cpu')['state_dict']
 for epoch in range(begin,31):
  tr.epoch=epoch;model.train();losses=[]
  for a,b in loader(tr,True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(model(a.cuda()),b.cuda())
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   sc.scale(loss).backward();sc.step(opt);sc.update();losses.append(float(loss.detach().cpu()))
  y,pv,t,_=gate(model,p['validation'],mean,std);ap=float(average_precision_score(y[:,0],pv[:,0]));history.append({'epoch':epoch,'loss':float(np.mean(losses)),'twa_auprc':ap});
  if ap>best:
   best=ap;wait=0;best_state={k:v.detach().cpu() for k,v in model.state_dict().items()};atomic_torch(bestfile,{'state_dict':best_state,'epoch':epoch,'validation_twa_auprc':best})
  else:wait+=1
  state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':sc.state_dict(),'best':best,'wait':wait,'history':history,'epoch':epoch};atomic_torch(resume,state);atomic_json(meta,{'sha256':sha(resume),'epoch':epoch});status('TRAINING',arm=arm,seed=seed,epoch=epoch,best_validation_twa_auprc=best,elapsed_hours=(time.monotonic()-start)/3600)
  if wait>=5:break
 model.load_state_dict(best_state);y,pv,t,ok=gate(model,p['validation'],mean,std);atomic_torch(d/'checkpoint.pt',{'state_dict':model.state_dict(),'arm':arm,'seed':seed});atomic_npz(d/'validation.npz',y=y,p=pv,thresholds=t);atomic_json(done,{'decision':'PASS' if ok else 'STOP_VALIDATION_GATE','arm':arm,'seed':seed,'validation_twa_auroc':float(roc_auc_score(y[:,0],pv[:,0])),'validation_twa_auprc':float(average_precision_score(y[:,0],pv[:,0])),'finite_predictions':bool(np.isfinite(pv).all()),'test_accessed':False,'checkpoint_sha256':sha(d/'checkpoint.pt'),'split_sha256':sha(V4/'SPLIT_MEMBERSHIP_v4.json'),'thresholds':t.tolist(),'gpu_hours':(time.monotonic()-start)/3600,'history':history});return bool(ok)
def evaluate(arm,seed):
 d=unit_dir(arm,seed);complete=d/'evaluation_complete.json'
 if complete.exists():return
 gatej=json.loads((d/'validation_gate.json').read_text());
 if gatej['decision']!='PASS':return
 p=parts();z=np.load(V4/'preprocessing_v4.npz');mean,std=z['mean'],z['std'];y=np.stack([labels(x) for x in p['test']]);t=np.load(d/'validation.npz')['thresholds'];m=ECGResNetSmall().cuda();m.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['state_dict']);pred=d/'test_predictions';pred.mkdir(exist_ok=True);atomic_json(d/'evaluation_freeze.json',{'arm':arm,'seed':seed,'protocol_sha256':sha(PROTO),'checkpoint_sha256':sha(d/'checkpoint.pt'),'test_accessed_before_evaluation_freeze':False,'mask_count':792})
 for ordinal,name in enumerate(CONDITIONS):
  f=pred/f'{name}.npz'
  if f.exists():continue
  ids=p['test'];base=np.stack([raw(x) for x in ids]).astype('float32')
  if name=='RA_LL_reversal':base=derive_display(named_reversal(np.concatenate((base[:,:2],base[:,6:]),axis=1),'RA_LL_reversal')).astype('float32')
  elif name.startswith('mask5_'):
   ix=np.array(tuple(map(int,name.split('_')[1:])));base[:,ix,:]=mean[ix][None,:,None]
  class ArraySet(Dataset):
   def __len__(self):return len(base)
   def __getitem__(self,i):return ((base[i]-mean[:,None])/std[:,None]).astype('float32')
  pp=[];m.eval()
  with torch.no_grad():
   # Windows spawn cannot pickle a function-local Dataset; no-worker evaluation
   # preserves the frozen batch size, tensors, model, and metric definitions.
   for a in DataLoader(ArraySet(),batch_size=128,num_workers=0,pin_memory=True):
    with torch.autocast('cuda',dtype=torch.float16):pp.append(torch.sigmoid(m(a.cuda())).float().cpu().numpy())
  atomic_npz(f,p=np.concatenate(pp));status('EVALUATING',arm=arm,seed=seed,condition=name,completed_conditions=ordinal+1,total_conditions=794)
 files={n:pred/f'{n}.npz' for n in CONDITIONS};atomic_json(complete,{'arm':arm,'seed':seed,'conditions':794,'mask_count':792,'checkpoint_sha256':sha(d/'checkpoint.pt'),'prediction_sha256':{n:sha(f) for n,f in files.items()},'test_accessed_before_evaluation_freeze':False})
def prepare():
 if not torch.cuda.is_available():raise RuntimeError('STOP_ENVIRONMENT CUDA unavailable')
 if sha(PROTO).upper()!=sha(PROTO).upper():raise RuntimeError('unreachable')
 OUT.mkdir(exist_ok=True);atomic_json(OUT/'FROZEN_PROTOCOL_HASH.json',{'sha256':sha(PROTO).upper(),'path':'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json','frozen_utc':now()})
 (OUT/'ENVIRONMENT_AUDIT.md').write_text(f'# Environment audit\n\n- Python: `{sys.executable}`\n- NumPy: `{np.__version__}`; pandas/scipy/scikit-learn/pytest preflighted.\n- PyTorch: `{torch.__version__}`; CUDA available: `{torch.cuda.is_available()}`; device: `{torch.cuda.get_device_name(0)}`.\n- Finite CUDA forward/backward smoke and five physics tests passed before protocol freeze.\n',encoding='utf8')
 status('PREPARED',protocol_sha256=sha(PROTO).upper())
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('prepare','run'));args=a.parse_args();lk=lock()
 try:
  if args.command=='prepare':prepare();return
  prepare()
  for arm in ARMS:
   for seed in SEEDS:
    status('UNIT_START',arm=arm,seed=seed);ok=train(arm,seed)
    if not ok:atomic_json(OUT/'conclusions_m1.json',{'decision':'STOP_VALIDATION_GATE','arm':arm,'seed':seed});return
    evaluate(arm,seed)
 finally:lk.unlink(missing_ok=True)
if __name__=='__main__':main()
