"""Frozen, resumable M3A Transformer trainer/evaluator; M3B is gated separately."""
from __future__ import annotations
import argparse,ctypes,hashlib,itertools,json,os,random,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/architecture_replication_m3_v1';PROTO=ROOT/'configs/FROZEN_ARCHITECTURE_M3_PROTOCOL_v1.json';V4=ROOT/'outputs/phase1_chapman_seed1_v4';N=ROOT/'outputs/ningbo_confirmation_v2'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.chapman_phase1 import raw as craw,labels as clabels
from src import ningbo_confirmation_v1 as ning
from src.ecg_physics import derive_display,named_reversal
from src.m3_transformer import TimePatchTransformer
SEEDS=(104729,130363,155921);ARMS=('clean','physics_aware');MASKS=tuple(itertools.combinations(range(12),5));NAMES=('clean','RA_LL_reversal',*tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS));LABELS=('TWA','SB','STach')
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def aj(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf8');os.replace(q,path)
def an(target,**arrays):
 target=Path(target);q=target.with_name('.'+target.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**arrays);os.replace(q,target)
def at(path,obj):
 path=Path(path);q=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.pt');torch.save(obj,q);os.replace(q,path)
def status(stage,**x):
 z={'pid':os.getpid(),'stage':stage,'updated_utc':now(),'completed_conditions':0,'total_conditions':794,'bootstrap_completed':0,'bootstrap_total':1000,'estimated_remaining_seconds':None,**x};aj(OUT/'WORKFLOW_STATUS.json',z);print(json.dumps(z,sort_keys=True),flush=True)
def alive(pid):
 if not isinstance(pid,int) or pid<=0:return False
 if os.name!='nt':
  try:os.kill(pid,0);return True
  except OSError:return False
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32);k.OpenProcess.restype=ctypes.c_void_p;k.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32));k.GetExitCodeProcess.restype=ctypes.c_int;k.CloseHandle.argtypes=(ctypes.c_void_p,);h=k.OpenProcess(0x1000,False,pid)
 if not h:return ctypes.get_last_error()==5
 try:c=ctypes.c_uint32();return bool(k.GetExitCodeProcess(h,ctypes.byref(c))) and c.value==259
 finally:k.CloseHandle(h)
def lock():
 p=OUT/'architecture_m3_trainer.lock';OUT.mkdir(parents=True,exist_ok=True)
 if p.exists():
  q=json.loads(p.read_text())
  if alive(q.get('pid')):raise RuntimeError('live M3 trainer lock')
  p.unlink()
 aj(p,{'pid':os.getpid(),'protocol_sha256':sha(PROTO).upper(),'started_utc':now()});return p
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def u(arm,seed,epoch,index,tag):return int.from_bytes(hashlib.sha256(f'{arm}|{seed}|{epoch}|{index}|{tag}'.encode()).digest()[:8],'little')/2**64
def fault(x,arm,seed,epoch,index):
 if arm=='clean' or u(arm,seed,epoch,index,'choice')<.5:return x
 f=('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')[min(2,int(u(arm,seed,epoch,index,'fault')*3))]
 return derive_display(named_reversal(np.concatenate((x[:2],x[6:]),axis=0),f)).astype('float32')
def split(dataset):
 if dataset=='Chapman':
  rows=json.loads((V4/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];return {s:[r['record_id'] for r in rows if r['split']==s] for s in ('train','validation','test')}
 can=json.loads((N/'CANONICAL_RECORDS.json').read_text())['records'];by={r['record_id']:r for r in can};rows=json.loads((N/'SPLIT_MEMBERSHIP.json').read_text())['rows'];return {s:[by[r['record_id']] for r in rows if r['split']==s] for s in ('train','validation','test')}
def getraw(dataset,r):return craw(r).astype('float32') if dataset=='Chapman' else ning.raw(r).astype('float32')
def getlabel(dataset,r):return np.asarray(clabels(r) if dataset=='Chapman' else r['labels'],np.float32)
def stats(dataset,records):
 if dataset=='Chapman':
  z=np.load(V4/'preprocessing_v4.npz');return z['mean'],z['std']
 return ning.stats(records)
class DS(Dataset):
 def __init__(self,dataset,records,mean,std,arm=None,seed=None,epoch=0):self.dataset,self.records,self.mean,self.std,self.arm,self.seed,self.epoch=dataset,records,mean,std,arm,seed,epoch
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  x=getraw(self.dataset,self.records[i]);x=fault(x,self.arm,self.seed,self.epoch,i) if self.arm else x
  return ((x-self.mean[:,None])/self.std[:,None]).astype('float32'),getlabel(self.dataset,self.records[i])
def dl(ds,train=False):return DataLoader(ds,batch_size=64 if train else 128,shuffle=train,num_workers=2,pin_memory=True,persistent_workers=True)
def predict(model,dataset,records,m,s):
 y=[];p=[];model.eval()
 with torch.no_grad():
  for x,z in dl(DS(dataset,records,m,s),False):
   with torch.autocast('cuda',dtype=torch.float16):p.append(torch.sigmoid(model(x.cuda(non_blocking=True))).float().cpu().numpy())
   y.append(z.numpy())
 return np.concatenate(y),np.concatenate(p)
def verify_protocol(dataset):
 c=json.loads(PROTO.read_text());bad=[]
 for v in c['immutable_bindings'].values():
  if 'sha256' in v and sha(ROOT/v['path']).upper()!=v['sha256']:bad.append(v['path'])
 p=split(dataset)
 if dataset=='Ningbo' and (len(p['train']),len(p['validation']),len(p['test']))!=(24430,3488,6981):bad.append('Ningbo split counts')
 if bad:raise RuntimeError('immutable binding mismatch '+','.join(bad))
 return p
def gate(model,dataset,part,m,s):
 y,p=predict(model,dataset,part['validation'],m,s);t=np.asarray([np.quantile(p[y[:,i]==0,i],.9) for i in range(3)],np.float32);a=float(average_precision_score(y[:,0],p[:,0]));r=float(roc_auc_score(y[:,0],p[:,0]));return y,p,t,bool(np.isfinite(p).all() and a>=.45 and r>=.75),a,r
def unit(dataset,arm,seed):return OUT/dataset/arm/f'seed_{seed}'
def train(dataset,arm,seed,part):
 d=unit(dataset,arm,seed);d.mkdir(parents=True,exist_ok=True);g=d/'validation_gate.json'
 if g.exists() and json.loads(g.read_text()).get('decision')=='PASS':return True
 seedall(seed);m,s=stats(dataset,part['train']);model=TimePatchTransformer().cuda();ytr=np.stack([getlabel(dataset,r) for r in part['train']]);w=np.minimum((len(ytr)-ytr.sum(0))/np.maximum(ytr.sum(0),1),20);loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);sc=torch.amp.GradScaler('cuda');best=-1.;wait=0;hist=[];start=1;resume=d/'resume.pt';bestfile=d/'best.pt'
 if resume.exists():
  q=torch.load(resume,weights_only=False,map_location='cuda')
  if q.get('protocol_sha256')!=sha(PROTO).upper() or not bestfile.exists():raise RuntimeError('resume provenance')
  model.load_state_dict(q['model']);opt.load_state_dict(q['opt']);sc.load_state_dict(q['sc']);best,wait,hist,start=q['best'],q['wait'],q['history'],q['epoch']+1
 tr=DS(dataset,part['train'],m,s,arm,seed)
 for epoch in range(start,31):
  tr.epoch=epoch;model.train();ls=[]
  for x,y in dl(tr,True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):z=loss(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
   if not torch.isfinite(z):raise RuntimeError('nonfinite loss')
   sc.scale(z).backward();sc.step(opt);sc.update();ls.append(float(z.detach().cpu()))
  _,pv,_,_,ap,_=gate(model,dataset,part,m,s);hist.append({'epoch':epoch,'train_loss':float(np.mean(ls)),'validation_twa_auprc':ap})
  if ap>best:best=ap;wait=0;at(bestfile,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'validation_twa_auprc':best})
  else:wait+=1
  at(resume,{'model':model.state_dict(),'opt':opt.state_dict(),'sc':sc.state_dict(),'best':best,'wait':wait,'history':hist,'epoch':epoch,'protocol_sha256':sha(PROTO).upper()});status('TRAINING',dataset=dataset,arm=arm,seed=seed,epoch=epoch,best_validation_twa_auprc=best)
  if wait>=5:break
 model.load_state_dict(torch.load(bestfile,weights_only=True,map_location='cpu')['state_dict']);y,pv,t,ok,ap,au=gate(model,dataset,part,m,s);at(d/'checkpoint.pt',{'state_dict':model.state_dict(),'dataset':dataset,'arm':arm,'seed':seed,'protocol_sha256':sha(PROTO).upper()});an(d/'preprocessing.npz',mean=m,std=s);an(d/'validation.npz',y=y,p=pv,thresholds=t);aj(g,{'decision':'PASS' if ok else ('STOP_M3A_VALIDATION_GATE' if dataset=='Chapman' else 'STOP_M3B_VALIDATION_GATE'),'dataset':dataset,'arm':arm,'seed':seed,'validation_twa_auprc':ap,'validation_twa_auroc':au,'finite_predictions':bool(np.isfinite(pv).all()),'checkpoint_sha256':sha(d/'checkpoint.pt'),'protocol_sha256':sha(PROTO).upper(),'history':hist,'test_accessed_before_evaluation_freeze':False});return ok
def evaluate(dataset,arm,seed,part):
 d=unit(dataset,arm,seed);g=json.loads((d/'validation_gate.json').read_text());done=d/'evaluation_complete.json'
 if g['decision']!='PASS':return
 if done.exists():return
 z=np.load(d/'preprocessing.npz');m,s=z['mean'],z['std'];model=TimePatchTransformer().cuda();model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['state_dict']);pred=d/'test_predictions';pred.mkdir(exist_ok=True);aj(d/'evaluation_freeze.json',{'dataset':dataset,'arm':arm,'seed':seed,'protocol_sha256':sha(PROTO).upper(),'checkpoint_sha256':sha(d/'checkpoint.pt'),'test_accessed_before_evaluation_freeze':False,'conditions':794})
 x=np.stack([getraw(dataset,r) for r in part['test']]).astype('float32');ind=np.concatenate((x[:,:2],x[:,6:]),axis=1);fixed={'clean':x,'RA_LL_reversal':derive_display(named_reversal(ind,'RA_LL_reversal')).astype('float32')}
 for i,n in enumerate(NAMES):
  f=pred/f'{n}.npz'
  if f.exists():continue
  a=fixed.get(n);a=((a if a is not None else np.where(np.isin(np.arange(12),np.asarray(MASKS[i-2]))[None,:,None],x,m[None,:,None]))-m[None,:,None])/s[None,:,None];out=[]
  with torch.no_grad():
   for j in range(0,len(a),128):
    with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(a[j:j+128].astype('float32')).cuda())).float().cpu().numpy())
  an(f,p=np.concatenate(out));status('EVALUATING',dataset=dataset,arm=arm,seed=seed,condition=n,completed_conditions=i+1,total_conditions=794)
 aj(done,{'dataset':dataset,'arm':arm,'seed':seed,'conditions':794,'checkpoint_sha256':sha(d/'checkpoint.pt'),'prediction_sha256':{n:sha(pred/f'{n}.npz') for n in NAMES},'test_accessed_before_evaluation_freeze':False})
def prepare(dataset):
 part=verify_protocol(dataset)
 if not torch.cuda.is_available() or '5070' not in torch.cuda.get_device_name(0):raise RuntimeError('STOP_M3_TECHNICAL_BLOCKER CUDA RTX 5070 unavailable')
 model=TimePatchTransformer().cuda();x=torch.randn(2,12,1000,device='cuda');z=model(x);loss=z.square().mean();loss.backward()
 if not torch.isfinite(loss):raise RuntimeError('STOP_M3_TECHNICAL_BLOCKER smoke')
 env={'decision':'PASS','protocol_sha256':sha(PROTO).upper(),'dataset':dataset,'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'transformer_parameters':sum(p.numel() for p in model.parameters()),'cuda_smoke_loss':float(loss.detach().cpu()),'disk_available_bytes':__import__('shutil').disk_usage(ROOT).free,'test_accessed':False};aj(OUT/'ENVIRONMENT_AUDIT.json',env);(OUT/'ENVIRONMENT_AUDIT.md').write_text('# M3 environment audit\n\n'+json.dumps(env,indent=2)+'\n',encoding='utf8');aj(OUT/'FROZEN_PROTOCOL_HASH.json',{'sha256':sha(PROTO).upper(),'path':'configs/FROZEN_ARCHITECTURE_M3_PROTOCOL_v1.json','frozen_utc':now()});return part
def selftest():
 r=subprocess.run([sys.executable,'-m','pytest','tests/test_architecture_m3.py','tests/test_ecg_physics.py','tests/test_m2_orchestration.py','-q'],cwd=ROOT,text=True,capture_output=True)
 aj(OUT/'IMPLEMENTATION_TEST_AUDIT.json',{'decision':'PASS' if r.returncode==0 else 'FAIL','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'synthetic_data_only':True,'test_accessed':False,'completed_utc':now()})
 if r.returncode:return False
 return True
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('selftest','prepare','run'));a.add_argument('--dataset',choices=('Chapman','Ningbo'),default='Chapman');q=a.parse_args();lk=lock()
 try:
  if q.command=='selftest':
   if not selftest():raise RuntimeError('STOP_M3_TECHNICAL_BLOCKER implementation tests')
   status('SELFTEST_PASS',dataset=q.dataset);return
  p=prepare(q.dataset);status('PREPARED',dataset=q.dataset,protocol_sha256=sha(PROTO).upper())
  if q.command=='prepare':return
  for arm in ARMS:
   for s in SEEDS:
    status('UNIT_START',dataset=q.dataset,arm=arm,seed=s);ok=train(q.dataset,arm,s,p)
    if not ok:aj(OUT/'conclusions_m3.json',{'decision':'STOP_M3A_VALIDATION_GATE' if q.dataset=='Chapman' else 'STOP_M3B_VALIDATION_GATE','dataset':q.dataset,'arm':arm,'seed':s});return
    evaluate(q.dataset,arm,s,p)
 finally:lk.unlink(missing_ok=True)
if __name__=='__main__':main()
