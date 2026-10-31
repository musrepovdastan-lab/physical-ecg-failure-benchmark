"""Frozen M2 Ningbo trainer/evaluator: physics-aware primary method only."""
from __future__ import annotations
import argparse,ctypes,hashlib,itertools,json,os,random,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mitigation_m2_ningbo_v1';N=ROOT/'outputs/ningbo_confirmation_v2';PROTO=ROOT/'configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import ningbo_confirmation_v1 as b
from src.chapman_phase1 import ECGResNetSmall
from src.ecg_physics import derive_display,named_reversal
SEEDS=(104729,130363,155921);LABELS=('TWA','SB','STach');MASKS=tuple(itertools.combinations(range(12),5));NAMES=('clean','RA_LL_reversal',*tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS));EXPECTED_TEST=6981
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def aj(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def an(target,**arrays):
 """Atomically save named arrays; ``p`` is a legal scientific array name."""
 target=Path(target);q=target.with_name('.'+target.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**arrays);os.replace(q,target)
def at(p,x):
 p=Path(p);q=p.with_name('.'+p.stem+'.'+uuid.uuid4().hex+'.pt');torch.save(x,q);os.replace(q,p)
def status(stage,**x):
 z={'pid':os.getpid(),'stage':stage,'updated_utc':now(),'completed_conditions':0,'total_conditions':794,'bootstrap_completed':0,'bootstrap_total':1000,'estimated_remaining_seconds':None,**x};aj(OUT/'WORKFLOW_STATUS.json',z);print(json.dumps(z,sort_keys=True),flush=True)
def alive(pid):
 if not isinstance(pid,int) or pid<=0:return False
 if os.name!='nt':
  try:os.kill(pid,0);return True
  except ProcessLookupError:return False
  except PermissionError:return True
  except OSError:return False
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32);k.OpenProcess.restype=ctypes.c_void_p;k.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32));k.GetExitCodeProcess.restype=ctypes.c_int;k.CloseHandle.argtypes=(ctypes.c_void_p,);h=k.OpenProcess(0x1000,False,pid)
 if not h:return ctypes.get_last_error()==5
 try:c=ctypes.c_uint32();return bool(k.GetExitCodeProcess(h,ctypes.byref(c))) and c.value==259
 finally:k.CloseHandle(h)
def lock():
 p=OUT/'mitigation_m2_ningbo_v1.lock';OUT.mkdir(parents=True,exist_ok=True)
 if p.exists():
  q=json.loads(p.read_text())
  if alive(q.get('pid')):raise RuntimeError('live M2 workflow lock')
  p.unlink()
 aj(p,{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTO)});return p
def seedall(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def u(seed,epoch,index,tag):return int.from_bytes(hashlib.sha256(f'physics_aware|{seed}|{epoch}|{index}|{tag}'.encode()).digest()[:8],'little')/2**64
def augment(x,seed,epoch,index):
 if u(seed,epoch,index,'choice')<.5:return x
 f=('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')[min(2,int(u(seed,epoch,index,'fault')*3))]
 return derive_display(named_reversal(np.concatenate((x[:2],x[6:]),axis=0),f)).astype('float32')
class DS(Dataset):
 def __init__(self,records,mean,std,seed=None,epoch=0):self.records,self.mean,self.std,self.seed,self.epoch=records,mean,std,seed,epoch
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  x=b.raw(self.records[i]).astype('float32');x=augment(x,self.seed,self.epoch,i) if self.seed is not None else x
  return ((x-self.mean[:,None])/self.std[:,None]).astype('float32'),np.asarray(self.records[i]['labels'],np.float32)
def dl(ds,train=False):return DataLoader(ds,batch_size=64 if train else 128,shuffle=train,num_workers=2,pin_memory=True,persistent_workers=True)
def predict(model,records,m,s):
 p=[];y=[];model.eval()
 with torch.no_grad():
  for x,z in dl(DS(records,m,s),False):
   with torch.autocast('cuda',dtype=torch.float16):p.append(torch.sigmoid(model(x.cuda(non_blocking=True))).float().cpu().numpy())
   y.append(z.numpy())
 return np.concatenate(y),np.concatenate(p)
def binding_check():
 cfg=json.loads(PROTO.read_text());bad=[]
 for x in cfg['immutable_bindings'].values():
  p=ROOT/x['path'];bad+=[] if p.is_file() and sha(p).upper()==x['sha256'] else [x['path']]
 for ss,x in cfg['baseline_artifacts'].items():
  d=N/f'seed_{ss}'
  for name,key in [('checkpoint.pt','checkpoint'),('validation.npz','validation'),('baseline.json','baseline'),('evaluation_complete.json','evaluation')]:
   if sha(d/name).upper()!=x[key]:bad.append(f'{ss}/{name}')
  # The cached baseline predictions are scientific inputs to the paired M2
  # comparison.  Verify every manifest-bound cache without loading data or
  # running inference; a mere 794-condition count is not sufficient.
  e=json.loads((d/'evaluation_complete.json').read_text())
  caches=e.get('prediction_sha256',{})
  if e.get('decision')!='PASS' or e.get('conditions')!=794 or len(caches)!=794:
   bad.append(f'{ss}/prediction_manifest')
  for condition,digest in caches.items():
   q=d/'test_predictions'/f'{condition}.npz'
   if not q.is_file() or sha(q).lower()!=str(digest).lower():bad.append(f'{ss}/test_predictions/{condition}')
 if bad:raise RuntimeError('immutable binding mismatch: '+','.join(bad))
 return cfg
def part():
 can=json.loads((N/'CANONICAL_RECORDS.json').read_text())['records'];split=json.loads((N/'SPLIT_MEMBERSHIP.json').read_text())['rows'];by={x['record_id']:x for x in can};out={s:[by[x['record_id']] for x in split if x['split']==s] for s in ('train','validation','test')}
 if (len(can),len(out['train']),len(out['validation']),len(out['test']))!=(34899,24430,3488,6981):raise RuntimeError('canonical split counts mismatch')
 proof=json.loads((N/'SPLIT_LEAKAGE_PROOF.json').read_text())
 if proof.get('record_id_overlap')!=0 or proof.get('waveform_hash_overlap')!=0:raise RuntimeError('split leakage proof mismatch')
 return out
def prepare():
 cfg=binding_check();p=part()
 npz_writer_selftest()
 # Run the small algebraic suite from the canonical root.  It uses only
 # synthetic terminals and therefore cannot access Ningbo test waveforms.
 test=subprocess.run([sys.executable,'-m','pytest','tests/test_ecg_physics.py','-q'],cwd=ROOT,text=True,capture_output=True)
 physics={'decision':'PASS' if test.returncode==0 else 'FAIL','returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr,'test_accessed':False,'completed_utc':now()}
 aj(OUT/'PHYSICS_TEST_AUDIT.json',physics)
 if test.returncode!=0:raise RuntimeError('STOP_M2_ENVIRONMENT physics test failure')
 m1_status=json.loads((ROOT/'outputs/mitigation_m1_v1/WORKFLOW_STATUS.json').read_text())
 m1_active=alive(m1_status.get('pid'))
 if m1_active:raise RuntimeError('STOP_M2_ENVIRONMENT M1 workflow remains active')
 if not torch.cuda.is_available() or '5070' not in torch.cuda.get_device_name(0):raise RuntimeError('STOP_M2_ENVIRONMENT CUDA/RTX 5070 unavailable')
 x=torch.randn(4,12,1000,device='cuda');y=torch.randn(4,3,device='cuda');m=ECGResNetSmall().cuda();loss=nn.MSELoss()(m(x),y);loss.backward()
 if not torch.isfinite(loss):raise RuntimeError('STOP_M2_ENVIRONMENT nonfinite CUDA smoke')
 for s in SEEDS:
  d=N/f'seed_{s}';e=json.loads((d/'evaluation_complete.json').read_text());
  if e.get('conditions')!=794 or len(e.get('prediction_sha256',{}))!=794:raise RuntimeError('baseline prediction cache mismatch')
 env={'decision':'PASS','python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_smoke_loss':float(loss.detach().cpu()),'protocol_sha256':sha(PROTO).upper(),'canonical_counts':{k:len(v) for k,v in p.items()},'disk_available_bytes':__import__('shutil').disk_usage(ROOT).free,'m1_status':m1_status,'m1_finalizer_active':m1_active,'baseline_prediction_hashes_verified':3*794}
 aj(OUT/'ENVIRONMENT_AUDIT.json',env);(OUT/'ENVIRONMENT_AUDIT.md').write_text('# M2 environment audit\n\n'+json.dumps(env,indent=2)+'\n',encoding='utf8');aj(OUT/'FROZEN_PROTOCOL_HASH.json',{'sha256':sha(PROTO).upper(),'frozen_utc':now(),'path':'configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json'});status('PREPARED',protocol_sha256=sha(PROTO).upper());return p
def npz_writer_selftest():
 """Regression guard for the former path/``p`` keyword collision; no ECG access."""
 target=OUT/'NPZ_WRITER_SELFTEST.npz';y=np.asarray([[0,1,0],[1,0,1]],np.int8);p=np.asarray([[.1,.8,.2],[.9,.3,.7]],np.float32);thresholds=np.asarray([.4,.5,.6],np.float32)
 an(target,y=y,p=p,thresholds=thresholds)
 with np.load(target,allow_pickle=False) as z:
  ok=(set(z.files)=={'y','p','thresholds'} and z['y'].shape==y.shape and z['p'].shape==p.shape and z['thresholds'].shape==thresholds.shape and np.array_equal(z['y'],y) and np.array_equal(z['p'],p) and np.array_equal(z['thresholds'],thresholds) and all(np.isfinite(z[k]).all() for k in z.files))
 aj(OUT/'NPZ_WRITER_SELFTEST.json',{'decision':'PASS' if ok else 'FAIL','arrays':{'y':list(y.shape),'p':list(p.shape),'thresholds':list(thresholds.shape)},'values_exact':bool(ok),'finite':bool(ok),'test_accessed':False,'completed_utc':now()})
 if not ok:raise RuntimeError('STOP_M2_TECHNICAL_BLOCKER NPZ writer self-test')
def recovery_integrity(seed):
 """Audit the interrupted seed before its validation-only recovery mutates it."""
 d=OUT/f'seed_{seed}';files=('best.pt','checkpoint.pt','preprocessing.npz','resume.pt');missing=[x for x in files if not (d/x).is_file()]
 if missing:raise RuntimeError('recovery artifacts missing: '+','.join(missing))
 best=torch.load(d/'best.pt',weights_only=True,map_location='cpu');checkpoint=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');resume=torch.load(d/'resume.pt',weights_only=False,map_location='cpu')
 with np.load(d/'preprocessing.npz',allow_pickle=False) as z:m,s=z['mean'],z['std']
 same_state=(set(best['state_dict'])==set(checkpoint['state_dict']) and all(torch.equal(best['state_dict'][k],checkpoint['state_dict'][k]) for k in best['state_dict']))
 history=resume.get('history',[]);ok=(resume.get('protocol_sha256')==sha(PROTO).upper() and resume.get('epoch')==22 and resume.get('wait',0)>=5 and len(history)==22 and m.shape==(12,) and s.shape==(12,) and np.isfinite(m).all() and np.isfinite(s).all() and same_state and not (d/'evaluation_freeze.json').exists() and not (d/'evaluation_complete.json').exists() and not (d/'test_predictions').exists())
 audit={'decision':'PASS' if ok else 'FAIL','seed':seed,'protocol_sha256':sha(PROTO).upper(),'artifact_sha256':{x:sha(d/x) for x in files},'best_epoch':best.get('epoch'),'best_validation_twa_auprc':best.get('validation_twa_auprc'),'resume_epoch':resume.get('epoch'),'resume_wait':resume.get('wait'),'history_length':len(history),'preprocessing_shapes':{'mean':list(m.shape),'std':list(s.shape)},'best_checkpoint_state_equal':same_state,'test_evaluation_started':(d/'evaluation_freeze.json').exists() or (d/'evaluation_complete.json').exists() or (d/'test_predictions').exists(),'completed_utc':now()}
 aj(d/'RECOVERY_INTEGRITY_AUDIT.json',audit)
 if not ok:raise RuntimeError('recovery seed integrity/provenance failure')
 return m,s,history
def validation_artifacts(seed,p,m,s,hist):
 """Validation-only completion/recovery from the frozen best checkpoint."""
 d=OUT/f'seed_{seed}';bestfile=d/'best.pt';model=ECGResNetSmall().cuda();best=torch.load(bestfile,weights_only=True,map_location='cpu');model.load_state_dict(best['state_dict']);y,q=predict(model,p['validation'],m,s);t=np.asarray([np.quantile(q[y[:,i]==0,i],.9) for i in range(3)],np.float32);au=float(roc_auc_score(y[:,0],q[:,0]));ap=float(average_precision_score(y[:,0],q[:,0]));ok=bool(np.isfinite(q).all() and au>=.75 and ap>=.45)
 at(d/'checkpoint.pt',{'state_dict':model.state_dict(),'seed':seed,'protocol_sha256':sha(PROTO).upper(),'source':'best.pt_validation_recovery_or_completion'});an(d/'preprocessing.npz',mean=m,std=s);an(d/'validation.npz',y=y,p=q,thresholds=t)
 aj(d/'validation_gate.json',{'decision':'PASS' if ok else 'STOP_M2_VALIDATION_GATE','seed':seed,'validation_twa_auroc':au,'validation_twa_auprc':ap,'finite_losses_predictions':bool(np.isfinite(q).all()),'checkpoint_sha256':sha(d/'checkpoint.pt'),'best_checkpoint_sha256':sha(bestfile),'split_sha256':sha(N/'SPLIT_MEMBERSHIP.json'),'protocol_sha256':sha(PROTO).upper(),'thresholds':t.tolist(),'history':hist,'test_accessed_before_evaluation_freeze':False})
 return ok
def train(seed,p):
 d=OUT/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);gate=d/'validation_gate.json'
 if gate.exists() and json.loads(gate.read_text()).get('decision')=='PASS':return True
 seedall(seed);resume=d/'resume.pt';bestfile=d/'best.pt'
 if resume.exists():
  q=torch.load(resume,weights_only=False,map_location='cpu')
  if q.get('protocol_sha256')!=sha(PROTO).upper() or not bestfile.exists():raise RuntimeError('resume provenance/best state mismatch')
  hist=q['history']
  # The writer failed after frozen early stopping.  Recover validation only;
  # do not permit an epoch 23 or a second model fit.
  if q.get('wait',0)>=5:
   m,s,hist=recovery_integrity(seed);return validation_artifacts(seed,p,m,s,hist)
 m,s=b.stats(p['train']);model=ECGResNetSmall().cuda();yt=np.asarray([x['labels'] for x in p['train']],np.float32);w=np.minimum((len(yt)-yt.sum(0))/np.maximum(yt.sum(0),1),20);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sc=torch.amp.GradScaler('cuda');best=-1.;wait=0;hist=[];start=1
 if resume.exists():
  q=torch.load(resume,weights_only=False,map_location='cuda');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);sc.load_state_dict(q['scaler']);best,wait,hist,start=q['best'],q['wait'],q['history'],q['epoch']+1
 for epoch in range(start,31):
  model.train();ls=[]
  for x,y in dl(DS(p['train'],m,s,seed,epoch),True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   sc.scale(loss).backward();sc.step(opt);sc.update();ls.append(float(loss.detach().cpu()))
  y,q=predict(model,p['validation'],m,s);ap=float(average_precision_score(y[:,0],q[:,0]));hist.append({'epoch':epoch,'train_loss':float(np.mean(ls)),'validation_twa_auprc':ap})
  if ap>best:best=ap;wait=0;at(bestfile,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'validation_twa_auprc':best})
  else:wait+=1
  at(resume,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':sc.state_dict(),'best':best,'wait':wait,'history':hist,'epoch':epoch,'protocol_sha256':sha(PROTO).upper()});status('TRAINING',seed=seed,epoch=epoch,best_validation_twa_auprc=best)
  if wait>=5:break
 return validation_artifacts(seed,p,m,s,hist)
def evaluate(seed,p):
 d=OUT/f'seed_{seed}';pred=d/'test_predictions';g=json.loads((d/'validation_gate.json').read_text());done=d/'evaluation_complete.json'
 if g['decision']!='PASS':return
 if done.exists():
  e=json.loads(done.read_text());c=e.get('prediction_sha256',{})
  if e.get('conditions')==794 and e.get('checkpoint_sha256')==g.get('checkpoint_sha256') and len(c)==794 and all((pred/f'{n}.npz').is_file() and sha(pred/f'{n}.npz').lower()==str(h).lower() for n,h in c.items()):return
  raise RuntimeError('completed evaluation manifest integrity mismatch')
 z=np.load(d/'preprocessing.npz');m,s=z['mean'],z['std'];model=ECGResNetSmall().cuda();model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['state_dict']);pred.mkdir(exist_ok=True);aj(d/'evaluation_freeze.json',{'seed':seed,'protocol_sha256':sha(PROTO).upper(),'checkpoint_sha256':sha(d/'checkpoint.pt'),'split_sha256':sha(N/'SPLIT_MEMBERSHIP.json'),'test_accessed_before_evaluation_freeze':False,'conditions':794})
 x=np.stack([b.raw(r) for r in p['test']]).astype('float32');ind=np.concatenate((x[:,:2],x[:,6:]),axis=1);fixed={'clean':x,'RA_LL_reversal':derive_display(named_reversal(ind,'RA_LL_reversal')).astype('float32')}
 for i,n in enumerate(NAMES):
  f=pred/f'{n}.npz'
  if f.exists():
   with np.load(f,allow_pickle=False) as old:valid=old['p'].shape==(EXPECTED_TEST,3) and np.isfinite(old['p']).all()
   if valid:continue
   raise RuntimeError('partial prediction artifact invalid: '+str(f))
  a=fixed.get(n);a=((a if a is not None else np.where(np.isin(np.arange(12),np.asarray(MASKS[i-2]))[None,:,None],x,m[None,:,None]))-m[None,:,None])/s[None,:,None]
  out=[];model.eval()
  with torch.no_grad():
   for j in range(0,len(a),128):
    with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(a[j:j+128].astype('float32')).cuda())).float().cpu().numpy())
 an(f,p=np.concatenate(out));status('EVALUATING',seed=seed,completed_conditions=i+1,total_conditions=794,condition=n)
 aj(done,{'seed':seed,'conditions':794,'checkpoint_sha256':sha(d/'checkpoint.pt'),'prediction_sha256':{n:sha(pred/f'{n}.npz') for n in NAMES},'test_accessed_before_evaluation_freeze':False})
def launch_finalizer_after_trainer_release(trainer_lock):
 """The cached-prediction finalizer is forbidden while the trainer lock exists."""
 if Path(trainer_lock).exists():raise RuntimeError('trainer lock must be released before finalizer launch')
 status('FINALIZING',completed_conditions=3*794,total_conditions=3*794)
 subprocess.run([sys.executable,'src/mitigation_m2_finalize_v1.py','run'],cwd=ROOT,check=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('prepare','run'));q=a.parse_args();lk=lock();launch_finalizer=False
 try:
  p=prepare()
  if q.command=='prepare':return
  for s in SEEDS:
   status('UNIT_START',seed=s);ok=train(s,p)
   if not ok:aj(OUT/'conclusions_m2.json',{'decision':'STOP_M2_VALIDATION_GATE','seed':s,'ningbo_accessed':False});return
   evaluate(s,p)
  launch_finalizer=True
 finally:lk.unlink(missing_ok=True)
 if launch_finalizer:launch_finalizer_after_trainer_release(lk)
if __name__=='__main__':main()
