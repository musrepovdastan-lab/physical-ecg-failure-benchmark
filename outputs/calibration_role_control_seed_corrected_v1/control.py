from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,csv,hashlib,random,time,shutil,itertools,subprocess,pickle,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
N=ROOT/'outputs/ningbo_confirmation_v2'; PREF=ROOT/'outputs/calibration_independence_preflight_v1'
SEEDS=(104729,130363,155921); METHODS=('ordinary','physics_aware_p050','combined')
FAULTS=('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'); ENDS=('TWA','SB','STach')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src import ningbo_confirmation_v1 as data, mask_semantics_recovery_v2 as recovery
from src.chapman_phase1 import ECGResNetSmall
from src.ecg_physics import derive_display,named_reversal
from src.heedb_huang_adapter_v1 import electrical_reversal_batch,corrected_batch,diagnostic_input

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def hobj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def configure_determinism():
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
def set_deterministic_seed(seed):
 configure_determinism();random.seed(seed);np.random.seed(seed % (2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def rng_state():
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
def restore_rng_state(q):
 random.setstate(q['python']);np.random.set_state(q['numpy']);torch.set_rng_state(q['torch_cpu']);torch.cuda.set_rng_state_all(q['torch_cuda'])
def rng_digest(q):
 h=hashlib.sha256();h.update(pickle.dumps(q['python'],protocol=4));n=q['numpy'];h.update(n[0].encode());h.update(n[1].tobytes());h.update(str((n[2],n[3],n[4])).encode());h.update(q['torch_cpu'].detach().cpu().contiguous().numpy().tobytes())
 for x in q['torch_cuda']:h.update(x.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def state_digest(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  x=t.detach().cpu().contiguous();meta=json.dumps([name,str(x.dtype),list(x.shape)],separators=(',',':')).encode();h.update(len(meta).to_bytes(8,'little'));h.update(meta);h.update(x.numpy().tobytes(order='C'))
 return h.hexdigest()
def seeded_model(seed):
 set_deterministic_seed(seed);before=rng_state();model=ECGResNetSmall();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};digest=state_digest(state);after=rng_state()
 return model,state,digest,before,after
def atomic_torch_save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8');os.replace(t,p)
def atomic_npz(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.stem+'.tmp.npz');np.savez_compressed(t,**x);os.replace(t,p)
def atomic_csv(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(t,p)
def log(stage,**x):
 q={'stage':stage,'utc':now(),**x};atomic_json(OUT/'progress.json',q);print(json.dumps(q,ensure_ascii=False),flush=True)
def protocol():
 p=json.loads((OUT/'FROZEN_EXECUTION_PROTOCOL.json').read_text(encoding='utf8'))
 if any(not (OUT/name).is_file() or sha(OUT/name)!=digest for name,digest in p['runner_sha256'].items()) or (OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').read_text(encoding='ascii').strip()!=sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json'):raise RuntimeError('frozen protocol/runner mismatch')
 if sha(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv')!=p['frozen_membership_copy_sha256'] or sha(OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')!=p['frozen_split_copy_sha256']:raise RuntimeError('frozen role/split snapshot mismatch')
 for path,digest in p['input_file_sha256'].items():
  if sha(ROOT/path)!=digest:raise RuntimeError('frozen input changed: '+path)
 return p
def records():
 can=json.loads((N/'CANONICAL_RECORDS.json').read_text(encoding='utf8'))['records'];by={r['record_id']:r for r in can}
 accepted=list(csv.DictReader((OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv').open(encoding='utf-8-sig')))
 split=json.loads((OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json').read_text(encoding='utf8'))['rows']
 tr=[by[x['record_id']] for x in split if x['split']=='train'];va=[by[x['record_id']] for x in split if x['split']=='validation'];te=[by[x['record_id']] for x in split if x['split']=='test']
 roles={r:[] for r in ('model_selection','calibration_A','threshold_B')}
 for m in accepted:
  if m['role'] not in roles:raise RuntimeError('unexpected membership role')
  r=by[m['record_id']]
  if r['labels']!=[int(v) for v in m['labels_TWA_SB_STach']]:raise RuntimeError('membership label mismatch')
  if r['waveform_sha256']!=m['waveform_sha256']:raise RuntimeError('membership waveform mismatch')
  roles[m['role']].append(r)
 if any(m['source_split']!='validation' for m in accepted) or {r['record_id'] for v in roles.values() for r in v}!={r['record_id'] for r in va}:raise RuntimeError('accepted roles are not an exact partition of prior validation')
 if (len(tr),*[len(roles[k]) for k in roles],len(te))!=(24430,1744,872,872,6981):raise RuntimeError('split size mismatch')
 ids={k:{r['record_id'] for r in v} for k,v in {'train':tr,**roles,'test':te}.items()}
 if any(ids[a]&ids[b] for a,b in itertools.combinations(ids,2)):raise RuntimeError('role ID overlap')
 wh={k:{r['waveform_sha256'] for r in v} for k,v in {'train':tr,**roles,'test':te}.items()}
 if any(wh[a]&wh[b] for a,b in itertools.combinations(wh,2)):raise RuntimeError('role waveform overlap')
 return {'train':tr,**roles,'test':te},accepted
def canonical_records():
 c,r=records();return c
def raw_array(rs): return np.stack([data.raw(r) for r in rs]).astype(np.float32)
def labels(rs): return np.asarray([r['labels'] for r in rs],dtype=np.uint8)
def determin_u(method,seed,epoch,index,kind):
 # Keep historical arm keys and key encoding from the frozen M2/four-arm loops.
 arm={'ordinary':'ordinary','physics_aware_p050':'physics_aware','combined':'combined'}[method]
 return int.from_bytes(hashlib.sha256(f'{arm}|{seed}|{epoch}|{index}|{kind}'.encode()).digest()[:8],'little')/2**64
def reverse_raw(x,fault): return derive_display(named_reversal(np.concatenate((x[:2],x[6:]),axis=0),fault)).astype(np.float32)
MASKS=tuple(itertools.combinations(range(12),5))
def augment(x,mean,method,seed,epoch,index):
 if method=='ordinary':return x
 arm='physics_aware' if method=='physics_aware_p050' else 'combined';u=determin_u(method,seed,epoch,index,'choice')
 if u<.5:return x
 if method=='combined' and u<.75:
  j=min(791,int(determin_u(method,seed,epoch,index,'mask')*792));return recovery.apply_mask_raw(x,mean,MASKS[j]).astype(np.float32)
 j=min(2,int(determin_u(method,seed,epoch,index,'fault')*3));return reverse_raw(x,FAULTS[j])
class TrainDS(Dataset):
 def __init__(self,rs,mean,std,method,seed):self.rs,self.mean,self.std,self.method,self.seed,self.epoch=rs,mean,std,method,seed,1
 def __len__(self):return len(self.rs)
 def __getitem__(self,i):
  r=self.rs[i];x=augment(data.raw(r).astype(np.float32),self.mean,self.method,self.seed,self.epoch,i)
  return torch.from_numpy(recovery.normalize_raw(x,self.mean,self.std)),torch.tensor(r['labels'],dtype=torch.float32)
class EvalDS(Dataset):
 def __init__(self,x,y,mean,std):self.x,self.y,self.mean,self.std=x,y,mean,std
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return torch.from_numpy(recovery.normalize_raw(self.x[i],self.mean,self.std)),torch.from_numpy(self.y[i].astype(np.float32))
def loader(ds,train=False,generator=None):return DataLoader(ds,batch_size=64 if train else 128,shuffle=train,num_workers=2,pin_memory=True,persistent_workers=False,generator=generator)
def infer(model,x,mean,std):
 model.eval();out=[]
 g=torch.Generator(device='cpu');g.manual_seed(0xEC600001)
 with torch.inference_mode():
  for xb,_ in loader(EvalDS(x,np.zeros((len(x),3),np.uint8),mean,std),generator=g):
   with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(xb.cuda(non_blocking=True))).float().cpu().numpy())
 return np.concatenate(out).astype(np.float32)
def train_one(method,seed,parts,prot):
 d=OUT/'fits'/method/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);gate=d/'training_gate.json'
 if gate.exists():
  q=json.loads(gate.read_text(encoding='utf8'))
  if q.get('decision') not in ('PASS','FAIL'):raise RuntimeError('invalid training gate')
  if q.get('runner_sha256')!=prot['runner_sha256'] or q.get('membership_sha256')!=prot['membership_file_sha256']:raise RuntimeError('fit reuse binding changed')
  return q
 pre=N/f'seed_{seed}'/'preprocessing.npz'
 all_stats=[N/f'seed_{s}'/'preprocessing.npz' for s in SEEDS]+[ROOT/'outputs/mitigation_m2_ningbo_v1'/f'seed_{s}'/'preprocessing.npz' for s in SEEDS]
 if len({sha(x) for x in all_stats})!=1:raise RuntimeError('frozen train-derived preprocessing files are not byte-identical')
 with np.load(pre,allow_pickle=False) as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 shutil.copy2(pre,d/'preprocessing.npz')
 ytr=labels(parts['train']);pw=np.minimum((len(ytr)-ytr.sum(0))/np.maximum(ytr.sum(0),1),20).astype(np.float32)
 model,initial_state,initial_hash,rng_before_init,rng_after_init=seeded_model(seed)
 initfile=d/'initialization_state.pt';initaudit=d/'INITIALIZATION_AUDIT.json'
 if initfile.exists():
  saved=torch.load(initfile,weights_only=False,map_location='cpu')
  if saved.get('initial_state_sha256')!=initial_hash:raise RuntimeError('initial state mismatch on restart')
 else:atomic_torch_save(initfile,{'initial_state':initial_state,'rng_before_init':rng_before_init,'rng_after_init':rng_after_init,'initial_state_sha256':initial_hash})
 model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda');lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device='cuda'))
 resume=d/'resume.pt';bestfile=d/'best.pt';hist=[];best=-1.;wait=0;start=1;elapsed_before=0.
 ds=TrainDS(parts['train'],mean,std,method,seed);xsel=raw_array(parts['model_selection']);ysel=labels(parts['model_selection'])
 if not len(xsel):raise RuntimeError('selection set empty')
 train_generator=torch.Generator(device='cpu');train_generator.manual_seed(seed);train_loader=loader(ds,True,generator=train_generator)
 if not initaudit.exists():
  before_training=rng_state();saved=torch.load(initfile,weights_only=False,map_location='cpu');saved['rng_before_training']=before_training;atomic_torch_save(initfile,saved)
  audit={'method':method,'seed':seed,'initial_state_sha256':initial_hash,'rng_before_init_sha256':rng_digest(rng_before_init),'rng_after_init_sha256':rng_digest(rng_after_init),'rng_before_training_sha256':rng_digest(before_training),'python_version':platform.python_version(),'numpy_version':np.__version__,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),'determinism':{'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32},'rng_state_file_sha256':sha(initfile),'before_training_state_saved_in_initialization_state':True}
  atomic_json(initaudit,audit)
 if resume.exists():
  q=torch.load(resume,weights_only=False,map_location='cpu')
  for k,v in [('runner_sha256',prot['runner_sha256']),('membership_sha256',prot['membership_file_sha256']),('method',method),('seed',seed)]:
   if q.get(k)!=v:raise RuntimeError('resume binding mismatch '+k)
  model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);scaler.load_state_dict(q['scaler']);hist,best,wait,start=q['history'],q['best'],q['wait'],q['next_epoch'];elapsed_before=float(q.get('elapsed_training_seconds',0.))
  if q.get('initial_state_sha256')!=initial_hash:raise RuntimeError('resume initialization binding mismatch')
  restore_rng_state(q['rng_state']);train_generator.set_state(q['train_loader_generator'])
 train_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start,31):
  ds.epoch=epoch;model.train();losses=[]
  for xb,yb in train_loader:
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(model(xb.cuda(non_blocking=True)),yb.cuda(non_blocking=True))
   if not torch.isfinite(loss):raise RuntimeError('nonfinite train loss')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
  ps=infer(model,xsel,mean,std);ap=float(average_precision_score(ysel[:,0],ps[:,0]));auc=float(roc_auc_score(ysel[:,0],ps[:,0]))
  hist.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'selection_twa_auprc':ap,'selection_twa_auroc':auc,'selection_role':'model_selection','selection_ids_sha256':hobj([r['record_id'] for r in parts['model_selection']])})
  if ap>best:
   best=ap;wait=0;torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'selection_twa_auprc':ap,'selection_twa_auroc':auc,'method':method,'seed':seed,'runner_sha256':prot['runner_sha256'],'membership_sha256':prot['membership_file_sha256']},bestfile)
  else:wait+=1
  elapsed=elapsed_before+time.monotonic()-train_started
  atomic_torch_save(resume,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'history':hist,'best':best,'wait':wait,'next_epoch':epoch+1,'method':method,'seed':seed,'runner_sha256':prot['runner_sha256'],'membership_sha256':prot['membership_file_sha256'],'initial_state_sha256':initial_hash,'rng_state':rng_state(),'train_loader_generator':train_generator.get_state(),'elapsed_training_seconds':elapsed})
  atomic_json(d/'history.json',{'method':method,'seed':seed,'rows':hist});log('TRAINING',method=method,seed=seed,epoch=epoch,best_epoch=int(max(hist,key=lambda z:z['selection_twa_auprc'])['epoch']),best_auprc=best,patience=wait,training_seconds=elapsed)
  if wait>=5:break
 training_seconds=elapsed_before+time.monotonic()-train_started;training_peak_bytes=torch.cuda.max_memory_allocated();bestq=torch.load(bestfile,weights_only=True,map_location='cpu');bestmodel=ECGResNetSmall().cuda();bestmodel.load_state_dict(bestq['state_dict']);selpred=infer(bestmodel,xsel,mean,std)
 replay_ap=float(average_precision_score(ysel[:,0],selpred[:,0]));replay_auc=float(roc_auc_score(ysel[:,0],selpred[:,0]));gatepass=(replay_ap>=.45 and replay_auc>=.75 and np.isfinite(selpred).all() and np.isfinite([z['train_loss'] for z in hist]).all() and bestq['epoch']==max(hist,key=lambda z:z['selection_twa_auprc'])['epoch'] and abs(replay_ap-bestq['selection_twa_auprc'])<1e-7)
 # Canonical checkpoint is a copy of the strictly selected best state, never the last epoch.
 ck=d/'checkpoint.pt';torch.save(bestq,ck)
 res={'method':method,'seed':seed,'decision':'PASS' if gatepass else 'FAIL','best_epoch':bestq['epoch'],'best_selection_twa_auprc':bestq['selection_twa_auprc'],'best_selection_twa_auroc':bestq['selection_twa_auroc'],'replay_selection_twa_auprc':replay_ap,'replay_selection_twa_auroc':replay_auc,'checkpoint_sha256':sha(ck),'best_sha256':sha(bestfile),'history_sha256':sha(d/'history.json'),'preprocessing_sha256':sha(d/'preprocessing.npz'),'membership_sha256':prot['membership_file_sha256'],'runner_sha256':prot['runner_sha256'],'initial_state_sha256':initial_hash,'elapsed_seconds':training_seconds,'training_peak_allocated_bytes':training_peak_bytes,'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'finite':bool(np.isfinite(selpred).all() and np.isfinite([z['train_loss'] for z in hist]).all()),'selection_rows':len(ysel),'selection_role':'model_selection'}
 atomic_json(gate,res);return res
def lock_all(parts,prot):
 out=[]
 for m in METHODS:
  for s in SEEDS:
   d=OUT/'fits'/m/f'seed_{s}';g=json.loads((d/'training_gate.json').read_text(encoding='utf8'))
   if g['decision']!='PASS':return None
   ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');hist=json.loads((d/'history.json').read_text(encoding='utf8'))['rows'];best=max(hist,key=lambda z:z['selection_twa_auprc'])
   if ck['epoch']!=best['epoch'] or sha(d/'checkpoint.pt')!=g['checkpoint_sha256']:raise RuntimeError('best epoch lock mismatch')
   out.append({**g,'history_best_epoch':best['epoch'],'history_best_selection_auprc':best['selection_twa_auprc']})
 atomic_json(OUT/'ALL_MODELS_LOCK.json',{'decision':'PASS','fits':out,'locked_utc':now(),'no_test_or_AB_before_lock':True,'runner_sha256':prot['runner_sha256']});return out
def engineering_smoke():
 parts,mem=records();pre=N/'seed_104729'/'preprocessing.npz'
 with np.load(pre,allow_pickle=False) as z:mean,std=z['mean'].astype(np.float32),z['std'].astype(np.float32)
 # Distinct role views: one gradient minibatch can only see train; scoring can only see model_selection.
 xt=raw_array(parts['train'][:2]);yt=labels(parts['train'][:2]);xs=raw_array(parts['model_selection'][:2]);ys=labels(parts['model_selection'][:2])
 mask=recovery.apply_mask_raw(xt[0],mean,(0,1,2,3,5));changed=np.flatnonzero(np.any(mask!=xt[0],axis=1))
 if not np.array_equal(changed,np.array([0,1,2,3,5])):raise RuntimeError('mask raw transform does not remove exactly five leads')
 rev=reverse_raw(xt[0],'RA_LL_reversal')
 if rev.shape!=xt[0].shape or not np.isfinite(rev).all():raise RuntimeError('physical reversal transform invalid')
 m,_,_,_,_=seeded_model(99173);m=m.cuda();o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.ones(3,device='cuda'))
 tr=recovery.normalize_raw(xt,mean,std);o.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(m(torch.from_numpy(tr).cuda()),torch.from_numpy(yt.astype(np.float32)).cuda())
 loss.backward();o.step();ps=infer(m,xs,mean,std)
 ok=bool(torch.isfinite(loss).item() and np.isfinite(ps).all() and len(parts['model_selection'])==1744)
 audit={'status':'PASS' if ok else 'FAIL','device':torch.cuda.get_device_name(0),'train_rows_in_smoke':len(xt),'selection_rows_in_smoke':len(xs),'calibration_A_loaded':False,'threshold_B_loaded':False,'test_loaded':False,'train_gradient_finite':bool(torch.isfinite(loss).item()),'selection_predictions_finite':bool(np.isfinite(ps).all()),'selection_predictions_shape':list(ps.shape),'mask_removed_indices':changed.tolist(),'physical_transform_shape':list(rev.shape),'membership_counts':{k:len(v) for k,v in parts.items()},'membership_file_sha256':sha(PREF/'PROPOSED_MEMBERSHIP.csv'),'completed_utc':now()}
 atomic_json(OUT/'ENGINEERING_SMOKE_AUDIT.json',audit)
 if not ok:raise RuntimeError('engineering smoke failed')
 return audit
def platt_fit(y,p):
 from scipy.optimize import minimize
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6);x=np.log(p/(1-p));y=np.asarray(y,dtype=np.float64)
 def fun(t):
  z=np.clip(np.exp(t[0])*x+t[1],-700,700);q=1/(1+np.exp(-z));loss=-np.mean(y*np.log(np.clip(q,1e-15,1))+(1-y)*np.log(np.clip(1-q,1e-15,1)));return loss
 r=minimize(fun,np.array([0.,0.]),method='L-BFGS-B',bounds=[(-12,12),(-30,30)],options={'maxiter':2000,'ftol':1e-13,'gtol':1e-8,'maxls':50})
 return {'theta0':float(r.x[0]),'b':float(r.x[1]),'a':float(np.exp(r.x[0])),'success':bool(r.success),'message':str(r.message),'loss':float(r.fun),'n_iter':int(r.nit)}
def calibrate_apply(p,f):
 p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6);z=np.log(p/(1-p))*f['a']+f['b'];return (1/(1+np.exp(-np.clip(z,-700,700)))).astype(np.float64)
def run():
 prot=protocol();parts,mem=records();fits=[]
 if json.loads((OUT/'INITIALIZATION_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':raise RuntimeError('initialization audit must PASS before training')
 if json.loads((OUT/'ENGINEERING_SMOKE_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':raise RuntimeError('deterministic engineering smoke must PASS before training')
 if json.loads((OUT/'HISTORICAL_INTEGRITY_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':raise RuntimeError('historical artifacts must PASS integrity audit before training')
 cache_audit=json.loads((OUT/'DETECTOR_CACHE_REUSE_AUDIT.json').read_text(encoding='utf8'))
 if cache_audit.get('status')!='PASS':raise RuntimeError('verified unchanged detector caches are required')
 for item in cache_audit['reused_detector_caches']:
  if item['role']!='HEEDB_detector' and sha(ROOT/item['reused_copy'])!=item['sha256']:raise RuntimeError('reused detector cache changed after verification')
 for m in METHODS:
  for s in SEEDS:
   q=train_one(m,s,parts,prot);fits.append(q);atomic_json(OUT/'TRAINING_STATUS.json',{'fits':fits,'completed':sum(x['decision'] in ('PASS','FAIL') for x in fits),'total':9})
 if any(x['decision']!='PASS' for x in fits):atomic_json(OUT/'STATUS.json',{'status':'STOP_GATE_FAILURE','failures':[x for x in fits if x['decision']!='PASS']});return
 lock=lock_all(parts,prot)
 if lock is None:atomic_json(OUT/'STATUS.json',{'status':'STOP_GATE_FAILURE'});return
 # Fit is locked before any external test cohort is opened.
 roles=('calibration_A','threshold_B');rolepred={};
 for m in METHODS:
  for s in SEEDS:
   d=OUT/'fits'/m/f'seed_{s}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict']);mean,std=np.load(d/'preprocessing.npz')['mean'],np.load(d/'preprocessing.npz')['std']
   for role in roles:
    xs=raw_array(parts[role]);ys=labels(parts[role]);pp=infer(model,xs,mean,std);rolepred[m,s,role]=(pp,ys)
    atomic_npz(OUT/f'scores/{role}/{m}/seed_{s}.npz',p=pp,y=ys,record_ids=np.asarray([r['record_id'] for r in parts[role]]),role=np.asarray(role),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),membership_sha256=np.asarray(prot['membership_file_sha256']))
   del model;torch.cuda.empty_cache()
 # Huang role scores use frozen detector for each simulated condition and corrected incoming waveform.
 huang_scores={}
 # Condition-specific probabilities were independently hash/order-verified and are reused unchanged.
 if any(not (OUT/f'detector/{role}.npz').is_file() for role in roles):raise RuntimeError('reused role detector scores missing')
 for s in SEEDS:
  m='ordinary';d=OUT/'fits'/m/f'seed_{s}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict']);mean,std=np.load(d/'preprocessing.npz')['mean'],np.load(d/'preprocessing.npz')['std']
  for role in roles:
   raw=raw_array(parts[role]);preds=[]
   for ci,condition in enumerate(('clean',*FAULTS)):
    incoming=raw if condition=='clean' else np.stack([reverse_raw(x,condition) for x in raw])
    det=np.load(OUT/f'detector/{role}.npz',allow_pickle=False)['probabilities'][ci];classes=np.argmax(det,axis=1).astype(np.int64)
    fixed=incoming if condition=='clean' else corrected_batch(incoming,classes)
    preds.append(infer(model,fixed,mean,std))
   hp=np.stack(preds);ys=labels(parts[role]);huang_scores[s,role]=(hp,ys)
   atomic_npz(OUT/f'scores/{role}/Huang_full_pipeline/seed_{s}.npz',p=hp,y=ys,record_ids=np.asarray([r['record_id'] for r in parts[role]]),role=np.asarray(role),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),detector_binding_sha256=np.asarray(sha(OUT/f'detector/{role}.npz')))
  del model;torch.cuda.empty_cache()
 # Per endpoint calibration and B-negative raw threshold.
 cals={};lockrows=[]
 for method in (*METHODS,'Huang_full_pipeline'):
  for seed in SEEDS:
   for ei,ep in enumerate(ENDS):
    if method=='Huang_full_pipeline':pa,ya=huang_scores[seed,'calibration_A'];pb,yb=huang_scores[seed,'threshold_B'];a=pa[0,:,ei];b=pb[0,:,ei]
    else:pa,ya=rolepred[method,seed,'calibration_A'];pb,yb=rolepred[method,seed,'threshold_B'];a=pa[:,ei];b=pb[:,ei]
    fit=platt_fit(ya[:,ei],a);neg=b[yb[:,ei]==0]
    if not len(neg) or len(np.unique(ya[:,ei]))!=2 or not fit['success'] or not np.isfinite([fit['a'],fit['b']]).all():
     lockrows.append({'method':method,'seed':seed,'endpoint':ep,'condition':'clean','status':'FAIL','n_negative_B':len(neg),**fit});continue
    q=float(np.quantile(neg,.9,method='linear'));mapped=float(calibrate_apply(np.array([q]),fit)[0]);cals[method,seed,ep]=(fit,q,mapped)
    lockrows.append({'method':method,'seed':seed,'endpoint':ep,'condition':'clean','status':'PASS','n_A':len(a),'n_B_negative':len(neg),'raw_q90':q,'mapped_calibrated_threshold':mapped,'slope_a':fit['a'],'intercept_b':fit['b'],'fit_success':fit['success'],'fit_loss':fit['loss'],'fit_iterations':fit['n_iter']})
 atomic_csv(OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv',lockrows)
 if len([r for r in lockrows if r['status']=='PASS'])!=36:atomic_json(OUT/'STATUS.json',{'status':'STOP_CALIBRATION_FAILURE','rows':len(lockrows)});return
 atomic_json(OUT/'CALIBRATION_AND_THRESHOLD_LOCK.json',{'status':'LOCKED','entries':lockrows,'fitted_on':'calibration_A','threshold_negatives':'threshold_B','locked_before_test':True,'quantile':'numpy linear','raw_decision':'>= q90','lock_hash':hobj(lockrows)})
 # TEST/HEEDB scoring only after all locks. Continued in evaluation command stage.
 atomic_json(OUT/'STATUS.json',{'status':'LOCKS_COMPLETE_READY_TEST','note':'evaluation stage begins only after calibration locks','utc':now()})
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='run':run()
 elif len(sys.argv)>1 and sys.argv[1]=='smoke':print(json.dumps(engineering_smoke(),ensure_ascii=False))
 elif len(sys.argv)>1 and sys.argv[1]=='init-probe':
  method=sys.argv[2];seed=int(sys.argv[3]);ambient=int(sys.argv[4]);random.seed(ambient);np.random.seed(ambient%(2**32));torch.manual_seed(ambient);torch.cuda.manual_seed_all(ambient)
  _,_,digest,before,after=seeded_model(seed);print(json.dumps({'method':method,'seed':seed,'ambient_state_tag':ambient,'initial_state_sha256':digest,'rng_before_init_sha256':rng_digest(before),'rng_after_init_sha256':rng_digest(after),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0)},sort_keys=True))
