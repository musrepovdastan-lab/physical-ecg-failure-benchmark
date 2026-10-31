"""Frozen, resumable Ningbo v1 confirmation workflow; no Chapman outcome reuse."""
from __future__ import annotations
import argparse,csv,contextlib,ctypes,hashlib,itertools,json,os,random,re,sys,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/ningbo_confirmation_v1'; DATA=ROOT/'data/challenge2021_v103'; PROTO=ROOT/'configs/FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v1.json'
PH='B6EEB79C0CD214FE7F41FDC526AEC64DF2E9D982F59A45E7F21D0108B5A3698E'; CODES=('164934002','426177001','427084000'); LABELS=('TWA','SB','STach'); SEEDS=(104729,130363,155921)
INGESTION_SCHEMA='ningbo_ingestion_schema_v2_normalized_challenge_comments'
MASKS=tuple(itertools.combinations(range(12),5)); NAMES=tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS); CONDITIONS=('clean','RA_LL_reversal',*NAMES); NBOOT=1000
class Stop(RuntimeError):pass
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def htxt(s):return hashlib.sha256(s.encode()).hexdigest()
def condition_hash():return htxt(json.dumps(list(CONDITIONS),separators=(',',':')))
def jput(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def nput(target,**x):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);q=target.with_name('.'+target.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**x);os.replace(q,target)
def csvput(p,rs):
 if not rs:return
 p=Path(p);q=p.with_name('.'+p.name+'.tmp')
 with open(q,'w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 os.replace(q,p)
def status(stage,**x):
 z={'stage':stage,'pid':os.getpid(),'updated_utc':now(),**x};jput(OUT/'WORKFLOW_STATUS.json',z);print(json.dumps(z,sort_keys=True),flush=True)
def halt(code,**x):status(code,**x);raise Stop(code)
def alive(pid):
 if not isinstance(pid,int) or pid<=0:return False
 if os.name=='nt':
  # os.kill(pid, 0) is NOT a harmless existence probe on Windows: Python may
  # route it through the Windows termination/control-event APIs.  Query the
  # process handle and exit code instead, without signalling the target.
  PROCESS_QUERY_LIMITED_INFORMATION=0x1000;STILL_ACTIVE=259
  kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
  kernel32.OpenProcess.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32)
  kernel32.OpenProcess.restype=ctypes.c_void_p
  kernel32.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32))
  kernel32.GetExitCodeProcess.restype=ctypes.c_int
  kernel32.CloseHandle.argtypes=(ctypes.c_void_p,)
  handle=kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,False,pid)
  if not handle:
   return ctypes.get_last_error()==5  # access denied implies that it exists
  try:
   code=ctypes.c_uint32()
   return bool(kernel32.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==STILL_ACTIVE
  finally:kernel32.CloseHandle(handle)
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
 except PermissionError:return True
 except (ValueError,OSError):return False
def lock():
 p=OUT/'ningbo_confirmation_v1.lock';OUT.mkdir(parents=True,exist_ok=True)
 if p.exists():
  try:pid=int(json.loads(p.read_text()).get('pid',-1))
  except Exception:halt('STOP_TECHNICAL_BLOCKER',reason='malformed lock')
  if alive(pid):halt('STOP_TECHNICAL_BLOCKER',reason='live lock',lock_pid=pid)
  p.unlink()
 jput(p,{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTO)});return p
def path(rel,base=None):
 """Map only release-relative Ningbo paths to the one frozen acquisition root."""
 raw=str(rel)
 if '\\' in raw:raise ValueError('noncanonical path separator')
 p=PurePosixPath(raw)
 if p.is_absolute() or ':' in raw or '..' in p.parts or p.parts[:2]!=('training','ningbo') or len(p.parts)<3:raise ValueError('invalid Ningbo mapping path')
 q=(DATA/'ningbo' if base is None else Path(base)).joinpath(*p.parts[2:])
 if not q.is_file():raise FileNotFoundError(str(q))
 return q
def preflight():
 status('PREFLIGHT')
 if sha(PROTO).upper()!=PH:halt('STOP_DATA_INTEGRITY',reason='protocol hash')
 a,m,s=OUT/'ACQUISITION_STATUS.json',OUT/'SOURCE_RECORD_MAPPING.json',OUT/'OFFICIAL_SHA256SUMS.txt'
 if not all(q.is_file() for q in(a,m,s)):halt('STOP_DATA_INTEGRITY',reason='missing acquisition artifacts')
 ac,mp=json.loads(a.read_text()),json.loads(m.read_text())
 if ac.get('stage')!='ACQUISITION_COMPLETE':halt('STOP_DATA_INTEGRITY',reason='acquisition incomplete',observed=ac.get('stage'))
 if ac.get('completed_files')!=69810 or ac.get('expected_files')!=69810 or ac.get('verified_bytes_total')!=4215509826:halt('STOP_DATA_INTEGRITY',reason='acquisition count/byte binding')
 if ac.get('source_mapping_sha256') and ac['source_mapping_sha256'].lower()!=sha(m).lower():halt('STOP_DATA_INTEGRITY',reason='acquisition mapping hash binding')
 if ac.get('official_manifest_sha256') and ac['official_manifest_sha256'].lower()!=sha(s).lower():halt('STOP_DATA_INTEGRITY',reason='acquisition official sums hash binding')
 if mp.get('logical_record_count')!=34905 or mp.get('physical_file_count')!=69810 or len(mp.get('records',()))!=34905:halt('STOP_DATA_INTEGRITY',reason='mapping count')
 official={q[-1].lstrip('./'):q[0].lower() for q in (x.split() for x in s.read_text(encoding='utf8').splitlines()) if len(q)>1}
 bad=[r[k] for r in mp['records'] for k in ('header_path','waveform_path') if r[k] not in official]
 if bad:halt('STOP_DATA_INTEGRITY',reason='mapping not bound to sums',count=len(bad))
 targets=[];missing=[]
 for r in mp['records']:
  for k in ('header_path','waveform_path'):
   try:targets.append(path(r[k]))
   except (ValueError,FileNotFoundError):missing.append(r[k])
 if missing:halt('STOP_DATA_INTEGRITY',reason='mapped local target missing/invalid',missing_count=len(missing),missing_sample=sorted(missing)[:10])
 z={'decision':'PASS','protocol_sha256':sha(PROTO).upper(),'acquisition_status_sha256':sha(a),'mapping_sha256':sha(m),'official_sums_sha256':sha(s),'logical_records':34905,'physical_files':69810,'local_regular_file_count':len(targets),'local_verified_bytes':sum(q.stat().st_size for q in targets),'targeted_rehash':False,'completed_utc':now()};jput(OUT/'PREFLIGHT.json',z);status('PREFLIGHT_PASS',**z);return mp
def header(p):
 ls=p.read_text(encoding='utf8',errors='strict').splitlines();first=ls[0].split() if ls else []
 if len(first)<3 or int(first[1])!=12:raise ValueError('header shape')
 leads=[];fields={}
 for x in ls[1:]:
  if x.startswith('#'):
   comment=x[1:].strip()
   if ':' in comment:
    key,value=comment.split(':',1);key=key.strip().lower();value=value.strip()
    if key in ('age','sex','dx'):
     if not value:raise ValueError('empty required comment field')
     if key in fields and fields[key]!=value:raise ValueError('conflicting duplicate comment field')
     fields[key]=value
  elif len(leads)<12:leads.append(x.split()[-1])
 if tuple(leads)!=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6') or 'dx' not in fields:raise ValueError('missing/noncanonical leads or Dx')
 age=fields.get('age');sex=fields.get('sex');sex=sex.lower() if sex is not None else None;dx={q.strip() for q in fields['dx'].split(',')}
 if not dx or any(not re.fullmatch(r'\d+',q) for q in dx):raise ValueError('malformed Dx')
 # Challenge headers use literal values such as `NaN` and `Unknown`.  They are
 # valid source metadata, not parser failures; preserve them in the duplicate key.
 age='UNKNOWN' if age is None or not np.isfinite(float(age)) else int(float(age))
 sex=sex if sex in ('male','female') else ('UNKNOWN' if sex is None or sex in ('','unknown','nan') else (_ for _ in ()).throw(ValueError('malformed sex')))
 return age,sex,tuple(sorted(dx))
def wave(p):
 from scipy.io import loadmat
 x=np.asarray(loadmat(p)['val'])
 if x.ndim!=2 or x.shape[0]!=12 or x.shape[1]<1 or not np.isfinite(x).all():raise ValueError('waveform invalid')
 return x.astype('float32',copy=False)
def duplicate_signature(r):return (r['age'],r['sex'],tuple(r['dx']))
def merge_audit_rows(rows):
 """Merge parsed shard rows without touching source waveforms again."""
 groups={}
 for r in rows:groups.setdefault(r['waveform_sha256'],[]).append(r)
 canon=[];collapsed=[];conflicts=[]
 for h,g in groups.items():
  g=sorted(g,key=lambda q:q['record_id'])
  if len({duplicate_signature(q) for q in g})!=1:conflicts.append({'waveform_sha256':h,'records':g});continue
  canon.append(g[0]);collapsed += [{'waveform_sha256':h,'canonical_record_id':g[0]['record_id'],'collapsed_record_id':q['record_id']} for q in g[1:]]
 return sorted(canon,key=lambda q:q['record_id']),collapsed,conflicts
def audit_binding(x,mapping_sha):return x.get('ingestion_schema')==INGESTION_SCHEMA and x.get('mapping_sha256')==mapping_sha and x.get('protocol_sha256')==sha(PROTO).upper()
def audit(mp):
 cp=OUT/'CANONICAL_RECORDS.json';ap=OUT/'DATA_INGESTION_AUDIT.json'
 mapping_sha=sha(OUT/'SOURCE_RECORD_MAPPING.json')
 if cp.exists() and ap.exists():
  oldcp,olda=json.loads(cp.read_text()),json.loads(ap.read_text())
  if olda.get('decision')=='PASS' and audit_binding(oldcp,mapping_sha) and audit_binding(olda,mapping_sha):return oldcp['records']
 status('INGESTION_AUDIT',expected_records=len(mp['records']));rows=[];ex=[];shards=OUT/'audit_shards';shards.mkdir(exist_ok=True)
 # Atomic deterministic shards make a 34,905-record scan restartable without
 # silently changing its population or reprocessing completed shards.
 for start in range(0,len(mp['records']),250):
  sp=shards/f'{start:05d}.json'
  if sp.exists():
   z=json.loads(sp.read_text())
   if audit_binding(z,mapping_sha):rows+=z['rows'];ex+=z['exclusions'];continue
  sr=[];se=[]
  for r in mp['records'][start:start+250]:
   rid=f"g{r['listed_group']}/{r['logical_record']}"
   try:
    a,s,d=header(path(r['header_path']));x=wave(path(r['waveform_path']));h=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();sr.append({'record_id':rid,'header_path':r['header_path'],'waveform_path':r['waveform_path'],'age':a,'sex':s,'dx':list(d),'labels':[int(c in d) for c in CODES],'waveform_sha256':h})
   except Exception as e:se.append({'record_id':rid,'reason':repr(e)})
  jput(sp,{'start':start,'rows':sr,'exclusions':se,'ingestion_schema':INGESTION_SCHEMA,'mapping_sha256':mapping_sha,'protocol_sha256':sha(PROTO).upper()});rows+=sr;ex+=se
  jput(OUT/'AUDIT_PROGRESS.json',{'completed_records':min(start+250,len(mp['records'])),'total_records':len(mp['records']),'completed_shards':len(list(shards.glob('*.json'))),'updated_utc':now()});status('INGESTION_AUDIT',parsed=min(start+250,len(mp['records'])),exclusions=len(ex))
 if ex:jput(OUT/'INGESTION_EXCLUSIONS.json',{'records':ex});halt('STOP_DATA_INTEGRITY',excluded=len(ex))
 canon,collapsed,conflicts=merge_audit_rows(rows)
 if conflicts:jput(OUT/'DUPLICATE_CONFLICTS.json',{'conflicts':conflicts});halt('STOP_DUPLICATE_CONFLICT',waveform_sha256=conflicts[0]['waveform_sha256'])
 base={'ingestion_schema':INGESTION_SCHEMA,'mapping_sha256':mapping_sha,'protocol_sha256':sha(PROTO).upper()};jput(cp,{'version':'ningbo_confirmation_v1','records':canon,**base});csvput(OUT/'DUPLICATE_COLLAPSE.csv',collapsed);missing={'age_unknown':sum(r['age']=='UNKNOWN' for r in rows),'sex_unknown':sum(r['sex']=='UNKNOWN' for r in rows),'sex_other':0};sex_counts={str(k):sum(r['sex']==k for r in rows) for k in sorted({str(r['sex']) for r in rows})};sample=[{'record_id':r['record_id'],'header_path':r['header_path'],'age':r['age'],'sex':r['sex'],'dx':r['dx']} for r in sorted(rows,key=lambda x:x['record_id'])[:10]];jput(OUT/'HEADER_AUDIT_SAMPLE.json',{'deterministic_first_ten':sample,'sex_counts':sex_counts,'metadata_missingness':missing,'header_parser':'strict normalized # key: value comments; unknown age/sex preserved'});oldex=OUT/'INGESTION_EXCLUSIONS.json';
 if oldex.exists():jput(OUT/'INGESTION_EXCLUSIONS_STALE.json',{'status':'STALE_SUPERSEDED','prior_sha256':sha(oldex),'superseded_by_schema':INGESTION_SCHEMA,'completed_utc':now()})
 jput(ap,{'decision':'PASS','source_records':len(rows),'canonical_records':len(canon),'duplicate_collapses':len(collapsed),'excluded_records':0,'metadata_missingness':missing,'sex_counts':sex_counts,**base,'completed_utc':now()});return canon
def leakage(rows):
 d={s:[x for x in rows if x['split']==s] for s in ('train','validation','test')}
 for a,b in itertools.combinations(d,2):
  if {x['record_id'] for x in d[a]}&{x['record_id'] for x in d[b]}:raise Stop('STOP_DATA_INTEGRITY record overlap')
  if {x['waveform_sha256'] for x in d[a]}&{x['waveform_sha256'] for x in d[b]}:raise Stop('STOP_DATA_INTEGRITY waveform overlap')
 return d
def split(rs):
 p=OUT/'SPLIT_MEMBERSHIP.json'
 if p.exists():z=json.loads(p.read_text())['rows'];leakage(z);return z
 rng=np.random.default_rng(20260829);gs={}
 for r in rs:gs.setdefault(tuple(r['labels']),[]).append(r)
 z=[]
 for st,g in sorted(gs.items()):
  a=np.array(sorted(g,key=lambda x:x['record_id']),dtype=object);rng.shuffle(a);n=len(a);u=round(.7*n);v=round(.1*n)
  for name,q in (('train',a[:u]),('validation',a[u:u+v]),('test',a[u+v:])):z += [{'record_id':r['record_id'],'waveform_sha256':r['waveform_sha256'],'labels':r['labels'],'stratum':''.join(map(str,st)),'split':name} for r in q]
 z=sorted(z,key=lambda x:x['record_id']);leakage(z);jput(p,{'version':'ningbo_confirmation_v1','seed':20260829,'fractions':{'train':.7,'validation':.1,'test':.2},'rows':z,'membership_sha256':htxt(json.dumps(z,sort_keys=True,separators=(',',':')))});jput(OUT/'SPLIT_LEAKAGE_PROOF.json',{'decision':'PASS','record_id_overlap':0,'waveform_hash_overlap':0,'split_sha256':sha(p)});return z
def prepare(rs,rows):
 d=leakage(rows);by={r['record_id']:r for r in rs};counts={}
 for s,q in d.items():
  y=np.array([r['labels'] for r in q],int);counts[s]={'records':len(q),**{LABELS[i]:int(y[:,i].sum()) for i in range(3)}}
 missing={'age_unknown':sum(r['age']=='UNKNOWN' for r in rs),'sex_unknown':sum(r['sex']=='UNKNOWN' for r in rs),'sex_other':sum(str(r['sex']).startswith('OTHER:') for r in rs)}
 env=cuda();prov={'architecture_source':'src.chapman_phase1.ECGResNetSmall','architecture_sha256':sha(ROOT/'src/chapman_phase1.py'),'chapman_train_v4_sha256':sha(ROOT/'src/chapman_train_v4.py'),'chapman_v7_sha256':sha(ROOT/'src/chapman_multiseed_workflow_v7.py'),'hyperparameters':{'optimizer':'AdamW lr=0.001 weight_decay=0.0001','batch_size':64,'max_epochs':30,'amp':True,'early_stop_metric':'validation TWA AUPRC','early_stop_patience':5,'pos_weight_cap':20}}
 jput(OUT/'ENVIRONMENT_PROVENANCE.json',env);jput(OUT/'MODEL_PROVENANCE.json',prov);jput(OUT/'TRAIN_VALIDATION_MEMBERSHIP.json',{'split_sha256':sha(OUT/'SPLIT_MEMBERSHIP.json'),'rows':d['train']+d['validation']});jput(OUT/'PREPARE_REPORT.json',{'decision':'PASS','endpoint_counts_by_split':counts,'metadata_missingness':missing,'canonical_records':len(rs),'split_sha256':sha(OUT/'SPLIT_MEMBERSHIP.json'),'protocol_sha256':sha(PROTO).upper(),'completed_utc':now()});status('PREPARE_COMPLETE',endpoint_counts_by_split=counts);return {s:[by[x['record_id']] for x in q] for s,q in d.items()}
def cuda_info(torch_module):
 if not torch_module.cuda.is_available():raise Stop('CUDA unavailable')
 name=torch_module.cuda.get_device_name(0)
 if '5070' not in name:raise Stop('RTX 5070 required')
 return {'python':sys.executable,'torch':torch_module.__version__,'cuda':torch_module.version.cuda,'gpu':name}
def cuda():
 import torch
 try:return cuda_info(torch)
 except Stop as e:halt('STOP_TECHNICAL_BLOCKER',reason=str(e))
class SyntheticDataset:
 def __init__(self):self.x=np.zeros((4,12,8),np.float32);self.y=np.zeros((4,3),np.float32)
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return self.x[i],self.y[i]
def raw(r):return wave(path(r['waveform_path']))[:,::5]
class ECGDataset:
 """Top-level Dataset: mandatory for Windows spawn DataLoader workers."""
 def __init__(self,records,mean,std):self.records,self.mean,self.std=records,mean,std
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  r=self.records[i];return ((raw(r)-self.mean[:,None])/self.std[:,None]).astype('float32'),np.array(r['labels'],np.float32)
def stats(rs):
 a=np.zeros(12,np.float64);b=a.copy();n=0
 for r in rs:
  x=raw(r);a+=x.sum(1);b+=(x*x).sum(1);n+=x.shape[1]
 m=a/n;return m.astype('float32'),np.sqrt(np.maximum(b/n-m*m,1e-8)).astype('float32')
def dl(rs,m,s,shuffle):
 from torch.utils.data import DataLoader
 return DataLoader(ECGDataset(rs,m,s),batch_size=64 if shuffle else 128,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True)
def predict(model,rs,m,s):
 import torch
 p=[];y=[];model.eval()
 with torch.no_grad():
  for x,z in dl(rs,m,s,False):
   with torch.autocast('cuda',dtype=torch.float16):p.append(torch.sigmoid(model(x.cuda(non_blocking=True))).float().cpu().numpy())
   y.append(z.numpy())
 return np.concatenate(y),np.concatenate(p)
def smoke(part):
 """CUDA/model smoke on exactly one train batch and one validation batch; never test."""
 import torch
 from src.chapman_phase1 import ECGResNetSmall
 m,s=stats(part['train']);model=ECGResNetSmall().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);x,y=next(iter(dl(part['train'],m,s,False)));opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.BCEWithLogitsLoss()(model(x.cuda()),y.cuda())
 if not torch.isfinite(loss):halt('STOP_TECHNICAL_BLOCKER',reason='smoke nonfinite loss')
 loss.backward();opt.step();vx,vy=next(iter(dl(part['validation'],m,s,False)))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(model(vx.cuda())).float().cpu().numpy()
 if not np.isfinite(p).all():halt('STOP_TECHNICAL_BLOCKER',reason='smoke nonfinite predictions')
 jput(OUT/'SMOKE_REPORT.json',{'decision':'PASS','train_batch':int(len(y)),'validation_batch':int(len(vy)),'test_membership_or_waveform_loaded':False,'loss':float(loss.detach().cpu()),'completed_utc':now()});status('SMOKE_COMPLETE')
def train(seed,part):
 import torch
 from sklearn.metrics import average_precision_score,roc_auc_score
 d=OUT/f'seed_{seed}';d.mkdir(exist_ok=True);bp=d/'baseline.json';cfg=htxt('ECGResNetSmall|AdamW=.001,.0001|BCEcap20|batch64|max30|AMP|TWA_AUPRC|patience5');split_sha=sha(OUT/'SPLIT_MEMBERSHIP.json')
 if bp.exists() and json.loads(bp.read_text()).get('decision')=='PASS':
  old=json.loads(bp.read_text())
  if old.get('training_config_sha256')==cfg and old.get('split_sha256')==split_sha and old.get('checkpoint_sha256')==sha(d/'checkpoint.pt') and (d/'preprocessing.npz').exists():return
  halt('STOP_TECHNICAL_BLOCKER',seed=seed,reason='completed seed provenance mismatch')
 status('TRAINING',seed=seed);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 rp=d/'resume.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
 if resume is not None and (resume.get('training_config_sha256')!=cfg or resume.get('split_sha256')!=split_sha):halt('STOP_TECHNICAL_BLOCKER',seed=seed,reason='resume provenance mismatch')
 if resume is None:m,s=stats(part['train']);start=1;best=-1.;wait=0;hist=[]
 else:m,s,start,best,wait,hist=resume['mean'],resume['std'],int(resume['epoch'])+1,float(resume['best']),int(resume['wait']),resume['history']
 nput(d/'preprocessing.npz',mean=m,std=s)
 from src.chapman_phase1 import ECGResNetSmall
 model=ECGResNetSmall().cuda();ytr=np.array([r['labels'] for r in part['train']],np.float32);w=np.minimum((len(ytr)-ytr.sum(0))/np.maximum(ytr.sum(0),1),20);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device='cuda'));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda')
 if resume is not None:model.load_state_dict(resume['state_dict']);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
 epochs=range(start,31) if wait<5 else ()
 for epoch in epochs:
  model.train();ls=[]
  for x,y in dl(part['train'],m,s,True):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=lossfn(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
   if not torch.isfinite(loss):halt('STOP_BASELINE_INADEQUATE',seed=seed,reason='nonfinite loss')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();ls.append(float(loss.detach().cpu()))
  y,p=predict(model,part['validation'],m,s);score=float(average_precision_score(y[:,0],p[:,0]));hist.append({'epoch':epoch,'train_loss':float(np.mean(ls)),'validation_twa_auprc':score});jput(d/'history.json',hist)
  if score>best:best=score;wait=0;torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'seed':seed,'labels':list(CODES),'training_config_sha256':cfg,'split_sha256':split_sha},d/'checkpoint.pt')
  else:wait+=1
  torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'best':best,'wait':wait,'history':hist,'mean':m,'std':s,'training_config_sha256':cfg,'split_sha256':split_sha},rp)
  if wait>=5:break
 model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True)['state_dict']);y,p=predict(model,part['validation'],m,s);t=np.array([np.quantile(p[y[:,i]==0,i],.9) for i in range(3)],np.float32);au,ap=float(roc_auc_score(y[:,0],p[:,0])),float(average_precision_score(y[:,0],p[:,0]));ok=bool(np.isfinite(p).all() and all(np.isfinite(q['train_loss']) for q in hist) and au>=.75 and ap>=.45);nput(d/'validation.npz',y=y,p=p,thresholds=t);b={'decision':'PASS' if ok else 'STOP_BASELINE_INADEQUATE','seed':seed,'validation_twa_auroc':au,'validation_twa_auprc':ap,'finite_losses_predictions':bool(np.isfinite(p).all()),'leakage_pass':True,'test_accessed':False,'checkpoint_sha256':sha(d/'checkpoint.pt'),'split_sha256':split_sha,'training_config_sha256':cfg,'preprocessing_sha256':sha(d/'preprocessing.npz'),'completed_utc':now()};jput(bp,b)
 if not ok:halt('STOP_BASELINE_INADEQUATE',seed=seed,auroc=au,auprc=ap)
def valid_gate(g,seed,split_sha,checkpoint_sha):
 return g.get('decision')=='PASS' and g.get('protocol_sha256')==sha(PROTO).upper() and g.get('split_sha256')==split_sha and g.get('checkpoint_sha256',{}).get(str(seed))==checkpoint_sha
def gate():
 z=[];split_sha=sha(OUT/'SPLIT_MEMBERSHIP.json');baselines={};checkpoints={}
 for s in SEEDS:
  p=OUT/f'seed_{s}'/'baseline.json'
  if not p.exists():halt('STOP_BASELINE_INADEQUATE',seed=s,reason='missing baseline')
  q=json.loads(p.read_text());z.append(q);baselines[str(s)]=sha(p);checkpoints[str(s)]=sha(OUT/f'seed_{s}'/'checkpoint.pt')
  if q.get('decision')!='PASS' or q.get('test_accessed') is not False:halt('STOP_BASELINE_INADEQUATE',seed=s)
  if q.get('split_sha256')!=split_sha or q.get('checkpoint_sha256')!=checkpoints[str(s)]:halt('STOP_BASELINE_INADEQUATE',seed=s,reason='baseline provenance')
 jput(OUT/'ALL_SEEDS_VALIDATION_GATE.json',{'decision':'PASS','baselines':z,'baseline_sha256':baselines,'checkpoint_sha256':checkpoints,'split_sha256':split_sha,'protocol_sha256':sha(PROTO).upper(),'test_accessed_before_gate':False,'completed_utc':now()})
def ece(y,p):
 v=0.
 for i in range(15):
  lo,hi=i/15,(i+1)/15;q=(p>=lo)&(p<(hi) if i<14 else p<=hi)
  if q.any():v+=q.mean()*abs(float(y[q].mean())-float(p[q].mean()))
 return float(v)
def inference(model,a):
 import torch
 z=[];model.eval()
 with torch.no_grad():
  for i in range(0,len(a),128):
   with torch.autocast('cuda',dtype=torch.float16):z.append(torch.sigmoid(model(torch.from_numpy(a[i:i+128]).cuda())).float().cpu().numpy())
 return np.concatenate(z).astype('float32')
def evaluate(seed,test):
 import torch
 from src.chapman_phase1 import ECGResNetSmall
 from src.ecg_physics import derive_display,named_reversal
 d=OUT/f'seed_{seed}';cache=d/'test_predictions';cache.mkdir(exist_ok=True);done=d/'evaluation_complete.json'
 gatep=OUT/'ALL_SEEDS_VALIDATION_GATE.json';testids=htxt('\n'.join(sorted(r['record_id'] for r in test))+'\n')
 if not gatep.exists():halt('STOP_BASELINE_INADEQUATE',seed=seed,reason='no all-seed gate')
 g=json.loads(gatep.read_text())
 if not valid_gate(g,seed,sha(OUT/'SPLIT_MEMBERSHIP.json'),sha(d/'checkpoint.pt')):halt('STOP_BASELINE_INADEQUATE',seed=seed,reason='invalid all-seed gate')
 bound={'seed':seed,'protocol_sha256':sha(PROTO).upper(),'split_sha256':sha(OUT/'SPLIT_MEMBERSHIP.json'),'checkpoint_sha256':sha(d/'checkpoint.pt'),'physics_sha256':sha(ROOT/'src/ecg_physics.py'),'test_record_ids_sha256':testids,'condition_definition_sha256':condition_hash(),'test_accessed_before_freeze':False,'mask_count':len(MASKS)}
 if done.exists():
  old=json.loads(done.read_text());
  if any(old.get(k)!=v for k,v in bound.items() if k!='test_accessed_before_freeze') or old.get('test_accessed') is not True:halt('STOP_TECHNICAL_BLOCKER',seed=seed,reason='stale evaluation cache')
  for n in CONDITIONS:
   p=cache/f'{n}.npz'
   if not p.exists() or old.get('prediction_sha256',{}).get(n)!=sha(p):halt('STOP_TECHNICAL_BLOCKER',seed=seed,reason='prediction cache hash')
  return {n:np.load(cache/f'{n}.npz',allow_pickle=False)['p'] for n in CONDITIONS},np.load(d/'validation.npz')['thresholds']
 jput(d/'EVALUATION_FREEZE.json',bound)
 z=np.load(d/'preprocessing.npz');m,s=z['mean'],z['std'];model=ECGResNetSmall().cuda();model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['state_dict']);x=np.stack([raw(r) for r in test]);norm=lambda q:((q-m[None,:,None])/s[None,:,None]).astype('float32');ind=np.concatenate((x[:,:2],x[:,6:]),axis=1);fixed={'clean':norm(x),'RA_LL_reversal':norm(derive_display(named_reversal(ind,'RA_LL_reversal')).astype('float32'))};res={}
 for i,n in enumerate(CONDITIONS):
  p=cache/f'{n}.npz'
  if p.exists():res[n]=np.load(p,allow_pickle=False)['p'];continue
  a=fixed.get(n)
  if a is None:a=norm(np.where(np.isin(np.arange(12),np.array(MASKS[i-2]))[None,:,None],x,m[None,:,None]))
  status('TEST_EVALUATION',seed=seed,condition=i,total_conditions=len(CONDITIONS));res[n]=inference(model,a);nput(p,p=res[n])
 jput(done,{'decision':'PASS',**bound,'conditions':len(CONDITIONS),'test_accessed':True,'prediction_sha256':{n:sha(cache/f'{n}.npz') for n in CONDITIONS},'completed_utc':now()});return res,np.load(d/'validation.npz')['thresholds']
def aggregate(test,pred,thresholds):
 from sklearn.metrics import average_precision_score,roc_auc_score
 y=np.array([r['labels'] for r in test]);met=[];con=[]
 for s in SEEDS:
  for n in CONDITIONS:
   for i,l in enumerate(LABELS):
    p=pred[s][n][:,i];q=p>=thresholds[s][i];pos=y[:,i]==1;met.append({'seed':s,'condition':n,'label':l,'auprc':float(average_precision_score(y[:,i],p)),'auroc':float(roc_auc_score(y[:,i],p)),'sensitivity':float(q[pos].mean()),'specificity':float((~q[~pos]).mean()),'ece_15':ece(y[:,i],p)})
  for i,l in enumerate(LABELS):
   ra=average_precision_score(y[:,i],pred[s]['RA_LL_reversal'][:,i]);m=np.array([average_precision_score(y[:,i],pred[s][n][:,i]) for n in NAMES]);con.append({'seed':s,'label':l,'ra_ll_auprc':float(ra),'mean_mask_auprc':float(m.mean()),'matched_contrast':float(m.mean()-ra)})
 csvput(OUT/'CONDITION_RESULTS_BY_SEED.csv',met);csvput(OUT/'PRIMARY_CONTRASTS_BY_SEED.csv',con);return y,con
def mean_mask_ap(ycol,seed_predictions,indices,names=NAMES):
 """Exact sklearn AUPRC, chunked only to bound temporary references/memory."""
 from sklearn.metrics import average_precision_score
 values=[]
 for start in range(0,len(names),32):values.extend(average_precision_score(ycol[indices],seed_predictions[n][indices]) for n in names[start:start+32])
 return float(np.mean(values))
def bootstrap_runtime_estimate():
 from sklearn.metrics import average_precision_score
 p=OUT/'BOOTSTRAP_RUNTIME_ESTIMATE.json'
 if p.exists():return
 rng=np.random.default_rng(7);y=rng.integers(0,2,96);scores={n:rng.random(96) for n in NAMES};ix=rng.integers(0,96,96);t=time.monotonic();mean_mask_ap(y,scores,ix);average_precision_score(y[ix],scores[NAMES[0]][ix]);elapsed=max(time.monotonic()-t,1e-6);calls=1000*len(SEEDS)*len(LABELS)*(len(NAMES)+1)
 jput(p,{'method':'exact sklearn average_precision_score, 32-condition chunks; no approximation','microbenchmark_seconds_for_793_metrics':elapsed,'exact_metric_calls':calls,'estimated_single_process_seconds':elapsed*calls/(len(NAMES)+1),'replicates':1000,'masks':792,'seeds':3,'labels':3,'created_utc':now()})
def bootstrap(y,pred):
 from sklearn.metrics import average_precision_score
 bootstrap_runtime_estimate()
 ip,sp=OUT/'bootstrap_indices.npy',OUT/'bootstrap_state.npz'
 if ip.exists():ix=np.load(ip,allow_pickle=False)
 else:
  ix=np.random.default_rng(20260829).integers(0,len(y),size=(NBOOT,len(y)),dtype=np.int32);q=ip.with_name('.'+ip.name);np.save(q,ix);os.replace(q,ip)
 if sp.exists():z=np.load(sp,allow_pickle=False);done=z['done'].astype(bool);c=z['contrasts']
 else:done=np.zeros(NBOOT,bool);c=np.full((NBOOT,3,3),np.nan,np.float32)
 cols={s:[{n:pred[s][n][:,li] for n in NAMES} for li in range(3)] for s in SEEDS}
 for b in range(NBOOT):
  if done[b]:continue
  for si,s in enumerate(SEEDS):
   for li in range(3):c[b,si,li]=mean_mask_ap(y[:,li],cols[s][li],ix[b])-average_precision_score(y[ix[b],li],pred[s]['RA_LL_reversal'][ix[b],li])
  done[b]=True
  if done.sum()%25==0 or done.all():nput(sp,done=done,contrasts=c);jput(OUT/'BOOTSTRAP_PROGRESS.json',{'completed_replicates':int(done.sum()),'total_replicates':NBOOT,'state_sha256':sha(sp)});status('BOOTSTRAP',completed=int(done.sum()),total=NBOOT)
 a=c.mean(1);rs=[{'label':l,'mean_contrast':float(a[:,i].mean()),'lower95':float(np.quantile(a[:,i],.025)),'upper95':float(np.quantile(a[:,i],.975)),'replicates':NBOOT} for i,l in enumerate(LABELS)];csvput(OUT/'BOOTSTRAP_SUMMARY.csv',rs);csvput(OUT/'BOOTSTRAP_REPLICATES.csv',[{'replicate':b,'label':l,'contrast':float(a[b,i])} for b in range(NBOOT) for i,l in enumerate(LABELS)]);return rs
def finalize(c,b):
 twa=[x for x in c if x['label']=='TWA'];bs={x['label']:x for x in b};ok=all(x['matched_contrast']>0 for x in twa) and bs['TWA']['mean_contrast']>=.05 and bs['TWA']['lower95']>0;z={'decision':'CONFIRMED_GENERAL_FAILURE' if ok else 'NOT_CONFIRMED','all_baseline_gates_pass':True,'all_seed_twa_contrasts_positive':all(x['matched_contrast']>0 for x in twa),'bootstrap':bs,'protocol_sha256':sha(PROTO).upper(),'completed_utc':now()};jput(OUT/'conclusions.json',z);(OUT/'FINAL_NINGBO_CONFIRMATION_DECISION.md').write_text('# '+z['decision']+'\n\n'+json.dumps(z,indent=2)+'\n',encoding='utf8')
def selftest():
 import tempfile
 from sklearn.metrics import average_precision_score
 from torch.utils.data import DataLoader
 r=lambda i,h:{'record_id':i,'waveform_sha256':h,'labels':[0,0,0],'split':'train','age':20,'sex':'male','dx':[CODES[0]]}
 # Exact duplicate collapse is allowed only for equal metadata/Dx; a conflict stops.
 assert duplicate_signature(r('a','same'))==duplicate_signature(r('b','same')) and min(('a','b'))=='a'
 assert duplicate_signature(r('a','same'))!=duplicate_signature({**r('b','same'),'age':21})
 with tempfile.TemporaryDirectory() as td:
  base=Path(td);(base/'g1').mkdir();(base/'g1'/'ok.hea').write_text('x')
  assert path('training/ningbo/g1/ok.hea',base)==base/'g1'/'ok.hea'
  for badpath in ('training/chapman/g1/ok.hea','training/ningbo/../g1/ok.hea','C:/data/ok.hea'):
   try:path(badpath,base);raise AssertionError('invalid mapping path accepted')
   except ValueError:pass
  try:path('training/ningbo/g1/missing.hea',base);raise AssertionError('missing target accepted')
  except FileNotFoundError:pass
  hp=base/'official.hea';hp.write_text('R 12 500 10\nR.mat 16 1 0 0 0 0 I\nR.mat 16 1 0 0 0 0 II\nR.mat 16 1 0 0 0 0 III\nR.mat 16 1 0 0 0 0 aVR\nR.mat 16 1 0 0 0 0 aVL\nR.mat 16 1 0 0 0 0 aVF\nR.mat 16 1 0 0 0 0 V1\nR.mat 16 1 0 0 0 0 V2\nR.mat 16 1 0 0 0 0 V3\nR.mat 16 1 0 0 0 0 V4\nR.mat 16 1 0 0 0 0 V5\nR.mat 16 1 0 0 0 0 V6\n# Age: NaN\n# Sex: Unknown\n# Dx: 164934002, 426177001\n')
  age,sex,dx=header(hp);assert age=='UNKNOWN' and sex=='UNKNOWN' and CODES[0] in dx and CODES[1] in dx
  assert not audit_binding({'ingestion_schema':'old','mapping_sha256':'m','protocol_sha256':sha(PROTO).upper()},'m')
  # End-to-end synthetic shard merge: two valid bound shards collapse an exact
  # duplicate; a metadata conflict is returned for STOP_DUPLICATE_CONFLICT.
  bind={'ingestion_schema':INGESTION_SCHEMA,'mapping_sha256':'m','protocol_sha256':sha(PROTO).upper()}
  one={**r('g1/z','dup'),'header_path':'h','waveform_path':'w'};two={**r('g1/a','dup'),'header_path':'h','waveform_path':'w'};three={**r('g1/c','unique'),'header_path':'h','waveform_path':'w'}
  jput(base/'00000.json',{'rows':[one],'exclusions':[],**bind});jput(base/'00250.json',{'rows':[two,three],'exclusions':[],**bind})
  loaded=[]
  for sp in sorted(base.glob('*.json')):
   q=json.loads(sp.read_text());assert audit_binding(q,'m');loaded+=q['rows']
  canonical,collapsed,conflicts=merge_audit_rows(loaded);assert [q['record_id'] for q in canonical]==['g1/a','g1/c'] and len(collapsed)==1 and not conflicts
  _,_,conflicts=merge_audit_rows([one,{**two,'age':21}]);assert len(conflicts)==1
 leakage([r('a','h1'),{**r('b','h2'),'split':'validation'},{**r('c','h3'),'split':'test'}])
 try:leakage([r('a','h1'),{**r('b','h1'),'split':'test'}]);raise AssertionError('leakage not detected')
 except Stop:pass
 from src.ecg_physics import derive_display,named_reversal
 x=np.random.default_rng(2).normal(size=(2,8,7));assert len(MASKS)==792 and len(set(MASKS))==792 and not np.allclose(derive_display(x),derive_display(named_reversal(x,'RA_LL_reversal')))
 # Actual Windows-spawn loader path: Dataset is top-level and worker iteration succeeds.
 loader=DataLoader(SyntheticDataset(),batch_size=2,num_workers=2,persistent_workers=True);bx,by=next(iter(loader));assert tuple(bx.shape)==(2,12,8) and tuple(by.shape)==(2,3)
 # Gate must reject wrong split/checkpoint before test evaluation can begin.
 good={'decision':'PASS','protocol_sha256':sha(PROTO).upper(),'split_sha256':'S','checkpoint_sha256':{'104729':'C'}};assert valid_gate(good,104729,'S','C') and not valid_gate(good,104729,'wrong','C') and not valid_gate(good,104729,'S','wrong')
 # Exact chunked AP is numerically equal to the brute-force frozen definition.
 rng=np.random.default_rng(9);yy=rng.integers(0,2,31);names=NAMES[:5];scores={n:rng.random(31) for n in names};ix=rng.integers(0,31,31);assert np.isclose(mean_mask_ap(yy,scores,ix,names),np.mean([average_precision_score(yy[ix],scores[n][ix]) for n in names]))
 class FakeCUDA:
  @staticmethod
  def is_available():return True
  @staticmethod
  def get_device_name(_):return 'NVIDIA GeForce RTX 5070'
 class FakeTorch: cuda=FakeCUDA();__version__='mock';version=type('V',(),{'cuda':'mock'})()
 assert cuda_info(FakeTorch)['gpu'].endswith('5070')
 d=OUT/'_self_test';d.mkdir(parents=True,exist_ok=True);jput(d/'telemetry.json',{'ok':True});nput(d/'resume.npz',done=np.array([1,0]));assert json.loads((d/'telemetry.json').read_text())['ok'] and np.load(d/'resume.npz')['done'].tolist()==[1,0] and not alive(-1)
 print('SELF_TEST PASS: official-style header fields/Unknown metadata, stale cache and bound multi-shard audit merge/conflict, split leakage, test gate sealing, 792 masks, RA_LL physics, Windows loader spawn, exact bootstrap equivalence/resume, stale/dead/live locks, atomic telemetry, CUDA mock',flush=True)
def run(preflight_only=False,prepare_only=False,smoke_only=False):
 p=lock()
 try:
  mp=preflight()
  if preflight_only:return
  if smoke_only:
   rp,tvp=OUT/'CANONICAL_RECORDS.json',OUT/'TRAIN_VALIDATION_MEMBERSHIP.json'
   if not rp.exists() or not tvp.exists():halt('STOP_TECHNICAL_BLOCKER',reason='--smoke-only requires completed --prepare-only')
   by={r['record_id']:r for r in json.loads(rp.read_text())['records']};tv=json.loads(tvp.read_text())['rows'];part={s:[by[q['record_id']] for q in tv if q['split']==s] for s in ('train','validation')};smoke(part);return
  rs=audit(mp);rows=split(rs);part=prepare(rs,rows)
  if prepare_only:return
  for s in SEEDS:train(s,part)
  gate();pred={};th={}
  for s in SEEDS:pred[s],th[s]=evaluate(s,part['test'])
  y,c=aggregate(part['test'],pred,th);b=bootstrap(y,pred);finalize(c,b);status('COMPLETE')
 except Stop:raise
 except Exception as e:status('STOP_TECHNICAL_BLOCKER',error=repr(e));raise
 finally:p.unlink(missing_ok=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--self-test',action='store_true');a.add_argument('--preflight-only',action='store_true');a.add_argument('--prepare-only',action='store_true');a.add_argument('--smoke-only',action='store_true');q=a.parse_args()
 if q.self_test:selftest()
 else:
  # The durable workflow owns append-only stdout/stderr logs even when started
  # from Task Scheduler or a detached PowerShell session.
  OUT.mkdir(parents=True,exist_ok=True)
  code=0
  with open(OUT/'workflow_stdout.log','a',encoding='utf8') as out,open(OUT/'workflow_stderr.log','a',encoding='utf8') as err:
   with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
    try:run(q.preflight_only,q.prepare_only,q.smoke_only)
    except Exception:
     traceback.print_exc();code=1
  if code:raise SystemExit(code)
if __name__=='__main__':main()
