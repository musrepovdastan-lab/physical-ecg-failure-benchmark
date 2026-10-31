"""Read-only, resumable finalization of locked Ningbo M2 saved predictions."""
from __future__ import annotations
import argparse,csv,ctypes,hashlib,itertools,json,os,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mitigation_m2_ningbo_v1';BASE=ROOT/'outputs/ningbo_confirmation_v2';PROTO=ROOT/'configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json'
SEEDS=(104729,130363,155921);LABELS=('TWA','SB','STach');MASKS=tuple(itertools.combinations(range(12),5));NAMES=('clean','RA_LL_reversal',*tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS));BOOT=1000
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def aj(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def an(target,**arrays):
 """Atomically save named arrays; prediction arrays may legitimately be named ``p``."""
 target=Path(target);q=target.with_name('.'+target.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**arrays);os.replace(q,target)
def wc(p,rows):
 p=Path(p);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
 with open(q,'w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(q,p)
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
def status(stage,**x):
 z={'pid':os.getpid(),'stage':stage,'updated_utc':now(),'completed_conditions':3*794,'total_conditions':3*794,'bootstrap_completed':0,'bootstrap_total':1000,'estimated_remaining_seconds':None,**x};aj(OUT/'WORKFLOW_STATUS.json',z);print(json.dumps(z,sort_keys=True),flush=True)
def lock():
 p=OUT/'mitigation_m2_finalizer_v1.lock'
 if p.exists():
  q=json.loads(p.read_text())
  if alive(q.get('pid')):raise RuntimeError('live M2 finalizer lock')
  p.unlink()
 trainer=OUT/'mitigation_m2_ningbo_v1.lock'
 if trainer.exists() and alive(json.loads(trainer.read_text()).get('pid')):raise RuntimeError('M2 trainer/evaluator remains active')
 aj(p,{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTO).upper()});return p
def load(p):
 z=np.load(p,allow_pickle=False);a=z['p'].astype(np.float32)
 if a.shape!=(6981,3) or not np.isfinite(a).all():raise RuntimeError(f'invalid saved prediction: {p}')
 return a
def labels():
 records={x['record_id']:x for x in json.loads((BASE/'CANONICAL_RECORDS.json').read_text())['records']};rows=json.loads((BASE/'SPLIT_MEMBERSHIP.json').read_text())['rows'];test=[x['record_id'] for x in rows if x['split']=='test']
 y=np.asarray([records[x]['labels'] for x in test],np.int8)
 if y.shape!=(6981,3):raise RuntimeError('test label shape mismatch')
 return y
def preflight():
 frozen=json.loads((OUT/'FROZEN_PROTOCOL_HASH.json').read_text())
 if frozen.get('sha256')!=sha(PROTO).upper():raise RuntimeError('M2 protocol hash mismatch')
 if sha(BASE/'bootstrap_indices.npy').upper()!=json.loads(PROTO.read_text())['immutable_bindings']['bootstrap_indices']['sha256']:raise RuntimeError('immutable bootstrap index hash mismatch')
 for s in SEEDS:
  d=OUT/f'seed_{s}';g=json.loads((d/'validation_gate.json').read_text());e=json.loads((d/'evaluation_complete.json').read_text());f=json.loads((d/'evaluation_freeze.json').read_text());checkpoint=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu')
  with np.load(d/'validation.npz',allow_pickle=False) as v:valid_validation=(v['y'].shape==(3488,3) and v['p'].shape==(3488,3) and v['thresholds'].shape==(3,) and np.isfinite(v['p']).all() and np.isfinite(v['thresholds']).all())
  if g.get('decision')!='PASS' or not valid_validation or checkpoint.get('protocol_sha256')!=sha(PROTO).upper() or checkpoint.get('seed')!=s or e.get('conditions')!=794 or e.get('checkpoint_sha256')!=g.get('checkpoint_sha256') or f.get('protocol_sha256')!=sha(PROTO).upper() or f.get('checkpoint_sha256')!=g.get('checkpoint_sha256') or f.get('conditions')!=794 or g.get('test_accessed_before_evaluation_freeze') is not False or f.get('test_accessed_before_evaluation_freeze') is not False or e.get('test_accessed_before_evaluation_freeze') is not False:raise RuntimeError(f'seed {s} validation/evaluation binding failure')
  for n,digest in e.get('prediction_sha256',{}).items():
   q=d/'test_predictions'/f'{n}.npz'
   if sha(q).lower()!=str(digest).lower():raise RuntimeError(f'seed {s} prediction hash failure {n}')
   load(q)  # required shape (6981,3) and finite-value validation
  if len(e.get('prediction_sha256',{}))!=794:raise RuntimeError(f'seed {s} condition manifest failure')
 return {'decision':'PASS','protocol_sha256':sha(PROTO).upper(),'bootstrap_indices_sha256':sha(BASE/'bootstrap_indices.npy').upper(),'seeds':list(SEEDS),'validation_gates_verified':3,'checkpoint_bindings_verified':3,'evaluation_freeze_provenance_verified':3,'saved_m2_prediction_hashes_verified':3*794,'prediction_shape_and_finite_verified':3*794,'test_waveforms_or_inference_used':False}
def predictions():
 base={};m2={}
 for s in SEEDS:
  base[s]={n:load(BASE/f'seed_{s}'/'test_predictions'/f'{n}.npz') for n in NAMES}
  m2[s]={n:load(OUT/f'seed_{s}'/'test_predictions'/f'{n}.npz') for n in NAMES}
 return base,m2
def ece(y,p):
 out=0.
 for k in range(15):
  z=(p>=k/15)&(p<=((k+1)/15) if k==14 else p<(k+1)/15)
  if z.any():out+=z.mean()*abs(y[z].mean()-p[z].mean())
 return float(out)
def rows(arm,pred,y,thresholds):
 r=[];primary=[]
 for s in SEEDS:
  for n in NAMES:
   for li,label in enumerate(LABELS):
    p=pred[s][n][:,li];z=y[:,li];c=p>=thresholds[s][li];pos=z==1;neg=~pos
    r.append({'arm':arm,'seed':s,'condition':n,'endpoint':label,'auprc':float(average_precision_score(z,p)),'auroc':float(roc_auc_score(z,p)),'sensitivity':float(c[pos].mean()),'specificity':float((~c[neg]).mean()),'ece_15':ece(z,p),'threshold':float(thresholds[s][li]),'n':len(z)})
  for li,label in enumerate(LABELS):
   c=average_precision_score(y[:,li],pred[s]['clean'][:,li]);v=average_precision_score(y[:,li],pred[s]['RA_LL_reversal'][:,li]);m=np.asarray([average_precision_score(y[:,li],pred[s][n][:,li]) for n in NAMES[2:]])
   primary.append({'arm':arm,'seed':s,'endpoint':label,'clean_auprc':float(c),'ra_ll_auprc':float(v),'mean_mask_auprc':float(m.mean()),'median_mask_auprc':float(np.median(m)),'p05_mask_auprc':float(np.quantile(m,.05)),'p95_mask_auprc':float(np.quantile(m,.95)),'matched_contrast':float(m.mean()-v),'ra_ll_loss':float(c-v)})
 return r,primary
def contrast(y,ix,pred,li):
 return float(np.mean([average_precision_score(y[ix,li],pred[n][ix,li]) for n in NAMES[2:]])-average_precision_score(y[ix,li],pred['RA_LL_reversal'][ix,li]))
def bootstrap(y,base,m2):
 idx=np.load(BASE/'bootstrap_indices.npy',allow_pickle=False)
 if idx.shape!=(BOOT,len(y)):raise RuntimeError('bootstrap index shape mismatch')
 state=OUT/'bootstrap_state.npz';progress=OUT/'BOOTSTRAP_PROGRESS.json'
 if state.exists():
  z=np.load(state,allow_pickle=False);done=z['done'].astype(bool);bc=z['baseline_contrasts'];mc=z['m2_contrasts']
  if done.shape!=(BOOT,) or bc.shape!=(BOOT,3,3) or mc.shape!=(BOOT,3,3):raise RuntimeError('bootstrap state shape mismatch')
 else:done=np.zeros(BOOT,bool);bc=np.full((BOOT,3,3),np.nan,np.float32);mc=np.full((BOOT,3,3),np.nan,np.float32)
 for b in range(3):
  for si,s in enumerate(SEEDS):
   for li in range(3):
    direct=float(np.mean([average_precision_score(y[idx[b],li],base[s][n][idx[b],li]) for n in NAMES[2:]])-average_precision_score(y[idx[b],li],base[s]['RA_LL_reversal'][idx[b],li]))
    if not np.isclose(direct,contrast(y,idx[b],base[s],li),rtol=0,atol=1e-12):raise RuntimeError('direct AUPRC bootstrap smoke mismatch')
 aj(OUT/'BOOTSTRAP_SMOKE.json',{'decision':'PASS','replicates_checked':3,'implementation':'direct condition-level AUPRC; no prediction-vector averaging; baseline contrast once per replicate/seed/endpoint'})
 started=time.monotonic();initial=int(done.sum())
 for b in range(BOOT):
  if done[b]:continue
  for si,s in enumerate(SEEDS):
   for li in range(3):bc[b,si,li]=contrast(y,idx[b],base[s],li);mc[b,si,li]=contrast(y,idx[b],m2[s],li)
  done[b]=True
  if done.sum()%10==0 or done.all():
   an(state,done=done,baseline_contrasts=bc,m2_contrasts=mc);rate=(done.sum()-initial)/(time.monotonic()-started) if done.sum()>initial else 0;eta=(BOOT-done.sum())/rate if rate else None
   aj(progress,{'completed_replicates':int(done.sum()),'total_replicates':BOOT,'indices_sha256':sha(BASE/'bootstrap_indices.npy').upper(),'state_sha256':sha(state),'estimated_remaining_seconds':eta,'updated_utc':now()});status('BOOTSTRAP',bootstrap_completed=int(done.sum()),bootstrap_total=BOOT,estimated_remaining_seconds=eta)
 out=[]
 for li,label in enumerate(LABELS):
  for metric,x in [('baseline_matched_contrast',bc[:,:,li].mean(1)),('m2_matched_contrast',mc[:,:,li].mean(1)),('contrast_reduction_vs_baseline',(bc[:,:,li]-mc[:,:,li]).mean(1))]:out.append({'endpoint':label,'metric':metric,'mean':float(x.mean()),'low95':float(np.quantile(x,.025)),'high95':float(np.quantile(x,.975)),'replicates':BOOT,'unit':'canonical_record','synchronized':True})
 return out
def figures(primary,boots):
 import matplotlib.pyplot as plt
 d=OUT/'figures';d.mkdir(exist_ok=True);b={(x['endpoint'],x['metric']):x for x in boots}
 f,ax=plt.subplots(figsize=(7,4))
 for j,label in enumerate(LABELS):
  x=b[(label,'contrast_reduction_vs_baseline')];ax.errorbar(x['mean'],j,xerr=[[x['mean']-x['low95']],[x['high95']-x['mean']]],fmt='o',label=label)
 ax.axvline(.15,color='black',ls='--',lw=1);ax.set(yticks=range(3),yticklabels=LABELS,xlabel='matched-contrast reduction (M2 vs baseline)',title='Ningbo M2 paired bootstrap');ax.legend();f.tight_layout();f.savefig(d/'m2_forest.svg');f.savefig(d/'m2_forest.png',dpi=220);plt.close(f)
 f,ax=plt.subplots(figsize=(7,4))
 for arm,color in [('cached_clean_baseline','#333333'),('physics_aware_m2','#4477AA')]:
  q=[x for x in primary if x['arm']==arm and x['endpoint']=='TWA'];ax.plot([x['seed'] for x in q],[x['ra_ll_auprc'] for x in q],'o-',label=arm.replace('_',' '),color=color)
 ax.set(xlabel='seed',ylabel='RA–LL TWA AUPRC',title='Ningbo M2 RA–LL recovery');ax.legend();f.tight_layout();f.savefig(d/'m2_twa_ra_ll.svg');f.savefig(d/'m2_twa_ra_ll.png',dpi=220);plt.close(f)
def decide(primary,boots):
 q=[x for x in primary if x['arm']=='cached_clean_baseline' and x['endpoint']=='TWA'];p=[x for x in primary if x['arm']=='physics_aware_m2' and x['endpoint']=='TWA'];base={x['seed']:x for x in q};m2={x['seed']:x for x in p};mean=lambda x,k:float(np.mean([z[k] for z in x]));b={(x['endpoint'],x['metric']):x for x in boots}
 checks={'ra_ll_twa_improvement_each_seed_min_0_15':all(m2[s]['ra_ll_auprc']-base[s]['ra_ll_auprc']>=.15 for s in SEEDS),'aggregate_matched_contrast_decrease_min_0_15':mean(q,'matched_contrast')-mean(p,'matched_contrast')>=.15,'aggregate_matched_contrast_upper95_max_0_15':b[('TWA','m2_matched_contrast')]['high95']<=.15,'clean_twa_mean_decline_max_0_02':mean(q,'clean_auprc')-mean(p,'clean_auprc')<=.02,'clean_twa_any_seed_decline_max_0_04':all(base[s]['clean_auprc']-m2[s]['clean_auprc']<=.04 for s in SEEDS),'mean_mask_twa_mean_decline_max_0_02':mean(q,'mean_mask_auprc')-mean(p,'mean_mask_auprc')<=.02,'mean_mask_twa_any_seed_decline_max_0_04':all(base[s]['mean_mask_auprc']-m2[s]['mean_mask_auprc']<=.04 for s in SEEDS),'all_three_seeds_validation_and_integrity':True}
 return ('PASS_M2_EXTERNAL_REPLICATION' if all(checks.values()) else 'FAIL_M2_NOT_REPLICATED'),checks
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('preflight','run'));args=a.parse_args();lk=lock()
 try:
  pf=preflight();aj(OUT/'CACHED_ARTIFACT_PREFLIGHT.json',pf)
  if args.command=='preflight':status('FINALIZER_PREFLIGHT_PASS');return
  y=labels();base,m2=predictions();bt={s:np.load(BASE/f'seed_{s}'/'validation.npz')['thresholds'] for s in SEEDS};mt={s:np.load(OUT/f'seed_{s}'/'validation.npz')['thresholds'] for s in SEEDS};rb,pb=rows('cached_clean_baseline',base,y,bt);rm,pm=rows('physics_aware_m2',m2,y,mt);gates=[];train=[]
  for s in SEEDS:
   g=json.loads((OUT/f'seed_{s}'/'validation_gate.json').read_text());gates.append(g);train.append({'seed':s,'status':'COMPLETE','validation_gate':g['decision'],'epochs_completed':len(g['history']),'best_validation_twa_auprc':max(x['validation_twa_auprc'] for x in g['history']),'checkpoint_sha256':g['checkpoint_sha256'],'evaluation_conditions':794})
  wc(OUT/'TRAINING_STATUS_BY_SEED.csv',train);aj(OUT/'VALIDATION_GATES.json',{'all_pass':True,'gates':gates});wc(OUT/'RESULTS_BY_SEED.csv',rb+rm);wc(OUT/'PRIMARY_M2_CONTRASTS.csv',pb+pm);bs=bootstrap(y,base,m2);wc(OUT/'BOOTSTRAP_SUMMARY.csv',bs);figures(pb+pm,bs);decision,checks=decide(pb+pm,bs);out={'decision':decision,'completed_utc':now(),'protocol_sha256':sha(PROTO).upper(),'interpretation':'locked external follow-up replication after observed historical Ningbo baseline outcomes','checks':checks,'bootstrap':{x['endpoint']+'_'+x['metric']:x for x in bs}};aj(OUT/'conclusions_m2.json',out);(OUT/'FINAL_M2_DECISION.md').write_text('# '+decision+'\n\n'+json.dumps(out,indent=2)+'\n',encoding='utf8');readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');block='<!-- M2_STATUS_START -->\n## Mitigation M2\n\nLocked Ningbo external follow-up of the Chapman-frozen physics-aware method. Final decision: `'+decision+'`. See `outputs/mitigation_m2_ningbo_v1/FINAL_M2_DECISION.md`.\n<!-- M2_STATUS_END -->';readme.write_text(__import__('re').sub(r'<!-- M2_STATUS_START -->.*?<!-- M2_STATUS_END -->',block,text,flags=__import__('re').S) if '<!-- M2_STATUS_START -->' in text else text+'\n\n'+block+'\n',encoding='utf8');status('COMPLETE',decision=decision,bootstrap_completed=BOOT,bootstrap_total=BOOT,bootstrap_state_sha256=sha(OUT/'bootstrap_state.npz').upper(),estimated_remaining_seconds=0.0)
 except Exception as exc:
  aj(OUT/'conclusions_m2.json',{'decision':'STOP_M2_TECHNICAL_BLOCKER','error':repr(exc),'completed_utc':now()});status('STOP_M2_TECHNICAL_BLOCKER',error=repr(exc));raise
 finally:lk.unlink(missing_ok=True)
if __name__=='__main__':main()
