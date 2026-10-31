"""Cached-prediction-only, resumable M1 finalizer; never trains or infers."""
from __future__ import annotations
import argparse,csv,ctypes,hashlib,itertools,json,os,re,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mitigation_m1_v1';V4=ROOT/'outputs/phase1_chapman_seed1_v4';V6=ROOT/'outputs/phase1b_matched_reversal_development_v6';V7=ROOT/'outputs/phase2_chapman_multiseed_v7';PROTO=ROOT/'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
EXPECTED='248976F691003DAA45C5F188C53F7CF75891F19F8723223477FC41E1D65667CE';SEEDS=(104729,130363,155921);ARMS=('physics_aware','masking','combined');LABELS=('TWA','SB','STach');MASKS=tuple(itertools.combinations(range(12),5));NAMES=('clean','RA_LL_reversal',*tuple('mask5_'+'_'.join(map(str,x)) for x in MASKS));BOOT=1000;BOOT_SEED=20260830
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def aj(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def an(p,**x):
 p=Path(p);q=p.with_name('.'+p.stem+'.'+uuid.uuid4().hex+'.npz');np.savez_compressed(q,**x);os.replace(q,p)
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
def status(stage,**x):aj(OUT/'WORKFLOW_STATUS.json',{'pid':os.getpid(),'stage':stage,'updated_utc':now(),**x})
def acquire():
 p=OUT/'mitigation_m1_finalizer_v1.lock'
 if p.exists():
  q=json.loads(p.read_text())
  if alive(q.get('pid')):raise RuntimeError('live finalizer lock')
  p.unlink()
 t=OUT/'mitigation_m1_v1.lock'
 if t.exists() and alive(json.loads(t.read_text()).get('pid')):raise RuntimeError('trainer or inference process remains active')
 aj(p,{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTO)});return p
def load(p):
 z=np.load(p,allow_pickle=False);a=z['p'].astype(np.float32)
 if a.shape!=(2050,3) or not np.isfinite(a).all():raise RuntimeError(f'invalid prediction array: {p}')
 return a
def baseline_thresholds():
 q={104729:np.asarray(json.loads((V4/'BASELINE_ADEQUACY_v4.json').read_text())['thresholds_at_90pct_specificity'],float)}
 for s in SEEDS[1:]:q[s]=np.asarray(json.loads((V7/f'seed_{s}'/'baseline.json').read_text())['thresholds'],float)
 return q
def preflight():
 problems=[]
 if len(NAMES)!=794 or len(set(NAMES))!=794:problems.append('condition enumeration is not 794 unique conditions')
 if sha(PROTO).upper()!=EXPECTED:problems.append('frozen protocol SHA-256 mismatch')
 frozen=json.loads((OUT/'FROZEN_PROTOCOL_HASH.json').read_text())
 if frozen.get('sha256')!=EXPECTED:problems.append('recorded frozen protocol SHA-256 mismatch')
 found={(a,int(d.name.split('_')[1])) for a in ARMS for d in (OUT/a).glob('seed_*') if d.is_dir() and d.name.split('_')[1].isdigit()}
 if found!={(a,s) for a in ARMS for s in SEEDS}:problems.append(f'expected exactly nine arm/seed units, found {sorted(found)}')
 trainer=OUT/'mitigation_m1_v1.lock'
 if trainer.exists() and alive(json.loads(trainer.read_text()).get('pid')):problems.append('trainer or inference process remains active')
 for a,s in sorted(found):
  d=OUT/a/f'seed_{s}';g=json.loads((d/'validation_gate.json').read_text()) if (d/'validation_gate.json').exists() else {};e=json.loads((d/'evaluation_complete.json').read_text()) if (d/'evaluation_complete.json').exists() else {}
  if g.get('decision')!='PASS':problems.append(f'{a}/{s}: validation gate is not PASS')
  if not e:problems.append(f'{a}/{s}: evaluation_complete missing')
  if e.get('conditions')!=794:problems.append(f'{a}/{s}: condition count is not 794')
  if e.get('checkpoint_sha256')!=g.get('checkpoint_sha256'):problems.append(f'{a}/{s}: evaluation/validation checkpoint hash mismatch')
  p=d/'test_predictions';files={x.stem for x in p.glob('*.npz')} if p.exists() else set()
  if set(NAMES)-files:problems.append(f'{a}/{s}: missing prediction file(s)')
  for n in NAMES:
   try:load(p/f'{n}.npz')
   except Exception as exc:problems.append(f'{a}/{s}/{n}: {exc}');break
 if problems:raise RuntimeError('; '.join(problems))
 return {'status':'PASS','checked_utc':now(),'protocol_sha256':EXPECTED,'unit_count':9,'conditions_per_unit':794,'prediction_shape':[2050,3],'all_validation_gates_pass':True,'all_evaluation_checkpoint_hashes_match':True,'trainer_or_inference_active':False}
def predictions():
 b={};a={x:{} for x in ARMS}
 for s in SEEDS:
  d=V4/'test_predictions_v4' if s==104729 else V7/f'seed_{s}'/'test_predictions';m=V6/'predictions' if s==104729 else d;b[s]={'clean':load(d/'clean.npz'),'RA_LL_reversal':load(d/'RA_LL_reversal.npz'),**{n:load(m/f'{n}.npz') for n in NAMES[2:]}}
  for arm in ARMS:a[arm][s]={n:load(OUT/arm/f'seed_{s}'/'test_predictions'/f'{n}.npz') for n in NAMES}
 return b,a
def ece(y,p):
 out=0.
 for k in range(15):
  z=(p>=k/15)&(p<=((k+1)/15) if k==14 else p<(k+1)/15)
  if z.any():out+=z.mean()*abs(y[z].mean()-p[z].mean())
 return float(out)
def result_rows(arm,pred,y,th):
 allrows=[];primary=[]
 for s in SEEDS:
  for n in NAMES:
   for i,label in enumerate(LABELS):
    p=pred[s][n][:,i];z=y[:,i];call=p>=th[s][i];pos=z==1;neg=~pos
    allrows.append({'arm':arm,'seed':s,'condition':n,'endpoint':label,'auprc':float(average_precision_score(z,p)),'auroc':float(roc_auc_score(z,p)),'sensitivity':float(call[pos].mean()),'specificity':float((~call[neg]).mean()),'ece_15':ece(z,p),'threshold':float(th[s][i]),'n':len(z)})
  for i,label in enumerate(LABELS):
   c=average_precision_score(y[:,i],pred[s]['clean'][:,i]);r=average_precision_score(y[:,i],pred[s]['RA_LL_reversal'][:,i]);m=np.array([average_precision_score(y[:,i],pred[s][n][:,i]) for n in NAMES[2:]])
   primary.append({'arm':arm,'seed':s,'endpoint':label,'clean_auprc':float(c),'ra_ll_auprc':float(r),'mean_mask_auprc':float(m.mean()),'median_mask_auprc':float(np.median(m)),'matched_contrast':float(m.mean()-r),'ra_ll_loss':float(c-r),'ra_ll_percentile_among_masks':float((m<=r).mean())})
 return allrows,primary
def point_audit(primary):
 q=[x for x in primary if x['arm']=='cached_clean_baseline' and x['endpoint']=='TWA'];p=[x for x in primary if x['arm']=='physics_aware' and x['endpoint']=='TWA'];mean=lambda a,k:float(np.mean([x[k] for x in a]));base={x['seed']:x for x in q};phy={x['seed']:x for x in p}
 actual={'baseline_mean_twa':{'clean':mean(q,'clean_auprc'),'ra_ll':mean(q,'ra_ll_auprc'),'mean_masks':mean(q,'mean_mask_auprc'),'matched_contrast':mean(q,'matched_contrast')},'physics_mean_twa':{'clean':mean(p,'clean_auprc'),'ra_ll':mean(p,'ra_ll_auprc'),'mean_masks':mean(p,'mean_mask_auprc'),'matched_contrast':mean(p,'matched_contrast')},'physics_ra_ll_improvement_by_seed':{str(s):phy[s]['ra_ll_auprc']-base[s]['ra_ll_auprc'] for s in SEEDS}}
 expected={'baseline_mean_twa':{'clean':.6531,'ra_ll':.3278,'mean_masks':.5738,'matched_contrast':.2460},'physics_mean_twa':{'clean':.6641,'ra_ll':.6372,'mean_masks':.5679,'matched_contrast':-.0693},'physics_ra_ll_improvement_by_seed':{'104729':.3631,'130363':.2884,'155921':.2768}}
 ok=all(abs(actual[g][k]-expected[g][k])<=.006 for g in ('baseline_mean_twa','physics_mean_twa') for k in expected[g]) and all(abs(actual['physics_ra_ll_improvement_by_seed'][k]-v)<=.006 for k,v in expected['physics_ra_ll_improvement_by_seed'].items());audit={'status':'PASS' if ok else 'FAIL','tolerance_absolute':.006,'expected':expected,'actual':actual};aj(OUT/'POINT_ESTIMATE_AUDIT.json',audit)
 if not ok:raise RuntimeError('cached point estimates materially disagree with audit expectations')
 return audit
def contrast(y,ix,q,li):return float(np.mean([average_precision_score(y[ix,li],q[n][ix,li]) for n in NAMES[2:]])-average_precision_score(y[ix,li],q['RA_LL_reversal'][ix,li]))
def bootstrap(y,base,arms):
 ip=OUT/'bootstrap_indices.npy';state=OUT/'bootstrap_state.npz';progress=OUT/'BOOTSTRAP_PROGRESS.json'
 if ip.exists():idx=np.load(ip,allow_pickle=False)
 else:
  idx=np.random.default_rng(BOOT_SEED).integers(0,len(y),(BOOT,len(y)),dtype=np.int32);q=ip.with_name('.bootstrap_indices.tmp.npy');np.save(q,idx);os.replace(q,ip)
 if idx.shape!=(BOOT,len(y)):raise RuntimeError('bootstrap indices shape mismatch')
 if state.exists():
  z=np.load(state,allow_pickle=False);done=z['done'].astype(bool);bc=z['baseline_contrasts'];ac=z['arm_contrasts']
  if done.shape!=(BOOT,) or bc.shape!=(BOOT,3,3) or ac.shape!=(BOOT,3,3,3):raise RuntimeError('bootstrap state shape mismatch')
 else:done=np.zeros(BOOT,bool);bc=np.full((BOOT,3,3),np.nan,np.float32);ac=np.full((BOOT,3,3,3),np.nan,np.float32)
 for b in range(3):
  for s in SEEDS:
   for li in range(3):
    direct=float(np.mean([average_precision_score(y[idx[b],li],base[s][n][idx[b],li]) for n in NAMES[2:]])-average_precision_score(y[idx[b],li],base[s]['RA_LL_reversal'][idx[b],li]))
    if not np.isclose(direct,contrast(y,idx[b],base[s],li),rtol=0,atol=1e-12):raise RuntimeError('bootstrap direct-formula smoke mismatch')
 aj(OUT/'BOOTSTRAP_SMOKE.json',{'status':'PASS','replicates_checked':3,'tolerance':1e-12,'implementation':'separate condition-level AUPRC; baseline contrast computed once per replicate/seed/endpoint'})
 started=time.monotonic();first=next((k for k in range(BOOT) if not done[k]),BOOT)
 for b in range(BOOT):
  if done[b]:continue
  ix=idx[b]
  for si,s in enumerate(SEEDS):
   for li in range(3):
    bc[b,si,li]=contrast(y,ix,base[s],li)
    for ai,arm in enumerate(ARMS):ac[b,ai,si,li]=contrast(y,ix,arms[arm][s],li)
  done[b]=True
  if done.sum()%10==0 or done.all():
   an(state,done=done,baseline_contrasts=bc,arm_contrasts=ac);elapsed=time.monotonic()-started;rate=(done.sum()-first)/elapsed if elapsed else 0;remaining=(BOOT-done.sum())/rate if rate else None
   aj(progress,{'completed_replicates':int(done.sum()),'total_replicates':BOOT,'indices_sha256':sha(ip),'state_sha256':sha(state),'estimated_remaining_seconds':remaining,'updated_utc':now()});status('BOOTSTRAP',completed_replicates=int(done.sum()),total_replicates=BOOT,estimated_remaining_seconds=remaining)
 out=[]
 for ai,arm in enumerate(ARMS):
  for li,label in enumerate(LABELS):
   for metric,x in [('matched_contrast',ac[:,ai,:,li].mean(1)),('contrast_reduction_vs_baseline',(bc[:,:,li]-ac[:,ai,:,li]).mean(1))]:out.append({'arm':arm,'endpoint':label,'metric':metric,'mean':float(x.mean()),'low95':float(np.quantile(x,.025)),'high95':float(np.quantile(x,.975)),'replicates':BOOT,'unit':'canonical_record','synchronized_across_arms_seeds_conditions':True})
 return out
def figs(primary):
 import matplotlib.pyplot as plt
 d=OUT/'figures';d.mkdir(exist_ok=True)
 for endpoint in LABELS:
  f,ax=plt.subplots(figsize=(7,4));q=[x for x in primary if x['endpoint']==endpoint]
  for arm,color in [('cached_clean_baseline','#333333'),('physics_aware','#4477AA'),('masking','#EE6677'),('combined','#228833')]:
   z=[x for x in q if x['arm']==arm];ax.plot([x['seed'] for x in z],[x['matched_contrast'] for x in z],'o-',label=arm.replace('_',' '),color=color)
  ax.axhline(.15,color='black',lw=1,ls='--');ax.set(xlabel='seed',ylabel='mean-mask minus RA–LL AUPRC',title=f'{endpoint}: matched contrast');ax.legend(fontsize=8);f.tight_layout();f.savefig(d/f'{endpoint.lower()}_matched_contrast.svg');f.savefig(d/f'{endpoint.lower()}_matched_contrast.png',dpi=220);plt.close(f)
def decide(primary,boots,gates):
 q=[x for x in primary if x['arm']=='cached_clean_baseline' and x['endpoint']=='TWA'];p=[x for x in primary if x['arm']=='physics_aware' and x['endpoint']=='TWA'];base={x['seed']:x for x in q};phy={x['seed']:x for x in p};mean=lambda a,k:float(np.mean([x[k] for x in a]));b={(x['arm'],x['endpoint'],x['metric']):x for x in boots}
 checks={'ra_ll_twa_improvement_each_seed_min_0_15':all(phy[s]['ra_ll_auprc']-base[s]['ra_ll_auprc']>=.15 for s in SEEDS),'aggregate_matched_contrast_decrease_min_0_15':mean(q,'matched_contrast')-mean(p,'matched_contrast')>=.15,'aggregate_matched_contrast_upper95_max_0_15':b[('physics_aware','TWA','matched_contrast')]['high95']<=.15,'clean_twa_mean_decline_max_0_02':mean(q,'clean_auprc')-mean(p,'clean_auprc')<=.02,'clean_twa_any_seed_decline_max_0_04':all(base[s]['clean_auprc']-phy[s]['clean_auprc']<=.04 for s in SEEDS),'mean_mask_twa_mean_decline_max_0_02':mean(q,'mean_mask_auprc')-mean(p,'mean_mask_auprc')<=.02,'mean_mask_twa_any_seed_decline_max_0_04':all(base[s]['mean_mask_auprc']-phy[s]['mean_mask_auprc']<=.04 for s in SEEDS),'all_physics_seeds_validation_and_integrity':all(g['decision']=='PASS' and g['evaluation_checkpoint_hash_match'] for g in gates if g['arm']=='physics_aware')}
 return ('PASS_M1_FREEZE_NINGBO_M2' if all(checks.values()) else 'FAIL_M1_NO_M2'),checks,{'baseline_mean_matched_contrast':mean(q,'matched_contrast'),'physics_mean_matched_contrast':mean(p,'matched_contrast'),'contrast_reduction':mean(q,'matched_contrast')-mean(p,'matched_contrast'),'physics_matched_contrast_bootstrap':b[('physics_aware','TWA','matched_contrast')]}
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('preflight','run'));args=a.parse_args();lock=acquire()
 try:
  pf=preflight();aj(OUT/'CACHED_ARTIFACT_PREFLIGHT.json',pf)
  if args.command=='preflight':status('PREFLIGHT_PASS');return
  split=json.loads((V4/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];ids=[x['record_id'] for x in split if x['split']=='test'];from src.chapman_phase1 import labels;y=np.stack([labels(x) for x in ids]).astype(np.int8)
  base,arms=predictions();th={'cached_clean_baseline':baseline_thresholds()};th.update({arm:{s:np.load(OUT/arm/f'seed_{s}'/'validation.npz')['thresholds'].astype(float) for s in SEEDS} for arm in ARMS});allrows=[];primary=[];gates=[];train=[]
  for arm,pred in [('cached_clean_baseline',base),*arms.items()]:r,p=result_rows(arm,pred,y,th[arm]);allrows+=r;primary+=p
  point_audit(primary)
  for arm in ARMS:
   for s in SEEDS:
    d=OUT/arm/f'seed_{s}';g=json.loads((d/'validation_gate.json').read_text());gates.append({'arm':arm,'seed':s,**g,'evaluation_complete':True,'evaluation_checkpoint_hash_match':True});train.append({'arm':arm,'seed':s,'validation_gate':g['decision'],'epochs_completed':len(g['history']),'best_validation_twa_auprc':max(x['twa_auprc'] for x in g['history']),'gpu_hours':g['gpu_hours'],'checkpoint_sha256':g['checkpoint_sha256'],'evaluation_conditions':794,'status':'COMPLETE'})
  wc(OUT/'TRAINING_STATUS_BY_ARM_SEED.csv',train);aj(OUT/'VALIDATION_GATES.json',{'all_pass':True,'gates':gates});wc(OUT/'RESULTS_BY_ARM_SEED.csv',allrows);wc(OUT/'PRIMARY_MITIGATION_CONTRASTS.csv',primary);b=bootstrap(y,base,arms);wc(OUT/'BOOTSTRAP_SUMMARY.csv',b);figs(primary);d,checks,twa=decide(primary,b,gates);out={'decision':d,'completed_utc':now(),'protocol_sha256':EXPECTED,'ningbo_accessed':False,'primary_arm':'physics_aware','checks':checks,'primary_twa':twa,'secondary_arms':['masking','combined']};aj(OUT/'conclusions_m1.json',out);(OUT/'FINAL_M1_DECISION.md').write_text('# '+d+'\n\n'+json.dumps(out,indent=2)+'\n\nM1 is a prespecified Chapman follow-up after observed Chapman outcomes, not a newly sealed confirmation. Ningbo M2 was not started.\n',encoding='utf8');readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');block='<!-- M1_STATUS_START -->\n## Mitigation M1\n\nChapman-only physics-aware mitigation is a prespecified follow-up. Final decision: `'+d+'`. See `outputs/mitigation_m1_v1/FINAL_M1_DECISION.md`; Ningbo M2 has not been started.\n<!-- M1_STATUS_END -->';readme.write_text(re.sub(r'<!-- M1_STATUS_START -->.*?<!-- M1_STATUS_END -->',block,text,flags=re.S) if '<!-- M1_STATUS_START -->' in text else text+'\n\n'+block+'\n',encoding='utf8');status('COMPLETE',decision=d)
 except KeyboardInterrupt:status('INTERRUPTED');raise
 except Exception as exc:aj(OUT/'conclusions_m1.json',{'decision':'STOP_TECHNICAL_BLOCKER','error':repr(exc),'completed_utc':now(),'ningbo_accessed':False});status('STOP_TECHNICAL_BLOCKER',error=repr(exc));raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
