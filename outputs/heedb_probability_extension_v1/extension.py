from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
N=ROOT/'outputs/augmentation_probability_study_v1/training/Ningbo'; PRIVATE=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1'
SEEDS=(104729,130363,155921); ENDS=('TWA','SB','STach'); Q=(0.,.001,.005,.01,.05,.1,1.)
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
import run_analysis as base
import extend_bootstrap_metrics as fastboot
from src.chapman_phase1 import ECGResNetSmall

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p=Path(p);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(t,p)
def atomic_npz(path,**arrays):
 path=Path(path);t=path.with_name('.'+path.stem+'.partial.npz');np.savez_compressed(t,**arrays);os.replace(t,path)
def write_csv(p,rows):
 p=Path(p);fields=list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix('.csv.partial')
 with t.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def protocol():
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json'; q=json.loads(p.read_text(encoding='utf-8'))
 if sha(p)!=(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').read_text().strip():raise RuntimeError('frozen protocol hash mismatch')
 for rel,item in q['bound_inputs'].items():
  if sha(ROOT/rel)!=item['sha256']:raise RuntimeError('frozen source changed: '+rel)
 return q
def apply(x,mean,std):return ((x-mean[None,:,None])/std[None,:,None]).astype(np.float32,copy=False)
def reverse_physical(raw):
 w=__import__('outputs.ningbo_four_arm_replication_v1.workflow',fromlist=['reverse_raw'])
 return np.stack([w.reverse_raw(np.asarray(row,dtype=np.float32),'RA_LL_reversal') for row in raw])
def predict(model,raw,mean,std):
 model.eval();out=[]
 with torch.inference_mode():
  for s in range(0,len(raw),128):
   z=apply(np.asarray(raw[s:s+128],dtype=np.float32),mean,std)
   with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(z)).cuda())).float().cpu().numpy())
 return np.concatenate(out)
def load_new(arm,seed):
 d=N/arm/f'seed_{seed}'; ck=ROOT/d/'checkpoint.pt'; prep=ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz';state=torch.load(ck,weights_only=True,map_location='cuda')
 if state.get('arm')!=arm or int(state.get('seed'))!=seed:raise RuntimeError(f'checkpoint identity mismatch {arm}/{seed}')
 z=np.load(prep,allow_pickle=False);mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 model=ECGResNetSmall().cuda();model.load_state_dict(state['state_dict']);model.eval()
 return model,mean,std
def replay():
 p=protocol(); tol=float(p['replay']['tolerance_max_abs_probability_delta']); htol=tol
 xh=np.load(PRIVATE/'signals_10000_float32.npy',mmap_mode='r',allow_pickle=False); old=np.load(PRIVATE/'predictions_float32.npy',mmap_mode='r',allow_pickle=False)
 nt=np.load(ROOT/'outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy',mmap_mode='r',allow_pickle=False); rows=[]
 for si,seed in enumerate(SEEDS):
  # Control: frozen ordinary model reproduces its original HEEDB scores on two complete 128-record batches.
  model,mean,std,_=__import__('src.heedb_external_v1',fromlist=['load_model']).load_model('baseline',seed)
  for ci,condition in enumerate(('clean','RA_LL_reversal')):
   raw=np.asarray(xh[:256],dtype=np.float32)[:,:,::5]
   if ci:raw=reverse_physical(raw)
   got=predict(model,raw,mean,std);ref=np.asarray(old[0,si,ci,:256])
   d=float(np.max(np.abs(got-ref)));rows.append({'kind':'existing_control','arm':'ordinary_resnet','seed':seed,'condition':condition,'n':256,'batch_size':128,'max_abs_probability_delta':d,'tolerance':tol,'status':'PASS' if d<=tol else 'FAIL'})
  del model;torch.cuda.empty_cache()
  model,mean,std=load_new('physics_p010',seed)
  for condition in ('clean','RA_LL_reversal'):
   f=ROOT/'outputs/augmentation_probability_study_v1/predictions/Ningbo/physics_p010'/f'seed_{seed}'/(condition+'.npz')
   with np.load(f,allow_pickle=False) as z:ref=z['p'][:256].astype(np.float32)
   raw=np.asarray(nt[:256],dtype=np.float32)
   if condition=='RA_LL_reversal':raw=reverse_physical(raw)
   got=predict(model,raw,mean,std);d=float(np.max(np.abs(got-ref)))
   rows.append({'kind':'new_model_ningbo_replay','arm':'physics_p010','seed':seed,'condition':condition,'n':256,'batch_size':128,'max_abs_probability_delta':d,'tolerance':htol,'status':'PASS' if d<=htol else 'FAIL'})
  del model;torch.cuda.empty_cache()
  model,mean,std=load_new('physics_p025',seed)
  for condition in ('clean','RA_LL_reversal'):
   f=ROOT/'outputs/augmentation_probability_study_v1/predictions/Ningbo/physics_p025'/f'seed_{seed}'/(condition+'.npz')
   with np.load(f,allow_pickle=False) as z:ref=z['p'][:256].astype(np.float32)
   raw=np.asarray(nt[:256],dtype=np.float32)
   if condition=='RA_LL_reversal':raw=reverse_physical(raw)
   got=predict(model,raw,mean,std);d=float(np.max(np.abs(got-ref)))
   rows.append({'kind':'new_model_ningbo_replay','arm':'physics_p025','seed':seed,'condition':condition,'n':256,'batch_size':128,'max_abs_probability_delta':d,'tolerance':htol,'status':'PASS' if d<=htol else 'FAIL'})
  del model;torch.cuda.empty_cache()
 write_csv(OUT/'REPLAY_AUDIT.csv',rows)
 bad=[r for r in rows if r['status']!='PASS']
 audit={'status':'PASS' if not bad else 'STOP_REPLAY_MISMATCH','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'replay_rows':rows,'diagnosis':'none; 128-record batch, frozen input path, exact stored transforms' if not bad else 'Mismatch observed; full HEEDB inference halted pending diagnosis; tolerance unchanged'}
 atomic_json(OUT/'REPLAY_AUDIT.json',audit);print(json.dumps({'status':audit['status'],'rows':len(rows),'max_delta':max(float(r['max_abs_probability_delta']) for r in rows)},indent=2))
 if bad:raise RuntimeError('Replay gate failed; full inference stopped')
def inference():
 p=protocol(); audit=json.loads((OUT/'REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 if audit.get('status')!='PASS':raise RuntimeError('replay gate missing')
 x=np.load(PRIVATE/'signals_10000_float32.npy',mmap_mode='r',allow_pickle=False)
 labels,patients=base.heedb_labels()
 sel=__import__('pandas').read_csv(PRIVATE/'selection_initial.csv',dtype=str,keep_default_na=False)
 # Every inference artifact is independently resumable and must match frozen bindings.
 rows=[]; start=time.monotonic()
 for arm in ('physics_p010','physics_p025'):
  for seed in SEEDS:
   dest=OUT/'predictions'/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
   model,mean,std=load_new(arm,seed);cksha=sha(N/arm/f'seed_{seed}'/'checkpoint.pt');prsha=sha(ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz')
   for condition in ('clean','RA_LL_reversal'):
    f=dest/f'{condition}.npz'
    if f.exists():
     with np.load(f,allow_pickle=False) as z:
      valid=z['p'].shape==(10000,3) and np.array_equal(z['labels'],labels) and np.array_equal(z['patient_ids'].astype(str),patients.astype(str)) and str(z['checkpoint_sha256'].item())==cksha and str(z['preprocessing_sha256'].item())==prsha and str(z['condition'].item())==condition
     if not valid:raise RuntimeError('invalid partial prediction artifact: '+str(f))
     rows.append({'arm':arm,'seed':seed,'condition':condition,'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'status':'REUSED'})
     continue
    preds=[]
    for s in range(0,10000,128):
     raw=np.asarray(x[s:min(s+128,10000)],dtype=np.float32)[:,:,::5]
     if condition=='RA_LL_reversal':raw=reverse_physical(raw)
     preds.append(predict(model,raw,mean,std))
     if s%1280==0: print(f'INFER {arm}/{seed}/{condition}: {min(s+128,10000)}/10000',flush=True)
    pred=np.concatenate(preds).astype(np.float32)
    if pred.shape!=(10000,3) or not np.isfinite(pred).all():raise RuntimeError('invalid HEEDB inference output')
    atomic_npz(f,p=pred,labels=labels,patient_ids=patients.astype(str),file_names=sel['FileName'].astype(str).to_numpy(),condition=np.asarray(condition),arm=np.asarray(arm),seed=np.asarray(seed),checkpoint_sha256=np.asarray(cksha),preprocessing_sha256=np.asarray(prsha),protocol_sha256=np.asarray(sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json')))
    rows.append({'arm':arm,'seed':seed,'condition':condition,'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'status':'COMPLETE'})
   del model;torch.cuda.empty_cache()
   atomic_json(OUT/'INFERENCE_PROGRESS.json',{'completed':rows,'elapsed_seconds':time.monotonic()-start,'status':'RUNNING'})
 atomic_json(OUT/'INFERENCE_COMPLETE.json',{'status':'PASS','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'new_prediction_count':len(rows),'predictions':rows,'elapsed_seconds':round(time.monotonic()-start,2),'reserved_test_evaluated':False})
 print(json.dumps({'status':'PASS','prediction_count':len(rows),'elapsed_seconds':round(time.monotonic()-start,1)},indent=2))

def metrics(y,p,threshold,w=None):return base.metrics(y,p,threshold,w)
def fits_by_method():
 v2=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['calibration_fits']
 hu=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['fits_by_seed']
 n4=json.loads((ROOT/'outputs/heedb_four_arm_extension_v1/CALIBRATION_PROVENANCE.json').read_text(encoding='utf-8'))['fits_by_method_and_seed']
 fits={}
 for m in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet'):
  fits[m]={int(seed):v for seed,v in v2[m].items()}
 rec=json.loads((ROOT/'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json').read_text(encoding='utf-8'))['new_fits']
 for m in ('physics_p010','physics_p025'):
  fits[m]={}
  for seed in SEEDS:
   entry=rec[f'{m}/{seed}']; fits[m][seed]={}
   for ep,fit in enumerate(entry['calibration_fits']):
    fits[m][seed][ep]={'a':fit['a'],'b':fit['b'],'threshold_raw':fit['threshold_raw_on_B_negative'],'threshold_calibrated':fit['threshold_calibrated_transform_of_raw'],'threshold_raw_validation_B_90pct':fit['threshold_raw_on_B_negative'],'threshold_calibrated_validation_B_90pct':fit['threshold_calibrated_transform_of_raw']}
 for m in ('masking','combined'):fits[m]={seed:n4[m][str(seed)] for seed in SEEDS}
 fits['huang_pipeline']={int(s):v for s,v in hu.items()}
 return fits
def source_arrays():
 data,_,_=base.get_data();labels,patients=base.heedb_labels();new={}
 for arm in ('physics_p010','physics_p025'):
  new[arm]={}
  for seed in SEEDS:
   pair=[]
   for cond in ('clean','RA_LL_reversal'):
    with np.load(OUT/'predictions'/arm/f'seed_{seed}'/f'{cond}.npz',allow_pickle=False) as z:
     if not np.array_equal(z['labels'],labels) or not np.array_equal(z['patient_ids'].astype(str),patients.astype(str)):raise RuntimeError('new prediction order/label binding mismatch')
     pair.append(z['p'].astype(np.float64))
   new[arm][seed]=(labels,pair[0],pair[1])
 for method in ('masking','combined'):
  new[method]={}
  for seed in SEEDS:
   pair=[]
   for cond in ('clean','RA_LL_reversal'):
    with np.load(ROOT/'outputs/heedb_four_arm_extension_v1/predictions'/method/f'seed_{seed}'/f'{cond}.npz',allow_pickle=False) as z:
     if not np.array_equal(z['labels'],labels) or not np.array_equal(z['patient_ids'].astype(str),patients.astype(str)):raise RuntimeError('historical prediction binding mismatch')
     pair.append(z['p'].astype(np.float64))
   new[method][seed]=(labels,pair[0],pair[1])
 for method in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline'):
  for seed in SEEDS:
   y,c,r=data[base.SOURCES[1]][method][seed]
   if not np.array_equal(y,labels):raise RuntimeError('reused HEEDB labels disagree')
   new.setdefault(method,{})[seed]=(y,c.astype(np.float64),r.astype(np.float64))
 return new,labels,patients
def analyze():
 protocol();done=json.loads((OUT/'INFERENCE_COMPLETE.json').read_text(encoding='utf-8'))
 if done.get('status')!='PASS':raise RuntimeError('missing complete inference')
 data,labels,patients=source_arrays();fits=fits_by_method();rows=[];clip=[]
 methods=('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined','physics_p010','physics_p025')
 for method in methods:
  for seed in SEEDS:
   y,clean,rev=data[method][seed]
   for ep,name in enumerate(ENDS):
    fit=fits[method][seed][ep]
    threshold_raw=float(fit.get('threshold_raw_validation_B_90pct',fit.get('threshold_raw')))
    threshold_cal=float(fit.get('threshold_calibrated_validation_B_90pct',fit.get('threshold_calibrated')))
    # Existing Huang full-pipeline validation-B fit is the primary source-validation operating point.
    for mode in ('raw','calibrated'):
     a,b=(1.,0.) if mode=='raw' else (float(fit['a']),float(fit['b']))
     pc,pr=(clean[:,ep],rev[:,ep]) if mode=='raw' else (base.platt(clean[:,ep],a,b),base.platt(rev[:,ep],a,b))
     th=threshold_raw if mode=='raw' else threshold_cal
     for q in Q:
      yy=np.r_[y[:,ep],y[:,ep]];pp=np.r_[pc,pr];w=np.r_[np.full(len(y),1-q),np.full(len(y),q)]
      met=metrics(yy,pp,th,w)
      rows.append({'method':method,'seed':seed,'endpoint':name,'q_fault':q,'score_mode':mode,'threshold_type':'validation_B_90pct','threshold':th,**met})
     if mode=='raw':
      cp=base.platt(clean[:,ep],float(fit['a']),float(fit['b']));rp=base.platt(rev[:,ep],float(fit['a']),float(fit['b']))
      for cond,raw,cal in (('clean',clean[:,ep],cp),('RA_LL_reversal',rev[:,ep],rp)):
       order=np.argsort(raw,kind='stable');ordered=cal[order]
       clip.append({'method':method,'seed':seed,'endpoint':name,'condition':cond,'raw_unique_scores':int(np.unique(raw).size),'calibrated_unique_scores':int(np.unique(cal).size),'added_ties':int(np.unique(raw).size-np.unique(cal).size),'clipped_low_count':int(np.sum(raw<1e-6)),'clipped_high_count':int(np.sum(raw>1-1e-6)),'raw_ties_at_threshold':int(np.sum(raw==threshold_raw)),'calibrated_ties_at_threshold':int(np.sum(cal==threshold_cal)),'max_monotonicity_violation':float(max(0.,-np.diff(ordered).min())),'exact_threshold_decision_mismatches':int(np.sum((raw>=threshold_raw)!=(cal>=threshold_cal)))})
      # Huang own full pipeline validation-B threshold is above. Keep historical comparator as a separate, explicitly labelled sensitivity analysis.
      if method=='huang_pipeline':
       old=json.loads((ROOT/'outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))['thresholds']['baseline'][str(seed)][ep]
       for q in Q:
        yy=np.r_[y[:,ep],y[:,ep]];pp=np.r_[clean[:,ep],rev[:,ep]];w=np.r_[np.full(len(y),1-q),np.full(len(y),q)]
        rows.append({'method':method,'seed':seed,'endpoint':name,'q_fault':q,'score_mode':'raw_historical_threshold','threshold_type':'historical_HEEDB_exploratory_v1','threshold':old,**metrics(yy,pp,float(old),w)})
 write_csv(OUT/'MIXTURE_RESULTS_BY_SEED.csv',rows);write_csv(OUT/'CALIBRATION_RANKING_AUDIT.csv',clip)
 # Mean and SD across three independently evaluated seeds.
 import pandas as pd
 df=pd.DataFrame(rows); metrics_cols=['auprc','auroc','brier','ece15','sensitivity','specificity','predicted_positive_fraction']
 grp=df.groupby(['method','endpoint','q_fault','score_mode','threshold_type'],dropna=False)
 means=grp[metrics_cols].agg(['mean','std']).reset_index();means.columns=['_'.join(str(x) for x in c if x!='').rstrip('_') if isinstance(c,tuple) else c for c in means.columns]
 means.to_csv(OUT/'MIXTURE_MEAN_SUMMARY.csv',index=False)
 # Paired patient bootstrap. Pre-sort fixed clean/reversal scores once; identical v1 fast metric implementation.
 fastboot.validate_fast_metrics()
 ix=np.load(OUT/'HEEDB_BOOTSTRAP_INDICES.npy',allow_pickle=False);multiplicities=np.stack([np.bincount(row,minlength=len(labels)) for row in ix]).astype(np.int32)
 pairs=[('physics_p025','physics_aware_resnet','physics_p025_minus_physics_aware_resnet','prespecified'),('physics_p025','combined','physics_p025_minus_combined','prespecified'),('physics_p010','ordinary_resnet','physics_p010_minus_ordinary_resnet','prespecified'),('physics_p010','physics_aware_resnet','physics_p010_minus_physics_aware_resnet','prespecified'),('physics_p010','huang_pipeline','physics_p010_minus_huang_pipeline','prespecified'),('physics_p025','huang_pipeline','physics_p025_minus_huang_pipeline','prespecified')]
 bootrows=[];bootstrap_draws={}; fast_cache={}; full_values={}
 bootmetrics=('auprc','auroc','brier','sensitivity','specificity')
 for method in methods:
  for mode in ('raw','calibrated'):
   for seed in SEEDS:
    y,c,r=data[method][seed];fit=fits[method][seed][0]
    if mode=='raw':cuse,ruse=c[:,0],r[:,0];threshold=float(fit.get('threshold_raw_validation_B_90pct',fit.get('threshold_raw')))
    else:cuse,ruse=base.platt(c[:,0],fit['a'],fit['b']),base.platt(r[:,0],fit['a'],fit['b']);threshold=float(fit.get('threshold_calibrated_validation_B_90pct',fit.get('threshold_calibrated')))
    fast_cache[(method,mode,seed)]=fastboot.make_cache(y[:,0],cuse,ruse,threshold)
 for mode in ('raw','calibrated'):
  for q in Q:
   for method in methods:
    full_values[(method,mode,q)]={}
    for seed in SEEDS:
     cache=fast_cache[(method,mode,seed)]
     full_values[(method,mode,q)][seed]=fastboot.fast_metrics(cache,np.ones(len(labels),dtype=np.int32),q)
   seed_means=np.empty((len(methods),len(bootmetrics),1000),dtype=np.float64)
   for bi,multiplicity in enumerate(multiplicities):
    if bi%100==0:print(f'HEEDB fast bootstrap {mode} q={q:g}: {bi}/1000',flush=True)
    for mi,method in enumerate(methods):
     perseed=[fastboot.fast_metrics(fast_cache[(method,mode,seed)],multiplicity,q) for seed in SEEDS]
     for xi,metric in enumerate(bootmetrics):seed_means[mi,xi,bi]=float(np.mean([v[metric] for v in perseed]))
   method_i={m:i for i,m in enumerate(methods)}
   for left,right,contrast,priority in pairs:
    for xi,metric in enumerate(bootmetrics):
     draws=seed_means[method_i[left],xi]-seed_means[method_i[right],xi]
     bootstrap_draws[f'{contrast}|{metric}|{mode}|q={q:g}']=draws.copy()
     leftfull=float(np.mean([full_values[(left,mode,q)][s][metric] for s in SEEDS]));rightfull=float(np.mean([full_values[(right,mode,q)][s][metric] for s in SEEDS]))
     bootrows.append({'endpoint':'TWA','q_fault':q,'score_mode':mode,'contrast':contrast,'priority':priority,'metric':metric,'left_method':left,'right_method':right,'left_full_sample_seed_mean':leftfull,'right_full_sample_seed_mean':rightfull,'full_sample_difference':leftfull-rightfull,'bootstrap_mean_difference':float(draws.mean()),'ci_2_5':float(np.quantile(draws,.025)),'ci_97_5':float(np.quantile(draws,.975)),'replicates':1000})
 bootrows=[r for r in bootrows if r['q_fault'] in (0.0,0.01,0.1,1.0) and ((r['metric']=='auprc' and r['score_mode']=='raw') or (r['metric']=='brier' and r['score_mode']=='calibrated'))]
 write_csv(OUT/'PAIRED_CONTRASTS_BOOTSTRAP.csv',bootrows)
 # Store only the prespecified TWA raw AUPRC and calibrated Brier contrasts.
 bootstrap_draws={k:v for k,v in bootstrap_draws.items() if any(f'|{metric}|{mode}|q={q:g}' in k for metric,mode in (('auprc','raw'),('brier','calibrated')) for q in (0,0.01,0.1,1))}
 draw_names=list(bootstrap_draws)
 atomic_npz(OUT/'PAIRED_BOOTSTRAP_REPLICATES.npz',indices_sha256=np.asarray(sha(OUT/'HEEDB_BOOTSTRAP_INDICES.npy')),draw_names=np.asarray(draw_names),draws=np.stack([bootstrap_draws[k] for k in draw_names]))
 # Plots for the three prespecified TWA summaries.
 import matplotlib.pyplot as plt
 mean_df=pd.read_csv(OUT/'MIXTURE_MEAN_SUMMARY.csv')
 for metric,mode,name in (('auprc','raw','TWA_AUPRC_BY_Q.png'),('brier','calibrated','TWA_CALIBRATED_BRIER_BY_Q.png'),('sensitivity','calibrated','TWA_CALIBRATED_SENSITIVITY_BY_Q.png')):
  fig,ax=plt.subplots(figsize=(8,5))
  for method in methods:
   part=mean_df[(mean_df.method==method)&(mean_df.endpoint=='TWA')&(mean_df.score_mode==mode)]
   ax.plot(part.q_fault,part[f'{metric}_mean'],marker='o',label=method)
  ax.set_xlabel('q: reversal share');ax.set_ylabel(metric);ax.set_title(f'HEEDB proxy TWA {metric} ({mode})');ax.set_xscale('symlog',linthresh=.001);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/name,dpi=160);plt.close(fig)
 complete={'status':'PASS','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'prediction_units':len(done['predictions']),'metric_rows':len(rows),'contrast_rows':len(bootrows),'bootstrap_replicates':1000,'bootstrap_indices_sha256':sha(OUT/'HEEDB_BOOTSTRAP_INDICES.npy'),'reserved_test_evaluated':False,'elapsed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 atomic_json(OUT/'ANALYSIS_COMPLETE.json',complete);print(json.dumps(complete,indent=2))

def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=('replay','infer','analyze'));cmd=a.parse_args().command
 if cmd=='replay':replay()
 elif cmd=='infer':inference()
 else:analyze()
if __name__=='__main__':main()
