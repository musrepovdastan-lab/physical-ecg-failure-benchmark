"""Independent numerical recheck from saved prediction artifacts and bootstrap draws."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"outputs/mixed_fault_calibration_v1"))
from outputs.mixed_fault_calibration_v1 import run_analysis as base
from src import published_detector_multifault_v2 as mf
SEEDS=(104729,130363,155921); METHODS=("ordinary_resnet","physics_aware_p050","combined","Huang_full_pipeline")
CONDS=("clean","RA_LL_reversal","RA_LA_reversal","LA_LL_reversal"); ENDS=("TWA","SB","STach")
QS=(0.,.001,.005,.01,.05,.1,1.); SCENARIOS=("equal_three_swap","RA_LL_only")
METRICS=("AP","AUROC","Brier","ECE15","sensitivity","specificity","predicted_positive_fraction","FP_per_1000","FN_per_1000")
CHECK_COLS=(*METRICS,"TP","FP","TN","FN")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def calibrate(p,fit):
 p=np.asarray(p,dtype=np.float64);p=np.clip(p,1e-6,1-1e-6)
 z=float(fit['a'])*np.log(p/(1-p))+float(fit['b'])
 return 1/(1+np.exp(-np.clip(z,-745,745)))

def metric_independent(y,p,threshold,w):
 y=np.asarray(y,dtype=np.uint8);p=np.asarray(p,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
 a=w>0;ya=y[a];pa=p[a];wa=w[a]
 if len(np.unique(ya))!=2:raise RuntimeError('single-class weighted sample')
 call=p>=float(threshold);pos=y==1;neg=~pos
 tp=float(w[pos&call].sum());fn=float(w[pos&~call].sum());fp=float(w[neg&call].sum());tn=float(w[neg&~call].sum());total=float(w.sum())
 ece=0.0
 for b in range(15):
  lo=b/15;hi=(b+1)/15
  ix=(p>=lo)&((p<hi) if b<14 else (p<=hi))&(w>0)
  if ix.any():
   mass=float(w[ix].sum());ece+=mass/total*abs(float(np.average(y[ix],weights=w[ix]))-float(np.average(p[ix],weights=w[ix])))
 return {'AP':float(average_precision_score(ya,pa,sample_weight=wa)),'AUROC':float(roc_auc_score(ya,pa,sample_weight=wa)),
  'Brier':float(np.average((p-y)**2,weights=w)),'ECE15':float(ece),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),
  'predicted_positive_fraction':(tp+fp)/total,'FP_per_1000':1000*fp/total,'FN_per_1000':1000*fn/total,'TP':tp,'FP':fp,'TN':tn,'FN':fn}

def coeff(scenario,q):
 if scenario=='equal_three_swap':return np.array([1-q,q/3,q/3,q/3],dtype=float)
 return np.array([1-q,q,0.,0.],dtype=float)

def get_inputs():
 fits0=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['calibration_fits']
 n4=json.loads((ROOT/'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['fits']
 hu=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['fits_by_seed']
 fits={'ordinary_resnet':{s:fits0['ordinary_resnet'][str(s)] for s in SEEDS},
       'physics_aware_p050':{s:fits0['physics_aware_resnet'][str(s)] for s in SEEDS},
       'combined':{s:n4[f'combined/{s}'] for s in SEEDS},
       'Huang_full_pipeline':{int(s):v for s,v in hu.items()}}
 rows=mf.records();y_n=np.asarray([r['labels'] for r in rows],dtype=np.uint8);ids_n=np.asarray([r['record_id'] for r in rows]).astype(str)
 y_h,pids=base.heedb_labels();y_h=np.asarray(y_h,dtype=np.uint8);pids=np.asarray(pids).astype(str)
 scores={'Ningbo':{},'HEEDB':{}}
 for src in scores:
  y=y_n if src=='Ningbo' else y_h
  for method in METHODS:
   scores[src][method]={}
   for seed in SEEDS:
    scores[src][method][seed]={}
    for cond in CONDS:
     if src=='Ningbo':
      if method=='ordinary_resnet':path=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_clean_resnet_no_correction_{cond}_seed_{seed}.npz'
      elif method=='physics_aware_p050':path=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_physics_aware_no_correction_{cond}_seed_{seed}.npz'
      elif method=='Huang_full_pipeline':path=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_huang_argmax_correct_clean_resnet_{cond}_seed_{seed}.npz'
      else:path=ROOT/f'outputs/ningbo_four_arm_replication_v1/combined/seed_{seed}/test_predictions/{cond}.npz'
     else:path=OUT/f'predictions/HEEDB/{method}/seed_{seed}/{cond}.npz'
     with np.load(path,allow_pickle=False) as z:
      p=np.asarray(z['p'],dtype=np.float64)
      if src=='Ningbo' and 'record_ids' in z.files and not np.array_equal(z['record_ids'].astype(str),ids_n):raise RuntimeError(f'Ningbo record order: {path}')
      if src=='Ningbo' and method=='combined' and not np.array_equal(z['y'],y):raise RuntimeError(f'Ningbo combined labels: {path}')
      if src=='HEEDB' and (not np.array_equal(z['labels'],y) or not np.array_equal(z['patient_ids'].astype(str),pids)):raise RuntimeError(f'HEEDB bindings: {path}')
      if p.shape!=(len(y),3) or not np.isfinite(p).all():raise RuntimeError(f'Invalid probability artifact: {path}')
      scores[src][method][seed][cond]=p
 return fits,{'Ningbo':y_n,'HEEDB':y_h},scores

def main():
 for f in ['RECOVERY_PROTOCOL.json','CONDITION_METRICS_BY_SEED.csv','MIXTURE_METRICS_BY_SEED.csv','MIXTURE_METRICS_MEAN_SD.csv','PAIRED_CONTRASTS_POINT_AND_CI.csv','PAIRED_BOOTSTRAP_DRAWS.npz','RA_LL_ANCHOR_AUDIT.csv']:
  if not (OUT/f).exists():raise RuntimeError('missing result '+f)
 fits,labels,scores=get_inputs()
 cond_saved=pd.read_csv(OUT/'CONDITION_METRICS_BY_SEED.csv');mix_saved=pd.read_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv')
 cond_ref=[];mix_ref=[];weight_checks=0;max_weight_error=0.0
 for src in ('Ningbo','HEEDB'):
  ymat=labels[src]
  for method in METHODS:
   for seed in SEEDS:
    for ep,endpoint in enumerate(ENDS):
     fit=fits[method][seed][ep]
     for mode in ('raw','calibrated'):
      th=float(fit.get('threshold_raw_validation_B_90pct',fit.get('threshold_raw'))) if mode=='raw' else float(fit.get('threshold_calibrated_validation_B_90pct',fit.get('threshold_calibrated')))
      ps={c:(scores[src][method][seed][c][:,ep] if mode=='raw' else calibrate(scores[src][method][seed][c][:,ep],fit)) for c in CONDS}
      for ci,c in enumerate(CONDS):
       m=metric_independent(ymat[:,ep],ps[c],th,np.ones(len(ymat)))
       cond_ref.append({'source':src,'method':method,'seed':seed,'endpoint':endpoint,'condition':c,'score_mode':mode,**m})
      for scenario in SCENARIOS:
       for q in QS:
        cw=coeff(scenario,q);err=abs(cw.sum()-1);max_weight_error=max(max_weight_error,err);weight_checks+=1
        if err>1e-15 or np.any(cw<0):raise RuntimeError('invalid mixture weights')
        yy=np.tile(ymat[:,ep],4);pp=np.concatenate([ps[c] for c in CONDS]);ww=np.concatenate([np.full(len(ymat),a,dtype=np.float64) for a in cw])
        m=metric_independent(yy,pp,th,ww)
        mix_ref.append({'source':src,'method':method,'seed':seed,'endpoint':endpoint,'scenario':scenario,'q_fault':q,'score_mode':mode,**m})
 cond_ref=pd.DataFrame(cond_ref);mix_ref=pd.DataFrame(mix_ref)
 comp=[]
 for label,ref,saved,keys in [('condition',cond_ref,cond_saved,['source','method','seed','endpoint','condition','score_mode']),('mixture',mix_ref,mix_saved,['source','method','seed','endpoint','scenario','q_fault','score_mode'])]:
   join=ref.merge(saved,on=keys,suffixes=('_ref','_saved'),validate='one_to_one')
   for c in CHECK_COLS:
    err=np.max(np.abs(join[f'{c}_ref'].astype(float)-join[f'{c}_saved'].astype(float)))
    comp.append({'table':label,'metric':c,'rows':len(join),'max_abs_error':float(err)})
    if err>2e-11:raise RuntimeError(f'{label} {c} metric mismatch {err}')
 # Saved paired draws: recompute all means/percentiles from the raw draws.
 z=np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False);draws=z['draws'];names=z['draw_names'].astype(str)
 if draws.shape!=(700,1000) or len(names)!=700:raise RuntimeError('paired bootstrap artifact shape mismatch')
 draw_map={n:draws[i] for i,n in enumerate(names)}
 point=pd.read_csv(OUT/'PAIRED_CONTRASTS_POINT_AND_CI.csv');max_boot_err=0.0
 for row in point.to_dict('records'):
  key=f"{row['source']}|{row['scenario']}|q={row['q_fault']:g}|{row['contrast']}|{row['metric']}"
  d=draw_map[key]
  vals={'bootstrap_mean_difference':float(np.mean(d)),'ci_2_5':float(np.quantile(d,.025)),'ci_97_5':float(np.quantile(d,.975))}
  for k,v in vals.items():
   err=abs(v-float(row[k]));max_boot_err=max(max_boot_err,err)
   if err>2e-12:raise RuntimeError(f'bootstrap summary mismatch {key} {k}: {err}')
 # Historical anchor rows must all be reproduced within CSV precision.
 anchor=pd.read_csv(OUT/'RA_LL_ANCHOR_AUDIT.csv')
 if len(anchor)!=2*4*3*3*2*7 or not (anchor.status=='PASS').all():raise RuntimeError(f'RA-LL anchor rows/status mismatch: {len(anchor)}')
 # Check actual persisted draw indices 0/417/999 against independent sklearn sample weights.
 draw_check=[]
 for src,indexpath in [('Ningbo',ROOT/'outputs/ningbo_confirmation_v2/bootstrap_indices.npy'),('HEEDB',ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy')]:
  ix=np.load(indexpath,allow_pickle=False)
  for di in (0,417,999):
   mult=np.bincount(ix[di],minlength=len(labels[src]));selected=np.repeat(np.arange(len(mult)),mult);y=labels[src][:,0]
   for method in METHODS:
    seed=SEEDS[0];raw=np.stack([scores[src][method][seed][c][:,0] for c in CONDS])
    for scenario in SCENARIOS:
     w=coeff(scenario,.05);yy=np.tile(y[selected],4);pp=np.concatenate([raw[c,selected] for c in range(4)])
     sw=np.concatenate([np.full(len(selected),w[c]) for c in range(4)])
     got=float(average_precision_score(yy,pp,sample_weight=sw))
     ordered=np.tile(y,4);values=raw.reshape(-1);weights=np.concatenate([mult*w[c] for c in range(4)])
     weighted=float(average_precision_score(ordered,values,sample_weight=weights))
     err=abs(got-weighted);draw_check.append({'source':src,'draw':di,'method':method,'scenario':scenario,'q':.05,'max_abs_error_repeated_vs_weighted':err})
     if err>1e-12:raise RuntimeError('sample-weight vs repeated-row AP mismatch')
 audit={'status':'PASS','type':'internal independent numerical verification; not external review','condition_metric_rows_recomputed':len(cond_ref),
        'mixture_metric_rows_recomputed':len(mix_ref),'weight_sum_checks':weight_checks,'max_weight_sum_error':max_weight_error,
        'metric_comparisons':comp,'paired_bootstrap_draws':int(draws.shape[0]*draws.shape[1]),'paired_bootstrap_summary_max_abs_error':max_boot_err,
        'historical_RA_LL_anchor_rows':len(anchor),'historical_RA_LL_anchor_all_pass':True,'real_draws_sklearn_repeated_vs_weighted':len(draw_check),
        'real_draws_max_abs_error':max(x['max_abs_error_repeated_vs_weighted'] for x in draw_check),
        'bootstrap_indices_sha256':{'Ningbo':sha(ROOT/'outputs/ningbo_confirmation_v2/bootstrap_indices.npy'),'HEEDB':sha(ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy')},
        'reserved_test_evaluated':False,'training_or_fitting':False}
 (OUT/'INDEPENDENT_NUMERICAL_VERIFICATION.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 (OUT/'VERIFICATION_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(audit,indent=2))
 
if __name__=='__main__':main()
