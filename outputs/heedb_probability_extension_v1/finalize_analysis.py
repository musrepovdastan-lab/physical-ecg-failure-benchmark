from __future__ import annotations
import hashlib,json,os,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p=Path(p);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 protocol=json.loads((OUT/'FROZEN_EXTENSION_PROTOCOL.json').read_text(encoding='utf-8'))
 ph=sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json')
 assert ph==(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').read_text().strip()
 replay=json.loads((OUT/'REPLAY_AUDIT.json').read_text(encoding='utf-8')); assert replay['status']=='PASS' and len(replay['replay_rows'])==18
 inference=json.loads((OUT/'INFERENCE_COMPLETE.json').read_text(encoding='utf-8')); assert inference['status']=='PASS' and len(inference['predictions'])==12
 idx=OUT/'HEEDB_BOOTSTRAP_INDICES.npy'; src=ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'
 assert sha(idx)==sha(src)==protocol['contrasts']['indices_sha256']
 # Check cohort/order bindings and identity hashes for all new external prediction files.
 import run_analysis as base
 lab,pat=base.heedb_labels()
 pred_audit=[]
 for arm in ('physics_p010','physics_p025'):
  for seed in (104729,130363,155921):
   model=protocol['new_models'][f'{arm}/{seed}']
   for cond in ('clean','RA_LL_reversal'):
    f=OUT/'predictions'/arm/f'seed_{seed}'/f'{cond}.npz'
    with np.load(f,allow_pickle=False) as z:
     assert z['p'].shape==(10000,3) and np.isfinite(z['p']).all()
     assert np.array_equal(z['labels'],lab) and np.array_equal(z['patient_ids'].astype(str),pat.astype(str))
     assert str(z['checkpoint_sha256'].item())==model['checkpoint_sha256']
     assert str(z['preprocessing_sha256'].item())==model['preprocessing_sha256']
     assert str(z['protocol_sha256'].item())==inference['protocol_sha256']
    pred_audit.append({'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'arm':arm,'seed':seed,'condition':cond,'n':10000,'order_labels_checkpoint_preprocessing_bound':True})
 # Recompute calibration metadata provenance, without fitting or changing any threshold.
 source_cal=ROOT/'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json'
 recovered=json.loads(source_cal.read_text(encoding='utf-8')); used=[]
 for arm in ('physics_p010','physics_p025'):
  for seed in (104729,130363,155921):
   item=recovered['new_fits'][f'{arm}/{seed}']
   for fit in item['calibration_fits']:
    used.append({'method':arm,'seed':seed,'endpoint':fit['endpoint'],'a':fit['a'],'b':fit['b'],'fit_split':item['fit_split'],'A_indices_sha256':item['A_indices_sha256'],'B_indices_sha256':item['B_indices_sha256'],'threshold_raw_validation_B':fit['threshold_raw_on_B_negative'],'threshold_calibrated_same_map':fit['threshold_calibrated_transform_of_raw'],'formula':item['fit_formula'],'full_validation_used_for_checkpoint_selection':item['full_validation_used_for_checkpoint_selection_before_A_B_split'],'source_json_sha256':sha(source_cal)})
 atomic_json(OUT/'CALIBRATION_PROVENANCE_USED.json',{'source':'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json','source_sha256':sha(source_cal),'HEEDB_refit':False,'models':used})
 rank=pd.read_csv(OUT/'CALIBRATION_RANKING_AUDIT.csv')
 assert int(rank['exact_threshold_decision_mismatches'].sum())==0
 assert float(rank['max_monotonicity_violation'].max())==0.0
 # Retain only the prespecified draws (the broad internal calculation also produced unreported exploratory intermediates).
 boot=OUT/'PAIRED_BOOTSTRAP_REPLICATES.npz'
 with np.load(boot,allow_pickle=False) as z: names=z['draw_names'].astype(str); draws=z['draws'].copy(); ih=str(z['indices_sha256'].item())
 keep=[]
 for i,name in enumerate(names):
  left,metric,mode,q=name.split('|'); qv=float(q.split('=')[1])
  if qv in (0.0,0.01,0.1,1.0) and ((metric=='auprc' and mode=='raw') or (metric=='brier' and mode=='calibrated')):keep.append(i)
 tmp=OUT/'.PAIRED_BOOTSTRAP_REPLICATES.partial.npz'
 np.savez_compressed(tmp,indices_sha256=np.asarray(ih),draw_names=names[keep],draws=draws[keep]);os.replace(tmp,boot)
 contrasts=pd.read_csv(OUT/'PAIRED_CONTRASTS_BOOTSTRAP.csv')
 assert len(contrasts)==48 and contrasts['replicates'].eq(1000).all() and len(keep)==48
 byname={n:d for n,d in zip(names[keep],draws[keep])}
 for row in contrasts.to_dict('records'):
  name=f"{row['contrast']}|{row['metric']}|{row['score_mode']}|q={row['q_fault']:g}"
  d=byname[name]
  assert abs(float(np.mean(d))-row['bootstrap_mean_difference'])<1e-12
  assert abs(float(np.quantile(d,.025))-row['ci_2_5'])<1e-12
  assert abs(float(np.quantile(d,.975))-row['ci_97_5'])<1e-12
 # A headless plotting backend avoids opening GUI windows and does not change metrics.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 mean=pd.read_csv(OUT/'MIXTURE_MEAN_SUMMARY.csv')
 for metric,mode,name in (('auprc','raw','TWA_RAW_AUPRC_BY_Q.png'),('brier','calibrated','TWA_CALIBRATED_BRIER_BY_Q.png'),('sensitivity','calibrated','TWA_CALIBRATED_SENSITIVITY_BY_Q.png')):
  fig,ax=plt.subplots(figsize=(8,5))
  for method in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined','physics_p010','physics_p025'):
   part=mean[(mean.method==method)&(mean.endpoint=='TWA')&(mean.score_mode==mode)&(mean.threshold_type!='historical_HEEDB_exploratory_v1')]
   ax.plot(part.q_fault,part[f'{metric}_mean'],marker='o',label=method)
  ax.set_xlabel('q_fault: reversal share');ax.set_ylabel(f'{metric} ({mode})');ax.set_title(f'HEEDB proxy TWA {metric}');ax.set_xscale('symlog',linthresh=.001);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/name,dpi=160);plt.close(fig)
 complete={'status':'PASS','protocol_sha256':ph,'inference_protocol_sha256':inference['protocol_sha256'],'prediction_count':len(pred_audit),'metric_rows':len(pd.read_csv(OUT/'MIXTURE_RESULTS_BY_SEED.csv')),'mean_summary_rows':len(mean),'prespecified_contrast_rows':len(contrasts),'bootstrap_replicates':1000,'bootstrap_indices_sha256':sha(idx),'replay_status':'PASS','calibration_decision_mismatches':0,'reserved_test_evaluated':False,'finalized_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 atomic_json(OUT/'ANALYSIS_COMPLETE.json',complete)
 print(json.dumps(complete,indent=2))
if __name__=='__main__':main()
