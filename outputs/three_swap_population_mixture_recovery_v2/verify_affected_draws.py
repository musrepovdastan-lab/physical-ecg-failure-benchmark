from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
from outputs.mixed_fault_calibration_v1 import run_analysis as base
from analyze_recovery import load_fits,probability,threshold,scenario_weights,SEEDS,CONDITIONS,METHODS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 labels,_=base.heedb_labels();y=np.asarray(labels[:,0],dtype=np.uint8)
 ix=np.load(ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy',allow_pickle=False)
 fits=load_fits();qvalues=(.001,.005,.01,.05,.1,1.0);draw_ids=(0,417,999)
 methods=('Huang_full_pipeline','ordinary_resnet','physics_aware_p050','combined')
 contrasts=(('Huang_full_pipeline','ordinary_resnet'),('physics_aware_p050','Huang_full_pipeline'),('combined','Huang_full_pipeline'))
 scores={m:{s:{} for s in SEEDS} for m in methods}
 for m in methods:
  for seed in SEEDS:
   for c in CONDITIONS:
    with np.load(OUT/f'predictions/HEEDB/{m}/seed_{seed}/{c}.npz',allow_pickle=False) as z:scores[m][seed][c]=z['p'][:,0].astype(np.float64)
 saved=np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False);dnames=saved['draw_names'].astype(str);drawarr=saved['draws'];dmap={k:drawarr[i] for i,k in enumerate(dnames)}
 rows=[]
 for q in qvalues:
  coeff=scenario_weights('equal_three_swap',q)
  for draw_id in draw_ids:
   selected=ix[draw_id]
   for metric in ('raw_TWA_AP','raw_TWA_Brier','calibrated_TWA_Brier','sensitivity','FP_per_1000'):
    agg={m:[] for m in methods}
    for m in methods:
     for seed in SEEDS:
      raw=[scores[m][seed][c][selected] for c in CONDITIONS]
      fit=fits[m][seed][0];cal=[probability(fit,p,'calibrated') for p in raw]
      if metric=='raw_TWA_AP':val=average_precision_score(np.tile(y[selected],4),np.concatenate(raw),sample_weight=np.concatenate([np.full(len(selected),coeff[j]) for j in range(4)]))
      elif metric=='raw_TWA_Brier':val=np.average(np.concatenate([(p-y[selected])**2 for p in raw]),weights=np.concatenate([np.full(len(selected),coeff[j]) for j in range(4)]))
      elif metric=='calibrated_TWA_Brier':val=np.average(np.concatenate([(p-y[selected])**2 for p in cal]),weights=np.concatenate([np.full(len(selected),coeff[j]) for j in range(4)]))
      else:
       th=threshold(fit,'calibrated');pred=[p>=th for p in cal];pos=np.tile(y[selected]==1,4);neg=~pos
       w=np.concatenate([np.full(len(selected),coeff[j]) for j in range(4)]);d=np.concatenate(pred)
       if metric=='sensitivity':val=float(w[pos&d].sum()/w[pos].sum())
       else:val=float(1000*w[neg&d].sum()/w.sum())
      agg[m].append(float(val))
    for left,right in contrasts:
     expected=float(np.mean(agg[left])-np.mean(agg[right]))
     key=f"HEEDB|equal_three_swap|q={q:g}|{left}-{right}|{metric}"
     saved_val=float(dmap[key][draw_id]);delta=abs(saved_val-expected)
     rows.append({'source':'HEEDB','scenario':'equal_three_swap','q_fault':q,'draw_index':draw_id,'contrast':f'{left}-{right}','metric':metric,
      'sklearn_independent_difference':expected,'saved_difference':saved_val,'abs_delta':delta,'tolerance':1e-11,'status':'PASS' if delta<=1e-11 else 'FAIL'})
 if any(r['status']!='PASS' for r in rows):raise RuntimeError('affected saved draw differs from independent sklearn recomputation')
 pd.DataFrame(rows).to_csv(OUT/'AFFECTED_CONTRAST_DRAW_AUDIT.csv',index=False)
 result={'status':'PASS','rows':len(rows),'draw_indices':list(draw_ids),'q_positive_values':list(qvalues),'changed_contrast_draws_recomputed':True,
  'bootstrap_indices_sha256':sha(ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'),'max_abs_delta':max(r['abs_delta'] for r in rows)}
 (OUT/'AFFECTED_CONTRAST_DRAW_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
