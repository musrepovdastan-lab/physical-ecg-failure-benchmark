import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import extension as e
def main():
 e.protocol();data,_,_=e.source_arrays();fits=e.fits_by_method();rows=[]
 for method in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined','physics_p010','physics_p025'):
  for seed in e.SEEDS:
   y,clean,rev=data[method][seed]
   for ep,name in enumerate(e.ENDS):
    fit=fits[method][seed][ep];tr=float(fit.get('threshold_raw_validation_B_90pct',fit.get('threshold_raw')));tc=float(fit.get('threshold_calibrated_validation_B_90pct',fit.get('threshold_calibrated')))
    for cond,raw in (('clean',clean[:,ep]),('RA_LL_reversal',rev[:,ep])):
     cal=e.base.platt(raw,float(fit['a']),float(fit['b']));order=np.argsort(raw,kind='stable');ordered=cal[order]
     rows.append({'method':method,'seed':seed,'endpoint':name,'condition':cond,'raw_unique_scores':int(np.unique(raw).size),'calibrated_unique_scores':int(np.unique(cal).size),'added_ties':int(np.unique(raw).size-np.unique(cal).size),'clipped_low_count':int(np.sum(raw<1e-6)),'clipped_high_count':int(np.sum(raw>1-1e-6)),'raw_ties_at_threshold':int(np.sum(raw==tr)),'calibrated_ties_at_threshold':int(np.sum(cal==tc)),'max_monotonicity_violation':float(max(0.,-np.diff(ordered).min())),'exact_threshold_decision_mismatches':int(np.sum((raw>=tr)!=(cal>=tc)))})
 p=OUT/'CALIBRATION_RANKING_AUDIT.csv';tmp=p.with_suffix('.csv.partial')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(p)
 bad=[r for r in rows if r['exact_threshold_decision_mismatches'] or r['max_monotonicity_violation']>0]
 print({'rows':len(rows),'mismatches':sum(r['exact_threshold_decision_mismatches'] for r in rows),'bad_rows':len(bad)})
 if bad:raise SystemExit('calibration mapping decision audit failed')
if __name__=='__main__':main()
