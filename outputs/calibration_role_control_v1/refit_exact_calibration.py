from pathlib import Path
import sys,json,csv,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from outputs.mixed_fault_calibration_v1 import run_analysis as base
def atomic_json(dest,obj):
 dest=Path(dest);tmp=dest.with_name('.'+dest.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8');os.replace(tmp,dest)
def main():
 rows=[]
 for method in ('ordinary','physics_aware_p050','combined','Huang_full_pipeline'):
  for seed in (104729,130363,155921):
   for ei,endpoint in enumerate(('TWA','SB','STach')):
    aroot=OUT/f'scores/calibration_A/{method}/seed_{seed}.npz';broot=OUT/f'scores/threshold_B/{method}/seed_{seed}.npz'
    with np.load(aroot,allow_pickle=False) as z:pa=z['p'].copy();ya=z['y'].copy()
    with np.load(broot,allow_pickle=False) as z:pb=z['p'].copy();yb=z['y'].copy()
    a,b,info=base.fit_platt(pa[:,ei],ya[:,ei]);neg=pb[yb[:,ei]==0,ei]
    if not len(neg) or not info['success'] or not np.isfinite([a,b]).all() or a<=0:raise RuntimeError('exact historical Platt fit failed')
    q=float(np.quantile(neg,.9,method='linear'));mapped=float(base.platt(np.asarray([q]),a,b)[0]);rows.append({'method':method,'seed':seed,'endpoint':endpoint,'condition':'clean','status':'PASS','n_A':len(ya),'n_A_positive':int(ya[:,ei].sum()),'n_B_negative':len(neg),'raw_q90':q,'mapped_calibrated_threshold':mapped,'slope_a':a,'intercept_b':b,'theta0':float(np.log(a)),'fit_success':info['success'],'fit_message':info['message'],'fit_loss':info['loss'],'fit_iterations':info['iterations'],'objective_source_sha256':hashlib.sha256((ROOT/'outputs/mixed_fault_calibration_v1/run_analysis.py').read_bytes()).hexdigest()})
 if len(rows)!=36:raise RuntimeError('expected 36 calibrated thresholds')
 with (OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lock={'status':'LOCKED','entries':rows,'locked_utc':'2026-10-20','fit_data':'clean calibration_A only','threshold_data':'clean negative threshold_B only','locked_after_all_models_before_test':True,'quantile':'numpy linear','raw_decision':'>= raw q90','positive_slope':'a=exp(theta0)','exact_objective_source':'outputs/mixed_fault_calibration_v1/run_analysis.py fit_platt/platt','all_36_pass':True,'initial_equivalent_formulation_lock_preserved':'CALIBRATION_AND_THRESHOLD_LOCK_PRECISE_FORMULA_REVIEW.json'}
 lock['canonical_sha256']=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();atomic_json(OUT/'CALIBRATION_AND_THRESHOLD_LOCK.json',lock);print(json.dumps({'status':'LOCKED','fits':len(rows),'objective':lock['exact_objective_source']}))
if __name__=='__main__':main()
