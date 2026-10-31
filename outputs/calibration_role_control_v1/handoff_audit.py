import json,csv,hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.heedb_huang_adapter_v1 import electrical_reversal_batch,corrected_batch
from src import huang_detector_infer_v2 as detector
OUT=Path(__file__).resolve().parent
def shaarr(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def explicit_swap(x,name):
 a=np.asarray(x,dtype=np.float64);i=a[:,0:1];ii=a[:,1:2]
 if name=='RA_LA_reversal':ni,nj=-i,ii-i
 elif name=='RA_LL_reversal':ni,nj=i-ii,-ii
 elif name=='LA_LL_reversal':ni,nj=ii,i
 else:raise ValueError(name)
 limb=np.concatenate([ni,nj,nj-ni,-(ni+nj)/2,ni-nj/2,nj-ni/2],axis=1)
 return np.concatenate([limb,a[:,6:12]],axis=1).astype(np.float32)
def main():
 rows=list(csv.DictReader((ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv').open(encoding='utf-8-sig')));rows=[r for r in rows if r['role']=='calibration_A'][:4]
 can=json.loads((ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json').read_text(encoding='utf8'))['records'];by={r['record_id']:r for r in can};clean=np.stack([detector.load_full_display(by[r['record_id']]) for r in rows]).astype(np.float32);pristine=clean.copy();checks=[]
 for cond in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'):
  incoming=np.stack([detector.apply_condition(x,cond) for x in clean]);formula=explicit_swap(clean,cond);delta=float(np.max(np.abs(incoming-formula)))
  if delta>1e-4:raise RuntimeError('incoming simulation differs from independent formula')
  for cls in range(4):
   actual=corrected_batch(incoming,np.full(len(incoming),cls,dtype=np.int64));expected=incoming.copy() if cls==0 else explicit_swap(incoming,('','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')[cls]);d=float(np.max(np.abs(actual-expected)))
   if d>2e-4 or (cls==0 and not np.array_equal(actual,incoming)):raise RuntimeError(f'handoff mismatch {cond}/{cls}: {d}')
   checks.append({'incoming_fault':cond,'predicted_class':cls,'incoming_sha256':shaarr(incoming),'corrector_input_is_same_faulty_signal':True,'result_sha256':shaarr(actual),'explicit_composition_max_abs_delta':d,'class_zero_leaves_signal_unchanged':cls!=0 or np.array_equal(actual,incoming),'status':'PASS'})
  if not np.array_equal(clean,pristine):raise RuntimeError('clean input mutated')
 result={'status':'PASS','input_source':'four accepted calibration_A 500-Hz Ningbo records','original_raw_shape':list(clean.shape),'source_buffer_unchanged':True,'predicted_classes_tested':[0,1,2,3],'faults_tested':['RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'],'known_class_inverse_used_for_expected_formula':False,'checks':checks,'tolerance_incoming':1e-4,'tolerance_corrected':2e-4}
 (OUT/'HUANG_HANDOFF_ENGINEERING_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf8');print(json.dumps({'status':'PASS','checks':len(checks)}))
if __name__=='__main__':main()
