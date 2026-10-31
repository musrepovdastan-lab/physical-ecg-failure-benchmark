from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.heedb_huang_adapter_v1 import electrical_reversal_batch,detector_input
from src.huang_detector_infer_v2 import load_model

def sha_array(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 started=time.monotonic()
 rawp=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy'
 raw=np.load(rawp,mmap_mode='r',allow_pickle=False)
 with np.load(ROOT/'outputs/three_swap_population_mixture_v1/huang_detector_probabilities.npz',allow_pickle=False) as z:
  cached=z['probabilities'];conds=z['conditions'].astype(str).tolist();classes=z['class_names'].astype(str).tolist()
 model,checkpoint,tfv=load_model();rows=[];all_rows=[]
 for cond in ('RA_LA_reversal','LA_LL_reversal'):
  ci=conds.index(cond)
  for start,stop in ((0,32),(4992,5024),(9984,10000)):
   x=np.asarray(raw[start:stop],dtype=np.float32).copy()
   faulty=electrical_reversal_batch(x,cond)
   # This is the same preprocessing of the same incoming signal used at detector inference.
   features=np.asarray([detector_input(row) for row in faulty],dtype=np.float32)
   got=np.asarray(model.predict(features,verbose=0),dtype=np.float32)
   ref=np.asarray(cached[ci,start:stop],dtype=np.float32)
   delta=float(np.max(np.abs(got-ref)))
   rec={'condition':cond,'block':[start,stop],'n':stop-start,'incoming_raw_sha256':sha_array(faulty),
        'detector_feature_sha256':sha_array(features),'replayed_detector_probabilities_sha256':sha_array(got),
        'cached_detector_probabilities_sha256':sha_array(ref),'max_abs_probability_delta':delta,
        'tolerance':1e-5,'status':'PASS' if delta<=1e-5 else 'FAIL'}
   rows.append(rec);all_rows.append(got)
   if rec['status']!='PASS':raise RuntimeError(f'detector block replay mismatch: {rec}')
 result={'status':'PASS','tensorflow_version':tfv,'checkpoint':checkpoint,
  'checkpoint_index_sha256':sha_file(Path(checkpoint).with_suffix('.index')),
  'checkpoint_data_sha256':sha_file(Path(checkpoint).with_suffix('.data-00000-of-00001')),
  'cached_artifact_sha256':sha_file(ROOT/'outputs/three_swap_population_mixture_v1/huang_detector_probabilities.npz'),
  'source_order_sha256':sha_file(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv'),
  'class_order':classes,'rows':rows,'max_abs_probability_delta':max(x['max_abs_probability_delta'] for x in rows),
  'elapsed_seconds':time.monotonic()-started}
 (OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':result['status'],'rows':len(rows),'max_delta':result['max_abs_probability_delta']}))
if __name__=='__main__':main()
