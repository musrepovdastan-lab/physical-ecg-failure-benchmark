"""Focused replay of the detected missing-fault bug; no full inference/training."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.chapman_phase1 import ECGResNetSmall
from src.heedb_huang_adapter_v1 import corrected_batch,electrical_reversal_batch
from src.ecg_physics import derive_display,named_reversal
if not torch.cuda.is_available():raise RuntimeError('Focused CUDA replay unavailable; code-level finding remains.')
raw=np.asarray(np.load(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy',mmap_mode='r')[:128],dtype=np.float32)
with np.load(OUT/'huang_detector_probabilities.npz') as z:det=z['probabilities'].copy();conditions=z['conditions'].astype(str).tolist()
def infer(model,x,mean,std):
 a=np.ascontiguousarray((x[:,:,::5]-mean[None,:,None])/std[None,:,None])
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):return torch.sigmoid(model(torch.from_numpy(a).cuda())).float().cpu().numpy()
rows=[]
for seed in [104729,130363,155921]:
 d=ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}'
 state=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True)
 model=ECGResNetSmall().cuda().eval();model.load_state_dict(state['state_dict'])
 with np.load(d/'preprocessing.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 for c in ['RA_LA_reversal','LA_LL_reversal']:
  ci=conditions.index(c);pred=det[ci,:128].argmax(1)
  fault=electrical_reversal_batch(raw,c)
  independent=np.concatenate([raw[:,:2],raw[:,6:]],axis=1)
  alt=derive_display(named_reversal(independent,c)).astype(np.float32)
  wrong=corrected_batch(raw,pred);right=corrected_batch(fault,pred)
  wp=infer(model,wrong,mean,std);rp=infer(model,right,mean,std)
  with np.load(OUT/f'predictions/HEEDB/Huang_full_pipeline/seed_{seed}/{c}.npz') as z:cached=z['p'][:128]
  truth={'RA_LA_reversal':1,'LA_LL_reversal':3}[c]
  restored=corrected_batch(fault,np.full(128,truth))
  identity=derive_display(independent).astype(np.float32)
  rows.append({'seed':seed,'condition':c,'processing_n':128,'compared_n':32,
   'fault_transform_path_max_abs_difference':float(np.max(np.abs(fault-alt))),
   'saved_vs_wrong_input_max_abs_probability_first32':float(np.max(np.abs(cached[:32]-wp[:32]))),
   'saved_vs_correct_input_max_abs_probability_first32':float(np.max(np.abs(cached[:32]-rp[:32]))),
   'wrong_vs_correct_input_max_abs_raw_difference':float(np.max(np.abs(wrong-right))),
   'true_inverse_vs_identity_rederived_max_abs_difference':float(np.max(np.abs(restored-identity))),
   'cached_replays_wrong_path_within_1e_4':bool(np.max(np.abs(cached[:32]-wp[:32]))<=1e-4)})
 del model;torch.cuda.empty_cache()
result={'overall_status':'FAIL_HUANG_HEEDB_NEW_FAULT_INPUT_PATH','affected_full_prediction_files':6,
 'affected_conditions':['RA_LA_reversal','LA_LL_reversal'],'affected_cohort':'HEEDB only',
 'meaning':'Detector processed corrupted ECG; submitted new diagnostic path applied predicted inverse to original clean ECG, omitting initial fault. Saved scores replay the wrong path.',
 'scope':'Focused first historical batch replay for three seeds and two conditions; no full inference, training, fitting or overwrite of predictions.',
 'rows':rows}
(OUT/'INFERENCE_PATH_REVIEW_20261019.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
