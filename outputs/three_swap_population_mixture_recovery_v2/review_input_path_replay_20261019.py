"""Independent explicit-signal GPU replay of recovery scores on fixed blocks."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.chapman_phase1 import ECGResNetSmall
def swap(x,name):
 a=x.astype(np.float64);i=a[:,0];ii=a[:,1]
 if name=='RA_LA_reversal':u,v=-i,ii-i
 elif name=='RA_LL_reversal':u,v=i-ii,-ii
 elif name=='LA_LL_reversal':u,v=ii,i
 else:raise ValueError(name)
 limb=np.stack([u,v,v-u,-(u+v)/2,u-v/2,v-u/2],axis=1)
 return np.concatenate([limb,a[:,6:]],axis=1).astype(np.float32)
def infer(model,x,mean,std):
 a=np.ascontiguousarray((x[:,:,::5]-mean[None,:,None])/std[None,:,None],dtype=np.float32)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):return torch.sigmoid(model(torch.from_numpy(a).cuda())).float().cpu().numpy()
raw=np.load(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy',mmap_mode='r')
with np.load(OUT/'huang_detector_probabilities.npz',allow_pickle=False) as z:dp=z['probabilities'].copy();conds=z['conditions'].astype(str).tolist()
rows=[]
for seed in [104729,130363,155921]:
 md=ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}';state=torch.load(md/'checkpoint.pt',weights_only=True,map_location='cpu')
 model=ECGResNetSmall().cuda().eval();model.load_state_dict(state['state_dict'])
 with np.load(md/'preprocessing.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 for condition in ['RA_LA_reversal','LA_LL_reversal']:
  with np.load(OUT/f'predictions/HEEDB/Huang_full_pipeline/seed_{seed}/{condition}.npz') as z:saved=z['p'].copy()
  ci=conds.index(condition)
  for start,end,n in [(0,128,32),(4992,5120,128),(9984,10000,16)]:
   original=np.asarray(raw[start:end],dtype=np.float32);faulty=swap(original,condition);pclasses=dp[ci,start:end].argmax(1)
   corrected=faulty.copy()
   for code,name in {1:'RA_LA_reversal',2:'RA_LL_reversal',3:'LA_LL_reversal'}.items():
    ix=pclasses==code
    if ix.any():corrected[ix]=swap(faulty[ix],name)
   probs=infer(model,corrected,mean,std)
   delta=float(np.max(np.abs(saved[start:start+n]-probs[:n])))
   rows.append({'seed':seed,'condition':condition,'processing_block':[start,end],'compared_n':n,'max_abs_probability_difference':delta,'pass':delta<=1e-4})
 del model;torch.cuda.empty_cache()
result={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','checks':len(rows),'max_abs_probability_difference':max(r['max_abs_probability_difference'] for r in rows),
 'scope':'Independent first/middle/last block explicit NumPy signal construction and predicted inverse; frozen cached detector outputs, original weights and normalization; no adapter/correction helper imports; no full inference or fitting. Detector neural replay not independently repeated.','rows':rows}
(OUT/'INDEPENDENT_INPUT_PATH_REPLAY_20261019.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
if result['status']!='PASS':raise SystemExit(1)
