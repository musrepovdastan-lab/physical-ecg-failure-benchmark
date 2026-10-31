"""Independent explicit-signal replay on selected HEEDB blocks; no fitting."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.chapman_phase1 import ECGResNetSmall
def swap(x,name):
 if name=='clean':return x.astype(np.float32,copy=True)
 a=x.astype(np.float64);i=a[:,0];ii=a[:,1]
 if name=='RA_LA_reversal':u,v=-i,ii-i
 elif name=='RA_LL_reversal':u,v=i-ii,-ii
 elif name=='LA_LL_reversal':u,v=ii,i
 else:raise ValueError(name)
 return np.concatenate([np.stack([u,v,v-u,-(u+v)/2,u-v/2,v-u/2],axis=1),a[:,6:]],axis=1).astype(np.float32)
def forward(model,x,mean,std):
 z=np.ascontiguousarray((x[:,:,::5]-mean[None,:,None])/std[None,:,None],dtype=np.float32)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):return torch.sigmoid(model(torch.from_numpy(z).cuda())).float().cpu().numpy()
raw=np.load(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy',mmap_mode='r')
with np.load(ROOT/'outputs/three_swap_population_mixture_recovery_v2/huang_detector_probabilities.npz',allow_pickle=False) as z:dp=z['probabilities'].copy();detconds=z['conditions'].astype(str).tolist()
conditions=['clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'];rows=[]
for seed in [104729,130363,155921]:
 for method in ['ordinary','physics_aware_p050','combined']:
  d=OUT/f'fits/{method}/seed_{seed}';st=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True)
  model=ECGResNetSmall().cuda().eval();model.load_state_dict(st['state_dict'])
  with np.load(d/'preprocessing.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
  methods=[method,'Huang_full_pipeline'] if method=='ordinary' else [method]
  for name in methods:
   for condition in conditions:
    with np.load(OUT/f'predictions/HEEDB/{name}/seed_{seed}/{condition}.npz',allow_pickle=False) as z:saved=z['p'].copy()
    blocks=[(0,128,32)]+([(4992,5120,128),(9984,10000,16)] if name=='Huang_full_pipeline' else [])
    for lo,hi,n in blocks:
     incoming=swap(np.asarray(raw[lo:hi],np.float32),condition);changed=incoming.copy()
     if name=='Huang_full_pipeline':
      cls=dp[detconds.index(condition),lo:hi].argmax(1)
      for code,fault in {1:'RA_LA_reversal',2:'RA_LL_reversal',3:'LA_LL_reversal'}.items():
       ix=cls==code
       if ix.any():changed[ix]=swap(incoming[ix],fault)
     p=forward(model,changed,mean,std);delta=float(np.max(np.abs(p[:n]-saved[lo:lo+n])))
     rows.append({'method':name,'seed':seed,'condition':condition,'block':[lo,hi],'compared_n':n,'max_abs_probability_delta':delta,'pass':delta<=1e-4})
  del model;torch.cuda.empty_cache()
result={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','checks':len(rows),'max_abs_probability_delta':max(r['max_abs_probability_delta'] for r in rows),'rows':rows,
 'scope':'HEEDB first historical block for all four methods, three seeds and four conditions; extra middle/last blocks for Huang including clean false corrections. Explicit NumPy transforms and predicted inverse; no adapter/evaluation workflow imports, no full inference or training. Ningbo GPU and detector neural replay not independently repeated.'}
(OUT/'INDEPENDENT_PREDICTION_REPLAY_20261020.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if result['status']!='PASS':raise SystemExit(1)
