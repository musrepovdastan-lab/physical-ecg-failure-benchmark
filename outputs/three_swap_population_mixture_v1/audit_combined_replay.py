from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src import published_detector_multifault_v2 as mf
from src.chapman_phase1 import ECGResNetSmall
SEEDS=(104729,130363,155921);CONDS=('clean','RA_LL_reversal','RA_LA_reversal','LA_LL_reversal')
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def main():
 rows=mf.records(); y=np.asarray([r['labels'] for r in rows],dtype=np.uint8)
 blocks=((0,128),(3400,3528),(6784,6912),(6912,6981));all_rows=[]
 raw_blocks=[np.stack([mf.raw_display_100hz(r) for r in rows[a:b]]).astype(np.float32) for a,b in blocks]
 ids=np.asarray([r['record_id'] for r in rows]).astype(str)
 for seed in SEEDS:
  d=ROOT/f'outputs/ningbo_four_arm_replication_v1/combined/seed_{seed}';state=torch.load(d/'checkpoint.pt',map_location='cuda',weights_only=True)
  model=ECGResNetSmall().cuda();model.load_state_dict(state['state_dict']);model.eval()
  pp=np.load(d/'preprocessing.npz',allow_pickle=False);mean=pp['mean'].astype(np.float32);std=pp['std'].astype(np.float32)
  for cond in CONDS:
   z=np.load(d/f'test_predictions/{cond}.npz',allow_pickle=False)
   if z['p'].shape!=(6981,3) or not np.array_equal(z['y'],y):raise RuntimeError(f'combined labels/shape mismatch seed={seed} condition={cond}')
   for (start,stop),raw in zip(blocks,raw_blocks):
    x=np.stack([mf.apply_condition(r,cond) for r in raw]).astype(np.float32)
    x=((x-mean[None,:,None])/std[None,:,None]).astype(np.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):got=torch.sigmoid(model(torch.from_numpy(x).cuda())).float().cpu().numpy()
    ref=z['p'][start:stop].astype(np.float32);delta=float(np.max(np.abs(got-ref)))
    all_rows.append({'method':'combined','seed':seed,'condition':cond,'block':[start,stop],
     'block_record_ids_sha256':sha_bytes('\n'.join(ids[start:stop]).encode()),'max_abs_probability_delta':delta,
     'tolerance':1e-4,'status':'PASS' if delta<=1e-4 else 'FAIL'})
  del model;torch.cuda.empty_cache()
 bad=[r for r in all_rows if r['status']!='PASS']
 audit={'status':'FAIL' if bad else 'PASS','record_count':len(rows),'record_order_sha256':sha_bytes('\n'.join(ids).encode()),
 'sampling':'first/middle/last fixed 128-record blocks; all 3 seeds and four conditions','rows':all_rows,
 'max_abs_probability_delta':max(r['max_abs_probability_delta'] for r in all_rows),'full_cached_label_array_matches_frozen_source':True}
 (OUT/'NINGBO_COMBINED_ORDER_REPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':audit['status'],'rows':len(all_rows),'max_abs_probability_delta':audit['max_abs_probability_delta']}))
 if bad:raise RuntimeError('combined replay mismatch')
if __name__=='__main__':main()

