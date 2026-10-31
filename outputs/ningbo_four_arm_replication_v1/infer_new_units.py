"""Inference after validation gates, with the frozen 128-batch preprocessing contract."""
from __future__ import annotations
import hashlib,json,os,sys,time,uuid
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from outputs.ningbo_four_arm_replication_v1 import workflow as w
from src.chapman_phase1 import ECGResNetSmall

def atomic_json(path,value):w.atomic_json(path,value)
def atomic_npz(path,**arrays):w.atomic_npz(path,**arrays)

def load_test_cache(parts):
    path=w.OUT/'test_waveforms_float32.npy';meta=w.OUT/'TEST_INPUT_CACHE.json'
    order=w.hash_obj([r['record_id'] for r in parts['test']]);label_hash=w.hash_obj([r['labels'] for r in parts['test']])
    if path.exists() and meta.exists():
        m=json.loads(meta.read_text(encoding='utf-8'))
        if m.get('test_order_sha256')!=order or m.get('test_label_order_sha256')!=label_hash or m.get('shape')!=[6981,12,1000] or m.get('cache_sha256')!=w.sha(path):raise RuntimeError('test waveform cache binding mismatch')
        x=np.load(path,mmap_mode='r',allow_pickle=False)
    else:
        tmp=path.with_name('.test_waveforms.'+uuid.uuid4().hex+'.npy')
        mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(parts['test']),12,1000))
        for i,r in enumerate(parts['test']):
            z=w.data.raw(r).astype(np.float32)
            if z.shape!=(12,1000) or not np.isfinite(z).all():raise RuntimeError(f'invalid test waveform row {i}')
            mm[i]=z
        mm.flush();del mm;os.replace(tmp,path)
        x=np.load(path,mmap_mode='r',allow_pickle=False)
        atomic_json(meta,{'test_order_sha256':order,'test_label_order_sha256':label_hash,'shape':list(x.shape),'dtype':str(x.dtype),'cache_sha256':w.sha(path),'source':'frozen canonical Ningbo test records; 100Hz downsample path src.ningbo_confirmation_v1.raw'})
    y=np.asarray([r['labels'] for r in parts['test']],np.uint8)
    return x,y,json.loads(meta.read_text(encoding='utf-8'))

def infer_array(model,x,mean,std):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            a=w.recovery.normalize_raw(np.asarray(x[start:start+128]),mean,std)
            with torch.autocast('cuda',dtype=torch.float16):
                out.append(torch.sigmoid(model(torch.from_numpy(a).cuda(non_blocking=True))).float().cpu().numpy())
    return np.concatenate(out)

def conditions(x,mean):
    for c in ('clean',*w.FAULTS):
        if c=='clean':yield c,x
        else:
            # Compute physical leads with the exact project-derived display implementation.
            yield c,np.stack([w.reverse_raw(np.asarray(x[i]),c) for i in range(len(x))])
    for ix,c in zip(w.MASKS,w.MASK_NAMES):
        yield c,w.recovery.apply_mask_raw(x,mean,ix)

def run_unit(arm,seed,x,y,test_order_sha):
    d=w.unit_dir(arm,seed);g=json.loads((d/'validation_gate.json').read_text(encoding='utf-8'))
    if g.get('decision')!='PASS':return {'arm':arm,'seed':seed,'status':'SKIPPED_VALIDATION_GATE'}
    protocol=w.protocol_check();ckpt=d/'checkpoint.pt';cksha=w.sha(ckpt)
    state=torch.load(ckpt,weights_only=True,map_location='cuda')
    if state.get('arm')!=arm or state.get('seed')!=seed or state.get('protocol_sha256')!=protocol['protocol_sha256'] or cksha!=g['checkpoint_sha256']:
        raise RuntimeError(f'checkpoint provenance mismatch: {arm}/{seed}')
    with np.load(d/'preprocessing.npz',allow_pickle=False) as z:mean,std=z['mean'],z['std']
    model=ECGResNetSmall().cuda();model.load_state_dict(state['state_dict']);model.eval()
    freeze=d/'evaluation_freeze.json';done=d/'evaluation_complete.json';pd=d/'test_predictions';pd.mkdir(exist_ok=True)
    atomic_json(freeze,{'arm':arm,'seed':seed,'protocol_sha256':protocol['protocol_sha256'],'checkpoint_sha256':cksha,'test_order_sha256':test_order_sha,'conditions':len(w.CONDITIONS),'test_accessed_before_freeze':False,'inference_source_sha256':w.sha(Path(__file__)),'batch_size':128,'model_eval':True,'autocast':'CUDA float16'})
    saved={}
    for ordinal,(cond,raw) in enumerate(conditions(x,mean),1):
        f=pd/f'{cond}.npz'
        if f.exists():
            with np.load(f,allow_pickle=False) as z:
                valid=(z['p'].shape==(len(y),3) and np.isfinite(z['p']).all() and str(z['protocol_sha256'].item())==protocol['protocol_sha256'] and str(z['checkpoint_sha256'].item())==cksha and str(z['condition'].item())==cond and np.array_equal(z['y'],y))
            if not valid:raise RuntimeError(f'unbound/invalid partial prediction cannot be resumed: {f}')
            saved[cond]=w.sha(f);continue
        p=infer_array(model,raw,mean,std)
        if p.shape!=(len(y),3) or not np.isfinite(p).all():raise RuntimeError(f'nonfinite/shape inference failure: {cond}')
        atomic_npz(f,p=p,y=y,protocol_sha256=np.asarray(protocol['protocol_sha256']),checkpoint_sha256=np.asarray(cksha),condition=np.asarray(cond),test_order_sha256=np.asarray(test_order_sha))
        saved[cond]=w.sha(f);w.status('INFERENCE',arm=arm,seed=seed,condition=cond,completed_conditions=ordinal,total_conditions=len(w.CONDITIONS))
    manifest={'arm':arm,'seed':seed,'protocol_sha256':protocol['protocol_sha256'],'checkpoint_sha256':cksha,'test_order_sha256':json.loads(freeze.read_text(encoding='utf-8'))['test_order_sha256'],'conditions':len(w.CONDITIONS),'predictions':saved,'inference_source_sha256':w.sha(Path(__file__)),'test_accessed_before_freeze':False}
    atomic_json(done,manifest);return {'arm':arm,'seed':seed,'status':'COMPLETE','prediction_manifest_sha256':w.sha(done)}

def main():
    w.protocol_check();parts=w.records_by_split();
    test_order_sha=w.hash_obj([r['record_id'] for r in parts['test']]);label_order_sha=w.hash_obj([r['labels'] for r in parts['test']])
    gates={};eligible=False
    for arm in w.ARMS:
      for seed in w.SEEDS:
        f=w.unit_dir(arm,seed)/'validation_gate.json';g=json.loads(f.read_text(encoding='utf-8'));gates[f'{arm}/{seed}']={'gate_sha256':w.sha(f),'decision':g.get('decision'),'checkpoint_sha256':g.get('checkpoint_sha256')}
        if g.get('decision')=='PASS':
            ck=torch.load(w.unit_dir(arm,seed)/'checkpoint.pt',weights_only=True,map_location='cpu')
            if ck.get('arm')!=arm or ck.get('seed')!=seed or ck.get('protocol_sha256')!=json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256'] or w.sha(w.unit_dir(arm,seed)/'checkpoint.pt')!=g.get('checkpoint_sha256'):raise RuntimeError(f'validation/checkpoint gate binding mismatch: {arm}/{seed}')
            eligible=True
    freeze={'protocol_sha256':json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256'],'inference_source_sha256':w.sha(Path(__file__)),
      'checkpoint_gate_bindings':gates,'test_order_sha256':test_order_sha,'test_label_order_sha256':label_order_sha,
      'conditions':len(w.CONDITIONS),'batch_size':128,'mode':'model.eval + no_grad/inference_mode + CUDA float16 autocast','preprocessing':'corrected recovery v2 masks and src.ecg_physics reversals; train mean replacement before normalization',
      'test_data_read_before_this_freeze':False}
    atomic_json(w.OUT/'FROZEN_INFERENCE_CONTRACT.json',freeze)
    if not eligible:raise RuntimeError('no new units passed the validation gate; test inference stopped')
    x,y,cache=load_test_cache(parts)
    atomic_json(w.OUT/'TEST_INPUT_CACHE.json',cache)
    rows=[]
    for arm in w.ARMS:
      for seed in w.SEEDS:rows.append(run_unit(arm,seed,x,y,test_order_sha))
    atomic_json(w.OUT/'NEW_UNIT_INFERENCE_STATUS.json',{'units':rows,'completed_utc':w.now()})
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
