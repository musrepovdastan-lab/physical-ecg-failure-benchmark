from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import control as c
SEED=49157
CHECKPOINT=OUT/'RNG_RESUME_AUDIT_TMP.pt'

def batch_data():
 g=torch.Generator(device='cpu');g.manual_seed(8191)
 x=torch.randn((6,12,1000),generator=g,dtype=torch.float32)
 y=torch.tensor([[1,0,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],[0,1,1]],dtype=torch.float32)
 return x,y
def epoch(model,opt,scaler,loader):
 model.train()
 for x,y in loader:
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x.cuda()),y.cuda())
  scaler.scale(loss).backward();scaler.step(opt);scaler.update()
def child(mode):
 c.set_deterministic_seed(SEED);model,_,_,_,_=c.seeded_model(SEED);model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda')
 x,y=batch_data();g=torch.Generator(device='cpu');g.manual_seed(SEED);loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=True,num_workers=0,generator=g)
 if mode=='full':
  epoch(model,opt,scaler,loader);epoch(model,opt,scaler,loader)
 elif mode=='split':
  epoch(model,opt,scaler,loader)
  c.atomic_torch_save(CHECKPOINT,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng_state':c.rng_state(),'loader_generator':g.get_state()})
 elif mode=='resume':
  q=torch.load(CHECKPOINT,weights_only=False,map_location='cpu');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);scaler.load_state_dict(q['scaler']);g.set_state(q['loader_generator']);c.restore_rng_state(q['rng_state']);epoch(model,opt,scaler,loader)
 else:raise ValueError(mode)
 final_rng=c.rng_state();rng_parts={'python':hashlib.sha256(__import__('pickle').dumps(final_rng['python'],protocol=4)).hexdigest(),'numpy':hashlib.sha256(__import__('pickle').dumps(final_rng['numpy'],protocol=4)).hexdigest(),'torch_cpu':hashlib.sha256(final_rng['torch_cpu'].numpy().tobytes()).hexdigest(),'torch_cuda':[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in final_rng['torch_cuda']]}
 print(json.dumps({'state_sha256':c.state_digest(model.state_dict()),'rng_sha256':c.rng_digest(final_rng),'rng_parts':rng_parts,'cuda_device':torch.cuda.get_device_name(0)}))
def main():
 if len(sys.argv)>1 and sys.argv[1]=='child':child(sys.argv[2]);return
 rows={}
 for mode in ('full','split','resume'):
  p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'child',mode],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
  if p.returncode:raise RuntimeError(f'{mode} failed: {p.stderr[-2000:]}')
  rows[mode]=json.loads(p.stdout.strip().splitlines()[-1])
 CHECKPOINT.unlink(missing_ok=True)
 passed=rows['full']['state_sha256']==rows['resume']['state_sha256'] and rows['full']['rng_sha256']==rows['resume']['rng_sha256']
 parts_equal=rows['full']['rng_parts']==rows['resume']['rng_parts']
 report={'status':'PASS' if passed else 'FAIL','test':'two-epoch fixed miniature trajectory, continuous versus checkpoint/resume in fresh process','full_process':rows['full'],'interrupted_then_resumed':rows['resume'],'interrupted_first_process':rows['split'],'same_final_model_state':rows['full']['state_sha256']==rows['resume']['state_sha256'],'same_final_rng_state':rows['full']['rng_sha256']==rows['resume']['rng_sha256'],'rng_componentwise_equal':parts_equal,'scope_note':'Engineering check only; not an additional fit, scientific result, or proof that all nine full training trajectories were independently replayed.'}
 (OUT/'RNG_RESUME_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':report['status'],'same_model_state':report['same_final_model_state'],'same_rng_state':report['same_final_rng_state']}))
 if not passed:raise SystemExit(2)
if __name__=='__main__':main()
