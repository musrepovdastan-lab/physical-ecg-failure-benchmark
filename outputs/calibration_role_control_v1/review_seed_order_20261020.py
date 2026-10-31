"""Audit RNG ordering without fitting a model or reading ECGs."""
from pathlib import Path
import ast,hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.chapman_phase1 import ECGResNetSmall
code=(OUT/'control.py').read_text(encoding='utf-8');tree=ast.parse(code)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one')
constructor=[];seed_calls=[]
for n in ast.walk(fn):
 if isinstance(n,ast.Call):
  name=ast.unparse(n.func)
  if name=='ECGResNetSmall':constructor.append(n.lineno)
  if name in ['random.seed','np.random.seed','torch.manual_seed','torch.cuda.manual_seed_all']:seed_calls.append({'call':name,'line':n.lineno})
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def observed_order(ambient,declared):
 torch.manual_seed(ambient);model=ECGResNetSmall();torch.manual_seed(declared);return digest(model)
def seeded_first(ambient,declared):
 torch.manual_seed(ambient);torch.manual_seed(declared);return digest(ECGResNetSmall())
a=observed_order(111,104729);b=observed_order(222,104729)
c=seeded_first(111,104729);d=seeded_first(222,104729)
result={'status':'INITIALIZATION_NOT_CONTROLLED_BY_DECLARED_FIT_SEED','model_constructor_lines':constructor,'seed_calls':seed_calls,
 'first_constructor_before_first_seed_call':min(constructor)<min(x['line'] for x in seed_calls),
 'same_declared_seed_different_ambient_rng_changes_initial_weights':a!=b,'seed_before_constructor_control_matches':c==d,
 'weight_digest_ambient111':a,'weight_digest_ambient222':b,
 'interpretation':'Declared seed controls subsequent shuffle/training RNG and deterministic augmentations, but is assigned after model initialization. Existing checkpoints remain evaluable; exact fresh initialization cannot be reproduced solely from their fit seed. Initial-state provenance needs a narrow audit; this is not evidence of A/B leakage or invalid prediction arithmetic.',
 'scope':'Static source audit plus four CPU model constructions; zero training, zero data inference, no historical artifact edits.'}
(OUT/'SEED_ORDER_REVIEW_20261020.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
