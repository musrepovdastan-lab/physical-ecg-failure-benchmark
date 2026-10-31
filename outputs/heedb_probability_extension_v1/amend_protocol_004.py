import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json';old=sha(p);shutil.copyfile(p,OUT/'FROZEN_EXTENSION_PROTOCOL_BEFORE_AMENDMENT_004.json');x=json.loads(p.read_text(encoding='utf-8'))
 for rel in ('outputs/heedb_probability_extension_v1/extension.py','outputs/heedb_probability_extension_v1/freeze_protocol.py'):
  f=ROOT/rel;x['bound_inputs'][rel]={'sha256':sha(f),'bytes':f.stat().st_size}
 x['analysis_implementation_note']='The six historical methods are reused from the immutable six-method HEEDB package with its published fits_by_method_and_seed structure.'
 p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');new=sha(p);(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(new+'\n',encoding='ascii')
 a={'amendment':'004','reason':'Analysis preflight after new inference clarified historical reuse must source masking and combined prediction bundles and calibration fits from the immutable six-method extension. The failed preflight produced no metrics and did not alter the 12 new prediction bundles. This correction adds the six-method reuse mapping only; methods and comparisons unchanged.','prior_protocol_sha256':old,'amended_protocol_sha256':new,'inference_protocol_sha256':'6cc3534e6c7e3f01f535578f995308b29f2425cae0ae2aa8ead093ccbfa6df87','inference_completed_before_amendment':True,'status':'ANALYSIS_IMPLEMENTATION_CORRECTION'}
 (OUT/'FROZEN_EXTENSION_AMENDMENT_004.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
