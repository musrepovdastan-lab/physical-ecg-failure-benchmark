import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json';old=sha(p);shutil.copyfile(p,OUT/'FROZEN_EXTENSION_PROTOCOL_BEFORE_AMENDMENT_003.json');x=json.loads(p.read_text(encoding='utf-8'))
 for rel in ('outputs/heedb_probability_extension_v1/extension.py','outputs/heedb_probability_extension_v1/freeze_protocol.py'):
  f=ROOT/rel;x['bound_inputs'][rel]={'sha256':sha(f),'bytes':f.stat().st_size}
 x['analysis_implementation_note']='Historical method fits are loaded from the already frozen heedb_four_arm_extension_v1 package; new p010/p025 fits are loaded only from augmentation_probability_recovery_v2.'
 p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');new=sha(p);(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(new+'\n',encoding='ascii')
 a={'amendment':'003','reason':'After all 12 new prediction bundles completed, analysis preflight found a stale path reference to a historical calibration JSON under the changed model root. Corrected the read path to the immutable six-method extension package. No metrics, calibrators, thresholds, predictions, or cohort selections were generated/changed by the failed analysis preflight. Statistical methods and frozen outcomes unchanged.','prior_protocol_sha256':old,'amended_protocol_sha256':new,'inference_protocol_sha256':old,'inference_completed_before_amendment':True,'status':'ANALYSIS_IMPLEMENTATION_CORRECTION'}
 (OUT/'FROZEN_EXTENSION_AMENDMENT_003.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
