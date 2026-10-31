import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json';old=sha(p);shutil.copyfile(p,OUT/'FROZEN_EXTENSION_PROTOCOL_BEFORE_AMENDMENT_005.json');x=json.loads(p.read_text(encoding='utf-8'))
 for rel in ('outputs/heedb_probability_extension_v1/extension.py','outputs/heedb_probability_extension_v1/freeze_protocol.py'):
  f=ROOT/rel;x['bound_inputs'][rel]={'sha256':sha(f),'bytes':f.stat().st_size}
 x['calibration']['audit_implementation_note']='Decision equivalence audit applies the fitted Platt coefficients in both score paths; calibration metrics already used those coefficients. Historical reports remain untouched.'
 p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');new=sha(p);(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(new+'\n',encoding='ascii')
 a={'amendment':'005','reason':'Post-analysis audit found the audit-only block used raw-mode placeholder coefficients when calculating the calibrated decision cross-check. Metric rows already used the correct fitted coefficients; corrected the audit cross-check to use the stored coefficients, then recompute the audit separately. No model probabilities, calibration fit, threshold, metric, bootstrap draw or contrast changes.','prior_protocol_sha256':old,'amended_protocol_sha256':new,'inference_protocol_sha256':'6cc3534e6c7e3f01f535578f995308b29f2425cae0ae2aa8ead093ccbfa6df87','status':'AUDIT_IMPLEMENTATION_CORRECTION'}
 (OUT/'FROZEN_EXTENSION_AMENDMENT_005.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
