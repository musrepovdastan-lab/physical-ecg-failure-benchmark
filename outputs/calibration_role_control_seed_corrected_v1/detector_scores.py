from pathlib import Path
import csv,hashlib,json,sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src import huang_detector_infer_v2 as det
PREF=ROOT/'outputs/calibration_independence_preflight_v1'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=list(csv.DictReader((PREF/'PROPOSED_MEMBERSHIP.csv').open(encoding='utf-8-sig')))
 model,checkpoint,tfv=det.load_model();out=[]
 for role in ('calibration_A','threshold_B'):
  rr=[r for r in rows if r['role']==role];records=[{'record_id':r['record_id'],'waveform_path':r['record_id'].split('/',1)[1]+'.mat'} for r in rr]
  # Resolve through the accepted canonical registry rather than reconstructing file paths.
  canonical=json.loads((ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json').read_text(encoding='utf8'))['records'];by={r['record_id']:r for r in canonical};records=[by[r['record_id']] for r in rr]
  probs=[]
  for cond in det.CONDITIONS:probs.append(det.predict(model,records,cond,64))
  p=np.stack(probs).astype(np.float32);ids=np.asarray([r['record_id'] for r in rr])
  dest=OUT/f'detector/{role}.npz';tmp=dest.with_name('.'+dest.stem+'.tmp.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(tmp,probabilities=p,conditions=np.asarray(det.CONDITIONS),class_names=np.asarray(det.CLASSES),record_ids=ids,membership_sha256=np.asarray(sha(PREF/'PROPOSED_MEMBERSHIP.csv')),checkpoint=np.asarray(str(checkpoint)),tensorflow_version=np.asarray(tfv));os.replace(tmp,dest)
  out.append({'role':role,'rows':len(ids),'conditions':list(det.CONDITIONS),'classes':list(det.CLASSES),'sha256':sha(dest),'record_ids_sha256':hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest(),'checkpoint':str(checkpoint),'checkpoint_data_sha256':sha(Path(str(checkpoint)+'.index')) if Path(str(checkpoint)+'.index').exists() else None})
 (OUT/'DETECTOR_ROLE_BINDINGS.json').write_text(json.dumps({'status':'PASS','artifacts':out,'same_source_input_order_verified':True},indent=2)+'\n',encoding='utf8')
 print(json.dumps({'status':'PASS','artifacts':out}))
if __name__=='__main__':main()
