import json,hashlib,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src import huang_detector_infer_v2 as det
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 canonical=json.loads((ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json').read_text(encoding='utf8'))['records'];split=json.loads((ROOT/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json').read_text(encoding='utf8'))['rows'];by={r['record_id']:r for r in canonical};rows=[by[x['record_id']] for x in split if x['split']=='test']
 model,checkpoint,tfv=det.load_model();probs=np.stack([det.predict(model,rows,cnd,64) for cnd in det.CONDITIONS]).astype(np.float32);ids=np.asarray([r['record_id'] for r in rows])
 dest=OUT/'detector/Ningbo_test.npz';dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.Ningbo_test.tmp.npz');np.savez_compressed(tmp,probabilities=probs,conditions=np.asarray(det.CONDITIONS),class_names=np.asarray(det.CLASSES),record_ids=ids,checkpoint=np.asarray(str(checkpoint)),tensorflow_version=np.asarray(tfv));os.replace(tmp,dest)
 old=np.load(ROOT/'outputs/published_detector_pipeline_v1/cache/detector_test_probs.npz',allow_pickle=False);same_order=np.array_equal(ids,old['record_ids']);clean_delta=float(np.max(np.abs(probs[0]-old['probs_clean'])));ra_ll_delta=float(np.max(np.abs(probs[2]-old['probs_ra_ll'])))
 audit={'status':'PASS' if same_order and clean_delta<=1e-5 and ra_ll_delta<=1e-5 else 'FAIL','rows':len(ids),'record_order_sha256':hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest(),'class_names':list(det.CLASSES),'conditions':list(det.CONDITIONS),'checkpoint':str(checkpoint),'checkpoint_index_sha256':sha(Path(str(checkpoint)+'.index')),'prediction_sha256':sha(dest),'legacy_clean_max_abs_delta':clean_delta,'legacy_RA_LL_max_abs_delta':ra_ll_delta,'legacy_order_match':bool(same_order),'tolerance':1e-5}
 (OUT/'DETECTOR_TEST_BINDINGS.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf8');print(json.dumps(audit))
 if audit['status']!='PASS':raise RuntimeError('Ningbo test detector cache replay mismatch')
if __name__=='__main__':main()
