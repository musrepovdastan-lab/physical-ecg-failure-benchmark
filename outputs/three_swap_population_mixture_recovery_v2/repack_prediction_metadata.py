from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for seed in (104729,130363,155921):
 for cond in ('RA_LA_reversal','LA_LL_reversal'):
  p=OUT/f'predictions/HEEDB/Huang_full_pipeline/seed_{seed}/{cond}.npz'
  with np.load(p,allow_pickle=True) as z:
   data={k:z[k] for k in z.files};before=hashlib.sha256(np.ascontiguousarray(data['p']).tobytes()).hexdigest()
  if data['file_names'].dtype.kind=='O':data['file_names']=data['file_names'].astype(str)
  if data['patient_ids'].dtype.kind=='O':data['patient_ids']=data['patient_ids'].astype(str)
  tmp=p.with_name('.'+p.stem+'.repack.npz');np.savez_compressed(tmp,**data);tmp.replace(p)
  with np.load(p,allow_pickle=False) as z:
   after=hashlib.sha256(np.ascontiguousarray(z['p']).tobytes()).hexdigest()
   if before!=after or z['file_names'].dtype.kind not in 'US' or z['patient_ids'].dtype.kind not in 'US':raise RuntimeError('score or identity array changed during metadata repack')
  rows.append({'path':p.relative_to(ROOT).as_posix(),'previous_artifact_sha256':None,'new_artifact_sha256':sha(p),'probability_array_sha256_before_after':before,'probability_array_unchanged':True,'file_names_dtype':'unicode','patient_ids_dtype':'unicode'})
completion=json.loads((OUT/'RECOVERY_INFERENCE_COMPLETE.json').read_text(encoding='utf-8'))
for r in completion['new_files']:
 r['sha256']=sha(ROOT/r['path']);r['probability_array_sha256']=next(x['probability_array_sha256_before_after'] for x in rows if x['path']==r['path'])
(OUT/'RECOVERY_INFERENCE_COMPLETE.json').write_text(json.dumps(completion,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(OUT/'METADATA_REPACK_AUDIT.json').write_text(json.dumps({'status':'PASS','prediction_values_changed':False,'rows':rows},indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','files':len(rows),'probability_arrays_unchanged':True}))
