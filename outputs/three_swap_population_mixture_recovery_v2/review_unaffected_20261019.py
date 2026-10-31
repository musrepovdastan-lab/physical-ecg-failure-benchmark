from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'three_swap_population_mixture_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for r in pd.read_csv(OLD/'SHA256_MANIFEST.csv').itertuples():
 checks.append({'kind':'v1_file_unchanged','path':r.path,'pass':sha(OLD/r.path)==r.sha256})
with np.load(OLD/'PAIRED_BOOTSTRAP_DRAWS.npz') as a,np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS.npz') as b:
 original=dict(zip(a['draw_names'].astype(str),a['draws']));current=dict(zip(b['draw_names'].astype(str),b['draws']))
 max_delta=0;count=0
 for key,v in current.items():
  ds,sc,q,contrast,metric=key.split('|')
  affected=ds=='HEEDB' and sc=='equal_three_swap' and float(q.split('=')[1])>0 and 'Huang_full_pipeline' in contrast
  if not affected:
   d=float(np.max(np.abs(v-original[key])));max_delta=max(max_delta,d);count+=1
 checks.append({'kind':'610_unaffected_bootstrap_groups','count':count,'max_abs_difference':max_delta,'pass':count==610 and max_delta<=1e-12})
result={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks}
(OUT/'INDEPENDENT_UNAFFECTED_AUDIT_20261019.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'v1_files':len(checks)-1,'unaffected_draw_groups':count,'max_draw_difference':max_delta}))
if result['status']!='PASS':raise SystemExit(1)
