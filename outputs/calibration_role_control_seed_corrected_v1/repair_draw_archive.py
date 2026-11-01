from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
combined={}
for cohort in ('Ningbo_test','HEEDB'):
    with np.load(ROOT/f'{cohort}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:
        for key in z.files:
            if not key.startswith(cohort+'|'):
                raise RuntimeError(f'noncanonical per-cohort key: {key}')
            combined[key]=z[key].copy()
if len(combined)!=700 or len(set(combined))!=700:
    raise RuntimeError(f'expected 700 unique draw keys, got {len(combined)}')
tmp=ROOT/'.PAIRED_BOOTSTRAP_DRAWS_ALL.tmp.npz'
np.savez_compressed(tmp,**combined)
tmp.replace(ROOT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz')
print({'status':'PASS','groups':len(combined),'draws_per_group':1000})
