from __future__ import annotations
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
import evaluate_full as ev

OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    cohort_draws={c:load(OUT/f'{c}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz') for c in ('Ningbo_test','HEEDB')}
    active=load(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz')
    expected=ev.expected_draw_keys()
    if len(active)!=700 or set(active)!=expected:raise RuntimeError('existing active archive is not the canonical 700-key archive')
    rebuilt=ev.combine_cohort_draws(cohort_draws)
    if len(rebuilt)!=700 or set(rebuilt)!=expected:raise RuntimeError('repack key set mismatch')
    mismatches=[]
    for key in sorted(expected):
        a,b=rebuilt[key],active[key]
        if a.shape!=(1000,) or b.shape!=(1000,) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a,b):
            mismatches.append(key)
    if mismatches:raise RuntimeError(f'repacked values differ: {mismatches[:3]}')
    # Exercise the exact write path used by evaluate_full without replacing the active archive.
    with tempfile.TemporaryDirectory(prefix='seed_corrected_repack_',dir=OUT) as td:
        target=Path(td)/'repacked.npz'
        written=ev.write_full_draw_archive(target,cohort_draws)
        roundtrip=load(target)
        if set(roundtrip)!=expected or any(not np.array_equal(roundtrip[k],active[k]) for k in expected):
            raise RuntimeError('write/read repack does not exactly match active archive')
    report={'status':'PASS','expected_keys':700,'rebuilt_keys':len(rebuilt),'duplicate_keys':0,
      'draw_shape':[1000],'all_draws_finite':True,'exact_value_mismatches':0,
      'active_archive_sha256':sha(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz'),
      'cohort_archives':{c:{'path':f'{c}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz','keys':len(v),'sha256':sha(OUT/f'{c}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz')} for c,v in cohort_draws.items()},
      'tested_combiner_sha256':sha(OUT/'evaluate_full.py'),'full_statistics_recomputed':False}
    (OUT/'BOOTSTRAP_ARCHIVE_REPACK_VERIFICATION.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
