from __future__ import annotations
import json, subprocess, sys, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CONTROL=OUT/'control.py'
METHODS=('ordinary','physics_aware_p050','combined')
SEEDS=(104729,130363,155921)
AMBIENTS=(111,987654321)

def main():
 rows=[]
 for seed in SEEDS:
  for method in METHODS:
   for ambient in AMBIENTS:
    p=subprocess.run([sys.executable,str(CONTROL),'init-probe',method,str(seed),str(ambient)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if p.returncode:
     raise RuntimeError(f'initialization probe failed {method}/{seed}/{ambient}: {p.stderr[-2000:]}')
    rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
 by_seed={s:{r['initial_state_sha256'] for r in rows if r['seed']==s} for s in SEEDS}
 by_method_seed={(m,s):{r['initial_state_sha256'] for r in rows if r['method']==m and r['seed']==s} for m in METHODS for s in SEEDS}
 same_across_methods=all(len({next(r['initial_state_sha256'] for r in rows if r['method']==m and r['seed']==s and r['ambient_state_tag']==a) for m in METHODS})==1 for s in SEEDS for a in AMBIENTS)
 same_across_fresh_processes=all(len(v)==1 for v in by_method_seed.values())
 differs_by_seed=len({next(iter(by_seed[s])) for s in SEEDS})==len(SEEDS)
 independent_of_preseeded_ambient=all(len(by_seed[s])==1 for s in SEEDS)
 report={'status':'PASS' if all((same_across_methods,same_across_fresh_processes,differs_by_seed,independent_of_preseeded_ambient)) else 'FAIL','runner_component':'the same seeded_model() called by train_one()','checks':{'same_seed_same_weights_across_all_methods':same_across_methods,'same_method_seed_same_weights_in_fresh_processes':same_across_fresh_processes,'different_seeds_produce_different_weights':differs_by_seed,'preexisting_rng_state_does_not_change_weights':independent_of_preseeded_ambient},'rows':rows,'n_fresh_processes':len(rows),'note':'Initial state identity is verified separately from complete trajectory reproducibility.'}
 (OUT/'INITIALIZATION_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':report['status'],'checks':report['checks'],'fresh_processes':len(rows)}))
 if report['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
