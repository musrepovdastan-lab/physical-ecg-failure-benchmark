"""Independent audit of completion; reuses unchanged previously reviewed predictions."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'calibration_role_control_v1'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
N={};MAX={};FAIL=[]
def ck(k,a,b=True,tol=1e-11):
 N[k]=N.get(k,0)+1
 if isinstance(b,(str,bool)):
  if a!=b:FAIL.append([k,str(a),str(b)])
 else:
  d=abs(float(a)-float(b));MAX[k]=max(MAX.get(k,0),d)
  if not np.isfinite(d) or d>tol:FAIL.append([k,str(a),str(b)])
mf=j(OUT/'MANIFEST.json')
for role,items in [('output',mf['new_outputs']),('source',mf['scientific_inputs']),('v1_unchanged',j(OLD/'REPRODUCIBILITY_MANIFEST.json')['files'])]:
 for r in items:
  p=ROOT/r['path']
  if role=='output' and sha(p)!=r['sha256']:
   archived=OUT/'archive_before_review_20261020'/p.name
   ck('original_output_archived_hash',sha(archived),r['sha256'])
   change=next(x for x in j(OUT/'DOCUMENT_REVIEW_CHANGES_20261020.json')['changes'] if x['path']==r['path'])
   ck('reviewed_document_hash',sha(p),change['reviewed_sha256'])
  else:ck(role+'_hash',sha(p),r['sha256']);ck(role+'_size',p.stat().st_size,r['bytes'],0)
ck('protocol_hash',sha(OUT/'COMPLETION_PROTOCOL.json'),(OUT/'COMPLETION_PROTOCOL_SHA256.txt').read_text().split()[0])
am=j(OUT/'COMPLETION_PROTOCOL_AMENDMENT_001.json');ck('amended_runner',sha(OUT/'complete_bootstrap.py'),am['corrected_runner_sha256'])
print('HASHES_COMPLETE',flush=True)
C=['clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'];M=['ordinary','physics_aware_p050','combined','Huang_full_pipeline'];S=[104729,130363,155921]
fits={(r['method'],int(r['seed'])):r for r in pd.read_csv(OLD/'CALIBRATION_AND_THRESHOLD_LOCK.csv',float_precision='round_trip').to_dict('records') if r['endpoint']=='TWA'}
data={};labels={}
for ds,m,s,c in itertools.product(['Ningbo_test','HEEDB'],M,S,C):
 with np.load(OLD/f'predictions/{ds}/{m}/seed_{s}/{c}.npz',allow_pickle=False) as z:
  data[ds,m,s,c]=z['p'][:,0].astype(np.float64)
  if ds in labels:ck('transitive_labels_match',np.array_equal(labels[ds],z['y'][:,0]))
  else:labels[ds]=z['y'][:,0].copy()
def coeff(sc,q):return np.asarray([1-q,0,q,0]) if sc=='RA_LL_only' else np.asarray([1-q,q/3,q/3,q/3])
def cal(p,f):
 v=np.clip(p,1e-6,1-1e-6);z=f['slope_a']*np.log(v/(1-v))+f['intercept_b'];return 1/(1+np.exp(-np.clip(z,-745,745)))
rawscores={(ds,m,s):np.concatenate([data[ds,m,s,c] for c in C]) for ds,m,s in itertools.product(labels,M,S)}
calprobs={k:cal(v,fits[k[1],k[2]]) for k,v in rawscores.items()}
boots={'Ningbo_test':np.load(ROOT/'outputs/ningbo_confirmation_v2/bootstrap_indices.npy',mmap_mode='r'),'HEEDB':np.load(ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy',mmap_mode='r')}
cache={}
def value(ds,sc,q,m,s,metric,bi=None):
 key=ds,sc,q,m,s,metric,bi
 if key in cache:return cache[key]
 n=len(labels[ds]);w=np.repeat(coeff(sc,q),n)
 if bi is not None:w=w*np.tile(np.bincount(boots[ds][bi],minlength=n),4)
 yy=np.tile(labels[ds],4);p=rawscores[ds,m,s];f=fits[m,s];t=f['raw_q90'];keep=w>0
 if metric=='AP_raw':v=average_precision_score(yy[keep],p[keep],sample_weight=w[keep])
 elif metric=='Brier_raw':v=np.average((p-yy)**2,weights=w)
 elif metric=='Brier_calibrated':v=np.average((calprobs[ds,m,s]-yy)**2,weights=w)
 elif metric=='sensitivity_raw':v=w[(yy==1)&(p>=t)].sum()/w[yy==1].sum()
 elif metric=='FP_per_1000_raw':v=1000*w[(yy==0)&(p>=t)].sum()/w.sum()
 else:raise ValueError(metric)
 cache[key]=v;return v
rows=pd.read_csv(OUT/'PAIRED_POINTS_CI_ALL_700.csv',float_precision='round_trip');coverage=pd.read_csv(OUT/'BOOTSTRAP_COVERAGE_700.csv')
with np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz',allow_pickle=False) as z:draws={k:z[k].copy() for k in z.files}
contrasts=[('physics_aware_p050','ordinary'),('combined','ordinary'),('Huang_full_pipeline','ordinary'),('physics_aware_p050','Huang_full_pipeline'),('combined','Huang_full_pipeline')]
metrics=['AP_raw','Brier_raw','Brier_calibrated','sensitivity_raw','FP_per_1000_raw'];qs=[0,.001,.005,.01,.05,.1,1]
expected={f'{ds}|{sc}|q{q:g}|{l}-{r}|{metric}' for ds,sc,q,(l,r),metric in itertools.product(labels,['RA_LL_only','equal_three_swap'],qs,contrasts,metrics)}
ck('exact_700_keys',set(draws)==expected);ck('table_rows',len(rows),700,0);ck('unique_coverage',coverage.group_key.nunique(),700,0)
for r in rows.to_dict('records'):
 ds,sc,q,l,rr,m=r['cohort'],r['scenario'],r['q_fault'],r['left'],r['right'],r['metric'];key=f'{ds}|{sc}|q{q:g}|{l}-{rr}|{m}';d=draws[key]
 ck('draw_shape',d.shape==(1000,));ck('all_draws_finite',bool(np.isfinite(d).all()))
 full=np.mean([value(ds,sc,q,l,s,m)-value(ds,sc,q,rr,s,m) for s in S])
 ck('full_point',r['full_cohort_point_mean_seed_contrast'],full)
 for f,v in [('bootstrap_mean',d.mean()),('ci95_low',np.quantile(d,.025)),('ci95_high',np.quantile(d,.975))]:ck('draw_summary',r[f],v)
 for bi in [0,417,999]:ck('actual_draw_spot',d[bi],np.mean([value(ds,sc,q,l,s,m,bi)-value(ds,sc,q,rr,s,m,bi) for s in S]),1e-10)
old_count=0
for ds in labels:
 with np.load(OLD/f'{ds}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:
  for key in z.files:ck('old_draw_exact',np.array_equal(draws[key],z[key]));old_count+=1
ck('old_count',old_count,300,0)
for r in coverage.to_dict('records'):
 ck('coverage_provenance',r['origin'],'reused_v1' if r['q_fault'] in [.01,.1,1.] else 'new_v2')
for ds,(l,r),metric in itertools.product(labels,contrasts,metrics):
 ck('q0_scenarios_equal',np.array_equal(draws[f'{ds}|RA_LL_only|q0|{l}-{r}|{metric}'],draws[f'{ds}|equal_three_swap|q0|{l}-{r}|{metric}']))
for name in ['FINAL_REPORT_RU.md','TRAINING_REPORT_RU.md','CURRENT_PLAN_STATUS_RU.md','INITIALIZATION_PROVENANCE_NOTE_RU.md','LOCK_CHRONOLOGY_RU.md']:
 text=(OUT/name).read_text(encoding='utf-8');ck('utf8_russian',sum('\u0400'<=c<='\u04ff' for c in text)>50 and '????' not in text)
result={'status':'PASS' if not FAIL else 'FAIL','counts':N,'max_abs_errors':MAX,'failure_count':len(FAIL),'failures':FAIL[:30],
 'scope':'Independent completion hashes, all700 points and summaries,2100 actual draw spots,old300 exact preservation,coverage/provenance and report encoding. No training, inference or calibration refit. Previously reviewed weights/predictions preserved by hashes; previous initialization/patient/proxy/chronology limitations remain.'}
(OUT/'INDEPENDENT_REVIEW_AUDIT_20261020.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if FAIL:raise SystemExit(1)
