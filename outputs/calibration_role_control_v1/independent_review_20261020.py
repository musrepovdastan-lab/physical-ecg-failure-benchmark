"""Independent saved-file audit; no training/inference or scientific recipe selection."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
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
mf=j(OUT/'REPRODUCIBILITY_MANIFEST.json')
for r in mf['files']:
 p=ROOT/r['path'];ck('output_hash',sha(p),r['sha256']);ck('output_bytes',p.stat().st_size,r['bytes'],0)
prot=j(OUT/'FROZEN_EXECUTION_PROTOCOL.json');ph=prot.pop('canonical_fingerprint_sha256')
ck('canonical_protocol',hashlib.sha256(json.dumps(prot,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),ph)
ck('protocol_file_hash',sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json'),(OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').read_text().strip())
for rel,h in prot['input_file_sha256'].items():
 p=ROOT/rel
 if rel=='NEXT_RESEARCH_DECISIONS_20261019.md' and sha(p)!=h:p=OUT/'ACTIVE_PLAN_CONTEXT_SNAPSHOT.md'
 ck('input_hash',sha(p),h)
for name,h in prot['runner_sha256'].items():ck('runner_hash',sha(OUT/name),h)
print('HASHES_COMPLETE',flush=True)
S=[104729,130363,155921];CORE=['ordinary','physics_aware_p050','combined'];M=CORE+['Huang_full_pipeline'];E=['TWA','SB','STach'];C=['clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal']
can=j(ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json')['records'];by={r['record_id']:r for r in can};split=j(OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')['rows']
nr=[by[r['record_id']] for r in split if r['split']=='test'];ids={'Ningbo_test':np.asarray([r['record_id'] for r in nr])};labels={'Ningbo_test':np.asarray([r['labels'] for r in nr],np.uint8)}
selection=pd.read_csv(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv',dtype=str,keep_default_na=False)
diagnoses=pd.read_csv(ROOT/'data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv',usecols=['FileName','codes_physician'],dtype=str,keep_default_na=False)
df=selection[['FileName','BDSPPatientID']].merge(diagnoses,on='FileName',how='left',validate='one_to_one',sort=False)
ck('heedb_unique_patients',df.BDSPPatientID.nunique(),10000,0);ids['HEEDB']=df.FileName.to_numpy()
codesets=[{1140,1145,1150,1151,1160,1161,1170,1171,1172,1180,1181},{21,24},{23}]
labels['HEEDB']=np.array([[int(bool({int(x) for x in re.findall(r'\d+',v)}&codes)) for codes in codesets] for v in df.codes_physician],np.uint8)
mem=pd.read_csv(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv',dtype=str,keep_default_na=False)
roles={role:mem[mem.role==role].record_id.to_numpy() for role in ['model_selection','calibration_A','threshold_B']}
for a in roles:
 for b in roles:
  if a<b:ck('role_disjoint',not bool(set(roles[a])&set(roles[b])))
ck('role_partition',set(np.concatenate(list(roles.values())))=={r['record_id'] for r in split if r['split']=='validation'})
locks=j(OUT/'ALL_MODELS_LOCK.json');ck('fit_count',len(locks['fits']),9,0)
for r in locks['fits']:
 m,s=r['method'],int(r['seed']);d=OUT/f'fits/{m}/seed_{s}';hist=j(d/'history.json')['rows'];best=max(hist,key=lambda x:x['selection_twa_auprc'])
 ck('fit_hash',sha(d/'checkpoint.pt'),r['checkpoint_sha256']);ck('fit_history_hash',sha(d/'history.json'),r['history_sha256'])
 ck('history_best_epoch',best['epoch'],r['best_epoch'],0)
 ck('history_selection_only',all(h['selection_role']=='model_selection' for h in hist))
 with np.load(OUT/f'scores/model_selection/{m}/seed_{s}.npz',allow_pickle=False) as z:
  ck('selection_ids',np.array_equal(z['record_ids'].astype(str),roles['model_selection']))
  yy=np.asarray([by[x]['labels'] for x in roles['model_selection']],np.uint8);ck('selection_labels',np.array_equal(z['y'],yy))
  ck('selection_checkpoint',str(z['checkpoint_sha256'].item()),r['checkpoint_sha256'])
  ck('selection_AP',average_precision_score(yy[:,0],z['p'][:,0]),best['selection_twa_auprc'])
  ck('selection_AUC',roc_auc_score(yy[:,0],z['p'][:,0]),best['selection_twa_auroc'])
calrows=pd.read_csv(OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv',float_precision='round_trip');fits={}
def platt(p,a,b):
 p=np.clip(p,1e-6,1-1e-6);return 1/(1+np.exp(-np.clip(a*np.log(p/(1-p))+b,-745,745)))
for r in calrows.to_dict('records'):
 m,s,e=r['method'],int(r['seed']),r['endpoint'];ei=E.index(e);slots={}
 for role in ['calibration_A','threshold_B']:
  with np.load(OUT/f'scores/{role}/{m}/seed_{s}.npz',allow_pickle=False) as z:
   ck('AB_role',str(z['role'].item()),role);ck('AB_ids',np.array_equal(z['record_ids'].astype(str),roles[role]))
   yy=np.asarray([by[x]['labels'] for x in roles[role]],np.uint8);ck('AB_labels',np.array_equal(z['y'],yy))
   md=OUT/f"fits/{'ordinary' if m=='Huang_full_pipeline' else m}/seed_{s}"
   ck('AB_checkpoint',str(z['checkpoint_sha256'].item()),sha(md/'checkpoint.pt'));slots[role]=(z['p'][:,ei].astype(np.float64),yy[:,ei])
 pa,ya=slots['calibration_A'];pb,yb=slots['threshold_B'];q=float(np.quantile(pb[yb==0].astype(np.float32),.9,method='linear'))
 ck('B_q90',r['raw_q90'],q,1e-7)
 x=np.log(np.clip(pa,1e-6,1-1e-6)/(1-np.clip(pa,1e-6,1-1e-6)))
 def obj(v):
  z=np.exp(v[0])*x+v[1];return float(np.mean(np.logaddexp(0,z)-ya*z))
 opt=minimize(obj,np.array([0.,0.]),method='L-BFGS-B',bounds=[(-12,12),(-30,30)],options={'maxiter':2000,'ftol':1e-13,'gtol':1e-8,'maxls':50})
 ck('A_fit_converged',bool(opt.success));ck('A_fit_a',r['slope_a'],np.exp(opt.x[0]),1e-8);ck('A_fit_b',r['intercept_b'],opt.x[1],1e-8)
 ck('mapped_threshold',r['mapped_calibrated_threshold'],platt(np.asarray([r['raw_q90']]),r['slope_a'],r['intercept_b'])[0])
 fits[m,s,e]=(r['slope_a'],r['intercept_b'],r['raw_q90'])
data={}
for ds in labels:
 for m in M:
  for s in S:
   for c in C:
    with np.load(OUT/f'predictions/{ds}/{m}/seed_{s}/{c}.npz',allow_pickle=False) as z:
     data[ds,m,s,c]=z['p'].astype(np.float64);ck('prediction_ids',np.array_equal(z['ids'].astype(str),ids[ds]));ck('prediction_labels',np.array_equal(z['y'],labels[ds]))
     md=OUT/f"fits/{'ordinary' if m=='Huang_full_pipeline' else m}/seed_{s}"
     ck('prediction_checkpoint',str(z['checkpoint_sha256'].item()),sha(md/'checkpoint.pt'));ck('prediction_preprocessing',str(z['preprocessing_sha256'].item()),sha(md/'preprocessing.npz'))
def coeff(s,q):return np.asarray([1-q,0,q,0]) if s=='RA_LL_only' else np.asarray([1-q,q/3,q/3,q/3])
def met(y,p,w,a,b,t):
 keep=w>0;y=y[keep];p=p[keep];w=w[keep];cp=platt(p,a,b);cd=cp>=platt(np.array([t]),a,b)[0];rd=p>=t;pos=y==1;neg=~pos;total=w.sum();out={}
 for mode,pr,dec in [('raw',p,rd),('calibrated',cp,cd)]:
  ece=0
  for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
   mask=(pr>=lo)&((pr<hi) if hi<1 else (pr<=hi));mass=w[mask].sum()
   if mass:ece+=mass/total*abs(np.average(y[mask],weights=w[mask])-np.average(pr[mask],weights=w[mask]))
  out.update({f'AP_{mode}':average_precision_score(y,pr,sample_weight=w),f'AUROC_{mode}':roc_auc_score(y,pr,sample_weight=w),f'Brier_{mode}':np.average((pr-y)**2,weights=w),f'ECE15_{mode}':ece,
   f'sensitivity_{mode}':w[pos&dec].sum()/w[pos].sum(),f'specificity_{mode}':w[neg&~dec].sum()/w[neg].sum(),f'predicted_positive_fraction_{mode}':w[dec].sum()/total,f'FP_per_1000_{mode}':1000*w[neg&dec].sum()/total,f'FN_per_1000_{mode}':1000*w[pos&~dec].sum()/total})
 out['raw_vs_calibrated_decision_mismatch_weight']=w[rd!=cd].sum()/total;return out
for filename in ['CONDITION_METRICS_BY_SEED.csv','MIXTURE_METRICS_BY_SEED.csv']:
 table=pd.read_csv(OUT/filename,float_precision='round_trip')
 for r in table.to_dict('records'):
  ds,m,s,e=r['cohort'],r['method'],int(r['seed']),r['endpoint'];n=len(labels[ds]);ei=E.index(e);a,b,t=fits[m,s,e]
  if 'condition' in r:p=data[ds,m,s,r['condition']][:,ei];y=labels[ds][:,ei];w=np.ones(n)
  else:p=np.concatenate([data[ds,m,s,c][:,ei] for c in C]);y=np.tile(labels[ds][:,ei],4);w=np.repeat(coeff(r['scenario'],r['q_fault']),n)
  for key,v in met(y,p,w,a,b,t).items():ck('saved_metric',r[key],v,1e-10)
for detail,summary,keys in [
 ('CONDITION_METRICS_BY_SEED.csv','CONDITION_METRICS_MEAN_SD.csv',['cohort','method','endpoint','condition']),
 ('MIXTURE_METRICS_BY_SEED.csv','MIXTURE_METRICS_MEAN_SD.csv',['cohort','method','endpoint','scenario','q_fault'])]:
 src=pd.read_csv(OUT/detail,float_precision='round_trip')
 for row in pd.read_csv(OUT/summary,float_precision='round_trip').to_dict('records'):
  sub=src
  for k in keys:sub=sub[sub[k]==row[k]]
  ck('summary_seed_count',len(sub),3,0)
  for field,value in row.items():
   if field.endswith('_mean') or field.endswith('_sd'):
    base=field[:-5] if field.endswith('_mean') else field[:-3]
    target=sub[base].mean() if field.endswith('_mean') else sub[base].std(ddof=1)
    ck('mean_sd',value,target,1e-10)
print('METRICS_AND_FITS_COMPLETE',flush=True)
points=pd.read_csv(OUT/'TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv',float_precision='round_trip');mi=pd.read_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv',float_precision='round_trip')
indices={'Ningbo_test':np.load(ROOT/'outputs/ningbo_confirmation_v2/bootstrap_indices.npy',mmap_mode='r'),'HEEDB':np.load(ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy',mmap_mode='r')}
cache={}
def spot(ds,sc,q,m,s,metric,bi):
 key=ds,sc,q,m,s,metric,bi
 if key not in cache:
  pp=np.concatenate([data[ds,m,s,c][:,0] for c in C]);yy=np.tile(labels[ds][:,0],4);a,b,t=fits[m,s,'TWA'];n=len(labels[ds]);w=np.tile(np.bincount(indices[ds][bi],minlength=n),4)*np.repeat(coeff(sc,q),n);keep=w>0
  if metric=='AP_raw':v=average_precision_score(yy[keep],pp[keep],sample_weight=w[keep])
  elif metric=='Brier_raw':v=np.average((pp-yy)**2,weights=w)
  elif metric=='Brier_calibrated':v=np.average((platt(pp,a,b)-yy)**2,weights=w)
  elif metric=='sensitivity_raw':v=w[(yy==1)&(pp>=t)].sum()/w[yy==1].sum()
  else:v=1000*w[(yy==0)&(pp>=t)].sum()/w.sum()
  cache[key]=v
 return cache[key]
with np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz',allow_pickle=False) as dr:
 for r in points.to_dict('records'):
  ds,sc,q,l,rr,m=r['cohort'],r['scenario'],r['q_fault'],r['left'],r['right'],r['metric'];name=f'{ds}|{ds}|{sc}|q{q:g}|{l}-{rr}|{m}';v=dr[name]
  ck('draw_count',len(v),1000,0)
  for f,x in [('bootstrap_mean',v.mean()),('ci95_low',np.quantile(v,.025)),('ci95_high',np.quantile(v,.975))]:ck('bootstrap_summary',r[f],x)
  part=mi[(mi.cohort==ds)&(mi.scenario==sc)&(mi.q_fault==q)&(mi.endpoint=='TWA')]
  ck('full_point',r['full_cohort_point_mean_seed_contrast'],part[part.method==l][m].mean()-part[part.method==rr][m].mean())
  for bi in [0,417,999]:ck('actual_draw_spot',v[bi],np.mean([spot(ds,sc,q,l,s,m,bi)-spot(ds,sc,q,rr,s,m,bi) for s in S]),1e-10)
required=2*2*7*5*5
result={'numerical_status':'PASS' if not FAIL else 'FAIL','counts':N,'max_abs_errors':MAX,'failure_count':len(FAIL),'failures':FAIL[:30],
 'completeness':{'expected_bootstrap_groups':required,'saved_bootstrap_groups':len(points),'missing_q':[q for q in [0,.001,.005,.01,.05,.1,1.] if q not in set(points.q_fault)],'full_scope_complete':len(points)==required},
 'scope':'Independent stored hashes, record roles, best-epoch selection, A-only Platt refit/B-only threshold, prediction bindings, metrics and draw spots. No new model training or full inference. Training initialization order and GPU input replay reviewed separately.'}
(OUT/'INDEPENDENT_REVIEW_AUDIT_20261020.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if FAIL:raise SystemExit(1)
