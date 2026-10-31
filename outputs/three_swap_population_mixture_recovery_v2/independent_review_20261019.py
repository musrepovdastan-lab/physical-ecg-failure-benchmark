"""Independent saved-artifact review. No imports of experiment workflows."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
COUNTS={}; MAX={}; FAIL=[]; notes=[]
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ck(k,a,b,tol=1e-11):
 COUNTS[k]=COUNTS.get(k,0)+1
 if isinstance(b,(str,bool)):
  if a!=b:FAIL.append([k,str(a),str(b)])
 else:
  d=abs(float(a)-float(b));MAX[k]=max(MAX.get(k,0),d)
  if not np.isfinite(d) or d>tol:FAIL.append([k,str(a),str(b)])
manifest=pd.read_csv(OUT/'OUTPUT_SHA256_MANIFEST.csv')
for r in manifest.itertuples():
 p=OUT/r.path;ck('output_hash',sha(p),r.sha256);ck('output_size',p.stat().st_size,r.bytes,0)
proto=j(OUT/'RECOVERY_PROTOCOL.json')
protocol_hash=sha(OUT/'RECOVERY_PROTOCOL.json')
ck('protocol_file_hash',protocol_hash,(OUT/'RECOVERY_PROTOCOL_SHA256.txt').read_text(encoding='utf-8').strip())
body={k:v for k,v in proto.items() if k!='canonical_json_fingerprint_sha256'}
canonical=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
ck('canonical_protocol',hashlib.sha256(canonical).hexdigest(),proto['canonical_json_fingerprint_sha256'])
for rel,b in proto['input_bindings'].items():
 actual=sha(ROOT/rel)
 if rel=='NEXT_RESEARCH_DECISIONS_20261019.md' and actual!=b['sha256']:
  notes.append('The explicitly mutable active plan changed after execution; all scientific inputs checked separately. Frozen text reconstruction is attempted by reviewer separately.')
  continue
 ck('input_hash',actual,b['sha256'])
for r in pd.read_csv(OUT/'INPUT_SHA256_MANIFEST.csv').itertuples():ck('input_manifest_hash',sha(ROOT/r.path),r.sha256)
coverage=pd.read_csv(OUT/'INPUT_COVERAGE.csv',keep_default_na=False)
coverage_index={(r.source,r.method,int(r.seed),r.condition):r for r in coverage.itertuples()}
ck('coverage_count',len(coverage_index),96,0)
print('HASHES_COMPLETE',flush=True)
S=(104729,130363,155921); M=('ordinary_resnet','physics_aware_p050','combined','Huang_full_pipeline')
C=('clean','RA_LL_reversal','RA_LA_reversal','LA_LL_reversal'); E=('TWA','SB','STach')
can=j(ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json')['records']; by={r['record_id']:r for r in can}
membership=j(ROOT/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json')['rows']
nr=[by[r['record_id']] for r in membership if r['split']=='test']; ids={'Ningbo':np.asarray([r['record_id'] for r in nr])}
labels={'Ningbo':np.asarray([r['labels'] for r in nr],dtype=np.uint8)}
selection=pd.read_csv(ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv',dtype=str,keep_default_na=False)
diagnoses=pd.read_csv(ROOT/'data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv',usecols=['FileName','codes_physician'],dtype=str,keep_default_na=False)
df=selection[['FileName','BDSPPatientID']].merge(diagnoses,on='FileName',how='left',sort=False,validate='one_to_one')
ck('heedb_unique_patients',df.BDSPPatientID.nunique(),10000,0)
core=({1140,1145,1150,1151,1160,1161,1170,1171,1172,1180,1181},{21,24},{23})
labels['HEEDB']=np.array([[int(bool({int(x) for x in re.findall(r'\d+',v)}&codes)) for codes in core] for v in df.codes_physician],dtype=np.uint8)
ids['HEEDB']=df.FileName.to_numpy(); pids=df.BDSPPatientID.to_numpy()
cohort=pd.read_csv(OUT/'COHORT_ORDER_AND_LABELS.csv',dtype={'record_or_file_id':str})
for ds in labels:
 part=cohort[cohort.source==ds]
 ck('saved_ids',bool(np.array_equal(part.record_or_file_id.to_numpy(),ids[ds])),True)
 ck('saved_labels',bool(np.array_equal(part[list(E)].to_numpy(dtype=np.uint8),labels[ds])),True)
mixfits=j(ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json')['calibration_fits']
n4=j(ROOT/'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json')['fits']
hu=j(ROOT/'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json')['fits_by_seed']
fits={}
for m in M:
 for s in S:
  fits[m,s]=n4[f'combined/{s}'] if m=='combined' else hu[str(s)] if m=='Huang_full_pipeline' else mixfits['physics_aware_resnet' if m=='physics_aware_p050' else m][str(s)]
def threshold(f,mode):return float(f.get('threshold_raw_validation_B_90pct',f.get('threshold_raw'))) if mode=='raw' else float(f.get('threshold_calibrated_validation_B_90pct',f.get('threshold_calibrated')))
def calibrate(p,f):
 c=np.clip(p.astype(np.float64),1e-6,1-1e-6);z=float(f['a'])*np.log(c/(1-c))+float(f['b'])
 return 1/(1+np.exp(-np.clip(z,-745,745)))
data={}; paths={}
for ds in labels:
 for m in M:
  for s in S:
   for c in C:
    binding=coverage_index[ds,m,s,c];p=ROOT/binding.final_prediction_path
    ck('coverage_final_hash',sha(p),binding.final_prediction_sha256)
    ck('coverage_source_hash',sha(ROOT/binding.input_path),binding.input_sha256)
    with np.load(p,allow_pickle=False) as z:
     pp=z['p'].astype(np.float64)
     for lk in ['y','labels']:
      if lk in z:ck('prediction_labels',bool(np.array_equal(z[lk],labels[ds])),True)
     if 'record_ids' in z:ck('prediction_ids',bool(np.array_equal(z['record_ids'].astype(str),ids[ds])),True)
     if 'patient_ids' in z:ck('prediction_patients',bool(np.array_equal(z['patient_ids'].astype(str),pids)),True)
     if binding.status=='NEW_CORRECTED_INFERENCE':
      ck('prediction_protocol',str(z['recovery_protocol_file_sha256'].item()),protocol_hash)
      ck('prediction_condition',str(z['incoming_condition'].item()),c)
    if binding.status!='NEW_CORRECTED_INFERENCE':
     with np.load(ROOT/binding.input_path,allow_pickle=False) as z:ck('unaffected_scores_exact',bool(np.array_equal(pp,z['p'])),True)
    ck('probability_shape',pp.shape==(len(labels[ds]),3),True)
    ck('probability_valid',bool(np.isfinite(pp).all() and (pp>=0).all() and (pp<=1).all()),True)
    data[ds,m,s,c]=pp;paths[ds,m,s,c]=p
def coeff(sc,q):return np.array([1-q,q/3,q/3,q/3]) if sc=='equal_three_swap' else np.array([1-q,q,0,0])
def metrics(y,p,t,w):
 a=w>0;y=y[a];p=p[a];w=w[a];total=w.sum();pos=y==1;dec=p>=t
 tp=w[pos&dec].sum();fn=w[pos&~dec].sum();fp=w[~pos&dec].sum();tn=w[~pos&~dec].sum()
 ece=0
 for b in range(15):
  ix=(p>=b/15)&((p<(b+1)/15) if b<14 else (p<=1))
  if ix.any():ece+=w[ix].sum()/total*abs(np.average(y[ix],weights=w[ix])-np.average(p[ix],weights=w[ix]))
 return dict(AP=average_precision_score(y,p,sample_weight=w),AUROC=roc_auc_score(y,p,sample_weight=w),Brier=np.average((p-y)**2,weights=w),ECE15=ece,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),predicted_positive_fraction=(tp+fp)/total,FP_per_1000=1000*fp/total,FN_per_1000=1000*fn/total,TP=tp,FP=fp,TN=tn,FN=fn)
tables={f:pd.read_csv(OUT/f,float_precision='round_trip') for f in ['CONDITION_METRICS_BY_SEED.csv','MIXTURE_METRICS_BY_SEED.csv','MIXTURE_METRICS_MEAN_SD.csv','PAIRED_CONTRASTS_POINT_AND_CI.csv']}
scores={}
for ds in labels:
 for m in M:
  for s in S:
   for ei,e in enumerate(E):
    f=fits[m,s][ei];raw=np.concatenate([data[ds,m,s,c][:,ei] for c in C])
    for mode in ['raw','calibrated']:scores[ds,m,s,e,mode]=raw if mode=='raw' else calibrate(raw,f)
for filename in ['CONDITION_METRICS_BY_SEED.csv','MIXTURE_METRICS_BY_SEED.csv']:
 for r in tables[filename].to_dict('records'):
  ds,m,s,e,mode=r['source'],r['method'],int(r['seed']),r['endpoint'],r['score_mode'];n=len(labels[ds]);f=fits[m,s][E.index(e)]
  p=scores[ds,m,s,e,mode];y=np.tile(labels[ds][:,E.index(e)],4)
  co=np.eye(4)[C.index(r['condition'])] if 'condition' in r else coeff(r['scenario'],r['q_fault'])
  w=np.repeat(co,n);ref=metrics(y,p,threshold(f,mode),w)
  ck('threshold',r['threshold'],threshold(f,mode),0)
  for k,v in ref.items():ck('condition_metric' if 'condition' in r else 'mixture_metric',r[k],v,1e-8 if k in ['TP','FP','TN','FN','FP_per_1000','FN_per_1000'] else 1e-11)
print('ALL_METRICS_COMPLETE',flush=True)
mi=tables['MIXTURE_METRICS_BY_SEED.csv']; points={}
for r in tables['PAIRED_CONTRASTS_POINT_AND_CI.csv'].to_dict('records'):
 ds,sc,q,c,metric=r['source'],r['scenario'],r['q_fault'],r['contrast'],r['metric'];a,b=c.split('-')
 col={'raw_TWA_AP':'AP','raw_TWA_Brier':'Brier','calibrated_TWA_Brier':'Brier'}.get(metric,metric)
 mode='raw' if metric.startswith('raw_') else 'calibrated'
 part=mi[(mi.source==ds)&(mi.scenario==sc)&(mi.q_fault==q)&(mi.endpoint=='TWA')&(mi.score_mode==mode)]
 ck('paired_point',r['full_sample_difference'],part[part.method==a][col].mean()-part[part.method==b][col].mean())
 points[ds,sc,q,c,metric]=r
with np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:
 names=z['draw_names'].astype(str);dd=z['draws'].copy()
boots={ds:np.load(ROOT/p,mmap_mode='r',allow_pickle=False) for ds,p in {'Ningbo':'outputs/ningbo_confirmation_v2/bootstrap_indices.npy','HEEDB':'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'}.items()}
cache={}
def spot(ds,sc,q,m,metric,bi):
 key=ds,sc,q,m,metric,bi
 if key in cache:return cache[key]
 co=coeff(sc,q);ix=boots[ds][bi];n=len(labels[ds]);w=np.repeat(co,n)*np.tile(np.bincount(ix,minlength=n),4);y=np.tile(labels[ds][:,0],4);keep=w>0;vals=[]
 for s in S:
  mode='raw' if metric.startswith('raw_') else 'calibrated';p=scores[ds,m,s,'TWA',mode];f=fits[m,s][0];t=threshold(f,mode)
  if metric=='raw_TWA_AP':v=average_precision_score(y[keep],p[keep],sample_weight=w[keep])
  elif metric.endswith('Brier'):v=np.average((p-y)**2,weights=w)
  elif metric=='sensitivity':v=w[(y==1)&(p>=t)].sum()/w[y==1].sum()
  else:v=1000*w[(y==0)&(p>=t)].sum()/w.sum()
  vals.append(v)
 cache[key]=np.mean(vals);return cache[key]
for name,d in zip(names,dd):
 ds,sc,qq,c,metric=name.split('|');q=float(qq.split('=')[1]);r=points[ds,sc,q,c,metric]
 ck('draw_length',len(d),1000,0)
 for field,v in [('bootstrap_mean_difference',d.mean()),('ci_2_5',np.quantile(d,.025)),('ci_97_5',np.quantile(d,.975))]:ck('bootstrap_summary',r[field],v)
 a,b=c.split('-')
 for bi in [0,417,999]:ck('actual_draw_spot',d[bi],spot(ds,sc,q,a,metric,bi)-spot(ds,sc,q,b,metric,bi),1e-9)
summary=tables['MIXTURE_METRICS_MEAN_SD.csv']
for r in summary.to_dict('records'):
 sub=mi[(mi.source==r['source'])&(mi.method==r['method'])&(mi.endpoint==r['endpoint'])&(mi.scenario==r['scenario'])&(mi.q_fault==r['q_fault'])&(mi.score_mode==r['score_mode'])]
 ck('mean_seed_count',len(sub),3,0)
 for col in ['AP','AUROC','Brier','ECE15','sensitivity','specificity','predicted_positive_fraction','FP_per_1000','FN_per_1000']:
  ck('mean_sd',r[col+'_mean'],sub[col].mean())
  ck('mean_sd',r[col+'_std'],sub[col].std(ddof=1))
special=tables['PAIRED_CONTRASTS_POINT_AND_CI.csv'];special=special[(special.source=='HEEDB')&(special.scenario=='equal_three_swap')&(special.q_fault.isin([.01,1]))&(special.metric.isin(['raw_TWA_AP','calibrated_TWA_Brier']))]
result={'status':'PASS' if not FAIL else 'FAIL','counts':COUNTS,'max_abs_errors':MAX,'failures':FAIL[:30],'failure_count':len(FAIL),'notes':notes,'scope':'Independent saved-file/label/metric/draw review. No GPU replay or new inference. Historical training/data selection limitations remain.','key_contrasts':special.to_dict('records')}
(OUT/'INDEPENDENT_REVIEW_AUDIT_20261019.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='key_contrasts'},ensure_ascii=False),flush=True)
if FAIL:raise SystemExit(1)
