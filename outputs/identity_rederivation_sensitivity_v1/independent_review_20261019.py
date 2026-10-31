"""Read-only audit of submitted artifacts; writes only this review's new JSON."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
counts={}; maxima={}; failures=[]
def js(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def check(kind,got,expected,tol=1e-12):
    counts[kind]=counts.get(kind,0)+1
    if isinstance(expected,(bool,str)):
        if got!=expected: failures.append([kind,str(got),str(expected)])
    else:
        d=abs(float(got)-float(expected)); maxima[kind]=max(maxima.get(kind,0),d)
        if not np.isfinite(d) or d>tol: failures.append([kind,str(got),str(expected)])
manifest=js(OUT/'REPRODUCIBILITY_MANIFEST.json')
for ent in manifest['files']:
    p=OUT/ent['path']; check('output_hash',sha(p),ent['sha256'])
    check('output_bytes',p.stat().st_size,ent['bytes'],0)
proto=js(OUT/'FROZEN_PROTOCOL.json'); ph=proto.pop('protocol_sha256')
canonical=json.dumps(proto,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
check('protocol_hash',hashlib.sha256(canonical).hexdigest(),ph)
for rel,h in proto['source_hashes'].items(): check('input_hash',sha(ROOT/rel),h)
print('HASHES_COMPLETE',len(proto['source_hashes']),flush=True)
endpoints=['TWA','SB','STach']; data={}; labels={}; ids={}
for key,u in proto['models'].items():
    ds,arm,seed=key.split('/'); seed=int(seed)
    for cond in ['original_clean','identity_rederived_clean','cycle_slots_120','cycle_slots_201']:
        path=(OUT/'predictions'/ds/arm/f'seed_{seed}'/'identity_rederived_clean.npz') if cond=='identity_rederived_clean' else ROOT/u['scores'][cond]['path']
        with np.load(path,allow_pickle=False) as z:
            pp=z['p'].copy()
            if cond=='identity_rederived_clean':
                check('prediction_protocol',str(z['protocol_sha256'].item()),ph)
                check('prediction_checkpoint',str(z['checkpoint_sha256'].item()),u['checkpoint']['sha256'])
                check('prediction_preprocessing',str(z['preprocessing_sha256'].item()),u['preprocessing']['sha256'])
            if 'y' in z and 'record_ids' in z:
                yy=z['y'].astype(np.uint8); ii=z['record_ids'].astype(str)
                if ds in labels:
                    check('labels_match',bool(np.array_equal(labels[ds],yy)),True)
                    check('ids_match',bool(np.array_equal(ids[ds],ii)),True)
                else: labels[ds]=yy; ids[ds]=ii
        check('prediction_finite',bool(np.isfinite(pp).all()),True)
        check('prediction_range',bool(((pp>=0)&(pp<=1)).all()),True)
        data[(ds,arm,seed,cond)]=(pp,np.asarray(u['thresholds'],dtype=np.float32))
for ds,yy in labels.items():
    check('canonical_labels',hashlib.sha256(yy.tobytes()).hexdigest(),proto['cohorts'][ds]['label_bytes_sha256'])
    encoded=json.dumps(ids[ds].tolist(),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    check('canonical_ids',hashlib.sha256(encoded).hexdigest(),proto['cohorts'][ds]['record_order_sha256'])
def metrics(yy,pp,threshold):
    pp=pp.astype(np.float64); dec=pp>=threshold
    return {'AP_AUPRC':average_precision_score(yy,pp),'AUROC':roc_auc_score(yy,pp),
            'raw_Brier':np.mean((pp-yy)**2),'sensitivity':np.mean(dec[yy==1]),'specificity':np.mean(~dec[yy==0])}
computed={}
for key,(pp,t) in data.items():
    for j,ep in enumerate(endpoints): computed[(*key,ep)]=metrics(labels[key[0]][:,j],pp[:,j],t[j])
for r in rows(OUT/'CONDITION_RESULTS_BY_SEED.csv'):
    key=(r['cohort'],r['arm'],int(r['seed']),r['condition'],r['endpoint'])
    for metric,value in computed[key].items(): check('metric_value',float(r[metric]),value)
    check('threshold',float(r['threshold']),data[key[:4]][1][endpoints.index(key[-1])],0)
for r in rows(OUT/'CONDITION_RESULTS_MEAN_SD.csv'):
    for metric in ['AP_AUPRC','AUROC','raw_Brier','sensitivity','specificity']:
        values=[computed[(r['cohort'],r['arm'],s,r['condition'],r['endpoint'])][metric] for s in proto['seeds']]
        check('mean_sd',float(r[metric+'_mean']),np.mean(values))
        check('mean_sd',float(r[metric+'_sd']),np.std(values,ddof=1))
bases={'identity_minus_original':('identity_rederived_clean','original_clean'),
       'cycle120_minus_original':('cycle_slots_120','original_clean'),'cycle120_minus_identity':('cycle_slots_120','identity_rederived_clean'),
       'cycle201_minus_original':('cycle_slots_201','original_clean'),'cycle201_minus_identity':('cycle_slots_201','identity_rederived_clean')}
for r in rows(OUT/'SCORE_CHANGES_BY_SEED.csv'):
    ds,arm,s=r['cohort'],r['arm'],int(r['seed']); j=endpoints.index(r['endpoint']); a,b=bases[r['comparison']]
    pa,ta=data[(ds,arm,s,a)]; pb,tb=data[(ds,arm,s,b)]; delta=np.abs(pa[:,j]-pb[:,j])
    vals={'score_max_abs_change':delta.max(),'score_mean_abs_change':delta.mean(),
          'score_p50_abs_change':np.quantile(delta,.5),'score_p95_abs_change':np.quantile(delta,.95),'score_p99_abs_change':np.quantile(delta,.99),
          'decision_flip_fraction':np.mean((pa[:,j]>=ta[j])!=(pb[:,j]>=tb[j]))}
    for k,v in vals.items(): check('score_change',float(r[k]),v)
contrasts=rows(OUT/'PAIRED_CONTRASTS.csv'); draws={}
for r in rows(OUT/'BOOTSTRAP_REPLICATES.csv'):
    k=(r['cohort'],r['contrast'],r['metric']); draws.setdefault(k,{})[int(r['replicate'])]=float(r['difference'])
boots={ds:np.load(ROOT/p['bootstrap']['path'],mmap_mode='r') for ds,p in proto['cohorts'].items()}
def decode(c):
    if '_minus_ordinary_on_' in c:
        arm,condition=c.split('_minus_ordinary_on_'); return arm,condition,'ordinary',condition
    expr,arm=c.rsplit('_for_',1)
    a,b=expr.split('_minus_',1); return arm,a,arm,b
cache={}
def sampled(ds,arm,s,cond,draw,metric):
    k=(ds,arm,s,cond,draw,metric)
    if k not in cache:
        yy=labels[ds][:,0]; pp,t=data[(ds,arm,s,cond)]; ix=boots[ds][draw]
        p=pp[ix,0].astype(np.float64); y=yy[ix]
        cache[k]={'AP_AUPRC':lambda:average_precision_score(y,p),'AUROC':lambda:roc_auc_score(y,p),'raw_Brier':lambda:np.mean((p-y)**2)}[metric]()
    return cache[k]
for r in contrasts:
    ds,c,m=r['cohort'],r['contrast'],r['metric']; a,ac,b,bc=decode(c)
    point=np.mean([computed[(ds,a,s,ac,'TWA')][m]-computed[(ds,b,s,bc,'TWA')][m] for s in proto['seeds']])
    # Submission's full-cohort Brier contrasts square float32 scores before
    # weighted averaging; the condition table and this audit square in float64.
    # Preserve this observed discrepancy (max ~3.24e-11), with explicit tolerance.
    check('contrast_point',float(r['full_cohort_point_estimate']),point,1e-9 if m=='raw_Brier' else 1e-12)
    d=draws[(ds,c,m)]; check('replicate_count',len(d),1000,0); check('replicate_index_set',set(d)==set(range(1000)),True)
    v=np.array([d[i] for i in range(1000)])
    for f,x in [('bootstrap_mean',v.mean()),('ci_2_5',np.quantile(v,.025)),('ci_97_5',np.quantile(v,.975))]: check('bootstrap_summary',float(r[f]),x)
    for bi in [0,417,999]:
        actual=np.mean([sampled(ds,a,s,ac,bi,m)-sampled(ds,b,s,bc,bi,m) for s in proto['seeds']])
        check('bootstrap_actual_spot',d[bi],actual,5e-12)
identity=[r for r in contrasts if r['contrast'].startswith('identity_') and r['metric']=='AP_AUPRC']
result={'status':'PASS' if not failures else 'FAIL','counts':counts,'max_abs_errors':maxima,
        'failure_count':len(failures),'failures':failures[:30],
        'identity_twa_ap_points':[{'cohort':r['cohort'],'contrast':r['contrast'],'point':float(r['full_cohort_point_estimate'])} for r in identity],
        'precision_note':'Brier full-cohort contrast points use float32 squaring in submission versus float64 in condition table/audit; explicit comparison tolerance 1e-9, observed maximum below 3.24e-11. Other point tolerance 1e-12; bootstrap spots 5e-12. Strict first attempt preserved separately.',
        'scope':'Independent saved-file hash and numerical review. No new GPU replay, full signal reconstruction, or new inference performed.'}
(OUT/'INDEPENDENT_REVIEW_AUDIT_20261019.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
if failures: raise SystemExit(1)
