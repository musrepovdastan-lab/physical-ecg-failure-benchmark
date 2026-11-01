from pathlib import Path
import sys,json,csv,hashlib,os,time
from functools import lru_cache
import numpy as np,torch,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT));import control as c
N=c.N
from src.chapman_phase1 import ECGResNetSmall
from src.ecg_physics import derive_display,named_reversal
from src.heedb_huang_adapter_v1 import corrected_batch,diagnostic_input,electrical_reversal_batch
from src import huang_detector_infer_v2 as detector
CONDS=('clean',*c.FAULTS);SCENARIOS=('RA_LL_only','equal_three_swap');Q=(0.,.001,.005,.01,.05,.1,1.)
METHODS=(*c.METHODS,'Huang_full_pipeline');CONTRASTS=(('physics_aware_p050','ordinary'),('combined','ordinary'),('Huang_full_pipeline','ordinary'),('physics_aware_p050','Huang_full_pipeline'),('combined','Huang_full_pipeline'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(dest,obj):
 dest=Path(dest);tmp=dest.with_name('.'+dest.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8');os.replace(tmp,dest)
def atomic_npz(dest,**kw):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.stem+'.tmp.npz');np.savez_compressed(tmp,**kw);os.replace(tmp,dest)
def write_csv(dest,rows):
 if not rows:return
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.name+'.tmp')
 with tmp.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(tmp,dest)
def verify_locks():
 p=json.loads((OUT/'CALIBRATION_AND_THRESHOLD_LOCK.json').read_text(encoding='utf8'));m=json.loads((OUT/'ALL_MODELS_LOCK.json').read_text(encoding='utf8'))
 if p.get('status')!='LOCKED' or len(p['entries'])!=36 or m.get('decision')!='PASS' or len(m['fits'])!=9:raise RuntimeError('test gate closed: incomplete locks')
 for f in m['fits']:
  d=OUT/'fits'/f['method']/f"seed_{f['seed']}"
  if sha(d/'checkpoint.pt')!=f['checkpoint_sha256']:raise RuntimeError('checkpoint changed after lock')
 return p,m
def load_cal():
 rows=list(csv.DictReader((OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv').open(encoding='utf8')));d={}
 for x in rows:d[x['method'],int(x['seed']),x['endpoint']]=(float(x['slope_a']),float(x['intercept_b']),float(x['raw_q90']),float(x['mapped_calibrated_threshold']))
 if len(d)!=36:raise RuntimeError('calibration entry count')
 return d
def platt(x,a,b):
 z=np.log(np.clip(x.astype(np.float64),1e-6,1-1e-6)/(1-np.clip(x.astype(np.float64),1e-6,1-1e-6)))*a+b
 return 1/(1+np.exp(-np.clip(z,-700,700)))
def raw500(rr):return np.stack([detector.load_full_display(r) for r in rr]).astype(np.float32)
def infer500(model,arr,mean,std):
 z=np.stack([diagnostic_input(x,mean,std) for x in arr]).astype(np.float32);out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(z),128):
   with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(z[i:i+128]).cuda(non_blocking=True))).float().cpu().numpy())
 return np.concatenate(out).astype(np.float32)
def save_pred(cohort,method,seed,cond,p,y,ids,metadata):
 dest=OUT/f'predictions/{cohort}/{method}/seed_{seed}/{cond}.npz';atomic_npz(dest,p=np.asarray(p,np.float32),y=np.asarray(y,np.uint8),ids=np.asarray(ids).astype(str),method=np.asarray(method),seed=np.asarray(seed),condition=np.asarray(cond),**metadata);return dest
def load_det_heedb():
 f=ROOT/'outputs/three_swap_population_mixture_recovery_v2/huang_detector_probabilities.npz'
 with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 expected=json.loads((ROOT/'outputs/three_swap_population_mixture_recovery_v2/RECOVERY_PROTOCOL.json').read_text(encoding='utf8'))['source_cohort']['selection_sha256']
 # The recovery package records selection/source order separately from its bytewise FileName-list hash.
 if str(d['detector_index_sha256'].item())!=json.loads((ROOT/'outputs/three_swap_population_mixture_recovery_v2/DETECTOR_BLOCK_REPLAY_AUDIT.json').read_text(encoding='utf8'))['checkpoint_index_sha256']:raise RuntimeError('HEEDB detector checkpoint/index binding mismatch')
 if list(d['conditions'].astype(str))!=list(CONDS) and set(d['conditions'].astype(str))!=set(CONDS):raise RuntimeError('HEEDB detector conditions mismatch')
 return d,sha(f),expected
def infer_cohort(cohort,records,y,ids,raw,detprob=None,detconds=None,detfilehash=None,raw_huang=None):
 rows=[];started=time.monotonic();
 for method in METHODS:
  for seed in c.SEEDS:
   d=OUT/'fits'/('ordinary' if method=='Huang_full_pipeline' else method)/f'seed_{seed}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict']);mean,std=np.load(d/'preprocessing.npz',allow_pickle=False)['mean'].astype(np.float32),np.load(d/'preprocessing.npz',allow_pickle=False)['std'].astype(np.float32)
   cksha=sha(d/'checkpoint.pt');prsha=sha(d/'preprocessing.npz')
   for ci,cond in enumerate(CONDS):
    dest=OUT/f'predictions/{cohort}/{method}/seed_{seed}/{cond}.npz'
    if dest.exists():
     with np.load(dest,allow_pickle=False) as z:
      if z['p'].shape!=(len(y),3) or not np.array_equal(z['y'],y) or not np.array_equal(z['ids'].astype(str),ids.astype(str)) or str(z['checkpoint_sha256'].item())!=cksha:raise RuntimeError('partial prediction binding mismatch '+str(dest))
      p=z['p'].copy()
     rows.append({'cohort':cohort,'method':method,'seed':seed,'condition':cond,'path':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest),'rows':len(y),'status':'REUSED'});continue
    if method!='Huang_full_pipeline':
     if cohort=='Ningbo_test':x=raw if ci==0 else np.stack([c.reverse_raw(a,cond) for a in raw])
     else:x=raw if ci==0 else electrical_reversal_batch(raw,cond)
     if cohort=='Ningbo_test':p=c.infer(model,x,mean,std)
     else:p=infer500(model,x,mean,std)
     meta={'checkpoint_sha256':np.asarray(cksha),'preprocessing_sha256':np.asarray(prsha),'source_order_sha256':np.asarray(hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest())}
    else:
     if detprob is None:raise RuntimeError('Huang detector probabilities missing')
     huang_input=raw if raw_huang is None else raw_huang
     if cohort=='Ningbo_test':incoming=huang_input if ci==0 else np.stack([detector.apply_condition(a,cond) for a in huang_input]);dpi=detconds.index(cond);probs=detprob[dpi];fixed=corrected_batch(incoming,np.argmax(probs,axis=1).astype(np.int64));p=infer500(model,fixed,mean,std)
     else:
      incoming=huang_input if ci==0 else electrical_reversal_batch(huang_input,cond);dpi=detconds.index(cond);probs=detprob[dpi];fixed=corrected_batch(incoming,np.argmax(probs,axis=1).astype(np.int64));p=infer500(model,fixed,mean,std)
     meta={'checkpoint_sha256':np.asarray(cksha),'preprocessing_sha256':np.asarray(prsha),'detector_probabilities_sha256':np.asarray(detfilehash),'detector_condition':np.asarray(cond),'source_order_sha256':np.asarray(hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest())}
    if p.shape!=(len(y),3) or not np.isfinite(p).all():raise RuntimeError('invalid full prediction')
    save_pred(cohort,method,seed,cond,p,y,ids,meta);rows.append({'cohort':cohort,'method':method,'seed':seed,'condition':cond,'path':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest),'rows':len(y),'status':'COMPLETE'})
    log={'cohort':cohort,'method':method,'seed':seed,'condition':cond,'elapsed_seconds':round(time.monotonic()-started,1),'complete_predictions':len(rows)};atomic_json(OUT/'EVALUATION_PROGRESS.json',log)
   del model;torch.cuda.empty_cache()
 atomic_json(OUT/f'{cohort}_PREDICTIONS_COMPLETE.json',{'status':'PASS','rows':rows,'source_order_sha256':hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest(),'label_sha256':hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started});return rows
def weighted_metrics(y,p,w,threshold,a,b):
 y=np.asarray(y,dtype=np.uint8);p=np.asarray(p,dtype=np.float64);w=np.asarray(w,dtype=np.float64);z=w.sum();cal=platt(p,a,b);raw_dec=p>=threshold;cal_dec=cal>=platt(np.asarray([threshold]),a,b)[0];pos=(y==1);neg=~pos
 def rate(mask,den):return float(np.sum(w*mask)/den) if den else float('nan')
 ece=0.;edges=np.linspace(0,1,16)
 for i in range(15):
  mask=(cal>=edges[i])&((cal<edges[i+1]) if i<14 else (cal<=edges[i+1]));bw=float(w[mask].sum())
  if bw:ece+=bw/z*abs(float(np.sum(w[mask]*y[mask])/bw)-float(np.sum(w[mask]*cal[mask])/bw))
 return {'AP_raw':float(average_precision_score(y,p,sample_weight=w)),'AUROC_raw':float(roc_auc_score(y,p,sample_weight=w)),'Brier_raw':float(np.sum(w*(p-y)**2)/z),'ECE15_raw':float(ece) if False else float(_ece(y,p,w)),
  'AP_calibrated':float(average_precision_score(y,cal,sample_weight=w)),'AUROC_calibrated':float(roc_auc_score(y,cal,sample_weight=w)),'Brier_calibrated':float(np.sum(w*(cal-y)**2)/z),'ECE15_calibrated':float(_ece(y,cal,w)),
  'sensitivity_raw':rate(raw_dec&pos,float(np.sum(w*pos))),'specificity_raw':rate((~raw_dec)&neg,float(np.sum(w*neg))),'predicted_positive_fraction_raw':float(np.sum(w*raw_dec)/z),'FP_per_1000_raw':float(1000*np.sum(w*raw_dec*neg)/z),'FN_per_1000_raw':float(1000*np.sum(w*(~raw_dec)*pos)/z),
  'sensitivity_calibrated':rate(cal_dec&pos,float(np.sum(w*pos))),'specificity_calibrated':rate((~cal_dec)&neg,float(np.sum(w*neg))),'predicted_positive_fraction_calibrated':float(np.sum(w*cal_dec)/z),'FP_per_1000_calibrated':float(1000*np.sum(w*cal_dec*neg)/z),'FN_per_1000_calibrated':float(1000*np.sum(w*(~cal_dec)*pos)/z),'raw_vs_calibrated_decision_mismatch_weight':float(np.sum(w*(raw_dec!=cal_dec))/z),'threshold_raw':threshold,'threshold_calibrated_mapped':float(platt(np.asarray([threshold]),a,b)[0])}
def _ece(y,p,w):
 edges=np.linspace(0,1,16);out=0.;total=w.sum()
 for i in range(15):
  m=(p>=edges[i])&((p<edges[i+1]) if i<14 else (p<=edges[i+1]));z=w[m].sum()
  if z:out+=z/total*abs(float(np.sum(w[m]*y[m])/z)-float(np.sum(w[m]*p[m])/z))
 return out
@lru_cache(maxsize=128)
def getpred(cohort,method,seed,cond):
 with np.load(OUT/f'predictions/{cohort}/{method}/seed_{seed}/{cond}.npz',allow_pickle=False) as z:return z['p'].astype(np.float64)
def metric_tables(cohort,y,cal):
 condition_rows=[];mix_rows=[]
 for method in METHODS:
  for seed in c.SEEDS:
   for ei,ep in enumerate(c.ENDS):
    a,b,t,_=cal[method,seed,ep]
    for cond in CONDS:
     p=getpred(cohort,method,seed,cond)[:,ei];m=weighted_metrics(y[:,ei],p,np.ones(len(y)),t,a,b);condition_rows.append({'cohort':cohort,'method':method,'seed':seed,'endpoint':ep,'condition':cond,**m})
    scores=[getpred(cohort,method,seed,co)[:,ei] for co in CONDS]
    for scen in SCENARIOS:
     for q in Q:
      if scen=='RA_LL_only':weights=np.asarray([1-q,0,q,0],float)
      else:weights=np.asarray([1-q,q/3,q/3,q/3],float)
      pp=np.concatenate(scores);yy=np.tile(y[:,ei],4);ww=np.repeat(weights,len(y));m=weighted_metrics(yy,pp,ww,t,a,b)
      mix_rows.append({'cohort':cohort,'method':method,'seed':seed,'endpoint':ep,'scenario':scen,'q_fault':q,'weighted_views_per_unit':float(weights.sum()),**m})
 return condition_rows,mix_rows
def _tie_group_ap(group_desc,yy,weight,n_groups):
 positive=np.bincount(group_desc,weights=weight*yy,minlength=n_groups).astype(np.float64)
 total=np.bincount(group_desc,weights=weight,minlength=n_groups).astype(np.float64)
 denom=positive.sum()
 if denom<=0:return float('nan')
 tp=np.cumsum(positive);seen=np.cumsum(total);precision=np.divide(tp,seen,out=np.zeros_like(tp),where=seen>0)
 return float(np.sum((positive/denom)*precision))
def bootstrap(cohort,y,cal,indices):
 # Exact weighted AP after precomputing tied score groups; every draw still uses the frozen unit multiplicities.
 rows=[];draws={};checks=[];n=len(y);metrics=('AP_raw','Brier_raw','Brier_calibrated','sensitivity_raw','FP_per_1000_raw');yy=np.tile(y[:,0],4)
 cache={}
 for seed in c.SEEDS:
  for method in METHODS:
   pp=np.concatenate([getpred(cohort,method,seed,co)[:,0] for co in CONDS]);u,inv=np.unique(pp,return_inverse=True);cache[method,seed]=(pp,len(u)-1-inv,len(u))
 for scen in SCENARIOS:
  for q in (.01,.1,1.0):
   condw=np.asarray([1-q,0,q,0] if scen=='RA_LL_only' else [1-q,q/3,q/3,q/3],float);values={m:np.full((1000,len(c.SEEDS),len(metrics)),np.nan,float) for m in METHODS}
   for bi,idx in enumerate(indices):
    counts=np.bincount(np.asarray(idx,dtype=np.int64),minlength=n).astype(float);mult=np.concatenate([counts*w for w in condw])
    for si,seed in enumerate(c.SEEDS):
     for method in METHODS:
      a,b,t,_=cal[method,seed,'TWA'];pp,grp,ng=cache[method,seed];dec=pp>=t;pos=(yy==1);neg=~pos;calp=platt(pp,a,b)
      vals=(_tie_group_ap(grp,yy,mult,ng),float(np.sum(mult*(pp-yy)**2)/mult.sum()),float(np.sum(mult*(calp-yy)**2)/mult.sum()),float(np.sum(mult*dec*pos)/np.sum(mult*pos)),float(1000*np.sum(mult*dec*neg)/mult.sum()))
      values[method][bi,si,:]=vals
      if bi in (0,417,999):checks.append({'cohort':cohort,'scenario':scen,'q_fault':q,'draw':bi,'method':method,'seed':seed,'ap_tie_group':vals[0],'ap_sklearn':float(average_precision_score(yy,pp,sample_weight=mult)),'absolute_delta':abs(vals[0]-float(average_precision_score(yy,pp,sample_weight=mult)))})
   for left,right in CONTRASTS:
    for mi,metric in enumerate(metrics):
     v=np.mean(values[left][:,:,mi]-values[right][:,:,mi],axis=1);name=f'{cohort}|{scen}|q{q:g}|{left}-{right}|{metric}';draws[name]=v
     point=float(np.mean([_point_contrast(cohort,y,cal,left,right,scen,q,metric,s) for s in c.SEEDS]))
     rows.append({'cohort':cohort,'scenario':scen,'q_fault':q,'left':left,'right':right,'metric':metric,'full_cohort_point_mean_seed_contrast':point,'bootstrap_mean':float(np.mean(v)),'ci95_low':float(np.quantile(v,.025)),'ci95_high':float(np.quantile(v,.975)),'n_draws':int(np.isfinite(v).sum()),'unit':'record' if cohort=='Ningbo_test' else 'patient','conditional_on_fitted_models_and_calibrators':True})
 if max(x['absolute_delta'] for x in checks)>1e-12:raise RuntimeError('tie-group AP does not match sklearn at selected draws')
 write_csv(OUT/f'{cohort}_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv',checks)
 atomic_npz(OUT/f'{cohort}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',**draws);return rows,draws
def _point_contrast(cohort,y,cal,left,right,scen,q,metric,seed):
 w=np.asarray([1-q,0,q,0] if scen=='RA_LL_only' else [1-q,q/3,q/3,q/3],float);mult=np.concatenate([np.full(len(y),z) for z in w]);vals=[]
 for method in (left,right):
  a,b,t,_=cal[method,seed,'TWA'];p=np.concatenate([getpred(cohort,method,seed,k)[:,0] for k in CONDS]);yy=np.tile(y[:,0],4);vals.append(weighted_metrics(yy,p,mult,t,a,b)[metric])
 return vals[0]-vals[1]
def mean_sd(rows,keys):
 df=pd.DataFrame(rows);metrics=['AP_raw','AUROC_raw','Brier_raw','ECE15_raw','AP_calibrated','AUROC_calibrated','Brier_calibrated','ECE15_calibrated','sensitivity_raw','specificity_raw','predicted_positive_fraction_raw','FP_per_1000_raw','FN_per_1000_raw','sensitivity_calibrated','specificity_calibrated','predicted_positive_fraction_calibrated','FP_per_1000_calibrated','FN_per_1000_calibrated']
 out=[]
 for group,g in df.groupby(keys,dropna=False):
  if not isinstance(group,tuple):group=(group,)
  r=dict(zip(keys,group))
  for m in metrics:r[m+'_mean']=float(g[m].mean());r[m+'_sd']=float(g[m].std(ddof=1)) if len(g)>1 else float('nan')
  out.append(r)
 return out
def main():
 c.configure_determinism()
 verify_locks();calrows=load_cal();cal={(r['method'],int(r['seed']),r['endpoint']):(float(r['slope_a']),float(r['intercept_b']),float(r['raw_q90']),float(r['mapped_calibrated_threshold'])) for r in csv.DictReader((OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv').open(encoding='utf8'))}
 parts,mem=c.records();# Open held-out cohorts only after immutable locks.
 nraw=c.raw_array(parts['test']);ny=c.labels(parts['test']);nids=np.asarray([r['record_id'] for r in parts['test']]);detn=np.load(OUT/'detector/Ningbo_test.npz',allow_pickle=False);det_rows=json.loads((OUT/'DETECTOR_TEST_BINDINGS.json').read_text(encoding='utf8'))
 if det_rows.get('status')!='PASS' or not np.array_equal(detn['record_ids'].astype(str),nids.astype(str)):raise RuntimeError('Ningbo detector test order/replay gate failed')
 raw500_n=raw500(parts['test']);infer_cohort('Ningbo_test',parts['test'],ny,nids,nraw,detn['probabilities'],list(detn['conditions'].astype(str)),sha(OUT/'detector/Ningbo_test.npz'),raw500_n)
 # HEEDB is historical sensitivity control; all 10,000 patients and source order are frozen.
 import pandas as pd
 from outputs.mixed_fault_calibration_v1 import run_analysis as base
 private=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1';rawh=np.load(private/'signals_10000_float32.npy',mmap_mode='r',allow_pickle=False)
 hy,pids=base.heedb_labels();sel=pd.read_csv(private/'selection_initial.csv',dtype=str,keep_default_na=False);hids=sel['FileName'].astype(str).to_numpy()
 if rawh.shape!=(10000,12,5000) or len(np.unique(pids))!=10000 or len(np.unique(hids))!=10000:raise RuntimeError('HEEDB cohort integrity failure')
 deth,dhsha,expected_order=load_det_heedb();
 if list(deth['conditions'].astype(str))!=list(detector.CONDITIONS):
  ix=[list(deth['conditions'].astype(str)).index(x) for x in detector.CONDITIONS];dprobs=deth['probabilities'][ix]
 else:dprobs=deth['probabilities']
 # Protocol's source order and current selection list must match the hash from recovery.
 if str(deth['source_record_order_sha256'].item())!=expected_order:raise RuntimeError('HEEDB detector/source order protocol mismatch')
 infer_cohort('HEEDB',[],hy,hids,rawh,dprobs,list(detector.CONDITIONS),dhsha)
 allc=[];allm=[];allboot=[];drawdict={}
 for cohort,y,idxfile in [('Ningbo_test',ny,N/'bootstrap_indices.npy'),('HEEDB',hy,ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy')]:
  cr,mr=metric_tables(cohort,y,cal);allc+=cr;allm+=mr;idx=np.load(idxfile,mmap_mode='r',allow_pickle=False)
  if idx.shape!=(1000,len(y)):raise RuntimeError('bootstrap index geometry mismatch '+cohort)
  br,dr=bootstrap(cohort,y,cal,idx);allboot+=br
  for k,v in dr.items():drawdict[cohort+'|'+k]=v
 write_csv(OUT/'CONDITION_METRICS_BY_SEED.csv',allc);write_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv',allm);write_csv(OUT/'MIXTURE_METRICS_MEAN_SD.csv',mean_sd(allm,['cohort','method','endpoint','scenario','q_fault']));write_csv(OUT/'CONDITION_METRICS_MEAN_SD.csv',mean_sd(allc,['cohort','method','endpoint','condition']));write_csv(OUT/'TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv',allboot)
 for cohort in ('Ningbo_test','HEEDB'):
  with np.load(OUT/f'{cohort}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:drawdict.update({cohort+'|'+k:z[k] for k in z.files})
 atomic_npz(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz',**drawdict)
 # Historical comparison is a direct same-q/same-cohort/metric comparison; no claim that the delta isolates calibration bias.
 old=pd.read_csv(ROOT/'outputs/three_swap_population_mixture_recovery_v2/MIXTURE_METRICS_BY_SEED.csv');new=pd.DataFrame(allm);omap={'ordinary':'ordinary_resnet','physics_aware_p050':'physics_aware_m2','combined':'combined','Huang_full_pipeline':'huang_pipeline'};old['method_norm']=old.method.map({v:k for k,v in omap.items()});new['method_norm']=new.method
 comp=[]
 for _,r in new.iterrows():
  match=old[(old.source==r.cohort.replace('_test',''))&(old.method_norm==r.method_norm)&(old.seed==r.seed)&(old.endpoint==r.endpoint)&(old.scenario==r.scenario)&(np.isclose(old.q_fault,r.q_fault))&(old.score_mode=='raw')]
  if len(match):
   o=match.iloc[0];comp.append({'cohort':r.cohort,'method':r.method,'seed':r.seed,'endpoint':r.endpoint,'scenario':r.scenario,'q_fault':r.q_fault,'metric':'AP_raw','old':o.AP,'new':r.AP_raw,'delta_new_minus_old':r.AP_raw-o.AP,'cause_interpretable_as_calibration_only':False})
   for col,oldcol in [('Brier_raw','Brier'),('FP_per_1000_raw','FP_per_1000')]:comp.append({'cohort':r.cohort,'method':r.method,'seed':r.seed,'endpoint':r.endpoint,'scenario':r.scenario,'q_fault':r.q_fault,'metric':col,'old':o[oldcol],'new':r[col],'delta_new_minus_old':r[col]-o[oldcol],'cause_interpretable_as_calibration_only':False})
 write_csv(OUT/'HISTORICAL_COMPARISON.csv',comp)
 atomic_json(OUT/'STATUS.json',{'status':'EVALUATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION','all_fits':9,'calibration_threshold_locks':36,'new_prediction_files':len(list((OUT/'predictions').rglob('*.npz'))),'cohorts':['Ningbo_test','HEEDB_historical_sensitivity'],'completed_utc':c.now()})
 print(json.dumps({'status':'EVALUATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION','mixture_rows':len(allm),'condition_rows':len(allc),'bootstrap_contrasts':len(allboot)}))
if __name__=='__main__':main()
