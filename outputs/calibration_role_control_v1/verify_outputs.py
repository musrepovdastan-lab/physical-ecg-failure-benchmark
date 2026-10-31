from pathlib import Path
import sys,json,csv,hashlib,math
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT));import evaluate as e
from functools import lru_cache
COHORTS=('Ningbo_test','HEEDB');EPS=('TWA','SB','STach')
def load_cal():
 rows=list(csv.DictReader((OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv').open(encoding='utf8')))
 return {(r['method'],int(r['seed']),r['endpoint']):(float(r['slope_a']),float(r['intercept_b']),float(r['raw_q90']),float(r['mapped_calibrated_threshold'])) for r in rows}
@lru_cache(maxsize=128)
def pred(cohort,method,seed,cond):
 with np.load(OUT/f'predictions/{cohort}/{method}/seed_{seed}/{cond}.npz',allow_pickle=False) as z:return z['p'].astype(np.float64)
def ece(y,p,w):
 y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float);total=w.sum();out=0.;edges=np.linspace(0,1,16)
 for j in range(15):
  ix=(p>=edges[j])&((p<edges[j+1]) if j<14 else (p<=edges[j+1]))&(w>0)
  if ix.any():out+=w[ix].sum()/total*abs(np.average(y[ix],weights=w[ix])-np.average(p[ix],weights=w[ix]))
 return float(out)
def metric_ind(y,p,w,t,a,b):
 y=np.asarray(y,np.uint8);p=np.asarray(p,float);w=np.asarray(w,float);cal=e.platt(p,a,b);rd=p>=t;ct=float(e.platt(np.asarray([t]),a,b)[0]);cd=cal>=ct;pos=y==1;neg=~pos;tot=w.sum()
 return {'AP_raw':float(average_precision_score(y,p,sample_weight=w)),'AUROC_raw':float(roc_auc_score(y,p,sample_weight=w)),'Brier_raw':float(np.sum(w*(p-y)**2)/tot),'ECE15_raw':ece(y,p,w),
 'AP_calibrated':float(average_precision_score(y,cal,sample_weight=w)),'AUROC_calibrated':float(roc_auc_score(y,cal,sample_weight=w)),'Brier_calibrated':float(np.sum(w*(cal-y)**2)/tot),'ECE15_calibrated':ece(y,cal,w),
 'sensitivity_raw':float(np.sum(w*rd*pos)/np.sum(w*pos)),'specificity_raw':float(np.sum(w*(~rd)*neg)/np.sum(w*neg)),'predicted_positive_fraction_raw':float(np.sum(w*rd)/tot),'FP_per_1000_raw':float(1000*np.sum(w*rd*neg)/tot),'FN_per_1000_raw':float(1000*np.sum(w*(~rd)*pos)/tot),
 'sensitivity_calibrated':float(np.sum(w*cd*pos)/np.sum(w*pos)),'specificity_calibrated':float(np.sum(w*(~cd)*neg)/np.sum(w*neg)),'predicted_positive_fraction_calibrated':float(np.sum(w*cd)/tot),'FP_per_1000_calibrated':float(1000*np.sum(w*cd*neg)/tot),'FN_per_1000_calibrated':float(1000*np.sum(w*(~cd)*pos)/tot),'raw_vs_calibrated_decision_mismatch_weight':float(np.sum(w*(rd!=cd))/tot)}
def cohort_labels(cohort):
 if cohort=='Ningbo_test':
  pp,_=e.c.records();return e.c.labels(pp['test'])
 from outputs.mixed_fault_calibration_v1 import run_analysis as base
 return base.heedb_labels()[0]
def verify_predictions_and_metrics(cal):
 cr=pd.read_csv(OUT/'CONDITION_METRICS_BY_SEED.csv');mr=pd.read_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv');checks=[];pred_bad=[]
 for cohort in COHORTS:
  y=cohort_labels(cohort)
  for _,r in cr[cr.cohort==cohort].iterrows():
   cond=r.condition;p=pred(cohort,r.method,int(r.seed),cond)[:,EPS.index(r.endpoint)];a,b,t,_=cal[r.method,int(r.seed),r.endpoint];z=metric_ind(y[:,EPS.index(r.endpoint)],p,np.ones(len(y)),t,a,b)
   for k,v in z.items():checks.append({'table':'condition','cohort':cohort,'method':r.method,'seed':int(r.seed),'endpoint':r.endpoint,'condition':cond,'metric':k,'observed':float(r[k]),'recalculated':v,'abs_delta':abs(float(r[k])-v)})
  for _,r in mr[mr.cohort==cohort].iterrows():
   ei=EPS.index(r.endpoint);q=float(r.q_fault);weights=np.asarray([1-q,0,q,0] if r.scenario=='RA_LL_only' else [1-q,q/3,q/3,q/3],float);p=np.concatenate([pred(cohort,r.method,int(r.seed),cnd)[:,ei] for cnd in e.CONDS]);yy=np.tile(y[:,ei],4);ww=np.concatenate([np.full(len(y),w) for w in weights]);a,b,t,_=cal[r.method,int(r.seed),r.endpoint];z=metric_ind(yy,p,ww,t,a,b)
   for k,v in z.items():checks.append({'table':'mixture','cohort':cohort,'method':r.method,'seed':int(r.seed),'endpoint':r.endpoint,'scenario':r.scenario,'q_fault':q,'metric':k,'observed':float(r[k]),'recalculated':v,'abs_delta':abs(float(r[k])-v)})
   if abs(float(r.weighted_views_per_unit)-weights.sum())>1e-12:pred_bad.append('view weight')
 maxdelta=max(x['abs_delta'] for x in checks)
 pd.DataFrame(checks).to_csv(OUT/'INDEPENDENT_METRICS_RECALC.csv',index=False)
 return maxdelta,len(checks),pred_bad
def verify_locks_and_roles(cal):
 membership=list(csv.DictReader((ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv').open(encoding='utf-8-sig')));roles={x:set() for x in ('model_selection','calibration_A','threshold_B')};wave={x:set() for x in roles}
 for r in membership:roles[r['role']].add(r['record_id']);wave[r['role']].add(r['waveform_sha256'])
 roleok=(len(roles['model_selection'])==1744 and len(roles['calibration_A'])==872 and len(roles['threshold_B'])==872 and all(not roles[a]&roles[b] for a,b in [('model_selection','calibration_A'),('model_selection','threshold_B'),('calibration_A','threshold_B')]) and all(not wave[a]&wave[b] for a,b in [('model_selection','calibration_A'),('model_selection','threshold_B'),('calibration_A','threshold_B')]))
 locks=json.loads((OUT/'ALL_MODELS_LOCK.json').read_text(encoding='utf8'));passed=True;best=[]
 for x in locks['fits']:
  d=OUT/'fits'/x['method']/f"seed_{x['seed']}";h=json.loads((d/'history.json').read_text(encoding='utf8'))['rows'];ck=__import__('torch').load(d/'checkpoint.pt',weights_only=True,map_location='cpu');b=max(h,key=lambda r:r['selection_twa_auprc']);best.append({'method':x['method'],'seed':x['seed'],'checkpoint_epoch':ck['epoch'],'history_best_epoch':b['epoch'],'score_delta':abs(ck['selection_twa_auprc']-b['selection_twa_auprc'])});passed &= ck['epoch']==b['epoch'] and all(v['selection_role']=='model_selection' for v in h)
 # Recompute q90 from clean negative B predictions and validate A-only exact Platt slope mapping.
 for (method,seed,ep),(a,b,q,mapped) in cal.items():
  with np.load(OUT/f'scores/threshold_B/{method}/seed_{seed}.npz',allow_pickle=False) as z:pp=z['p'];yy=z['y']
  ei=EPS.index(ep)
  q2=float(np.quantile(pp[yy[:,ei]==0,ei],.9,method='linear'));m2=float(e.platt(np.asarray([q]),a,b)[0]);passed &= abs(q2-q)<1e-14 and abs(m2-mapped)<1e-14 and a>0
 return roleok,bool(passed),best
def verify_bootstrap():
 rows=pd.read_csv(OUT/'TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv');checks=[];draw_deltas=[]
 for _,r in rows.iterrows():
  c0=r.cohort;key=f"{c0}|{r.scenario}|q{r.q_fault:g}|{r.left}-{r.right}|{r.metric}"
  with np.load(OUT/f'{c0}_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as z:v=z[key]
  d={'mean':float(np.mean(v)),'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975))};checks.append({'key':key,'mean_delta':abs(d['mean']-r.bootstrap_mean),'low_delta':abs(d['low']-r.ci95_low),'high_delta':abs(d['high']-r.ci95_high),'n':len(v)})
 max_ci=max(max(x['mean_delta'],x['low_delta'],x['high_delta']) for x in checks)
 # Actual synchronized draws are independently evaluated with sklearn at indices 0, 417, and 999.
 draw_checks=pd.concat([pd.read_csv(OUT/f'{cohort}_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv') for cohort in COHORTS],ignore_index=True);max_tie=float(draw_checks.absolute_delta.max())
 if not (max_ci<1e-12 and max_tie<1e-12):raise RuntimeError(f'bootstrap verification failed {max_ci}/{max_tie}')
 pd.DataFrame(checks).to_csv(OUT/'BOOTSTRAP_SUMMARY_RECALC.csv',index=False)
 return max_ci,max_tie,len(checks),len(draw_checks)
def main():
 cal=load_cal();maxd,nchecks,bad=verify_predictions_and_metrics(cal);roleok,modelok,best=verify_locks_and_roles(cal);maxci,maxtie,nci,ndraw=verify_bootstrap()
 report={'status':'PASS' if maxd<1e-10 and not bad and roleok and modelok and maxci<1e-12 and maxtie<1e-12 else 'FAIL','independent_metric_rows_recalculated':nchecks,'max_metric_abs_delta':maxd,'metric_anomalies':bad,'role_membership_audit_pass':roleok,'model_best_epoch_history_and_selection_only_pass':modelok,'best_checkpoint_replays':best,'bootstrap_summary_rows_recalculated':nci,'bootstrap_summary_max_abs_delta':maxci,'actual_sklearn_draw_checks':ndraw,'actual_sklearn_draw_max_abs_delta':maxtie,'patient_independence':'Ningbo patient IDs unavailable; HEEDB uses one unique patient per row','reserved_test_evaluated':False,'completed_utc':'2026-10-20'}
 (OUT/'VERIFICATION_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
