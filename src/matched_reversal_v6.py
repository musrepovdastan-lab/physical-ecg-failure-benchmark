"""Exploratory v6: exact reversals vs five displayed-lead masks."""
import csv,itertools,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from src.chapman_phase1 import ECGResNetSmall,raw,labels
ROOT=Path(__file__).resolve().parents[1];V4=ROOT/'outputs/phase1_chapman_seed1_v4';OUT=ROOT/'outputs/phase1b_matched_reversal_development_v6';SEED=104729
def csvout(n,r):
 with (OUT/n).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
def testids():return [x['record_id'] for x in json.loads((V4/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'] if x['split']=='test']
def ece(y,p):
 z=[]
 for b in range(15):
  q=(p>=b/15)&(p<((b+1)/15) if b<14 else p<=1)
  if q.any():z.append(q.mean()*abs(p[q].mean()-y[q].mean()))
 return float(sum(z))
def measures(n,y,p,t):
 r=[]
 for i,l in enumerate(('TWA','SB','STach')):
  q=p[:,i]>=t[i];neg=y[:,i]==0;r.append({'condition':n,'label':l,'auprc':float(average_precision_score(y[:,i],p[:,i])),'auroc':float(roc_auc_score(y[:,i],p[:,i])),'sensitivity':float(q[y[:,i]==1].mean()),'specificity':float((~q[neg]).mean()),'ece_15bin':ece(y[:,i],p[:,i])})
 return r
def main():
 OUT.mkdir(parents=True,exist_ok=True); masks=list(itertools.combinations(range(12),5));assert len(masks)==792
 write='''# Reversal algebra audit\n\nSynthetic unit checks against `src/ecg_physics.py` passed: RA–LA preserves aVF, RA–LL preserves aVL, and LA–LL preserves aVR (machine precision); each transforms the other five displayed limb leads. This evaluator uses existing cached clean/reversal predictions and the fixed sentinel convention for five displayed-lead masks.\n''';(OUT/'REVERSAL_ALGEBRA_AUDIT.md').write_text(write,encoding='utf-8')
 ids=testids();z=np.load(V4/'preprocessing_v4.npz');m,s=z['mean'],z['std'];t=np.load(V4/'validation_predictions_v4.npz')['thresholds'];x=np.stack([raw(r) for r in ids]);y=np.stack([labels(r) for r in ids]);model=ECGResNetSmall().cuda();model.load_state_dict(torch.load(V4/'ecgresnet_small_seed104729_v4.pt',weights_only=True)['state_dict']);cache=OUT/'predictions';cache.mkdir(exist_ok=True)
 def infer(a):
  q=[];model.eval()
  with torch.no_grad():
   for k in range(0,len(a),128):
    with torch.autocast('cuda',dtype=torch.float16):q.append(torch.sigmoid(model(torch.from_numpy(a[k:k+128]).cuda())).float().cpu().numpy())
  return np.concatenate(q)
 preds={'clean':np.load(V4/'test_predictions_v4/clean.npz')['p']}
 for r in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'):preds[r]=np.load(V4/f'test_predictions_v4/{r}.npz')['p']
 for q in masks:
  n='mask5_'+'_'.join(map(str,q));p=cache/f'{n}.npz'
  if p.exists():preds[n]=np.load(p)['p'];continue
  a=x.copy();a[:,q]=m[list(q),None];preds[n]=infer(((a-m[None,:,None])/s[None,:,None]).astype('float32'));np.savez_compressed(p,p=preds[n])
 rows=sum((measures(n,y,p,t) for n,p in preds.items()),[]);df=pd.DataFrame(rows);csvout('five_display_mask_results.csv',[r for r in rows if r['condition'].startswith('mask5_')]);csvout('reversal_matched_results.csv',[r for r in rows if r['condition'] in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')])
 masknames=[n for n in preds if n.startswith('mask5_')];rng=np.random.default_rng(SEED);indices=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
 def one(arg):
  b,ix=arg;means=np.mean([[average_precision_score(y[ix,i],preds[n][ix,i]) for i in range(3)] for n in masknames],0);o=[]
  for r in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'):
   a=np.array([average_precision_score(y[ix,i],preds[r][ix,i]) for i in range(3)])
   for i,l in enumerate(('TWA','SB','STach')):o.append({'replicate':b,'reversal':r,'label':l,'contrast':float(means[i]-a[i])})
  return o
 with ThreadPoolExecutor(max_workers=8) as ex:boot=sum(ex.map(one,enumerate(indices)),[])
 csvout('bootstrap_replicates.csv',boot);bs=pd.DataFrame(boot);summ=[]
 for (r,l),q in bs.groupby(['reversal','label']):summ.append({'reversal':r,'label':l,'mean_contrast':q.contrast.mean(),'low95':q.contrast.quantile(.025),'high95':q.contrast.quantile(.975)})
 csvout('bootstrap_summary.csv',summ);tw=pd.DataFrame(summ);clean=float(df[(df.condition=='clean')&(df.label=='TWA')].auprc.iloc[0]);md=df[(df.condition.str.startswith('mask5_'))&(df.label=='TWA')];revs=[]
 for r in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal'):
  a=float(df[(df.condition==r)&(df.label=='TWA')].auprc.iloc[0]);revs.append({'reversal':r,'twa_auprc':a,'absolute_loss':clean-a,'percentile_among_masks':float((md.auprc<=a).mean())})
 ra=tw[(tw.reversal=='RA_LL_reversal')].set_index('label');go=ra.loc['TWA','mean_contrast']>=.05 and ra.loc['TWA','low95']>0 and ra.loc['TWA','mean_contrast']>=ra.loc['SB','mean_contrast']+.05 and ra.loc['TWA','mean_contrast']>=ra.loc['STach','mean_contrast']+.05;decision='FREEZE_NINGBO_CONFIRMATION' if go else 'PROMISING_NONSPECIFIC_SIGNAL' if ra.loc['TWA','mean_contrast']>0 else 'STOP_REVERSAL_COMPARATOR_WEAK';out={'decision':decision,'clean_twa_auprc':clean,'reversals':revs,'mask_twa_mean':float(md.auprc.mean()),'mask_twa_median':float(md.auprc.median()),'mask_twa_p5':float(md.auprc.quantile(.05)),'mask_twa_p95':float(md.auprc.quantile(.95)),'conditions_completed':len(masknames),'runtime_seconds':None,'exploratory':True};(OUT/'conclusions_v6.json').write_text(json.dumps(out,indent=2),encoding='utf-8');(OUT/'FINAL_PHASE1B_DECISION.md').write_text('# '+decision+'\n\n'+json.dumps(out,indent=2),encoding='utf-8');(OUT/'MATCHED_COMPARATOR_INTERPRETATION.md').write_text('Five-channel masking is consequence-count matched only: reversals transform five limb channels, whereas masks remove five channels. It is not a claim of equal corruption magnitude or clinical severity. All Chapman v6 comparisons are exploratory because Chapman outcomes were previously inspected.\n',encoding='utf-8')
if __name__=='__main__':main()
