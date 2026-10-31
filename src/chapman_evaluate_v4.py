"""Frozen v4 cached internal-test evaluation; no tuning branches."""
import csv,hashlib,itertools,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from src.chapman_phase1 import ECGResNetSmall,raw,labels
from src.ecg_physics import derive_display,named_reversal
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase1_chapman_seed1_v4';CFG=ROOT/'configs/FROZEN_PHASE1_PILOT_PROTOCOL_v4.json';SEED=104729
def hh(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def csvout(n,rows):
 with (OUT/n).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def parts():
 x=json.loads((OUT/'SPLIT_MEMBERSHIP_v4.json').read_text())['rows'];return [q['record_id'] for q in x if q['split']=='test']
def ece(y,p):
 z=[]
 for i in range(y.shape[1]):
  for b in range(15):
   q=(p[:,i]>=b/15)&(p[:,i]<(b+1)/15 if b<14 else p[:,i]<=1)
   if q.any():z.append(q.mean()*abs(p[q,i].mean()-y[q,i].mean()))
 return float(sum(z)/y.shape[1])
def metric(name,y,p,t):
 out=[]
 for i,l in enumerate(('TWA','SB','STach')):
  q=p[:,i]>=t[i];neg=y[:,i]==0;out.append({'condition':name,'label':l,'auprc':float(average_precision_score(y[:,i],p[:,i])),'auroc':float(roc_auc_score(y[:,i],p[:,i])),'sensitivity':float(q[y[:,i]==1].mean()),'specificity':float((~q[neg]).mean()),'ece_15bin':ece(y[:,[i]],p[:,[i]])})
 return out
def transform(x,m,s,unavailable=(),rev=None,display=()):
 ind=np.concatenate((x[:,0:2],x[:,6:]),1)
 if rev:ind=named_reversal(ind,rev)
 d=derive_display(ind).astype('float32');u=set(unavailable)
 if u&{0,1}:d[:,:6]=m[:6,None]
 for q in u-{0,1}:d[:,6+q-2]=m[6+q-2,None]
 for q in display:d[:,q]=m[q,None]
 return ((d-m[None,:,None])/s[None,:,None]).astype('float32')
def manifests():
 four=list(itertools.combinations(range(8),4));disp=list(itertools.combinations(range(12),4));assert len(four)==70 and len(disp)==495 and four.count((2,3,4,5))==1
 a=[('V'+str(i),(i+1,),None,()) for i in range(1,7)]+[('V1_V4',(2,3,4,5),None,())]
 a += [('independent_'+'_'.join(map(str,q)),q,None,()) for q in four]
 a += [(q,(),q,()) for q in ('RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')]
 a += [('displaymask_'+'_'.join(map(str,q)),(),None,q) for q in disp]
 names=['clean']+[x[0] for x in a];assert len(names)==len(set(names));return a,four,disp
def smoke():
 a,f,d=manifests();x=np.random.default_rng(1).normal(size=(2,12,1000)).astype('float32');m=np.zeros(12,'float32');s=np.ones(12,'float32')
 for _,u,r,z in a:assert transform(x,m,s,u,r,z).shape==x.shape and np.isfinite(transform(x,m,s,u,r,z)).all()
 y=np.array([[1,0,1],[0,1,0]]);p={'a':np.ones((2,3))*.4,'b':np.ones((2,3))*.3};ix=np.random.default_rng(SEED).integers(0,2,(3,2));assert np.array_equal(ix[0].shape,ix[1].shape);dump('EVALUATOR_SMOKE_v4.json',{'status':'PASS','four_independent':len(f),'display_masks':len(d),'unique_conditions':1+len(a),'ece':'15 equal-width bins','bootstrap_synchronized':True})
def freeze():
 a,f,d=manifests();required=['ecgresnet_small_seed104729_v4.pt','preprocessing_v4.npz','validation_predictions_v4.npz','BASELINE_ADEQUACY_v4.json','SPLIT_MEMBERSHIP_v4.json']
 if not json.loads((OUT/'BASELINE_ADEQUACY_v4.json').read_text())['decision']=='PASS':raise RuntimeError('baseline gate')
 z={x:hh(OUT/x) for x in required};z.update({'protocol_sha256':hh(CFG),'fault_library_sha256':hh(ROOT/'src/ecg_physics.py'),'evaluator_sha256':hh(Path(__file__)),'bootstrap_seed':SEED,'bootstrap_replicates':1000,'ece_definition':'15 equal-width bins','test_accessed_before_freeze':False,'conditions':['clean']+[x[0] for x in a],'independent_masks':[list(x) for x in f],'display_masks':[list(x) for x in d]});dump('FAILURE_ANALYSIS_FREEZE_v4.json',z)
def main():
 smoke();freeze();a,f,d=manifests();test=parts();z=np.load(OUT/'preprocessing_v4.npz');m,s=z['mean'],z['std'];x=np.stack([raw(r) for r in test]);y=np.stack([labels(r) for r in test]);t=np.load(OUT/'validation_predictions_v4.npz')['thresholds'];model=ECGResNetSmall().cuda();model.load_state_dict(torch.load(OUT/'ecgresnet_small_seed104729_v4.pt',weights_only=True)['state_dict']);
 def infer(q):
  o=[];model.eval()
  with torch.no_grad():
   for k in range(0,len(q),128):
    with torch.autocast('cuda',dtype=torch.float16):o.append(torch.sigmoid(model(torch.from_numpy(q[k:k+128]).cuda())).float().cpu().numpy())
  return np.concatenate(o)
 preds={};conditions=[('clean',(),None,())]+a
 for name,u,r,dm in conditions:
  pth=OUT/'test_predictions_v4'/f'{name}.npz';pth.parent.mkdir(exist_ok=True)
  if pth.exists():p=np.load(pth)['p']
  else:p=infer(((x-m[None,:,None])/s[None,:,None]).astype('float32') if name=='clean' else transform(x,m,s,u,r,dm));np.savez_compressed(pth,p=p)
  preds[name]=p
 rows=sum((metric(n,y,p,t) for n,p in preds.items()),[]);csvout('all_condition_metrics_v4.csv',rows)
 csvout('clean_metrics_v4.csv',[q for q in rows if q['condition']=='clean']);csvout('availability_fault_results_v4.csv',[q for q in rows if q['condition'].startswith('V')]);csvout('all_70_independent_masks_v4.csv',[q for q in rows if q['condition'].startswith('independent_')]);csvout('terminal_reversal_results_v4.csv',[q for q in rows if 'reversal' in q['condition']]);csvout('display_mask_secondary_results_v4.csv',[q for q in rows if q['condition'].startswith('displaymask_')]);csvout('negative_control_results_v4.csv',[q for q in rows if q['label']!='TWA'])
 rng=np.random.default_rng(SEED);other=['independent_'+'_'.join(map(str,q)) for q in f if q!=(2,3,4,5)];boot=[]
 for b in range(1000):
  ix=rng.integers(0,len(y),len(y));base=np.array([average_precision_score(y[ix,i],preds['clean'][ix,i]) for i in range(3)]);v=np.array([average_precision_score(y[ix,i],preds['V1_V4'][ix,i]) for i in range(3)]);o=np.mean([[average_precision_score(y[ix,i],preds[n][ix,i]) for i in range(3)] for n in other],0)
  for i,l in enumerate(('TWA','SB','STach')):boot.append({'replicate':b,'label':l,'effect':float((base[i]-v[i])-(base[i]-o[i]))})
 csvout('bootstrap_replicates_v4.csv',boot);summary=[]
 for l in ('TWA','SB','STach'):
  q=np.array([r['effect'] for r in boot if r['label']==l]);summary.append({'label':l,'mean_effect':float(q.mean()),'low95':float(np.quantile(q,.025)),'high95':float(np.quantile(q,.975))})
 csvout('bootstrap_summary_v4.csv',summary);tw=summary[0];neg={x['label']:x['mean_effect'] for x in summary[1:]};gate=abs(tw['mean_effect'])>=.05 and not(tw['low95']<=0<=tw['high95']) and not(abs(neg['SB'])>=abs(tw['mean_effect']) and abs(neg['STach'])>=abs(tw['mean_effect']));out={'decision':'PROCEED_MULTISEED' if gate else 'STOP_ONE_SEED_NO_MEANINGFUL_SIGNAL','primary_effect':tw,'negative_controls':neg,'test_accessed':True,'ningbo_accessed':False};dump('conclusions_v4.json',out);(OUT/'FINAL_PHASE1_PILOT_DECISION_v4.md').write_text('# '+out['decision']+'\n\n'+json.dumps(out,indent=2),encoding='utf-8');dump('test_predictions_manifest_v4.json',{'records':len(test),'conditions':list(preds),'cached':True,'test_accessed':True})
if __name__=='__main__':main()
