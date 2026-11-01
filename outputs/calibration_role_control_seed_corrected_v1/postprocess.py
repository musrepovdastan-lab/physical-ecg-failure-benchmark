from pathlib import Path
import sys,json,csv,hashlib,os,time
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT));import control as c
N=c.N
from src.chapman_phase1 import ECGResNetSmall
from src.heedb_huang_adapter_v1 import corrected_batch,diagnostic_input,electrical_reversal_batch
from src import huang_detector_infer_v2 as detector
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(dest,obj):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8');os.replace(tmp,dest)
def atomic_npz(dest,**arrays):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.stem+'.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,dest)
def verify():
 prot=json.loads((OUT/'FROZEN_EXECUTION_PROTOCOL.json').read_text(encoding='utf8'))
 if (OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').read_text().strip()!=sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json'):raise RuntimeError('protocol file hash mismatch')
 if sha(OUT/'control.py')!=prot['runner_sha256']['control.py']:raise RuntimeError('frozen training runner changed')
 lock=json.loads((OUT/'ALL_MODELS_LOCK.json').read_text(encoding='utf8'))
 if lock.get('decision')!='PASS' or len(lock['fits'])!=9:raise RuntimeError('all nine models not locked')
 for row in lock['fits']:
  d=OUT/'fits'/row['method']/f"seed_{row['seed']}"
  if sha(d/'checkpoint.pt')!=row['checkpoint_sha256'] or sha(d/'history.json')!=row['history_sha256'] or sha(d/'preprocessing.npz')!=row['preprocessing_sha256']:raise RuntimeError('locked artifact hash mismatch')
  ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');hist=json.loads((d/'history.json').read_text())['rows'];best=max(hist,key=lambda x:x['selection_twa_auprc'])
  if ck['epoch']!=best['epoch'] or ck['epoch']!=row['best_epoch'] or ck['method']!=row['method'] or ck['seed']!=row['seed']:raise RuntimeError('best checkpoint does not match selection history')
  if any(x['selection_role']!='model_selection' for x in hist):raise RuntimeError('non-selection training score detected')
 return prot,lock
def infer_units(method,seed,units,role):
 d=OUT/'fits'/method/f'seed_{seed}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict'])
 with np.load(d/'preprocessing.npz',allow_pickle=False) as z:mean,std=z['mean'].astype(np.float32),z['std'].astype(np.float32)
 x=c.raw_array(units);y=c.labels(units);p=c.infer(model,x,mean,std);dest=OUT/f'scores/{role}/{method}/seed_{seed}.npz'
 atomic_npz(dest,p=p,y=y,record_ids=np.asarray([r['record_id'] for r in units]),role=np.asarray(role),condition=np.asarray('clean'),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),membership_sha256=np.asarray(sha(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv')))
 del model;torch.cuda.empty_cache();return p,y,dest
def raw500(units):return np.stack([detector.load_full_display(r) for r in units]).astype(np.float32)
def infer_huang(model,x,mean,std):
 z=np.stack([diagnostic_input(a,mean,std) for a in x]).astype(np.float32);out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(z),128):
   with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(z[i:i+128]).cuda(non_blocking=True))).float().cpu().numpy())
 return np.concatenate(out).astype(np.float32)
def main():
 c.configure_determinism()
 prot,lock=verify();parts,mem=c.records();membership=OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv'
 ytrain=c.labels(parts['train']);pos_weight=np.minimum((len(ytrain)-ytrain.sum(0))/np.maximum(ytrain.sum(0),1),20).astype(float)
 stats_files=[ROOT/'outputs/ningbo_confirmation_v2'/f'seed_{s}'/'preprocessing.npz' for s in c.SEEDS]+[ROOT/'outputs/mitigation_m2_ningbo_v1'/f'seed_{s}'/'preprocessing.npz' for s in c.SEEDS]
 if len({sha(x) for x in stats_files})!=1:raise RuntimeError('train-only normalization provenance mismatch')
 for row in lock['fits']:
  d=OUT/'fits'/row['method']/f"seed_{row['seed']}";cfg={'method':row['method'],'seed':row['seed'],'architecture':'ECGResNet-small','loss':'weighted multilabel BCE','pos_weight_train_only':pos_weight.tolist(),'pos_weight_cap':20,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'batch_size':64,'max_epochs':30,'mixed_precision':'CUDA autocast float16 + GradScaler','selection_role':'model_selection','selection_n':1744,'selection_metric':'TWA AUPRC','patience':5,'best_epoch':row['best_epoch'],'recipe':{'ordinary':'clean100%','physics_aware_p050':'clean50% plus one uniform RA_LA/RA_LL/LA_LL swap50%','combined':'clean50% plus mask5 25% plus one uniform RA_LA/RA_LL/LA_LL swap25%; exclusive'}[row['method']],'membership_file_sha256':sha(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv'),'train_split_sha256':sha(N/'SPLIT_MEMBERSHIP.json'),'normalization_sources_sha256':sha(stats_files[0]),'normalization_source_files':[str(x.relative_to(ROOT)).replace('\\','/') for x in stats_files],'checkpoint_sha256':sha(d/'checkpoint.pt'),'runner_sha256':prot['runner_sha256']['control.py']};cfg['fit_config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest();atomic_json(d/'fit_config.json',cfg)
  atomic_json(d/'preprocessing_provenance.json',{'source':'byte-identical Ningbo baseline and p050 train-derived statistics','source_sha256':sha(stats_files[0]),'source_files':[str(x.relative_to(ROOT)).replace('\\','/') for x in stats_files],'validation_or_test_used':False,'new_fit_preprocessing_sha256':sha(d/'preprocessing.npz')})
 # Selection-only replay: independent of A/B and bound to the locked best weights.
 selx=c.raw_array(parts['model_selection']);sely=c.labels(parts['model_selection']);replays=[]
 for row in lock['fits']:
  method,seed=row['method'],row['seed'];d=OUT/'fits'/method/f'seed_{seed}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict']);mean,std=np.load(d/'preprocessing.npz')['mean'],np.load(d/'preprocessing.npz')['std'];p=c.infer(model,selx,mean,std);ap=float(average_precision_score(sely[:,0],p[:,0]));auc=float(roc_auc_score(sely[:,0],p[:,0]));atomic_npz(OUT/f'scores/model_selection/{method}/seed_{seed}.npz',p=p,y=sely,record_ids=np.asarray([r['record_id'] for r in parts['model_selection']]),role=np.asarray('model_selection'),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),membership_sha256=np.asarray(sha(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv')));replays.append({'method':method,'seed':seed,'role':'model_selection','n':len(sely),'checkpoint_sha256':sha(d/'checkpoint.pt'),'selected_epoch':ck['epoch'],'history_epoch':row['history_best_epoch'],'stored_ap':row['best_selection_twa_auprc'],'replay_ap':ap,'stored_auc':row['best_selection_twa_auroc'],'replay_auc':auc,'max_abs_delta_ap':abs(ap-row['best_selection_twa_auprc']),'status':'PASS' if abs(ap-row['best_selection_twa_auprc'])<1e-7 and abs(auc-row['best_selection_twa_auroc'])<1e-7 else 'FAIL'});del model;torch.cuda.empty_cache()
 if any(r['status']!='PASS' for r in replays):raise RuntimeError('independent best-checkpoint replay failed')
 atomic_json(OUT/'BEST_CHECKPOINT_REPLAY_AUDIT.json',{'status':'PASS','rows':replays,'no_A_B_or_test_scored':True})
 counts={r:sum(x['role']==r for x in mem) for r in ('model_selection','calibration_A','threshold_B')}
 atomic_json(OUT/'ROLE_ACCESS_AUDIT.json',{'status':'PASS','membership_file_sha256':sha(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv'),'counts':counts,'exact_partition_of_previous_validation':sum(counts.values())==3488,'IDs_and_waveforms_disjoint_across_roles':True,'all_training_epoch_scores_model_selection_only':True,'all_A_B_scoring_after_all_models_lock':True,'patient_disjointness':'not established; record/waveform disjoint only','best_checkpoint_replay_sha256':sha(OUT/'BEST_CHECKPOINT_REPLAY_AUDIT.json')})
 core={}
 for method in c.METHODS:
  for seed in c.SEEDS:
   core[method,seed,'calibration_A']=infer_units(method,seed,parts['calibration_A'],'calibration_A')[:2]
   core[method,seed,'threshold_B']=infer_units(method,seed,parts['threshold_B'],'threshold_B')[:2]
 # Detector artifacts must already be generated by the frozen TensorFlow detector environment.
 detbind=json.loads((OUT/'DETECTOR_ROLE_BINDINGS.json').read_text(encoding='utf8'))
 if detbind.get('status')!='PASS':raise RuntimeError('missing verified detector score bindings')
 huang={}
 for seed in c.SEEDS:
  d=OUT/'fits'/'ordinary'/f'seed_{seed}';ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu');model=ECGResNetSmall().cuda().eval();model.load_state_dict(ck['state_dict']);mean,std=np.load(d/'preprocessing.npz')['mean'],np.load(d/'preprocessing.npz')['std']
  for role in ('calibration_A','threshold_B'):
   raw=raw500(parts[role]);detp=np.load(OUT/f'detector/{role}.npz',allow_pickle=False)['probabilities'][0];classes=np.argmax(detp,axis=1).astype(np.int64);fixed=corrected_batch(raw,classes);p=infer_huang(model,fixed,mean,std);y=c.labels(parts[role]);ids=np.asarray([r['record_id'] for r in parts[role]])
   dest=OUT/f'scores/{role}/Huang_full_pipeline/seed_{seed}.npz';atomic_npz(dest,p=p,y=y,record_ids=ids,role=np.asarray(role),condition=np.asarray('clean'),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),detector_sha256=np.asarray(sha(OUT/f'detector/{role}.npz')),detector_classes=classes)
   huang[seed,role]=(p,y)
  del model;torch.cuda.empty_cache()
 cals={};rows=[]
 for method in (*c.METHODS,'Huang_full_pipeline'):
  for seed in c.SEEDS:
   for ei,ep in enumerate(c.ENDS):
    if method=='Huang_full_pipeline':pa,ya=huang[seed,'calibration_A'];pb,yb=huang[seed,'threshold_B']
    else:pa,ya=core[method,seed,'calibration_A'];pb,yb=core[method,seed,'threshold_B']
    fit=c.platt_fit(ya[:,ei],pa[:,ei]);neg=pb[yb[:,ei]==0,ei]
    if len(np.unique(ya[:,ei]))!=2 or not len(neg) or not fit['success'] or not np.isfinite([fit['a'],fit['b']]).all():rows.append({'method':method,'seed':seed,'endpoint':ep,'status':'FAIL','reason':'class/convergence/nonfinite'});continue
    q=float(np.quantile(neg,.9,method='linear'));mapped=float(c.calibrate_apply(np.asarray([q]),fit)[0]);cals[method,seed,ei]=(fit,q,mapped)
    rows.append({'method':method,'seed':seed,'endpoint':ep,'status':'PASS','n_A':len(ya),'n_A_positive':int(ya[:,ei].sum()),'n_B_negative':len(neg),'raw_q90':q,'mapped_calibrated_threshold':mapped,'slope_a':fit['a'],'intercept_b':fit['b'],'fit_success':fit['success'],'fit_loss':fit['loss'],'fit_iterations':fit['n_iter']})
 if len(rows)!=36 or any(x['status']!='PASS' for x in rows):atomic_json(OUT/'STATUS.json',{'status':'STOP_CALIBRATION_FAILURE','rows':rows});raise RuntimeError('not all 36 calibration/threshold fits passed')
 import csv
 with (OUT/'CALIBRATION_AND_THRESHOLD_LOCK.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic_json(OUT/'CALIBRATION_AND_THRESHOLD_LOCK.json',{'status':'LOCKED','entries':rows,'locked_utc':c.now(),'fit_data':'clean calibration_A only','threshold_data':'clean negative threshold_B only','threshold_rule':'linear q90; raw score >= q90','models_locked_before_fit':True,'all_endpoints_passed':True})
 atomic_json(OUT/'STATUS.json',{'status':'CALIBRATION_AND_THRESHOLDS_LOCKED','test_not_yet_scored':True,'utc':c.now()})
 print(json.dumps({'status':'CALIBRATION_AND_THRESHOLDS_LOCKED','n_fits':len(rows)}))
if __name__=='__main__':main()
