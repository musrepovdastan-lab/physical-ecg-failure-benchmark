from pathlib import Path
import sys,json,hashlib,csv
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT));import evaluate as e
from src.chapman_phase1 import ECGResNetSmall
from src.heedb_huang_adapter_v1 import corrected_batch,electrical_reversal_batch,diagnostic_input
from src import huang_detector_infer_v2 as detector
FAULTS=e.CONDS[1:]
def shaarr(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def explicit_swap(x,name):
 a=np.asarray(x,dtype=np.float64);i=a[:,0:1];ii=a[:,1:2]
 if name=='RA_LA_reversal':ni,nj=-i,ii-i
 elif name=='RA_LL_reversal':ni,nj=i-ii,-ii
 elif name=='LA_LL_reversal':ni,nj=ii,i
 else:raise ValueError(name)
 limb=np.concatenate([ni,nj,nj-ni,-(ni+nj)/2,ni-nj/2,nj-ni/2],axis=1)
 return np.concatenate([limb,a[:,6:12]],axis=1).astype(np.float32)
def explicit_predicted_inverse(x,codes):
 out=x.copy()
 for cls,name in ((1,'RA_LA_reversal'),(2,'RA_LL_reversal'),(3,'LA_LL_reversal')):
  ix=np.flatnonzero(codes==cls)
  if len(ix):out[ix]=explicit_swap(x[ix],name)
 return out
def infer_explicit(model,x,mean,std):
 z=np.stack([diagnostic_input(r,mean,std) for r in x]).astype(np.float32);out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(z),128):
   with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(z[i:i+128]).cuda(non_blocking=True))).float().cpu().numpy())
 return np.concatenate(out)
def main():
 from outputs.mixed_fault_calibration_v1 import run_analysis as base
 # Ningbo source order and detector cache.
 parts,_=e.c.records();nr=parts['test'];nids=np.asarray([r['record_id'] for r in nr]);nraw=e.raw500(nr);nd=np.load(OUT/'detector/Ningbo_test.npz',allow_pickle=False);ndp=nd['probabilities'];nconds=list(nd['conditions'].astype(str))
 private=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1';hraw=np.load(private/'signals_10000_float32.npy',mmap_mode='r',allow_pickle=False);sel=__import__('pandas').read_csv(private/'selection_initial.csv',dtype=str,keep_default_na=False);hids=sel['FileName'].astype(str).to_numpy();hd,_dh,_ord=e.load_det_heedb();hconds=list(hd['conditions'].astype(str));horder=[hconds.index(c) for c in detector.CONDITIONS];hdp=hd['probabilities'][horder]
 rows=[];misc=[]
 for cohort,raw,ids,dp,dconds in (('Ningbo_test',nraw,nids,ndp,nconds),('HEEDB',hraw,hids,hdp,list(detector.CONDITIONS))):
  n=len(ids);mid=(n//2//128)*128;last=((n-1)//128)*128;blocks=((0,min(128,n)),(mid,min(mid+128,n)),(last,n))
  for seed in e.c.SEEDS:
   d=OUT/'fits'/'ordinary'/f'seed_{seed}';model=ECGResNetSmall().cuda().eval();model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cpu')['state_dict']);mean,std=np.load(d/'preprocessing.npz',allow_pickle=False)['mean'],np.load(d/'preprocessing.npz',allow_pickle=False)['std']
   for cond in e.CONDS:
    ci=e.CONDS.index(cond);di=dconds.index(cond);dest=OUT/f'predictions/{cohort}/Huang_full_pipeline/seed_{seed}/{cond}.npz'
    with np.load(dest,allow_pickle=False) as z:stored=z['p'].copy()
    classes_all=np.argmax(dp[di],axis=1).astype(np.int64)
    if cond=='clean':misc.append({'cohort':cohort,'seed':seed,'n':n,'clean_records_with_nonzero_predicted_inverse':int(np.sum(classes_all!=0)),'fraction_clean_records_miscorrected':float(np.mean(classes_all!=0))})
    for bi,(lo,hi) in enumerate(blocks):
     clean=np.asarray(raw[lo:hi],dtype=np.float32)
     incoming=clean if cond=='clean' else (np.stack([detector.apply_condition(x,cond) for x in clean]) if cohort=='Ningbo_test' else electrical_reversal_batch(clean,cond))
     codes=classes_all[lo:hi];actual=corrected_batch(incoming,codes);manual=explicit_predicted_inverse(incoming,codes);delta=float(np.max(np.abs(actual-manual)))
     pred=infer_explicit(model,manual,mean,std);saved=stored[lo:hi];pdelta=float(np.max(np.abs(pred-saved)))
     row={'cohort':cohort,'seed':seed,'condition':cond,'block_index':bi,'source_record_ids_sha256':hashlib.sha256('\n'.join(ids[lo:hi].astype(str)).encode()).hexdigest(),'range':[int(lo),int(hi)],'detector_classes_sha256':shaarr(codes),'incoming_sha256':shaarr(incoming),'corrected_helper_vs_explicit_max_abs_delta':delta,'explicit_prediction_vs_saved_max_abs_delta':pdelta,'detector_path':'frozen class argmax on matching incoming condition','classifier_batch_size':128,'autocast':'float16','signal_rate_hz':500,'tolerance_input':2e-4,'tolerance_prediction':1e-4,'status':'PASS' if delta<=2e-4 and pdelta<=1e-4 else 'FAIL'};rows.append(row)
     if row['status']!='PASS':print(json.dumps(row),flush=True)
    if any(x['status']!='PASS' for x in rows if x['cohort']==cohort and x['seed']==seed):raise RuntimeError('Huang replay failed')
   del model;torch.cuda.empty_cache()
 e.write_csv(OUT/'HUANG_SOURCE_ALIGNED_REPLAY.csv',rows);e.write_csv(OUT/'HUANG_CLEAN_MISCORRECTION_RATES.csv',misc)
 audit={'status':'PASS' if len(rows)==72 and all(x['status']=='PASS' for x in rows) else 'FAIL','rows':len(rows),'blocks':'first/middle/last source-aligned inference batches; batch size 128 except original shorter final batch','tolerance_input_max_abs':2e-4,'tolerance_prediction_max_abs':1e-4,'all_fault_handoffs_keep_same_incoming_signal':True,'clean_detector_false_corrections_measured':True,'detector_class_map':[0,1,2,3]}
 e.atomic_json(OUT/'HUANG_SOURCE_ALIGNED_REPLAY_AUDIT.json',audit);print(json.dumps(audit))
if __name__=='__main__':main()
