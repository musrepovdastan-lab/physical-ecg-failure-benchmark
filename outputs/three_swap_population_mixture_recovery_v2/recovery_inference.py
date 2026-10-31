from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from src.chapman_phase1 import ECGResNetSmall
from src.heedb_huang_adapter_v1 import electrical_reversal_batch,corrected_batch,diagnostic_input
from src.ecg_physics import DISPLAY_LEADS,derive_display,named_reversal

SEEDS=(104729,130363,155921);FAULTS=('RA_LA_reversal','LA_LL_reversal')
CONDS=('clean','RA_LL_reversal','RA_LA_reversal','LA_LL_reversal');ALL_FAULTS=CONDS[1:];CLASS_TO_FAULT={1:'RA_LA_reversal',2:'RA_LL_reversal',3:'LA_LL_reversal'}
PRIVATE=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1';V1=ROOT/'outputs/three_swap_population_mixture_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_array(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def atomic_npz(dest,**kw):
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.stem+'.partial.npz');np.savez_compressed(tmp,**kw);tmp.replace(dest)
def check_protocol():
 p=OUT/'RECOVERY_PROTOCOL.json';h=sha(p)
 if h!=(OUT/'RECOVERY_PROTOCOL_SHA256.txt').read_text(encoding='ascii').strip():raise RuntimeError('protocol file hash mismatch')
 d=json.loads(p.read_text(encoding='utf-8'))
 for rel,v in d['input_bindings'].items():
  if sha(ROOT/rel)!=v['sha256']:raise RuntimeError('frozen binding changed: '+rel)
 return d

def explicit_swap(x,condition):
 """Independent terminal-potential formula used as the replay oracle."""
 a=np.asarray(x,dtype=np.float64);i=a[:,0:1,:];ii=a[:,1:2,:]
 if condition=='RA_LA_reversal':new_i,new_ii=-i,ii-i
 elif condition=='RA_LL_reversal':new_i,new_ii=i-ii,-ii
 elif condition=='LA_LL_reversal':new_i,new_ii=ii,i
 else:raise ValueError(condition)
 limb=np.concatenate([new_i,new_ii,new_ii-new_i,-(new_i+new_ii)/2,new_i-new_ii/2,new_ii-new_i/2],axis=1)
 return np.concatenate([limb,a[:,6:12,:]],axis=1).astype(np.float32)

def explicit_inverse_chain(faulty,classes):
 x=np.asarray(faulty,dtype=np.float32).copy();codes=np.asarray(classes)
 out=x.copy()
 for code,condition in CLASS_TO_FAULT.items():
  ix=np.flatnonzero(codes==code)
  if len(ix):out[ix]=explicit_swap(x[ix],condition)
 return out

def manual_identity(x):
 a=np.asarray(x,dtype=np.float32);i=a[:,0:1,:];ii=a[:,1:2,:]
 limb=np.concatenate([i,ii,ii-i,-(i+ii)/2,i-ii/2,ii-i/2],axis=1)
 return np.concatenate([limb,a[:,6:12,:]],axis=1).astype(np.float32)

def infer(model,x,mean,std):
 x=np.asarray(x,dtype=np.float32)
 y=((x[:,:,::5]-mean[None,:,None])/std[None,:,None]).astype(np.float32)
 if not np.array_equal(y[0],diagnostic_input(x[0],mean,std)):raise RuntimeError('batch normalizer differs from frozen per-record adapter')
 if y.shape!=(len(x),12,1000) or y.dtype!=np.float32:raise RuntimeError('diagnostic input geometry/dtype changed')
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  p=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(y)).cuda())).float().cpu().numpy()
 return p

def engineering_handoff(raw,det):
 rows=[];source=raw[:4].copy();original_hash=sha_array(source)
 synth=np.random.default_rng(99201).normal(0,1,(4,12,5000)).astype(np.float32)
 synth[:,6:]=np.random.default_rng(771).normal(0,1,(4,6,5000)).astype(np.float32)
 for kind,x in [('real',source),('synthetic',synth)]:
  pristine=x.copy()
  for fault in ALL_FAULTS:
   incoming=electrical_reversal_batch(x,fault);expected_in=explicit_swap(x,fault)
   transform_delta=float(np.max(np.abs(incoming-expected_in)))
   # Derived limb leads differ by at most one float32 rounding step at this signal scale.
   if transform_delta>1e-4:raise RuntimeError('incoming swap differs from independent physical equations')
   for predclass in range(4):
    classes=np.full(len(x),predclass,dtype=np.int64)
    passed=corrected_batch(incoming,classes)
    expected=explicit_inverse_chain(expected_in,classes)
    delta=float(np.max(np.abs(passed-expected)))
    if delta>2e-4:raise RuntimeError(f'predicted class composition mismatch {fault}/{predclass}/{kind}: {delta}')
    if predclass==0 and not np.array_equal(passed,incoming):raise RuntimeError('class 0 modified incoming faulty signal')
    rows.append({'signal_kind':kind,'fault':fault,'predicted_class':predclass,'incoming_transform_max_abs_delta':transform_delta,
      'correction_vs_independent_composition_max_abs_delta':delta,'incoming_sha256':sha_array(incoming),
      'correction_input_sha256':sha_array(incoming),'post_correction_sha256':sha_array(passed),'class_zero_keeps_faulty':predclass!=0 or np.array_equal(passed,incoming),
      'dtype':str(passed.dtype),'shape':list(passed.shape),'status':'PASS'})
   # Correct-class inverse engineered control only; not used in evaluated pipeline.
   truecode={'RA_LA_reversal':1,'RA_LL_reversal':2,'LA_LL_reversal':3}[fault]
   correct=corrected_batch(incoming,np.full(len(x),truecode,dtype=np.int64));identity=manual_identity(x)
   err=float(np.max(np.abs(correct-identity)))
   if err>2e-4:raise RuntimeError(f'known-class inverse identity control failed {fault}/{kind}: {err}')
   rows.append({'signal_kind':kind,'fault':fault,'predicted_class':truecode,'control':'known_class_inverse_engineering_only',
     'identity_rederived_max_abs_delta':err,'tolerance':2e-4,'status':'PASS'})
  if not np.array_equal(x,pristine):raise RuntimeError('source raw buffer mutated')
 # Tensor geometry and normalization/stride use the unchanged adapter.
 probe=diagnostic_input(source[0],np.zeros(12,dtype=np.float32),np.ones(12,dtype=np.float32))
 if probe.shape!=(12,1000) or probe.dtype!=np.float32 or not np.array_equal(source,raw[:4]):raise RuntimeError('stride/channel/order/dtype/raw immutability check failed')
 result={'status':'PASS','checks':rows,'raw_channel_order':list(DISPLAY_LEADS),'batch_geometry':[4,12,5000],
  'diagnostic_stride':5,'diagnostic_geometry':[4,12,1000],'raw_dtype':'float32','normalized_dtype':'float32',
  'known_class_inverse_is_engineering_control_only':True,'source_buffer_sha256_before_after':original_hash}
 (OUT/'INPUT_HANDOFF_ENGINEERING_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 return result

def direct_explicit_infer(model,x,mean,std):
 # Explicit frozen model path, independent of adapter and correction helpers.
 z=np.asarray(x,dtype=np.float32)[:,:,::5]
 z=(z-mean[None,:,None])/std[None,:,None];z=np.ascontiguousarray(z,dtype=np.float32)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  return torch.sigmoid(model(torch.from_numpy(z).cuda())).float().cpu().numpy()

def main():
 protocol=check_protocol();detfile=V1/'huang_detector_probabilities.npz'
 with np.load(detfile,allow_pickle=False) as z:
  dp=z['probabilities'];detconds=z['conditions'].astype(str).tolist();classes=z['class_names'].astype(str).tolist()
 raw=np.load(PRIVATE/'signals_10000_float32.npy',mmap_mode='r',allow_pickle=False)
 if raw.shape!=(10000,12,5000) or raw.dtype!=np.float32:raise RuntimeError('raw HEEDB source geometry changed')
 handoff=engineering_handoff(raw,dp)
 detector_audit=json.loads((OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 if detector_audit.get('status')!='PASS':raise RuntimeError('detector probability replay failed')
 # Verify the inference path before full scoring using three frozen batch blocks.
 blocks=((0,128,32),(4992,5120,None),(9984,10000,None));replay=[];full_records=[];started=time.monotonic()
 sel=PRIVATE/'selection_initial.csv';import pandas as pd
 fileids=pd.read_csv(sel,dtype=str,keep_default_na=False)['FileName'].astype(str).to_numpy(dtype=str)
 sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
 from outputs.mixed_fault_calibration_v1 import run_analysis as base
 labels,pids=base.heedb_labels();labels=np.asarray(labels,dtype=np.uint8);pids=np.asarray(pids).astype(str)
 for seed in SEEDS:
  md=ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}';ck=md/'checkpoint.pt';prep=md/'preprocessing.npz'
  state=torch.load(ck,map_location='cpu',weights_only=True)
  model=ECGResNetSmall().cuda().eval();model.load_state_dict(state['state_dict'])
  with np.load(prep,allow_pickle=False) as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
  for fault in FAULTS:
   unit_started=time.monotonic();ci=detconds.index(fault);chunks=[];perfile=[]
   for start in range(0,10000,128):
    stop=min(start+128,10000);clean=np.asarray(raw[start:stop],dtype=np.float32)
    pristine_hash=sha_array(clean);incoming=electrical_reversal_batch(clean,fault)
    predclasses=np.argmax(dp[ci,start:stop],axis=1).astype(np.int64)
    corrected=corrected_batch(incoming,predclasses)
    handoff_delta=float(np.max(np.abs(incoming-explicit_swap(clean,fault))))
    if handoff_delta>1e-4:raise RuntimeError(f'fault handoff mismatch {fault} {start}: {handoff_delta}')
    # Classes are only detector argmax; the true fault is never used to choose correction.
    p=infer(model,corrected,mean,std);chunks.append(p)
    if not np.array_equal(clean,raw[start:stop]):raise RuntimeError('read-only source buffer changed')
    perfile.append({'block':[start,stop],'incoming_sha256':sha_array(incoming),'detector_classes_sha256':sha_array(predclasses),
     'detector_probabilities_slice_sha256':sha_array(dp[ci,start:stop]),'corrector_input_sha256':sha_array(incoming),
     'corrected_signal_sha256':sha_array(corrected),'source_clean_sha256':pristine_hash,'batch_size':stop-start})
    if (stop % 1024 == 0) or stop == 10000:print(f'INFER Huang/{seed}/{fault} {stop}/10000',flush=True)
    if start in (0,4992,9984):
     n=32 if start==0 else stop-start;record_end=start+n
     explicit=explicit_swap(clean,fault)
     classes_n=np.argmax(dp[ci,start:stop],axis=1).astype(np.int64)
     expcorr=explicit_inverse_chain(explicit,classes_n)
     gotp=direct_explicit_infer(model,expcorr,mean,std)
     helperp=p[:n]
     delta=float(np.max(np.abs(helperp-gotp[:n])))
     oldpath=corrected_batch(clean,classes_n)
     wrongp=direct_explicit_infer(model,oldpath,mean,std)
     oldf=V1/f'predictions/HEEDB/Huang_full_pipeline/seed_{seed}/{fault}.npz'
     with np.load(oldf,allow_pickle=False) as oz:oldscore=oz['p'][start:record_end]
     wrongdelta=float(np.max(np.abs(oldscore-wrongp[:n])))
     rec={'seed':seed,'condition':fault,'block':[start,record_end],'n':n,'corrected_helper_vs_independent_explicit_max_abs_delta':delta,
       'wrong_historical_path_vs_legacy_score_max_abs_delta':wrongdelta,'wrong_vs_correct_raw_signal_max_abs_delta':float(np.max(np.abs(oldpath-expcorr))),
       'detector_classes_sha256':sha_array(classes_n[:n]),'incoming_sha256':sha_array(explicit[:n]),'historical_model_batch_size':stop-start,'tolerance':1e-4,
       'wrong_legacy_witness_match':wrongdelta<=1e-4,'correct_path_replay_pass':delta<=1e-4,'status':'PASS' if delta<=1e-4 and wrongdelta<=1e-4 else 'FAIL'}
     replay.append(rec)
     if rec['status']!='PASS':raise RuntimeError(f'correct/wrong path replay failure: {rec}')
   p_all=np.concatenate(chunks,axis=0)
   dest=OUT/f'predictions/HEEDB/Huang_full_pipeline/seed_{seed}/{fault}.npz'
   # Add immutable source and inference bindings for downstream audits.
   atomic_npz(dest,p=p_all,labels=labels,patient_ids=pids,file_names=fileids,condition=np.asarray(fault),method=np.asarray('Huang_full_pipeline'),seed=np.asarray(seed),
    checkpoint_sha256=np.asarray(sha(ck)),preprocessing_sha256=np.asarray(sha(prep)),source_ids_sha256=np.asarray(sha(sel)),
    detector_probabilities_sha256=np.asarray(sha(detfile)),detector_condition_index=np.asarray(ci),incoming_condition=np.asarray(fault),
    recovery_protocol_file_sha256=np.asarray(sha(OUT/'RECOVERY_PROTOCOL.json')),
    recovery_protocol_canonical_fingerprint=np.asarray(protocol['canonical_json_fingerprint_sha256']))
   full_records.append({'method':'Huang_full_pipeline','seed':seed,'condition':fault,'status':'NEW_CORRECTED_FULL_INFERENCE',
    'path':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest),'rows':len(p_all),'file_ids_sha256':protocol['source_cohort']['file_ids_sha256'],
    'patient_ids_sha256':protocol['source_cohort']['patient_ids_sha256'],'labels_sha256':protocol['source_cohort']['labels_sha256'],
    'checkpoint_sha256':sha(ck),'preprocessing_sha256':sha(prep),'detector_artifact_sha256':sha(detfile),'detector_condition_index':ci,
    'block_input_records':perfile,'elapsed_seconds':time.monotonic()-unit_started})
  del model;torch.cuda.empty_cache()
 elapsed=time.monotonic()-started
 complete={'status':'PASS','new_files':full_records,'elapsed_seconds_total':elapsed,'elapsed_seconds_sum_of_units':elapsed,
  'cuda_device':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'precision':'float32 input, float16 CUDA autocast',
  'no_training_or_fitting':True,'reserved_test_evaluated':False,'shared_detector_probabilities_across_seeds':True}
 (OUT/'RECOVERY_INFERENCE_COMPLETE.json').write_text(json.dumps(complete,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 ra={'status':'PASS','protocol_file_sha256':sha(OUT/'RECOVERY_PROTOCOL.json'),'protocol_canonical_fingerprint_sha256':protocol['canonical_json_fingerprint_sha256'],
  'detector_block_replay_audit_sha256':sha(OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json'),'engineering_handoff_audit_sha256':sha(OUT/'INPUT_HANDOFF_ENGINEERING_AUDIT.json'),
  'rows':replay,'max_correct_path_delta':max(r['corrected_helper_vs_independent_explicit_max_abs_delta'] for r in replay),
  'wrong_path_control_rows':sum(r['wrong_legacy_witness_match'] for r in replay)}
 (OUT/'REPLAY_AUDIT.json').write_text(json.dumps(ra,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','units':len(full_records),'elapsed_seconds':elapsed,'max_replay_delta':ra['max_correct_path_delta']}))
if __name__=='__main__':main()
