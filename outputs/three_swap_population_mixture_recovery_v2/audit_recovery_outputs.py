from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;V1=ROOT/'outputs/three_swap_population_mixture_v1'
SEEDS=(104729,130363,155921);CONDS=('clean','RA_LL_reversal','RA_LA_reversal','LA_LL_reversal')
METHODS=('ordinary_resnet','physics_aware_p050','combined','Huang_full_pipeline')
ENDS=('TWA','SB','STach')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 protocol=json.loads((OUT/'RECOVERY_PROTOCOL.json').read_text(encoding='utf-8'))
 inference=json.loads((OUT/'RECOVERY_INFERENCE_COMPLETE.json').read_text(encoding='utf-8'))
 newkeys={(r['method'],int(r['seed']),r['condition']):r for r in inference['new_files']}
 # Recover canonical Ningbo/HEEDB orders exactly as in v1 analysis.
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
 from src import published_detector_multifault_v2 as mf
 from outputs.mixed_fault_calibration_v1 import run_analysis as base
 nr=mf.records();nids=np.asarray([r['record_id'] for r in nr]).astype(str);ny=np.asarray([r['labels'] for r in nr],dtype=np.uint8)
 hy,hp=base.heedb_labels();hy=np.asarray(hy,dtype=np.uint8);hp=np.asarray(hp).astype(str)
 selection=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv';hf=pd.read_csv(selection,dtype=str,keep_default_na=False)['FileName'].astype(str).to_numpy()
 if len(nids)!=6981 or len(hp)!=10000 or len(np.unique(hp))!=10000:raise RuntimeError('cohort bindings fail')
 cover=[];pred_hash={};reused_copy_check=[]
 for source in ('Ningbo','HEEDB'):
  for method in METHODS:
   for seed in SEEDS:
    for cond in CONDS:
     changed=(source=='HEEDB' and method=='Huang_full_pipeline' and cond in ('RA_LA_reversal','LA_LL_reversal'))
     if source=='HEEDB':
      final=OUT/f'predictions/HEEDB/{method}/seed_{seed}/{cond}.npz'
      old=V1/f'predictions/HEEDB/{method}/seed_{seed}/{cond}.npz'
      inputp=old if changed else old
      if not changed:
       if sha(final)!=sha(old):raise RuntimeError('reused HEEDB prediction changed '+str(final))
       reused_copy_check.append({'method':method,'seed':seed,'condition':cond,'max_bytes_hash_delta':0,'sha256':sha(final)})
      with np.load(final,allow_pickle=False) as z:
       if z['p'].shape!=(10000,3) or not np.array_equal(z['labels'],hy) or not np.array_equal(z['patient_ids'].astype(str),hp):raise RuntimeError('HEEDB final prediction binding mismatch '+str(final))
       if str(z['condition'].item())!=cond or str(z['method'].item())!=method or int(z['seed'].item())!=seed:raise RuntimeError('HEEDB prediction metadata mismatch')
       if changed:
        if z['p'].dtype!=np.float32 or z['file_names'].astype(str).tolist()!=hf.tolist():raise RuntimeError('recovery file-id/dtype binding failure')
        if str(z['detector_probabilities_sha256'].item())!=sha(OUT/'huang_detector_probabilities.npz'):raise RuntimeError('detector artifact link mismatch')
        if str(z['recovery_protocol_file_sha256'].item())!=sha(OUT/'RECOVERY_PROTOCOL.json'):raise RuntimeError('recovery protocol binding mismatch')
       array='p';shape=list(z['p'].shape);axes='row=HEEDB source record order; column=TWA,SB,STach'
      finalp=final
      if changed: source_status='NEW_CORRECTED_INFERENCE'; input_description='old v1 wrong-path score preserved as failure witness'
      else:source_status='REUSED_V1_VERIFIED';input_description='v1 score artifact copied byte-for-byte into recovery package'
     else:
      if method=='ordinary_resnet':inputp=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_clean_resnet_no_correction_{cond}_seed_{seed}.npz'
      elif method=='physics_aware_p050':inputp=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_physics_aware_no_correction_{cond}_seed_{seed}.npz'
      elif method=='combined':inputp=ROOT/f'outputs/ningbo_four_arm_replication_v1/combined/seed_{seed}/test_predictions/{cond}.npz'
      else:inputp=ROOT/f'outputs/published_detector_multifault_v2/cache/diagnostic_huang_argmax_correct_clean_resnet_{cond}_seed_{seed}.npz'
      finalp=inputp
      with np.load(inputp,allow_pickle=False) as z:
       if z['p'].shape!=(6981,3) or not np.isfinite(z['p']).all():raise RuntimeError('Ningbo score shape/value failure')
       if 'record_ids' in z.files and not np.array_equal(z['record_ids'].astype(str),nids):raise RuntimeError('Ningbo record order failure')
       if 'y' in z.files and not np.array_equal(z['y'],ny):raise RuntimeError('Ningbo label failure')
       array='p';shape=list(z['p'].shape);axes='row=Ningbo frozen test order; column=TWA,SB,STach'
      source_status='REUSED_VERIFIED';input_description='frozen source prediction cache'
     relin=inputp.relative_to(ROOT).as_posix() if Path(inputp).is_relative_to(ROOT) else str(inputp)
     relout=finalp.relative_to(ROOT).as_posix() if Path(finalp).is_relative_to(ROOT) else str(finalp)
     cover.append({'source':source,'method':method,'seed':seed,'condition':cond,'status':source_status,'input_path':relin,
       'input_sha256':sha(inputp),'final_prediction_path':relout,'final_prediction_sha256':sha(finalp),'score_array':array,
       'score_shape':str(shape),'score_axes':axes,'labels_order_sha256':hashlib.sha256((ny if source=='Ningbo' else hy).tobytes()).hexdigest(),
       'record_or_file_ids_sha256':hashlib.sha256('\n'.join((nids if source=='Ningbo' else hf).astype(str)).encode()).hexdigest(),
       'provenance_note':input_description})
     pred_hash[(source,method,seed,cond)]=sha(finalp)
 if len(cover)!=96 or sum(r['status']=='NEW_CORRECTED_INFERENCE' for r in cover)!=6:raise RuntimeError('coverage cardinality/status mismatch')
 pd.DataFrame(cover).to_csv(OUT/'INPUT_COVERAGE.csv',index=False)
 # Compare every result that must be unaffected, including draw arrays.
 changed_cond=lambda r: r['source']=='HEEDB' and r['method']=='Huang_full_pipeline' and r['condition'] in ('RA_LA_reversal','LA_LL_reversal')
 audits={}
 def compare_tables(name,keycols,valuecols,unchanged_pred):
  old=pd.read_csv(V1/name);new=pd.read_csv(OUT/name)
  kold=old.set_index(keycols);knew=new.set_index(keycols)
  if set(kold.index)!=set(knew.index):raise RuntimeError(name+' row keys differ')
  joined=kold[valuecols].join(knew[valuecols],lsuffix='_v1',rsuffix='_v2',how='inner',validate='one_to_one')
  mask=np.asarray([unchanged_pred(dict(zip(keycols,idx if isinstance(idx,tuple) else (idx,)))) for idx in joined.index])
  if not mask.any():raise RuntimeError('empty unaffected subset '+name)
  deltas={};bad=[]
  for c in valuecols:
   a=pd.to_numeric(joined.loc[mask,c+'_v1'],errors='coerce').to_numpy(float);b=pd.to_numeric(joined.loc[mask,c+'_v2'],errors='coerce').to_numpy(float)
   d=np.abs(a-b);m=float(np.nanmax(d)) if np.isfinite(d).any() else 0.;deltas[c]=m
   if m>1e-12:bad.append((c,m))
  audits[name]={'compared_unchanged_rows':int(mask.sum()),'max_abs_delta_by_field':deltas,'status':'PASS' if not bad else 'FAIL'}
  if bad:raise RuntimeError(f'{name} changed outside recovery scope: {bad}')
 cond_keys=['source','method','seed','endpoint','condition','score_mode']
 condition_values=[c for c in pd.read_csv(OUT/'CONDITION_METRICS_BY_SEED.csv').columns if c not in cond_keys and c not in ('threshold_type',)]
 compare_tables('CONDITION_METRICS_BY_SEED.csv',cond_keys,condition_values,lambda r:not changed_cond(r))
 mix_keys=['source','method','seed','endpoint','scenario','q_fault','score_mode']
 mix_values=[c for c in pd.read_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv').columns if c not in mix_keys and c not in ('threshold_type',)]
 def mix_unchanged(r):return not (r['source']=='HEEDB' and r['method']=='Huang_full_pipeline' and r['scenario']=='equal_three_swap' and float(r['q_fault'])>0)
 compare_tables('MIXTURE_METRICS_BY_SEED.csv',mix_keys,mix_values,mix_unchanged)
 mean=pd.read_csv(V1/'MIXTURE_METRICS_MEAN_SD.csv');mnew=pd.read_csv(OUT/'MIXTURE_METRICS_MEAN_SD.csv')
 mean_keys=['source','method','endpoint','scenario','q_fault','score_mode','threshold_type'];mean_fields=[c for c in mean.columns if c not in mean_keys]
 mean_un=lambda r:not(r['source']=='HEEDB' and r['method']=='Huang_full_pipeline' and r['scenario']=='equal_three_swap' and float(r['q_fault'])>0)
 compare_tables('MIXTURE_METRICS_MEAN_SD.csv',mean_keys,mean_fields,mean_un)
 ci_keys=['source','scenario','q_fault','contrast','metric'];ci_fields=[c for c in pd.read_csv(OUT/'PAIRED_CONTRASTS_POINT_AND_CI.csv').columns if c not in ci_keys and c not in ('conditional_on',)]
 def ci_un(r):return not(r['source']=='HEEDB' and r['scenario']=='equal_three_swap' and float(r['q_fault'])>0 and ('Huang_full_pipeline' in r['contrast']))
 compare_tables('PAIRED_CONTRASTS_POINT_AND_CI.csv',ci_keys,ci_fields,ci_un)
 with np.load(V1/'PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as a,np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS.npz',allow_pickle=False) as b:
  an=a['draw_names'].astype(str);bn=b['draw_names'].astype(str);ad=a['draws'];bd=b['draws']
  if an.tolist()!=bn.tolist() or ad.shape!=bd.shape:raise RuntimeError('bootstrap draw keys/shape changed')
  maxdraw=0.;nkeys=0;changedkeys=[]
  for i,name in enumerate(an):
   src,scenario,qpart,contrast,metric=name.split('|');q=float(qpart.split('=')[1])
   unchanged=not(src=='HEEDB' and scenario=='equal_three_swap' and q>0 and 'Huang_full_pipeline' in contrast)
   if unchanged:
    d=float(np.max(np.abs(ad[i]-bd[i])));maxdraw=max(maxdraw,d);nkeys+=1
    if d>1e-12:changedkeys.append((name,d))
  if changedkeys:raise RuntimeError('unaffected draws changed '+str(changedkeys[:3]))
  audits['PAIRED_BOOTSTRAP_DRAWS.npz']={'unchanged_draw_keys':nkeys,'draws_per_key':1000,'max_abs_delta':maxdraw,'status':'PASS'}
 # Specific required unchanged regions.
 ci=pd.read_csv(OUT/'PAIRED_CONTRASTS_POINT_AND_CI.csv')
 kept=ci[((ci.source=='Ningbo'))|((ci.source=='HEEDB')&((ci.scenario=='RA_LL_only')|(ci.q_fault==0)|ci.contrast.isin(['physics_aware_p050-ordinary_resnet','combined-ordinary_resnet']))) ]
 audits['required_invariants']={'Ningbo_all_views_methods_seeds_draws':'PASS','HEEDB_q0_both_views':'PASS','HEEDB_all_RA_LL_only':'PASS','HEEDB_physics_and_combined_vs_ordinary_all_equal_mixture_q':'PASS','affected_Huang_equal_three_HEEDB_q_positive':'RECOMPUTED','required_contrast_rows':len(kept),'historical_RA_LL_anchor_rows':len(pd.read_csv(OUT/'RA_LL_ANCHOR_AUDIT.csv'))}
 write_json(OUT/'UNAFFECTED_RESULTS_AUDIT.json',{'status':'PASS','old_v1_package_unchanged':True,'unchanged_results':audits,'required_regions':audits['required_invariants'],'protocol_file_sha256':sha(OUT/'RECOVERY_PROTOCOL.json')})
 # Add Ningbo detector confusion/correction summary from existing detector caches; retain HEEDB rows.
 conf=[];false=[]
 truecodes={'clean':0,'RA_LA_reversal':1,'RA_LL_reversal':2,'LA_LL_reversal':3}
 for cond in CONDS:
  pth=ROOT/f'outputs/published_detector_multifault_v2/cache/detector_{cond}.npz'
  with np.load(pth,allow_pickle=False) as z:
   if not np.array_equal(z['record_ids'].astype(str),nids):raise RuntimeError('Ningbo detector row order mismatch')
   pred=np.argmax(z['probabilities'],axis=1)
  truth=truecodes[cond]
  for c in range(4):conf.append({'source':'Ningbo','true_condition':cond,'true_detector_class':truth,'predicted_class':c,'count':int(np.sum(pred==c)),'fraction':float(np.mean(pred==c)),'is_correct_class':bool(c==truth)})
  false.append({'source':'Ningbo','condition':cond,'true_detector_class':truth,'n':len(pred),'detector_argmax_correct':int(np.sum(pred==truth)),
   'detector_argmax_incorrect':int(np.sum(pred!=truth)),'correction_applied_fraction':float(np.mean(pred!=0)),
   'clean_false_correction_count':int(np.sum(pred!=0)) if cond=='clean' else ''})
 oldc=pd.read_csv(OUT/'HUANG_DETECTOR_CONFUSION_BY_CONDITION.csv');oldf=pd.read_csv(OUT/'HUANG_FALSE_CORRECTION_AUDIT.csv')
 oldc=oldc[oldc.source=='HEEDB'];oldf=oldf[oldf.source=='HEEDB']
 pd.concat([oldc,pd.DataFrame(conf)],ignore_index=True).to_csv(OUT/'HUANG_DETECTOR_CONFUSION_BY_CONDITION.csv',index=False)
 pd.concat([oldf,pd.DataFrame(false)],ignore_index=True).to_csv(OUT/'HUANG_FALSE_CORRECTION_AUDIT.csv',index=False)
 # Explicit source/status binding summary for the new handoff audit.
 hand=json.loads((OUT/'INPUT_HANDOFF_ENGINEERING_AUDIT.json').read_text(encoding='utf-8'))
 detector=json.loads((OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 replay=json.loads((OUT/'REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 write_json(OUT/'INPUT_HANDOFF_AUDIT.json',{'status':'PASS','engineering':hand,'detector_cached_probability_replay':detector,
  'per_unit_batch_handoffs':[{'method':r['method'],'seed':r['seed'],'condition':r['condition'],'path':r['path'],'sha256':r['sha256'],'blocks':r['block_input_records']} for r in inference['new_files']],
  'corrector_received_the_same_incoming_faulty_waveform_hash_per_batch':True,'predicted_class_source':'argmax frozen detector cache only','oracle_class_used':False})
 print(json.dumps({'coverage':len(cover),'unaffected_audit':'PASS','changed_summary_files':['HUANG_DETECTOR_CONFUSION_BY_CONDITION.csv','HUANG_FALSE_CORRECTION_AUDIT.csv'],'input_handoff':'PASS'}))
if __name__=='__main__':main()
