from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEEDS=(104729,130363,155921)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 if (OUT/'FROZEN_EXTENSION_PROTOCOL.json').exists(): raise FileExistsError('protocol already frozen')
 old=json.loads((ROOT/'outputs/heedb_four_arm_extension_v1/FROZEN_EXTENSION_PROTOCOL.json').read_text(encoding='utf-8'))
 recovery=json.loads((ROOT/'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json').read_text(encoding='utf-8'))
 bindings=dict(old['bound_inputs'])
 rels=[
  'outputs/heedb_four_arm_extension_v1/FROZEN_EXTENSION_PROTOCOL.json',
  'outputs/heedb_four_arm_extension_v1/REPLAY_AUDIT.json',
  'outputs/heedb_four_arm_extension_v1/REPRODUCIBILITY_MANIFEST.json',
  'outputs/heedb_four_arm_extension_v1/extension.py',
  'outputs/augmentation_probability_recovery_v2/REVIEW_NOTES_20261018.md',
  'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json',
  'outputs/augmentation_probability_recovery_v2/INDEPENDENT_REVIEW_AUDIT_20261018.json',
  'outputs/augmentation_probability_study_v1/FROZEN_PROTOCOL.json',
  'outputs/augmentation_probability_study_v1/RUN_MANIFEST.json',
  'outputs/augmentation_probability_study_v1/FINAL_PROBABILITY_STUDY_REPORT_RU.md',
  'outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy',
  'outputs/heedb_probability_extension_v1/extension.py',
  'outputs/heedb_probability_extension_v1/freeze_protocol.py',
 ]
 for arm in ('physics_p010','physics_p025'):
  for seed in SEEDS:
   d=f'outputs/augmentation_probability_study_v1/training/Ningbo/{arm}/seed_{seed}'
   for f in ('checkpoint.pt','best.pt','validation_gate.json','validation.npz'):
    rel=f'{d}/{f}'; rels.append(rel)
   rels.append(f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz')
   rels.extend([f'outputs/augmentation_probability_study_v1/predictions/Ningbo/{arm}/seed_{seed}/clean.npz',f'outputs/augmentation_probability_study_v1/predictions/Ningbo/{arm}/seed_{seed}/RA_LL_reversal.npz'])
 for rel in rels:
  p=ROOT/rel
  if not p.is_file(): raise FileNotFoundError(rel)
  bindings[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
 models={}
 for arm in ('physics_p010','physics_p025'):
  for seed in SEEDS:
   d=ROOT/f'outputs/augmentation_probability_study_v1/training/Ningbo/{arm}/seed_{seed}'
   import torch
   ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True)
   best=torch.load(d/'best.pt',map_location='cpu',weights_only=True)
   if ck.get('arm')!=arm or int(ck.get('seed'))!=seed or int(ck.get('best_epoch'))!=int(best.get('epoch')): raise RuntimeError(f'checkpoint identity mismatch: {arm}/{seed}')
   if not all(torch.equal(ck['state_dict'][k],best['state_dict'][k]) for k in best['state_dict']): raise RuntimeError(f'checkpoint is not selected best: {arm}/{seed}')
   gate=json.loads((d/'validation_gate.json').read_text(encoding='utf-8'))
   if gate.get('decision')!='PASS' or gate.get('checkpoint_sha256')!=sha(d/'checkpoint.pt'): raise RuntimeError(f'checkpoint gate failed {arm}/{seed}')
   prep=ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz'
   with __import__('numpy').load(prep,allow_pickle=False) as z:
    if z['mean'].shape!=(12,) or z['std'].shape!=(12,) or (__import__('numpy').any(z['std']<=0)): raise RuntimeError('preprocessing invalid')
   models[f'{arm}/{seed}']={'checkpoint_sha256':sha(d/'checkpoint.pt'),'best_pt_sha256':sha(d/'best.pt'),'preprocessing_sha256':sha(prep),'validation_sha256':sha(d/'validation.npz'),'gate_sha256':sha(d/'validation_gate.json'),'selected_epoch':int(best['epoch']),'checkpoint_equals_selected_best_state':True}
 # Bootstrap indices are byte-identical to the accepted historical source-order draws.
 src=ROOT/'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'; dst=OUT/'HEEDB_BOOTSTRAP_INDICES.npy'
 shutil.copyfile(src,dst)
 if sha(src)!=sha(dst): raise RuntimeError('bootstrap index copy changed')
 protocol={
  'version':'heedb_probability_extension_v1','status':'FROZEN_BEFORE_NEW_EXTERNAL_RESULTS',
  'purpose':'Exploratory extension comparison of already-trained Ningbo ResNet physics_p010 and physics_p025 on the previously analyzed HEEDB 10,000-patient cohort; not a new blind independent confirmatory set.',
  'scope':{'cohort':'HEEDB_I0006_independent_confirmation_v1','n_patients':10000,'one_record_per_patient':True,'conditions':['clean','RA_LL_reversal'],'seeds':list(SEEDS),'new_training':False,'new_cohort':False,'reserved_test_evaluated':False,'founder_included':False,'parameter_or_model_selection_on_HEEDB':False,'existing_signals_order_proxy_labels_exclusions_unchanged':True},
  'methods':['ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined','physics_p010','physics_p025'],
  'new_models':models,
  'preprocessing':{'input':'Existing frozen HEEDB float32 waveform in microvolts, 12-lead order I,II,III,aVR,aVL,aVF,V1-V6, 5000 samples at 500Hz','downsampling':'stride-5 decimation to 1000 samples at 100Hz','scaling':'raw microvolt units, checkpoint-specific Ningbo train-derived per-lead mean/std','fault':'src.heedb_external_v1.reversal_ra_ll applied physically before normalization','model':'ECGResNetSmall eval/inference_mode, CUDA float16 autocast, batch 128'},
  'replay':{'existing_baseline':'all three seeds, clean and RA_LL_reversal, first 256 canonical HEEDB records, original 128 batch context against saved baseline predictions','new_models':'all six models, clean and RA_LL_reversal, first 256 canonical Ningbo test records against saved test probabilities','tolerance_max_abs_probability_delta':0.0001,'tolerance_frozen_before_comparison':True,'posthoc_widening':False},
  'calibration':{'source':'outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json','fit_split':'frozen calibration A','threshold':'raw negative q90 from frozen validation B; calibrated threshold is the same positive-slope Platt map applied to that raw threshold','HEEDB_refit_or_threshold_selection':False,'full_validation_checkpoint_selection_disclosed':True,'probability_study_v1_validation_or_calibration_assets_used':False},
  'mixture':{'q_fault':[0,0.001,0.005,0.01,0.05,0.1,1.0],'scheme':'paired copies of each patient record, clean weight 1-q and reversal weight q; weighted pooled metrics; no averaging of condition AP'},
  'metrics':{'endpoints':['TWA','SB','STach'],'score_modes':['raw','calibrated'],'metrics':['AUPRC','AUROC','Brier','ECE15','sensitivity','specificity','predicted_positive_fraction'],'thresholds':'validation-B raw q90 and its same-calibrator transform'},
  'contrasts':{'endpoint':'TWA','methods':['physics_p025-physics_aware_resnet','physics_p025-combined','physics_p010-ordinary_resnet','physics_p010-physics_aware_resnet','physics_p010-huang_pipeline','physics_p025-huang_pipeline'],'metrics_and_modes':['raw AUPRC','calibrated Brier'],'q_fault':[0,0.01,0.1,1.0],'bootstrap_replicates':1000,'indices_file':'HEEDB_BOOTSTRAP_INDICES.npy','indices_sha256':sha(dst),'unit':'patient','same_patient_multiplicity_for_clean_and_reversal':True,'seed_specific_metrics_then_mean_within_draw_then_contrast':True,'full_cohort_point_separate_from_bootstrap_mean':True,'interval':'percentile 2.5%,97.5%; zero-crossing does not establish equivalence'},
  'historical_reuse':{'six_methods':'reuse existing verified ordinary, p050, masking, combined, chest-only and full Huang predictions from heedb_four_arm_extension_v1 unchanged','huang':'retain real detector-to-predicted-correction cascade; no known-fault substitution'},
  'limitations':['HEEDB cohort was analyzed before; this is an extension comparison, not a new blind independent confirmation.','Proxy labels are unchanged and are not clinical adjudication.','Full validation influenced checkpoint selection before calibration A/B split.','Bootstrap intervals condition on selected checkpoints and calibration; they do not represent all training uncertainty.','No optimal p or q will be selected on HEEDB.','AUPRC restoration is not an increase in the count of detected patients.'],
  'bound_inputs':bindings
 }
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json';p.write_text(json.dumps(protocol,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 (OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(sha(p)+'\n',encoding='ascii')
 print(json.dumps({'status':protocol['status'],'protocol_sha256':sha(p),'inputs':len(bindings),'new_models':len(models),'bootstrap_sha256':sha(dst)},indent=2))
if __name__=='__main__':main()
