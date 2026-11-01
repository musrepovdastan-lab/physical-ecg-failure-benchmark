from __future__ import annotations
import hashlib,json,shutil,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ATTACH=Path(r'C:\Users\User\.codex\attachments\e6d2c713-c5c0-4480-a48b-9aa61b9c4d01\Вставленный текст.txt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 base_path=ROOT/'outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL.json'
 base=json.loads(base_path.read_text(encoding='utf-8'))
 old_plan=base['input_file_sha256'].get('NEXT_RESEARCH_DECISIONS_20261019.md')
 old_prompt=base['input_file_sha256'].get('LUNA_PROMPT_CALIBRATION_CONTROL_20261019.md')
 if not (OUT/'INITIALIZATION_AUDIT.json').is_file() or json.loads((OUT/'INITIALIZATION_AUDIT.json').read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('initialization audit must PASS before freeze')
 if json.loads((OUT/'HISTORICAL_INTEGRITY_AUDIT.json').read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('historical artifact audit must PASS before freeze')
 if json.loads((OUT/'DETECTOR_CACHE_REUSE_AUDIT.json').read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('reused detector cache audit must PASS before freeze')
 if json.loads((OUT/'ENGINEERING_SMOKE_AUDIT.json').read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('engineering smoke must PASS before freeze')
 if json.loads((OUT/'RNG_RESUME_AUDIT.json').read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('RNG resume audit must PASS before freeze')
 new_inputs={}
 for rel in base['input_file_sha256']:
  p=ROOT/rel
  if p.is_file():new_inputs[rel]=sha(p)
 extra=[
  'outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL.json','outputs/calibration_role_control_v1/REPRODUCIBILITY_MANIFEST.json','outputs/calibration_role_control_v1/ALL_MODELS_LOCK.json','outputs/calibration_role_control_v1/CALIBRATION_AND_THRESHOLD_LOCK.csv','outputs/calibration_role_control_v1/CONDITION_METRICS_BY_SEED.csv','outputs/calibration_role_control_v1/MIXTURE_METRICS_BY_SEED.csv','outputs/calibration_role_control_v1/PAIRED_BOOTSTRAP_DRAWS_ALL.npz','outputs/calibration_role_control_v1/Ningbo_test_TWA_PAIRED_BOOTSTRAP_DRAWS.npz','outputs/calibration_role_control_v1/HEEDB_TWA_PAIRED_BOOTSTRAP_DRAWS.npz','outputs/calibration_role_control_v1/DETECTOR_ROLE_BINDINGS.json','outputs/calibration_role_control_v1/DETECTOR_TEST_BINDINGS.json',
  'outputs/calibration_role_control_completion_v2/COMPLETION_PROTOCOL.json','outputs/calibration_role_control_completion_v2/MANIFEST.json','outputs/calibration_role_control_completion_v2/REVIEW_NOTES_20261020.md','outputs/calibration_role_control_completion_v2/FINAL_REPORT_RU.md','outputs/calibration_role_control_completion_v2/PAIRED_POINTS_CI_ALL_700.csv','outputs/calibration_role_control_completion_v2/PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz','outputs/calibration_role_control_completion_v2/BOOTSTRAP_COVERAGE_AUDIT.json',
  'outputs/ningbo_confirmation_v2/bootstrap_indices.npy','outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy','outputs/three_swap_population_mixture_recovery_v2/huang_detector_probabilities.npz','outputs/three_swap_population_mixture_recovery_v2/RECOVERY_PROTOCOL.json','outputs/three_swap_population_mixture_recovery_v2/DETECTOR_BLOCK_REPLAY_AUDIT.json',
  'outputs/calibration_role_control_seed_corrected_v1/AMENDMENT_001.md','outputs/calibration_role_control_seed_corrected_v1/INITIALIZATION_AUDIT.json','outputs/calibration_role_control_seed_corrected_v1/HISTORICAL_INTEGRITY_AUDIT.json','outputs/calibration_role_control_seed_corrected_v1/DETECTOR_CACHE_REUSE_AUDIT.json','outputs/calibration_role_control_seed_corrected_v1/ENGINEERING_SMOKE_AUDIT.json','outputs/calibration_role_control_seed_corrected_v1/RNG_RESUME_AUDIT.json','docs/next_steps.md','docs/RESULTS.md','README.md']
 for rel in extra:
  if not (ROOT/rel).is_file():raise FileNotFoundError(rel)
  new_inputs[rel]=sha(ROOT/rel)
 components=['control.py','postprocess.py','evaluate.py','detector_scores.py','detector_test_scores.py','initialization_audit.py','rng_resume_audit.py','verify_history_integrity.py','verify_reused_detector_caches.py','freeze_seed_corrected.py']
 runner={name:sha(OUT/name) for name in components}
 smoke=json.loads((OUT/'ENGINEERING_SMOKE_AUDIT.json').read_text(encoding='utf-8'))
 if smoke.get('status')!='PASS':raise RuntimeError('deterministic engineering smoke must PASS before freeze')
 new_inputs['outputs/calibration_role_control_seed_corrected_v1/ENGINEERING_SMOKE_AUDIT.json']=sha(OUT/'ENGINEERING_SMOKE_AUDIT.json')
 p=base.copy();p['status']='FROZEN_BEFORE_SEED_CORRECTED_TRAINING';p['created_utc']=datetime.now(timezone.utc).isoformat();p['client_timezone_date']='2026-11-01';p['input_file_sha256']=new_inputs;p['runner_sha256']=runner
 p['randomness_policy']={'seed_order':'Python, NumPy, torch.manual_seed, torch.cuda.manual_seed_all, deterministic backend flags; all before ECGResNetSmall constructor','python_seed':'fit seed','numpy_seed':'fit seed modulo 2^32','torch_cpu_seed':'fit seed','cuda_seeds':'fit seed for every visible CUDA device','deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'tf32_matmul':False,'tf32_cudnn':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8','training_dataloader_generator':'CPU generator seeded with fit seed; state saved/restored each epoch','inference_dataloader_generator':'separate local CPU generator; does not consume training RNG','resume_state':['Python random','NumPy legacy RNG','Torch CPU RNG','all Torch CUDA RNG states','training DataLoader generator','model','optimizer','GradScaler','epoch/history/patience/best score'],'initial_state_digest':'SHA-256 over sorted tensor names, dtype, shape, and contiguous tensor bytes; not torch.save file hash'}
 p['minimum_implementation_differences']=['Correct seed-before-constructor ordering.','Strict deterministic backend configuration and explicit independent DataLoader generators.','Save canonical initial state and RNG snapshots; atomically checkpoint/restore all training RNG state.','Measure training elapsed time and peak CUDA allocation separately.']
 active_plan=ROOT/'docs/next_steps.md'
 p['source_lineage_disclosures']={'prior_plan_hash':old_plan,'prior_root_plan_present':(ROOT/'NEXT_RESEARCH_DECISIONS_20261019.md').is_file(),'current_active_plan_path':'docs/next_steps.md','current_active_plan_sha256':sha(active_plan),'prior_calibration_prompt_hash':old_prompt,'prior_calibration_prompt_present':(ROOT/'LUNA_PROMPT_CALIBRATION_CONTROL_20261019.md').is_file(),'current_attached_instruction_sha256':sha(ATTACH),'historical_artifact_audit_sha256':sha(OUT/'HISTORICAL_INTEGRITY_AUDIT.json'),'detector_cache_reuse_audit_sha256':sha(OUT/'DETECTOR_CACHE_REUSE_AUDIT.json'),'initialization_audit_sha256':sha(OUT/'INITIALIZATION_AUDIT.json'),'rng_resume_audit_sha256':sha(OUT/'RNG_RESUME_AUDIT.json')}
 p['run_limits']={'exact_fits':9,'method_order':['ordinary','physics_aware_p050','combined'],'seeds':[104729,130363,155921],'no_new_seeds_or_parameter_search':True,'no_reserved_test':True,'raw_data_predictions_and_weights_stay_local':True}
 p['comparison_protocol']={'old_series':'calibration_role_control_v1','completed_old_statistics':'calibration_role_control_completion_v2 all 700 paired groups','q':[0,0.001,0.005,0.01,0.05,0.1,1.0],'scenarios':['RA_LL_only','equal_three_swap'],'bootstrap_draws_per_group':1000,'bootstrap_units':{'Ningbo_test':'record','HEEDB':'patient'},'frozen_index_files':['outputs/ningbo_confirmation_v2/bootstrap_indices.npy','outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'],'comparison_claim':'sensitivity/reproducibility on previously examined data; old-to-new delta is not a pure causal effect of initialization-order defect'}
 shutil.copy2(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv',OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv')
 shutil.copy2(ROOT/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json',OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')
 p['frozen_membership_copy_sha256']=sha(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv');p['frozen_split_copy_sha256']=sha(OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')
 p['environment']={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),'torch_deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()}
 dest=OUT/'FROZEN_EXECUTION_PROTOCOL.json';dest.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');(OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').write_text(sha(dest)+'\n',encoding='ascii')
 print(json.dumps({'status':p['status'],'protocol_sha256':sha(dest),'bound_inputs':len(new_inputs),'runner_components':len(runner),'fits':9},sort_keys=True))
if __name__=='__main__':main()
