import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
files=[
 'outputs/calibration_role_control_v1/FAILED_ATTEMPT_001.json','outputs/calibration_role_control_v1/AMENDMENT_001.md',
 'outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL_FAILED_001.json','outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL_FAILED_001_SHA256.txt',
 'outputs/calibration_role_control_v1/FAILED_ATTEMPT_002.json','outputs/calibration_role_control_v1/AMENDMENT_002.md',
 'outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL_FAILED_002.json','outputs/calibration_role_control_v1/FROZEN_EXECUTION_PROTOCOL_FAILED_002_SHA256.txt',
 'NEXT_RESEARCH_DECISIONS_20261019.md','LUNA_PROMPT_CALIBRATION_CONTROL_20261019.md',
 'outputs/calibration_independence_preflight_v1/REVIEW_NOTES_20261019.md',
 'outputs/calibration_independence_preflight_v1/DRAFT_CONTROL_PROTOCOL.json',
 'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv',
 'outputs/calibration_independence_preflight_v1/SPLIT_COUNTS.csv',
 'outputs/calibration_independence_preflight_v1/IDENTITY_AND_LEAKAGE_AUDIT.json',
 'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json','outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json',
 'outputs/ningbo_confirmation_v2/SPLIT_LEAKAGE_PROOF.json','outputs/ningbo_confirmation_v2/bootstrap_indices.npy',
 'outputs/ningbo_confirmation_v2/seed_104729/preprocessing.npz','outputs/ningbo_confirmation_v2/seed_130363/preprocessing.npz','outputs/ningbo_confirmation_v2/seed_155921/preprocessing.npz',
 'outputs/mitigation_m2_ningbo_v1/seed_104729/preprocessing.npz','outputs/mitigation_m2_ningbo_v1/seed_130363/preprocessing.npz','outputs/mitigation_m2_ningbo_v1/seed_155921/preprocessing.npz',
 'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json','configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json',
 'src/chapman_phase1.py','src/ecg_physics.py','src/ningbo_confirmation_v1.py','src/mask_semantics_recovery_v2.py','src/mitigation_m2_ningbo_v1.py',
 'src/heedb_huang_adapter_v1.py','src/huang_detector_infer_v2.py','outputs/three_swap_population_mixture_recovery_v2/REVIEW_NOTES_20261019.md',
 'outputs/three_swap_population_mixture_recovery_v2/recovery_inference.py','outputs/three_swap_population_mixture_recovery_v2/INPUT_HANDOFF_ENGINEERING_AUDIT.json',
 'outputs/three_swap_population_mixture_recovery_v2/DETECTOR_BLOCK_REPLAY_AUDIT.json',
 'outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable/checkpoint'
]
# Resolve TensorFlow checkpoint prefix into the frozen index/data shards.
cp=ROOT/'outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable'
files=[x for x in files if x!='outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable/checkpoint']
det_files=[p.relative_to(ROOT).as_posix() for p in cp.glob('LDenseNet.*') if p.is_file()]
inputs={p:sha(ROOT/p) for p in files if (ROOT/p).is_file()};inputs.update({p:sha(ROOT/p) for p in det_files})
if len(inputs)<30:raise RuntimeError('required source files absent; refusing incomplete freeze')
for p in ('control.py','detector_scores.py','freeze.py'):
 if not (OUT/p).is_file():raise RuntimeError('runner component absent: '+p)
shutil.copy2(ROOT/'NEXT_RESEARCH_DECISIONS_20261019.md',OUT/'ACTIVE_PLAN_CONTEXT_SNAPSHOT.md')
shutil.copy2(ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv',OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv')
shutil.copy2(ROOT/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json',OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')
protocol={
 'status':'FROZEN_BEFORE_TRAINING','created_utc':'2026-10-19','design':'B_NINGBO_RECORD_DISJOINT_CONTROL',
 'scientific_question':'Whether protective method comparisons remain after separate record roles for checkpoint selection, calibration fitting, and threshold selection.',
 'roles':{'train':24430,'model_selection':1744,'calibration_A':872,'threshold_B':872,'Ningbo_test':6981,'HEEDB':10000},
 'membership_file_sha256':inputs['outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv'],
 'membership_sha256_canonical':hashlib.sha256(json.dumps(list(__import__('csv').DictReader((ROOT/'outputs/calibration_independence_preflight_v1/PROPOSED_MEMBERSHIP.csv').open(encoding='utf-8-sig'))),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'membership_seed':20261019,'independence':'Ningbo record/waveform disjoint only; patient IDs not established.',
 'models':{'methods':['ordinary','physics_aware_p050','combined'],'seeds':[104729,130363,155921],'n_fits':9,'architecture':'ECGResNet-small','loss':'train-only pos_weight capped at 20, weighted multilabel BCE','optimizer':'AdamW lr=.001 wd=.0001','batch_size':64,'epochs_max':30,'autocast':'CUDA float16 + GradScaler','selection':'model_selection only TWA AUPRC, patience 5, strict improvement, earliest tie','gate':{'TWA_AUROC_min':.75,'TWA_AUPRC_min':.45,'finite':True,'checkpoint_replay':True}},
 'recipes':{'ordinary':'clean 100%','physics_aware_p050':'clean 50%; 50% one uniform RA_LA/RA_LL/LA_LL reversal','combined':'clean 50%; 25% one uniform five-displayed-lead mask; 25% one uniform RA_LA/RA_LL/LA_LL reversal; exclusive'},
 'mask':{'space':'uniform lexicographic C(12,5)=792','indices_removed':5,'remaining':7,'sentinel':'unchanged train per-displayed-lead mean in raw units before normalization'},
 'normalization':'Reuse byte-identical existing train-only mean/std with bound exact source hashes; no validation-derived statistics.',
 'calibration':{'fit':'per method/seed/endpoint Platt on clean calibration_A only','form':'sigmoid(exp(theta0)*logit(clip(p))+b)','clip':[1e-6,.999999],'loss':'unweighted logistic','initial':[0,0],'optimizer':'L-BFGS-B','bounds':[[-12,12],[-30,30]],'maxiter':2000,'ftol':1e-13,'gtol':1e-8,'maxls':50,'threshold':'q90 of negative clean raw scores in threshold_B; numpy linear; raw score >= q90','fits':36,'thresholds':36},
 'huang':'frozen published detector; argmax class inverse applied to same incoming raw faulty waveform; corresponding new ordinary classifier; no oracle; clean correction errors retained.',
 'future_evaluation':{'cohorts':['previous Ningbo test','previous HEEDB I0006'],'conditions':['clean','RA_LL_reversal','RA_LA_reversal','LA_LL_reversal'],'q':[0,.001,.005,.01,.05,.1,1.0],'scenarios':['RA_LL_only','equal_three_swap'],'equal_three_weights':'[1-q,q/3,q/3,q/3]','bootstraps':1000,'unit':{'Ningbo':'record','HEEDB':'patient'},'primary_endpoint':'TWA'},
 'no_new_scope':['no new model family','no p-grid','no new seeds','no 792-mask evaluation','no additional endpoints','no threshold tuning'],
 'historical_caveat':'HEEDB has historical access and is sensitivity control; Ningbo patient IDs absent; old-to-new difference combines checkpoint and sample-size changes.',
 'input_file_sha256':inputs,'runner_sha256':{p:sha(OUT/p) for p in ('control.py','detector_scores.py','freeze.py')},
 'active_plan_snapshot_sha256':sha(OUT/'ACTIVE_PLAN_CONTEXT_SNAPSHOT.md'),'accepted_membership_snapshot_sha256':sha(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv')}
protocol['canonical_fingerprint_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
p=OUT/'FROZEN_EXECUTION_PROTOCOL.json';p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
(OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').write_text(sha(p)+'\n',encoding='ascii')
(OUT/'IMMUTABLE_BINDINGS.json').write_text(json.dumps({'protocol_file_sha256':sha(p),'canonical_fingerprint_sha256':protocol['canonical_fingerprint_sha256'],'runner_sha256':protocol['runner_sha256'],'input_file_sha256':inputs,'split_snapshot_sha256':{'membership':sha(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv'),'original_split':sha(OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json'),'active_plan_context':sha(OUT/'ACTIVE_PLAN_CONTEXT_SNAPSHOT.md')}},indent=2,sort_keys=True)+'\n',encoding='utf8')
print(json.dumps({'protocol_file_sha256':sha(p),'canonical_fingerprint':protocol['canonical_fingerprint_sha256'],'inputs':len(inputs)}))
