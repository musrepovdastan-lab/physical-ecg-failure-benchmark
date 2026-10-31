from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = (104729, 130363, 155921)

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    if (OUT/'FROZEN_EXTENSION_PROTOCOL.json').exists():
        raise FileExistsError('Refusing to replace frozen extension protocol')
    nroot=ROOT/'outputs/ningbo_four_arm_replication_v1'
    mroot=ROOT/'outputs/mixed_fault_calibration_v2'
    v1=ROOT/'outputs/mixed_fault_calibration_v1'
    private=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1'
    files=[
      'outputs/heedb_independent_confirmation_v1/COHORT_FREEZE.json',
      'outputs/heedb_independent_confirmation_v1/EVALUATION_FREEZE.json',
      'outputs/heedb_independent_confirmation_v1/INFERENCE_COMPLETE.json',
      'data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv',
      'data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy',
      'data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv',
      'data/heedb_restricted/I0006/independent_confirmation_v1/predictions_float32.npy',
      'data/heedb_restricted/I0006/mixed_fault_calibration_v1/chest_only_predictions_float32.npy',
      'data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy',
      'outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_PROTOCOL.json',
      'outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_AMENDMENT_001.json',
      'outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_AMENDMENT_002.json',
      'outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_AMENDMENT_003.json',
      'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json',
      'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json',
      'outputs/mixed_fault_calibration_v2/BOOTSTRAP_COMPLETION_AUDIT.json',
      'outputs/mixed_fault_calibration_v2/BOOTSTRAP_METRICS_REPLICATES.npz',
      'outputs/mixed_fault_calibration_v1/FROZEN_ANALYSIS_PROTOCOL.json',
      'outputs/mixed_fault_calibration_v1/calibration_split_indices.npz',
      'outputs/mixed_fault_calibration_v1/BOOTSTRAP_METRICS_REPLICATES.npz',
      'outputs/mixed_fault_calibration_v1/run_analysis.py',
      'outputs/mixed_fault_calibration_v1/extend_bootstrap_metrics.py',
      'outputs/ningbo_four_arm_replication_v1/FROZEN_NINGBO_FOUR_ARM_PROTOCOL.json',
      'outputs/ningbo_four_arm_replication_v1/FROZEN_INFERENCE_CONTRACT.json',
      'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json',
      'outputs/ningbo_four_arm_replication_v1/REPRODUCIBILITY_MANIFEST.json',
      'outputs/ningbo_four_arm_replication_v1/infer_new_units.py',
      'outputs/ningbo_four_arm_replication_v1/workflow.py',
      'src/chapman_phase1.py','src/ecg_physics.py','src/heedb_external_v1.py'
    ]
    bindings={}
    for rel in files:
        p=ROOT/rel
        if not p.is_file(): raise FileNotFoundError(rel)
        bindings[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
    units={}
    for arm in ('masking','combined'):
      for seed in SEEDS:
        d=nroot/arm/f'seed_{seed}'
        checkpoint=d/'checkpoint.pt'; prep=d/'preprocessing.npz'; gate=d/'validation_gate.json'; val=d/'validation.npz'
        g=json.loads(gate.read_text(encoding='utf-8'))
        if g.get('decision')!='PASS' or g.get('checkpoint_sha256')!=sha(checkpoint): raise RuntimeError(f'checkpoint gate failed {arm}/{seed}')
        with np.load(prep,allow_pickle=False) as z:
            if z['mean'].shape!=(12,) or z['std'].shape!=(12,) or not np.isfinite(z['mean']).all() or not np.isfinite(z['std']).all() or np.any(z['std']<=0): raise RuntimeError(f'preprocessing invalid {arm}/{seed}')
        units[f'{arm}/{seed}']={'checkpoint_sha256':sha(checkpoint),'preprocessing_sha256':sha(prep),'validation_sha256':sha(val),'gate_sha256':sha(gate),'status':g['decision']}
    # Fixed source-order RNG stream used by mixed_fault_calibration_v2: Ningbo draw(s), then HEEDB.
    rng=np.random.default_rng(20261017)
    rng.integers(0,6981,size=(1000,6981),dtype=np.int64)
    heedb_indices=rng.integers(0,10000,size=(1000,10000),dtype=np.int64)
    idx_path=OUT/'HEEDB_BOOTSTRAP_INDICES.npy'
    with idx_path.open('wb') as f: np.save(f,heedb_indices,allow_pickle=False)
    protocol={
      'version':'heedb_four_arm_extension_v1','status':'FROZEN_BEFORE_NEW_EXTERNAL_METRICS',
      'purpose':'Exploratory extension of previously accessed Emory HEEDB analysis using existing Ningbo masking and combined checkpoints.',
      'scope':{'cohort':'HEEDB_I0006_independent_confirmation_v1','n_patients':10000,'one_record_per_patient':True,'same_selected_record_order_labels_and_transform':True,'new_cohorts':False,'reserved_test':False,'heemask_panel':False,'model_or_parameter_selection_on_HEEDB':False,'new_training':False},
      'cohort':{'cohort_freeze_sha256':bindings['outputs/heedb_independent_confirmation_v1/COHORT_FREEZE.json']['sha256'],'evaluation_freeze_sha256':bindings['outputs/heedb_independent_confirmation_v1/EVALUATION_FREEZE.json']['sha256'],'selection_sha256':bindings['data/heedb_restricted/I0006/independent_confirmation_v1/selection_initial.csv']['sha256'],'signals_sha256':bindings['data/heedb_restricted/I0006/independent_confirmation_v1/signals_10000_float32.npy']['sha256'],'labels_source_sha256':bindings['data/heedb_restricted/I0006/12SL_diagnoses/diagnoses_acquisition.csv']['sha256'],'proxy_label':'codes_physician TWA/SB/STach; existing mapping and ordering unchanged'},
      'models':units,
      'preprocessing':{'input':'frozen HEEDB float32 waveforms in microvolts, order I,II,III,aVR,aVL,aVF,V1-V6; 5000 samples at 500Hz','to_model':'exact existing path: stride-5 decimation to 1000 samples at 100Hz; no other transform; physical RA_LL reversal before normalization','normalization':'checkpoint-specific train mean/std from corresponding frozen preprocessing.npz, raw units then per-lead normalization','inference':'ECGResNetSmall, model.eval(), inference_mode(), CUDA float16 autocast, batch size 128','physical_fault':'RA_LL reversal via existing src.heedb_external_v1.reversal_ra_ll'},
      'conditions':['clean','RA_LL_reversal'],'endpoints':{'primary':'TWA','secondary':['SB','STach']},'q':[0,0.001,0.005,0.01,0.05,0.1,1.0],
      'methods':['ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined'],
      'reuse':{'existing_four_methods':'unchanged verified predictions from mixed_fault_calibration_v2 inputs; Huang is complete frozen detector→predicted correction→diagnostic pipeline','new_outputs':'only masking/combined × 3 seeds × clean/RA_LL; no seed prediction averaging'},
      'calibration':{'masking_combined':'unchanged per-seed/per-endpoint validation-A Platt coefficients and validation-B 90th-percentile thresholds from ningbo_four_arm_replication_v1','others':'unchanged per-seed/per-endpoint Platt coefficients and validation-B thresholds from mixed_fault_calibration_v2','huang_primary':'full-pipeline validation-B 90th-percentile threshold','huang_historical':'retained only as separately named sensitivity mode','HEEDB_fit':False},
      'metrics':['auprc','auroc','brier','ece15','sensitivity','specificity','predicted_positive_fraction'],
      'mixture':'For each record, pair its clean and reversed prediction/label copies with weights 1-q and q; weighted pooled metrics; never average condition AUPRCs.',
      'bootstrap':{'replicates':1000,'unit':'patient','rng':'numpy.default_rng(20261017); exact source-ordered stream from mixed_fault_calibration_v2 (discard Ningbo draws then use HEEDB draws)','indices_file':'HEEDB_BOOTSTRAP_INDICES.npy','indices_sha256':sha(idx_path),'same_multiplicity_clean_reversed':True,'seed_aggregation':'calculate each seed metric within each draw, average three seed metrics, then form paired differences','interval':'percentile 2.5%,97.5%; conditional on fixed checkpoints/calibrators; zero crossing is not equivalence'},
      'contrasts':{'primary':['masking_minus_ordinary_resnet','combined_minus_physics_aware_resnet'],'secondary':['combined_minus_huang_pipeline','combined_minus_chest_only_resnet'],'metrics':['auprc','auroc','brier','sensitivity','specificity'],'modes':['raw','calibrated'],'q':'all frozen q values','multiplicity':'no prespecified adjustment; exploratory'},
      'replay':{'existing_prediction_replay_tolerance':0.0001,'new_ningbo_replay_tolerance':0.0001,'batch_size':128,'tolerance_posthoc_widening':False,'diagnose_before_retry':True},
      'limitations':['HEEDB diagnoses are proxy labels, not clinical adjudication or identical source labels.','HEEDB analyses share Emory and are not independent centers.','This is an extension of prior HEEDB access, not new independent confirmation.','combined has half the physical-reversal exposure of physics-aware; comparison does not isolate mask addition.','Bootstrap intervals condition on selected checkpoints and calibrators; they do not include training uncertainty.'],
      'bound_inputs':bindings
    }
    path=OUT/'FROZEN_EXTENSION_PROTOCOL.json'
    path.write_text(json.dumps(protocol,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    (OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(sha(path)+'\n',encoding='ascii')
    print(json.dumps({'status':protocol['status'],'protocol_sha256':sha(path),'input_files':len(bindings),'new_units':len(units),'bootstrap_indices_sha256':sha(idx_path)},indent=2))

if __name__=='__main__': main()
