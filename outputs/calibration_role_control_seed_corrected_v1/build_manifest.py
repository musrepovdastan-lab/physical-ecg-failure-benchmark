"""Build a curated, explicit manifest. No recursive directory discovery."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TARGET=OUT/'REPRODUCIBILITY_MANIFEST.json'
METHODS=('ordinary','physics_aware_p050','combined')
ALL_METHODS=(*METHODS,'Huang_full_pipeline')
SEEDS=(104729,130363,155921)
CONDITIONS=('clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')
TOP_LEVEL=(
 'ALL_MODELS_LOCK.json','AMENDMENT_001.md','AMENDMENT_002.md','AMENDMENT_003.md','AMENDMENT_004.md',
 'AP_LEADERSHIP_OLD_VS_CORRECTED.csv','audit_compare.py','BEST_CHECKPOINT_REPLAY_AUDIT.json',
 'BOOTSTRAP_ARCHIVE_REPACK_VERIFICATION.json','build_manifest.py','CALIBRATION_AND_THRESHOLD_LOCK.csv',
 'CALIBRATION_AND_THRESHOLD_LOCK.json','CONDITION_METRICS_BY_SEED.csv','CONDITION_METRICS_MEAN_SD.csv',
 'CONTRAST_OLD_VS_CORRECTED_700.csv','control.py','control_replay.py','DETECTOR_CACHE_REUSE_AUDIT.json',
 'DETECTOR_ROLE_BINDINGS_REUSED.json','DETECTOR_ROLE_BINDINGS.json','DETECTOR_TEST_BINDINGS_REUSED.json',
 'DETECTOR_TEST_BINDINGS.json','detector_scores.py','detector_test_scores.py','ENGINEERING_SMOKE_AUDIT.json',
 'evaluate.py','evaluate_full.py','EVALUATION_PROGRESS.json','FAILED_ATTEMPT_BOOTSTRAP_ARCHIVE.json',
 'FAILED_ATTEMPT_HUANG_FORMAT.json','FAILED_ATTEMPT_POSTFIT_WRITER.json','FINAL_REPORT_RU.md','FOLLOWUP_REPORT_RU.md',
 'freeze_seed_corrected.py','FROZEN_EXECUTION_PROTOCOL.json','FROZEN_EXECUTION_PROTOCOL_SHA256.txt',
 'HEEDB_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv','HEEDB_PREDICTIONS_COMPLETE.json','HEEDB_TWA_PAIRED_BOOTSTRAP_DRAWS.npz',
 'HISTORICAL_COMPARISON.csv','HISTORICAL_INTEGRITY_AUDIT.json','INDEPENDENT_RESULT_AUDIT.json',
 'INITIALIZATION_AUDIT.json','initialization_audit.py','INPUT_VALIDATION_AMENDMENT_001.json','INPUT_VALIDATION_AMENDMENT_001.md',
 'INPUT_VALIDATION_AMENDMENT_001_SHA256.txt','KEY_AP_FINDINGS.csv','MEAN_OLD_VS_CORRECTED.csv',
 'MIXTURE_METRICS_BY_SEED.csv','MIXTURE_METRICS_MEAN_SD.csv','NINGBO_SPLIT_MEMBERSHIP_FROZEN.json',
 'Ningbo_test_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv','Ningbo_test_PREDICTIONS_COMPLETE.json',
 'Ningbo_test_TWA_PAIRED_BOOTSTRAP_DRAWS.npz','PAIRED_BOOTSTRAP_DRAWS_ALL.npz',
 'POSTFIT_RECOVERY_AMENDMENT_001.md','POSTFIT_RECOVERY_AUDIT.json','postprocess.py',
 'PROPOSED_MEMBERSHIP_FROZEN.csv','REPRODUCTION_INPUT_CHECK.json',
 'RNG_RESUME_AUDIT.json','rng_resume_audit.py','ROLE_ACCESS_AUDIT.json','SEED_LEVEL_OLD_VS_CORRECTED.csv',
 'STATUS.json','STATUS.md','train_all_fresh.py','TRAINING_REPORT_RU.md','TRAINING_STATUS.json',
 'TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv','verify_archive_repack.py','verify_history_integrity.py',
 'verify_reused_detector_caches.py','verify_input_provenance.py','SUPPORTED_REPRODUCTION.md',
 'DOCUMENTATION_PUBLICATION_AMENDMENT_002.json'
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    relative=set(TOP_LEVEL)
    for method in METHODS:
        for seed in SEEDS:
            base=f'fits/{method}/seed_{seed}'
            relative.update(f'{base}/{name}' for name in ('best.pt','checkpoint.pt','fit_config.json','history.json',
                'INITIALIZATION_AUDIT.json','initialization_state.pt','preprocessing_provenance.json','preprocessing.npz',
                'resume.pt','training_gate.json'))
            for role in ('model_selection','calibration_A','threshold_B'):
                relative.add(f'scores/{role}/{method}/seed_{seed}.npz')
    for seed in SEEDS:
        for role in ('calibration_A','threshold_B'):
            relative.add(f'scores/{role}/Huang_full_pipeline/seed_{seed}.npz')
    for cohort in ('Ningbo_test','HEEDB'):
        for method in ALL_METHODS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    relative.add(f'predictions/{cohort}/{method}/seed_{seed}/{condition}.npz')
    relative.update(f'detector/{name}.npz' for name in ('calibration_A','threshold_B','Ningbo_test'))
    relative.update(f'frozen_input_snapshots/{name}' for name in ('README.md','docs/RESULTS.md','docs/next_steps.md'))
    relative.update(f'documentation_before_publication_20261101/{name}' for name in
        ('README.md','docs/RESULTS.md','docs/next_steps.md',
         'INPUT_VALIDATION_AMENDMENT_001.json','INPUT_VALIDATION_AMENDMENT_001_SHA256.txt'))
    missing=sorted(x for x in relative if not (OUT/x).is_file())
    if missing:raise RuntimeError('required manifest files missing: '+', '.join(missing[:10]))
    entries=[]
    for rel in sorted(relative):
        path=OUT/rel
        local=(rel.startswith(('fits/','predictions/','scores/','detector/')) or
            rel in {'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json','PROPOSED_MEMBERSHIP_FROZEN.csv'})
        entries.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path),
            'local_only':local,'kind':'model_or_individual_artifact' if local else 'protocol_code_or_aggregate'})
    protocol=json.loads((OUT/'FROZEN_EXECUTION_PROTOCOL.json').read_text(encoding='utf8'))
    amendment=json.loads((OUT/'INPUT_VALIDATION_AMENDMENT_001.json').read_text(encoding='utf8'))
    docs={rel:{'current_sha256':sha(ROOT/rel),'original_frozen_sha256':protocol['input_file_sha256'][rel],
       'historical_snapshot_sha256':sha(OUT/amendment['historical_document_snapshots'][rel]['snapshot_path'])}
       for rel in amendment['historical_document_snapshots']}
    corrected={'control_replay.py':sha(OUT/'control_replay.py'),'train_all_fresh.py':sha(OUT/'train_all_fresh.py'),
       'evaluate_full.py':sha(OUT/'evaluate_full.py'),'verify_archive_repack.py':sha(OUT/'verify_archive_repack.py'),
       'verify_input_provenance.py':sha(OUT/'verify_input_provenance.py'),'build_manifest.py':sha(OUT/'build_manifest.py')}
    manifest={'manifest_version':3,'status':'COMPLETE_REVIEWED','generated_utc':datetime.now(timezone.utc).isoformat(),
      'package':'calibration_role_control_seed_corrected_v1','explicit_required_file_count':len(entries),
      'files':entries,'frozen_protocol_sha256':sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json'),
      'method_fingerprint_sha256':protocol['canonical_fingerprint_sha256'],
      'frozen_training_runner_sha256':protocol['runner_sha256']['control.py'],
      'frozen_runner_hash_bindings':protocol['runner_sha256'],
      'corrected_reproduction_tool_hashes':corrected,
      'input_validation_amendment_sha256':sha(OUT/'INPUT_VALIDATION_AMENDMENT_001.json'),
      'documentation_publication_amendment_sha256':sha(OUT/'DOCUMENTATION_PUBLICATION_AMENDMENT_002.json'),
      'historical_document_provenance':docs,
      'historical_integrity_audit':json.loads((OUT/'HISTORICAL_INTEGRITY_AUDIT.json').read_text(encoding='utf8')),
      'initialization_audit_status':json.loads((OUT/'INITIALIZATION_AUDIT.json').read_text(encoding='utf8'))['status'],
      'independent_result_audit':json.loads((OUT/'INDEPENDENT_RESULT_AUDIT.json').read_text(encoding='utf8')),
      'historical_artifacts_modified':False,'raw_ecg_individual_predictions_and_weights_are_local_only':True,
      'excluded_from_manifest':['**/__pycache__/**','**/*.pyc','**/*.tmp*','failed_attempts/**',
        'partial_bootstrap_first_three_q/**','outputs/seed_corrected_review_20261101/**',
        'review attachments and transient repack temporary directory']}
    TARGET.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'required_files':len(entries),'manifest_sha256':sha(TARGET)}))
if __name__=='__main__':main()
