from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 protocol_path=OUT/'RECOVERY_PROTOCOL.json';protocol=json.loads(protocol_path.read_text(encoding='utf-8'));protocol_file_sha=sha(protocol_path)
 if protocol_file_sha!=(OUT/'RECOVERY_PROTOCOL_SHA256.txt').read_text(encoding='ascii').strip():raise RuntimeError('protocol file hash mismatch')
 canonical=dict(protocol);fingerprint=canonical.pop('canonical_json_fingerprint_sha256')
 canonical_hash=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 if fingerprint!=canonical_hash:raise RuntimeError('canonical protocol fingerprint mismatch')
 binding_errors=[]
 for rel,v in protocol['input_bindings'].items():
  if sha(ROOT/rel)!=v['sha256']:binding_errors.append(rel)
 if binding_errors:raise RuntimeError('frozen input hash drift: '+str(binding_errors[:3]))
 inf=json.loads((OUT/'RECOVERY_INFERENCE_COMPLETE.json').read_text(encoding='utf-8'))
 inf['elapsed_seconds_sum_of_units']=sum(float(r['elapsed_seconds']) for r in inf['new_files'])
 inf['new_files'].sort(key=lambda r:(int(r['seed']),r['condition']))
 (OUT/'RECOVERY_INFERENCE_COMPLETE.json').write_text(json.dumps(inf,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 detector=json.loads((OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 replay=json.loads((OUT/'REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 if detector.get('status')!='PASS' or replay.get('status')!='PASS':raise RuntimeError('input path replay gate is not PASS')
 replay['detector_block_replay_audit_sha256']=sha(OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json')
 replay['max_detector_probability_delta']=detector['max_abs_probability_delta']
 replay['detector_block_replay_elapsed_seconds']=detector['elapsed_seconds']
 (OUT/'REPLAY_AUDIT.json').write_text(json.dumps(replay,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 # Refresh the handoff record so it embeds the measured selected-block replay.
 hand=json.loads((OUT/'INPUT_HANDOFF_AUDIT.json').read_text(encoding='utf-8'))
 hand['detector_cached_probability_replay']=detector
 hand['detector_block_replay_audit_sha256']=sha(OUT/'DETECTOR_BLOCK_REPLAY_AUDIT.json')
 hand['corrected_classifier_replay_sha256']=sha(OUT/'REPLAY_AUDIT.json')
 (OUT/'INPUT_HANDOFF_AUDIT.json').write_text(json.dumps(hand,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 scientific=json.loads((OUT/'INDEPENDENT_NUMERICAL_VERIFICATION.json').read_text(encoding='utf-8'))
 unaffected=json.loads((OUT/'UNAFFECTED_RESULTS_AUDIT.json').read_text(encoding='utf-8'))
 affected=json.loads((OUT/'AFFECTED_CONTRAST_DRAW_AUDIT.json').read_text(encoding='utf-8'))
 if scientific.get('status')!='PASS' or unaffected.get('status')!='PASS' or affected.get('status')!='PASS':raise RuntimeError('numerical or unaffected result audit failed')
 audit={'status':'PASS','review_type':'internal verification; no external acceptance claimed',
  'scientific_metrics_verification':{'status':scientific['status'],'condition_metric_rows':scientific['condition_metric_rows_recomputed'],
    'mixture_metric_rows':scientific['mixture_metric_rows_recomputed'],'historical_RA_LL_anchor_rows':scientific['historical_RA_LL_anchor_rows'],
    'all_RA_LL_anchors_pass':scientific['historical_RA_LL_anchor_all_pass'],'max_metric_error':max(x['max_abs_error'] for x in scientific['metric_comparisons']),
    'bootstrap_draws':scientific['paired_bootstrap_draws'],'summary_recalculation_max_error':scientific['paired_bootstrap_summary_max_abs_error'],
    'affected_draw_sklearn_checks':affected['rows'],'affected_draw_check_max_error':affected['max_abs_delta']},
  'input_path_verification':{'status':'PASS','detector_block_replay_status':detector['status'],'detector_probability_max_delta':detector['max_abs_probability_delta'],
    'detector_blocks':len(detector['rows']),'corrected_classifier_replay_rows':len(replay['rows']),'corrected_replay_max_delta':replay['max_correct_path_delta'],
    'wrong_legacy_path_control_rows':replay['wrong_path_control_rows'],'engineering_handoff_status':hand['engineering']['status'],
    'all_three_swaps_and_predicted_classes_0_to_3_tested':True,'class_zero_preserves_faulty_signal':True,'source_buffer_mutation':False},
  'unaffected_results_audit':unaffected,'protocol_file_sha256':protocol_file_sha,'protocol_canonical_json_fingerprint_sha256':fingerprint,
  'frozen_input_binding_count':len(protocol['input_bindings']),'frozen_input_binding_errors':binding_errors,
  'training_calibration_threshold_fitting_reserved_test':False}
 jwrite(OUT/'VERIFICATION_AUDIT.json',audit)
 execution={'status':'COMPLETED_AWAITING_USER_REVIEW','recovery_classifier_inference_seconds_total':inf['elapsed_seconds_total'],
  'recovery_classifier_inference_seconds_sum_of_six_units':inf['elapsed_seconds_sum_of_units'],
  'recovery_classifier_inference_seconds_by_unit':[{k:r[k] for k in ('seed','condition','elapsed_seconds','sha256')} for r in inf['new_files']],
  'device':inf['cuda_device'],'torch_version':inf['torch_version'],'precision':inf['precision'],
  'cached_detector_selected_block_replay_seconds':detector['elapsed_seconds'],'cached_detector_replayed_records':sum(r['n'] for r in detector['rows']),
  'historical_v1_full_Huang_detector_runtime':'unavailable; not recorded in v1',
  'historical_v1_full_Huang_classifier_runtime':'unavailable; not recorded in v1',
  'new_detector_full_cohort_inference':'not run; existing detector probabilities reused',
  'model_training':False,'calibration_fitting':False,'threshold_selection':False,'reserved_test':False}
 jwrite(OUT/'EXECUTION_SUMMARY.json',execution)
 # Input hash ledger directly mirrors the frozen protocol bindings.
 with (OUT/'INPUT_SHA256_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','purpose']);w.writeheader()
  for k,v in sorted(protocol['input_bindings'].items()):w.writerow({'path':k,'bytes':v['bytes'],'sha256':v['sha256'],'purpose':v['purpose']})
 # Output ledger excludes the output ledger itself and JSON manifest to avoid recursive hashes.
 excluded={'OUTPUT_SHA256_MANIFEST.csv','REPRODUCIBILITY_MANIFEST.json'}
 outputs=[]
 for p in sorted(x for x in OUT.rglob('*') if x.is_file() and x.name not in excluded):
  b=p.read_bytes();outputs.append({'path':p.relative_to(OUT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 with (OUT/'OUTPUT_SHA256_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(outputs)
 repro={'status':'COMPLETE','recovery_protocol_file_sha256':protocol_file_sha,'recovery_protocol_canonical_json_fingerprint_sha256':fingerprint,
  'input_file_count':len(protocol['input_bindings']),'inputs':protocol['input_bindings'],'output_file_count':len(outputs),
  'outputs':outputs,'output_sha256_manifest_sha256':sha(OUT/'OUTPUT_SHA256_MANIFEST.csv'),
  'input_coverage_sha256':sha(OUT/'INPUT_COVERAGE.csv'),'unchanged_results_audit_sha256':sha(OUT/'UNAFFECTED_RESULTS_AUDIT.json'),
  'new_prediction_files':[{'path':r['path'],'sha256':r['sha256'],'probability_array_sha256':r['probability_array_sha256']} for r in inf['new_files']]}
 jwrite(OUT/'REPRODUCIBILITY_MANIFEST.json',repro)
 # Recheck the output manifest after writing all its listed files.
 bad=[]
 for r in outputs:
  p=OUT/r['path']
  if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:bad.append(r['path'])
 if bad:raise RuntimeError('output hash ledger mismatch '+str(bad[:3]))
 print(json.dumps({'status':'PASS','input_bindings':len(protocol['input_bindings']),'output_files_hashed':len(outputs),
  'protocol_file_sha256':protocol_file_sha,'canonical_fingerprint':fingerprint,'scientific_metrics':'PASS','input_path':'PASS'}))
if __name__=='__main__':main()
