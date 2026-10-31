import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 old=ROOT/'outputs/heedb_four_arm_extension_v1'; mf=json.loads((old/'REPRODUCIBILITY_MANIFEST.json').read_text(encoding='utf-8'))
 old_art=mf.get('artifacts',{})
 reused=[]
 sources=[
  ('ordinary_resnet','data/heedb_restricted/I0006/independent_confirmation_v1/predictions_float32.npy','arm index 0; [seed,condition,patient,endpoint]'),
  ('physics_aware_resnet','data/heedb_restricted/I0006/independent_confirmation_v1/predictions_float32.npy','arm index 1; [seed,condition,patient,endpoint]'),
  ('chest_only_resnet','data/heedb_restricted/I0006/mixed_fault_calibration_v1/chest_only_predictions_float32.npy','[seed,condition,patient,endpoint]'),
  ('huang_pipeline','data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy','[seed,condition,patient,endpoint]; outputs of frozen detector→predicted correction→diagnostic cascade'),
 ]
 for method,rel,order in sources:
  p=ROOT/rel;reused.append({'method':method,'path':rel,'sha256':sha(p),'bytes':p.stat().st_size,'binding':order})
 for method in ('masking','combined'):
  for seed in (104729,130363,155921):
   for cond in ('clean','RA_LL_reversal'):
    rel=f'outputs/heedb_four_arm_extension_v1/predictions/{method}/seed_{seed}/{cond}.npz';p=ROOT/rel
    key=f'predictions/{method}/seed_{seed}/{cond}.npz';prior=old_art.get(key,{}).get('sha256') if isinstance(old_art.get(key,{}),dict) else old_art.get(key)
    actual=sha(p)
    if prior and prior!=actual:raise RuntimeError(f'historical output differs from frozen manifest: {rel}')
    reused.append({'method':method,'seed':seed,'condition':cond,'path':rel,'sha256':actual,'bytes':p.stat().st_size,'prior_manifest_sha256':prior})
 obj={'package':'heedb_probability_extension_v1','historical_source_package':'outputs/heedb_four_arm_extension_v1','historical_manifest_sha256':sha(old/'REPRODUCIBILITY_MANIFEST.json'),'historical_protocol_sha256':sha(old/'FROZEN_EXTENSION_PROTOCOL.json'),'huang_known_fault_substitution':False,'legacy_helper_failure_evidence':{'path':'outputs/heedb_four_arm_extension_v1/REPLAY_AUDIT_LEGACY_HELPER_DIAGNOSTIC.json','sha256':sha(old/'REPLAY_AUDIT_LEGACY_HELPER_DIAGNOSTIC.json'),'status':json.loads((old/'REPLAY_AUDIT_LEGACY_HELPER_DIAGNOSTIC.json').read_text(encoding='utf-8'))['status'],'used_in_new_pipeline':False},'reused_predictions':reused}
 (OUT/'REUSED_PREDICTION_BINDINGS.json').write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 inference=json.loads((OUT/'INFERENCE_COMPLETE.json').read_text(encoding='utf-8'))
 progress={'status':'COMPLETE','protocol_sha256':inference['protocol_sha256'],'prediction_count':12,'predictions':inference['predictions'],'elapsed_seconds':inference['elapsed_seconds']}
 (OUT/'INFERENCE_PROGRESS.json').write_text(json.dumps(progress,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 status='''# Статус пакета HEEDB probability extension v1

Статус: **ЗАВЕРШЕНО**. Ограниченное внешнее расширение p010/p025 выполнено на уже анализировавшейся когорте HEEDB (10 000 пациентов); результат не является новой слепой независимой подтверждающей выборкой.

- До нового inference заморожен протокол; поправки 001–002 внесены до replay, 003–005 документируют только пути и аудитные/анализные реализации. Исходные варианты протокола сохранены.
- Bounded replay: 18/18 PASS; максимальное различие 0 при допуске 0.0001.
- HEEDB inference: 12/12 prediction files, 2 arms × 3 seeds × 2 conditions; обучение отсутствовало.
- Анализ: 1 071 seed-specific строки с отдельными score_mode; 357 строк средних; 48 предписанных bootstrap-интервалов и 48 draws.
- Calibration decision audit: 144 строки, 0 несовпадений; calibration provenance и checkpoint/preprocessing hashes сохранены.
- Reserved test не оценивался. Founder и новые модели/настройки не добавлялись.

Итог: `RESULTS_REPORT_RU.md`. Протокол: `FROZEN_EXTENSION_PROTOCOL.json`. Машиночитаемый индекс артефактов: `REPRODUCIBILITY_MANIFEST.json`.
'''
 (OUT/'STATUS.md').write_text(status,encoding='utf-8')
 # Hash every package artifact except the manifest being written.
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='REPRODUCIBILITY_MANIFEST.json' and '.partial' not in p.name:
   files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest={'package':'heedb_probability_extension_v1','status':'COMPLETE','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'inference_protocol_sha256':inference['protocol_sha256'],'analysis_complete_sha256':sha(OUT/'ANALYSIS_COMPLETE.json'),'scope_disclosure':'Previously analyzed HEEDB cohort; extension comparison, not new blind independent confirmation. No new training. Founder excluded.','counts':{'new_prediction_files':12,'methods':8,'endpoints':3,'q_values':7,'seed_specific_metric_rows':1071,'mean_summary_rows':357,'prespecified_bootstrap_contrast_rows':48,'bootstrap_draws_per_contrast':1000,'replay_rows':18,'calibration_audit_rows':144},'historical_reuse_bindings_sha256':sha(OUT/'REUSED_PREDICTION_BINDINGS.json'),'files':files}
 (OUT/'REPRODUCIBILITY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'status':'COMPLETE','files':len(files),'manifest_sha256':sha(OUT/'REPRODUCIBILITY_MANIFEST.json')},indent=2))
if __name__=='__main__':main()
