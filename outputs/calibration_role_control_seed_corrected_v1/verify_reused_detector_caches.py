from __future__ import annotations
import hashlib,json,csv,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
V1=ROOT/'outputs/calibration_role_control_v1'
N=ROOT/'outputs/ningbo_confirmation_v2'
PREF=ROOT/'outputs/calibration_independence_preflight_v1'
CONDS=['clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal']
CLASSES=['clean','RA_LA','RA_LL','LA_LL']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order_sha(ids):return hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()
def main():
 membership=PREF/'PROPOSED_MEMBERSHIP.csv'
 mrows=list(csv.DictReader(membership.open(encoding='utf-8-sig')))
 source_bind=json.loads((V1/'DETECTOR_ROLE_BINDINGS.json').read_text(encoding='utf-8'))
 if source_bind.get('status')!='PASS' or not source_bind.get('same_source_input_order_verified'):raise RuntimeError('source role detector bindings not PASS')
 canonical=json.loads((N/'CANONICAL_RECORDS.json').read_text(encoding='utf-8'))['records'];by={x['record_id']:x for x in canonical}
 audits=[]
 role_binding={x['role']:x for x in source_bind['artifacts']}
 for role in ('calibration_A','threshold_B'):
  src=V1/'detector'/f'{role}.npz';dst=OUT/'detector'/f'{role}.npz';binding=role_binding[role]
  expected=[by[x['record_id']]['record_id'] for x in mrows if x['role']==role]
  with np.load(src,allow_pickle=False) as z:
   ids=z['record_ids'].astype(str).tolist();conds=z['conditions'].astype(str).tolist();classes=z['class_names'].astype(str).tolist();membership_hash=str(z['membership_sha256'].item());checkpoint=str(z['checkpoint'].item())
   shape=list(z['probabilities'].shape)
  valid=(sha(src)==binding['sha256'] and ids==expected and order_sha(ids)==binding['record_ids_sha256'] and conds==CONDS and classes==CLASSES and shape==[4,872,4] and membership_hash==sha(membership))
  if not valid:raise RuntimeError(f'role cache binding mismatch: {role}')
  shutil.copy2(src,dst)
  if sha(dst)!=sha(src):raise RuntimeError(f'copied cache hash mismatch: {role}')
  audits.append({'role':role,'source':str(src.relative_to(ROOT)).replace('\\','/'),'reused_copy':str(dst.relative_to(ROOT)).replace('\\','/'),'sha256':sha(dst),'record_order_sha256':order_sha(ids),'rows':len(ids),'conditions':conds,'classes':classes,'membership_sha256':membership_hash,'detector_checkpoint':checkpoint,'detector_checkpoint_data_sha256':binding['checkpoint_data_sha256'],'status':'PASS'})
 test_src=V1/'detector/Ningbo_test.npz';test_dst=OUT/'detector/Ningbo_test.npz';test_binding=json.loads((V1/'DETECTOR_TEST_BINDINGS.json').read_text(encoding='utf-8'))
 split=json.loads((N/'SPLIT_MEMBERSHIP.json').read_text(encoding='utf-8'))['rows'];expected=[by[x['record_id']]['record_id'] for x in split if x['split']=='test']
 with np.load(test_src,allow_pickle=False) as z:
  ids=z['record_ids'].astype(str).tolist();conds=z['conditions'].astype(str).tolist();classes=z['class_names'].astype(str).tolist();checkpoint=str(z['checkpoint'].item());shape=list(z['probabilities'].shape)
 valid=(test_binding.get('status')=='PASS' and sha(test_src)==test_binding['prediction_sha256'] and ids==expected and order_sha(ids)==test_binding['record_order_sha256'] and conds==CONDS and classes==CLASSES and shape==[4,6981,4] and test_binding['legacy_order_match'] is True)
 if not valid:raise RuntimeError('Ningbo test detector cache binding mismatch')
 shutil.copy2(test_src,test_dst)
 if sha(test_dst)!=sha(test_src):raise RuntimeError('Ningbo test cache copy hash mismatch')
 shutil.copy2(V1/'DETECTOR_ROLE_BINDINGS.json',OUT/'DETECTOR_ROLE_BINDINGS.json')
 shutil.copy2(V1/'DETECTOR_TEST_BINDINGS.json',OUT/'DETECTOR_TEST_BINDINGS.json')
 audits.append({'role':'Ningbo_test','source':str(test_src.relative_to(ROOT)).replace('\\','/'),'reused_copy':str(test_dst.relative_to(ROOT)).replace('\\','/'),'sha256':sha(test_dst),'record_order_sha256':order_sha(ids),'rows':len(ids),'conditions':conds,'classes':classes,'detector_checkpoint':checkpoint,'status':'PASS'})
 recovery=ROOT/'outputs/three_swap_population_mixture_recovery_v2'
 hfile=recovery/'huang_detector_probabilities.npz';ha=json.loads((recovery/'DETECTOR_BLOCK_REPLAY_AUDIT.json').read_text(encoding='utf-8'))
 with np.load(hfile,allow_pickle=False) as z:
  hconds=z['conditions'].astype(str).tolist();hclasses=z['class_names'].astype(str).tolist();hindex=str(z['detector_index_sha256'].item()) if 'detector_index_sha256' in z.files else ''
 recovery_protocol=json.loads((recovery/'RECOVERY_PROTOCOL.json').read_text(encoding='utf-8'))
 expected_heedb_order=recovery_protocol['source_cohort']['selection_sha256']
 with np.load(hfile,allow_pickle=False) as z:horder=str(z['source_record_order_sha256'].item())
 if ha.get('status')!='PASS' or sha(hfile)!=ha['cached_artifact_sha256'] or hindex!=ha['checkpoint_index_sha256'] or set(hconds)!=set(CONDS) or hclasses!=CLASSES or horder!=expected_heedb_order:
  raise RuntimeError('HEEDB Huang probability cache binding mismatch')
 audits.append({'role':'HEEDB_detector','source':str(hfile.relative_to(ROOT)).replace('\\','/'),'sha256':sha(hfile),'checkpoint_index_sha256':hindex,'source_order_sha256':horder,'conditions_in_cache_order':hconds,'conditions_expected_after_bound_reordering':CONDS,'classes':hclasses,'prior_block_replay_status':ha['status'],'status':'PASS'})
 report={'status':'PASS','reused_detector_caches':audits,'detector_inference_will_not_be_repeated':True}
 (OUT/'DETECTOR_CACHE_REUSE_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','cache_count':len(audits),'rows':{x['role']:x.get('rows') for x in audits}}))
if __name__=='__main__':main()
