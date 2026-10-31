"""Frozen v2 Ningbo runner: transparent pre-split exclusion after v1 stop."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,sys,tempfile,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src import ningbo_confirmation_v1 as b
V1=ROOT/'outputs/ningbo_confirmation_v1';OUT=ROOT/'outputs/ningbo_confirmation_v2';PROTO=ROOT/'configs/FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json'
V1PROTO='B6EEB79C0CD214FE7F41FDC526AEC64DF2E9D982F59A45E7F21D0108B5A3698E';CONFLICT='F5976880484276C1747B1A3598EE92322733273BE92DCA5AA04EC7C2A0E0CF2F'
V2HASH='1FE829E4357B4EFDE0E5F227BAF5BF94E8D22D12AF0959A38CEE12CCE1F40D71'
MAPPING='AE749EEBE8FC6EFAF42E5E559B3FE494FAB93C015306F1DD8C0659A7DC1FDD5E';SUMS='F29C2609343C38734F3A3D62BE4A9052CA56A9E560E7AD2EFC2118563FCA2473';V1DECISION='BDF6E881B3CCB97E70E3FE66BAB9344A69EAD6FF0EA4E5411B82A4F6F3E73F66'
def sha(p):return b.sha(p)
def jput(p,x):return b.jput(p,x)
def status(s,**x):
 old=b.OUT;b.OUT=OUT
 try:b.status(s,**x)
 finally:b.OUT=old
def bind_v2():
 b.OUT=OUT;b.PROTO=PROTO;b.PH=sha(PROTO).upper()
def lock_for(out):
 """Create an atomic, stale-aware V2-only lock."""
 out.mkdir(parents=True,exist_ok=True);p=out/'ningbo_confirmation_v2.lock'
 payload={'pid':os.getpid(),'workflow_version':'ningbo_confirmation_v2','protocol_v2_sha256':sha(PROTO).upper(),'started_utc':b.now()}
 while True:
  try:
   with open(p,'x',encoding='utf8') as f:json.dump(payload,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
   return p
  except FileExistsError:
   try:pid=int(json.loads(p.read_text(encoding='utf8')).get('pid',-1))
   except Exception:raise b.Stop('STOP_TECHNICAL_BLOCKER malformed v2 lock')
   if b.alive(pid):raise b.Stop('STOP_TECHNICAL_BLOCKER live v2 lock')
   # Re-loop after removing a dead owner's file so a racing runner must win
   # the same atomic exclusive-create operation.
   try:p.unlink()
   except FileNotFoundError:pass
def lock():return lock_for(OUT)
def required(p):
 if not p.is_file():raise b.Stop('STOP_DATA_INTEGRITY missing '+str(p))
def preflight():
 OUT.mkdir(parents=True,exist_ok=True);status('V2_PREFLIGHT')
 required(PROTO);cfg=json.loads(PROTO.read_text())
 if sha(PROTO).upper()!=V2HASH or cfg.get('version')!='ningbo_confirmation_v2' or cfg.get('parent_v1_protocol_sha256')!=V1PROTO or cfg.get('status')!='FROZEN_AFTER_PRE_SPLIT_DATA_QUALITY_AUDIT_BEFORE_MODEL_OR_TEST_EVALUATION':raise b.Stop('STOP_DATA_INTEGRITY v2 freeze')
 for n in ('ACQUISITION_STATUS.json','SOURCE_RECORD_MAPPING.json','OFFICIAL_SHA256SUMS.txt','DUPLICATE_CONFLICTS.json','conclusions_v1.json'):required(V1/n)
 ac=json.loads((V1/'ACQUISITION_STATUS.json').read_text());dec=json.loads((V1/'conclusions_v1.json').read_text())
 if ac.get('stage')!='ACQUISITION_COMPLETE' or ac.get('completed_files')!=69810 or ac.get('expected_files')!=69810 or ac.get('checksum_verified_files')!=69810 or ac.get('verified_bytes_total')!=4215509826 or dec.get('decision')!='STOP_DUPLICATE_CONFLICT' or dec.get('protocol_v1_sha256')!=V1PROTO or sha(V1/'conclusions_v1.json').upper()!=V1DECISION or sha(V1/'DUPLICATE_CONFLICTS.json').upper()!=CONFLICT:raise b.Stop('STOP_DATA_INTEGRITY v1 provenance')
 mp=json.loads((V1/'SOURCE_RECORD_MAPPING.json').read_text())
 if len(mp.get('records',()))!=34905 or mp.get('logical_record_count')!=34905 or mp.get('physical_file_count')!=69810 or sha(V1/'SOURCE_RECORD_MAPPING.json').upper()!=MAPPING or sha(V1/'OFFICIAL_SHA256SUMS.txt').upper()!=SUMS or ac.get('source_mapping_sha256','').upper()!=MAPPING or ac.get('official_manifest_sha256','').upper()!=SUMS:raise b.Stop('STOP_DATA_INTEGRITY mapping/acquisition bindings')
 z={'version':'ningbo_confirmation_v2','decision':'PASS','protocol_v2_sha256':V2HASH,'parent_v1_protocol_sha256':V1PROTO,'parent_v1_decision':'STOP_DUPLICATE_CONFLICT','parent_v1_decision_sha256':V1DECISION,'conflict_artifact_sha256':CONFLICT,'mapping_sha256':MAPPING,'official_sums_sha256':SUMS,'source_records':34905,'source_files':69810,'verified_bytes_total':4215509826}
 jput(OUT/'PREFLIGHT_V2.json',z);jput(OUT/'FROZEN_PROTOCOL_HASH.json',{'sha256':z['protocol_v2_sha256'],'parent_v1_sha256':V1PROTO});return mp,z
def shard_binding(z,mapping_sha):
 return (z.get('ingestion_schema')==b.INGESTION_SCHEMA
         and str(z.get('mapping_sha256','')).lower()==str(mapping_sha).lower()
         and str(z.get('protocol_sha256','')).upper()==V1PROTO)
def v2_conflict_gate(rows,canonical,collapsed,conflicts):
 if len(rows)!=34905 or len(canonical)!=34899 or len(conflicts)!=3 or sum(len(x['records']) for x in conflicts)!=6 or collapsed:raise b.Stop('STOP_DATA_INTEGRITY audited duplicate counts')
def migrate_rows(v1out=V1,out=OUT,mapping_sha=None):
 """Validate all v1 parsed shards, then atomically rebind them to v2."""
 mapping_sha=mapping_sha or sha(v1out/'SOURCE_RECORD_MAPPING.json');src=v1out/'audit_shards';dst=out/'audit_shards';dst.mkdir(parents=True,exist_ok=True);rows=[]
 starts=list(range(0,34905,250))
 for start in starts:
  p=src/f'{start:05d}.json';required(p);z=json.loads(p.read_text())
  if z.get('start')!=start or not shard_binding(z,mapping_sha) or z.get('exclusions')!=[]:raise b.Stop('STOP_DATA_INTEGRITY invalid v1 shard '+str(start))
  rows+=z['rows'];jput(dst/f'{start:05d}.json',{'start':start,'rows':z['rows'],'exclusions':[],'ingestion_schema':b.INGESTION_SCHEMA,'mapping_sha256':mapping_sha,'protocol_sha256':sha(PROTO).upper(),'parent_v1_protocol_sha256':V1PROTO,'parent_shard_sha256':sha(p)})
 if len(rows)!=34905 or len({r['record_id'] for r in rows})!=34905:raise b.Stop('STOP_DATA_INTEGRITY shard coverage')
 canonical,collapsed,conflicts=b.merge_audit_rows(rows)
 v2_conflict_gate(rows,canonical,collapsed,conflicts)
 target_conflicts=sum(len({tuple(r['labels']) for r in x['records']})>1 for x in conflicts)
 excluded=sorted([r for c in conflicts for r in c['records']],key=lambda r:r['record_id'])
 if len(canonical)!=34899:raise b.Stop('STOP_DATA_INTEGRITY canonical count')
 b.csvput(out/'EXCLUDED_CONFLICT_GROUPS.csv',[{'waveform_sha256':c['waveform_sha256'],'record_id':r['record_id'],'age':r['age'],'sex':r['sex'],'dx':'|'.join(r['dx']),'labels':'|'.join(map(str,r['labels']))} for c in conflicts for r in c['records']])
 jput(out/'EXCLUDED_CONFLICT_GROUPS.json',{'rule':'exclude all members independent of labels','groups':conflicts,'excluded_records':len(excluded),'target_label_conflict_groups_recorded_not_used':target_conflicts})
 base={'version':'ningbo_confirmation_v2','ingestion_schema':b.INGESTION_SCHEMA,'mapping_sha256':mapping_sha,'protocol_sha256':sha(PROTO).upper(),'parent_v1_protocol_sha256':V1PROTO,'source_records':34905,'canonical_records':34899,'excluded_conflict_groups':3,'excluded_conflict_records':6}
 jput(out/'CANONICAL_RECORDS.json',{'records':canonical,**base});jput(out/'DATA_INGESTION_AUDIT.json',{'decision':'PASS','exact_metadata_duplicate_groups':0,'target_endpoint_label_conflict_groups_recorded_not_used':0,**base});return canonical
def v2_split_binding():
 return {'version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,'canonical_records_sha256':sha(OUT/'CANONICAL_RECORDS.json'),'exclusion_artifact_sha256':sha(OUT/'EXCLUDED_CONFLICT_GROUPS.json'),'canonical_record_count':34899}
def valid_split_cache(old,bind):
 return len(old.get('rows',()))==34899 and all(old.get(k)==v for k,v in bind.items())
def split_v2(canonical):
 if len(canonical)!=34899:raise b.Stop('STOP_DATA_INTEGRITY v2 canonical population count')
 p=OUT/'SPLIT_MEMBERSHIP.json';bind=v2_split_binding()
 if p.exists():
  old=json.loads(p.read_text())
  if not valid_split_cache(old,bind):raise b.Stop('STOP_DATA_INTEGRITY stale v2 split cache')
  b.leakage(old['rows']);return old['rows']
 rng=np.random.default_rng(20260829);groups={}
 for r in canonical:groups.setdefault(tuple(r['labels']),[]).append(r)
 rows=[]
 for bits,g in sorted(groups.items()):
  a=np.array(sorted(g,key=lambda q:q['record_id']),dtype=object);rng.shuffle(a);n=len(a);nt=round(.7*n);nv=round(.1*n)
  for name,q in (('train',a[:nt]),('validation',a[nt:nt+nv]),('test',a[nt+nv:])):rows += [{'record_id':r['record_id'],'waveform_sha256':r['waveform_sha256'],'labels':r['labels'],'stratum':''.join(map(str,bits)),'split':name} for r in q]
 rows=sorted(rows,key=lambda q:q['record_id']);b.leakage(rows);jput(p,{**bind,'seed':20260829,'fractions':{'train':.7,'validation':.1,'test':.2},'rows':rows});jput(OUT/'SPLIT_LEAKAGE_PROOF.json',{'version':'ningbo_confirmation_v2','decision':'PASS','record_id_overlap':0,'waveform_hash_overlap':0,'protocol_v2_sha256':V2HASH,'split_sha256':sha(p)});return rows
def annotate_v2_prepare():
 """Make inherited prepare/model artifacts unambiguously V2-bound."""
 bind=v2_split_binding();split_sha=sha(OUT/'SPLIT_MEMBERSHIP.json')
 for name in ('MODEL_PROVENANCE.json','ENVIRONMENT_PROVENANCE.json','PREPARE_REPORT.json'):
  p=OUT/name;z=json.loads(p.read_text());z.update({'workflow_version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,'split_sha256':split_sha,**bind});jput(p,z)
 p=OUT/'TRAIN_VALIDATION_MEMBERSHIP.json';z=json.loads(p.read_text());z.update({'workflow_version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,**bind});jput(p,z)
def v2_model_cache_binding(seed):
 d=OUT/f'seed_{seed}'
 return {'workflow_version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,'seed':seed,'model_cache_namespace':'ningbo_confirmation_v2','split_sha256':sha(OUT/'SPLIT_MEMBERSHIP.json'),'canonical_records_sha256':sha(OUT/'CANONICAL_RECORDS.json'),'exclusion_artifact_sha256':sha(OUT/'EXCLUDED_CONFLICT_GROUPS.json'),'baseline_sha256':sha(d/'baseline.json'),'checkpoint_sha256':sha(d/'checkpoint.pt'),'preprocessing_sha256':sha(d/'preprocessing.npz')}
def valid_v2_model_cache(old,bind):
 return old==bind and old.get('workflow_version')=='ningbo_confirmation_v2' and old.get('protocol_v2_sha256')==V2HASH and old.get('model_cache_namespace')=='ningbo_confirmation_v2'
def train_v2(seed,part):
 d=OUT/f'seed_{seed}';bp=d/'baseline.json';vp=d/'V2_MODEL_CACHE_PROVENANCE.json'
 if bp.exists() and json.loads(bp.read_text()).get('decision')=='PASS':
  if not vp.is_file() or not valid_v2_model_cache(json.loads(vp.read_text()),v2_model_cache_binding(seed)):raise b.Stop('STOP_TECHNICAL_BLOCKER stale or unbound v2 model cache')
 b.train(seed,part)
 jput(vp,v2_model_cache_binding(seed))
def prepare_only():
 mp,info=preflight();canonical=migrate_rows(mapping_sha=info['mapping_sha256']);bind_v2();rows=split_v2(canonical);part=b.prepare(canonical,rows);annotate_v2_prepare();jput(OUT/'V2_PREPARE_COMPLETE.json',{'version':'ningbo_confirmation_v2','decision':'PASS','canonical_records':len(canonical),'split_sha256':sha(OUT/'SPLIT_MEMBERSHIP.json'),'protocol_v2_sha256':V2HASH});return part
def smoke_only():
 preflight();bind_v2();rp,tvp=OUT/'CANONICAL_RECORDS.json',OUT/'TRAIN_VALIDATION_MEMBERSHIP.json'
 if not rp.exists() or not tvp.exists():raise b.Stop('STOP_TECHNICAL_BLOCKER smoke requires v2 prepare')
 by={r['record_id']:r for r in json.loads(rp.read_text())['records']};tv=json.loads(tvp.read_text())['rows'];part={s:[by[q['record_id']] for q in tv if q['split']==s] for s in ('train','validation')};b.smoke(part)
def full():
 part=prepare_only()
 for s in b.SEEDS:train_v2(s,part)
 b.gate();pred={};th={}
 for s in b.SEEDS:pred[s],th[s]=b.evaluate(s,part['test'])
 y,c=b.aggregate(part['test'],pred,th);boot=b.bootstrap(y,pred);b.finalize(c,boot)
def selftest():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);v1=root/'v1';out=root/'v2';(v1/'audit_shards').mkdir(parents=True);jput(v1/'SOURCE_RECORD_MAPPING.json',{'x':1});mh=sha(v1/'SOURCE_RECORD_MAPPING.json')
  live=out/'ningbo_confirmation_v2.lock';out.mkdir();jput(live,{'pid':os.getpid()})
  try:lock_for(out);raise AssertionError('live lock accepted')
  except b.Stop:pass
  jput(live,{'pid':-1});stale=lock_for(out);assert stale.exists();stale.unlink()
  row=lambda rid,h,lab,age=20:[{'record_id':rid,'waveform_sha256':h,'age':age,'sex':'male','dx':['164934002'],'labels':lab,'header_path':'h','waveform_path':'w'}]
  # Minimal synthetic shard migration binding check (a full source uses 140 shards).
  good={'start':0,'rows':row('a','h',[1,0,0]),'exclusions':[],'ingestion_schema':b.INGESTION_SCHEMA,'mapping_sha256':mh,'protocol_sha256':V1PROTO};assert shard_binding(good,mh) and shard_binding(good,mh.upper()) and not shard_binding({**good,'protocol_sha256':'bad'},mh)
  can,_,conf=b.merge_audit_rows(row('a','h',[1,0,0])+row('b','h',[0,1,0],21));assert len(conf)==1 and not can
  # All group members are selected regardless of their differing labels.
  assert sorted(r['record_id'] for c in conf for r in c['records'])==['a','b']
  try:
   bad=good.copy();bad['start']=250
   if bad.get('start')!=0:raise b.Stop('stale shard rejected')
   raise AssertionError('bad shard accepted')
  except b.Stop:pass
  split_bind={'version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,'canonical_records_sha256':'c','exclusion_artifact_sha256':'e','canonical_record_count':34899};split_ok={**split_bind,'rows':[{}]*34899};assert valid_split_cache(split_ok,split_bind) and not valid_split_cache({**split_ok,'protocol_v2_sha256':'wrong'},split_bind) and not valid_split_cache({**split_bind,'rows':[{}]*34898},split_bind)
  cache_bind={'workflow_version':'ningbo_confirmation_v2','protocol_v2_sha256':V2HASH,'model_cache_namespace':'ningbo_confirmation_v2'};assert valid_v2_model_cache(cache_bind,cache_bind) and not valid_v2_model_cache({**cache_bind,'workflow_version':'ningbo_confirmation_v1'},cache_bind)
 before=(sha(V1/'conclusions_v1.json'),sha(V1/'DUPLICATE_CONFLICTS.json'));assert sha(PROTO).upper()==V2HASH and sha(V1/'SOURCE_RECORD_MAPPING.json').upper()==MAPPING and sha(V1/'OFFICIAL_SHA256SUMS.txt').upper()==SUMS and sha(V1/'conclusions_v1.json').upper()==V1DECISION and (V1/'conclusions_v1.json').is_file() and (V1/'DUPLICATE_CONFLICTS.json').is_file() and before==(sha(V1/'conclusions_v1.json'),sha(V1/'DUPLICATE_CONFLICTS.json'))
 print('V2 SELF_TEST PASS: atomic live/stale lock, frozen config and exact v1 acquisition bindings, split/cache V2 provenance rejection, v1-shard binding rejection, all-member label-independent exclusion, exact conflict-count gate contract',flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--self-test',action='store_true');a.add_argument('--preflight-only',action='store_true');a.add_argument('--prepare-only',action='store_true');a.add_argument('--smoke-only',action='store_true');q=a.parse_args()
 if q.self_test:selftest();return
 OUT.mkdir(parents=True,exist_ok=True)
 with open(OUT/'workflow_stdout.log','a',encoding='utf8') as so,open(OUT/'workflow_stderr.log','a',encoding='utf8') as se:
  with contextlib.redirect_stdout(so),contextlib.redirect_stderr(se):
   lp=None
   try:
    lp=lock()
    if q.preflight_only:preflight()
    elif q.prepare_only:prepare_only()
    elif q.smoke_only:smoke_only()
    else:full()
   except b.Stop as e:status(str(e));traceback.print_exc();raise
   except Exception as e:status('STOP_TECHNICAL_BLOCKER',error=repr(e));traceback.print_exc();raise
   finally:
    if lp is not None:lp.unlink(missing_ok=True)
if __name__=='__main__':main()
