import csv,json,hashlib
from pathlib import Path
R=Path.cwd();O=R/'outputs/calibration_independence_preflight_v1'
def h(b):return hashlib.sha256(b).hexdigest()
m=list(csv.DictReader((O/'PROPOSED_MEMBERSHIP.csv').open(encoding='utf-8-sig')))
assert len(m)==3488
roles={k:[x for x in m if x['role']==k] for k in ('model_selection','calibration_A','threshold_B')}
assert {k:len(v) for k,v in roles.items()}=={'model_selection':1744,'calibration_A':872,'threshold_B':872}
for key in ('record_id','waveform_sha256'):
 sets={k:{x[key] for x in v} for k,v in roles.items()}
 assert all(not(sets[a]&sets[b]) for a in sets for b in sets if a<b)
src=json.loads((R/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json').read_text(encoding='utf-8'))
hist={k:{x['record_id'] for x in src['rows'] if x['split']==k} for k in ('train','validation','test')}
assert set(x['record_id'] for x in m)==hist['validation']
assert not(set(x['record_id'] for x in m)&(hist['train']|hist['test']))
y={x['record_id']:''.join(map(str,x['labels'])) for x in src['rows']}
assert all(x['labels_TWA_SB_STach']==y[x['record_id']] for x in m)
for role in roles:
 for j,e in enumerate(('TWA','SB','STach')):
  p=sum(x['labels_TWA_SB_STach'][j]=='1' for x in roles[role])
  assert p>0 and p<len(roles[role])
audit=json.loads((O/'PREFLIGHT_AUDIT.json').read_text(encoding='utf-8'))
assert audit['partition_exact'] and audit['checked_intersections_zero'] and audit['no_training'] and audit['no_new_inference']
manifest=json.loads((O/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
bad=[]
for x in manifest['files']:
 p=R/x['path']
 if not p.exists() or h(p.read_bytes())!=x['sha256']:bad.append(x['path'])
assert not bad,bad
excluded=json.loads((R/'outputs/ningbo_confirmation_v2/EXCLUDED_CONFLICT_GROUPS.json').read_text(encoding='utf-8'))
exids={z['record_id'] for group in excluded['groups'] for z in group['records']}; exwaves={group['waveform_sha256'] for group in excluded['groups']}
assert len(exids)==6 and not {x['record_id'] for x in m}&exids and not {x['waveform_sha256'] for x in m}&exwaves
counts=list(csv.DictReader((O/'SPLIT_COUNTS.csv').open(encoding='utf-8-sig')))
result={'verification':'PASS','membership_rows':len(m),'role_counts':{k:len(v) for k,v in roles.items()},'counts_rows':len(counts),'manifest_files_unchanged':len(manifest['files']),'excluded_conflict_records':len(exids),'excluded_record_or_waveform_overlap':0,'calibration_A_positives':{e:sum(x['labels_TWA_SB_STach'][i]=='1' for x in roles['calibration_A']) for i,e in enumerate(('TWA','SB','STach'))},'threshold_B_positives':{e:sum(x['labels_TWA_SB_STach'][i]=='1' for x in roles['threshold_B']) for i,e in enumerate(('TWA','SB','STach'))}}
(O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
