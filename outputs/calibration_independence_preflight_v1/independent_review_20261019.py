"""Review metadata and candidate split; does not read waveform/prediction arrays."""
from pathlib import Path
import csv,json,hashlib,itertools
import numpy as np
from scipy.stats import beta,binom
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def membership_hash(items):return hashlib.sha256(('\n'.join(sorted(items))+'\n').encode()).hexdigest()
counts={};errors=[];notes=[]
def ck(kind,good):
 counts[kind]=counts.get(kind,0)+1
 if not good:errors.append(kind)
manifest=j(OUT/'INPUT_MANIFEST.json')
for ent in manifest['files']:
 got=sha(ROOT/ent['path'])
 if ent['path']=='NEXT_RESEARCH_DECISIONS_20261019.md' and got!=ent['sha256']:
  snapshot=OUT/'archive_before_review_20261019/ACTIVE_PLAN_AT_REVIEW.md'
  ck('archived_context_hash',snapshot.exists() and sha(snapshot)==ent['sha256'])
  notes.append('Active-plan status changed after preflight; original context preserved with matching hash in archive_before_review_20261019/ACTIVE_PLAN_AT_REVIEW.md. Scientific inputs unchanged.')
 else:ck('input_hash',got==ent['sha256'])
source=j(ROOT/'outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json')['rows']
canonical=j(ROOT/'outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json')['records']
by={r['record_id']:r for r in source};bycan={r['record_id']:r for r in canonical}
for r in source:
 ck('source_canonical_label',r['labels']==bycan[r['record_id']]['labels'])
 ck('source_canonical_waveform',r['waveform_sha256']==bycan[r['record_id']]['waveform_sha256'])
validation=[r for r in source if r['split']=='validation']
candidate=read(OUT/'PROPOSED_MEMBERSHIP.csv');roles={k:[r for r in candidate if r['role']==k] for k in ['model_selection','calibration_A','threshold_B']}
ck('candidate_count',len(candidate)==3488)
ck('validation_partition',len({r['record_id'] for r in candidate})==3488 and {r['record_id'] for r in candidate}=={r['record_id'] for r in validation})
for role,expected in [('model_selection',1744),('calibration_A',872),('threshold_B',872)]:ck('role_size',len(roles[role])==expected)
for r in candidate:
 s=by[r['record_id']]
 ck('source_index',validation[int(r['source_index'])]['record_id']==r['record_id'])
 ck('source_membership',s['split']=='validation' and r['source_split']=='validation')
 ck('candidate_labels',r['labels_TWA_SB_STach']==''.join(map(str,s['labels'])))
 ck('candidate_waveform',r['waveform_sha256']==s['waveform_sha256'])
 ck('record_digest',r['record_id_sha256']==hashlib.sha256(r['record_id'].encode()).hexdigest())
 ck('identity_not_invented',r['patient_id']=='NOT_AVAILABLE')
parts={k:[r for r in source if r['split']==k] for k in ['train','validation','test']}
for kind in ['record_id','waveform_sha256']:
 for a,b in itertools.combinations(parts,2):ck('source_overlap',not {r[kind] for r in parts[a]}&{r[kind] for r in parts[b]})
 for a,b in itertools.combinations(roles,2):ck('role_overlap',not {r[kind] for r in roles[a]}&{r[kind] for r in roles[b]})
excluded=j(ROOT/'outputs/ningbo_confirmation_v2/EXCLUDED_CONFLICT_GROUPS.json')['groups']
exids={r['record_id'] for g in excluded for r in g['records']};exwaves={g['waveform_sha256'] for g in excluded}
ck('exclusions',not {r['record_id'] for r in candidate}&exids and not {r['waveform_sha256'] for r in candidate}&exwaves)
allparts={**parts,**{k:[by[r['record_id']] for r in v] for k,v in roles.items()}}
for r in read(OUT/'SPLIT_COUNTS.csv'):
 part=allparts[r['split']];endpoint=r['endpoint']
 positive=sum(x['labels'][['TWA','SB','STach'].index(endpoint)] for x in part) if not endpoint.startswith('JOINT_') else sum(''.join(map(str,x['labels']))==endpoint[6:] for x in part)
 ck('counts_n',int(r['n'])==len(part));ck('counts_positive',int(r['positive'])==positive);ck('counts_negative',int(r['negative'])==len(part)-positive)
 ck('counts_prevalence',abs(float(r['prevalence'])-positive/len(part))<1e-15)
identity=j(OUT/'IDENTITY_AND_LEAKAGE_AUDIT.json')
for role,rr in roles.items():
 ck('role_id_hash',membership_hash([r['record_id'] for r in rr])==identity['proposed_membership_sha256_by_role'][role])
 ck('role_wave_hash',membership_hash([r['waveform_sha256'] for r in rr])==identity['proposed_waveform_hash_sha256_by_role'][role])
for r in read(OUT/'OPERATING_POINT_PRECISION.csv'):
 rr=allparts[r['role']];ep=['TWA','SB','STach'].index(r['endpoint']);p=sum(x['labels'][ep] for x in rr);n=len(rr)-p;k=round(.9*p)
 ck('precision_counts',int(r['positives'])==p and int(r['negatives'])==n)
 ck('precision_rank',int(r['q90_rank95_low'])==max(1,int(binom.ppf(.025,n,.9))) and int(r['q90_rank95_high'])==min(n,int(binom.ppf(.975,n,.9))+1))
 ck('precision_sensitivity',abs(float(r['illustrative_sensitivity_ci95_low'])-beta.ppf(.025,k,p-k+1))<1e-14 and abs(float(r['illustrative_sensitivity_ci95_high'])-beta.ppf(.975,k+1,p-k))<1e-14)
budget=j(OUT/'PREFLIGHT_AUDIT.json')['budget']
ck('fit_budget',budget['new_fits']==9 and budget['max_total_epochs']==270)
ck('runtime_projection',abs(budget['median_hours']-9*budget['median']/3600)<1e-12)
result={'status':'PASS_METADATA_AND_SPLIT','failure_count':len(errors),'failures':errors,'counts':counts,'notes':notes,
 'review_corrections_recorded':['HEEDB was evaluated with transferred source calibrators/thresholds, not directly fitted in verified historical pipelines.','TWA has the fewest positives (129); SB has the fewest negatives (555/556), so blanket STach precision bottleneck was corrected.','Total future calibration/threshold operations include Huang: 36 endpoint fits/selections, not just 27 core operations.'],
 'scope':'Independent metadata/count/hash/overlap/precision audit; no training, forward pass, calibration fit, threshold choice or test/HEEDB outcome analysis.'}
if errors:result['status']='FAIL'
(OUT/'INDEPENDENT_REVIEW_AUDIT_20261019.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
if errors:raise SystemExit(1)
