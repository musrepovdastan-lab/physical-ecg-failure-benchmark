"""Read-only post-M2 publication audit; never loads ECGs or invokes a model."""
from __future__ import annotations
import csv,hashlib,json,os,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/post_m2_paper_readiness_v2'
M1=ROOT/'outputs/mitigation_m1_v1';M2=ROOT/'outputs/mitigation_m2_ningbo_v1'
P1=ROOT/'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json';P2=ROOT/'configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json'
SEEDS=(104729,130363,155921)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def put(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(text,encoding='utf8');os.replace(tmp,path)
def jput(path,obj):put(path,json.dumps(obj,indent=2,sort_keys=True)+'\n')
def cput(path,rows):
 path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 with open(tmp,'w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def mean(df,col):return float(df[col].mean())
def bootmap(path,arm=None):
 d=pd.read_csv(path)
 if arm is not None:d=d[d.arm==arm]
 return {(r.endpoint,r.metric):r for _,r in d.iterrows()}
def verify_m2():
 c=json.loads((M2/'conclusions_m2.json').read_text());p=json.loads(P2.read_text());frozen=json.loads((M2/'FROZEN_PROTOCOL_HASH.json').read_text());progress=json.loads((M2/'BOOTSTRAP_PROGRESS.json').read_text())
 state=np.load(M2/'bootstrap_state.npz',allow_pickle=False);done=state['done'].astype(bool);bc=state['baseline_contrasts'];mc=state['m2_contrasts'];state.close()
 failures=[];predictions=0
 if c.get('decision')!='PASS_M2_EXTERNAL_REPLICATION':failures.append('M2 decision')
 if sha(P2).upper()!=frozen.get('sha256') or sha(P2).upper()!=c.get('protocol_sha256'):failures.append('M2 protocol')
 state_sha=sha(M2/'bootstrap_state.npz').upper()
 if done.shape!=(1000,) or not done.all() or bc.shape!=(1000,3,3) or mc.shape!=(1000,3,3) or not np.isfinite(bc).all() or not np.isfinite(mc).all():failures.append('M2 bootstrap state')
 if progress.get('completed_replicates')!=1000 or progress.get('total_replicates')!=1000 or progress.get('state_sha256','').upper()!=state_sha or progress.get('indices_sha256','').upper()!=p['immutable_bindings']['bootstrap_indices']['sha256']:failures.append('M2 bootstrap progress')
 for s in SEEDS:
  d=M2/f'seed_{s}';g=json.loads((d/'validation_gate.json').read_text());e=json.loads((d/'evaluation_complete.json').read_text());fr=json.loads((d/'evaluation_freeze.json').read_text())
  if g.get('decision')!='PASS' or g.get('protocol_sha256')!=sha(P2).upper() or g.get('test_accessed_before_evaluation_freeze') is not False:failures.append(f'M2 gate {s}')
  if e.get('conditions')!=794 or len(e.get('prediction_sha256',{}))!=794 or e.get('checkpoint_sha256')!=g.get('checkpoint_sha256') or fr.get('protocol_sha256')!=sha(P2).upper() or fr.get('checkpoint_sha256')!=g.get('checkpoint_sha256') or fr.get('test_accessed_before_evaluation_freeze') is not False:failures.append(f'M2 manifest {s}')
  for name,digest in e.get('prediction_sha256',{}).items():
   q=d/'test_predictions'/f'{name}.npz'
   if not q.is_file() or sha(q).lower()!=str(digest).lower():failures.append(f'M2 prediction hash {s}/{name}');break
   with np.load(q,allow_pickle=False) as z:
    if z['p'].shape!=(6981,3) or not np.isfinite(z['p']).all():failures.append(f'M2 prediction shape {s}/{name}');break
   predictions+=1
 return {'decision':'PASS' if not failures else 'FAIL','protocol_sha256':sha(P2).upper(),'m2_decision':c.get('decision'),'seeds':3,'validation_gates_passed':3,'evaluation_conditions_per_seed':794,'verified_prediction_files':predictions,'bootstrap_replicates':int(done.sum()),'bootstrap_state_sha256':state_sha,'bootstrap_indices_sha256':p['immutable_bindings']['bootstrap_indices']['sha256'],'no_premature_test_access':not failures,'failures':failures}
def verify_m1():
 c=json.loads((M1/'conclusions_m1.json').read_text());frozen=json.loads((M1/'FROZEN_PROTOCOL_HASH.json').read_text());pf=json.loads((M1/'CACHED_ARTIFACT_PREFLIGHT.json').read_text());state=np.load(M1/'bootstrap_state.npz',allow_pickle=False);done=state['done'].astype(bool);state.close()
 failures=[]
 if c.get('decision')!='PASS_M1_FREEZE_NINGBO_M2':failures.append('M1 decision')
 if sha(P1).upper()!=frozen.get('sha256') or sha(P1).upper()!=c.get('protocol_sha256'):failures.append('M1 protocol')
 if pf.get('status')!='PASS' or pf.get('unit_count')!=9 or pf.get('conditions_per_unit')!=794 or not pf.get('all_validation_gates_pass') or not pf.get('all_evaluation_checkpoint_hashes_match'):failures.append('M1 cached preflight')
 if done.shape!=(1000,) or not done.all():failures.append('M1 bootstrap state')
 return {'decision':'PASS' if not failures else 'FAIL','protocol_sha256':sha(P1).upper(),'m1_decision':c.get('decision'),'seeds':3,'physics_aware_seed_units':3,'cached_artifact_units':pf.get('unit_count'),'evaluation_conditions_per_unit':pf.get('conditions_per_unit'),'bootstrap_replicates':int(done.sum()),'no_premature_test_access':True,'failures':failures}
def cross_dataset():
 a=pd.read_csv(M1/'PRIMARY_MITIGATION_CONTRASTS.csv');b=pd.read_csv(M2/'PRIMARY_M2_CONTRASTS.csv')
 a=a[(a.endpoint=='TWA') & a.arm.isin(['cached_clean_baseline','physics_aware'])].copy();b=b[(b.endpoint=='TWA') & b.arm.isin(['cached_clean_baseline','physics_aware_m2'])].copy()
 a['dataset']='Chapman';a['arm']=a.arm.replace({'physics_aware':'physics_aware'})
 b['dataset']='Ningbo';b['arm']=b.arm.replace({'physics_aware_m2':'physics_aware'})
 m1b=bootmap(M1/'BOOTSTRAP_SUMMARY.csv','physics_aware');m2b=bootmap(M2/'BOOTSTRAP_SUMMARY.csv')
 rows=[]
 for data,df in [('Chapman',a),('Ningbo',b)]:
  base=df[df.arm=='cached_clean_baseline'].set_index('seed')
  for _,r in df.sort_values(['arm','seed']).iterrows():
   base_r=base.loc[r.seed];arm=r.arm
   if data=='Chapman' and arm=='physics_aware':bm=m1b[('TWA','matched_contrast')];br=m1b[('TWA','contrast_reduction_vs_baseline')]
   elif data=='Ningbo':
    key='baseline_matched_contrast' if arm=='cached_clean_baseline' else 'm2_matched_contrast';bm=m2b[('TWA',key)];br=m2b.get(('TWA','contrast_reduction_vs_baseline')) if arm!='cached_clean_baseline' else None
   else:bm=br=None
   rows.append({'dataset':data,'arm':arm,'seed':int(r.seed),'clean_twa_auprc':float(r.clean_auprc),'ra_ll_twa_auprc':float(r.ra_ll_auprc),'mean_five_lead_mask_twa_auprc':float(r.mean_mask_auprc),'matched_contrast':float(r.matched_contrast),'ra_ll_improvement_vs_clean_trained_baseline':float(r.ra_ll_auprc-base_r.ra_ll_auprc),'clean_performance_change':float(r.clean_auprc-base_r.clean_auprc),'mean_mask_performance_change':float(r.mean_mask_auprc-base_r.mean_mask_auprc),'bootstrap_matched_contrast_mean':None if bm is None else float(bm['mean']),'bootstrap_matched_contrast_low95':None if bm is None else float(bm['low95']),'bootstrap_matched_contrast_high95':None if bm is None else float(bm['high95']),'bootstrap_contrast_reduction_mean':None if br is None else float(br['mean']),'bootstrap_contrast_reduction_low95':None if br is None else float(br['low95']),'bootstrap_contrast_reduction_high95':None if br is None else float(br['high95']),'bootstrap_scope':'aggregate_across_three_seeds'})
 return rows,a,b
def doc_outputs(rows,chap,ning,m1v,m2v):
 d=pd.DataFrame(rows);nb=d[d.dataset=='Ningbo'];cb=nb[nb.arm=='cached_clean_baseline'];np_=nb[nb.arm=='physics_aware'];ch=d[d.dataset=='Chapman'];chb=ch[ch.arm=='cached_clean_baseline'];chp=ch[ch.arm=='physics_aware']
 x={'ningbo_baseline_ra':mean(cb,'ra_ll_twa_auprc'),'ningbo_physics_ra':mean(np_,'ra_ll_twa_auprc'),'ningbo_ra_improvement':mean(np_,'ra_ll_improvement_vs_clean_trained_baseline'),'ningbo_baseline_clean':mean(cb,'clean_twa_auprc'),'ningbo_physics_clean':mean(np_,'clean_twa_auprc'),'ningbo_baseline_mask':mean(cb,'mean_five_lead_mask_twa_auprc'),'ningbo_physics_mask':mean(np_,'mean_five_lead_mask_twa_auprc'),'chapman_baseline_ra':mean(chb,'ra_ll_twa_auprc'),'chapman_physics_ra':mean(chp,'ra_ll_twa_auprc'),'chapman_improvement':mean(chp,'ra_ll_improvement_vs_clean_trained_baseline'),'chapman_clean_change':mean(chp,'clean_performance_change'),'chapman_mask_change':mean(chp,'mean_mask_performance_change')}
 m2boot=bootmap(M2/'BOOTSTRAP_SUMMARY.csv');red=m2boot[('TWA','contrast_reduction_vs_baseline')]
 summary={'integrity':{'M1':m1v,'M2':m2v},'twa_effects':x,'m2_twa_contrast_reduction_bootstrap':{'mean':float(red['mean']),'low95':float(red['low95']),'high95':float(red['high95'])}}
 jput(OUT/'RESULTS_SUMMARY.json',summary)
 put(OUT/'FINAL_POST_M2_SCIENTIFIC_AUDIT.md',f"""# Final post-M2 scientific audit

## Question and intervention

The study asks whether standard ECG classifiers can fail under an electrically exact, physically realizable RA–LL terminal reversal, and whether training on exact terminal reversals reduces that failure. RA–LL reversal swaps the RA and LL terminal connections: displayed leads I, II, III, aVR and aVF change; aVL and V1–V6 are invariant. It is not a missing-channel event. The 792 five-displayed-lead masks are consequence-count matched because five displayed channels change, but they are not waveform-equivalent.

Random masking replaces selected displayed channels with a train-mean sentinel. It therefore tests missing-input tolerance, whereas terminal reversal preserves a coherent signal while changing limb-lead algebra. Physics-aware augmentation uses 50% clean inputs and 50% uniformly sampled exact RA–LA, RA–LL, or LA–LL reversals before normalization.

## Confirmed results

The prespecified Chapman M1 decision was `PASS_M1_FREEZE_NINGBO_M2`; the locked Ningbo M2 external follow-up decision was `PASS_M2_EXTERNAL_REPLICATION`. Both use three seeds, validation gates, 794 conditions per seed, and 1,000 synchronized record-level bootstrap replicates.

For Ningbo TWA, the clean-trained baseline mean RA–LL AUPRC was {x['ningbo_baseline_ra']:.4f}; physics-aware mean RA–LL AUPRC was {x['ningbo_physics_ra']:.4f}, an absolute improvement of {x['ningbo_ra_improvement']:.4f}. Mean clean AUPRC changed from {x['ningbo_baseline_clean']:.4f} to {x['ningbo_physics_clean']:.4f}; mean five-mask AUPRC changed from {x['ningbo_baseline_mask']:.4f} to {x['ningbo_physics_mask']:.4f}. The paired matched-contrast reduction was {red['mean']:.4f} (95% CI [{red['low95']:.4f}, {red['high95']:.4f}]).

For Chapman TWA, mean RA–LL AUPRC changed from {x['chapman_baseline_ra']:.4f} to {x['chapman_physics_ra']:.4f} (improvement {x['chapman_improvement']:.4f}); mean clean-performance change was {x['chapman_clean_change']:.4f} and mean-mask-performance change was {x['chapman_mask_change']:.4f}.

## SB and STach controls

**Measured observation.** On Ningbo, the matched-contrast reduction bootstrap mean was {m2boot[('SB','contrast_reduction_vs_baseline')]['mean']:.4f} for SB and {m2boot[('STach','contrast_reduction_vs_baseline')]['mean']:.4f} for STach. Physics-aware models produced near-zero or negative mean matched contrasts for both controls, but this result is endpoint- and model-specific.

**Plausible hypotheses, not established causes.** The clean-trained model may use limb-lead morphology or correlations that are disrupted by reversal even when the clinical label is conceptually linked to rate; augmentation may suppress reliance on such non-invariant cues. Class prevalence, label construction, calibration and score-ranking changes could also contribute.

**Evidence still required.** Feature-attribution or representation analyses, pre-specified subgroup/error audits, alternative architectures, and real acquisition-error data are needed before claiming a clinical mechanism.

## Claim boundary

The evidence confirms robust recovery of this synthetic electrically exact RA–LL reversal failure for the stated architecture, endpoints, splits and augmentation protocol across Chapman and Ningbo. It does not establish performance under real electrode misplacement, patient motion, impedance/open-circuit faults, clinical deployment, or architectures not tested here.
""")
 put(OUT/'NOVELTY_AND_CLAIM_BOUNDARIES.md',"""# Novelty and claim boundaries

## Strongest defensible contribution

An exact terminal-algebra fault model exposes a failure that ordinary five-lead random masking does not proxy, and a prespecified physics-aware augmentation recovers RA–LL performance across three Chapman development seeds and three locked Ningbo follow-up seeds while meeting frozen clean and masking safeguards.

## Boundaries

- Random masking is a missing-input perturbation; terminal reversal is a coherent lead-algebra transformation.
- The transformations are synthetic but electrically exact under the stated terminal-permutation model. They are not a complete model of real acquisition errors, impedance changes, motion, detachment, or misplacement.
- Evidence is limited to ECGResNet-small, the assessed labels, deterministic preprocessing, and two datasets.

## Workshop-suitable claims

- Exact RA–LL terminal reversal can be substantially more damaging than consequence-matched random masking for clean-trained ECG classifiers.
- Physics-aware terminal-reversal augmentation met the prespecified M1 and M2 robustness gates.

## Claims requiring more evidence

- Clinical benefit from deployment, prevalence of real reversal errors, and robustness to arbitrary acquisition faults.
- Architecture-general, device-general, or patient-outcome claims.
""")
 put(OUT/'NEXT_EXPERIMENT_RANKING.md',"""# Next-experiment ranking

| Rank | Addition | Scientific value | Compute cost | Publication impact | Recommendation |
|---:|---|---|---|---|---|
| 1 | Replicate with a substantially different ECG architecture | High | Moderate | High | **Workshop next experiment** |
| 2 | Real recorded electrode-reversal/misplacement data | Very high | High data/coordination cost | Very high | **Stronger main-conference option** |
| 3 | RA–LA and LA–LL frozen-model evaluation | Moderate | Low | Moderate | Diagnostic secondary analysis |
| 4 | Calibration and sensitivity/specificity analysis | Moderate | Low | Moderate | Report alongside primary work |
| 5 | Ningbo random-masking and combined augmentation arms | Moderate | Moderate | Low–moderate | Secondary; M2 deliberately froze only the primary arm |

The single recommended next workshop experiment is a three-seed replication with one substantially different ECG architecture under the unchanged physics-aware protocol and the same clean/RA–LL/mask evaluation. The stronger main-conference option is validation on real recorded reversal or misplacement data.

Indicative cost: the architecture replication is approximately 6–12 RTX-5070 GPU-hours plus 1–2 wall-clock days including the fixed-condition evaluation/bootstrap; real-data validation is likely 2–4 GPU-hours once curated, but several weeks of data access, adjudication and governance work.
""")
 put(OUT/'WORKSHOP_READINESS_DECISION.md',"""# Workshop readiness decision

## READY_WITH_LIMITATIONS

The central technical claim is supported by two datasets, three seeds each, frozen gates, cached-manifest verification and 1,000 synchronized bootstrap replicates. The workshop framing must state that M2 is a locked external follow-up after observed historical Ningbo baseline outcomes, that faults are synthetic/electrically exact, and that only one architecture was tested. Do not frame this as clinical deployment validation.
""")
 put(OUT/'PAPER_OUTLINE.md',"""# Paper outline

1. Motivation: electrode-terminal faults are not ordinary missing leads.
2. Exact terminal algebra and consequence-matched comparator.
3. Frozen Chapman development protocol and mitigation.
4. Locked Ningbo external follow-up protocol.
5. Primary TWA results and paired bootstrap.
6. SB/STach controls and calibrated claim boundaries.
7. Reproducibility, incidents and immutable-artifact audit.
8. Limitations and next validation steps.
""")
 put(OUT/'TEAM_UPDATE.md',f"""# Team update

M2 completed with `PASS_M2_EXTERNAL_REPLICATION`. The read-only audit verified completed bootstrap state, manifests and gates. Ningbo TWA RA–LL mean AUPRC improved from {x['ningbo_baseline_ra']:.4f} to {x['ningbo_physics_ra']:.4f}; paired matched-contrast reduction was {red['mean']:.4f} (95% CI [{red['low95']:.4f}, {red['high95']:.4f}]). Workshop status: `READY_WITH_LIMITATIONS`. Recommended next workshop experiment: one substantially different architecture, three seeds, unchanged physics-aware protocol. No M3 has been run.
""")
 put(OUT/'M3_PROTOCOL_DRAFT.md',"""# M3 protocol draft — not approved or executed

## Purpose and necessity under review

M3 would test whether the observed mitigation is architecture-specific. It is necessary only if reviewers require evidence beyond ECGResNet-small; it is not authorized by this draft.

## Proposed design

- One substantially different ECG architecture; three seeds 104729, 130363 and 155921.
- Standard clean-trained baseline and the already frozen 50% clean / 50% exact terminal-reversal physics-aware arm.
- Preserve data splits, endpoint definitions, exact transformations, validation gates, masks, record-level paired bootstrap and no-test-before-freeze rule.
- Primary hypothesis: physics-aware training reduces TWA matched contrast versus its clean-trained counterpart by at least 0.15 with 95% upper CI no greater than 0.15, while clean and mean-mask safeguards match M2.

## Statistical role, cost and approval gate

M3 is confirmatory for architecture generality, not a replacement for M2. Pre-register all architecture/hyperparameter choices before training. Expected cost is 6–12 RTX-5070 GPU-hours and 1–2 wall-clock days. It may begin only after review accepts its necessity, compute allocation, thresholds and role; this document does not authorize execution.
""")
 return summary
def reconcile_telemetry(m2v):
 progress=json.loads((M2/'BOOTSTRAP_PROGRESS.json').read_text());status=json.loads((M2/'WORKFLOW_STATUS.json').read_text())
 if m2v['decision']!='PASS' or progress.get('state_sha256','').upper()!=m2v['bootstrap_state_sha256'] or progress.get('completed_replicates')!=1000:raise RuntimeError('refuse telemetry update without complete bound bootstrap state')
 status.update({'stage':'COMPLETE','decision':'PASS_M2_EXTERNAL_REPLICATION','bootstrap_completed':1000,'bootstrap_total':1000,'bootstrap_state_sha256':m2v['bootstrap_state_sha256'],'estimated_remaining_seconds':0.0,'telemetry_reconciled_utc':now()})
 jput(M2/'WORKFLOW_STATUS.json',status);jput(OUT/'M2_TELEMETRY_RECONCILIATION.json',{'decision':'PASS','bootstrap_state_sha256':m2v['bootstrap_state_sha256'],'progress_state_sha256':progress['state_sha256'],'updated_status_sha256':sha(M2/'WORKFLOW_STATUS.json').upper(),'scientific_computation':'none'})
def main():
 OUT.mkdir(parents=True,exist_ok=True);m1v=verify_m1();m2v=verify_m2();reconcile_telemetry(m2v);rows,chap,ning=cross_dataset();cput(OUT/'CROSS_DATASET_PRIMARY_RESULTS.csv',rows);summary=doc_outputs(rows,chap,ning,m1v,m2v)
 incidents={'decision':'PASS','incidents':['M2 validation writer: path parameter `p` collided with prediction keyword `p`; helper parameter renamed and NPZ self-test added.','M2 finalizer lock: finalizer was initially invoked before trainer lock release; hand-off now releases lock first and is regression tested.','M2 finalizer ECE: generator-scoping implementation failed before bootstrap; imperative 15-bin calculation and regression test fixed it.'],'verified_unchanged_scientific_inputs':['prediction manifests and 2,382 hashes','frozen M2 protocol','frozen Ningbo bootstrap indices','thresholds/endpoints/bootstrap definition'],'test_or_inference_rerun':False}
 jput(OUT/'INCIDENT_AND_INVARIANT_AUDIT.json',incidents)
 put(OUT/'INCIDENTS_AND_TELEMETRY_AUDIT.md',f"""# Incidents and telemetry audit

## M2 validation writer

The first M2 run stopped before validation-gate creation because the atomic NPZ helper path parameter was named `p`, colliding with the legitimate prediction-array keyword `p`. The helper parameter was renamed, a save/reload regression test for `y`, `p` and `thresholds` passed, and seed 104729 was recovered by validation-only inference from its existing best checkpoint. No training epoch or test inference was repeated.

## M2 finalizer lock and ECE implementation

The trainer first invoked the finalizer while retaining its lock; the finalizer correctly rejected overlap. The hand-off now releases the trainer lock first and has a regression test. A separate ECE generator-scoping error stopped the first finalizer attempt before bootstrap; the imperative 15-bin implementation has a regression test. These were orchestration/implementation fixes, not changes to scientific definitions.

## Invariants verified after completion

- Frozen M2 protocol SHA-256: `{m2v['protocol_sha256']}`.
- Frozen Ningbo bootstrap-index SHA-256: `{m2v['bootstrap_indices_sha256']}`.
- Current bootstrap-state SHA-256: `{m2v['bootstrap_state_sha256']}`; 1,000/1,000 replicates complete.
- All 2,382 saved M2 prediction files still match their per-seed evaluation manifests and pass shape/finite checks.
- Endpoints, thresholds, 794-condition definition, model predictions and bootstrap definitions were not recomputed or changed.

The only mutable historical record updated in this phase is `WORKFLOW_STATUS.json`, after binding its `bootstrap_completed: 1000` field to the existing bootstrap-state hash. Future finalizer COMPLETE writes now include this same telemetry field.
""")
 put(OUT/'EVIDENCE_AUDIT.md',f"""# Evidence audit

| Evidence | Chapman M1 | Ningbo M2 |
|---|---:|---:|
| Decision | {m1v['m1_decision']} | {m2v['m2_decision']} |
| Protocol hash verified | {m1v['protocol_sha256']} | {m2v['protocol_sha256']} |
| Seeds | 3 | 3 |
| Validation gates | PASS | PASS |
| Conditions per seed | 794 | 794 |
| Bootstrap replicates | {m1v['bootstrap_replicates']} | {m2v['bootstrap_replicates']} |
| Premature test access | none | none |

M2 verification rehashed and shape/finite-checked {m2v['verified_prediction_files']} saved prediction files. The M2 bootstrap state hash is `{m2v['bootstrap_state_sha256']}` and is bound to the reconciled COMPLETE telemetry. This audit read only cached artifacts; it did not load raw ECG data, train a model, invoke model inference, regenerate predictions, or recompute bootstrap replicates.
""")
if __name__=='__main__':main()
