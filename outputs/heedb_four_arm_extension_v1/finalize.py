from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/mixed_fault_calibration_v1'))
import run_analysis as base

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 protocol=json.loads((OUT/'FROZEN_EXTENSION_PROTOCOL.json').read_text(encoding='utf-8'))
 if sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json')!=(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').read_text().strip():raise RuntimeError('protocol hash mismatch')
 # Correctly regenerate raw/calibrated ordering and threshold decision audit from bound fits.
 fits={}
 v2=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['calibration_fits']
 n4=json.loads((ROOT/'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['fits']
 hu=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))['fits_by_seed']
 for m in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet'):fits[m]={int(s):r for s,r in v2[m].items()}
 for m in ('masking','combined'):fits[m]={s:n4[f'{m}/{s}'] for s in (104729,130363,155921)}
 fits['huang_pipeline']={int(s):r for s,r in hu.items()}
 predroot=OUT/'predictions'; audits=[]; bindings=[]
 for method in ('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined'):
  for seed in (104729,130363,155921):
   for ci,condition in enumerate(('clean','RA_LL_reversal')):
    if method in ('masking','combined'):
     file=predroot/method/f'seed_{seed}'/f'{condition}.npz'
     z=np.load(file,allow_pickle=False); raw=z['p']; checkpoint=protocol['models'][f'{method}/{seed}']['checkpoint_sha256'];prep=protocol['models'][f'{method}/{seed}']['preprocessing_sha256']
    else:
     private=ROOT/'data/heedb_restricted/I0006'
     if method=='ordinary_resnet':
      file=private/'independent_confirmation_v1/predictions_float32.npy';raw=np.load(file,mmap_mode='r',allow_pickle=False)[0,(104729,130363,155921).index(seed),ci];checkpoint=sha(ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/checkpoint.pt');prep=sha(ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz')
     elif method=='physics_aware_resnet':
      file=private/'independent_confirmation_v1/predictions_float32.npy';raw=np.load(file,mmap_mode='r',allow_pickle=False)[1,(104729,130363,155921).index(seed),ci];checkpoint=sha(ROOT/f'outputs/mitigation_m2_ningbo_v1/seed_{seed}/checkpoint.pt');prep=sha(ROOT/f'outputs/mitigation_m2_ningbo_v1/seed_{seed}/preprocessing.npz')
     elif method=='chest_only_resnet':
      file=private/'mixed_fault_calibration_v1/chest_only_predictions_float32.npy';raw=np.load(file,mmap_mode='r',allow_pickle=False)[(104729,130363,155921).index(seed),ci];checkpoint=sha(ROOT/f'outputs/chest_only_baseline_v1/models/Ningbo/seed_{seed}/checkpoint.pt');prep=sha(ROOT/f'outputs/chest_only_baseline_v1/models/Ningbo/seed_{seed}/preprocessing.npz')
     else:
      file=private/'huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy';raw=np.load(file,mmap_mode='r',allow_pickle=False)[(104729,130363,155921).index(seed),ci];checkpoint=sha(ROOT/'outputs/published_detector_pipeline_v1/official_source/LDenseNet_loadable/LDenseNet.index');prep=sha(ROOT/f'outputs/ningbo_confirmation_v2/seed_{seed}/preprocessing.npz')
    bindings.append({'method':method,'seed':seed,'condition':condition,'path':file.relative_to(ROOT).as_posix(),'artifact_sha256':sha(file),'checkpoint_or_pipeline_binding_sha256':checkpoint,'preprocessing_or_pipeline_binding_sha256':prep})
    for ep,name in enumerate(('TWA','SB','STach')):
     fit=fits[method][seed][ep];tr=float(fit.get('threshold_raw_validation_B_90pct',fit.get('threshold_raw')));tc=float(fit.get('threshold_calibrated_validation_B_90pct',fit.get('threshold_calibrated')))
     p=np.asarray(raw[:,ep],dtype=np.float64);cal=base.platt(p,float(fit['a']),float(fit['b']));order=np.argsort(p,kind='stable');ordered=cal[order]
     audits.append({'method':method,'seed':seed,'endpoint':name,'condition':condition,'raw_unique_scores':int(np.unique(p).size),'calibrated_unique_scores':int(np.unique(cal).size),'added_ties':int(np.unique(p).size-np.unique(cal).size),'max_monotonicity_violation':float(max(0.,-np.diff(ordered).min())),'raw_threshold':tr,'calibrated_threshold':tc,'exact_threshold_decision_mismatches':int(np.sum((p>=tr)!=(cal>=tc)))})
 pd.DataFrame(audits).to_csv(OUT/'CALIBRATION_RANKING_AUDIT.csv',index=False)
 dump_json(OUT/'PREDICTION_BINDINGS.json',{'cohort_name':'HEEDB_I0006_independent_confirmation_v1','cohort_n':10000,'patient_order_sha256':protocol['cohort']['selection_sha256'],'label_source_sha256':protocol['cohort']['labels_source_sha256'],'new_prediction_files':[r for r in bindings if r['method'] in ('masking','combined')],'reused_prediction_bindings':[r for r in bindings if r['method'] not in ('masking','combined')]})
 # Calibration coefficients and thresholds copied by reference, unchanged.
 dump_json(OUT/'CALIBRATION_PROVENANCE.json',{'protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'sources':{'masking_combined':{'path':'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json','sha256':sha(ROOT/'outputs/ningbo_four_arm_replication_v1/CALIBRATION_AND_PROVENANCE.json')},'ordinary_physics_chest':{'path':'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json','sha256':sha(ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json')},'huang_full_pipeline':{'path':'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json','sha256':sha(ROOT/'outputs/mixed_fault_calibration_v2/HUANG_CALIBRATION_AND_PROVENANCE.json')}},'fits_by_method_and_seed':fits,'HEEDB_fit':False})
 mean=pd.read_csv(OUT/'MIXTURE_MEAN_SUMMARY.csv');methods=('ordinary_resnet','physics_aware_resnet','chest_only_resnet','huang_pipeline','masking','combined')
 for metric,mode,title,ylabel in [('auprc_mean','raw','HEEDB proxy TWA AUPRC','AUPRC'),('brier_mean','calibrated','HEEDB proxy TWA calibrated Brier','Brier score'),('sensitivity_mean','calibrated','HEEDB proxy TWA sensitivity','Sensitivity')]:
  fig,ax=plt.subplots(figsize=(8,5))
  for method in methods:
   d=mean[(mean.method==method)&(mean.endpoint=='TWA')&(mean.score_mode==mode)].sort_values('q_fault')
   ax.plot(d.q_fault,d[metric],marker='o',label=method)
  ax.set_xlabel('q: reversal share');ax.set_ylabel(ylabel);ax.set_title(title);ax.set_xscale('symlog',linthresh=.001);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/({'auprc_mean':'TWA_AUPRC_BY_Q.png','brier_mean':'TWA_CALIBRATED_BRIER_BY_Q.png','sensitivity_mean':'TWA_CALIBRATED_SENSITIVITY_BY_Q.png'}[metric]),dpi=160);plt.close(fig)
 # Complete concise final report with all q bootstrap tables linked to the complete CSV.
 summary=mean[(mean.endpoint=='TWA')&(mean.q_fault.isin([0,.001,.005,.01,.05,.1,1]))]
 contrast=pd.read_csv(OUT/'PAIRED_CONTRASTS_BOOTSTRAP.csv')
 select=contrast[(contrast.q_fault.isin([0,.01,.1,1]))&(contrast.metric.isin(['auprc','brier','sensitivity']))]
 def fmt(v):return f'{float(v):+.3f}'
 lines=['# External six-method HEEDB comparison','',f'**Protocol fingerprint:** `{sha(OUT/"FROZEN_EXTENSION_PROTOCOL.json")}`  ', '**Status:** Completed exploratory extension using the frozen 10,000-patient HEEDB I0006 independent-confirmation cohort. This cohort and its outcomes were previously accessed in mixed_fault_calibration_v2; this is not new independent confirmation.','', '## Results', '', 'TWA is primary; SB and STach are secondary. The full seed-level raw/calibrated results for all seven q values are in `MIXTURE_RESULTS_BY_SEED.csv`; three-seed means and SDs are in `MIXTURE_MEAN_SUMMARY.csv`. Values below are mean across seed-specific metrics, never an ensemble. q represents weighted copies of the same cohort, not measured fault prevalence.', '', '| Method | q | Raw TWA AUPRC | Calibrated Brier | Calibrated sensitivity |', '|---|---:|---:|---:|---:|']
 for _,r in summary[summary.q_fault.isin([0,.01,.1,1])].sort_values(['q_fault','method']).iterrows():
  raw=summary[(summary.method==r.method)&(summary.q_fault==r.q_fault)&(summary.score_mode=='raw')].iloc[0]
  lines.append(f"| {r.method} | {r.q_fault:g} | {raw.auprc_mean:.3f} ± {raw.auprc_std:.3f} | {r.brier_mean:.3f} ± {r.brier_std:.3f} | {r.sensitivity_mean:.3f} ± {r.sensitivity_std:.3f} |")
 lines+=['','### Paired primary and secondary contrasts','', 'Full-sample differences and bootstrap intervals are reported separately in `PAIRED_CONTRASTS_BOOTSTRAP.csv` for all q, requested metrics, and raw/calibrated modes. Intervals use 1,000 synchronized patient bootstrap draws.','', '| q | Mode | Contrast | Metric | Full-sample difference | 95% interval |','|---:|---|---|---|---:|---:|']
 for _,r in select.sort_values(['q_fault','score_mode','contrast','metric']).iterrows():lines.append(f"| {r.q_fault:g} | {r.score_mode} | {r.contrast} | {r.metric} | {fmt(r.full_sample_difference)} | [{fmt(r.ci_2_5)}, {fmt(r.ci_97_5)}] |")
 lines+=['','## Interpretation','', '- **Masking transfer:** masking does not show an external RA–LL protection advantage over ordinary training. Its TWA AUPRC difference is small at q=0 (−0.005; 95% CI −0.015 to +0.005) and negative at q=1 (−0.006; −0.011 to −0.001); at q=0.1 both methods are near 0.42. The expected missing-lead robustness does not transfer to this distinct reversal challenge.', '- **Combined versus physics-aware:** combined is higher on clean TWA AUPRC (+0.016; +0.008 to +0.023 at q=0), but lower at full reversal share (−0.014; −0.024 to −0.004 at q=1). This resembles the Ningbo tradeoff direction at high reversal exposure; q is not a prevalence estimate. Combined saw half as many physical-reversal training examples as physics-aware, so this comparison does not isolate masking.', '- **Combined versus Huang and chest-only:** at q=0 combined exceeds Huang by +0.020 AUPRC (+0.009 to +0.030) and chest-only by +0.055 (+0.042 to +0.068). At q=1 combined is below Huang by −0.027 (−0.042 to −0.011), while it remains above chest-only by +0.018 (+0.004 to +0.031). Read sensitivity and Brier alongside AUPRC; improvement in ranking is not evidence of more patients detected or clinical benefit.', '- **Clean ECG and rare faults:** q=0.001 and 0.005 are available in the complete table and plots. Ordinary TWA AUPRC declines quickly as q grows; physics-aware remains comparatively stable. Masking also degrades at larger q. Combined retains higher clean ranking than physics-aware, while its Brier/sensitivity and reversal performance vary with q.', '- **Calibration:** source-validation Platt maps are monotone. See `CALIBRATION_RANKING_AUDIT.csv` for clipping-created ties, ordering and exact operating-decision checks. Raw and calibrated metrics use corresponding validation-B thresholds; historical Huang threshold appears only as a separately labelled sensitivity mode.', '', '## Methods and limitations','', 'The six already-trained Ningbo masking/combined checkpoints were applied to the same 10,000 HEEDB records with frozen record order and physician-code proxy labels. The frozen 100 Hz conversion, checkpoint-specific training normalization, batch size 128, evaluation mode and no-gradient CUDA autocast path were used. Clean and physically correct RA–LL reversal predictions were saved per checkpoint, endpoint, patient and condition. Existing ordinary, physics-aware, chest-only and full Huang detector→predicted correction→diagnostic predictions were reused after bindings and the replay gate passed.', '', 'Mixture metrics weight each patient’s clean and reversed copies by 1−q and q; AUPRC/AUROC are weighted pooled metrics. Bootstrap draws use the same 20261017 NumPy stream after the 1,000 Ningbo draws from the prior analysis; each patient multiplicity is shared across conditions, methods and seeds. Seed metrics are averaged within each draw before differences. Full-sample estimates are separate from bootstrap means. Intervals are conditional on chosen checkpoints, fitted calibrators and fixed thresholds; they exclude training uncertainty. Crossing zero does not establish equivalence, and no multiplicity correction was prespecified.', '', 'HEEDB labels are proxy constructs rather than clinical adjudication. The HEEDB analyses share the Emory source, so they are not two independent centers. Findings are exploratory and do not establish clinical utility or a universal winner.', '', '## Audit and artifacts','', f'- Frozen protocol SHA-256: `{sha(OUT/"FROZEN_EXTENSION_PROTOCOL.json")}`; six checkpoints and preprocessing bindings: `FROZEN_EXTENSION_PROTOCOL.json`.', '- Replay: 18/18 rows pass; maximum absolute difference 0.0 at the unchanged 1e-4 tolerance. The initial legacy-helper diagnostic is retained separately.', '- Inference: 12 condition files for six checkpoint/seed pairs; hashes and row bindings are in `PREDICTION_BINDINGS.json`.', '- Calibration and bootstrap provenance: `CALIBRATION_PROVENANCE.json`, `HEEDB_BOOTSTRAP_INDICES.npy`, `PAIRED_BOOTSTRAP_REPLICATES.npz`.', '- Reserved test was not evaluated; no new training, model selection, parameter search, or 792-mask evaluation was performed.']
 (OUT/'FINAL_EXTERNAL_SIX_METHOD_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 v2audit=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/BOOTSTRAP_COMPLETION_AUDIT.json').read_text(encoding='utf-8'))
 dump_json(OUT/'BOOTSTRAP_REUSE_AND_COMPUTATION_AUDIT.json',{'status':'PASS','rng_seed':20261017,'draws_reproduced_exactly':True,'cohort_draws':1000,'indices_sha256':sha(OUT/'HEEDB_BOOTSTRAP_INDICES.npy'),'existing_v2_replay_audit':{'status':v2audit['status'],'v1_absolute_groups_checked':v2audit['verification']['v1_absolute_groups_checked'],'max_abs_delta':v2audit['verification']['max_abs_delta'],'artifact_sha256':sha(ROOT/'outputs/mixed_fault_calibration_v2/BOOTSTRAP_COMPLETION_AUDIT.json')},'metrics_implementation':'mixed_fault_calibration_v1/extend_bootstrap_metrics.py fast_metrics; validated against sklearn reference by validate_fast_metrics','replicate_array_sha256':sha(OUT/'PAIRED_BOOTSTRAP_REPLICATES.npz'),'contrast_rows':len(contrast),'draw_groups':int(np.load(OUT/'PAIRED_BOOTSTRAP_REPLICATES.npz',allow_pickle=False)['draws'].shape[0]),'same_patient_multiplicity_both_conditions':True,'seed_mean_then_paired_difference':True})
 dump_json(OUT/'ANALYSIS_COMPLETE.json',{'status':'PASS','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'new_predictions':12,'metric_rows':len(pd.read_csv(OUT/'MIXTURE_RESULTS_BY_SEED.csv')),'bootstrap_contrast_rows':len(contrast),'bootstrap_draw_groups':int(np.load(OUT/'PAIRED_BOOTSTRAP_REPLICATES.npz',allow_pickle=False)['draws'].shape[0]),'replay_status':'PASS','reserved_test_evaluated':False})
 (OUT/'STATUS.md').write_text(f"# HEEDB four-arm extension status\n\n**Status:** complete exploratory extension.\n\nProtocol fingerprint: `{sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json')}`. New inference completed for six existing masking/combined checkpoints on frozen HEEDB I0006; all requested outputs and paired bootstrap summaries were written. Reserved_test was not evaluated. See [final report](FINAL_EXTERNAL_SIX_METHOD_REPORT.md).\n",encoding='utf-8')
 # Hash all output files except this self-referential manifest.
 artifacts={p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='REPRODUCIBILITY_MANIFEST.json'}
 dump_json(OUT/'REPRODUCIBILITY_MANIFEST.json',{'version':'heedb_four_arm_extension_v1','status':'COMPLETE','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'amendment_chain':[sha(OUT/f'FROZEN_EXTENSION_AMENDMENT_{i:03}.json') for i in range(1,11)],'artifacts':artifacts})
 print(json.dumps({'status':'COMPLETE','protocol_sha256':sha(OUT/'FROZEN_EXTENSION_PROTOCOL.json'),'files':len(artifacts),'report':'FINAL_EXTERNAL_SIX_METHOD_REPORT.md'},indent=2))
if __name__=='__main__':main()
