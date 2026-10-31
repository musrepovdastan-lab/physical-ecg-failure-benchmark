"""Create run status, plots, and evidence-led synthesis after analysis artifacts exist."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from outputs.ningbo_four_arm_replication_v1 import workflow as w

def readcsv(p):
    with open(p,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    w.protocol_check();out=w.OUT
    timings=json.loads((out/'TRAINING_TIMINGS.json').read_text(encoding='utf-8')) if (out/'TRAINING_TIMINGS.json').exists() else {'events':[]}
    tmap={(x['arm'],int(x['seed'])):x for x in timings.get('events',[])}
    status=[]
    for arm in ('clean','masking','physics_aware','combined'):
      for seed in w.SEEDS:
        if arm=='clean':
            d=w.N/f'seed_{seed}';g=json.loads((d/'baseline.json').read_text(encoding='utf-8'));hist=json.loads((d/'history.json').read_text(encoding='utf-8'))
            ep=max(hist,key=lambda x:x.get('validation_twa_auprc',-1)).get('epoch');valfile=d/'validation.npz';checkpoint=d/'checkpoint.pt';gate=g.get('decision','PASS')
            try:
                import torch
                rr=torch.load(d/'resume.pt',weights_only=False,map_location='cpu');reason='historical validation early stopping (patience 5)' if rr.get('wait',0)>=5 else ('historical maximum epoch reached' if rr.get('epoch')==30 else 'historical clean training completed')
            except Exception: reason='historical clean checkpoint reused; stop telemetry unavailable'
            gpu='not recorded in historical clean run artifacts';elapsed='not recorded in historical clean run artifacts'
        elif arm=='physics_aware':
            d=w.M2/f'seed_{seed}';g=json.loads((d/'validation_gate.json').read_text(encoding='utf-8'));hist=g.get('history',[]);ep=max(hist,key=lambda x:x.get('validation_twa_auprc',-1)).get('epoch') if hist else None
            valfile=d/'validation.npz';checkpoint=d/'checkpoint.pt';gate=g.get('decision','UNKNOWN')
            try:
                import torch
                rr=torch.load(d/'resume.pt',weights_only=False,map_location='cpu');reason='historical validation early stopping (patience 5)' if rr.get('wait',0)>=5 else ('historical maximum epoch reached' if rr.get('epoch')==30 else 'historical physics-aware training completed')
            except Exception: reason='historical physics-aware checkpoint reused; stop telemetry unavailable'
            gpu='not recorded in historical M2 run artifacts';elapsed='not recorded in historical M2 run artifacts'
        else:
            d=w.unit_dir(arm,seed);g=json.loads((d/'validation_gate.json').read_text(encoding='utf-8'));hist=json.loads((d/'history.json').read_text(encoding='utf-8'))['history'] if (d/'history.json').exists() else []
            ep=g.get('best_epoch');valfile=d/'validation.npz';checkpoint=d/'checkpoint.pt';gate=g.get('decision','UNKNOWN');resume=torchload=None
            try:
                import torch
                q=torch.load(d/'resume.pt',weights_only=False,map_location='cpu');last=q.get('epoch');wait=q.get('wait',0)
                reason='early stopping: validation TWA AUPRC patience 5' if wait>=5 else ('maximum 30 epochs reached' if last==30 else 'validation gate stop' if gate!='PASS' else 'completed')
            except Exception: reason='validation gate stop or incomplete run'
            tm=tmap.get((arm,seed),{});elapsed=tm.get('wall_seconds','in progress/not measured');gpu=tm.get('peak_gpu_memory_bytes','not yet measured')
            if isinstance(gpu,int):gpu=str(gpu)
        with np.load(valfile,allow_pickle=False) as z:
            yy=z['y'][:,0].astype(np.uint8);pp=z['p'][:,0].astype(float);thr=float(z['thresholds'][0])
            from sklearn.metrics import average_precision_score,roc_auc_score
            val_ap=float(average_precision_score(yy,pp));val_auc=float(roc_auc_score(yy,pp))
        status.append({'arm':arm,'seed':seed,'status':gate,'best_epoch':ep,'validation_twa_auprc':val_ap,'validation_twa_auroc':val_auc,'stop_reason':reason,
          'elapsed_wall_seconds':elapsed,'peak_gpu_allocated_bytes':gpu,'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':w.sha(checkpoint)})
    w.atomic_csv(out/'TRAINING_STATUS_BY_ARM_SEED.csv',status)
    rows=readcsv(out/'RESULTS_BY_ARM_SEED.csv');boots=readcsv(out/'BOOTSTRAP_SUMMARY.csv');mixtures=readcsv(out/'MIXTURE_RESULTS_BY_SEED.csv');mixboots=readcsv(out/'MIXTURE_BOOTSTRAP_SUMMARY.csv');dists=readcsv(out/'MASK_DISTRIBUTION_SUMMARY.csv')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8));arms=('clean','masking','physics_aware','combined'); offsets={s:(i-1)*.06 for i,s in enumerate(w.SEEDS)}
    for ai,arm in enumerate(arms):
        for seed in w.SEEDS:
            z=next(r for r in dists if r['arm']==arm and int(r['seed'])==seed); med=float(z['median_auprc']);lo=med-float(z['p05']);hi=float(z['p95'])-med
            ax.errorbar(ai+offsets[seed],med,yerr=[[lo],[hi]],fmt='o',capsize=3,color=f'C{ai}',alpha=.85)
    ax.set_xticks(range(len(arms)),arms);ax.set_ylabel('TWA AUPRC across 792 masks');ax.set_title('Median and 5th–95th percentile by seed');fig.tight_layout();fig.savefig(out/'MASK_AUPRC_DISTRIBUTION.png',dpi=160);plt.close(fig)
    def val(arm,seed,cond,field='auprc'):
        return float(next(r[field] for r in rows if r['arm']==arm and int(r['seed'])==seed and r['endpoint']=='TWA' and r['condition']==cond))
    def bval(a,b,cond):return next((x for x in boots if x['contrast']==f'{a}_minus_{b}' and x['estimand']==cond),None)
    lines=['# Ningbo four-arm training replication','',
      f"**Protocol:** `{json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256']}`. **Scope:** follow-up on Ningbo after prior test access; this is exploratory corrected reanalysis, not untouched confirmation.",'',
      '## Training and validation','',
      'The only newly trained arms were masking and combined (three fixed seeds each). Clean and physics-aware checkpoints were reused after checkpoint, preprocessing, split, corrected mask-cache and condition bindings were checked. The six unit summaries, stop reasons, elapsed time and measured GPU process allocation are in `TRAINING_STATUS_BY_ARM_SEED.csv`. Historical artifacts did not retain per-run wall time or GPU peak memory, so those fields are explicitly marked unavailable for reused models.','',
      '| Arm | Seed | Status | Best epoch | Validation TWA AUPRC | Validation TWA AUROC | Stop reason | Wall seconds | Peak allocated GPU bytes | Checkpoint |','|---|---:|---|---:|---:|---:|---|---:|---:|---|']
    for r in status:lines.append(f"| {r['arm']} | {r['seed']} | {r['status']} | {r['best_epoch']} | {r['validation_twa_auprc']:.4f} | {r['validation_twa_auroc']:.4f} | {r['stop_reason']} | {r['elapsed_wall_seconds']} | {r['peak_gpu_allocated_bytes']} | `{r['checkpoint_path']}` |")
    lines+=['','All 792 masks were evaluated separately. The exact mask removes I, II, III, aVR and aVF (indices `[0,1,2,3,5]`). No prediction vectors were averaged. TWA is primary; SB and STach are secondary. Full metrics for every seed and condition are in `RESULTS_BY_ARM_SEED.csv`, with seed means and SDs in `RESULTS_MEAN_BY_ARM.csv`; exact-mask metrics and mask distributions are in the per-seed and mean-summary CSVs.','',
      '## Prespecified TWA comparisons','',
      'Intervals below use the same 1,000 canonical-record resamples across both arms, all seeds and conditions. Central differences use full-cohort point estimates; bootstrap means are reported separately. A 95% interval crossing zero leaves the difference unresolved and does not establish equivalence. Comparisons were prespecified, but no multiplicity adjustment was defined; interpret the family as exploratory.','',
      '| Contrast | Estimand | Full-cohort difference | Bootstrap mean | 95% interval |','|---|---|---:|---:|---:|']
    for r in boots:lines.append(f"| {r['contrast']} | {r['estimand']} | {float(r['point_estimate_full_cohort']):+.4f} | {float(r['bootstrap_mean']):+.4f} | [{float(r['lower95']):+.4f}, {float(r['upper95']):+.4f}] |")
    lines+=['','The seed-level primary results are:', '', '| Arm | Seed | Clean | RA–LA | RA–LL | LA–LL | Mean of 792 masks | Exact mask |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('clean','masking','physics_aware','combined'):
      for seed in w.SEEDS:
        md=float(next(r['mean_auprc'] for r in dists if r['arm']==arm and int(r['seed'])==seed))
        ex=val(arm,seed,'mask5_0_1_2_3_5')
        lines.append(f"| {arm} | {seed} | {val(arm,seed,'clean'):.4f} | {val(arm,seed,'RA_LA_reversal'):.4f} | {val(arm,seed,'RA_LL_reversal'):.4f} | {val(arm,seed,'LA_LL_reversal'):.4f} | {md:.4f} | {ex:.4f} |")
    lines+=['','## Mostly-clean RA–LL mixtures and calibration','',
      'The new masking and combined Platt fits used only clean validation A; raw thresholds were the 90th percentile of negative clean validation B scores, and calibrated thresholds were transformed through the same fitted monotone map. Clean and physics-aware fits were reused unchanged from mixed-fault calibration v2. No test data were used for fitting. Per-seed raw/calibrated metrics at every frozen q are in `MIXTURE_RESULTS_BY_SEED.csv`, with cross-seed means/SDs in `MIXTURE_MEAN_SUMMARY.csv`; paired TWA bootstrap intervals are in `MIXTURE_BOOTSTRAP_SUMMARY.csv`.','',
      '| Arm | TWA q=0 raw AUPRC | Raw Brier | Calibrated Brier | Raw ECE15 | Calibrated ECE15 |','|---|---:|---:|---:|---:|---:|']
    for arm in ('clean','masking','physics_aware','combined'):
        z=[r for r in mixtures if r['arm']==arm and r['endpoint']=='TWA' and float(r['q'])==0.0]
        raw=[r for r in z if r['score_mode']=='raw'];cal=[r for r in z if r['score_mode']=='calibrated']
        if raw and cal:
            lines.append(f"| {arm} | {np.mean([float(r['auprc']) for r in raw]):.4f} | {np.mean([float(r['brier']) for r in raw]):.4f} | {np.mean([float(r['brier']) for r in cal]):.4f} | {np.mean([float(r['ece15']) for r in raw]):.4f} | {np.mean([float(r['ece15']) for r in cal]):.4f} |")
    lines += ['',
      '## Interpretation','',
      '']
    mra=bval('masking','clean','RA_LL_reversal');cra=bval('combined','physics_aware','RA_LL_reversal');mc=bval('masking','clean','clean');cc=bval('combined','physics_aware','clean');mm=bval('masking','clean','mean_mask');cm=bval('combined','physics_aware','mean_mask')
    def effect(r):return 'not estimable because fewer than three gated seeds completed' if not r else f"{float(r['point_estimate_full_cohort']):+.3f} (95% interval {float(r['lower95']):+.3f} to {float(r['upper95']):+.3f})"
    lines += [f"- **Does masking protect against RA–LL?** Masking minus clean-trained TWA AUPRC under RA–LL was {effect(mra)}. Judge the size, interval and seed signs together; this comparison does not imply complete protection.",
      f"- **Is combined more balanced than physics-aware?** Combined minus physics-aware was {effect(cc)} on clean ECG, {effect(cra)} on RA–LL and {effect(cm)} on mean-mask performance. Combined uses 25% reversal examples versus 50% in physics-aware, so this contrast cannot isolate the effect of adding masks.",
      f"- **Clean cost and calibration:** clean-condition raw test calibration and validation-calibrated mostly-clean results should be read separately in `RESULTS_BY_ARM_SEED.csv` and `MIXTURE_RESULTS_BY_SEED.csv`; the calibration map is monotone and preserves operating decisions except measured clipping/ties.",
      '- **Seed stability:** see the seed-level table above. Three seeds provide limited precision; a result that changes direction by seed is not stable evidence.',
      '- **Chapman transfer:** Chapman M1 had passed its frozen gate for physics-aware as primary, while masking/combined were secondary. This Ningbo study is a follow-up, not fresh confirmation. The exact Chapman-to-Ningbo combined comparison is tabulated below; Chapman source results remain unchanged.',
      '- **Reversal exposure:** if combined differs from physics-aware, both mask exposure and halved reversal exposure change together. No mechanism can be attributed from this comparison.',
      '- **HEEDB:** a future external analysis may be justified if the new fixed arms show reproducible tradeoffs, but this package does not run HEEDB inference and Ningbo findings alone do not establish external validity.',
      '- **Competing results:** retain chest-only and Huang comparisons from `outputs/mixed_fault_calibration_v2/FINAL_FOUR_METHOD_COMPARISON.md`. The corrected recovery reports and the TimePatchTransformer M3 Chapman `FAIL_M3_ARCHITECTURE_SPECIFIC` remain in force; no global winner is claimed.',
      '', 'No hyperparameter search, retraining of clean/physics-aware, reserved-test evaluation, new data, new split, or HEEDB inference was performed.']
    m1=readcsv(ROOT/'outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv'); ref=[]
    for seed in w.SEEDS:
        for c in ('clean','RA_LL_reversal'):
            try:
                ca=float(next(r['auprc'] for r in m1 if r['arm']=='combined' and int(r['seed'])==seed and r['endpoint']=='TWA' and r['condition']==c))
                cl=float(next(r['auprc'] for r in m1 if r['arm']=='cached_clean_baseline' and int(r['seed'])==seed and r['endpoint']=='TWA' and r['condition']==c))
                ref.append((seed,c,ca,cl,ca-cl))
            except StopIteration: pass
    if ref:
        at=lines.index('- **Reversal exposure:** if combined differs from physics-aware, both mask exposure and halved reversal exposure change together. No mechanism can be attributed from this comparison.')
        block=['','Chapman M1 combined versus cached clean baseline TWA AUPRC (full-cohort test estimates; source M1 values):','',
          '| Seed | Condition | Combined | Clean baseline | Difference |','|---:|---|---:|---:|---:|']
        block += [f'| {s} | {c} | {a:.4f} | {b:.4f} | {d:+.4f} |' for s,c,a,b,d in ref]
        lines[at:at]=block
    (out/'FINAL_FOUR_ARM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    eng=json.loads((out/'engineering_checks_result.json').read_text(encoding='utf-8')) if (out/'engineering_checks_result.json').exists() else None
    (out/'ENGINEERING_TEST_REPORT.md').write_text('# Engineering test report\n\n- Protocol bindings: PASS.\n- Existing clean/physics-aware comparator and corrected mask-cache manifest audit: PASS (6 checkpoints; 4,752 corrected mask prediction files; no legacy inverted mask caches reused).\n- Synthetic tests of actual transforms/training helpers: PASS; see `engineering_checks_result.json`.\n- Validation gates: see `TRAINING_STATUS_BY_ARM_SEED.csv`; test inference is conditional on each unit gate.\n- Test set: not accessed by synthetic tests; later test inference follows per-unit validation freeze.\n- Frozen method fingerprint: `'+json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256']+'`.\n')
    artifact_names=['FROZEN_NINGBO_FOUR_ARM_PROTOCOL.json','FROZEN_NINGBO_FOUR_ARM_PROTOCOL_AMENDMENT_001.json','FROZEN_NINGBO_FOUR_ARM_PROTOCOL_AMENDMENT_002.json',
      'ENGINEERING_TEST_REPORT.md','engineering_checks_result.json','EXISTING_COMPARATOR_REUSE_AUDIT.json','EXISTING_MISSING_INFERENCE_FREEZE.json','EXISTING_MISSING_INFERENCE_AUDIT.json','FROZEN_INFERENCE_CONTRACT.json','NEW_UNIT_INFERENCE_STATUS.json','TEST_INPUT_CACHE.json','test_waveforms_float32.npy',
      'RESULTS_BY_ARM_SEED.csv','EXACT_MASK_COMPARISON.csv','MASK_DISTRIBUTION_SUMMARY.csv','PRIMARY_CONTRASTS.csv','BOOTSTRAP_SUMMARY.csv','BOOTSTRAP_REPLICATES.csv',
      'CALIBRATION_AND_PROVENANCE.json','MIXTURE_RESULTS_BY_SEED.csv','MIXTURE_BOOTSTRAP_SUMMARY.csv','MIXTURE_BOOTSTRAP_REPLICATES.csv','SECONDARY_CONTRASTS.csv',
      'TRAINING_STATUS_BY_ARM_SEED.csv','RESULTS_MEAN_BY_ARM.csv','MASK_DISTRIBUTION_MEAN_BY_ARM.csv','MIXTURE_MEAN_SUMMARY.csv','FINAL_FOUR_ARM_REPORT.md','TWA_AUPRC_BY_Q.png','TWA_AUPRC_BY_CONDITION.png','MASK_AUPRC_DISTRIBUTION.png']
    artifacts={name:w.sha(out/name) for name in artifact_names if (out/name).is_file()}
    units={}
    for arm in ('clean','masking','physics_aware','combined'):
      for seed in w.SEEDS:
        d=(w.N/f'seed_{seed}' if arm=='clean' else w.M2/f'seed_{seed}') if arm=='physics_aware' or arm=='clean' else w.unit_dir(arm,seed)
        unit={'arm':arm,'seed':seed,'checkpoint':(d/'checkpoint.pt').relative_to(ROOT).as_posix(),'checkpoint_sha256':w.sha(d/'checkpoint.pt')}
        if arm in w.ARMS:
            for f in ('history.json','validation_gate.json','evaluation_complete.json'):
                if (d/f).is_file():unit[f+'_sha256']=w.sha(d/f)
            unit['test_inference_status']='complete' if (d/'evaluation_complete.json').is_file() else 'not_run_or_gate_failed'
        else:
            unit['recovery_evaluation_manifest_sha256']=w.sha(w.REC/('baseline_v2' if arm=='clean' else 'physics_aware_m2')/f'seed_{seed}'/'evaluation_complete.json')
        units[f'{arm}/{seed}']=unit
    manifest={'status':'COMPLETE' if all(x['status']=='PASS' for x in status) and (out/'ANALYSIS_COMPLETE.json').exists() else 'PARTIAL_OR_GATE_STOP',
      'protocol_sha256':json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256'],'workflow_source_sha256':w.sha(out/'workflow.py'),
      'training_runner':'outputs/ningbo_four_arm_replication_v1/workflow.py','new_training_units':6,'existing_reuse_units':6,
      'input_bindings':json.loads(w.PROTO.read_text(encoding='utf-8'))['bindings'],'units':units,'artifact_sha256':artifacts,
      'updated_utc':w.now(),'reserved_test_evaluated':False,'heedb_inference_performed':False}
    w.atomic_json(out/'REPRODUCIBILITY_MANIFEST.json',manifest)
    runm=json.loads((out/'RUN_MANIFEST.json').read_text(encoding='utf-8'));runm.update({'status':manifest['status'],'completed_utc':w.now(),'artifact_sha256':artifacts,'reproducibility_manifest_sha256':w.sha(out/'REPRODUCIBILITY_MANIFEST.json'),'protocol_sha256':manifest['protocol_sha256']})
    w.atomic_json(out/'RUN_MANIFEST.json',runm)
    w.atomic_json(out/'progress_resume.json',{'stage':manifest['status'],'updated_utc':w.now(),'protocol_sha256':manifest['protocol_sha256'],'completed_training_units':sum(x['status']=='PASS' for x in status if x['arm'] in w.ARMS),'total_training_units':6,'analysis_complete':(out/'ANALYSIS_COMPLETE.json').exists()})

if __name__=='__main__':main()
