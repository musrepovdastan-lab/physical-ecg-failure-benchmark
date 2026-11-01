from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'outputs/calibration_role_control_v1'
COMP = ROOT / 'outputs/calibration_role_control_completion_v2'
COHORTS = ('Ningbo_test', 'HEEDB')
METHODS = ('ordinary', 'physics_aware_p050', 'combined', 'Huang_full_pipeline')
SEEDS = (104729, 130363, 155921)
SCENARIOS = ('RA_LL_only', 'equal_three_swap')
QS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
CONTRASTS = (('physics_aware_p050','ordinary'), ('combined','ordinary'),
             ('Huang_full_pipeline','ordinary'), ('physics_aware_p050','Huang_full_pipeline'),
             ('combined','Huang_full_pipeline'))
METRICS = ('AP_raw','Brier_raw','Brier_calibrated','sensitivity_raw','FP_per_1000_raw')

def write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')

def main():
    mix0 = pd.read_csv(OLD/'MIXTURE_METRICS_BY_SEED.csv')
    mix1 = pd.read_csv(OUT/'MIXTURE_METRICS_BY_SEED.csv')
    key = ['cohort','method','seed','endpoint','scenario','q_fault']
    metrics = ('AP_raw','AP_calibrated','Brier_raw','Brier_calibrated','sensitivity_raw',
               'sensitivity_calibrated','FP_per_1000_raw','FP_per_1000_calibrated',
               'AUROC_raw','ECE15_raw','ECE15_calibrated')
    joined = mix0[key+list(metrics)].merge(mix1[key+list(metrics)], on=key, suffixes=('_original','_corrected'), validate='one_to_one')
    if len(joined) != 1008: raise RuntimeError(f'old/new seed-level coverage {len(joined)} != 1008')
    for m in metrics: joined[m+'_delta'] = joined[m+'_corrected'] - joined[m+'_original']
    joined.to_csv(OUT/'SEED_LEVEL_OLD_VS_CORRECTED.csv',index=False,encoding='utf-8')

    # Bind every saved test prediction to frozen record order, labels, method, seed and checkpoint.
    sys.path.insert(0,str(OUT))
    import control as c
    parts,_=c.records(); conditions=('clean',*c.FAULTS); pred_audit=[]
    nids=np.asarray([r['record_id'] for r in parts['test']]).astype(str)
    ny=c.labels(parts['test'])
    private=ROOT/'data/heedb_restricted/I0006/independent_confirmation_v1'
    from outputs.mixed_fault_calibration_v1 import run_analysis as base
    hy,_=base.heedb_labels(); hids=pd.read_csv(private/'selection_initial.csv',dtype=str,keep_default_na=False)['FileName'].astype(str).to_numpy()
    expected={'Ningbo_test':(nids,ny),'HEEDB':(hids,hy)}
    detector_hash={'Ningbo_test':c.sha(OUT/'detector/Ningbo_test.npz'),
                   'HEEDB':c.sha(ROOT/'outputs/three_swap_population_mixture_recovery_v2/huang_detector_probabilities.npz')}
    for cohort in COHORTS:
        ids0,y0=expected[cohort]
        for method in METHODS:
            ckmethod='ordinary' if method=='Huang_full_pipeline' else method
            for seed in SEEDS:
                ck=OUT/'fits'/ckmethod/f'seed_{seed}'/'checkpoint.pt';ckhash=c.sha(ck)
                for condition in conditions:
                    path=OUT/f'predictions/{cohort}/{method}/seed_{seed}/{condition}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        pz=z['p'];yz=z['y'];idz=z['ids'].astype(str)
                        ok=(pz.shape==(len(y0),3) and np.isfinite(pz).all() and np.array_equal(yz,y0)
                            and np.array_equal(idz,ids0) and str(z['method'].item())==method
                            and int(z['seed'].item())==seed and str(z['condition'].item())==condition
                            and str(z['checkpoint_sha256'].item())==ckhash
                            and str(z['source_order_sha256'].item())==hashlib.sha256('\n'.join(ids0).encode()).hexdigest())
                        if method=='Huang_full_pipeline':
                            ok=ok and str(z['detector_probabilities_sha256'].item())==detector_hash[cohort] and str(z['detector_condition'].item())==condition
                    if not ok: raise RuntimeError(f'prediction identity/label/checkpoint binding failed: {path}')
                    pred_audit.append({'cohort':cohort,'method':method,'seed':seed,'condition':condition,'status':'PASS'})
    meanrows=[]
    for group, g in joined.groupby(['cohort','method','endpoint','scenario','q_fault'], sort=False):
        row=dict(zip(['cohort','method','endpoint','scenario','q_fault'],group))
        for m in metrics:
            for tag in ('original','corrected','delta'):
                col=m+'_delta' if tag=='delta' else m+'_'+tag
                row[f'{m}_{tag}_mean']=float(g[col].mean())
                row[f'{m}_{tag}_sd']=float(g[col].std(ddof=1))
        meanrows.append(row)
    write(OUT/'MEAN_OLD_VS_CORRECTED.csv',meanrows)

    oldp=pd.read_csv(COMP/'PAIRED_POINTS_CI_ALL_700.csv')
    newp=pd.read_csv(OUT/'TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv')
    pkey=['cohort','scenario','q_fault','left','right','metric']
    p=oldp.merge(newp,on=pkey,suffixes=('_original','_corrected'),validate='one_to_one')
    if len(p)!=700: raise RuntimeError(f'contrast table coverage {len(p)} != 700')
    with np.load(COMP/'PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz',allow_pickle=False) as z: olddraw={k:z[k].copy() for k in z.files}
    with np.load(OUT/'PAIRED_BOOTSTRAP_DRAWS_ALL.npz',allow_pickle=False) as z: newdraw={k:z[k].copy() for k in z.files}
    if set(olddraw)!=set(newdraw) or len(newdraw)!=700: raise RuntimeError('old/new paired bootstrap draw keys differ')
    chglo=[]; chghi=[]; chgmean=[]
    for _,r in p.iterrows():
        k=f"{r.cohort}|{r.scenario}|q{r.q_fault:g}|{r.left}-{r.right}|{r.metric}"
        d=newdraw[k]-olddraw[k]
        if d.shape!=(1000,) or not np.isfinite(d).all(): raise RuntimeError('invalid paired old-new bootstrap delta '+k)
        chgmean.append(float(np.mean(d))); chglo.append(float(np.quantile(d,.025))); chghi.append(float(np.quantile(d,.975)))
    p['delta_point_corrected_minus_original']=p.full_cohort_point_mean_seed_contrast_corrected-p.full_cohort_point_mean_seed_contrast_original
    p['delta_bootstrap_mean']=chgmean; p['delta_ci95_low']=chglo; p['delta_ci95_high']=chghi
    p.to_csv(OUT/'CONTRAST_OLD_VS_CORRECTED_700.csv',index=False,encoding='utf-8')

    leaders=[]
    for (cohort,scenario,q), g in mix1[(mix1.endpoint=='TWA')].groupby(['cohort','scenario','q_fault']):
        oldg=mix0[(mix0.endpoint=='TWA')&(mix0.cohort==cohort)&(mix0.scenario==scenario)&(mix0.q_fault==q)]
        oldmeans=oldg.groupby('method').AP_raw.mean();newmeans=g.groupby('method').AP_raw.mean()
        oldlead=str(oldmeans.idxmax());newlead=str(newmeans.idxmax())
        leaders.append({'cohort':cohort,'scenario':scenario,'q_fault':q,'original_leader':oldlead,'corrected_leader':newlead,
                        'leader_changed':oldlead!=newlead,'original_leader_AP':float(oldmeans.max()),'corrected_leader_AP':float(newmeans.max())})
    write(OUT/'AP_LEADERSHIP_OLD_VS_CORRECTED.csv',leaders)

    # Directional mid-fault and q=1 checks use paired intervals from each series.
    interpretation=[]
    for cohort in COHORTS:
        for scenario in SCENARIOS:
            for q in (.01,.05,.1,1.0):
                for method in ('physics_aware_p050','combined'):
                    row=p[(p.cohort==cohort)&(p.scenario==scenario)&(np.isclose(p.q_fault,q))&
                          (p.left==method)&(p.right=='ordinary')&(p.metric=='AP_raw')].iloc[0]
                    interpretation.append({'question':'method_vs_ordinary_AP','cohort':cohort,'scenario':scenario,'q_fault':q,'method':method,
                        'original_point':row.full_cohort_point_mean_seed_contrast_original,
                        'original_ci_low':row.ci95_low_original,'original_ci_high':row.ci95_high_original,
                        'corrected_point':row.full_cohort_point_mean_seed_contrast_corrected,
                        'corrected_ci_low':row.ci95_low_corrected,'corrected_ci_high':row.ci95_high_corrected,
                        'corrected_interval_excludes_zero':bool(row.ci95_low_corrected>0 or row.ci95_high_corrected<0)})
                # q=1 AP Huang versus physics-aware and combined; negate method-minus-Huang rows.
                if q==1.0:
                    for method in ('physics_aware_p050','combined'):
                        row=p[(p.cohort==cohort)&(p.scenario==scenario)&(np.isclose(p.q_fault,q))&
                              (p.left==method)&(p.right=='Huang_full_pipeline')&(p.metric=='AP_raw')].iloc[0]
                        interpretation.append({'question':'Huang_vs_method_AP_at_q1','cohort':cohort,'scenario':scenario,'q_fault':q,'method':method,
                            'original_point':-row.full_cohort_point_mean_seed_contrast_original,
                            'original_ci_low':-row.ci95_high_original,'original_ci_high':-row.ci95_low_original,
                            'corrected_point':-row.full_cohort_point_mean_seed_contrast_corrected,
                            'corrected_ci_low':-row.ci95_high_corrected,'corrected_ci_high':-row.ci95_low_corrected,
                            'corrected_interval_excludes_zero':bool(row.ci95_high_corrected<0 or row.ci95_low_corrected>0)})
    write(OUT/'KEY_AP_FINDINGS.csv',interpretation)

    # Independent dimensional, q coverage, and actual sklearn tie checks.
    expected={f'{c}|{s}|q{q:g}|{a}-{b}|{m}' for c in COHORTS for s in SCENARIOS for q in QS for a,b in CONTRASTS for m in METRICS}
    if set(newdraw)!=expected or not all(v.shape==(1000,) and np.isfinite(v).all() for v in newdraw.values()): raise RuntimeError('draw coverage/shape/finiteness failed')
    if sorted(newp.q_fault.unique().tolist())!=list(QS): raise RuntimeError('q coverage failed')
    q0=[]
    for cohort in COHORTS:
        for scenario in SCENARIOS:
            for a,b in CONTRASTS:
                for metric in METRICS:
                    k=f'{cohort}|{scenario}|q0|{a}-{b}|{metric}'
                    if k not in newdraw: raise RuntimeError('q0 group absent')
                    q0.append(k)
    for cohort in COHORTS:
        x=pd.read_csv(OUT/f'{cohort}_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv')
        if x.absolute_delta.max()>1e-12: raise RuntimeError(f'{cohort} weighted AP tie check failed')
    # Group-bootstrap vectors are paired; independently check the two q=0 scenario arrays.
    q0delta=max(float(np.max(np.abs(newdraw[f'{c}|RA_LL_only|q0|{a}-{b}|{m}']-newdraw[f'{c}|equal_three_swap|q0|{a}-{b}|{m}'])))
                for c in COHORTS for a,b in CONTRASTS for m in METRICS)
    if q0delta>1e-14: raise RuntimeError(f'q=0 scenarios differ: {q0delta}')
    if len(newp)!=700 or len(joined)!=1008: raise RuntimeError('table row count mismatch')
    training=json.loads((OUT/'TRAINING_STATUS.json').read_text(encoding='utf8'))
    init=json.loads((OUT/'INITIALIZATION_AUDIT.json').read_text(encoding='utf8'))
    resume=json.loads((OUT/'RNG_RESUME_AUDIT.json').read_text(encoding='utf8'))
    if training.get('completed')!=9 or any(x.get('decision')!='PASS' for x in training['fits']): raise RuntimeError('training fit gate incomplete')
    if init.get('status')!='PASS' or resume.get('status')!='PASS': raise RuntimeError('initialization or resume audit did not pass')
    summary={'status':'PASS','original_models':9,'corrected_models':9,'seed_level_metric_rows':len(joined),
        'prediction_files_verified':len(pred_audit),
        'seed_level_metrics':['AP_raw','AP_calibrated','Brier_raw','Brier_calibrated','sensitivity_raw','sensitivity_calibrated','FP_per_1000_raw','FP_per_1000_calibrated','AUROC_raw','ECE15_raw','ECE15_calibrated'],
        'paired_contrast_groups':len(newp),'draw_groups':len(newdraw),'draws_per_group':1000,
        'q_values':list(QS),'scenarios':list(SCENARIOS),'cohorts':list(COHORTS),'q0_scenario_max_abs_delta':q0delta,
        'sklearn_weighted_AP_max_abs_delta':max(float(pd.read_csv(OUT/f'{c}_BOOTSTRAP_SKLEARN_DRAW_CHECKS.csv').absolute_delta.max()) for c in COHORTS),
        'historical_series':'calibration_role_control_v1; bootstrap intervals from calibration_role_control_completion_v2',
        'interpretation_limit':'Old initial weights were not retained; old-new differences cannot isolate a pure initialization-seed effect.'}
    (OUT/'INDEPENDENT_RESULT_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
