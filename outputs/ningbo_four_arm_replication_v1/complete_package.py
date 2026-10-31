"""Finish missing comparator inference, metrics, paired bootstrap and calibration."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from outputs.ningbo_four_arm_replication_v1 import workflow as w

def import_v1():
    f=ROOT/'outputs/mixed_fault_calibration_v1/run_analysis.py'; spec=importlib.util.spec_from_file_location('frozen_mixture_v1',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def rows_records(): return w.records_by_split()

def existing_source(arm,seed,condition):
    mode='baseline_v2' if arm=='clean' else 'physics_aware_m2'
    rd=w.REC/mode/f'seed_{seed}'
    if condition.startswith('mask5_'):return rd/'test_predictions'/f'{condition}.npz'
    recovered=rd/'verified_conditions'/f'{condition}.npz'
    if recovered.exists():return recovered
    base=w.N/f'seed_{seed}' if arm=='clean' else w.M2/f'seed_{seed}'
    return base/'test_predictions'/f'{condition}.npz'

def pred_path(arm,seed,condition):
    if arm in w.ARMS:return w.unit_dir(arm,seed)/'test_predictions'/f'{condition}.npz'
    if condition in ('clean','RA_LL_reversal') or condition.startswith('mask5_'):return existing_source(arm,seed,condition)
    return w.OUT/'existing_reinference'/arm/f'seed_{seed}'/'test_predictions'/f'{condition}.npz'

def sha(p):return w.sha(p)

def infer_missing_existing(parts):
    gates={};eligible=[]
    for arm in ('clean','physics_aware'):
      for seed in w.SEEDS:
        d=w.N/f'seed_{seed}' if arm=='clean' else w.M2/f'seed_{seed}'
        g=json.loads((d/('baseline.json' if arm=='clean' else 'validation_gate.json')).read_text(encoding='utf-8'))
        if g.get('decision')!='PASS':raise RuntimeError(f'existing comparator failed validation gate: {arm}/{seed}')
        ck=d/'checkpoint.pt';m=d/'preprocessing.npz';gatefile=d/('baseline.json' if arm=='clean' else 'validation_gate.json')
        gates[f'{arm}/{seed}']={'checkpoint_sha256':sha(ck),'preprocessing_sha256':sha(m),'gate_sha256':sha(gatefile)};eligible.append((arm,seed,d))
    contract={'protocol_sha256':json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256'],'inference_source_sha256':sha(Path(__file__)),
      'test_order_sha256':w.hash_obj([r['record_id'] for r in parts['test']]),'test_label_order_sha256':w.hash_obj([r['labels'] for r in parts['test']]),
      'checkpoint_gate_bindings':gates,'conditions_added':['RA_LA_reversal','LA_LL_reversal'],'batch_size':128,'mode':'model.eval + inference_mode + CUDA float16 autocast',
      'test_data_read_before_freeze':False,'existing_conditions_only':True}
    w.atomic_json(w.OUT/'EXISTING_MISSING_INFERENCE_FREEZE.json',contract)
    x,y=w.validation_arrays(parts['test']); audit=[]
    for arm,seed,d in eligible:
        m,std=np.load(d/'preprocessing.npz',allow_pickle=False)['mean'],np.load(d/'preprocessing.npz',allow_pickle=False)['std']
        import torch
        from src.chapman_phase1 import ECGResNetSmall
        model=ECGResNetSmall().cuda();ck=torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda');model.load_state_dict(ck['state_dict'])
        pred_dir=w.OUT/'existing_reinference'/arm/f'seed_{seed}'/'test_predictions';pred_dir.mkdir(parents=True,exist_ok=True)
        for fault in ('RA_LA_reversal','LA_LL_reversal'):
            f=pred_dir/f'{fault}.npz'
            if f.exists():
                with np.load(f,allow_pickle=False) as z:
                    valid=(z['p'].shape==(len(y),3) and np.isfinite(z['p']).all() and str(z['condition'].item())==fault and str(z['checkpoint_sha256'].item())==sha(d/'checkpoint.pt') and str(z['protocol_sha256'].item())==contract['protocol_sha256'] and str(z['test_order_sha256'].item())==contract['test_order_sha256'] and np.array_equal(z['y'],y))
                    if not valid:raise RuntimeError(f'bad partial inference binding {f}')
                    audit.append({'arm':arm,'seed':seed,'condition':fault,'status':'RESUMED','sha256':sha(f)});continue
            xt=np.stack([w.reverse_raw(a,fault) for a in x]);model.eval();out=[]
            with torch.inference_mode():
                for start in range(0,len(xt),128):
                    batch=w.recovery.normalize_raw(xt[start:start+128],m,std)
                    with torch.autocast('cuda',dtype=torch.float16):out.append(torch.sigmoid(model(torch.from_numpy(batch).cuda())).float().cpu().numpy())
            p=np.concatenate(out);w.atomic_npz(f,p=p,y=y,condition=np.asarray(fault),checkpoint_sha256=np.asarray(sha(d/'checkpoint.pt')),protocol_sha256=np.asarray(contract['protocol_sha256']),test_order_sha256=np.asarray(contract['test_order_sha256']))
            audit.append({'arm':arm,'seed':seed,'condition':fault,'status':'PASS','sha256':sha(f)})
    w.atomic_json(w.OUT/'EXISTING_MISSING_INFERENCE_AUDIT.json',{'decision':'PASS','models_conditions':audit,'only_missing_conditions_rerun':True,'freeze_sha256':sha(w.OUT/'EXISTING_MISSING_INFERENCE_FREEZE.json')})
    return audit

def load_p(arm,seed,condition):
    f=pred_path(arm,seed,condition)
    if not f.is_file():raise FileNotFoundError(f)
    with np.load(f,allow_pickle=False) as z:
        p=z['p'].astype(np.float32)
        if p.shape!=(6981,3) or not np.isfinite(p).all():raise RuntimeError(f'invalid predictions {f}: {p.shape}')
    return p

def evaluate_conditions(parts):
    y=np.asarray([r['labels'] for r in parts['test']],np.uint8); rows=[]; mask_dist=[]; exact=[]; points={}
    for arm in ('clean','masking','physics_aware','combined'):
      for seed in w.SEEDS:
        if arm in w.ARMS:
            g=json.loads((w.unit_dir(arm,seed)/'validation_gate.json').read_text(encoding='utf-8'))
            if g['decision']!='PASS':continue
            ed=w.unit_dir(arm,seed);ev=json.loads((ed/'evaluation_complete.json').read_text(encoding='utf-8'))
            proto=json.loads(w.PROTO.read_text(encoding='utf-8'))
            if ev.get('protocol_sha256')!=proto['protocol_sha256'] or ev.get('checkpoint_sha256')!=g['checkpoint_sha256'] or len(ev.get('predictions',{}))!=len(w.CONDITIONS):raise RuntimeError(f'evaluation manifest binding mismatch: {arm}/{seed}')
            for cond,digest in ev['predictions'].items():
                f=ed/'test_predictions'/f'{cond}.npz'
                if not f.is_file() or sha(f)!=digest:raise RuntimeError(f'prediction file hash mismatch: {arm}/{seed}/{cond}')
            val=np.load(w.unit_dir(arm,seed)/'validation.npz',allow_pickle=False)
        else:
            d=w.N/f'seed_{seed}' if arm=='clean' else w.M2/f'seed_{seed}'
            val=np.load(d/'validation.npz',allow_pickle=False)
        th=val['thresholds']
        for ci,condition in enumerate(w.CONDITIONS):
            p=load_p(arm,seed,condition);points[arm,seed,condition]=p
            for j,ep in enumerate(w.ENDPOINTS):rows.append({'arm':arm,'seed':seed,'condition':condition,'endpoint':ep,**w.metric(y[:,j],p[:,j],float(th[j]))})
            if condition==f'mask5_0_1_2_3_5':
                for j,ep in enumerate(w.ENDPOINTS):exact.append({'arm':arm,'seed':seed,'endpoint':ep,'mask':'I,II,III,aVR,aVF','removed_indices':'0,1,2,3,5','auprc':float(average_precision_score(y[:,j],p[:,j]))})
        ap=np.array([average_precision_score(y[:,0],points[arm,seed,n][:,0]) for n in w.MASK_NAMES])
        mask_dist.append({'arm':arm,'seed':seed,'endpoint':'TWA','mask_count':792,'mean_auprc':float(ap.mean()),'median_auprc':float(np.median(ap)),'sd_auprc':float(ap.std(ddof=1)),
          'p05':float(np.quantile(ap,.05)),'p25':float(np.quantile(ap,.25)),'p75':float(np.quantile(ap,.75)),'p95':float(np.quantile(ap,.95)),'min':float(ap.min()),'max':float(ap.max()),'exact_mask_auprc':float(ap[w.MASKS.index(w.EXACT)])})
    w.atomic_csv(w.OUT/'RESULTS_BY_ARM_SEED.csv',rows);w.atomic_csv(w.OUT/'EXACT_MASK_COMPARISON.csv',exact);w.atomic_csv(w.OUT/'MASK_DISTRIBUTION_SUMMARY.csv',mask_dist)
    mean_rows=[]
    metric_fields=('auprc','auroc','brier','ece15','sensitivity','specificity','predicted_positive_fraction')
    for arm in ('clean','masking','physics_aware','combined'):
      conds=sorted({r['condition'] for r in rows if r['arm']==arm})
      for cond in conds:
       for ep in w.ENDPOINTS:
        group=[r for r in rows if r['arm']==arm and r['condition']==cond and r['endpoint']==ep]
        if not group:continue
        row={'arm':arm,'condition':cond,'endpoint':ep,'seed_count':len(group)}
        for field in metric_fields:
            values=np.asarray([float(r[field]) for r in group]);row[field+'_mean']=float(values.mean());row[field+'_sd_across_seeds']=float(values.std(ddof=1)) if len(values)>1 else None
        mean_rows.append(row)
    w.atomic_csv(w.OUT/'RESULTS_MEAN_BY_ARM.csv',mean_rows)
    mask_means=[]
    for arm in ('clean','masking','physics_aware','combined'):
        group=[r for r in mask_dist if r['arm']==arm]
        if group:mask_means.append({'arm':arm,'seed_count':len(group),**{k:float(np.mean([float(r[k]) for r in group])) for k in ('mean_auprc','median_auprc','sd_auprc','p05','p25','p75','p95','min','max','exact_mask_auprc')}})
    w.atomic_csv(w.OUT/'MASK_DISTRIBUTION_MEAN_BY_ARM.csv',mask_means)
    secondary=[]
    for seed in w.SEEDS:
      for cond in ('clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal',f'mask5_0_1_2_3_5','mean_mask'):
        if cond=='mean_mask':
            a=float(np.mean([average_precision_score(y[:,0],points['combined',seed,n][:,0]) for n in w.MASK_NAMES]))
            b=float(np.mean([average_precision_score(y[:,0],points['masking',seed,n][:,0]) for n in w.MASK_NAMES]))
        else:
            a=float(average_precision_score(y[:,0],points['combined',seed,cond][:,0]));b=float(average_precision_score(y[:,0],points['masking',seed,cond][:,0]))
        secondary.append({'contrast':'combined_minus_masking','seed':seed,'endpoint':'TWA','condition_estimand':cond,'combined_auprc':a,'masking_auprc':b,'difference':a-b,'uncertainty':'secondary descriptive point estimate; no interval claim'})
    w.atomic_csv(w.OUT/'SECONDARY_CONTRASTS.csv',secondary)
    return y,points,rows,mask_dist

def frozen_bootstrap_tasks():
    """Verify and reuse old recovery-v2 arrays only under their original content binding."""
    protocol=json.loads((w.REC/'RECOVERY_PROTOCOL.json').read_text(encoding='utf-8'));idx=np.load(w.N/'bootstrap_indices.npy',allow_pickle=False)
    out={}; audits=[]; index_sha=sha(w.N/'bootstrap_indices.npy')
    for arm,mode in (('clean','baseline_v2'),('physics_aware','physics_aware_m2')):
      for seed in w.SEEDS:
       predhash={n:sha(existing_source(arm,seed,n)) for n in ('clean','RA_LL_reversal',*w.MASK_NAMES)}
       for ep in w.ENDPOINTS:
        f=w.REC/'bootstrap_tasks'/f'{mode}_seed_{seed}_{ep}.npz';z=np.load(f,allow_pickle=False)
        bind={'protocol':protocol['binding_sha256'],'mode':mode,'seed':seed,'endpoint':ep,'bootstrap_indices_sha256':index_sha,'prediction_hashes':predhash}
        expected=w.hash_obj(bind)
        if str(z['binding_sha256'].item())!=expected:raise RuntimeError(f'old bootstrap binding mismatch: {f}')
        if z['mask_ap'].shape!=(1000,792) or z['clean_ap'].shape!=(1000,) or z['ra_ll_ap'].shape!=(1000,) or not np.isfinite(z['mask_ap']).all():raise RuntimeError(f'old bootstrap task malformed: {f}')
        out[arm,seed,ep]={k:z[k].copy() for k in ('mask_ap','clean_ap','ra_ll_ap','exact_ap')};audits.append({'arm':arm,'seed':seed,'endpoint':ep,'source':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'binding_sha256':expected,'status':'REUSED_VERIFIED'})
    return idx,out,audits

def bootstrap_new(arm,seed,endpoint_idx,y,points,idx):
    import src.mask_semantics_recovery_v2 as r
    counts=r.bootstrap_counts(idx)
    mask_matrix=np.stack([points[arm,seed,n][:,endpoint_idx] for n in w.MASK_NAMES])
    mask_ap=r.weighted_ap_resamples(y[:,endpoint_idx],mask_matrix,counts).T
    clean=r.weighted_ap_resamples(y[:,endpoint_idx],points[arm,seed,'clean'][:,endpoint_idx][None,:],counts)[0]
    ra=r.weighted_ap_resamples(y[:,endpoint_idx],points[arm,seed,'RA_LL_reversal'][:,endpoint_idx][None,:],counts)[0]
    exact=mask_ap[:,w.MASKS.index(w.EXACT)]
    return {'mask_ap':mask_ap,'clean_ap':clean,'ra_ll_ap':ra,'exact_ap':exact}

def paired_bootstrap(y,points,mask_dist):
    idx,task,audits=frozen_bootstrap_tasks();
    for arm in w.ARMS:
      for seed in w.SEEDS:
        g=json.loads((w.unit_dir(arm,seed)/'validation_gate.json').read_text(encoding='utf-8'))
        if g['decision']!='PASS':continue
        z=bootstrap_new(arm,seed,0,y,points,idx);task[arm,seed,'TWA']=z
        w.atomic_npz(w.OUT/'bootstrap_tasks'/f'{arm}_seed_{seed}_TWA.npz',mask_ap=z['mask_ap'],clean_ap=z['clean_ap'],ra_ll_ap=z['ra_ll_ap'],exact_ap=z['exact_ap'],bootstrap_indices_sha256=np.asarray(sha(w.N/'bootstrap_indices.npy')),prediction_binding_sha256=np.asarray(w.hash_obj({c:sha(pred_path(arm,seed,c)) for c in ('clean','RA_LL_reversal',*w.MASK_NAMES)})))
        audits.append({'arm':arm,'seed':seed,'endpoint':'TWA','status':'COMPUTED','replicates':1000})
    contrasts=[];boot=[];replicates=[]
    for new,old in (('masking','clean'),('combined','physics_aware')):
      for cond in ('clean','RA_LL_reversal','mean_mask'):
       seed_arrays=[];full=[]
       for seed in w.SEEDS:
        if (new,seed,'TWA') not in task:continue
        a,b=task[new,seed,'TWA'],task[old,seed,'TWA']
        if cond=='clean':x,z=a['clean_ap'],b['clean_ap'];ptx=average_precision_score(y[:,0],points[new,seed,'clean'][:,0]);ptz=average_precision_score(y[:,0],points[old,seed,'clean'][:,0])
        elif cond=='RA_LL_reversal':x,z=a['ra_ll_ap'],b['ra_ll_ap'];ptx=average_precision_score(y[:,0],points[new,seed,'RA_LL_reversal'][:,0]);ptz=average_precision_score(y[:,0],points[old,seed,'RA_LL_reversal'][:,0])
        else:x,z=a['mask_ap'].mean(1),b['mask_ap'].mean(1);ptx=np.mean([average_precision_score(y[:,0],points[new,seed,n][:,0]) for n in w.MASK_NAMES]);ptz=np.mean([average_precision_score(y[:,0],points[old,seed,n][:,0]) for n in w.MASK_NAMES])
        seed_arrays.append(x-z);full.append(ptx-ptz)
       if len(seed_arrays)!=3:continue
       draws=np.mean(np.stack(seed_arrays),axis=0);point=float(np.mean(full))
       contrasts.append({'contrast':f'{new}_minus_{old}','endpoint':'TWA','estimand':cond,'point_estimate_full_cohort':point,'seed_differences':json.dumps(full)})
       boot.append({'contrast':f'{new}_minus_{old}','endpoint':'TWA','estimand':cond,'point_estimate_full_cohort':point,'bootstrap_mean':float(draws.mean()),'lower95':float(np.quantile(draws,.025)),'upper95':float(np.quantile(draws,.975)),'replicates':1000,'resampling':'synchronized frozen Ningbo canonical-record draws'})
       for i,v in enumerate(draws):replicates.append({'contrast':f'{new}_minus_{old}','estimand':cond,'replicate':i,'difference':float(v)})
    w.atomic_csv(w.OUT/'PRIMARY_CONTRASTS.csv',contrasts);w.atomic_csv(w.OUT/'BOOTSTRAP_SUMMARY.csv',boot);w.atomic_csv(w.OUT/'BOOTSTRAP_REPLICATES.csv',replicates)
    w.atomic_json(w.OUT/'BOOTSTRAP_REUSE_AND_COMPUTATION_AUDIT.json',{'decision':'PASS','indices_sha256':sha(w.N/'bootstrap_indices.npy'),'index_shape':list(idx.shape),'tasks':audits,'all_new_primary_tasks_use_same_draws':True,'point_estimates_from_full_cohort':True,'mask_panel_ap_per_mask_then_mean':True})
    return boot

def calibration_mixtures(y,parts,points,condition_rows):
    v1=import_v1();cv2=json.loads((ROOT/'outputs/mixed_fault_calibration_v2/CALIBRATION_AND_PROVENANCE.json').read_text(encoding='utf-8'))
    split=np.load(ROOT/'outputs/mixed_fault_calibration_v1/calibration_split_indices.npz',allow_pickle=False);A=split['A'].astype(int);B=split['B'].astype(int)
    fits={};rows=[];calprov={};tie_audit=[]
    for arm in ('clean','physics_aware','masking','combined'):
      for seed in w.SEEDS:
        if arm in ('clean','physics_aware'):
            f= cv2['calibration_fits']['ordinary_resnet' if arm=='clean' else 'physics_aware_resnet'][str(seed)]
            fit=[(float(x['a']),float(x['b']),float(x['threshold_raw']),float(x['threshold_calibrated']),x['optimizer']) for x in f]
        else:
            val=np.load(w.unit_dir(arm,seed)/'validation.npz',allow_pickle=False); yy=val['y'].astype(np.uint8); pp=val['p'].astype(float);fit=[]
            for j in range(3):
                a,b,opt=v1.fit_platt(pp[A,j],yy[A,j]);thr=float(np.quantile(pp[B,j][yy[B,j]==0],.90,method='linear'));tc=float(v1.platt(np.asarray([thr]),a,b)[0]);fit.append((a,b,thr,tc,opt))
        fits[arm,seed]=fit;calprov[f'{arm}/{seed}']=[{'endpoint':w.ENDPOINTS[j],'a':fit[j][0],'b':fit[j][1],'threshold_raw':fit[j][2],'threshold_calibrated':fit[j][3],'optimizer':fit[j][4]} for j in range(3)]
        for j,ep in enumerate(w.ENDPOINTS):
          clean=points[arm,seed,'clean'][:,j].astype(float);rev=points[arm,seed,'RA_LL_reversal'][:,j].astype(float);calclean=v1.platt(clean,fit[j][0],fit[j][1]);calrev=v1.platt(rev,fit[j][0],fit[j][1]);
          tie_audit.append({'arm':arm,'seed':seed,'endpoint':ep,'condition':'clean','raw_unique_scores':int(np.unique(clean).size),'calibrated_unique_scores':int(np.unique(calclean).size),'added_clipped_ties':int(np.unique(clean).size-np.unique(calclean).size),'operating_decision_mismatches':int(np.count_nonzero((clean>=fit[j][2])!=(calclean>=fit[j][3]))),'auprc_raw':float(average_precision_score(y[:,j],clean)),'auprc_calibrated':float(average_precision_score(y[:,j],calclean)),'auroc_raw':float(roc_auc_score(y[:,j],clean)),'auroc_calibrated':float(roc_auc_score(y[:,j],calclean))})
          tie_audit.append({'arm':arm,'seed':seed,'endpoint':ep,'condition':'RA_LL_reversal','raw_unique_scores':int(np.unique(rev).size),'calibrated_unique_scores':int(np.unique(calrev).size),'added_clipped_ties':int(np.unique(rev).size-np.unique(calrev).size),'operating_decision_mismatches':int(np.count_nonzero((rev>=fit[j][2])!=(calrev>=fit[j][3]))),'auprc_raw':float(average_precision_score(y[:,j],rev)),'auprc_calibrated':float(average_precision_score(y[:,j],calrev)),'auroc_raw':float(roc_auc_score(y[:,j],rev)),'auroc_calibrated':float(roc_auc_score(y[:,j],calrev))})
          yy=np.r_[y[:,j],y[:,j]]
          for q in (0,.001,.005,.01,.05,.1,1.):
            weight=np.r_[np.full(len(y),1-q),np.full(len(y),q)]
            for mode,pc,pr,thr in [('raw',clean,rev,fit[j][2]),('calibrated',calclean,calrev,fit[j][3])]:
                pp=np.r_[pc,pr];met=v1.metrics(yy,pp,float(thr),weight)
                rows.append({'arm':arm,'seed':seed,'endpoint':ep,'q':q,'score_mode':mode,'auprc':met['auprc'],'auroc':met['auroc'],'brier':met['brier'],'ece15':met['ece15'],'sensitivity':met['sensitivity'],'specificity':met['specificity'],'predicted_positive_fraction':met['predicted_positive_fraction'],'threshold':thr,'calibration_a':fit[j][0] if mode=='calibrated' else None,'calibration_b':fit[j][1] if mode=='calibrated' else None})
    w.atomic_csv(w.OUT/'MIXTURE_RESULTS_BY_SEED.csv',rows)
    mean_rows=[]
    for arm in ('clean','masking','physics_aware','combined'):
      for ep in w.ENDPOINTS:
       for q in (0,.001,.005,.01,.05,.1,1.):
        for mode in ('raw','calibrated'):
         group=[r for r in rows if r['arm']==arm and r['endpoint']==ep and float(r['q'])==q and r['score_mode']==mode]
         if not group:continue
         z={'arm':arm,'endpoint':ep,'q':q,'score_mode':mode,'seed_count':len(group)}
         for field in ('auprc','auroc','brier','ece15','sensitivity','specificity','predicted_positive_fraction'):
            arr=np.asarray([float(r[field]) for r in group]);z[field+'_mean']=float(arr.mean());z[field+'_sd_across_seeds']=float(arr.std(ddof=1)) if len(arr)>1 else None
         mean_rows.append(z)
    w.atomic_csv(w.OUT/'MIXTURE_MEAN_SUMMARY.csv',mean_rows)
    w.atomic_json(w.OUT/'CALIBRATION_AND_PROVENANCE.json',{
      'calibrator_sources':{'clean':'reused mixed_fault_calibration_v2 ordinary_resnet fits',
        'physics_aware':'reused mixed_fault_calibration_v2 physics_aware_resnet fits',
        'masking_combined':'fit on frozen clean validation A with exact v1 Platt implementation'},
      'validation_split_indices_sha256':sha(ROOT/'outputs/mixed_fault_calibration_v1/calibration_split_indices.npz'),
      'test_fitting':False,'raw_threshold':'90th percentile of negative clean validation B',
      'calibrated_threshold':'same raw threshold transformed through Platt fit','fits':calprov,
      'metrics':'raw and calibrated, clean/RA-LL weighted mixtures; same threshold calls preserved except clipping/ties',
      'ranking_and_operating_point_audit':tie_audit})
    # Synchronized record bootstrap for the primary TWA mixture contrasts.
    idx=np.load(w.N/'bootstrap_indices.npy',allow_pickle=False); yy=np.r_[y[:,0],y[:,0]]; draws={};rep_rows=[]
    rmod=__import__('src.mask_semantics_recovery_v2',fromlist=['bootstrap_counts'])
    counts=rmod.bootstrap_counts(idx)
    for arm in ('clean','masking','physics_aware','combined'):
      for seed in w.SEEDS:
        fit=fits[arm,seed][0];c=points[arm,seed,'clean'][:,0].astype(float);r=points[arm,seed,'RA_LL_reversal'][:,0].astype(float)
        cp=v1.platt(c,fit[0],fit[1]);rp=v1.platt(r,fit[0],fit[1])
        for q in (0,.001,.005,.01,.05,.1,1.):
          for mode,pc,pr in (('raw',c,r),('calibrated',cp,rp)):
            pp=np.r_[pc,pr]; vals=np.empty(1000,np.float32)
            for bi,mult in enumerate(counts):
                mult=mult.astype(np.float64)
                ww=np.r_[mult*(1-q),mult*q]
                vals[bi]=average_precision_score(yy,pp,sample_weight=ww)
            draws[arm,seed,q,mode]=vals
    full={(r['arm'],int(r['seed']),float(r['q']),r['score_mode']):r['auprc'] for r in rows if r['endpoint']=='TWA'}
    summaries=[]
    for q in (0,.001,.005,.01,.05,.1,1.):
      for mode in ('raw','calibrated'):
       for new,old in (('masking','clean'),('combined','physics_aware')):
        paired=[]
        for seed in w.SEEDS:paired.append(draws[new,seed,q,mode]-draws[old,seed,q,mode])
        d=np.mean(np.stack(paired),axis=0);pt=float(np.mean([full[new,s,q,mode]-full[old,s,q,mode] for s in w.SEEDS]))
        summaries.append({'contrast':f'{new}_minus_{old}','endpoint':'TWA','q':q,'score_mode':mode,'point_estimate_full_cohort':pt,'bootstrap_mean':float(d.mean()),'lower95':float(np.quantile(d,.025)),'upper95':float(np.quantile(d,.975)),'replicates':1000,'bootstrap_indices_sha256':sha(w.N/'bootstrap_indices.npy')})
        for i,val in enumerate(d):rep_rows.append({'contrast':f'{new}_minus_{old}','q':q,'score_mode':mode,'replicate':i,'difference':float(val)})
    w.atomic_csv(w.OUT/'MIXTURE_BOOTSTRAP_SUMMARY.csv',summaries);w.atomic_csv(w.OUT/'MIXTURE_BOOTSTRAP_REPLICATES.csv',rep_rows)
    # Compact comparison figures; all plotted values are seed-level metrics, not prediction ensembles.
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.8))
        for arm in ('clean','masking','physics_aware','combined'):
            qv=[];mv=[]
            for q in (0,.001,.005,.01,.05,.1,1.):
                qv.append(q);mv.append(np.mean([full[arm,s,q,'raw'] for s in w.SEEDS]))
            ax.plot(qv,mv,marker='o',label=arm)
        ax.set_xscale('symlog',linthresh=.001);ax.set_xlabel('RA–LL mixture share q');ax.set_ylabel('TWA AUPRC (raw)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(w.OUT/'TWA_AUPRC_BY_Q.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.8));conditions=('clean','RA_LA_reversal','RA_LL_reversal','LA_LL_reversal')
        for arm in ('clean','masking','physics_aware','combined'):
            ys=[]
            for c in conditions:
                ys.append(np.mean([float(next(r['auprc'] for r in condition_rows if r['arm']==arm and int(r['seed'])==s and r['endpoint']=='TWA' and r['condition']==c)) for s in w.SEEDS]))
            ax.plot(conditions,ys,marker='o',label=arm)
        ax.set_ylabel('TWA AUPRC');ax.tick_params(axis='x',rotation=18);ax.legend(frameon=False);fig.tight_layout();fig.savefig(w.OUT/'TWA_AUPRC_BY_CONDITION.png',dpi=160);plt.close(fig)
    except Exception as exc: w.atomic_json(w.OUT/'PLOT_GENERATION_WARNING.json',{'reason':str(exc)})
    return rows,calprov

def main():
    w.protocol_check(); parts=rows_records()
    infer_missing_existing(parts)
    y,points,rows,dist=evaluate_conditions(parts)
    boot=paired_bootstrap(y,points,dist)
    mixture,cal=calibration_mixtures(y,parts,points,rows)
    w.atomic_json(w.OUT/'ANALYSIS_COMPLETE.json',{'status':'COMPLETE','created_utc':w.now(),'condition_rows':len(rows),'primary_contrasts':len(boot),'mixture_rows':len(mixture),'protocol_sha256':json.loads(w.PROTO.read_text(encoding='utf-8'))['protocol_sha256']})
    print('FOUR_ARM_ANALYSIS_COMPLETE',len(rows),len(boot),len(mixture),flush=True)

if __name__=='__main__':main()
