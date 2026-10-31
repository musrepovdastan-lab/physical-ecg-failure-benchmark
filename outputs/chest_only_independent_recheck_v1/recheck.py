"""Independent AP recheck of existing frozen models; no inference or fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/chest_only_baseline_v1'
SEEDS = [104729, 130363, 155921]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cache(y, p):
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    return order, ends, y[order].astype(float)


def ap_from_multiplicities(c, multiplicity):
    order, ends, y = c
    w = multiplicity[order]
    tp = np.cumsum(w * y)[ends]
    total = np.cumsum(w)[ends]
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    return float(np.sum(np.diff(np.r_[0., tp]) * precision) / tp[-1])


def main():
    specification = {
        'date': '2026-10-18', 'scope': 'Recompute AP and synchronized 1000-draw intervals from existing frozen predictions only; previously accessed cohorts.',
        'seeds': SEEDS, 'datasets': ['Chapman', 'Ningbo', 'HEEDB'],
        'conditions': ['clean', 'RA_LL_reversal'], 'primary_endpoint': 'TWA',
        'contrast': 'physics-aware 12-lead minus chest-only',
        'statistic': 'mean of seed-specific AP differences; percentile interval; no ensemble',
        'new_training': False, 'new_inference': False,
        'interpretation': 'Conditional record/patient uncertainty; not equivalence or all training uncertainty.',
    }
    protocol = OUT / 'RECHECK_SPECIFICATION.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == specification
    else:
        protocol.write_text(json.dumps(specification, indent=2) + '\n')
    local_reference = pd.read_csv(BASE / 'BOOTSTRAP_SUMMARY.csv')
    external_reference = pd.read_csv(ROOT / 'outputs/mixed_fault_calibration_v2/PAIRED_CONTRASTS.csv')
    points, summaries, hashes, draws_out = [], [], {}, {}
    independent_spots = 0
    for dataset in specification['datasets']:
        if dataset == 'HEEDB':
            priv = ROOT / 'data/heedb_restricted/I0006'
            fp = priv / 'independent_confirmation_v1/predictions_float32.npy'
            fc = priv / 'mixed_fault_calibration_v1/chest_only_predictions_float32.npy'
            yfile = ROOT / 'outputs/heedb_four_arm_extension_v1/predictions/combined/seed_104729/clean.npz'
            y = np.load(yfile, allow_pickle=False)['labels'][:, 0]
            phys = np.load(fp, mmap_mode='r')[1]
            chest = np.load(fc, mmap_mode='r')
            matrices = [(phys[s, :, :, 0].astype(float), chest[s, :, :, 0].astype(float)) for s in range(3)]
            indices_file = ROOT / 'outputs/heedb_four_arm_extension_v1/HEEDB_BOOTSTRAP_INDICES.npy'
            for f in [fp, fc, yfile]: hashes[f.relative_to(ROOT).as_posix()] = sha(f)
        else:
            matrices = []
            for seed in SEEDS:
                pairs = []
                for model in ['physics_aware_12lead', 'chest_only']:
                    d = BASE / ('models' if model == 'chest_only' else 'comparators') / dataset
                    d = d / f'seed_{seed}' / 'test_predictions' if model == 'chest_only' else d / model / f'seed_{seed}'
                    conditions = []
                    for condition in specification['conditions']:
                        f = d / f'{condition}.npz'; z = np.load(f, allow_pickle=False)
                        if pairs or matrices or conditions: assert np.array_equal(y, z['y'][:, 0])
                        y = z['y'][:, 0]
                        conditions.append(z['p'][:, 0].astype(float)); hashes[f.relative_to(ROOT).as_posix()] = sha(f)
                    pairs.append(np.stack(conditions))
                matrices.append(tuple(pairs))
            indices_file = ROOT / ('outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy' if dataset == 'Chapman' else 'outputs/ningbo_confirmation_v2/bootstrap_indices.npy')
        ix = np.load(indices_file, mmap_mode='r'); assert ix.shape == (1000, len(y))
        hashes[indices_file.relative_to(ROOT).as_posix()] = sha(indices_file)
        for ci, condition in enumerate(specification['conditions']):
            caches = []
            for si, (phys, chest) in enumerate(matrices):
                p, c = phys[ci], chest[ci]
                assert np.isfinite(p).all() and np.isfinite(c).all()
                cp, cc = cache(y, p), cache(y, c); caches.append((cp, cc))
                ap, ac = average_precision_score(y, p), average_precision_score(y, c)
                assert np.isclose(ap_from_multiplicities(cp, np.ones(len(y))), ap, atol=1e-12)
                assert np.isclose(ap_from_multiplicities(cc, np.ones(len(y))), ac, atol=1e-12)
                points.append({'dataset': dataset, 'condition': condition, 'seed': SEEDS[si], 'physics_auprc': ap, 'chest_auprc': ac,
                               'auprc_difference': ap-ac, 'physics_auroc': roc_auc_score(y,p), 'chest_auroc': roc_auc_score(y,c),
                               'physics_brier': float(np.mean((p-y)**2)), 'chest_brier': float(np.mean((c-y)**2))})
            draws = np.empty(1000)
            for bi, indices in enumerate(ix):
                multiplicity = np.bincount(indices, minlength=len(y))
                draws[bi] = np.mean([ap_from_multiplicities(cp, multiplicity)-ap_from_multiplicities(cc, multiplicity) for cp, cc in caches])
                if bi in [0, 417, 999]:
                    direct = np.mean([average_precision_score(y, p[ci], sample_weight=multiplicity)-average_precision_score(y, c[ci], sample_weight=multiplicity) for p,c in matrices])
                    assert np.isclose(direct, draws[bi], atol=1e-12); independent_spots += 1
            row = {'dataset': dataset, 'condition': condition, 'physics_auprc': float(np.mean([p['physics_auprc'] for p in points[-3:]])),
                   'chest_auprc': float(np.mean([p['chest_auprc'] for p in points[-3:]])),
                   'difference': float(np.mean([p['auprc_difference'] for p in points[-3:]])),
                   'bootstrap_mean': float(draws.mean()), 'lower95': float(np.quantile(draws,.025)), 'upper95': float(np.quantile(draws,.975))}
            if dataset == 'HEEDB':
                ref = external_reference[(external_reference.source=='HEEDB_I0006_independent_confirmation') & (external_reference.endpoint=='TWA') &
                     (external_reference.q_fault==(0 if ci==0 else 1)) & (external_reference.contrast=='physics_aware_minus_chest_only_resnet') &
                     (external_reference.score_mode=='raw') & (external_reference.metric=='auprc')].iloc[0]
                expected = [ref.full_sample_difference, ref.bootstrap_mean_difference, ref.ci_2_5, ref.ci_97_5]
            else:
                name = 'physics_aware_12lead_minus_chest_only_' + ('clean' if ci==0 else 'RA_LL')
                ref = local_reference[(local_reference.dataset==dataset)&(local_reference.endpoint=='TWA')&(local_reference.comparison==name)&(local_reference.metric=='AUPRC')].iloc[0]
                expected = [ref.point_mean_of_seed_differences, ref.bootstrap_mean, ref.lower95, ref.upper95]
            assert np.allclose([row[k] for k in ['difference','bootstrap_mean','lower95','upper95']], expected, atol=1e-7), row
            summaries.append(row);draws_out[dataset+'|'+condition] = draws
            print(json.dumps(row), flush=True)
    pd.DataFrame(points).to_csv(OUT/'SEED_LEVEL_RECHECK.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'PAIRED_RECHECK.csv',index=False)
    np.savez_compressed(OUT/'BOOTSTRAP_RECHECK_DRAWS.npz',**draws_out)
    (OUT/'AUDIT.json').write_text(json.dumps({'status':'PASS','summary_groups':6,'bootstrap_draws_per_group':1000,'sklearn_bootstrap_spots':independent_spots,'input_hashes':hashes},indent=2)+'\n')


if __name__ == '__main__': main()
