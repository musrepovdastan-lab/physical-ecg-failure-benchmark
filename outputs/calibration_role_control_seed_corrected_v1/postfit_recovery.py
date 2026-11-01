"""Recover post-fit scoring after the frozen runner's NPZ keyword collision.

This continuation never calls train_one. It consumes and verifies the nine
already completed checkpoints, then performs the protocol's locked A/B work.
"""
import json
import numpy as np
import torch
from pathlib import Path
import control as c


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def correct_ningbo_length(incoming, predicted_classes):
    """Apply the same named lead reversal to Ningbo's 12x1000 inputs."""
    x = np.asarray(incoming, dtype=np.float32)
    codes = np.asarray(predicted_classes)
    if x.ndim != 3 or x.shape[1:] != (12, 1000) or codes.shape != (len(x),):
        raise ValueError('expected aligned Ningbo (N,12,1000) batch and detector classes')
    if not np.isfinite(x).all() or not np.isin(codes, (0, 1, 2, 3)).all():
        raise ValueError('invalid Ningbo Huang input or predicted classes')
    result = x.copy()
    class_fault = {1: 'RA_LA_reversal', 2: 'RA_LL_reversal', 3: 'LA_LL_reversal'}
    for code, fault in class_fault.items():
        for row in np.flatnonzero(codes == code):
            result[row] = c.reverse_raw(x[row], fault)
    return result


def main():
    prot = c.protocol()
    parts, _ = c.records()
    status = json.loads((c.OUT / 'TRAINING_STATUS.json').read_text(encoding='utf8'))
    if status.get('completed') != 9 or status.get('total') != 9:
        raise RuntimeError('expected all nine completed fits before post-fit recovery')
    if len(status.get('fits', [])) != 9 or any(x.get('decision') != 'PASS' for x in status['fits']):
        raise RuntimeError('one or more fit gates are not PASS')
    locked = c.lock_all(parts, prot)
    if locked is None or len(locked) != 9:
        raise RuntimeError('best-checkpoint lock did not pass for all nine fits')

    roles = ('calibration_A', 'threshold_B')
    rolepred = {}
    for method in c.METHODS:
        for seed in c.SEEDS:
            d = c.OUT / 'fits' / method / f'seed_{seed}'
            ck = torch.load(d / 'checkpoint.pt', weights_only=True, map_location='cpu')
            model = c.ECGResNetSmall().cuda().eval()
            model.load_state_dict(ck['state_dict'])
            with np.load(d / 'preprocessing.npz', allow_pickle=False) as z:
                mean, std = z['mean'], z['std']
            for role in roles:
                xs, ys = c.raw_array(parts[role]), c.labels(parts[role])
                pp = c.infer(model, xs, mean, std)
                rolepred[method, seed, role] = (pp, ys)
                save_npz(c.OUT / f'scores/{role}/{method}/seed_{seed}.npz',
                         p=pp, y=ys,
                         record_ids=np.asarray([r['record_id'] for r in parts[role]]),
                         role=np.asarray(role),
                         checkpoint_sha256=np.asarray(c.sha(d / 'checkpoint.pt')),
                         membership_sha256=np.asarray(prot['membership_file_sha256']))
            del model
            torch.cuda.empty_cache()

    huang_scores = {}
    if any(not (c.OUT / f'detector/{role}.npz').is_file() for role in roles):
        raise RuntimeError('verified reused detector scores missing')
    for seed in c.SEEDS:
        d = c.OUT / 'fits' / 'ordinary' / f'seed_{seed}'
        ck = torch.load(d / 'checkpoint.pt', weights_only=True, map_location='cpu')
        model = c.ECGResNetSmall().cuda().eval()
        model.load_state_dict(ck['state_dict'])
        with np.load(d / 'preprocessing.npz', allow_pickle=False) as z:
            mean, std = z['mean'], z['std']
        for role in roles:
            raw, ys = c.raw_array(parts[role]), c.labels(parts[role])
            preds = []
            detfile = c.OUT / f'detector/{role}.npz'
            with np.load(detfile, allow_pickle=False) as dz:
                detector_scores = dz['probabilities']
            for ci, condition in enumerate(('clean', *c.FAULTS)):
                incoming = raw if condition == 'clean' else np.stack([c.reverse_raw(x, condition) for x in raw])
                det = detector_scores[ci]
                classes = np.argmax(det, axis=1).astype(np.int64)
                fixed = incoming if condition == 'clean' else correct_ningbo_length(incoming, classes)
                preds.append(c.infer(model, fixed, mean, std))
            hp = np.stack(preds)
            huang_scores[seed, role] = (hp, ys)
            save_npz(c.OUT / f'scores/{role}/Huang_full_pipeline/seed_{seed}.npz',
                     p=hp, y=ys,
                     record_ids=np.asarray([r['record_id'] for r in parts[role]]),
                     role=np.asarray(role),
                     checkpoint_sha256=np.asarray(c.sha(d / 'checkpoint.pt')),
                     detector_binding_sha256=np.asarray(c.sha(detfile)))
        del model
        torch.cuda.empty_cache()

    lockrows = []
    for method in (*c.METHODS, 'Huang_full_pipeline'):
        for seed in c.SEEDS:
            for ei, endpoint in enumerate(c.ENDS):
                if method == 'Huang_full_pipeline':
                    pa, ya = huang_scores[seed, 'calibration_A']
                    pb, yb = huang_scores[seed, 'threshold_B']
                    a, b = pa[0, :, ei], pb[0, :, ei]
                else:
                    pa, ya = rolepred[method, seed, 'calibration_A']
                    pb, yb = rolepred[method, seed, 'threshold_B']
                    a, b = pa[:, ei], pb[:, ei]
                fit = c.platt_fit(ya[:, ei], a)
                neg = b[yb[:, ei] == 0]
                if not len(neg) or len(np.unique(ya[:, ei])) != 2 or not fit['success'] or not np.isfinite([fit['a'], fit['b']]).all():
                    lockrows.append({'method': method, 'seed': seed, 'endpoint': endpoint,
                                     'condition': 'clean', 'status': 'FAIL',
                                     'n_negative_B': len(neg), **fit})
                    continue
                q = float(np.quantile(neg, .9, method='linear'))
                mapped = float(c.calibrate_apply(np.array([q]), fit)[0])
                lockrows.append({'method': method, 'seed': seed, 'endpoint': endpoint,
                                 'condition': 'clean', 'status': 'PASS', 'n_A': len(a),
                                 'n_B_negative': len(neg), 'raw_q90': q,
                                 'mapped_calibrated_threshold': mapped,
                                 'slope_a': fit['a'], 'intercept_b': fit['b'],
                                 'fit_success': fit['success'], 'fit_loss': fit['loss'],
                                 'fit_iterations': fit['n_iter']})
    c.atomic_csv(c.OUT / 'CALIBRATION_AND_THRESHOLD_LOCK.csv', lockrows)
    if len([r for r in lockrows if r['status'] == 'PASS']) != 36:
        c.atomic_json(c.OUT / 'STATUS.json', {'status': 'STOP_CALIBRATION_FAILURE', 'rows': len(lockrows)})
        raise RuntimeError('calibration/threshold lock failed')
    c.atomic_json(c.OUT / 'CALIBRATION_AND_THRESHOLD_LOCK.json', {
        'status': 'LOCKED', 'entries': lockrows, 'fitted_on': 'calibration_A',
        'threshold_negatives': 'threshold_B', 'locked_before_test': True,
        'quantile': 'numpy linear', 'raw_decision': '>= q90',
        'lock_hash': c.hobj(lockrows),
        'postfit_recovery_runner_sha256': c.sha(Path(__file__))})
    c.atomic_json(c.OUT / 'POSTFIT_RECOVERY_AUDIT.json', {
        'status': 'PASS', 'completed_fits_reused': 9, 'training_repeated': False,
        'A_B_prediction_files': 36, 'calibration_threshold_rows': len(lockrows),
        'calibration_pass_rows': sum(x['status'] == 'PASS' for x in lockrows),
        'frozen_protocol_sha256': c.sha(c.OUT / 'FROZEN_EXECUTION_PROTOCOL.json'),
        'postfit_recovery_runner_sha256': c.sha(Path(__file__))})
    c.atomic_json(c.OUT / 'STATUS.json', {
        'status': 'LOCKS_COMPLETE_READY_TEST',
        'note': 'Nine fits verified; post-fit A/B lock completed by separately hashed recovery runner.',
        'utc': c.now()})


if __name__ == '__main__':
    main()
