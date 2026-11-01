"""Reproduce the nine fits in a fresh corrected-series output directory.

This intentionally refuses existing checkpoints so it cannot accidentally
repeat a completed training series. Run only on a newly prepared copy.
"""
import json
import argparse
import control_replay as c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-inputs',action='store_true',help='validate frozen scientific inputs and documentation provenance without training')
    args=parser.parse_args()
    prot=c.protocol()
    if args.check_inputs:
        report=c.input_provenance_report()
        (c.OUT/'REPRODUCTION_INPUT_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))
        return
    if any((c.OUT/'fits').rglob('checkpoint.pt')):
        raise RuntimeError('refusing to train over existing checkpoints; use the saved series or a fresh package copy')
    if json.loads((c.OUT/'INITIALIZATION_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':
        raise RuntimeError('initialization audit must PASS before training')
    if json.loads((c.OUT/'ENGINEERING_SMOKE_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':
        raise RuntimeError('engineering smoke must PASS before training')
    if json.loads((c.OUT/'HISTORICAL_INTEGRITY_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':
        raise RuntimeError('historical artifact audit must PASS before training')
    if json.loads((c.OUT/'DETECTOR_CACHE_REUSE_AUDIT.json').read_text(encoding='utf8')).get('status')!='PASS':
        raise RuntimeError('verified frozen detector caches are required')
    parts,_=c.records();fits=[]
    for method in c.METHODS:
        for seed in c.SEEDS:
            result=c.train_one(method,seed,parts,prot);fits.append(result)
            c.atomic_json(c.OUT/'TRAINING_STATUS.json',{'fits':fits,'completed':len(fits),'total':9})
            if result.get('decision')!='PASS':
                c.atomic_json(c.OUT/'STATUS.json',{'status':'STOP_GATE_FAILURE','failed_fit':result})
                return
    lock=c.lock_all(parts,prot)
    if lock is None:
        c.atomic_json(c.OUT/'STATUS.json',{'status':'STOP_GATE_FAILURE','reason':'best-checkpoint lock failed'})
        return
    c.atomic_json(c.OUT/'STATUS.json',{'status':'ALL_NINE_FITS_LOCKED_READY_POSTPROCESS','fit_count':len(lock),'utc':c.now()})
    print(json.dumps({'status':'ALL_NINE_FITS_LOCKED_READY_POSTPROCESS','fits':len(lock)}))


if __name__=='__main__':main()
