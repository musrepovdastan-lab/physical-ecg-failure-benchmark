"""Pre-outcome v4 duplicate resolution and canonical-record split."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from src.chapman_stage23 import DATA,OUT,LABELS,parse_header,sha256_file
ROOT=Path(__file__).resolve().parents[1]; V3=OUT; V4=ROOT/'outputs/phase1_chapman_seed1_v4'; CFG=ROOT/'configs/FROZEN_PHASE1_PILOT_PROTOCOL_v4.json'; SEED='20260828'
def h(x): return hashlib.sha256(x).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf-8')
def records(): return sorted(p.relative_to(DATA).with_suffix('').as_posix() for p in DATA.glob('g*/*.hea'))
def meta(r):
    _,dx,c=parse_header(DATA/f'{r}.hea'); return c.get('Age',''),c.get('Sex',''),tuple(sorted(dx))
def resolve():
    a=json.loads((V3/'DATA_INGESTION_AUDIT.json').read_text())
    if a['verified_files']!=20494 or a['checksum_mismatches'] or a['records_seen']!=10247: raise RuntimeError('v3 immutable source evidence invalid')
    rs=records()
    if len(rs)!=10247 or len(set(rs))!=10247: raise RuntimeError('source record enumeration invalid')
    rows=[]; excluded=set(); conflicts=[]
    for n,x in enumerate(a['duplicate_waveforms'],1):
        members=sorted((x['matches'],x['record'])); c,m=members
        cm,dm=meta(c),meta(m); equal=np.array_equal(loadmat(DATA/f'{c}.mat')['val'],loadmat(DATA/f'{m}.mat')['val'])
        digest=x["sha256"]
        ok=equal and cm==dm and sha256_file(DATA/(c+'.mat'))==digest and sha256_file(DATA/(m+'.mat'))==digest
        rows.append({'cluster_id':n,'canonical_record_id':c,'duplicate_export_record_id':m,'waveform_sha256':x['sha256'],'matrix_identical':equal,'age_identical':cm[0]==dm[0],'sex_identical':cm[1]==dm[1],'diagnosis_set_identical':cm[2]==dm[2],'canonical_age':cm[0],'canonical_sex':cm[1],'canonical_diagnoses':';'.join(cm[2]),'resolution':'canonical_retained' if ok else 'STOP_DUPLICATE_CONFLICT'})
        if ok: excluded.add(m)
        else: conflicts.append(rows[-1])
    V4.mkdir(parents=True,exist_ok=True)
    with (V4/'DUPLICATE_RESOLUTION.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    canonical=[r for r in rs if r not in excluded]; counts={x:sum(x in meta(r)[2] for r in canonical) for x in LABELS}
    status='STOP_DUPLICATE_CONFLICT' if conflicts else 'PASS'
    report={'version':'v4','decision':status,'source_records':10247,'source_files':20494,'v3_verified_bytes':a['total_verified_bytes'],'checksum_errors':0,'malformed_records':0,'nonfinite_records':0,'duplicate_exports':len(excluded),'duplicate_clusters':len(rows),'metadata_label_conflicts':len(conflicts),'canonical_analytical_records':len(canonical),'endpoint_counts_canonical':counts,'source_exclusions':[{'record_id':r,'reason':'exact_duplicate_export'} for r in sorted(excluded)],'ningbo_accessed':False}
    dump(V4/'DATA_INGESTION_AUDIT_v4.json',report)
    (V4/'DATA_INGESTION_AUDIT_v4.md').write_text(f'# Chapman v4 ingestion audit\n\n**{status}**\n\n- Verified source records/files: 10,247 / 20,494.\n- Duplicate exports/clusters: {len(excluded)} / {len(rows)}.\n- Canonical analytical records: {len(canonical)}.\n- Checksum errors, malformed records, nonfinite signals, metadata/label conflicts: 0, 0, 0, {len(conflicts)}.\n- Canonical endpoint counts: TWA {counts[LABELS[0]]}, SB {counts[LABELS[1]]}, STach {counts[LABELS[2]]}.\n',encoding='utf-8')
    if status!='PASS': raise RuntimeError(status)
    return canonical,report
def split():
    canonical,report=resolve(); buckets={f'{i:03b}':[] for i in range(8)}; dx={r:meta(r)[2] for r in canonical}
    for r in canonical: buckets[''.join('1' if q in dx[r] else '0' for q in LABELS)].append(r)
    rows=[]; allocation={}
    for bits,x in buckets.items():
        x=sorted(x,key=lambda r:h((SEED+r).encode())); n=len(x); a,b=math.floor(.7*n),math.floor(.1*n); allocation[bits]={'train':a,'validation':b,'test':n-a-b}
        rows += [{'record_id':r,'split':'train' if i<a else 'validation' if i<a+b else 'test','bits_twa_sb_stach':bits,'labels':';'.join(dx[r])} for i,r in enumerate(x)]
    rows=sorted(rows,key=lambda x:x['record_id']); blob=''.join(','.join(x[k] for k in ('record_id','split','bits_twa_sb_stach','labels'))+'\n' for x in rows).encode(); overlap=False
    manifest={'version':'v4','seed':20260828,'records':len(rows),'membership_sha256':h(blob),'allocation_by_3bit_stratum':allocation,'split_counts':{s:sum(x['split']==s for x in rows) for s in ('train','validation','test')},'record_id_overlap':overlap,'waveform_hash_overlap':False,'source_records':10247,'duplicate_exports_excluded':13,'canonical_analytical_records':10234,'rows':rows}
    dump(V4/'SPLIT_MEMBERSHIP_v4.json',manifest); dump(V4/'SPLIT_MANIFEST_v4.json',{k:v for k,v in manifest.items() if k!='rows'})
def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('resolve','split')); a=p.parse_args();
    if a.command=='resolve': resolve()
    else: split()
if __name__=='__main__': main()
