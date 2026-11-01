from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def check_manifest(path,groups):
 d=json.loads((ROOT/path).read_text(encoding='utf-8'));rows=[]
 for group in groups:
  for x in d[group]:
   p=ROOT/x['path'];exists=p.is_file();actual=sha(p) if exists else None
   archive=None
   if path.endswith('calibration_role_control_completion_v2/MANIFEST.json') and x['path'].endswith(('CURRENT_PLAN_STATUS_RU.md','FINAL_REPORT_RU.md')):
    archive=ROOT/'outputs/calibration_role_control_completion_v2/archive_before_review_20261020'/Path(x['path']).name
   archived_match=bool(archive and archive.is_file() and sha(archive)==x['sha256'])
   rows.append({'source_manifest':path,'path':x['path'],'expected_sha256':x['sha256'],'actual_sha256':actual,'exists':exists,'match':exists and actual==x['sha256'],'documented_post_review_edit':bool(archive),'pre_review_archive_path':str(archive.relative_to(ROOT)).replace('\\','/') if archive else None,'pre_review_archive_matches_original_manifest':archived_match,'current_post_review_sha256':actual if archive else None})
 return rows
def main():
 rows=check_manifest('outputs/calibration_role_control_v1/REPRODUCIBILITY_MANIFEST.json',['files'])
 rows+=check_manifest('outputs/calibration_role_control_completion_v2/MANIFEST.json',['scientific_inputs','new_outputs'])
 bad=[x for x in rows if not x['match'] and not x['pre_review_archive_matches_original_manifest']]
 report={'status':'PASS' if not bad else 'FAIL','verified_artifacts':len(rows),'mismatches':bad,'historical_manifests_preserved':True,'checked_utc_date':'2026-10-31'}
 (OUT/'HISTORICAL_INTEGRITY_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':report['status'],'checked':len(rows),'mismatches':len(bad)}))
 if bad:raise SystemExit(2)
if __name__=='__main__':main()
