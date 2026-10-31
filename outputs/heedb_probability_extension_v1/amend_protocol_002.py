import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=OUT/'FROZEN_EXTENSION_PROTOCOL.json';old=sha(p);shutil.copyfile(p,OUT/'FROZEN_EXTENSION_PROTOCOL_BEFORE_AMENDMENT_002.json');x=json.loads(p.read_text(encoding='utf-8'))
 for rel in ('outputs/heedb_probability_extension_v1/extension.py','outputs/heedb_probability_extension_v1/freeze_protocol.py'):
  f=ROOT/rel;x['bound_inputs'][rel]={'sha256':sha(f),'bytes':f.stat().st_size}
 x['replay']['implementation_note']='HEEDB uses existing frozen signals; Ningbo bounded replay waveform source is the six-method extension frozen test_waveforms_float32.npy.'
 p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');new=sha(p);(OUT/'FROZEN_EXTENSION_PROTOCOL_SHA256.txt').write_text(new+'\n',encoding='ascii')
 a={'amendment':'002','reason':'First runner invocation stopped before any replay comparison because the Ningbo test waveform path was incorrectly inferred from the training directory. Corrected to the existing six-method extension frozen waveform file. No inference comparison or HEEDB outcome was produced before the amendment; scope and analysis unchanged.','prior_protocol_sha256':old,'amended_protocol_sha256':new,'status':'AMENDED_BEFORE_REPLAY'}
 (OUT/'FROZEN_EXTENSION_AMENDMENT_002.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
