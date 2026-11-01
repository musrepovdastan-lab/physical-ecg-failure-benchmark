"""Input validation for supported fresh reproductions.

The frozen control.py is intentionally imported unchanged because it is the
runner bound to the nine saved checkpoints. This wrapper separates immutable
scientific inputs from the three documentation files updated after freezing.
"""
from pathlib import Path
import hashlib,json
import control as frozen
from control import *  # re-export the frozen training implementation verbatim

ROOT=frozen.ROOT
OUT=frozen.OUT
DOC_PATHS=('README.md','docs/RESULTS.md','docs/next_steps.md')
BINDING=OUT/'INPUT_VALIDATION_AMENDMENT_001.json'
BINDING_SHA=OUT/'INPUT_VALIDATION_AMENDMENT_001_SHA256.txt'

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def protocol():
    p=json.loads((OUT/'FROZEN_EXECUTION_PROTOCOL.json').read_text(encoding='utf8'))
    protocol_sha=_sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json')
    if (OUT/'FROZEN_EXECUTION_PROTOCOL_SHA256.txt').read_text(encoding='ascii').strip()!=protocol_sha:
        raise RuntimeError('original frozen protocol hash mismatch')
    amendment_sha=_sha(BINDING)
    if BINDING_SHA.read_text(encoding='ascii').strip()!=amendment_sha:
        raise RuntimeError('input validation amendment hash mismatch')
    amendment=json.loads(BINDING.read_text(encoding='utf8'))
    if amendment.get('frozen_protocol_sha256')!=protocol_sha:
        raise RuntimeError('input validation amendment is bound to another frozen protocol')
    if amendment.get('control_replay_sha256')!=_sha(OUT/'control_replay.py'):
        raise RuntimeError('replay validator changed after amendment')
    if amendment.get('train_all_fresh_sha256')!=_sha(OUT/'train_all_fresh.py'):
        raise RuntimeError('supported trainer changed after amendment')
    # The original training runner and companion frozen tools remain byte-bound.
    for name,digest in p['runner_sha256'].items():
        if not (OUT/name).is_file() or _sha(OUT/name)!=digest:
            raise RuntimeError('frozen runner changed: '+name)
    if _sha(OUT/'PROPOSED_MEMBERSHIP_FROZEN.csv')!=p['frozen_membership_copy_sha256']:
        raise RuntimeError('frozen membership copy mismatch')
    if _sha(OUT/'NINGBO_SPLIT_MEMBERSHIP_FROZEN.json')!=p['frozen_split_copy_sha256']:
        raise RuntimeError('frozen split copy mismatch')
    # Strictly validate every original input, except mutable docs whose historical
    # bytes and current provenance are both checked in their dedicated bindings.
    frozen_inputs=p['input_file_sha256']
    if set(amendment['historical_document_snapshots'])!=set(DOC_PATHS):
        raise RuntimeError('historical document snapshot inventory mismatch')
    if set(amendment['current_documentation_sha256'])!=set(DOC_PATHS):
        raise RuntimeError('current documentation inventory mismatch')
    for rel,entry in amendment['historical_document_snapshots'].items():
        if frozen_inputs.get(rel)!=entry['expected_frozen_sha256']:
            raise RuntimeError('documentation expected hash differs from original protocol: '+rel)
        snapshot=OUT/entry['snapshot_path']
        if not snapshot.is_file() or _sha(snapshot)!=entry['expected_frozen_sha256']:
            raise RuntimeError('historical documentation snapshot mismatch: '+rel)
        current=ROOT/rel
        if not current.is_file() or _sha(current)!=amendment['current_documentation_sha256'][rel]:
            raise RuntimeError('current documentation provenance mismatch: '+rel)
    docs=set(DOC_PATHS)
    for rel,digest in frozen_inputs.items():
        if rel in docs:
            continue
        path=ROOT/rel
        if not path.is_file() or _sha(path)!=digest:
            raise RuntimeError('frozen scientific input changed: '+rel)
    if len(frozen_inputs)-len(docs)!=65:
        raise RuntimeError('frozen input inventory count changed')
    return p

def input_provenance_report():
    p=protocol();amendment=json.loads(BINDING.read_text(encoding='utf8'))
    return {'status':'PASS','frozen_protocol_sha256':_sha(OUT/'FROZEN_EXECUTION_PROTOCOL.json'),
      'frozen_runner_hashes_pass':'PASS','scientific_inputs_checked':len(p['input_file_sha256'])-len(DOC_PATHS),
      'mutable_documentation_snapshots_checked':len(DOC_PATHS),
      'current_documentation_hashes_checked':len(DOC_PATHS),
      'historical_document_snapshots':amendment['historical_document_snapshots'],
      'training_runner_checkpoint_binding':p['runner_sha256']['control.py']}
