"""Read-only direct validation path used by the supported reproduction wrapper."""
import json
from pathlib import Path
import control_replay as c

def main():
    report=c.input_provenance_report()
    if report['status']!='PASS' or report['scientific_inputs_checked']!=65 or report['mutable_documentation_snapshots_checked']!=3:
        raise RuntimeError('input provenance check failed')
    existing=json.loads((c.OUT/'REPRODUCTION_INPUT_CHECK.json').read_text(encoding='utf8'))
    for key in ('status','frozen_protocol_sha256','scientific_inputs_checked','mutable_documentation_snapshots_checked','current_documentation_hashes_checked','training_runner_checkpoint_binding'):
        if existing.get(key)!=report.get(key):raise RuntimeError('input report does not match current provenance: '+key)
    print(json.dumps({'status':'PASS','scientific_inputs':65,'historic_document_snapshots':3,'training_started':False}))
if __name__=='__main__':main()
