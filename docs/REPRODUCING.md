# Reproduction guide

Run commands from the repository root. Python 3.12 was used locally.

## Synthetic checks

The minimal install and `python -m pytest tests/test_ecg_physics.py -q` need only
NumPy and pytest. They do not access ECG data or train models.

Install `requirements-research.txt` for the broader regression suite. Requirements
specify compatible ranges, not an exact environment lock. Historical environment
audits record the original study environments. Results can depend on package
versions, CUDA, hardware and randomization.

## Research workflow

1. Choose a study and read its protocol in `configs/` or `outputs/<study>/`.
2. Obtain its datasets under their access terms; see [data/README.md](../data/README.md).
3. Recreate prerequisite ingestion audits, cohort/split artifacts, training outputs
   and validation gates. Public aggregate reports do not replace them.
4. Run the matching module from the root. Preserve version suffixes: a recovery
   script is not interchangeable with the original analysis.
5. Compare generated results with the corresponding final report and hashes.

There is no single command that reproduces the complete research history.
`python -m src.chapman_stage23 --help` describes Chapman acquisition/audit commands;
`python -m src.ningbo_confirmation_v2 --help` describes the separate Ningbo workflow.
Archived authorization/status statements do not authorize new experiments or
replace dataset-access permission. Some orchestration/latency tools require Windows.

## External models

**ECGFounder:** obtain [the authors' code](https://github.com/PKUDigitalHealth/ECGFounder),
commit `04edac702b61c91face519774ddcc0cd712fef23`, and place it at:

```text
outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/
```

It must include `net1d.py` and `util.py`. Obtain `12_lead_ECGFounder.pth` separately
from [the authors' model repository](https://huggingface.co/PKUDigitalHealth/ECGFounder)
and place it in `outputs/ecgfounder_baseline_v1/official/`. The source revision,
checkpoint hash and preprocessing settings are recorded in
[the frozen v2 protocol](../configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json).
Synthetic tests need no weights; tests requiring the authors' code skip if absent.

**Huang detector:** use the source/model described in
[the frozen detector protocol](../outputs/published_detector_pipeline_v1/FROZEN_PROTOCOL.json)
from [wjcai/ECG_lead_check](https://github.com/wjcai/ECG_lead_check). The workflow
expects source at `outputs/published_detector_pipeline_v1/official_source/ECG_lead_check-main/`
and separately prepared loadable weights. TensorFlow inference used an isolated
`detector_env`, not the main PyTorch environment. Its model/framework compatibility
is documented in that study's reports. These dependencies are not bundled.

**HEEDB:** acquisition needs AWS CLI and separately authorized access. Credentials,
identifiers, labels and predictions stay under `data/heedb_restricted/`. Aggregate
outputs cannot reconstruct the restricted cohort.

## Repository maintenance

New code belongs in `src/` or its existing study directory. Generated outputs are
ignored by default; selected aggregate reports can be added explicitly with
`git add -f outputs/<study>/<report>`. Review contents first. Never force-add raw or
restricted data, individual-level exports, credentials, dependencies or weights.

`.gitattributes` preserves original file bytes for historical SHA-256 checks.
`PUBLICATION_MANIFEST.json` lists this release's files. Historical manifests can
reference omitted local artifacts; editing documentation/tests does not re-freeze
the original experiments.
