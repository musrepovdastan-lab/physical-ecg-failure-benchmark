# Physical ECG Failure Benchmark

Research code and results for comparing 12-lead ECG classifiers under missing
channels and physically valid limb-electrode cable reversals. The project covers
Chapman-Shaoxing, Ningbo and restricted HEEDB evaluations, mitigation experiments,
chest-only and ECGFounder baselines, and a published detection/correction pipeline.

Start with the [updated manuscript](outputs/submission_evidence_v3/MANUSCRIPT_DRAFT.md),
[corrected-series tables](outputs/submission_evidence_v3/SEED_CORRECTED_RESULTS.md)
and [results guide](docs/RESULTS.md). This repository contains project-owned code,
tests, frozen protocols and selected aggregate results.

## Structure

```text
src/                 Core transformations, training, evaluation and audits
tests/               Mathematical and workflow regression tests
configs/             Frozen study protocols; original versions retained
outputs/<study>/     Study scripts, reports and selected aggregate tables
docs/                Methods, reproduction, results and next steps
docs/archive/        Historical Phase-0 assessments
data/README.md       Data sources, access and expected local layout
requirements.txt     Minimal dependencies for the physics tests
requirements-research.txt  Research dependencies
DATA_MANIFEST.json   Historical Phase-0 data audit; required by a legacy runner
```

Study-specific scripts remain at their original `outputs/<study>/` paths because
other scripts and frozen protocols refer to them. Historical versions and failed
analyses are retained; use the corrected results linked in [docs/RESULTS.md](docs/RESULTS.md).

Documentation: [methods](docs/methods.md),
[Ningbo protocol notes](docs/ningbo_protocol_notes.md),
[reproduction](docs/REPRODUCING.md) and [current status](docs/next_steps.md).

## Quick start: no datasets or GPU needed

Use Python 3.12 and run commands from the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_ecg_physics.py -q
```

These synthetic tests check Einthoven/Goldberger identities, terminal permutations
and unavailable-measurement masks. For the broader regression suite:

```bash
python -m pip install -r requirements-research.txt
python -m pytest -q
```

Tests requiring the authors' ECGFounder source are skipped until that source is
installed at the documented path. Training, inference and restricted-data studies
have additional prerequisites; see [docs/REPRODUCING.md](docs/REPRODUCING.md).
Some orchestration and latency utilities were written for Windows/CUDA.

## Research status

The latest synthesis is dated 1 November 2026. The evidence supports a bounded
comparative evaluation: missing-channel robustness does not automatically predict
robustness to simulated cable reversals. Mitigation trade-offs depend on cohort,
endpoint, fault composition and operating point. Universal superiority and
clinical effectiveness have not been established.

Earlier masking analyses were corrected in separate recovery studies. The
Transformer M3 branch failed its full prespecified criterion. HEEDB evaluations
use coded proxy targets. The calibration-role initialization-order defect has
been addressed in a separate corrected nine-fit series, and its predictions and
700 bootstrap comparisons are complete. This is a sensitivity check on prior
data; the old initial weights are unknown, so the result does not isolate a pure
causal seed effect. At equal-three-swap q=0.1, augmentation-versus-ordinary AP
gains persist, but p050/combined AP advantages over Huang are not supported by
the paired intervals in either cohort. Manuscript version 3 distinguishes these
results from historical model families; version 2 remains unchanged.
See the [results guide](docs/RESULTS.md), the
[corrected-series report](outputs/calibration_role_control_seed_corrected_v1/FINAL_REPORT_RU.md),
and the [current status](docs/next_steps.md) for evidence and limitations.

## Publication scope

Included: project-owned code, frozen protocols, reports, selected aggregate metrics
and figures. Excluded: raw ECGs, individual-level labels/predictions, record
membership tables, model weights, environments, third-party source copies,
downloaded papers, large generated tables and intermediate caches.
See [data/README.md](data/README.md).

Historical hashes describe the original studies, including artifacts omitted from
this public snapshot. Full historical runs cannot be reproduced from a fresh
clone alone. The historical Phase-0 assessments are kept in [docs/archive/](docs/archive/).
External resources retain their own access/license terms. A project-wide
open-source license has not yet been selected.
