# M2 validation-writer incident

## Root cause

The initial M2 process stopped after frozen early stopping for seed `104729` while
attempting to atomically save `validation.npz`.  The helper was declared as
`an(p, **x)`, while the validation prediction array was correctly named `p`.
Thus `an(validation_path, y=y, p=predictions, thresholds=t)` supplied two values
for parameter `p`, producing `TypeError: an() got multiple values for argument
'p'`.

## Preserved scientific state

No historical output was changed.  The existing seed-104729 `best.pt`,
`checkpoint.pt`, `preprocessing.npz`, and `resume.pt` are retained in place.  The
resume state records the frozen 22-epoch early-stopping history and is used only
to recover validation from `best.pt`; recovery must not run another training
epoch.

## Test-access proof

At the time of this incident, `WORKFLOW_STATUS.json` still reported stage
`TRAINING`, seed `104729`, epoch `22`.  No `validation_gate.json`,
`evaluation_freeze.json`, `evaluation_complete.json`, or `test_predictions/`
directory existed for that seed.  The traceback is in the preserved
`m2_runner.stderr.log`; its final failing statement is the validation NPZ write,
before the evaluator call.  Therefore no M2 test waveform or M2 test inference
was accessed before this recovery.

## Recovery controls

The atomic NPZ helper path argument was renamed to `target` in both M2 code
paths, and a regression self-test writes and reloads arrays named `y`, `p`, and
`thresholds` before scientific recovery starts.  A validation-only recovery path
loads the saved best checkpoint and preserves the original training history.
