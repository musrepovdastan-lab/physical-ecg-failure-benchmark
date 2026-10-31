# Amendment 003 — score archive writer argument

All nine training fits and their checkpoint/history replays passed the frozen adequacy gates and were locked before any A/B scoring. The first clean A score forward pass then stopped at serialization because the writer's path argument `p` conflicts with an array keyword also named `p`. The failure is preserved in `FAILED_ATTEMPT_003.json`; no test or HEEDB waveforms were read, and no calibration or threshold was fit.

Continue with a separate postprocessing script that uses a destination argument name distinct from the `p` prediction array, revalidates all nine frozen checkpoint and membership bindings, reruns A/B scores, and resumes the already authorized calibration/evaluation. The nine completed fits are reused, not repeated. Scientific scope and memberships stay frozen.
