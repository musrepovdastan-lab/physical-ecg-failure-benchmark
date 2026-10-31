# ?????? ? calibration role control v1

**?????????:** ?????????, ??????? ???????????????? ????????.

- Execution protocol frozen before training; canonical fingerprint: `ae1fc61c15eeffbdc0847c4ba69f78cf1a82ee24e5a0ca5c40e686c4ad40654a`.
- 9/9 trainings PASS; 36/36 calibration and threshold locks PASS.
- Role access and exact record/waveform partition audit: PASS; patient independence is not established.
- Best checkpoint/history replay: PASS for all nine.
- Huang source-aligned handoff replay: PASS (72 checks).
- Independent metric recomputation: PASS (24,624 rows; max absolute delta 5.68e-14).
- Bootstrap summary recomputation: PASS (300 rows); 432 sklearn draw checks PASS.
- Reserved test: not evaluated.
- Historical artifacts: preserved; new outputs only in this directory.

See `FINAL_REPORT_RU.md` for findings and limitations. No further experiments or paper work started.
