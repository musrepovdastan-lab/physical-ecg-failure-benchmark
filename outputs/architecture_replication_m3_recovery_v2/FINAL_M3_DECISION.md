# Final M3 decision

**FAIL_M3_ARCHITECTURE_SPECIFIC**

Stage M3A fails the frozen gate because the physics-aware 792-mask mean TWA AUPRC for seed 155921 declines by 0.1126 relative to its clean-trained counterpart; the allowed per-seed decline is 0.04. Other Stage M3A criteria pass, including all three RA–LL gains, aggregate contrast reduction, physics-aware contrast upper 95% limit, mean clean decline, mean mask decline, validation, and integrity.

This gate failure is a performance-retention failure at one seed, not an absence of the measured RA–LL protection benefit. The unchanged protocol therefore blocks Stage B. No Ningbo Transformer training or evaluation was performed.

The result is a corrected Chapman reanalysis after prior test access. It is not untouched confirmation. See `CHAPMAN_REPORT.md` and `conclusions_m3.json` for all estimates and criteria.
