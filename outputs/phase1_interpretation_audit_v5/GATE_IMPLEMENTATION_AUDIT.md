# Gate implementation audit

The frozen primary estimand was `(clean − V1V4) − (clean − mean_other69)`. Its observed signed value is -0.163688 (95% CI -0.193069 to -0.134923). Negative means V1–V4 was **less damaging** than the mean of the other 69 masks.

The v4 implementation used `abs(primary_effect) >= 0.05`, accepting a significant contrast in either direction. It therefore did not enforce a prespecified positive direction and did not directly verify that one particular physical scenario itself had AUPRC loss ≥0.05 with its own synchronized interval. The automatic `PROCEED_MULTISEED` result is not a valid frozen confirmation decision.
