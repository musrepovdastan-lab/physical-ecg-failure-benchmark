# Reversal algebra audit

Synthetic unit checks against `src/ecg_physics.py` passed: RA–LA preserves aVF, RA–LL preserves aVL, and LA–LL preserves aVR (machine precision); each transforms the other five displayed limb leads. This evaluator uses existing cached clean/reversal predictions and the fixed sentinel convention for five displayed-lead masks.
