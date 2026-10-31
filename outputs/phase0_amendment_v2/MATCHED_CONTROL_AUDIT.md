# Matched-control audit — Phase-0 amendment v2

## Corrected question

> Do conventional random lead-dropout evaluations predict the diagnostic robustness of 12-lead ECG classifiers under a preregistered library of physically realizable acquisition faults?

This is deliberately directional-neutral. A model may be more, less, or differently robust under conventional random dropout than under a physical-fault family. Class-specific disagreement, rank disagreement, or calibration disagreement is informative.

## Family A — availability failures

The analysis operates on eight independent signals: `I`, `II`, and `V1`–`V6`. The six derived limb displays are reconstructed after any availability operation using the equations in `src/ecg_physics.py`; a physical failure is never counted as a number of stored displayed channels.

| Physical availability scenario | Independent measurements unavailable | Matched random condition | Estimand |
|---|---:|---|---|
| One unavailable precordial measurement | 1 of V1–V6 | Each random chest measurement (`V1`…`V6`), separately evaluated | physical degradation − mean random degradation |
| Coherent precordial connector/block loss | fixed preregistered subset of V1–V6, initially V1–V4 | random k-of-8 independent-measurement unavailability, k equal to the fixed subset size, separately evaluated | physical degradation − mean random degradation |

An unavailable measurement is a missingness indicator / model-input availability state, not an invented zero-valued physiological trace. Random masks are sampled once per specified replicate, inference is run separately for every random draw, and a metric—not prediction vectors—is averaged inside each patient bootstrap draw.

Open limb-electrode loss and WCT corruption are excluded from this primary comparison because clean 12-lead data cannot identify their actual post-fault waveform. Spatial chest-electrode misplacement, impedance, motion artefact, and arbitrary additive noise remain excluded without paired physical measurements.

## Family B — connection errors

RA↔LA, RA↔LL, and LA↔LL are exact terminal permutations. They preserve information but alter its representation; matching them to dropout by channel count would be physically and statistically false. Their primary outcomes are clean-to-reversal class-specific AUPRC degradation, macro-AUPRC, ECE, frozen-specificity sensitivity, and model ranking.

The comparison is predictive agreement, not a synthetic control: across architectures × seeds × source/class rows, compute Spearman rank correlation between clean/random-dropout robustness ranks and terminal-reversal robustness ranks; report the complete matrix and class-specific degradation differences. A low or sign-changing correlation rejects the proposition that conventional dropout rankings predict reversal robustness.

## Leakage and sealing safeguards

- Fault library, primary labels, metrics, split algorithm, random-mask count, and correlation endpoint are frozen before internal-test or Ningbo prediction access.
- Chapman record IDs are one-record-per-patient by the authoritative Challenge table; a deterministic, stratified `70/10/20` train/validation/internal-test allocation therefore is patient-disjoint.
- Ningbo labels are used only for eligibility counts. No Ningbo waveform, model prediction, training, tuning, threshold fitting, or fault selection occurs in Phase 0.
- The earlier PTB-XL V1–V4 displayed-lead result remains historical exploratory motivation only; it has no role in endpoint, fault, or threshold selection.

## Test status

The existing source-level unit tests cover the exact equations and all implemented named transformations. All four named test functions passed in this amendment using the project runtime with transient access to legacy read-only NumPy; no legacy file or environment was changed. The available host lacks pytest and its runtime lacks `venv`, so `pytest` command-line execution remains a Phase-1 setup checkpoint. Python syntax compilation and JSON parsing succeeded. No transformation is altered in v2.
