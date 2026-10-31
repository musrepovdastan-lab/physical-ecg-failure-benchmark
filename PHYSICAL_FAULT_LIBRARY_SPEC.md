# Physical fault library specification

## Scope and representation

Let the independent recorded signals be `x = (I, II, V1, …, V6)`. The conventional 12 displayed leads contain only eight independent measurements. In an arbitrary voltage gauge set RA = 0, LA = I, LL = II. Then

```
III = II − I
aVR = −(I + II)/2
aVL = I − II/2
aVF = II − I/2
WCT = (RA + LA + LL)/3 = (I + II)/3
Vk = Ck − WCT,  k = 1,…,6
```

The implementation in `src/ecg_physics.py` uses these equations. It tests Einthoven (`II = I + III`) and Goldberger (`aVR + aVL + aVF = 0`) identities. The signal is mathematically a 12-lead representation; it is not a patient-specific body-surface potential map or an acquisition-device circuit simulator.

## Library categories

| Category | Fault | What can be derived from a clean 12-lead ECG | Implementation status | Phase-1 control requirement |
|---|---|---|---|---|
| Conventional benchmark | Whole displayed-lead zeroing | An availability stressor only | allowed as legacy comparator, never called electrode failure | match number of displayed inputs only; secondary |
| Exact physical connection error | RA↔LA, RA↔LL, LA↔LL limb cable reversal | Exact terminal permutation; WCT is invariant | implemented | random terminal permutation of same two terminals; metrics separately per draw |
| Physically appropriate polarity change | Sign changes implied by a terminal permutation, e.g. RA↔LA makes I′=−I | Exact only as part of the cable permutation | implemented through reversal; isolated sign inversion prohibited | same as reversal |
| Single chest electrode disconnection | V1 … V6 contact lost | The corresponding independent chest measurement is unknown; open-circuit waveform is not inferable | implemented as NaN/unavailable, not zero | random one-of-six chest-measurement unavailability |
| Coherent chest-system failure | all V1–V6 contacts/cables unavailable | Six independent chest measurements unknown | implemented as NaN/unavailable | random 6-of-8 independent measurement unavailability, reported per mask |
| Limb electrode disconnection | RA, LA, or LL contact lost | I/II and WCT-referenced chest signals are not identifiable from the clean output as a physical post-fault waveform | represented as all independent measurements unavailable | random all-8 unavailability is not scientifically useful; this scenario is descriptive/quality-failure only unless raw terminal hardware data exist |
| Shared WCT fault | WCT electrode/reference chain corrupted | Unknown time-varying reference contamination cannot be recovered or parameterised from clean leads | represented as all independent measurements unavailable | no fabricated matched waveform; exclude from primary classifier benchmark unless hardware-characterised contamination is available |
| Precordial position misplacement | V1–V6 shifted/incorrectly placed | Requires body-surface potential at a new spatial position | **not implemented** | exclude unless paired mapping/controlled-position data support a transformation |

## Exact limb-reversal consequences

For RA↔LA: `I′=−I`, `II′=III`, `III′=II`, `aVR′=aVL`, `aVL′=aVR`, `aVF′=aVF`, and every chest lead is unchanged because WCT is a symmetric terminal average. Other two-terminal swaps follow directly from the terminal equations. The implementation recomputes every displayed lead from permuted RA/LA/LL terminal potentials rather than manually swapping stored derived channels.

## Non-negotiable restrictions

- Do not set an open-circuit electrode to zero and label it a physical waveform. Zero may be an inference-adapter sentinel only, after its semantics are separately frozen and matched.
- Do not independently flip a derived lead, swap displayed channels, or perturb a WCT-referenced chest lead without deriving it from terminal connections.
- Do not simulate precordial positional errors using interpolation, lead copying, arbitrary amplitude scaling, or noise; conventional 12-lead data do not observe the new position.
- Do not call a four-lead displayed mask “four electrodes.”
- Do not estimate fault prevalence from PTB-XL or Challenge 2020 without a logged acquisition-fault source.

## Test contract

`tests/test_ecg_physics.py` covers every implemented function and named fault: lead identities; RA↔LA exact equations and chest invariance; the remaining two terminal reversals; all six single-chest unavailability cases; coherent chest failure; and RA/LA/LL/WCT unavailability. Tests use synthetic signals and no clinical data.

## Sources

- [Kligfield et al., AHA ECG standardization recommendations](https://doi.org/10.1161/CIRCULATIONAHA.106.180200)
- [Challenge 2020 header and lead description](https://physionet.org/content/challenge-2020/1.0.2/)
- [Electrocardiographic lead reversals review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10685096/)
- [Machine-learning review of electrode misplacement/interchange](https://www.sciencedirect.com/science/article/pii/S0022073620305331)
