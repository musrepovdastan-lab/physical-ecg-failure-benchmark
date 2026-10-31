# Related work and the remaining gap

## Search scope and conclusion

This Phase-0 search covered 2022–2026 primary literature and authoritative conference/preprint records on random and adversarial masking, reduced-lead classification/reconstruction, electrode disconnection/reversal, precordial misplacement, and physiologically valid transformations. It does not justify the absolute statement “no one has done this.” Rather, it supports the narrower conclusion that no located 2024–2026 paper performed the same **predeclared matched physical-fault-versus-random benchmark**: exact limb cable permutations plus explicitly non-fabricated loss states, matched by independent measurement impact / transformed or unavailable signal budget / prevalence / population / inference budget, evaluated with cluster inference across sealed external sources.

## Strongest adjacent papers and precise difference

1. **Oh et al. (2022), “Lead-agnostic Self-supervised Learning for Local and Global Representations of ECG.”** The work uses random lead masking during self-supervised pretraining to support arbitrary lead availability and evaluates arrhythmia classification and identification. It is the strongest random-mask comparator. It does not model electrode terminal faults, enforce physically matched controls, or report a fault-versus-random clinical-classification degradation estimand. [Paper](https://proceedings.mlr.press/v174/oh22a.html)

2. **Bo et al. (2022), “Pretraining ECG Data with Adversarial Masking Improves Model Generalizability for Data-Scarce Tasks.”** This learns adversarial **time-domain** masks for SSL and compares downstream transfer accuracy. It is not a physical electrode/cable fault model, does not study whole-electrode consequences or WCT, and does not execute a matched benchmark of clinical classifier degradation. [Paper](https://arxiv.org/abs/2211.07889)

3. **Huang et al. (2024), “A lightweight deep learning approach for detecting electrocardiographic lead misplacement.”** This develops detectors for limb and chest lead misplacement. It establishes clinical relevance of misplacement and detection, but its outcome is error detection rather than the degradation of an independent diagnostic classifier relative to matched random corruption. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/38663434/)

## Other material that narrows the claim

- **Reduced-lead classification:** Wickramasinghe & Athif (2022) train/evaluate a reduced-lead classifier. Selecting a smaller intended acquisition differs from testing a fixed 12-lead classifier after a fault. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/35617943/)
- **Reconstruction:** Seo et al. (2025) reconstruct a 12-lead surrogate from I or I+II and quantify its diagnostic feasibility. Reconstruction is not fault robustness; their observed regression-to-the-mean in precordial amplitudes makes it especially unsuitable as a hidden replacement for a lost lead in this benchmark. [Article](https://www.nature.com/articles/s43856-025-00814-w)
- **Precordial misplacement:** Kligfield-style physical implications and the 2021 deep-learning V1/V2 placement detector demonstrate that spatial position changes morphology, but standard 12-lead recordings alone do not contain measurements at the displaced body-surface locations. [Detector study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8087970/)
- **Limb reversals:** A 2023 review describes limb and precordial reversals as clinically consequential, and supports treating terminal permutation—not independent display-channel sign flipping—as the physically correct operation. [Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10685096/)
- **Recent diagnostic-model robustness:** 2025 work on LV systolic dysfunction reports lead-reversal robustness across configurations. It is an important near-neighbour; it does not, from the available report, specify a matched random comparator or the proposed endpoint across multiple non-PTB sources. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12346549/)

## What remains potentially new

If the data gate is repaired, the contribution is an **evaluation protocol**, not a new augmentation method:

- distinguish exact terminal permutations from non-identifiable electrode-open and position faults;
- compare each physical scenario with random conditions matched by *independent* measurements affected and mechanism (permutation vs unavailable), rather than simply displayed-lead count;
- calculate every random-mask metric separately within each cluster bootstrap then average metrics, avoiding an artificial random-prediction ensemble;
- separate source development from a sealed source confirmation and keep augmentation success out of the primary claim.

The present project is NO-GO because the data design cannot support the final two bullets, not because the gap has been disproved.
