# Cao et al. (ICASSP 2020): full-text review

**Review date:** 2026-10-17  
**Primary source:** [conference paper PDF, pp. 951–955](https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000951.pdf); [DOI](https://doi.org/10.1109/ICASSP40776.2020.9053307); [ICASSP program record](https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=3120).

The proceedings PDF is the primary source. Direct page opening timed out in the web reader, but its indexed full-text extraction returned the methods, Table 1, experimental setup, Tables 2–3, and conclusion. Page references below refer to the printed page numbers in the PDF.

## What the paper did

| Question | What the authors report | Location |
|---|---|---|
| Diagnostic task and labels | Six-class rhythm/noise classification: atrial fibrillation (710), sinus arrhythmia (589), ventricular extrasystole (700), atrial extrasystole (874), sinus rhythm (785), plus 500 pure-noise examples drawn from the PhysioNet/CinC 2017 noise class. The five ECG classes sum to 3,658 patients/recordings. | §3.1, p. 953 |
| Data and split | A bespoke 12-lead set collected across China; the paper reports 3,658 patients. Train/validation/test split is 3:1:1. It does not report a separate external cohort. | §3.1–3.2, p. 953 |
| Reversal conditions | Table 1 gives limb-electrode interchange consequences for LA/RA, LA/LL, RA/LL; clockwise and counter-clockwise RA–LA–LL cycles; RA/RL, LA/RL, LL/RL; and RA/RL plus LA/LL. The paper excludes LL/RL from experiments because it says the resulting waveforms barely change. Precordial electrode reversals are not evaluated. | Table 1, p. 952; §3.3, p. 953 |
| RL handling | The table distinguishes cases involving the right-leg/neutral electrode and lists their changed limb-lead morphology. LL/RL is omitted from testing as nearly unchanged. The work does not claim to cover all open-electrode, impedance, spatial-placement, or arbitrary fault states. | Table 1, p. 952; §3.3, p. 953 |
| Were reversals used for supervised training? | No evidence that reversal examples were used for training. The methods say the proposed model and the two baselines were trained on the training set; each reversal-specific test set was simulated from the original test set. Thus this is a clean-trained robustness architecture evaluated on reversal inputs, not reversal augmentation. | §3.2–3.3, p. 953 |
| Proposed architecture | A multi-scale feature encoder plus a lead-wise/shared-weight encoder and learned attention over lead representations. The authors motivate shared encoder weights as reducing sensitivity to channel order and attention as learning cross-lead relevance. The paper gives the attention equations and an architecture diagram. | §2, pp. 952–953, Fig. 1–3 |
| Baselines and ablations | CNN+LSTM and a residual network are trained and tested alongside the proposed model. Table 3 ablates multi-scale encoding and attention separately and together on normal data. No chest-only comparator, random-mask comparator, or detect-and-correct pipeline is evaluated. | §3.3, p. 953; Tables 2–3, p. 954 |
| Metrics and repeated fits | Macro F1 across six classes. Training/testing is repeated ten times per network to address random training variation; Table 2 reports mean ± variation. Accuracy is discussed in the narrative but the central reported comparison is macro F1. | §3.3, p. 953; Tables 2–3, p. 954 |
| Main result | Normal macro-F1: proposed 73.68%, CNN+LSTM 67.66%, residual network 69.81%. Under RA/LL, proposed 73.17% (−0.51 points), versus 36.45% (−31.21) and 49.49% (−20.32) for the baselines. Other tested swaps/cycles are also reported in Table 2. | Table 2, p. 954 |
| External validation, calibration, prevalence | No external source, probability calibration, mostly-clean fault-prevalence mixture, or calibration analysis is reported in the paper. | §§3–4, pp. 953–954 |
| Data, code, weights | The paper describes its custom cohort but provides no linked dataset, implementation repository, or pretrained checkpoint in the paper/proceedings record reviewed here. The described architecture and equations support a carefully labeled adaptation, but not a verified exact reproduction of weights or the original cohort. | Paper and [ICASSP record](https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=3120); reviewed 2026-10-17 |

## Interpretation: authors' claim versus this review

**Authors' claim:** the shared-weight multi-scale network with lead-attention maintains high multi-class macro-F1 on the simulated reversal sets, and ablations indicate contributions from both attention and multi-scale features. The paper presents this as a reversal-robust arrhythmia detector.

**Review interpretation:** this is a direct prior demonstration that diagnostic classification can be tested under coherent simulated limb-electrode swaps and that a clean-trained model can be robust to several such transformations. It directly overlaps any broad claim that this project is the first to expose diagnostic models to physical lead reversal, or that reversal augmentation itself is a new general idea. It does not answer the project's narrower question about TWA/SB/STach on Chapman/Ningbo/HEEDB, the matched missing-lead control, chest-only performance, calibrated mixtures, or downstream fault correction. Its custom data, different rhythm endpoints, lack of a mask comparator, and lack of external validation limit direct numerical comparison.

## Is this an exact baseline we can reproduce?

The paper provides architecture diagrams, attention equations, reversal definitions, split ratio, optimizer, initial learning rate, weight decay, dropout, and early stopping. However, the original data are not linked, the paper/proceedings record has no implementation or weights, and several details needed for a faithful replication and exact dataset reconstruction are unavailable. Training a network that merely contains attention would not reproduce Cao et al.

A faithful port to the project's three outcomes could be attempted only as a clearly named **Cao-inspired adaptation** after recording the architecture-level decisions and verifying them against the published design. It would not be an exact reproduction because dataset, labels, preprocessing, cohort and endpoint differ. This adaptation is potentially useful but is not required to support the narrower, explicitly non-first contribution proposed in `REVISED_CONTRIBUTION.md`. The existing published Huang detector/correction chain is a more reproducible operational comparator and already has an exploratory HEEDB evaluation.

## Direct source links

- [Cao, Shi & Wu, ICASSP 2020, DOI](https://doi.org/10.1109/ICASSP40776.2020.9053307)
- [Full proceedings paper, pp. 951–955](https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000951.pdf)
- [ICASSP program entry](https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=3120)
