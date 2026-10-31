# FINAL_DECISION

Detector RA-LL recall: 0.9811. Clean specificity: 0.9851.

Mean TWA AUPRC under RA-LL reversal:
- clean_resnet_no_correction: 0.1822
- physics_aware_no_correction: 0.6665
- huang_argmax_correct_clean_resnet: 0.6636

Best deployable arm by mean RA-LL TWA AUPRC: physics_aware_no_correction.
This is an operational evaluation of the published pretrained detector plus our frozen clean diagnostic model, not a direct reproduction of Huang et al.'s reported paper scores and not a SOTA claim.
No diagnostic model, split, or 792-mask benchmark was retrained or rerun.

What remains untested: detector calibration or threshold policies beyond argmax, real-world low-prevalence workflow costs beyond the optional illustrative mixture, other cable faults as clinical prevalence scenarios, and any adaptation of Huang's detector to Ningbo.
