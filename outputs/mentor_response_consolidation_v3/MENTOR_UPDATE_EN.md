# Mentor update

We completed the planned computational follow-up. Missing-lead robustness did not ensure robustness to physical lead reversals. We added chest-only controls, exact affected-lead masks, four training recipes, two unseen cycle compositions, and the 0/10/25/50% augmentation grid. At matched 25% reversal exposure, combined training improved mean-mask performance over reversal-only training in both sources, with remaining implementation differences limiting causal attribution.

Augmentation also helped heads on a frozen ECGFounder encoder, including both unseen cycles, but reduced clean Ningbo performance and did not establish a general gain across its 32-mask panel. The Transformer's failed mask-retention gate remains part of the evidence.

On the previously examined HEEDB proxy cohort, results depended on fault share and metric; Huang remained competitive. We frame the contribution as a controlled evaluation of distinct robustness modes and their tradeoffs, not a new general augmentation principle or the best architecture. Clinician review of endpoint definitions and proxy mappings remains outstanding. Confirmed real miswirings and additional acquisition stresses would strengthen external validity but are optional extensions to the current simulation scope.
