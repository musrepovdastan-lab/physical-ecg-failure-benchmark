# Amendment 011 — exact accelerated paired bootstrap

The first full bootstrap implementation repeatedly sorted identical prediction scores inside sklearn for every resample and had not written final outputs after ten CPU minutes. The run was stopped before summary tables or intervals were finalized; source predictions and lock files are unchanged.

The replacement precomputes descending unique-score groups per method/seed and calculates weighted AP from exact tie-group positive/total mass under the same frozen synchronized multiplicities. It also computes weighted Brier, sensitivity, and FP directly. Before accepting intervals, actual draws 0, 417, and 999 are recalculated independently with sklearn for every saved contrast key. Point estimates remain the full-cohort sklearn results; bootstrap means and percentile intervals are derived from all 1,000 frozen draws.
