# Engineering amendment 007

The legacy Huang table uses explicit score-mode names `raw_validation_B_90pct` and `calibrated_validation_B_90pct`. The first history lookup used generic names and stopped before bootstrap. The comparator lookup now maps only Huang to those exact labels, preserving the required full-pipeline source-validation-B threshold. No historical threshold or result was substituted.
