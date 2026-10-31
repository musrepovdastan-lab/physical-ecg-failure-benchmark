# FINAL DEPLOYMENT TRADE-OFF DECISION

## Plain answer

Yes. The single physics-aware model has a substantial demonstrated deployment advantage on this machine: its warm batch-1 CPU median was 10.71 ms versus 34.04 ms for Huang detect-correct-diagnose (3.2x slower for Huang), and its measured peak working set was 36% lower than the conservative two-process Huang total. Diagnostic TWA AUPRC was statistically unresolved between the two approaches in v2, so this operational result is the main demonstrated reason to prefer the single model.

This does not establish clinical superiority, equivalence, or SOTA performance.

## Compact comparison

AUPRC triples are ordered TWA/SB/STach.

| Dimension | Physics-aware single model | Huang detect-correct-clean ResNet | Source |
|---|---:|---:|---|
| Clean AUPRC (TWA/SB/STach) | 0.6869/0.9962/0.9867 | 0.6754/0.9960/0.9842 | [worst_fault_summary.csv](../published_detector_multifault_v2/worst_fault_summary.csv) |
| Worst-fault AUPRC (TWA/SB/STach) | 0.6665/0.9954/0.9843 | 0.6636/0.9960/0.9780 | [worst_fault_summary.csv](../published_detector_multifault_v2/worst_fault_summary.csv) |
| Detector error | Not applicable | 2.18% reversal misclassification; 1.49% clean false correction | [detector_metrics_all_faults.json](../published_detector_multifault_v2/detector_metrics_all_faults.json) |
| Warm batch-1 CPU latency median/p95 | 10.71/11.93 ms | 34.04/40.18 ms | [latency_summary.json](latency_summary.json) |
| Peak working set | 614.2 MiB | 963.0 MiB (two-process sum) | [latency_summary.json](latency_summary.json) |
| Model artifacts | 3.70 MiB | 3.81 MiB | [latency_summary.json](latency_summary.json) |
| Core limitation | One diagnostic model; no explicit fault label | Two runtimes/models; detector mistakes can trigger wrong correction | [detector_mistake_diagnostic_errors.csv](detector_mistake_diagnostic_errors.csv) |

The three reversals are electrically exact simulations on the same 6,981 Ningbo records. The balanced latency fixture is not a measured clinical prevalence distribution.

## Detector failure cases

The detector produced 239 missed reversal condition-record instances, 217 wrong-reversal-type instances, and 104 false corrections of clean ECGs.
Missed-reversal breakdown: RA-LA 50, RA-LL 22, LA-LL 167. Wrong-type breakdown: RA-LA 78, RA-LL 110, LA-LL 29.

Diagnostic error rates below pool the three frozen seeds at their existing validation-fixed thresholds and are descriptive seed-record error rates.

| Detector mistake | Diagnosis | Physics-aware error | Huang pipeline error |
|---|---|---:|---:|
| Missed reversal (239) | TWA | 15.62% (112/717) | 20.36% (146/717) |
| Missed reversal (239) | SB | 10.74% (77/717) | 9.76% (70/717) |
| Missed reversal (239) | STach | 24.55% (176/717) | 23.15% (166/717) |
| Wrong reversal type (217) | TWA | 5.07% (33/651) | 29.80% (194/651) |
| Wrong reversal type (217) | SB | 11.06% (72/651) | 8.76% (57/651) |
| Wrong reversal type (217) | STach | 24.12% (157/651) | 27.19% (177/651) |
| False correction of clean (104) | TWA | 13.46% (42/312) | 17.31% (54/312) |
| False correction of clean (104) | SB | 14.42% (45/312) | 14.10% (44/312) |
| False correction of clean (104) | STach | 25.00% (78/312) | 27.24% (85/312) |

Illustrative one-way anonymized IDs:

- Missed reversals: NGB-01B0F4A469AF, NGB-01BC4A6B0918, NGB-02595F39AB9D.
- Wrong reversal types: NGB-00A3BE8F5969, NGB-00D943357402, NGB-043CA1C4BE5F.
- False corrections of clean ECGs: NGB-02D482A5CC33, NGB-08899A665217, NGB-09B7762149AB.

Five examples per group and their simulated condition are stored in [detector_mistake_examples.json](detector_mistake_examples.json); no original record IDs are exported.

## Publication assessment

Yes, the evidence is sufficient for a carefully scoped workshop paper framed as a reproducible operational audit and negative performance-comparison result. The contribution would be the frozen multifault comparison, detector-failure decomposition, and deployment trade-off, not a new model or superiority claim.

A stronger conference claim is blocked by specific missing evidence: real rather than simulated cable reversals; independent multi-device and multi-center external validation; a prospectively prespecified noninferiority or equivalence design; deployment measurements in one integrated target implementation rather than a conservative two-process component sum; and workflow evidence at a measured clinical fault prevalence showing the consequences of false corrections. Calibration, noise/artifact robustness, partial lead failures, and prospective utility also remain untested.

No new training method is proposed by this audit.
