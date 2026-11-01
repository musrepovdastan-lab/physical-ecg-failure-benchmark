"""Export manuscript tables from aggregate, already completed analyses only.

No ECGs, individual predictions, fitting, inference or bootstrap generation.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/calibration_role_control_seed_corrected_v1"
LABELS = {"ordinary": "Ordinary", "physics_aware_p050": "Physics-aware p050",
          "combined": "Combined", "Huang_full_pipeline": "Huang pipeline"}
METHODS = tuple(LABELS)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mean_path = SOURCE / "MIXTURE_METRICS_MEAN_SD.csv"
    seed_path = SOURCE / "MIXTURE_METRICS_BY_SEED.csv"
    contrast_path = SOURCE / "TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv"
    leader_path = SOURCE / "AP_LEADERSHIP_OLD_VS_CORRECTED.csv"
    means = pd.read_csv(mean_path)
    means = means[means.endpoint == "TWA"].copy()
    seeds = pd.read_csv(seed_path)
    seeds = seeds[seeds.endpoint == "TWA"].copy()
    contrasts = pd.read_csv(contrast_path)
    assert len(means) == 112 and len(seeds) == 336 and len(contrasts) == 700
    means.to_csv(OUT / "SEED_CORRECTED_TWA_MEANS.csv", index=False)
    seeds.to_csv(OUT / "SEED_CORRECTED_TWA_BY_SEED.csv", index=False)
    # Retain exact source bytes for the complete paired-contrast table.
    (OUT / "SEED_CORRECTED_TWA_CONTRASTS.csv").write_bytes(contrast_path.read_bytes())

    def row(cohort, method, q):
        g = means[(means.cohort == cohort) & (means.method == method) &
                  (means.scenario == "equal_three_swap") & (means.q_fault == q)]
        assert len(g) == 1
        return g.iloc[0]

    def interval(cohort, method, metric, right="ordinary", q=.1):
        g = contrasts[(contrasts.cohort == cohort) & (contrasts.left == method) &
                      (contrasts.right == right) & (contrasts.metric == metric) &
                      (contrasts.scenario == "equal_three_swap") & (contrasts.q_fault == q)]
        assert len(g) == 1
        r = g.iloc[0]
        precision = 1 if metric == "FP_per_1000_raw" else 4
        return (f"{r.full_cohort_point_mean_seed_contrast:+.{precision}f} "
                f"[{r.ci95_low:+.{precision}f}, {r.ci95_high:+.{precision}f}]")

    lines = ["# Corrected initialization series: manuscript tables", "",
        "These tables summarize completed fits; generating them does not run models or statistics. "
        "TWA is primary. AP means and SD are calculated over three seed-specific metrics, not an ensemble. "
        "q is a constructed evaluation mixture weight, not observed fault prevalence.", "",
        "## Table 1. Raw TWA AP in the equal-three-swap scenario", "",
        "| Cohort | Pipeline | q=0 | q=0.1 | q=1 |",
        "|---|---|---:|---:|---:|"]
    for cohort in ("Ningbo_test", "HEEDB"):
        for method in METHODS:
            cells = [f"{row(cohort, method, q).AP_raw_mean:.4f} ± "
                     f"{row(cohort, method, q).AP_raw_sd:.4f}" for q in (0., .1, 1.)]
            lines.append(f"| {cohort} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "## Table 2. Differences from ordinary at equal-three q=0.1", "",
        "Entries are full-cohort mean-seed differences [unadjusted 95% paired percentile interval]. "
        "Higher AP/sensitivity is favorable; lower Brier/false positives is favorable. "
        "False positives are per 1,000 evaluated ECGs, not per 1,000 negatives.", "",
        "| Cohort | Pipeline | AP raw | Brier calibrated | Sensitivity raw | FP per 1,000 raw |",
        "|---|---|---:|---:|---:|---:|"]
    for cohort in ("Ningbo_test", "HEEDB"):
        for method in ("physics_aware_p050", "combined"):
            cells = [interval(cohort, method, m) for m in
                     ("AP_raw", "Brier_calibrated", "sensitivity_raw", "FP_per_1000_raw")]
            lines.append(f"| {cohort} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "## Table 3. AP comparisons against Huang", "",
        "Positive values favor the named training method over the complete Huang pipeline.", "",
        "| Cohort | Pipeline minus Huang | equal-three q=0.1 | equal-three q=1 |",
        "|---|---|---:|---:|"]
    for cohort in ("Ningbo_test", "HEEDB"):
        for method in ("physics_aware_p050", "combined"):
            cells = [interval(cohort, method, "AP_raw", right="Huang_full_pipeline", q=q)
                     for q in (.1, 1.)]
            lines.append(f"| {cohort} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "At q=0.1, neither p050 nor combined has an AP interval excluding zero against "
        "Huang in either cohort. The historical positive comparisons must not be carried forward "
        "as evidence of the corrected series. At q=1 Huang is the point leader in both cohorts; "
        "its difference from p050 on Ningbo includes zero. Crossing zero does not establish equivalence.", "",
        "## Table 4. Point leaders before and after corrected initialization", "",
        "| Cohort | Scenario | q | Original leader | Corrected leader |",
        "|---|---|---:|---|---|"]
    leaders = pd.read_csv(leader_path)
    for r in leaders[leaders.q_fault.isin([0., .1, 1.])].itertuples():
        lines.append(f"| {r.cohort} | {r.scenario} | {r.q_fault:g} | "
                     f"{LABELS[r.original_leader]} | {LABELS[r.corrected_leader]} |")
    lines += ["", "## Scope and provenance", "",
        "The complete CSVs contain both RA–LL-only and equal-three-swap scenarios at "
        "q = 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1. The 700 contrast groups use 1,000 synchronized "
        "bootstrap replicates each, with records resampled for Ningbo and patients for HEEDB. "
        "Intervals condition on the fitted models/calibrators and are not multiplicity-adjusted. "
        "The original calibration-role series and the corrected series use the same data roles "
        "and scientific recipes; original initialization states are unknown. The comparison is "
        "a sensitivity analysis on previously examined data, not an isolated causal estimate "
        "or independent external validation.", "",
        "Sources: [corrected mean/SD table](../calibration_role_control_seed_corrected_v1/MIXTURE_METRICS_MEAN_SD.csv), "
        "[paired contrasts](../calibration_role_control_seed_corrected_v1/TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv), "
        "[old/new leaders](../calibration_role_control_seed_corrected_v1/AP_LEADERSHIP_OLD_VS_CORRECTED.csv).", ""]
    (OUT / "SEED_CORRECTED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    provenance = {"scope": "Aggregate table export only; no training, inference, calibration or bootstrap generation.",
                  "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                   (mean_path, seed_path, contrast_path, leader_path)},
                  "export_rows": {"means": 112, "by_seed": 336, "contrasts": 700}}
    (OUT / "TABLE_SOURCE_BINDINGS.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["export_rows"]))


if __name__ == "__main__":
    main()
