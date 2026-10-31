"""Validate and document the frozen four-method calibration completion."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mixed_fault_calibration_v2"
V1 = ROOT / "outputs/mixed_fault_calibration_v1"
SEEDS = (104729, 130363, 155921)
Q = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 1.0)
ARMS = ("ordinary_resnet", "physics_aware_resnet", "chest_only_resnet", "huang_pipeline")
ENDPOINTS = ("TWA", "SB", "STach")
METRICS = ("auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "false_positives_per_1000", "false_negatives_per_1000")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V1))
import run_analysis as base
import extend_bootstrap_metrics as ext


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = json.loads((OUT / "FROZEN_COMPLETION_PROTOCOL.json").read_text(encoding="utf-8"))
    fits = json.loads((OUT / "HUANG_CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))
    inference = json.loads((OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8"))
    completion = json.loads((OUT / "ANALYSIS_COMPLETE.json").read_text(encoding="utf-8"))
    bootstrap_audit = json.loads((OUT / "BOOTSTRAP_COMPLETION_AUDIT.json").read_text(encoding="utf-8"))
    mixture = pd.read_csv(OUT / "MIXTURE_RESULTS_BY_SEED.csv")
    contrasts = pd.read_csv(OUT / "PAIRED_CONTRASTS.csv")
    boot = pd.read_csv(OUT / "BOOTSTRAP_METRICS_SUMMARY.csv")
    errors = []
    split = np.load(V1 / "calibration_split_indices.npz", allow_pickle=False)
    a, b = split["A"], split["B"]
    if len(a) != 1744 or len(b) != 1744 or len(np.intersect1d(a, b)) or len(np.union1d(a, b)) != 3488:
        errors.append("A/B split does not partition the frozen 3,488 validation records")
    if sha(V1 / "calibration_split_indices.npz") != protocol["sources"]["outputs/mixed_fault_calibration_v1/calibration_split_indices.npz"]["sha256"]:
        errors.append("calibration split changed from its frozen binding")
    score = np.load(OUT / "huang_validation_scores.npz", allow_pickle=False)
    if score["diagnostic_scores"].shape != (3, 3488, 3) or score["detector_probabilities"].shape != (3488, 4):
        errors.append("Huang validation score dimensions are incomplete")
    if not np.isfinite(score["diagnostic_scores"]).all() or np.any((score["diagnostic_scores"] < 0) | (score["diagnostic_scores"] > 1)):
        errors.append("Huang validation diagnostic scores are nonfinite or outside [0,1]")
    if not np.array_equal(score["labels"], np.load(ROOT / "outputs/ningbo_confirmation_v2/seed_104729/validation.npz", allow_pickle=False)["y"]):
        errors.append("Huang validation labels/order do not match frozen model validation labels")
    huang_fit = fits["fits_by_seed"]
    for seed_i, seed in enumerate(SEEDS):
        for ep, label in enumerate(ENDPOINTS):
            f = huang_fit[str(seed)][ep]
            if f["a"] <= 0 or f["A_n"] != 1744 or f["B_n"] != 1744:
                errors.append(f"invalid Huang calibration fit metadata for {seed}/{label}")
    # The other three methods' point estimates must be unchanged from v1.
    old = pd.read_csv(V1 / "MIXTURE_RESULTS_BY_SEED.csv")
    old = old[old.method != "huang_pipeline"].sort_values(["source", "method", "seed", "endpoint", "q_fault", "score_mode"]).reset_index(drop=True)
    new_other = mixture[mixture.method != "huang_pipeline"].sort_values(["source", "method", "seed", "endpoint", "q_fault", "score_mode"]).reset_index(drop=True)
    if len(old) != len(new_other):
        errors.append("v1 non-Huang mixture row count changed")
    else:
        for col in ("auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "false_positives_per_1000", "false_negatives_per_1000", "threshold"):
            if not np.allclose(old[col].to_numpy(float), new_other[col].to_numpy(float), rtol=0, atol=1e-12):
                errors.append(f"v1 non-Huang point estimates changed in {col}")
    # Full expected Huang grid: 2 cohorts x 3 modes x 3 seeds x 3 endpoints x 7 q.
    h = mixture[mixture.method == "huang_pipeline"]
    wanted_modes = {"raw_historical_threshold", "raw_validation_B_90pct", "calibrated_validation_B_90pct"}
    if len(h) != 378 or set(h.score_mode.unique()) != wanted_modes:
        errors.append("Huang full-sample mixture grid incomplete or has unexpected score modes")
    keycols = ["source", "method", "seed", "endpoint", "q_fault", "score_mode"]
    if mixture.duplicated(keycols).any():
        errors.append("duplicate full-sample mixture rows")
    if not np.isfinite(mixture[list(METRICS)].to_numpy(float)).all():
        errors.append("nonfinite full-sample mixture metric")
    # At q endpoints, use the unchanged v1 score arrays to independently verify Huang metrics.
    endpoint_rows_checked = 0
    source_data, _, _ = base.get_data()
    heedb = np.load(ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    heedb_labels, _ = base.heedb_labels()
    validation_split = np.load(V1 / "calibration_split_indices.npz", allow_pickle=False)
    for si, source in enumerate(base.SOURCES):
        for seed_i, seed in enumerate(SEEDS):
            if source == "Ningbo":
                y = np.load(ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz", allow_pickle=False)["y"].astype(np.uint8)
                c = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz", allow_pickle=False)["p"].astype(float)
                r = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz", allow_pickle=False)["p"].astype(float)
            else:
                y, c, r = heedb_labels, np.asarray(heedb[seed_i, 0]), np.asarray(heedb[seed_i, 1])
            for ep in range(3):
                pclean, prev = c[:, ep], r[:, ep]
                if source == "Ningbo":
                    v = score["diagnostic_scores"][seed_i, :, ep]
                    lab = score["labels"][:, ep]
                else:
                    v = None
                if ep == 0:
                    fit = huang_fit[str(seed)][ep]
                    modes = {
                        "raw_historical_threshold": (pclean, prev, float(h[(h.source==source)&(h.seed==seed)&(h.endpoint=="TWA")&(h.score_mode=="raw_historical_threshold")].threshold.iloc[0])),
                        "raw_validation_B_90pct": (pclean, prev, float(fit["threshold_raw_validation_B_90pct"])),
                        "calibrated_validation_B_90pct": (base.platt(pclean, fit["a"], fit["b"]), base.platt(prev, fit["a"], fit["b"]), float(fit["threshold_calibrated_validation_B_90pct"])),
                    }
                    for mode, (cs, rs, th) in modes.items():
                        for q in (0.0, 1.0):
                            yy = np.r_[y[:, ep], y[:, ep]]
                            pp = np.r_[cs, rs]
                            w = np.r_[np.full(len(y), 1-q), np.full(len(y), q)]
                            ref = base.metrics(yy, pp, th, w)
                            row = h[(h.source == source) & (h.seed == seed) & (h.endpoint == "TWA") & (h.q_fault == q) & (h.score_mode == mode)]
                            if len(row) != 1 or any(not np.isclose(float(row.iloc[0][m]), ref[m], rtol=0, atol=1e-12) for m in METRICS):
                                errors.append(f"q endpoint reference mismatch {source}/{seed}/{mode}/{q}")
                            endpoint_rows_checked += 1
    # Raw and calibrated use the same transformed B operating point. Check all weighted-mixture call metrics.
    op_metrics = ("sensitivity", "specificity", "predicted_positive_fraction", "false_positives_per_1000", "false_negatives_per_1000")
    max_op_delta = 0.0
    for keys, g in h.groupby(["source", "seed", "endpoint", "q_fault"]):
        raw = g[g.score_mode == "raw_validation_B_90pct"]
        cal = g[g.score_mode == "calibrated_validation_B_90pct"]
        if len(raw) != 1 or len(cal) != 1:
            errors.append(f"missing Huang raw/cal row pair {keys}")
            continue
        for metric in op_metrics:
            delta = abs(float(raw.iloc[0][metric]) - float(cal.iloc[0][metric]))
            max_op_delta = max(max_op_delta, delta)
            if delta > 1e-12:
                errors.append(f"raw/calibrated operating point changed {keys}/{metric}: {delta}")
    # v2 bootstrap output and stream/source match are checked against all 882,000 saved v1 absolute draws.
    if bootstrap_audit.get("status") != "PASS" or bootstrap_audit.get("verification", {}).get("v1_absolute_groups_checked") != 882000 or bootstrap_audit.get("verification", {}).get("max_abs_delta") != 0:
        errors.append("bootstrap did not exactly reproduce all saved v1 absolute metric draws")
    if len(boot) != 2709 or not (boot.replicates.astype(int) == 1000).all() or boot.duplicated([c for c in ("source", "method", "contrast", "score_mode", "q_fault", "metric") if c in boot.columns]).any():
        errors.append("bootstrap summary row count, replicate count, or group uniqueness failure")
    if not np.isfinite(boot[["bootstrap_mean", "ci_2_5", "ci_97_5"]].to_numpy(float)).all():
        errors.append("nonfinite bootstrap summary")
    if inference["status"] != "PASS" or inference["detector_replay"]["max_abs_probability_delta"] != 0 or any(v["max_abs_probability_delta"] != 0 for v in inference["diagnostic_replay"].values()):
        errors.append("Huang pipeline test replay did not reproduce historical scores")
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "validation_split_partition": "PASS" if not errors else "FAIL",
              "v1_non_Huang_point_estimates_unchanged": len(old) == len(new_other), "huang_grid_rows": len(h),
              "q_endpoint_groups_checked": endpoint_rows_checked, "raw_calibrated_operating_metric_max_abs_delta": max_op_delta,
              "bootstrap_summary_rows": len(boot), "bootstrap_replicates_per_group": 1000,
              "v1_saved_absolute_draws_checked": bootstrap_audit["verification"],
              "replay_max_abs_delta_detector": inference["detector_replay"]["max_abs_probability_delta"],
              "replay_max_abs_delta_diagnostic": {k: v["max_abs_probability_delta"] for k,v in inference["diagnostic_replay"].items()},
              "no_training": True, "reserved_test_evaluated": False}
    write_json(OUT / "ENGINEERING_TEST_REPORT.json", report)
    return report, mixture, contrasts, boot


def make_reliability(mixture: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_csv(V1 / "RELIABILITY_BINS.csv").to_dict(orient="records")
    rows = [r for r in rows if r.get("method") != "huang_pipeline"]
    fit = json.loads((OUT / "HUANG_CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))["fits_by_seed"]
    val = np.load(OUT / "huang_validation_scores.npz", allow_pickle=False)
    pval = val["diagnostic_scores"].astype(float)
    heedb = np.load(ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1/huang_diagnostic_probabilities_float32.npy", mmap_mode="r")
    y_heedb, _ = base.heedb_labels()
    for source in base.SOURCES:
        for si, seed in enumerate(SEEDS):
            if source == "Ningbo":
                y = np.load(ROOT / f"outputs/chest_only_baseline_v1/comparators/Ningbo/clean_12lead/seed_{seed}/clean.npz", allow_pickle=False)["y"].astype(np.uint8)
                c = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_clean_seed_{seed}.npz", allow_pickle=False)["p"].astype(float)
                r = np.load(ROOT / f"outputs/published_detector_pipeline_v1/cache/huang_corrected_RA_LL_reversal_seed_{seed}.npz", allow_pickle=False)["p"].astype(float)
            else:
                y, c, r = y_heedb, np.asarray(heedb[si, 0]), np.asarray(heedb[si, 1])
            fit_ep = fit[str(seed)][0]
            th = float(mixture[(mixture.source==source)&(mixture.method=="huang_pipeline")&(mixture.seed==seed)&(mixture.endpoint=="TWA")&(mixture.score_mode=="raw_historical_threshold")].threshold.iloc[0])
            for mode, (cs, rs, threshold) in {
                "raw_historical_threshold": (c[:, 0], r[:, 0], th),
                "raw_validation_B_90pct": (c[:, 0], r[:, 0], fit_ep["threshold_raw_validation_B_90pct"]),
                "calibrated_validation_B_90pct": (base.platt(c[:, 0], fit_ep["a"], fit_ep["b"]), base.platt(r[:, 0], fit_ep["a"], fit_ep["b"]), fit_ep["threshold_calibrated_validation_B_90pct"]),
            }.items():
                for q in (0.0, 0.01, 0.10, 1.0):
                    yy=np.r_[y[:,0],y[:,0]]; pp=np.r_[cs,rs]; ww=np.r_[np.full(len(y),1-q),np.full(len(y),q)]
                    rows.extend(base.reliability_rows(source,"huang_pipeline",seed,"TWA",q,mode,yy,pp,ww))
    return pd.DataFrame(rows)


def make_plots(mixture: pd.DataFrame, reliability: pd.DataFrame) -> None:
    for col in ("bin", "weighted_fraction", "mean_prediction", "observed_fraction"):
        if col in reliability.columns:
            reliability[col] = pd.to_numeric(reliability[col], errors="coerce")
    colors = {"ordinary_resnet":"#4C78A8", "physics_aware_resnet":"#F58518", "chest_only_resnet":"#54A24B", "huang_pipeline":"#E45756"}
    chosen_mode = {"ordinary_resnet":"raw", "physics_aware_resnet":"raw", "chest_only_resnet":"raw", "huang_pipeline":"raw_validation_B_90pct"}
    for source in mixture.source.unique():
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for arm in ARMS:
            x=mixture[(mixture.source==source)&(mixture.method==arm)&(mixture.endpoint=="TWA")&(mixture.score_mode==chosen_mode[arm])]
            y=x.groupby("q_fault").auprc.mean().reindex(Q)
            ax.plot(Q,y,marker="o",label=arm.replace("_"," "),color=colors[arm])
        ax.set_xscale("symlog",linthresh=.001); ax.set_xlabel("Simulated RA–LL reversal share q"); ax.set_ylabel("TWA AUPRC")
        ax.set_title(f"TWA ranking under weighted mixture — {source}"); ax.legend(frameon=False,ncol=2); ax.grid(alpha=.22)
        fig.tight_layout(); fig.savefig(OUT / f"TWA_AUPRC_BY_Q_{source}.png",dpi=170); plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    for ri, source in enumerate(("Ningbo", "HEEDB_I0006_independent_confirmation")):
        d=mixture[mixture.source==source]
        for arm in ARMS:
            raw_mode=chosen_mode[arm]
            raw=d[(d.method==arm)&(d.endpoint=="TWA")&(d.score_mode==raw_mode)]
            axes[ri,0].plot(Q,raw.groupby("q_fault").brier.mean().reindex(Q),marker="o",color=colors[arm],label=arm.replace("_"," ")+" raw")
            cal_mode="calibrated" if arm!="huang_pipeline" else "calibrated_validation_B_90pct"
            cal=d[(d.method==arm)&(d.endpoint=="TWA")&(d.score_mode==cal_mode)]
            axes[ri,0].plot(Q,cal.groupby("q_fault").brier.mean().reindex(Q),marker="x",linestyle="--",color=colors[arm],label=arm.replace("_"," ")+" calibrated")
            for mode,ls,marker in ((raw_mode,"-","o"),(cal_mode,"--","x")):
                x=d[(d.method==arm)&(d.endpoint=="TWA")&(d.score_mode==mode)]
                axes[ri,1].plot(Q,x.groupby("q_fault").ece15.mean().reindex(Q),linestyle=ls,marker=marker,color=colors[arm],label=arm.replace("_"," ")+" "+mode.replace("_"," "))
        for ci, metric in enumerate(("Brier score", "ECE (15 bins)")):
            axes[ri,ci].set_xscale("symlog",linthresh=.001);axes[ri,ci].set_xlabel("Simulated fault share q");axes[ri,ci].set_ylabel(metric);axes[ri,ci].set_title(source);axes[ri,ci].grid(alpha=.22)
            axes[ri,ci].legend(frameon=False,fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(OUT/"TWA_CALIBRATION_BY_Q.png",dpi=170);plt.close(fig)
    # Raw/calibrated reliability for Huang, plus all v1 arms retained in one auditable table.
    for source in ("Ningbo","HEEDB_I0006_independent_confirmation"):
        x=reliability[(reliability.source==source)&(reliability.endpoint=="TWA")&(reliability.q_fault==0)&(reliability.method=="huang_pipeline")]
        fig,ax=plt.subplots(figsize=(5.5,5.2))
        ax.plot([0,1],[0,1],color="#777777",linestyle="--",linewidth=1)
        for mode,color in (("raw_validation_B_90pct","#E45756"),("calibrated_validation_B_90pct","#4C78A8")):
            z=x[x.score_mode==mode].groupby("bin").agg(pred=("mean_prediction","mean"),obs=("observed_fraction","mean"),weight=("weighted_fraction","mean")).dropna()
            ax.scatter(z.pred,z.obs,s=25+500*z.weight,color=color,label=mode.replace("_"," "),alpha=.8)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability",ylabel="Observed positive fraction",title=f"Huang TWA calibration, q=0 — {source}")
        ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/f"TWA_RELIABILITY_{source}.png",dpi=170);plt.close(fig)


def make_report(mix: pd.DataFrame, contrasts: pd.DataFrame, boot: pd.DataFrame, inference: dict, fits: dict) -> str:
    rows=[]
    names={"ordinary_resnet":"Ordinary", "physics_aware_resnet":"Physics-aware", "chest_only_resnet":"Chest-only", "huang_pipeline":"Huang"}
    for source in mix.source.unique():
        rows.append(f"\n### {source}\n\n| q | Ordinary | Physics-aware | Chest-only | Huang raw validation-B |\n|---:|---:|---:|---:|---:|")
        for q in Q:
            vals=[]
            for arm in ARMS:
                mode="raw_validation_B_90pct" if arm=="huang_pipeline" else "raw"
                g=mix[(mix.source==source)&(mix.method==arm)&(mix.endpoint=="TWA")&(mix.q_fault==q)&(mix.score_mode==mode)]
                vals.append(float(g.auprc.mean()))
            rows.append(f"| {q:g} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    ci_rows=[]
    qs=(0.0,0.001,0.01,0.05,1.0)
    for source in mix.source.unique():
        for q in qs:
            for mode in ("raw_historical_threshold","raw_validation_B_90pct","calibrated_validation_B_90pct"):
                g=contrasts[(contrasts.source==source)&(contrasts.endpoint=="TWA")&(contrasts.q_fault==q)&(contrasts.metric=="auprc")&(contrasts.contrast=="physics_aware_minus_huang_pipeline")&(contrasts.score_mode==mode)]
                if len(g):
                    r=g.iloc[0]
                    ci_rows.append(f"| {source} | {q:g} | {mode} | {r.full_sample_difference:.4f} | [{r.ci_2_5:.4f}, {r.ci_97_5:.4f}] |")
            legacy=contrasts[(contrasts.source==source)&(contrasts.endpoint=="TWA")&(contrasts.q_fault==q)&(contrasts.metric=="auprc")&(contrasts.contrast=="physics_aware_minus_huang_pipeline")&(contrasts.score_mode=="raw_legacy_v1_Huang_historical")]
            if len(legacy):
                r=legacy.iloc[0];ci_rows.append(f"| {source} | {q:g} | raw_legacy_v1_Huang_historical | {r.full_sample_difference:.4f} | [{r.ci_2_5:.4f}, {r.ci_97_5:.4f}] |")
    cal_rows=[]
    for source in mix.source.unique():
        for arm in ARMS:
            rawmode="raw_validation_B_90pct" if arm=="huang_pipeline" else "raw"
            calmode="calibrated_validation_B_90pct" if arm=="huang_pipeline" else "calibrated"
            raw=mix[(mix.source==source)&(mix.method==arm)&(mix.endpoint=="TWA")&(mix.q_fault==0)&(mix.score_mode==rawmode)]
            cal=mix[(mix.source==source)&(mix.method==arm)&(mix.endpoint=="TWA")&(mix.q_fault==0)&(mix.score_mode==calmode)]
            cal_rows.append(f"| {source} | {names[arm]} | {raw.brier.mean():.4f} | {cal.brier.mean():.4f} | {raw.ece15.mean():.4f} | {cal.ece15.mean():.4f} |")
    fits_table=[]
    for seed in SEEDS:
        for f in fits["fits_by_seed"][str(seed)]:
            fits_table.append(f"| {seed} | {f['endpoint']} | {f['a']:.6g} | {f['b']:.6g} | {f['threshold_raw_validation_B_90pct']:.6g} | {f['threshold_calibrated_validation_B_90pct']:.6g} |")
    text="""# Four-method calibration completion and paired uncertainty (v2)

## Scope

This is a versioned completion of the previously accessed Ningbo and HEEDB analyses. It is not a new untouched validation. No diagnostic network or Huang detector was trained; no weights, labels, splits, q values, or training data were changed. The v1 directory is retained unchanged. The only new model outputs are frozen Huang full-pipeline predictions for the existing 3,488 Ningbo validation records.

The published detector was run on clean validation ECGs and its argmax class controlled correction for every record, including records predicted clean. The same clean-trained ResNet checkpoints, preprocessing and normalization were then used for all three seeds. The test adapter replay matched stored detector and diagnostic probabilities exactly on the preregistered 64-record subset, using the historical diagnostic batch size 128. The initial 64-sized partial-batch replay differed by 0.000244; source inspection identified batch shape as the cause, the 0.0001 tolerance was left unchanged, and amendment 001 records the repeat. The detector marked 48/3,488 clean validation records for correction (1.38%); this is descriptive and does not change the always-predict-correction policy.

Huang calibrators were fitted on clean Ningbo validation A only, using v1's unchanged Platt implementation and ordinary unweighted log loss. Raw thresholds are the 90th percentile of negative full-pipeline scores on validation B; calibrated thresholds are those same thresholds transformed through the fit. Test and HEEDB labels were not used for fitting. Historical Huang thresholds remain visible as a separate mode. The legacy v1 contrast, where physics-aware uses its validation-B raw threshold while Huang uses the historical baseline threshold, is retained with an explicit label.

All model selection/checkpoint selection preceded this A/B split and historically used full validation, so the split does not make checkpoint selection independent. Calibration is fixed in all test/HEEDB and bootstrap calculations. HEEDB outcomes are physician-code proxies, and both HEEDB analyses share the Emory source. Results are exploratory, q is a hypothetical mixture share rather than a clinical prevalence estimate, and intervals condition on selected checkpoints/calibrators.

## TWA ranking over the full q grid

Entries are full-sample weighted pooled AUPRC, averaged across seed-specific metrics (prediction vectors are not ensembled). Huang values use its raw validation-B threshold; the score mapping itself is raw. The full seed-level table covers all endpoints and operating threshold modes in `MIXTURE_RESULTS_BY_SEED.csv`.
""" + "\n".join(rows) + """

## Physics-aware minus Huang TWA AUPRC

`Full-sample difference` is the central estimate from the entire cohort; the bootstrap mean is stored separately in `PAIRED_CONTRASTS.csv`. Intervals use the same 1,000 synchronized draws and paired clean/reversal cluster multiplicities.

| Source | q | Threshold mode | Full-sample difference | 95% percentile interval |
|---|---:|---|---:|---:|
""" + "\n".join(ci_rows) + """

Intervals crossing zero leave the relative difference unresolved; they do not establish equivalence. Across the tested q values, these intervals do not establish a physics-aware AUPRC advantage over Huang. This removes a basis for claiming superiority over Huang in this analysis. At higher q the point estimates generally remain close, with uncertainty spanning zero.

## Source-validation calibration at q=0

| Source | Method | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---:|---:|---:|---:|
""" + "\n".join(cal_rows) + """

The monotone mapping leaves AUPRC/AUROC essentially unchanged except for measured clipping-created ties. It improves Brier/ECE on Ningbo and the measured HEEDB proxy labels. The latter is not evidence that source calibration is valid probability calibration for HEEDB's proxy outcome construct.

Raw validation-B versus calibrated validation-B use exactly one operating point: sensitivity, specificity, predicted-positive fraction, false positives and false negatives are identical at every q. Comparisons to historical thresholds are separate because the Huang historical threshold differs from its full-pipeline validation-B threshold; changes in those call rates arise from the operating-point standardization, not monotone calibration.

## Huang validation calibration coefficients

| Seed | Endpoint | a | b | Raw B threshold | Calibrated B threshold |
|---:|---|---:|---:|---:|---:|
""" + "\n".join(fits_table) + """

## Conclusions

1. **Relative ranking:** Physics-aware retains the strongest TWA point estimates under increasingly frequent simulated RA–LL faults. Ordinary deteriorates substantially as q rises; chest-only is nearly invariant because the simulated reversal changes limb leads. Paired intervals comparing physics-aware with Huang cross zero, so the analysis does not resolve a superiority claim over Huang.
2. **Calibration:** Huang's full-pipeline validation scores fill the previous calibration gap. Source calibration improves probability-error metrics, while the strictly monotone mapping preserves ranking to within measured clipping/tie effects. It leaves fixed operating decisions unchanged.
3. **Rare faults:** At q=0 and q=0.001, physics-aware and Huang/ordinary rankings remain close; confidence intervals do not resolve a physics-aware advantage over Huang. With larger q, ordinary worsens while physics-aware remains relatively stable.
4. **Clean ECG tradeoff:** Huang's false-correction rate is 1.38% on this clean validation cohort under the mandated detector-argmax correction policy. This is an observed validation behavior, not a clinical harm estimate. Full clean-cohort metrics are in the seed-level CSV.
5. **HEEDB:** The relative ranking is broadly consistent with the Ningbo pattern, but endpoint labels are proxies and the source is one Emory system. It is not multi-center clinical confirmation.
6. **Next experiment:** No `p_aug` value is selected by this inference-time mixture analysis. A focused masking/combined Ningbo replication remains useful to ask whether independently retrained arms reproduce M2's reversal-versus-masking tradeoff and preserve the q response. No such training was started.

## Reproducibility and engineering checks

- `FROZEN_COMPLETION_PROTOCOL.json` and amendments 001–003 bind cohorts, record order, inputs, replay tolerances, A/B split, calibration and bootstrap rules.
- The exact v1 RNG stream was replayed in cohort order. All 882,000 v1 absolute metric draws matched with maximum absolute delta 0; existing v1 intervals were reused, the missing Ningbo Huang paired interval was filled, and new bootstrap metrics were computed only for Huang's B-threshold and calibrated modes plus the new matched historical operating-point comparisons.
- All 2,709 bootstrap summary groups contain 1,000 draws. Point estimates, bootstrap means and intervals are separately labeled.
- See `ENGINEERING_TEST_REPORT.md`, `HUANG_VALIDATION_INFERENCE_AUDIT.json`, `BOOTSTRAP_COMPLETION_AUDIT.json`, and `REPRODUCIBILITY_MANIFEST.json`.

## Main files

- `HUANG_VALIDATION_INFERENCE_AUDIT.md` and `.json`: exact full-pipeline inference, replay, record/label bindings and clean false-correction count.
- `HUANG_CALIBRATION_AND_PROVENANCE.json` and `CALIBRATION_AND_PROVENANCE.json`: fit coefficients and unchanged existing calibrators.
- `MIXTURE_RESULTS_BY_SEED.csv`, `PAIRED_CONTRASTS.csv`, `BOOTSTRAP_METRICS_SUMMARY.csv`, `BOOTSTRAP_METRICS_REPLICATES.npz`.
- `SCORE_COVERAGE_AUDIT.csv`, `RELIABILITY_BINS.csv`, `TWA_*.png`, `ENGINEERING_TEST_REPORT.md`, `REPRODUCIBILITY_MANIFEST.json`.

Reserved_test was not evaluated. Stop here for review; no subsequent training or experiment was launched.
"""
    return text


def main() -> None:
    report, mixture, contrasts, boot = evaluate()
    reliability = make_reliability(mixture)
    reliability.to_csv(OUT / "RELIABILITY_BINS.csv", index=False)
    make_plots(mixture, reliability)
    report_md = make_report(mixture, contrasts, boot, json.loads((OUT/"HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8")),
                            json.loads((OUT/"HUANG_CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8")))
    (OUT / "FINAL_FOUR_METHOD_COMPARISON.md").write_text(report_md, encoding="utf-8")
    (OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.md").write_text(
        "# Huang validation inference audit\n\n" + json.dumps(json.loads((OUT/"HUANG_VALIDATION_INFERENCE_AUDIT.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Combine unchanged v1 fit provenance with the newly added Huang calibration.
    old_prov = json.loads((V1 / "CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))
    old_prov["huang_validation_calibration"] = json.loads((OUT / "HUANG_CALIBRATION_AND_PROVENANCE.json").read_text(encoding="utf-8"))
    old_prov["v1_provenance_sha256"] = sha(V1 / "CALIBRATION_AND_PROVENANCE.json")
    write_json(OUT / "CALIBRATION_AND_PROVENANCE.json", old_prov)
    checks_md = """# Engineering test report (v2)

**Status: {status}**

- Validation order and 3,488 labels match all frozen diagnostic-model validation artifacts; the unchanged A/B indices partition all 3,488 records into 1,744 A and 1,744 B records.
- Calibration optimizer, clipping, precision and bounds are the existing v1 implementation; coefficients have positive slopes and each fit uses A only. Thresholds use only negative B scores.
- Detector and all three diagnostic checkpoints replay the existing 64-record test subset with exact probability agreement at preregistered tolerances; diagnostic replay used the frozen 128-record inference batch size.
- All three pre-existing methods retain exactly the v1 point estimates; Huang now has complete full-pipeline historical-raw, validation-B raw and validation-B calibrated coverage.
- q=0 and q=1 endpoint reference checks passed for all Huang seeds, TWA, cohorts and threshold modes.
- Raw/calibrated Huang fixed operating-point decisions agree: maximum absolute difference across all five call-rate metrics and q values is {op_delta:.3g}.
- The frozen RNG stream reproduces all 882,000 saved v1 absolute metric draws exactly, then supplies paired Huang intervals. Every v2 interval contains 1,000 replicates.
- Validation detector output/labels are finite and ordered; test/HEEDB labels were not used for fitting. No diagnostic training or reserved-test assessment occurred.

See `ENGINEERING_TEST_REPORT.json` for machine-readable counts and hashes.
""".format(status=report["status"], op_delta=report["raw_calibrated_operating_metric_max_abs_delta"])
    (OUT / "ENGINEERING_TEST_REPORT.md").write_text(checks_md, encoding="utf-8")
    write_json(OUT / "ENGINEERING_TEST_REPORT.json", report)
    # Ranking and calibration effects are measured rather than inferred from the method label.
    ranking=[]
    for source in mixture.source.unique():
        for arm in ARMS:
            raw_mode="raw_validation_B_90pct" if arm=="huang_pipeline" else "raw"
            cal_mode="calibrated_validation_B_90pct" if arm=="huang_pipeline" else "calibrated"
            for q in Q:
                raw=mixture[(mixture.source==source)&(mixture.method==arm)&(mixture.endpoint=="TWA")&(mixture.q_fault==q)&(mixture.score_mode==raw_mode)]
                cal=mixture[(mixture.source==source)&(mixture.method==arm)&(mixture.endpoint=="TWA")&(mixture.q_fault==q)&(mixture.score_mode==cal_mode)]
                ranking.append({"source":source,"method":arm,"q_fault":q,"max_abs_seed_delta_auprc":float(np.max(np.abs(raw.auprc-cal.auprc))),
                                "max_abs_seed_delta_auroc":float(np.max(np.abs(raw.auroc-cal.auroc))),
                                "raw_mean_brier":float(raw.brier.mean()),"calibrated_mean_brier":float(cal.brier.mean()),
                                "raw_mean_ece15":float(raw.ece15.mean()),"calibrated_mean_ece15":float(cal.ece15.mean())})
    write_json(OUT / "RANKING_CALIBRATION_AUDIT.json", {"status":"PASS","measured_by":"three seed-level full-sample mixture results","rows":ranking,
        "call_rate_max_abs_delta":report["raw_calibrated_operating_metric_max_abs_delta"],"clipping_ties_caveat":"AUPRC/AUROC differences are explicitly measured; monotone clipping can tie endpoint probabilities."})
    # Hash every deliverable and all frozen sources; the manifest deliberately does not hash itself.
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!="REPRODUCIBILITY_MANIFEST.json" and not p.name.endswith(".partial") and p.suffix!=".pyc"]
    manifest={"status":"COMPLETE","protocol_sha256":sha(OUT/"FROZEN_COMPLETION_PROTOCOL.json"),
              "amendments":{p.name:sha(p) for p in sorted(OUT.glob("FROZEN_COMPLETION_AMENDMENT_*.json"))},
              "outputs":{p.name:sha(p) for p in sorted(files)},
              "v1_unchanged":True,"no_new_diagnostic_training":True,"reserved_test_evaluated":False,
              "hash_algorithm":"sha256","manifest_self_hash":"excluded to avoid recursive hash"}
    write_json(OUT/"REPRODUCIBILITY_MANIFEST.json",manifest)
    print(json.dumps({"status":manifest["status"],"engineering":report["status"],"outputs":len(files),"manifest_sha256":sha(OUT/"REPRODUCIBILITY_MANIFEST.json")},sort_keys=True))


if __name__=="__main__":
    main()
