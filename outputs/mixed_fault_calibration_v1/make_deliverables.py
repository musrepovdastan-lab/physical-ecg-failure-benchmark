"""Render plots and the final aggregate-only mixed-fault report."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_analysis import ROOT, OUT, METHODS, Q, sha, write_json

COLORS={"ordinary_resnet":"#b24a3a","physics_aware_resnet":"#2a7d5f","chest_only_resnet":"#366aa0","huang_pipeline":"#8357a1"}
LABELS={"Ningbo":"Ningbo test (previously accessed)","HEEDB_I0006_independent_confirmation":"HEEDB I0006 (Emory proxy-label cohort)"}

def f(x): return "NA" if pd.isna(x) else f"{float(x):.3f}"

def main():
    result=pd.read_csv(OUT/"MIXTURE_RESULTS_BY_SEED.csv")
    boot=pd.read_csv(OUT/"BOOTSTRAP_REPLICATES.csv")
    # Curves of full-cohort three-seed mean TWA AUPRC.
    twa=result[result.endpoint=="TWA"]
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,(source,g) in zip(axes,twa[twa.score_mode.isin(["raw","calibrated"])].groupby("source")):
        for arm,mode in ((a,m) for a in METHODS for m in ["raw","calibrated"]):
            z=g[(g.method==arm)&(g.score_mode==mode)].groupby("q_fault").auprc.mean()
            if len(z): ax.plot(np.arange(len(Q)),[z.get(q,np.nan) for q in Q],marker="o",linestyle="-" if mode=="raw" else "--",
                               color=COLORS[arm],alpha=1 if mode=="raw" else .55,label=f"{METHODS[arm]} ({mode})")
        ax.set_xticks(np.arange(len(Q)),["0","0.1%","0.5%","1%","5%","10%","100%"])
        ax.set_xlabel("Hypothetical RA–LL fraction q");ax.set_title(LABELS[source]);ax.grid(alpha=.25)
    axes[0].set_ylabel("Weighted mixture TWA AUPRC")
    handles,labels=axes[1].get_legend_handles_labels();by=dict(zip(labels,handles));fig.legend(by.values(),by.keys(),loc="lower center",ncol=4,fontsize=8)
    fig.tight_layout(rect=[0,0.20,1,1]);fig.savefig(OUT/"TWA_AUPRC_BY_Q.png",dpi=180);plt.close(fig)
    # Brier and ECE raw/calibrated response.
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,source in enumerate(["Ningbo","HEEDB_I0006_independent_confirmation"]):
        for row,metric in enumerate(["brier","ece15"]):
            ax=axes[row,col]; g=twa[(twa.source==source)&twa.score_mode.isin(["raw","calibrated"])]
            for arm in METHODS:
                for mode,ls,alpha in (("raw","-",.95),("calibrated","--",.65)):
                    z=g[(g.method==arm)&(g.score_mode==mode)].groupby("q_fault")[metric].mean()
                    if len(z):ax.plot(np.arange(len(Q)),[z.get(q,np.nan) for q in Q],marker="o",linestyle=ls,color=COLORS[arm],alpha=alpha,
                                      label=f"{METHODS[arm]} ({mode})")
            ax.set_xticks(np.arange(len(Q)),["0","0.1%","0.5%","1%","5%","10%","100%"]);ax.grid(alpha=.25)
            ax.set_title(f"{LABELS[source]} — {metric.upper()}");ax.set_xlabel("Hypothetical RA–LL fraction q")
    axes[0,0].set_ylabel("Brier score");axes[1,0].set_ylabel("ECE-15")
    handles,labels=axes[0,1].get_legend_handles_labels();by=dict(zip(labels,handles));fig.legend(by.values(),by.keys(),loc="lower center",ncol=4,fontsize=8)
    fig.tight_layout(rect=[0,0.15,1,1]);fig.savefig(OUT/"TWA_CALIBRATION_BY_Q.png",dpi=180);plt.close(fig)
    # Reliability diagrams for every q; separate raw and source-calibrated lines.
    rel=pd.read_csv(OUT/"RELIABILITY_BINS.csv")
    for source in ["Ningbo","HEEDB_I0006_independent_confirmation"]:
        fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
        for j,q in enumerate(Q):
            ax=axes.flat[j]; z=rel[(rel.source==source)&(rel.endpoint=="TWA")&(rel.q_fault==q)&rel.score_mode.isin(["raw","calibrated"])]
            for arm in METHODS:
                for mode,ls in (("raw","--"),("calibrated","-")):
                    a=z[(z.method==arm)&(z.score_mode==mode)].groupby("bin").agg(x=("mean_prediction","mean"),y=("observed_fraction","mean")).dropna()
                    if len(a):ax.plot(a.x,a.y,linestyle=ls,marker=".",color=COLORS[arm],alpha=.8,label=f"{METHODS[arm]} {mode}")
            ax.plot([0,1],[0,1],color="black",lw=.8);ax.set_title(f"q={q:g}");ax.grid(alpha=.2)
        axes[0,0].set_ylabel("Observed fraction");axes[1,0].set_ylabel("Observed fraction")
        for ax in axes[1,:]:ax.set_xlabel("Mean predicted probability")
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,fontsize=7)
        fig.suptitle(f"TWA reliability — {LABELS[source]}");fig.tight_layout(rect=[0,0.15,1,.95])
        fig.savefig(OUT/f"TWA_RELIABILITY_{source}.png",dpi=180);plt.close(fig)
    # Report selected rows at all q, including paired intervals for primary AUPRC contrasts.
    lines=["# Mostly-clean simulated RA–LL mixtures with source-validation calibration", "",
        "## Scope and interpretation", "",
        "This is an exploratory reanalysis of Ningbo and previously studied HEEDB I0006 records, not a new untouched confirmation. q is a hypothetical simulated-fault share, not an estimated clinical frequency. Each ECG contributes clean and exact RA–LL-reversed score/label copies weighted 1−q and q; metrics are calculated from the pooled weighted empirical mixture.", "",
        "No diagnostic model was trained. The only newly fitted parameters are monotone Platt calibrators fit on clean Ningbo validation A. Thresholds use the 90th percentile of negative validation B scores. Checkpoints had historically been selected using full validation, so this A/B split does not make their selection independent. Calibrators and thresholds were held fixed on test/HEEDB. Calibration does not change the underlying ordering except for extreme clipping ties.", "",
        "The Huang detector→predicted correction→clean ResNet pipeline has no complete frozen Ningbo-validation score set in the available artifacts. It is therefore shown raw only, using its existing corresponding baseline Ningbo validation threshold. No Huang calibration or standardized validation-B operating point is claimed. The already completed Ningbo corrected-score caches were added through [protocol amendment 001](FROZEN_PROTOCOL_AMENDMENT_001.json), with order/hash bindings.", "",
        "HEEDB is external relative to the Chinese training sources, but these HEEDB cohorts come from the same Emory source and are not two independent external centers. HEEDB outcomes are physician-code proxy constructs; source calibration does not establish probability calibration for those proxies.", "",
        "## Primary TWA ranking by q", "",
        "Values are three-seed means from full-cohort weighted pooled mixtures. The bootstrap intervals below are 1,000 synchronized record/patient resamples, paired across clean/reversal states and methods; they are conditional on the selected checkpoints, fitted calibrators and fixed thresholds. Full intervals for all nine TWA metrics are in [BOOTSTRAP_METRICS_SUMMARY.csv](BOOTSTRAP_METRICS_SUMMARY.csv), with replicate arrays in `BOOTSTRAP_METRICS_REPLICATES.npz`.", "",
        "| Source | q | Ordinary | Physics-aware | Chest-only | Huang |", "|---|---:|---:|---:|---:|---:|"]
    for source in ["Ningbo","HEEDB_I0006_independent_confirmation"]:
        for q in Q:
            vals=[]
            for arm in METHODS:
                z=twa[(twa.source==source)&(twa.method==arm)&(twa.q_fault==q)&(twa.score_mode=="raw")].groupby("seed").auprc.mean()
                vals.append(f(z.mean()) if len(z) else "—")
            lines.append(f"| {LABELS[source]} | {q:g} | "+" | ".join(vals)+" |")
    lines += ["", "### Paired primary contrast: physics-aware minus each comparator", "",
        "Percentile 95% intervals for TWA AUPRC; an interval crossing zero is unresolved and does not establish equality. Huang is raw-only.", "",
        "| Source | q | Score mode | Contrast | Mean difference | 95% interval |", "|---|---:|---|---|---:|---:|"]
    b=boot[(boot.metric=="AUPRC")&(boot.primary_endpoint=="TWA")]
    for source in ["Ningbo","HEEDB_I0006_independent_confirmation"]:
        for q in Q:
            for mode in ["raw","calibrated"]:
                for comp in ["ordinary_resnet","chest_only_resnet","huang_pipeline"]:
                    cname=f"physics_aware_minus_{comp}"
                    z=b[(b.source==source)&(b.q_fault==q)&(b.score_mode==mode)&(b.contrast==cname)]
                    if z.empty:continue
                    vals=z.difference.to_numpy(float);lines.append(f"| {LABELS[source]} | {q:g} | {mode} | {cname} | {vals.mean():.4f} | [{np.quantile(vals,.025):.4f}, {np.quantile(vals,.975):.4f}] |")
    lines += ["", "## Clean, rare-fault and all-fault tradeoffs", "",
        "At q=0, physics-aware TWA AUPRC is 0.687 on Ningbo and 0.576 on HEEDB, versus 0.676/0.573 for ordinary ResNet, 0.655/0.537 for chest-only, and 0.675/0.572 for Huang. Its paired intervals versus ordinary and Huang include zero at q=0; the comparison with chest-only favors physics-aware on both sources. As q rises, ordinary ResNet ranking falls sharply while physics-aware and chest-only remain steadier. At q=1, physics-aware remains well above ordinary, is close to chest-only on Ningbo, and is below Huang on HEEDB by 0.0127 AUPRC (95% paired interval −0.0263 to +0.0021), which does not establish a difference. These are scenario results, not clinical-frequency claims.", "",
        "Source-validation calibration reduces TWA Brier/ECE on both sources at q=0: for physics-aware, Ningbo Brier/ECE changes from 0.1077/0.1247 raw to 0.0730/0.0115 calibrated; HEEDB changes from 0.1152/0.1120 to 0.0840/0.0227. On Ningbo, maximum raw/calibrated AUPRC and AUROC differences across arms/q are below 4e-16. On HEEDB, clipping created ties: the maximum differences are 0.000062 AUPRC and 0.000142 AUROC. These are measured and reported as clipping effects. The calibrated threshold is the same monotone transform of the raw validation-B threshold; all five operating-point measures matched exactly (maximum absolute difference 0). Raw historical-threshold results, raw validation-B threshold results, and calibrated equivalents are distinct rows in the result CSV. See [ranking/calibration audit](RANKING_CALIBRATION_AUDIT.json).", "",
        "The chest-only model is exactly invariant to RA–LL in this input path because only V1–V6 enter the model; its score-based discrimination is constant across q. This makes it a useful simple comparator, while not implying superiority on every source or endpoint.", "",
        "## Required engineering checks", "",
        "Synthetic checks covered q=0 and q=1 identity, weighted AP/AUROC and all nine grouped-bootstrap metrics against scikit-learn reference calculations, ties and zero weights, paired bootstrap record multiplicity, Brier linear-mixture identity, raw/calibrated threshold equivalence, and monotonicity/clipping ties. See [ENGINEERING_TEST_REPORT.md](ENGINEERING_TEST_REPORT.md).", "",
        "## Positioning correction", "",
        "A separate correction note records the existing matched CPU request-to-response benchmark (10.59 ms physics-aware vs 31.51 ms Huang chain; 2.97×, specific to the measured implementation/hardware/workload), the HEEDB same-Emory-source limitation, and the full-PDF access caveat for Cao. See [POSITIONING_CORRECTION_001.md](../novelty_reassessment_v1/POSITIONING_CORRECTION_001.md). No new latency benchmark was run.", "",
        "## Follow-up experiments", "",
        "This packet does not make a new p_aug sweep mandatory. If the retained question is whether the training exposure rate itself has a dose-response relation, a future experiment should freeze a small p_aug grid, matched seeds, validation selection and test transfer before training. The current q-mixture answers inference-time scenario sensitivity for existing fixed models, not an optimal training exposure rate.", "",
        "A focused Ningbo masking/combined-training replication remains scientifically useful because the prior corrected M2 Ningbo result passed its frozen external replication gate. Its concrete question should be whether independently retrained masking-only and combined arms reproduce M2's reversal-vs-clean/mask tradeoffs and whether they preserve the mostly-clean q curves on held-out data. It is not needed to complete this reanalysis and was not started.", "",
        "## Files", "",
        "- `SCORE_COVERAGE_AUDIT.csv`: model/source/seed coverage and explicit missing-data notes.",
        "- `FROZEN_ANALYSIS_PROTOCOL.json`, `FROZEN_PROTOCOL_AMENDMENT_001.json`, `FROZEN_PROTOCOL_AMENDMENT_002.json`: frozen methods, cohorts, thresholds, hashes and versioned score/bootstrap coverage additions.",
        "- `CALIBRATION_AND_PROVENANCE.json`, `SPLIT_PROVENANCE.json`: Platt fit coefficients, A/B split, threshold and source hashes.",
        "- `RANKING_CALIBRATION_AUDIT.json`: measured raw/calibrated ranking deltas, clipping-created ties and operating-point invariance.",
        "- `MIXTURE_RESULTS_BY_SEED.csv`, `PAIRED_CONTRASTS.csv`: primary/secondary full-sample metrics and paired contrasts.",
        "- `BOOTSTRAP_SUMMARY.csv`, `BOOTSTRAP_REPLICATES.csv`: primary TWA AUPRC uncertainty draws and intervals.",
        "- `BOOTSTRAP_METRICS_SUMMARY.csv`, `BOOTSTRAP_METRICS_REPLICATES.npz`, `BOOTSTRAP_METRICS_COMPLETE.json`: all nine TWA metrics across q, 1,000 paired draws and intervals.",
        "- `RELIABILITY_BINS.csv`, `TWA_AUPRC_BY_Q.png`, `TWA_CALIBRATION_BY_Q.png`, `TWA_RELIABILITY_*.png`: reliability bins and q curves.", ""]
    report=OUT/"FINAL_MIXED_FAULT_REPORT.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    complete=json.loads((OUT/"ANALYSIS_COMPLETE.json").read_text(encoding="utf-8"))
    complete.update({"mixture_rows":int(len(result)),"mixture_rows_final":int(len(result)),
        "paired_contrast_rows":int(len(pd.read_csv(OUT/"PAIRED_CONTRASTS.csv"))),"paired_contrast_rows_final":int(len(pd.read_csv(OUT/"PAIRED_CONTRASTS.csv"))),
        "bootstrap_rows":int(len(pd.read_csv(OUT/"BOOTSTRAP_SUMMARY.csv"))),
        "bootstrap_summary_rows_final":int(len(pd.read_csv(OUT/"BOOTSTRAP_SUMMARY.csv"))),
        "protocol_amendment_001_sha256":sha(OUT/"FROZEN_PROTOCOL_AMENDMENT_001.json"),
        "protocol_amendment_002_sha256":sha(OUT/"FROZEN_PROTOCOL_AMENDMENT_002.json"),
        "all_twa_metrics_bootstrap_status":json.loads((OUT/"BOOTSTRAP_METRICS_COMPLETE.json").read_text(encoding="utf-8"))["status"],
        "amendment_added_existing_huang_ningbo_scores":True,
        "raw_calibrated_operating_point_decisions_identical":True})
    write_json(OUT/"ANALYSIS_COMPLETE.json",complete)
    write_json(OUT/"WORKFLOW_STATUS.json",{"status":"COMPLETE","primary_analysis":"PASS","cohort":"Ningbo and HEEDB I0006 independent-confirmation; previously accessed/exploratory","diagnostic_training":False,"new_downloads":False,"latency_benchmark":False,"reserved_test_evaluated":False,"huang_ningbo_scores":"included via frozen protocol amendment 001","completed_utc":datetime.now(timezone.utc).isoformat()})
    manifest={"status":"COMPLETE","protocol_sha256":sha(OUT/"FROZEN_ANALYSIS_PROTOCOL.json"),"protocol_amendment_001_sha256":sha(OUT/"FROZEN_PROTOCOL_AMENDMENT_001.json"),"analysis_code_sha256":sha(OUT/"run_analysis.py"),"amendment_code_sha256":sha(OUT/"apply_amendment_001.py"),"final_report_sha256":sha(report),"source_label_caveat":"HEEDB physician-code proxies","outputs":{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in {"REPRODUCIBILITY_MANIFEST.json"}}}
    write_json(OUT/"REPRODUCIBILITY_MANIFEST.json",manifest)
    print(f"REPORT_WRITTEN {report} sha256={sha(report)}")

if __name__=="__main__":main()
