"""Generate aggregate-only, source-verified scientific figures for the manuscript."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "submission_evidence_v1"
SEEDS = (104729, 130363, 155921)
BASE = "#6b6b6b"
PHYSICS = "#0072B2"
HUANG = "#D55E00"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_pair(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    prior = pd.read_csv(source("outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv"))
    heedb = pd.read_csv(source("outputs/heedb_huang_exploratory_v1/TWA_THREE_SEED_MEANS.csv"))
    ci = pd.read_csv(source("outputs/heedb_huang_exploratory_v1/PAIRED_PATIENT_BOOTSTRAP.csv"))
    if len(prior) != 12 or len(heedb) != 6 or set(prior.seed) != set(SEEDS):
        raise RuntimeError("figure source cardinality changed")
    if set(heedb.condition) != {"clean", "RA_LL_reversal"} or set(heedb.arm) != {
        "baseline", "physics_aware", "huang_detect_correct_clean_resnet"
    }:
        raise RuntimeError("figure source condition/arm mapping changed")
    verified = json.loads(source("outputs/heedb_huang_exploratory_v1/ANALYSIS_COMPLETE.json").read_text(encoding="utf-8"))
    for name in ("TWA_THREE_SEED_MEANS.csv", "PAIRED_PATIENT_BOOTSTRAP.csv"):
        if sha(source("outputs/heedb_huang_exploratory_v1/" + name)) != verified["aggregate_hashes"][name]:
            raise RuntimeError("Huang figure source hash changed")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300,
                         "svg.fonttype": "none"})
    figure_data: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True, constrained_layout=True)
    for ax, dataset, n, prevalence in zip(axes, ("Chapman", "Ningbo"), (2050, 6981), (.183902439, .1481163157)):
        for arm, label, color, offset in (("cached_clean_baseline", "Clean-trained", BASE, -.045),
                                          ("physics_aware", "Physics-aware", PHYSICS, .045)):
            part = prior.loc[(prior.dataset == dataset) & (prior.arm == arm)].sort_values("seed")
            vals = part[["clean_twa_auprc", "mean_five_lead_mask_twa_auprc", "ra_ll_twa_auprc"]].to_numpy(dtype=float)
            x = np.arange(3) + offset
            for vector in vals:
                ax.plot(x, vector, color=color, alpha=.20, linewidth=.8, marker="o", markersize=2.4)
            ax.plot(x, vals.mean(axis=0), color=color, linewidth=2.1, marker="o", markersize=5, label=label)
            for row, vector in zip(part.itertuples(index=False), vals):
                for condition, value in zip(("clean", "mean_792_five_lead_masks", "RA_LL_reversal"), vector):
                    figure_data.append({"figure": "F1", "dataset": dataset, "arm": arm, "seed": int(row.seed),
                                        "condition": condition, "metric": "TWA_AUPRC", "value": float(value),
                                        "low95": "", "high95": ""})
        ax.axhline(prevalence, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(f"{dataset} (test n={n:,})")
        ax.set_xticks(range(3), ["Clean", "Mean of 792\nfive-lead masks", "Exact simulated\nRA-LL"])
        ax.set_ylim(0, .85)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0].set_ylabel("TWA AUPRC")
    axes[0].legend(frameon=False, loc="lower left")
    save_pair(fig, figdir / "figure1_cross_dataset_ranking")

    arms = (("baseline", "Unprotected", BASE, "o"),
            ("physics_aware", "Physics-aware", PHYSICS, "s"),
            ("huang_detect_correct_clean_resnet", "Huang-to-clean ResNet*", HUANG, "^"))
    metrics = (("auprc", "TWA AUPRC", (0, .68)), ("sensitivity", "Sensitivity", (0, .85)),
               ("specificity", "Specificity", (0, 1.0)), ("ece15", "ECE15 (lower is better)", (0, .60)))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.1), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes.flat, metrics):
        for arm, label, color, marker in arms:
            part = heedb.loc[heedb.arm == arm].set_index("condition")
            values = [float(part.loc[condition, metric]) for condition in ("clean", "RA_LL_reversal")]
            ax.plot((0, 1), values, marker=marker, markersize=5, color=color,
                    linestyle="--" if arm.startswith("huang") else "-", linewidth=1.7, label=label)
            for condition, value in zip(("clean", "RA_LL_reversal"), values):
                figure_data.append({"figure": "F2", "dataset": "HEEDB_independent", "arm": arm, "seed": "three_seed_mean",
                                    "condition": condition, "metric": metric, "value": value, "low95": "", "high95": ""})
        if metric == "auprc":
            ax.axhline(.1456, color="#999999", linestyle=":", linewidth=1, label="TWA prevalence")
        ax.set_title(title)
        ax.set_xticks((0, 1), ("Clean", "Exact simulated RA-LL"))
        ax.set_xlim(-.18, 1.18)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("HEEDB physician-code TWA proxy; Huang comparison post-hoc exploratory", fontsize=10)
    save_pair(fig, figdir / "figure2_heedb_tradeoffs")

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.3), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, (("auprc", "TWA AUPRC"), ("auroc", "TWA AUROC"),
                                                 ("sensitivity", "Sensitivity"), ("specificity", "Specificity"))):
        left, right = "physics_aware", "huang_detect_correct_clean_resnet"
        part = ci.loc[(ci.type == "paired_contrast") & (ci.left == left) & (ci.right == right) &
                      (ci.condition == "RA_LL_reversal") & (ci.metric == metric)]
        if len(part) != 1:
            raise RuntimeError("paired Huang figure row not unique")
        row = part.iloc[0]
        x, low, high = float(row.estimate), float(row.low95), float(row.high95)
        ax.errorbar(x, 0, xerr=[[x - low], [high - x]], color=PHYSICS,
                    fmt="o", markersize=6, capsize=4, linewidth=1.6)
        figure_data.append({"figure": "F3", "dataset": "HEEDB_independent", "arm": left + "_minus_" + right,
                            "seed": "three_seed_mean", "condition": "RA_LL_reversal", "metric": metric,
                            "value": x, "low95": low, "high95": high})
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1)
        margin = max(high - low, .008) * .35
        ax.set_xlim(min(low, 0) - margin, max(high, 0) + margin)
        ax.set_ylim(-.5, .5)
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("Physics-aware minus Huang (95% CI)")
        ax.grid(axis="x", color="#e0e0e0", linewidth=.6)
    fig.suptitle("HEEDB simulated RA-LL: post-hoc exploratory paired contrasts (1,000 bootstraps)", fontsize=10)
    save_pair(fig, figdir / "figure3_heedb_paired_contrasts")

    with (OUT / "FIGURE_DATA.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_data[0]))
        writer.writeheader()
        writer.writerows(figure_data)
    files = [OUT / "FIGURE_DATA.csv", *sorted(figdir.glob("*.svg")), *sorted(figdir.glob("*.png"))]
    manifest = {
        "status": "AGGREGATE_ONLY_FIGURES",
        "source_hashes": {rel: sha(source(rel)) for rel in (
            "outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv",
            "outputs/heedb_huang_exploratory_v1/TWA_THREE_SEED_MEANS.csv",
            "outputs/heedb_huang_exploratory_v1/PAIRED_PATIENT_BOOTSTRAP.csv")},
        "artifact_hashes": {str(path.relative_to(OUT)).replace("\\", "/"): sha(path) for path in files},
        "figure_data_rows": len(figure_data),
        "source_code_sha256": sha(Path(__file__)),
    }
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figure_data_rows": len(figure_data),
                      "figures": len(list(figdir.glob("*.svg")))}, sort_keys=True))


if __name__ == "__main__":
    main()
