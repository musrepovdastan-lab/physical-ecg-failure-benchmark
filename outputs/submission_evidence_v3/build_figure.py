"""Plot already calculated aggregate AP; never reads ECGs or predictions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    frame = pd.read_csv(OUT / "SEED_CORRECTED_TWA_MEANS.csv")
    styles = {"ordinary": ("Ordinary", "#697586", "o", "--"),
              "physics_aware_p050": ("Physics-aware p050", "#007f86", "s", "-"),
              "combined": ("Combined", "#bd5c25", "^", "-"),
              "Huang_full_pipeline": ("Huang pipeline", "#7047a3", "D", "-")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "seed-corrected-v3"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for ax, cohort in zip(axes, ("Ningbo_test", "HEEDB")):
        for method, (label, color, marker, line) in styles.items():
            rows = frame[(frame.cohort == cohort) & (frame.method == method) &
                         (frame.scenario == "equal_three_swap")].sort_values("q_fault")
            ax.plot(rows.q_fault, rows.AP_raw_mean, label=label, color=color,
                    marker=marker, markersize=4, linewidth=1.8, linestyle=line)
        ax.set_xscale("symlog", linthresh=.001)
        ax.set_xticks([0, .001, .005, .01, .05, .1, 1],
                      ["0", ".001", ".005", ".01", ".05", ".1", "1"])
        ax.set_ylim(.15, .73)
        ax.set_title("Ningbo" if cohort == "Ningbo_test" else "HEEDB (reused proxy-label cohort)",
                     loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel("Simulated fault mixture weight q")
        ax.grid(axis="y", color="#e3e7eb", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Raw TWA average precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Corrected initialization: equal-three-swap mixtures", x=.075,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(.075, .015, "Means of three seed-specific metrics; no ensembling. "
             "Curves are descriptive and show no confidence intervals. q is not observed prevalence.",
             fontsize=8, color="#465362")
    fig.subplots_adjust(top=.82, bottom=.26, left=.075, right=.985, wspace=.12)
    fig.savefig(OUT / "SEED_CORRECTED_AP.png", dpi=190)
    fig.savefig(OUT / "SEED_CORRECTED_AP.svg", metadata={"Date": None})
    vector = OUT / "SEED_CORRECTED_AP.svg"
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines())
                      + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
