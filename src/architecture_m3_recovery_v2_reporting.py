"""Generate M3 recovery reports without touching inference caches or predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/architecture_replication_m3_recovery_v2"
SEEDS = (104729, 130363, 155921)
ARMS = ("clean", "physics_aware")
ENDPOINTS = ("TWA", "SB", "STach")
EXACT = "mask5_0_1_2_3_5"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for q in iter(lambda: f.read(1 << 20), b""):
            h.update(q)
    return h.hexdigest()


def f(v):
    return float(v)


def generate():
    protocol = read_json(OUT / "RECOVERY_PROTOCOL.json")
    decision = read_json(OUT / "conclusions_m3.json")
    status = read_json(OUT / "WORKFLOW_STATUS.json")
    progress = read_json(OUT / "BOOTSTRAP_PROGRESS.json")
    units = protocol["source_audit"]["units"]
    conditions = read_csv(OUT / "CONDITION_RESULTS_BY_SEED.csv")
    distributions = read_csv(OUT / "MASK_DISTRIBUTION_SUMMARY.csv")
    contrasts = read_csv(OUT / "PRIMARY_CONTRASTS.csv")
    boot = read_csv(OUT / "BOOTSTRAP_SUMMARY.csv")

    unit_validation = []
    for u in units:
        gate = read_json(ROOT / u["artifacts"]["validation_gate.json"]["path"])
        unit_validation.append({"arm": u["arm"], "seed": u["seed"],
                                "checkpoint_identity_and_validation_artifact": gate.get("decision") == "PASS",
                                "finite_validation_predictions": bool(gate.get("finite_predictions")),
                                "validation_test_separation_flag": gate.get("test_accessed_before_evaluation_freeze") is False,
                                "validation_twa_auprc": u["validation_metrics_recomputed"]["auprc"],
                                "validation_twa_auroc": u["validation_metrics_recomputed"]["auroc"],
                                "checkpoint_sha256": u["artifacts"]["checkpoint.pt"]["sha256"],
                                "validation_artifact_sha256": u["artifacts"]["validation.npz"]["sha256"]})

    cache_units = []
    for arm in ARMS:
        for seed in SEEDS:
            d = OUT / "Chapman" / arm / f"seed_{seed}" / "predictions"
            files = sorted(d.glob("*.npz"))
            mask_files = [p for p in files if p.stem.startswith("mask5_")]
            cache_units.append({"arm": arm, "seed": seed, "conditions": len(files), "mask_conditions": len(mask_files),
                                "expected_conditions": 794, "expected_masks": 792,
                                "summary_loader_verified_binding_shape_finiteness": True,
                                "binding_semantic": protocol["evaluator"]["semantic_version"]})
    counts_ok = all(q["conditions"] == 794 and q["mask_conditions"] == 792 for q in cache_units)
    report_integrity = {
        "decision": "PASS" if (status.get("stage") == "COMPLETE" and progress.get("completed_tasks") == 18
                                  and progress.get("total_tasks") == 18 and counts_ok
                                  and len(conditions) == 6 * 794 * 3 and len(distributions) == 18
                                  and len(contrasts) == 18 and len(boot) == 15
                                  and len(read_csv(OUT / "BOOTSTRAP_REPLICATES.csv")) == 3000
                                  and all(x["checkpoint_identity_and_validation_artifact"] and x["finite_validation_predictions"] for x in unit_validation)) else "FAIL",
        "protocol_binding_sha256": protocol["binding_sha256"],
        "evaluation_completion": status,
        "prediction_cache_units": cache_units,
        "prediction_cache_count": sum(x["conditions"] for x in cache_units),
        "corrected_mask_prediction_count": sum(x["mask_conditions"] for x in cache_units),
        "mask_conditions_expected": 6 * 792,
        "condition_metric_rows": len(conditions),
        "bootstrap_tasks": progress,
        "bootstrap_replicate_rows": len(read_csv(OUT / "BOOTSTRAP_REPLICATES.csv")),
        "validation_units": unit_validation,
        "legacy_mask_predictions_reused": False,
        "legacy_clean_ra_ll_replay": "failed frozen tolerance for all 12 probes; all 12 conditions fully re-inferred under amended protocol",
        "test_accessed_historically": True,
        "interpretation": "corrected reanalysis, not an untouched confirmation",
    }
    (OUT / "MASK_INFERENCE_INTEGRITY.json").write_text(json.dumps(report_integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "VALIDATION_GATES.json").write_text(json.dumps({"protocol_binding_sha256": protocol["binding_sha256"],
        "thresholds_unchanged": True, "validation_units": unit_validation,
        "all_validation_units_passed": all(x["checkpoint_identity_and_validation_artifact"] and x["finite_validation_predictions"] for x in unit_validation),
        "stage_m3a_decision": decision["decision"], "stage_m3a_gate_checks": decision["gate_checks"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    byc = {(r["arm"], int(r["seed"]), r["endpoint"], r["condition"]): r for r in conditions}
    byd = {(r["arm"], int(r["seed"]), r["endpoint"]): r for r in distributions}
    bycontrast = {(r["arm"], int(r["seed"]), r["endpoint"]): r for r in contrasts}
    bs = {(r["endpoint"], r["estimand"]): r for r in boot}
    table = []
    for ep in ENDPOINTS:
        for seed in SEEDS:
            for arm in ARMS:
                q = bycontrast[(arm, seed, ep)]
                dist = byd[(arm, seed, ep)]
                table.append({"endpoint": ep, "seed": seed, "arm": arm,
                              "clean_auprc": q["clean_auprc"], "ra_ll_auprc": q["ra_ll_auprc"],
                              "exact_mask_auprc": q["exact_mask_auprc"], "mean_792_mask_auprc": q["mean_mask_auprc"],
                              "exact_mask_minus_ra_ll": q["exact_mask_minus_ra_ll"],
                              "mean_mask_minus_ra_ll": q["mean_mask_minus_ra_ll"],
                              "mask_median": dist["mask_median_auprc"], "mask_p05": dist["mask_p05_auprc"],
                              "mask_p25": dist["mask_p25_auprc"], "mask_p75": dist["mask_p75_auprc"],
                              "mask_p95": dist["mask_p95_auprc"]})
    write_csv(OUT / "EXACT_MASK_AND_DISTRIBUTION_BY_SEED.csv", table)

    check_lines = []
    for name, q in decision["gate_checks"].items():
        val = q.get("values", q.get("value", q.get("upper95")))
        check_lines.append(f"| {name} | {'PASS' if q['pass'] else 'FAIL'} | `{val}` | `{q.get('threshold', 'integrity')}` |")
    per_seed_lines = []
    for seed in SEEDS:
        b = bycontrast[("clean", seed, "TWA")]
        m = bycontrast[("physics_aware", seed, "TWA")]
        bd = byd[("clean", seed, "TWA")]
        md = byd[("physics_aware", seed, "TWA")]
        per_seed_lines.append(f"| {seed} | {f(b['clean_auprc']):.4f} | {f(b['ra_ll_auprc']):.4f} | {f(bd['exact_mask_auprc']):.4f} | {f(bd['mask_mean_auprc']):.4f} | {f(m['clean_auprc']):.4f} | {f(m['ra_ll_auprc']):.4f} | {f(md['exact_mask_auprc']):.4f} | {f(md['mask_mean_auprc']):.4f} |")
    clean_changes = decision["gate_checks"]["mean_clean_decline_max_0_02"]["values"]
    mask_changes = decision["gate_checks"]["mean_mask_decline_max_0_02"]["values"]
    boot_lines = []
    for key, label in (("physics_minus_clean_ra_ll", "RA–LL TWA AUPRC improvement"),
                       ("contrast_reduction", "Mean-mask minus RA–LL contrast reduction"),
                       ("physics_mean_mask_minus_ra_ll", "Physics-aware mean-mask minus RA–LL contrast"),
                       ("physics_minus_clean_clean", "Clean TWA AUPRC change"),
                       ("physics_minus_clean_mean_mask", "Mean-mask TWA AUPRC change")):
        q = bs[("TWA", key)]
        boot_lines.append(f"| {label} | {f(q['bootstrap_mean']):+.4f} | [{f(q['lower95']):+.4f}, {f(q['upper95']):+.4f}] | 1,000 |")

    m3report = ["# TimePatchTransformer M3 corrected evaluation", "",
        f"**Decision: {decision['decision']}**. The frozen Stage M3A gate was not passed, so Stage B (Ningbo training/evaluation) was not run.", "",
        "This is a corrected reanalysis of the previously accessed Chapman test set, not an untouched confirmation. No Chapman model was retrained. Historical inverted mask predictions were not reused.", "",
        f"Recovery protocol binding: `{protocol['binding_sha256']}`. Original M3 protocol SHA-256: `{protocol['source_audit']['original_protocol']['sha256']}`.",
        f"Integrity: **{report_integrity['decision']}**; six units, {report_integrity['corrected_mask_prediction_count']}/4,752 corrected mask predictions, 14,292 condition/endpoint metric rows, and 18/18 synchronized bootstrap tasks.", "",
        "## Stage M3A decision", "", "| Frozen criterion | Result | Estimate | Threshold |", "|---|---|---:|---:|"] + check_lines + ["",
        "The sole failed criterion is the per-seed mean-mask retention limit: seed 155921 declined by 0.1126 TWA AUPRC, exceeding the allowed 0.04 decline. The mean across seeds was a 0.0188 decline and passed the 0.02 limit. This is a mask-performance cost at one seed, not evidence that the RA–LL protection effect is absent.",
        "RA–LL TWA AUPRC improved by at least 0.15 in all three seeds; aggregate mask-minus-RA–LL contrast reduction passed; the physics-aware contrast upper 95% limit also passed. Mean and per-seed clean-performance limits passed.", "",
        "## Per-seed TWA AUPRC", "", "| Seed | Clean arm clean | Clean arm RA–LL | Clean arm exact mask | Clean arm 792-mask mean | Physics-aware clean | Physics-aware RA–LL | Physics-aware exact mask | Physics-aware 792-mask mean |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"] + per_seed_lines + ["",
        "The exact mask removes I, II, III, aVR, and aVF (indices [0,1,2,3,5]); it matches the corresponding column in the lexicographic 792-mask panel. Every mask metric was computed separately before taking the panel mean.", "",
        "## Synchronized bootstrap", "", "| TWA estimand | Bootstrap mean | Percentile 95% interval | Replicates |", "|---|---:|---:|---:|"] + boot_lines + ["",
        "Bootstrap resamples use the existing Chapman multiseed indices (1000 × 2050, canonical test record order; synchronized across conditions, seeds, and arms). Central estimates use the complete test sample; bootstrap means are reported separately.", "",
        "## Interpretation and next steps", "",
        "The corrected Chapman results show a large RA–LL benefit for physics-aware training and a substantial reduction in the mask-versus-RA–LL contrast. The frozen Stage M3A gate nevertheless fails because seed 155921 misses the mean-mask retention bound. Under the protocol, this is `FAIL_M3_ARCHITECTURE_SPECIFIC`; it is not a passed architecture replication.",
        "Ningbo Transformer training and test evaluation were not run because Stage M3A did not pass. No new method, architecture, threshold, split, or probability was selected.",
        "Chapman ResNet corrected mask panel status remains unavailable/not fully verified; this package did not restore it. Ningbo ResNet values below are from the separate corrected recovery package, after historical Ningbo test access.", "",
        "| Model family | Chapman corrected mask status | Ningbo corrected panel/status | Interpretation |", "|---|---|---|---|"]
    m3report += ["| ECGResNet-small | Not fully verified; not regenerated here | Corrected recovery complete for baseline and M2 (6 units, 4,752 predictions); M2 retained its reported frozen-gate PASS | Ningbo is corrected reanalysis after prior test access, not independent untouched confirmation. |",
                 "| TimePatchTransformer | Corrected 6-unit Chapman panel complete; Stage M3A **FAIL** on one-seed mean-mask retention | **Not run** because Stage M3A gate failed | Ningbo transfer of the intervention remains untested. |", "",
                 "Open research questions remain chest-only performance, calibration, published baselines, and clinical interpretation. This work does not rank Transformer versus ResNet as an architecture in general.", "",
                 "Detailed tables: `CONDITION_RESULTS_BY_SEED.csv`, `EXACT_MASK_AND_DISTRIBUTION_BY_SEED.csv`, `MASK_DISTRIBUTION_SUMMARY.csv`, `PRIMARY_CONTRASTS.csv`, `BOOTSTRAP_SUMMARY.csv`, and `BOOTSTRAP_REPLICATES.csv`.", ""]
    (OUT / "CHAPMAN_REPORT.md").write_text("\n".join(m3report), encoding="utf-8")
    (OUT / "FINAL_M3_DECISION.md").write_text("\n".join(["# Final M3 decision", "", f"**{decision['decision']}**", "",
        "Stage M3A fails the frozen gate because the physics-aware 792-mask mean TWA AUPRC for seed 155921 declines by 0.1126 relative to its clean-trained counterpart; the allowed per-seed decline is 0.04. Other Stage M3A criteria pass, including all three RA–LL gains, aggregate contrast reduction, physics-aware contrast upper 95% limit, mean clean decline, mean mask decline, validation, and integrity.", "",
        "This gate failure is a performance-retention failure at one seed, not an absence of the measured RA–LL protection benefit. The unchanged protocol therefore blocks Stage B. No Ningbo Transformer training or evaluation was performed.", "",
        "The result is a corrected Chapman reanalysis after prior test access. It is not untouched confirmation. See `CHAPMAN_REPORT.md` and `conclusions_m3.json` for all estimates and criteria.", ""]), encoding="utf-8")

    # A compact ResNet vs Transformer status table, with no cross-family superiority claim.
    write_csv(OUT / "RESNET_VS_TRANSFORMER_STATUS.csv", [
        {"model_family": "ECGResNet-small", "Chapman_corrected_792_mask_panel": "not fully verified; not run in this package",
         "Ningbo_corrected_panel": "complete: baseline and physics-aware M2; 6 units, 4,752 corrected mask predictions",
         "gate_status": "Ningbo M2 reported PASS under its own frozen M2 criteria",
         "test_provenance": "Ningbo corrected reanalysis after prior test access; not untouched confirmation"},
        {"model_family": "TimePatchTransformer", "Chapman_corrected_792_mask_panel": "complete: 6 units; Stage M3A FAIL at seed 155921 mask-retention criterion",
         "Ningbo_corrected_panel": "not run because frozen Stage M3A gate failed",
         "gate_status": "FAIL_M3_ARCHITECTURE_SPECIFIC",
         "test_provenance": "Chapman corrected reanalysis after historical test access; not untouched confirmation"},
    ])

    # Distribution plot for the primary TWA endpoint.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    positions, values, labels = [], [], []
    pos = 1
    for arm in ARMS:
        for seed in SEEDS:
            a = [f(r["auprc"]) for r in conditions if r["arm"] == arm and int(r["seed"]) == seed and r["endpoint"] == "TWA" and r["condition"].startswith("mask5_")]
            values.append(a); positions.append(pos); labels.append(f"{arm.replace('_', ' ')}\n{seed}"); pos += 1
        pos += 0.6
    ax.boxplot(values, positions=positions, widths=.55, showfliers=False, whis=(5, 95), patch_artist=True,
               boxprops={"facecolor": "#dbeafe", "edgecolor": "#2563eb"}, medianprops={"color": "#111827"})
    for i, (arm, seed) in enumerate(((a, s) for a in ARMS for s in SEEDS)):
        p = positions[i]
        q = bycontrast[(arm, seed, "TWA")]
        ax.scatter(p, f(q["ra_ll_auprc"]), color="#dc2626", marker="D", s=30, zorder=4,
                   label="RA–LL" if i == 0 else None)
        ax.scatter(p, f(q["exact_mask_auprc"]), color="#059669", marker="*", s=90, zorder=5,
                   label="Exact five-lead mask" if i == 0 else None)
    ax.axvline(3.8, color="#9ca3af", lw=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("TWA AUPRC")
    ax.set_title("Corrected Chapman TWA performance: each box contains 792 masks")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "chapman_twa_mask_distribution.png", dpi=180)
    plt.close(fig)

    artifacts = [OUT / n for n in ("RECOVERY_PROTOCOL.json", "RECOVERY_PROTOCOL_AMENDMENT_001.json", "RECOVERY_PROTOCOL_AMENDMENT_002.json",
                                    "PILOT_REPORT.json", "LEGACY_REPLAY_MISMATCH_BEFORE_REINFERENCE.json", "MASK_INFERENCE_INTEGRITY.json",
                                    "VALIDATION_GATES.json", "CONDITION_RESULTS_BY_SEED.csv", "EXACT_MASK_AND_DISTRIBUTION_BY_SEED.csv",
                                    "MASK_DISTRIBUTION_SUMMARY.csv", "PRIMARY_CONTRASTS.csv", "BOOTSTRAP_SUMMARY.csv", "BOOTSTRAP_REPLICATES.csv",
                                    "conclusions_m3.json", "FINAL_M3_DECISION.md", "CHAPMAN_REPORT.md", "RESNET_VS_TRANSFORMER_STATUS.csv",
                                    "chapman_twa_mask_distribution.png")]
    manifest = {"protocol_binding_sha256": protocol["binding_sha256"], "reporting_source_sha256": sha(Path(__file__)),
                "artifacts": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in artifacts}}
    (OUT / "REPORTING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "integrity": report_integrity["decision"],
                      "corrected_mask_predictions": report_integrity["corrected_mask_prediction_count"],
                      "conditions": len(conditions), "bootstrap_tasks": progress["completed_tasks"],
                      "output": str(OUT / "CHAPMAN_REPORT.md")}, indent=2))


if __name__ == "__main__":
    generate()
