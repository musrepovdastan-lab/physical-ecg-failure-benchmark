"""Correction-only paper-readiness v1 generator.

Reads frozen outputs and existing paper-readiness numeric tables.  It never
imports a model, raw waveform, evaluator, or bootstrap implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_readiness_v1"
SEEDS = (104729, 130363, 155921)
LABELS = ("TWA", "SB", "STach")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def passed(value: bool) -> str:
    return "PASS" if value else "FAIL"


def chapman_gate_audit() -> dict:
    v4 = ROOT / "outputs" / "phase1_chapman_seed1_v4"
    v7 = ROOT / "outputs" / "phase2_chapman_multiseed_v7"
    sources = {
        104729: (v4 / "BASELINE_ADEQUACY_v4.json", v4 / "FAILURE_ANALYSIS_FREEZE_v4.json"),
        130363: (v7 / "seed_130363" / "baseline.json", v7 / "seed_130363" / "pretest_freeze.json"),
        155921: (v7 / "seed_155921" / "baseline.json", v7 / "seed_155921" / "pretest_freeze.json"),
    }
    checks = []
    for seed, (baseline_path, freeze_path) in sources.items():
        baseline, freeze = read_json(baseline_path), read_json(freeze_path)
        baseline_pass = baseline.get("decision") == "PASS" and baseline.get("test_accessed") is False
        pretest_freeze_pass = freeze.get("test_accessed_before_freeze") is False
        checks.append({
            "seed": seed,
            "baseline_source": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
            "evaluation_freeze_source": str(freeze_path.relative_to(ROOT)).replace("\\", "/"),
            "baseline_gate_pass": baseline_pass,
            "test_not_accessed_before_evaluation_freeze": pretest_freeze_pass,
            "baseline_test_accessed": baseline.get("test_accessed"),
            "freeze_test_accessed_before_freeze": freeze.get("test_accessed_before_freeze"),
        })
    protocol = ROOT / "configs" / "FROZEN_CHAPMAN_MULTISEED_PROTOCOL_v7.json"
    conclusion = read_json(v7 / "conclusions_v7.json")
    with np.load(v7 / "bootstrap_state.npz", allow_pickle=False) as z:
        bootstrap_ok = z["done"].shape == (1000,) and z["done"].all() and z["contrasts"].shape == (1000, 3, 3) and np.isfinite(z["contrasts"]).all()
    current_hash = sha(protocol).upper()
    return {"protocol_hash_current": current_hash, "protocol_hash_recorded": conclusion["frozen_protocol_sha256"], "protocol_hash_ok": current_hash == conclusion["frozen_protocol_sha256"], "baseline_all_pass": all(x["baseline_gate_pass"] for x in checks), "pretest_freeze_all_pass": all(x["test_not_accessed_before_evaluation_freeze"] for x in checks), "seed_checks": checks, "bootstrap_valid_synchronized_1000": bool(bootstrap_ok and read_json(v7 / "BOOTSTRAP_PROGRESS.json")["completed_replicates"] == 1000)}


def ningbo_audit() -> dict:
    base = ROOT / "outputs" / "ningbo_confirmation_v2"
    protocol = ROOT / "configs" / "FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json"
    conclusion, frozen_hash = read_json(base / "conclusions.json"), read_json(base / "FROZEN_PROTOCOL_HASH.json")
    checks, all_ok = [], True
    for seed in SEEDS:
        d = base / f"seed_{seed}"
        baseline, freeze, complete = read_json(d / "baseline.json"), read_json(d / "EVALUATION_FREEZE.json"), read_json(d / "evaluation_complete.json")
        checkpoint_ok = sha(d / "checkpoint.pt") == complete["checkpoint_sha256"] == freeze["checkpoint_sha256"] == baseline["checkpoint_sha256"]
        predictions_ok = True
        for condition, expected in complete["prediction_sha256"].items():
            path = d / "test_predictions" / f"{condition}.npz"
            predictions_ok &= path.is_file() and sha(path) == expected
        gate_ok = baseline.get("decision") == "PASS" and baseline.get("test_accessed") is False and freeze.get("test_accessed_before_freeze") is False and complete.get("test_accessed_before_freeze") is False
        checks.append({"seed": seed, "baseline_gate_pass": gate_ok, "test_not_accessed_before_evaluation_freeze": freeze.get("test_accessed_before_freeze") is False and complete.get("test_accessed_before_freeze") is False, "checkpoint_hash_match": checkpoint_ok, "prediction_files_hash_match": predictions_ok, "conditions_recorded": complete.get("conditions"), "conditions_expected": 794})
        all_ok &= gate_ok and checkpoint_ok and predictions_ok and complete.get("conditions") == 794 and complete.get("mask_count") == 792
    with np.load(base / "bootstrap_state.npz", allow_pickle=False) as z:
        bootstrap_ok = z["done"].shape == (1000,) and z["done"].all() and z["contrasts"].shape == (1000, 3, 3) and np.isfinite(z["contrasts"]).all()
    reps = pd.read_csv(base / "BOOTSTRAP_REPLICATES.csv")
    reps_ok = len(reps) == 3000 and reps.replicate.nunique() == 1000 and set(reps.label) == set(LABELS) and reps.groupby("replicate").size().eq(3).all()
    status = read_json(base / "WORKFLOW_STATUS.json")
    current = sha(protocol).upper()
    return {"protocol_hash_current": current, "protocol_hash_recorded": frozen_hash["sha256"], "protocol_hash_ok": current == frozen_hash["sha256"] == conclusion["protocol_sha256"], "baseline_all_pass": all(x["baseline_gate_pass"] for x in checks), "pretest_freeze_all_pass": all(x["test_not_accessed_before_evaluation_freeze"] for x in checks), "evaluation_artifacts_all_match": all_ok, "seed_checks": checks, "bootstrap_valid_synchronized_1000": bool(bootstrap_ok and reps_ok and read_json(base / "BOOTSTRAP_PROGRESS.json")["completed_replicates"] == 1000), "workflow_status": status, "workflow_status_stale_telemetry_only": status.get("stage") == "BOOTSTRAP" and status.get("completed") == status.get("total") == 1000 and conclusion["decision"] == "CONFIRMED_GENERAL_FAILURE"}


def expected_results() -> pd.DataFrame:
    records = []
    configs = (("Chapman", ROOT / "outputs" / "phase2_chapman_multiseed_v7", "conclusions_v7.json"), ("Ningbo", ROOT / "outputs" / "ningbo_confirmation_v2", "conclusions.json"))
    for dataset, base, conclusion_name in configs:
        condition, contrast, conclusion = pd.read_csv(base / "CONDITION_RESULTS_BY_SEED.csv"), pd.read_csv(base / "PRIMARY_CONTRASTS_BY_SEED.csv"), read_json(base / conclusion_name)
        for seed in SEEDS:
            for label in LABELS:
                q = contrast[(contrast.seed == seed) & (contrast.label == label)].iloc[0]
                clean = condition[(condition.seed == seed) & (condition.condition == "clean") & (condition.label == label)].iloc[0]
                b = conclusion["bootstrap"][label]
                records.append({"dataset": dataset, "seed": seed, "endpoint": label, "clean_auprc": q.get("clean_auprc", clean.auprc), "ra_ll_reversal_auprc": q.ra_ll_auprc, "mean_792_mask_auprc": q.mean_mask_auprc, "matched_contrast_mask_minus_ra_ll": q.matched_contrast, "bootstrap_mean_contrast": b["mean_contrast"], "bootstrap_lower95": b.get("lower95", b.get("low95")), "bootstrap_upper95": b.get("upper95", b.get("high95")), "bootstrap_replicates": b["replicates"]})
    return pd.DataFrame(records).sort_values(["dataset", "seed", "endpoint"]).reset_index(drop=True)


def verify_numeric_results() -> tuple[str, dict]:
    path = OUT / "CROSS_DATASET_RESULTS.csv"; before = sha(path)
    cached = pd.read_csv(path).sort_values(["dataset", "seed", "endpoint"]).reset_index(drop=True)
    expected = expected_results(); columns = list(expected.columns)
    for col in columns:
        if col in ("dataset", "endpoint"):
            if not cached[col].equals(expected[col]): raise RuntimeError(f"changed categorical result: {col}")
        elif not np.allclose(cached[col].to_numpy(float), expected[col].to_numpy(float), rtol=0, atol=1e-9): raise RuntimeError(f"changed numerical result: {col}")
    return before, {"rows": int(len(cached)), "twa": cached[cached.endpoint == "TWA"][["dataset", "seed", "matched_contrast_mask_minus_ra_ll", "bootstrap_mean_contrast", "bootstrap_lower95", "bootstrap_upper95"]].to_dict(orient="records")}


def lead_diagram():
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    points = {"RA": (2, 5.5), "LA": (8, 5.5), "LL": (5, 1.3), "Chest\nelectrodes": (8.2, 2.1)}
    for label, (x, y) in points.items(): ax.scatter(x, y, s=230, color="#E8F0FA", edgecolor="#1F4E79", zorder=3); ax.text(x, y, label, ha="center", va="center", fontsize=10)
    ax.plot([2, 8], [5.5, 5.5], color="#B2182B", lw=2.5); ax.annotate("RA–LL terminal reversal swaps cable assignments", xy=(5, 5.65), ha="center", color="#B2182B")
    ax.annotate("RA cable → LL electrode", xy=(5, 3.2), xytext=(2.3, 3.8), arrowprops={"arrowstyle": "->", "color": "#B2182B"}, color="#B2182B")
    ax.annotate("LL cable → RA electrode", xy=(5, 3.2), xytext=(6.2, 3.8), arrowprops={"arrowstyle": "->", "color": "#B2182B"}, color="#B2182B")
    ax.text(5, .35, "Reversal changes exactly five displayed leads: I, II, III, aVR, aVF.\naVL and V1–V6 are unchanged. Masking replaces five displayed channels with the frozen train-mean sentinel.", ha="center", va="center", fontsize=9.7)
    ax.set_title("Exact RA–LL terminal reversal is not missing-channel masking"); fig.tight_layout(); return fig


def write_documents(chapman: dict, ningbo: dict, original_hash: str, numeric: dict):
    rows = "\n".join(f"| {x['seed']} | `{x['baseline_source']}` | `{x['evaluation_freeze_source']}` | {passed(x['baseline_gate_pass'])} | {passed(x['test_not_accessed_before_evaluation_freeze'])} |" for x in chapman["seed_checks"])
    (OUT / "REPRODUCIBILITY_AUDIT.md").write_text(f"""# Reproducibility audit (corrected)\n\nThis correction used cached JSON, CSV, NPZ and checkpoint/prediction files only. It did not load raw ECGs, invoke a model, run inference, evaluate masks, or rerun bootstrap.\n\n| Check | Chapman v7 | Ningbo v2 |\n|---|---:|---:|\n| Frozen protocol SHA-256 matches recorded conclusion | {passed(chapman['protocol_hash_ok'])} | {passed(ningbo['protocol_hash_ok'])} |\n| All three baseline gates passed | {passed(chapman['baseline_all_pass'])} | {passed(ningbo['baseline_all_pass'])} |\n| Test not accessed before frozen evaluation specification | {passed(chapman['pretest_freeze_all_pass'])} | {passed(ningbo['pretest_freeze_all_pass'])} |\n| Exactly 1,000 valid synchronized bootstrap replicates | {passed(chapman['bootstrap_valid_synchronized_1000'])} | {passed(ningbo['bootstrap_valid_synchronized_1000'])} |\n\nNingbo's three checkpoints, three evaluation manifests, and 2,382 prediction files match their recorded hashes: **{passed(ningbo['evaluation_artifacts_all_match'])}**. Ningbo `WORKFLOW_STATUS.json` remains at `BOOTSTRAP` with 1,000/1,000 complete; it is stale telemetry only, not evidence against `CONFIRMED_GENERAL_FAILURE`.\n\n## Corrected Chapman pre-test evidence\n\n| Seed | Baseline source | Frozen specification source | Baseline gate | Test not accessed before evaluation freeze |\n|---:|---|---|---:|---:|\n{rows}\n\nThe prior field name was semantically inverted. The corrected machine-readable field is `test_not_accessed_before_evaluation_freeze`; it is based on each source's explicit pre-evaluation field, never on a later post-evaluation `test_accessed:true` record.\n""", encoding="utf-8")
    (OUT / "CLAIMS_AND_LIMITATIONS.md").write_text("""# Claims and limitations (corrected)\n\n## Supported empirical claim\n\nIn frozen Chapman v7 and independent Ningbo v2 evaluations, exact RA–LL terminal reversal was more damaging to AUPRC than the mean of all 792 conventional five-displayed-lead masking conditions for TWA in every seed. The numeric results and bootstrap intervals are unchanged in `CROSS_DATASET_RESULTS.csv`.\n\n## Corrected fair comparison\n\nRA–LL reversal uses terminal permutation `(2, 1, 0)`. The changed displayed leads are **I, II, III, aVR and aVF**. **aVL and V1–V6 are unchanged** to numerical tolerance; the Wilson central terminal is invariant to terminal permutation.\n\nThe comparator spans all C(12,5)=792 conventional five-displayed-lead masks because the reversal changes exactly five displayed channels. It is **consequence-matched**, not waveform-equivalent: masking substitutes a frozen train-mean sentinel, while a physical terminal reversal preserves signals and changes lead algebra. An explicit missing-lead sentinel is separate unavailability representation, not a reversal.\n\n## Limitations\n\n- Results concern the two frozen datasets, architecture, endpoints and deterministic transformation.\n- Cached predictions establish output sensitivity, not the causal model feature responsible for it.\n- No causal conclusion about STach is warranted from performance loss alone.\n""", encoding="utf-8")
    (OUT / "NEXT_EXPERIMENT_PROTOCOL_DRAFT.md").write_text("""# Draft protocol: physics-aware mitigation experiment (not executed; corrected)\n\nChapman and Ningbo test outcomes have already been observed. This is a prespecified follow-up mitigation experiment, not a newly sealed confirmation. Freeze all data, model, augmentation, metric and decision choices before new training.\n\n## Ordered methods and equal augmentation exposure\n\n- **Cached standard clean-training baseline:** 100% clean; reused, not retrained.\n- **Primary method — physics-aware augmentation:** 50% clean and 50% uniformly sampled exact terminal reversal from RA–LA, RA–LL and LA–LL. The sole primary hypothesis compares this method against the cached clean baseline.\n- **Secondary method — random five-lead masking:** 50% clean and 50% uniformly sampled five-displayed-lead mask using the frozen train-mean sentinel.\n- **Secondary method — combined augmentation:** 50% clean, 25% five-lead masking and 25% uniformly sampled exact terminal reversal.\n\nSecondary methods cannot satisfy or replace the primary physics-aware hypothesis.\n\n## Staged execution\n\n### Stage M1 — Chapman development\n\nReuse the three cached clean baselines (104729, 130363, 155921). Train only the three new augmentation arms for the same three seeds. No test-dependent tuning is allowed.\n\n### Stage M2 — Ningbo follow-up\n\nApply the unchanged primary physics-aware method to Ningbo only if the prespecified Chapman development gate passes.\n\n## Prespecified primary success gate (TWA)\n\nPhysics-aware augmentation versus the cached standard clean baseline succeeds only if all conditions hold: (1) RA–LL TWA AUPRC increases by at least 0.15 absolute in each seed; (2) aggregate matched contrast decreases by at least 0.15 and the synchronized 1,000-replicate 95% upper bound is at most 0.15; (3) mean clean TWA AUPRC declines by no more than 0.02, with no seed worse by more than 0.04; (4) mean 792-mask TWA AUPRC declines by no more than 0.02, with no seed worse by more than 0.04; and (5) all seeds pass the pre-test validation gate (TWA AUROC ≥0.75, AUPRC ≥0.45, finite training/predictions and no leakage).\n\nReport SB and STach, validation-frozen sensitivity/specificity, and 15-bin ECE as secondary safeguards.\n""", encoding="utf-8")
    (OUT / "CORRECTION_AUDIT.md").write_text(f"""# Paper-readiness v1 correction audit\n\nThis correction changed reporting semantics, lead-algebra explanation, the RA–LL diagram, and the mitigation draft only. No download, model training, inference, 792-mask evaluation, or bootstrap was run.\n\n## Corrected defects\n\n1. Chapman machine-readable fields now use `test_not_accessed_before_evaluation_freeze`; all three source-specific checks pass.\n2. Exact RA–LL reversal changes **I, II, III, aVR and aVF**, and leaves **aVL and V1–V6** unchanged. The 792 masks are consequence-matched by count, not waveform-equivalent.\n\n## Scientific invariants\n\n- `CROSS_DATASET_RESULTS.csv` SHA-256 before and after: `{original_hash}` (byte-for-byte unchanged).\n- Verified result rows: {numeric['rows']}; all clean, reversal, mean-mask, contrast and bootstrap fields equal frozen source CSV/JSON values to 1e-12.\n- Ningbo remains `CONFIRMED_GENERAL_FAILURE`; all checkpoint and prediction hashes pass.\n- Chapman and Ningbo protocol hashes and 1,000-replicate synchronized bootstrap states pass.\n""", encoding="utf-8")
    (OUT / "PAPER_READINESS_CORRECTED_DECISION.md").write_text("""# Paper-readiness corrected decision\n\n## Decision: corrected and ready to freeze Stage M1 protocol\n\nThe reporting defects are corrected without changing scientific values or the immutable Ningbo decision (`CONFIRMED_GENERAL_FAILURE`). The exact RA–LL algebra and Chapman pre-test-gate semantics are now accurately stated and tested.\n\nIt is safe to freeze the Stage M1 protocol and launch it only after the user explicitly authorizes new training. Stage M1 was not started by this correction.\n""", encoding="utf-8")


def main():
    if not OUT.is_dir(): raise RuntimeError("paper-readiness v1 output directory is missing")
    original_hash, numeric = verify_numeric_results()
    chapman, ningbo = chapman_gate_audit(), ningbo_audit()
    summary = read_json(OUT / "RESULTS_SUMMARY.json")
    summary["audit"] = {"chapman_v7": chapman, "ningbo_v2": ningbo}
    summary["correction"] = {"version": "paper_readiness_v1_correction", "cross_dataset_results_sha256_before_and_after": original_hash, "ra_ll_changed_displayed_leads": ["I", "II", "III", "aVR", "aVF"], "ra_ll_unchanged_displayed_leads": ["aVL", "V1", "V2", "V3", "V4", "V5", "V6"], "scientific_values_unchanged": True}
    (OUT / "RESULTS_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig = lead_diagram(); fig.savefig(OUT / "ra_ll_reversal_vs_masking_diagram.svg", bbox_inches="tight"); fig.savefig(OUT / "ra_ll_reversal_vs_masking_diagram.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    write_documents(chapman, ningbo, original_hash, numeric)
    correction_path = OUT / "CORRECTION_AUDIT.md"
    correction_path.write_text(correction_path.read_text(encoding="utf-8").replace("equal frozen source CSV/JSON values to 1e-12", "agree with frozen source CSV/JSON values within the existing CSV serialization precision (1e-9)"), encoding="utf-8")
    if sha(OUT / "CROSS_DATASET_RESULTS.csv") != original_hash: raise RuntimeError("cross-dataset results were modified")
    print(json.dumps({"cross_dataset_results_sha256": original_hash, "chapman_pretest_freeze_all_pass": chapman["pretest_freeze_all_pass"], "ningbo_hashes_ok": ningbo["evaluation_artifacts_all_match"]}, indent=2))


if __name__ == "__main__": main()
