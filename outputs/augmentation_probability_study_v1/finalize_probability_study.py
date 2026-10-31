from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ROOT_REL = ROOT.relative_to(Path(__file__).resolve().parents[2])
REPORT = ROOT / "FINAL_PROBABILITY_STUDY_REPORT_RU.md"
MANIFEST = ROOT / "RUN_MANIFEST.json"
ARMS = ["ordinary", "physics_p010", "physics_p025", "physics_aware", "combined"]
SEEDS = [104729, 130363, 155921]
CONDITIONS = ["clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201", "mask5_0_1_2_3_5"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Нет строк._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    metrics = pd.read_csv(ROOT / "METRICS_ALL_ARMS_BY_SEED.csv")
    summaries = {cohort: pd.read_csv(ROOT / f"METRICS_{cohort.upper()}_SUMMARY.csv") for cohort in ["Chapman", "Ningbo"]}
    bootstrap = pd.read_csv(ROOT / "PAIRED_BOOTSTRAP_TWA_AUPRC.csv")
    mixtures = pd.read_csv(ROOT / "MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv")

    lines = [
        "# Исследование вероятности физической аугментации: итоговый отчёт",
        "",
        "## Краткие ответы",
        "",
        "1. **10%, 25% и 50% swaps.** Новые `physics_p010` и `physics_p025` сравнили с историческим `physics_aware` (50% swaps), `ordinary` (0%) и `combined` (25% swaps + 25% masks). На clean у новых режимов TWA AUPRC близок к историческим режимам: различия малы и не дают основания выбирать вероятность. При физически нарушенных входах оба новых режима заметно превосходят ordinary. 50% режим сохраняет более сильные средние результаты на знакомой RA_LL перестановке и части cycle условий; преимущество по маскам зависит от когорты и от того, рассматривается конкретная маска или среднее по 792 маскам.",
        "2. **Меньшая доля повреждений.** При 10% swaps польза относительно обычного обучения сохраняется на RA_LL, обоих циклах и точной маске в обеих когортах. Для средней метрики 792 масок прирост также положительный, но скромнее, особенно в Ningbo. Это показывает перенос на изученные типы нарушений, не доказывает перенос на новые пациенты или независимую когорту.",
        "3. **25% swaps против combined.** Доля swaps одинакова, но combined заменяет оставшуюся четверть чистых примеров масками. `physics_p025` имел почти одинаковый clean AUPRC, но ниже AUPRC на RA_LL, циклах и масках в Ningbo; на Chapman разница также отрицательна для RA_LL, cycle_slots_120 и точной маски. По среднему 792 масок `physics_p025` выше combined на Chapman, но ниже на Ningbo. Поэтому эффекта «маски помогают/мешают» без оговорок нет.",
        "4. **Условия теста.** Сводная таблица ниже показывает TWA AUPRC по clean, знакомой перестановке, двум проверенным циклам и точной маске. Ordinary резко проседает на нарушенных сигналах. p010/p025 улучшают устойчивость, но не достигают одинакового уровня во всех стресс-условиях. Для остальных endpoint/метрик (AUPRC, AUROC, Brier, ECE15 и фиксированные пороговые показатели) доступны полные таблицы по каждому seed; отдельные метрики не следует смешивать с TWA AUPRC.",
        "5. **Chapman и Ningbo.** Направление сравнения p010 с ordinary совпало: практически нулевое изменение clean и положительное на RA_LL, обоих циклах и точной маске. p025 против combined совпал по направлению на clean, RA_LL, cycle_slots_120 и точной маске; cycle_slots_201 и средние 792 маски различались. Это согласованность части сигналов, а не независимое подтверждение: те же заранее исследованные наборы и три seed.",
        "6. **Что остаётся неопределённым.** Только три seed; bootstrap интервалы условны на выбранных моделях и test когортах. Интервалы не измеряют неопределённость выбора моделей и не являются доказательством равенства при пересечении нуля. Не установлена оптимальная вероятность swaps; нет оценки HEEDB, reserved test или иных новых данных. Сравнение p025 с combined изолирует замену части clean примеров масками при фиксированной доле swaps, но не все различия в процессе обучения.",
        "",
        "## Результаты по TWA AUPRC",
        "",
        "Среднее ± SD по трём seed; здесь приведены значения из исходных seed-specific метрик. Для 792 масок отдельный bootstrap отчёт хранит среднюю метрику по маскам, а не смешивает её с конкретной маской.",
        "",
    ]

    for cohort in ["Chapman", "Ningbo"]:
        tab = summaries[cohort]
        tab = tab[(tab.endpoint == "TWA") & tab.condition.isin(CONDITIONS)].copy()
        tab["AUPRC mean ± SD"] = tab.apply(lambda r: f"{r.auprc_mean:.3f} ± {r.auprc_sd:.3f}", axis=1)
        pivot = tab.pivot(index="condition", columns="arm", values="AUPRC mean ± SD").reindex(CONDITIONS)
        pivot = pivot.reindex(columns=ARMS)
        pivot.index.name = "Условие"
        pivot.columns.name = None
        lines += [f"### {cohort}", "", md_table(pivot.reset_index()), ""]

    lines += [
        "Для ключевых условий ниже приведены те же seed mean ± SD для AUROC и Brier. Более низкий Brier лучше; AUROC не заменяет AUPRC при дисбалансе классов.",
        "",
    ]
    for cohort in ["Chapman", "Ningbo"]:
        tab = summaries[cohort]
        tab = tab[(tab.endpoint == "TWA") & tab.condition.isin(CONDITIONS)].copy()
        tab["AUROC / Brier"] = tab.apply(lambda r: f"{r.auroc_mean:.3f} ± {r.auroc_sd:.3f} / {r.brier_mean:.3f} ± {r.brier_sd:.3f}", axis=1)
        pivot = tab.pivot(index="condition", columns="arm", values="AUROC / Brier").reindex(CONDITIONS)
        pivot = pivot.reindex(columns=ARMS)
        pivot.index.name = "Условие"
        pivot.columns.name = None
        lines += [f"### {cohort}: AUROC / Brier", "", md_table(pivot.reset_index()), ""]

    mask_tables = []
    for cohort in ["Chapman", "Ningbo"]:
        df = pd.read_csv(ROOT / f"MASK_DISTRIBUTION_{cohort.upper()}.csv")
        agg = df.groupby("arm", as_index=False)[["mean_auprc", "median_auprc", "p05_auprc", "p25_auprc", "p75_auprc", "p95_auprc", "exact_mask_auprc"]].mean()
        agg.insert(0, "cohort", cohort)
        mask_tables.append(agg)
    mask_summary = pd.concat(mask_tables, ignore_index=True).round(4)
    lines += [
        "### Распределение по 792 маскам",
        "",
        "Ниже средние по seed для распределительных характеристик mask AUPRC; SD по seed доступен в исходном файле. `mean_auprc` усредняет 792 отдельных mask condition, а `exact_mask_auprc` относится к одной заранее заданной маске.",
        "",
        md_table(mask_summary),
        "",
    ]

    lines += [
        "## Парные bootstrap интервалы",
        "",
        "Первичная метрика — TWA AUPRC. Bootstrap синхронно пересэмплирует одни и те же test индексы для сравниваемых моделей и усредняет seed-specific различия внутри повтора. `full_cohort_point_difference` — оценка по полному test набору; интервал — 2.5–97.5 percentile из 1000 повторов.",
        "",
    ]
    b = bootstrap.copy()
    b["point"] = b.full_cohort_point_difference
    b["bootstrap mean"] = b.bootstrap_mean
    b["95% CI"] = b.apply(lambda r: f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]", axis=1)
    b = b[["cohort", "contrast", "condition", "point", "bootstrap mean", "95% CI"]]
    b.columns = ["Когорта", "Контраст (A−B)", "Условие", "Полный test", "Bootstrap mean", "95% CI"]
    lines += [md_table(b), ""]

    # Mixture data are intentionally linked as a complete table rather than reduced to a selected q.
    new_mix = mixtures[mixtures.arm.isin(["physics_p010", "physics_p025"]) & (mixtures.endpoint == "TWA")]
    mix_overview = new_mix.groupby(["arm", "q", "score_mode"], as_index=False)[["auprc", "auroc", "brier", "ece15"]].mean()
    lines += [
        "## Ningbo смеси clean/RA_LL",
        "",
        "Сохранена вся сетка `q_fault = 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1` для всех endpoint и пяти режимов в `MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv`. Таблица ниже показывает новые режимы для TWA, усреднение по seed; raw и calibrated значения разделены. Не выбирайте q по test как рабочий порог.",
        "",
        md_table(mix_overview.round(4)),
        "",
        "Для новых моделей Platt calibrator fitted только на прежней Ningbo validation A; threshold определён только на прежней validation B. Параметры и происхождение по каждой модели записаны в `CALIBRATION_PROVENANCE_NEW.json`. Исторические arms используют сохранённые calibrators/thresholds соответствующей фазы; raw и calibrated оценки хранятся отдельно.",
        "",
        "## Выполнение 12 запусков",
        "",
        "Все шесть Chapman запусков прошли frozen validation/integrity gate, затем были оценены; после этого так же завершены шесть Ningbo. Остановка каждого обучения — `patience_5`. Значение памяти — пиковая CUDA allocation по gate, а не системная RAM.",
        "",
    ]
    run_rows = []
    for cohort in ["Chapman", "Ningbo"]:
        for arm in ["physics_p010", "physics_p025"]:
            for seed in SEEDS:
                unit = ROOT / "training" / cohort / arm / f"seed_{seed}"
                gate = json.loads((unit / "validation_gate.json").read_text(encoding="utf-8"))
                eval_manifest = json.loads((unit / "evaluation_complete.json").read_text(encoding="utf-8"))
                run_rows.append({
                    "Когорта": cohort,
                    "Режим": arm,
                    "Seed": seed,
                    "Статус": gate["decision"],
                    "Лучшая эпоха": gate["best_epoch"],
                    "Val TWA AUPRC": gate["validation_twa_auprc"],
                    "Val TWA AUROC": gate["validation_twa_auroc"],
                    "Остановка": gate["stop_reason"],
                    "Время, ч": gate["elapsed_hours"],
                    "Пик GPU, MiB": gate["gpu_peak_bytes"] / (1024 * 1024),
                    "Условия inference": eval_manifest.get("conditions", 0) if isinstance(eval_manifest.get("conditions", 0), int) else len(eval_manifest.get("conditions", [])),
                    "Checkpoint": f"training/{cohort}/{arm}/seed_{seed}/best.pt",
                })
    lines += [md_table(pd.DataFrame(run_rows)), ""]

    lines += [
        "## Метод и ограничения сопоставимости",
        "",
        "Использованы прежние ECGResNet-small, labels, splits, train-derived normalization и фиксированный recipe: AdamW, lr 0.001, weight decay 0.0001, weighted multilabel BCE с positive-weight cap 20, batch 64, максимум 30 эпох, AMP, выбор checkpoint по clean-validation TWA AUPRC, patience 5. Swap применялся к raw сигналу до normalization. Новые режимы содержали только clean или один из трёх reversal swaps RA_LA/RA_LL/LA_LL; маски и циклы в train augmentation не добавлялись.",
        "",
        "Исторические результаты переиспользованы без перезапуска: ordinary, physics-aware 50%, combined и mask comparators взяты из проверенных сохранённых outputs, prediction caches и heldout-cycle пакета. Точные файлы-источники зафиксированы в `FROZEN_PROTOCOL.json`, `PROTOCOL_AMENDMENT_001..004.json`, `build_final_report.py` и `paired_bootstrap.py`. Поэтому это не полностью контролируемое повторное сравнение и не независимая репликация.",
        "",
        "Во время первоначального запуска первой Chapman p010 единицы завершились train и validation эпохи 1, но запись validation артефакта сорвалась из-за конфликта аргумента функции; checkpoint epoch не был зафиксирован и test inference не запускался. Попытка сохранена в `INTERRUPTED_ATTEMPTS.json`, исправление документировано Amendment 004; unit чисто повторён с исходным seed и учтён один раз среди 12 завершённых запусков.",
        "",
        "**Методический SHA256:** `49c3fdef06868f951aaf96803b42c31146e67362b0eb71290848501d6ec7e382`.\n",
        "**Runner SHA256:** `c6052fe7e0b4f27e93b24a509736b2231574392a3e06944bfe1e1646690befb2`.\n",
        "Engineering checks: PASS; validation gates: 12/12 PASS; test conditions: 796 на unit (4 фиксированных + 792 маски); mask records: 792 на unit. HEEDB/reserved test не оценивались; новые модели и подбор параметров не проводились.",
        "",
        "## Артефакты для аудита",
        "",
        "- Полные seed-specific метрики и исторические comparators: `METRICS_ALL_ARMS_BY_SEED.csv`.",
        "- Сводки по условию/endpoint: `METRICS_CHAPMAN_SUMMARY.csv`, `METRICS_NINGBO_SUMMARY.csv`.",
        "- Распределения 792 масок новых моделей: `MASK_DISTRIBUTION_CHAPMAN.csv`, `MASK_DISTRIBUTION_NINGBO.csv`.",
        "- Парные разности по seed и 1000-repeat intervals: `PAIRED_CONTRASTS_POINT_BY_SEED.csv`, `PAIRED_BOOTSTRAP_TWA_AUPRC.csv`.",
        "- Mixtures/calibration: `MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv`, `CALIBRATION_PROVENANCE_NEW.json`.",
        "- Все checkpoints, validation outputs, per-condition test predictions и integrity manifests: подкаталоги `training/` и `predictions/`.",
        "- Текущий file/hash inventory: `RUN_MANIFEST.json`.",
        "",
        "Отчёт фиксирует результаты для проверки. Дальнейшая оценка данных или новые эксперименты не запускались.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    units = []
    for cohort in ["Chapman", "Ningbo"]:
        for arm in ["physics_p010", "physics_p025"]:
            for seed in SEEDS:
                unit = ROOT / "training" / cohort / arm / f"seed_{seed}"
                gate = json.loads((unit / "validation_gate.json").read_text(encoding="utf-8"))
                eval_manifest = json.loads((unit / "evaluation_complete.json").read_text(encoding="utf-8"))
                units.append({
                    "cohort": cohort, "arm": arm, "seed": seed, "status": gate["decision"],
                    "best_epoch": gate["best_epoch"], "stop_reason": gate["stop_reason"],
                    "elapsed_hours": gate["elapsed_hours"], "gpu_peak_bytes": gate["gpu_peak_bytes"],
                    "validation_gate": unit.relative_to(ROOT).as_posix() + "/validation_gate.json",
                    "validation_gate_sha256": sha256(unit / "validation_gate.json"),
                    "checkpoint": unit.relative_to(ROOT).as_posix() + "/best.pt",
                    "checkpoint_sha256": sha256(unit / "best.pt"),
                    "evaluation_complete": unit.relative_to(ROOT).as_posix() + "/evaluation_complete.json",
                    "evaluation_complete_sha256": sha256(unit / "evaluation_complete.json"),
                    "condition_count": eval_manifest.get("conditions", 0) if isinstance(eval_manifest.get("conditions", 0), int) else len(eval_manifest.get("conditions", [])),
                })
    tracked = [
        "FROZEN_PROTOCOL.json", "PROTOCOL.md", "PROTOCOL_AMENDMENT_001.json", "PROTOCOL_AMENDMENT_002.json",
        "PROTOCOL_AMENDMENT_003.json", "PROTOCOL_AMENDMENT_004.json", "ENGINEERING_CHECKS.json", "INTERRUPTED_ATTEMPTS.json",
        "run_probability_study.py", "engineering_checks.py", "build_final_report.py", "analyze_ningbo_mixtures.py",
        "paired_bootstrap.py", "finalize_probability_study.py", "METRICS_CHAPMAN_NEW_BY_SEED.csv", "METRICS_NINGBO_NEW_BY_SEED.csv",
        "METRICS_ALL_ARMS_BY_SEED.csv", "METRICS_CHAPMAN_SUMMARY.csv", "METRICS_NINGBO_SUMMARY.csv",
        "METRICS_MEAN_SD_BY_ARM.csv", "PAIRED_CONTRASTS_POINT_BY_SEED.csv", "PAIRED_BOOTSTRAP_TWA_AUPRC.csv",
        "MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv", "MIXTURE_RESULTS_NEW_BY_SEED.csv", "CALIBRATION_PROVENANCE_NEW.json",
        "MASK_DISTRIBUTION_CHAPMAN.csv", "MASK_DISTRIBUTION_NINGBO.csv", "FINAL_PROBABILITY_STUDY_REPORT_RU.md",
    ]
    file_hashes = {f: {"sha256": sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in tracked}
    inv = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != MANIFEST.name):
        inv.append((path.relative_to(ROOT).as_posix(), sha256(path), path.stat().st_size))
    inv_hasher = hashlib.sha256()
    for rel, digest, size in inv:
        inv_hasher.update(f"{rel}\t{digest}\t{size}\n".encode("utf-8"))
    protocol = json.loads((ROOT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    manifest = {
        "study": "augmentation_probability_study_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study_root": ROOT_REL.as_posix(),
        "method_sha256": protocol.get("protocol_sha256", "49c3fdef06868f951aaf96803b42c31146e67362b0eb71290848501d6ec7e382"),
        "runner_sha256": sha256(ROOT / "run_probability_study.py"),
        "conditions_per_unit": 796,
        "mask_count_per_unit": 792,
        "bootstrap": {"file": "PAIRED_BOOTSTRAP_TWA_AUPRC.csv", "rows": len(bootstrap), "replicates_per_row": 1000, "all_numeric_finite": bool(bootstrap.select_dtypes("number").notna().all().all())},
        "units": units,
        "tracked_artifacts": file_hashes,
        "inventory": {"file_count": len(inv), "algorithm": "sha256 of sorted UTF-8 lines: relative_path<TAB>file_sha256<TAB>bytes<LF>", "inventory_sha256": inv_hasher.hexdigest()},
        "scope": {"training_runs_completed": 12, "validation_gates_passed": sum(u["status"] == "PASS" for u in units), "cohorts_order": ["Chapman", "Ningbo"], "seeds": SEEDS, "HEEDB_evaluated": False, "reserved_test_evaluated": False},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "manifest": str(MANIFEST), "units": len(units), "gates_passed": manifest["scope"]["validation_gates_passed"], "bootstrap_rows": len(bootstrap), "inventory_files": len(inv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
