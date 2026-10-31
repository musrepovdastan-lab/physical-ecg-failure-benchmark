from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
METRICS = OUT / "METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(METRICS)
    value_cols = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "accuracy", "threshold"]
    grouped = df.groupby(["cohort", "arm", "condition", "endpoint"], sort=True)[value_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(OUT / "METRICS_SUMMARY_ALL_ARMS.csv", index=False, float_format="%.12g")

    mask = df[df.condition.str.startswith("mask5_")]
    mask_seed = mask.groupby(["cohort", "arm", "seed", "endpoint"], as_index=False)[value_cols].mean()
    mask_summary = mask_seed.groupby(["cohort", "arm", "endpoint"], as_index=False)[value_cols].agg(["mean", "std"])
    mask_summary.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in mask_summary.columns]
    mask_summary = mask_summary.reset_index(drop=True)
    mask_summary.to_csv(OUT / "MEAN_792_MASK_METRICS_ALL_ARMS.csv", index=False, float_format="%.12g")
    return grouped, mask_summary


def main() -> None:
    summary, mask_summary = write_summaries()
    auprc = pd.read_csv(OUT / "PAIRED_BOOTSTRAP_TWA_AUPRC_PRESERVED.csv")
    other = pd.read_csv(OUT / "PAIRED_BOOTSTRAP_TWA_AUROC_BRIER.csv")
    intervals = pd.concat([auprc, other], ignore_index=True, sort=False)
    intervals.to_csv(OUT / "ALL_PAIRED_BOOTSTRAP_INTERVALS.csv", index=False, float_format="%.12g")

    metrics = pd.read_csv(METRICS)
    common = metrics[metrics.condition.isin(["clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201"]) ]
    required_metric_cols = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction", "accuracy", "threshold"]
    missing = int(common[required_metric_cols].isna().sum().sum())
    if missing:
        raise RuntimeError(f"Common fixed-condition table has {missing} missing cells")
    fixed_rows = summary[summary.condition.isin(["clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201"])].copy()
    if len(fixed_rows) != 2 * 5 * 4 * 3:
        raise RuntimeError(f"Unexpected fixed-condition summary row count: {len(fixed_rows)}")
    if len(intervals) != 128 or int(auprc.shape[0]) != 48 or int(other.shape[0]) != 80:
        raise RuntimeError("Unexpected paired bootstrap row count")

    validation = pd.read_csv(OUT / "VALIDATION_RESTORATION.csv")
    raw_audit = json.loads((OUT / "RAW_TEST_METRIC_REPRODUCTION_AUDIT.json").read_text(encoding="utf-8"))
    validation_audit = json.loads((OUT / "VALIDATION_RESTORATION_AUDIT.json").read_text(encoding="utf-8"))
    boot_audit = json.loads((OUT / "BOOTSTRAP_RECOVERY_AUDIT.json").read_text(encoding="utf-8"))
    hist_audit = json.loads((OUT / "HISTORICAL_COMPARATOR_COMPLETENESS_AUDIT.json").read_text(encoding="utf-8"))
    cal_audit = json.loads((OUT / "NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
    supplement = json.loads((OUT / "COMPARATOR_COMPLETENESS_SUPPLEMENT.json").read_text(encoding="utf-8"))

    def md_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
        view = frame[columns].copy()
        for c in view.columns:
            if pd.api.types.is_numeric_dtype(view[c]):
                view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        headers = [str(c) for c in view.columns]
        rows = [[str(v) for v in row] for row in view.itertuples(index=False, name=None)]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    order = ["ordinary", "physics_p010", "physics_p025", "physics_p050", "combined"]
    clean = fixed_rows[(fixed_rows.condition == "clean") & (fixed_rows.endpoint == "TWA")].copy()
    clean["arm"] = pd.Categorical(clean.arm, order, ordered=True)
    clean = clean.sort_values(["cohort", "arm"])
    clean_table = md_table(clean, ["cohort", "arm", "auprc_mean", "auprc_std", "auroc_mean", "auroc_std", "brier_mean", "ece15_mean", "sensitivity_mean", "specificity_mean", "predicted_positive_fraction_mean", "accuracy_mean"])
    reversal = fixed_rows[(fixed_rows.condition == "RA_LL_reversal") & (fixed_rows.endpoint == "TWA")].copy()
    reversal["arm"] = pd.Categorical(reversal.arm, order, ordered=True)
    reversal = reversal.sort_values(["cohort", "arm"])
    reversal_table = md_table(reversal, ["cohort", "arm", "auprc_mean", "auroc_mean", "brier_mean", "ece15_mean", "sensitivity_mean", "specificity_mean", "predicted_positive_fraction_mean", "accuracy_mean"])
    mask_twa = mask_summary[mask_summary.endpoint == "TWA"].copy()
    mask_twa["arm"] = pd.Categorical(mask_twa.arm, order, ordered=True)
    mask_twa = mask_twa.sort_values(["cohort", "arm"])
    mask_table = md_table(mask_twa, ["cohort", "arm", "auprc_mean", "auprc_std", "auroc_mean", "brier_mean", "ece15_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"])

    sel = intervals[(intervals.metric == "AUPRC") & (intervals.condition.isin(["clean", "mean_792_masks"]))]
    sel = sel[sel.contrast.isin(["physics_p025_minus_combined", "physics_p025_minus_physics_aware", "physics_p010_minus_ordinary", "physics_p010_minus_physics_aware"])]
    ci_table = md_table(sel, ["cohort", "contrast", "condition", "full_cohort_point_difference", "bootstrap_mean", "ci95_low", "ci95_high"], 4)
    auroc_brier = intervals[(intervals.metric.isin(["AUROC", "raw Brier"])) & intervals.condition.isin(["clean", "RA_LL_reversal", "mask5_0_1_2_3_5"])].copy()
    auroc_brier_table = md_table(auroc_brier, ["cohort", "contrast", "condition", "metric", "full_cohort_point_difference", "bootstrap_mean", "ci95_low", "ci95_high"], 4)

    val_table = md_table(validation, ["cohort", "arm", "seed", "best_epoch", "last_epoch", "old_saved_validation_twa_auprc", "restored_best_checkpoint_twa_auprc", "validation_gate_twa_auprc", "restored_best_checkpoint_twa_auroc", "validation_gate_twa_auroc"], 6)

    # Calibrated-v1 changes are summarized across seed/endpoints for each q.
    changes = pd.read_csv(OUT / "NINGBO_MIXTURE_CHANGE_SUMMARY.csv")
    cal = changes[changes.score_mode == "calibrated"].copy()
    cal_rows = cal[cal.q.isin([0.0, 0.1, 1.0])]
    cal_table = md_table(cal_rows, ["arm", "q", "delta_brier", "delta_ece15", "delta_sensitivity", "delta_specificity", "delta_accuracy", "delta_threshold"], 5)

    # Key paired TWA AP rows for the main questions; the complete 128-row file is adjacent.
    report = f"""# Восстановленный отчёт: augmentation probability study

Пакет восстановления: `outputs/augmentation_probability_recovery_v2/`. Это постфактум-восстановление по уже обученным моделям и сохранённым test probabilities. Оно не было зарегистрировано до первоначального просмотра test. Исходные данные и артефакты v1 не перезаписывались.

## Объём и метод

- 12 уже обученных моделей: Chapman/Ningbo × physics_p010/physics_p025 × seeds 104729, 130363, 155921. Новых обучений и полных повторных прогонов 792 масок не было.
- Для каждого запуска восстановлена validation prediction по `best.pt`; входные predictions, y и record IDs записаны в `validation/`. Первичный протокол и SHA256 зафиксированы в `RECOVERY_PROTOCOL.json` (отпечаток протокола: `{protocol['recovery_protocol_sha256']}`). Дополнительный frozen supplement исторических cached predictions: `{supplement['supplement_sha256']}`.
- Протокол явно помечен постфактум: первичный test review уже состоялся. Supplement создан позднее, до повторного заполнения исторических comparator-полей.
- Численный допуск для validation TWA AUPRC/AUROC заранее установлен как 1e-6. Применённый порядок записей, split, preprocessing и inference соответствуют сохранённым данным и контрольным артефактам.

## Исправление validation

Все 12 `best.pt` state_dict побитно совпали с `checkpoint.pt`, использованным для test inference. Восстановленные TWA AUPRC/AUROC совпали с `validation_gate.json` во всех 12 случаях; максимальная абсолютная разница 0. Старый `validation.npz` действительно содержал метрики последней эпохи, тогда как тестовые предсказания были получены из best checkpoint.

| Cohort | Arm | Seed | Best epoch | Last epoch | Старый AP | Восстановленный AP | Gate AP | Восстановленный AUROC | Gate AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{val_table}

Проверка небольшого набора test probabilities для каждого checkpoint прошла: 24 сравнения (clean и RA_LL для 12 моделей) совпали в пределах фиксированного same-batch контекста. Первичная проба с отдельным batch из 16 записей показала разницу 0.00048828125 при допуске 1e-6; диагностика установила зависимость AMP вычислений от размера batch. Поэтому amendment сохранил заранее выбранные индексы и допуск, а повтор воспроизводит исходный batch размером 128; подробности — `RECOVERY_AMENDMENT_001.json`.

## Test-метрики и пороги

Для новых моделей пороги пересчитаны из отрицательных clean validation scores по правилу линейного 90-го процентиля, отдельно для каждой модели и endpoint. Пересчитаны sensitivity, specificity, predicted-positive fraction и accuracy для всех 796 условий × 3 endpoint × 12 моделей = 28 656 строк. Raw AUPRC/AUROC/Brier/ECE оставлены из v1 после проверки SHA256 всех файлов предсказаний и канонического порядка записей; максимальное отличие от v1 равно 0. Исторические пороги существующих comparator-моделей оставлены прежними, а отсутствовавшие comparator поля восстановлены из cached predictions без inference. Для всех общих фиксированных условий в таблице нет пропусков; 54 поля accuracy по двум дополнительным Ningbo условиям RA_LA/LA_LL остаются недоступны из-за отсутствия сохранённых исторических предсказаний для этих условий.

Полная summary-таблица по всем пяти arm, четырём фиксированным условиям и трём endpoint: `METRICS_SUMMARY_ALL_ARMS.csv` (mean и SD по seed). Полная строковая таблица по всем 796 условиям на каждый доступный arm: `METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv`. Файлы не содержат специально восстановленных/выдуманных значений для отсутствующих исторических предсказаний.

### TWA, clean

Пороговые показатели относятся к сохранённому порогу соответствующей модели: для новых arm — к восстановленному clean validation q90, для исторических comparator — к исходному native validation threshold.

{clean_table}

### TWA, RA_LL reversal

{reversal_table}

## Среднее по 792 маскам

Значения — сначала среднее AP по 792 маскам внутри каждого seed, затем mean и SD по трём seeds. Полная таблица всех метрик и endpoint находится в `MEAN_792_MASK_METRICS_ALL_ARMS.csv`.

{mask_table}

Для TWA разница AUPRC `p025 − combined` составила −0.047445 на Chapman и −0.037691 на Ningbo. Парные 95% bootstrap CI соответственно [−0.058512, −0.037227] и [−0.044864, −0.030394]. Здесь отрицательное значение означает ниже AUPRC у p025. Это сравнение двух обученных стратегий по зафиксированному набору масок; оно само по себе не изолирует причинный эффект маски.

## Парные интервалы

Сохранены прежние 48 TWA AUPRC интервалов и добавлены 80 TWA AUROC/raw Brier интервалов (1000 синхронизированных bootstrap выборок; seeds усреднены внутри каждой реплики). Full-cohort point estimate приведён отдельно от bootstrap mean. Для разницы Brier `arm_a − arm_b` отрицательное значение означает ниже Brier у arm_a. Полный набор из 128 групп находится в `ALL_PAIRED_BOOTSTRAP_INTERVALS.csv`.

### AUPRC: clean и среднее по 792 маскам

{ci_table}

### AUROC и raw Brier: clean, RA_LL и одна заранее заданная маска

{auroc_brier_table}

Для p050 против новых p010/p025 на чистом Ningbo AUPRC интервалы разницы `p025 − p050` и `p010 − p050` не включают ноль: соответственно −0.009299 [−0.017904, −0.001550] и −0.012707 [−0.023890, −0.002316]. Это наблюдение по одной метрике и одной когорте; другие метрики не дают основания объявлять общую оптимальную вероятность или эквивалентность вариантов.

## Ningbo calibration и смеси

Platt-модели переоценены для шести новых Ningbo запусков и трёх endpoint на validation A; пороги рассчитаны по raw scores отрицательных примеров B, а calibrated threshold получен преобразованием raw threshold. Использованы прежние split indices, clipping и настройки L-BFGS-B. Тестовая сетка `q_fault`: 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1; полный результат по raw/calibrated режимам — `NINGBO_MIXTURES_RECOVERED_BY_SEED.csv`.

Есть важное ограничение: полный validation ранее участвовал в выборе checkpoint до разделения A/B. Поэтому A/B не являются независимыми от отбора модели и калибровка имеет утечку через checkpoint selection. На clean и RA_LL raw/calibrated бинарные решения совпали во всех 18 проверках; ties и clipping отдельно перечислены в `NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json`.

Изменение средних calibrated показателей относительно ошибочного v1 для q=0, 0.1 и 1:

{cal_table}

Raw test ranking/probability metrics для новых arm не менялись; изменения calibrated Brier/ECE и пороговых метрик связаны с исправленным validation, Platt fit и порогом. Тест не использовался для настройки параметров.

## Ответы на вопросы

1. **Что из v1 подтвердилось?** Сами новые модели, test probabilities, порядок записей и raw AUPRC/AUROC/Brier/ECE: контрольные хеши предсказаний подтверждены, raw метрики совпали с v1 (максимальная разница 0). Восстановленные test sample probabilities совпали с кешем при исходном контексте batch.
2. **Что изменилось после исправления?** Validation перешла с последней эпохи на best checkpoint; из-за этого изменились новые validation-derived пороги, пороговые метрики и Ningbo Platt calibration/mixtures. Старые значения оставлены в v1 и перечислены в `SUPERSEDED_ARTIFACTS.csv`.
3. **Помогают ли маски при одинаковой доле перестановок?** В среднем по 792 маскам TWA AUPRC ниже у p025, чем combined, на 0.0474 для Chapman и 0.0377 для Ningbo; парные интервалы не пересекают ноль. Это поддерживает разницу между этими arms в данной оценке, но не является самостоятельным доказательством причинного эффекта маски.
4. **Есть ли выигрыш на clean от снижения вероятности augmentation?** Убедительного общего выигрыша p010/p025 над p050 по всем метрикам и когортам нет. На clean Ningbo TWA AUPRC p050 выше обоих новых arm с указанными выше интервалами; это не доказывает общего превосходства p050. Интервал, включающий ноль, не трактуется как равенство.
5. **Готов ли пакет к следующему архитектурному тесту?** Да, как восстановленный и проверяемый пакет для планирования следующего теста. Он не даёт основания заявлять клиническое превосходство или независимое подтверждение; calibration A/B ограничен отбором checkpoint по полному validation. Новые обучения, HEEDB, Founder, reserved_test и работа над статьёй не выполнялись.

## Происхождение и заменённые результаты

- `RECOVERY_PROTOCOL.json`, `RECOVERY_AMENDMENT_001.json`, `COMPARATOR_COMPLETENESS_SUPPLEMENT.json` фиксируют протокол, поправку по batch-dependent replay и дополнительный пересчёт исторических comparator.
- `RECOVERY_MANIFEST.json` содержит SHA256 и размер всех файлов пакета, кроме самого manifest.
- `SUPERSEDED_ARTIFACTS.csv` перечисляет ошибочные v1 артефакты и соответствующие замены в этом каталоге. Файлы v1 сохранены без изменений.
- `VALIDATION_RESTORATION_AUDIT.json`, `RAW_TEST_METRIC_REPRODUCTION_AUDIT.json`, `HISTORICAL_COMPARATOR_COMPLETENESS_AUDIT.json`, `BOOTSTRAP_RECOVERY_AUDIT.json` содержат машинно-читаемые проверки.
"""
    (OUT / "FINAL_PROBABILITY_STUDY_REPORT_RECOVERED_RU.md").write_text(report, encoding="utf-8")

    superseded = pd.DataFrame([
        {"v1_artifact": "outputs/augmentation_probability_study_v1/{cohort}/{arm}/seed_{seed}/validation.npz", "issue": "contains last-epoch validation predictions although test inference used best checkpoint", "replacement": "outputs/augmentation_probability_recovery_v2/validation/{cohort}/{arm}/seed_{seed}.npz"},
        {"v1_artifact": "outputs/augmentation_probability_study_v1/METRICS_CHAPMAN_NEW_BY_SEED.csv and METRICS_NINGBO_NEW_BY_SEED.csv threshold-derived fields", "issue": "threshold metrics used last-epoch validation scores", "replacement": "outputs/augmentation_probability_recovery_v2/METRICS_NEW_RECOVERED_BY_SEED.csv"},
        {"v1_artifact": "outputs/augmentation_probability_study_v1 Ningbo new-arm Platt fits, thresholds, and mixtures", "issue": "calibration used last-epoch validation probabilities", "replacement": "outputs/augmentation_probability_recovery_v2/NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json and NINGBO_MIXTURES_RECOVERED_BY_SEED.csv"},
        {"v1_artifact": "outputs/augmentation_probability_study_v1/FINAL_PROBABILITY_STUDY_REPORT_RU.md", "issue": "reports validation-dependent outputs from the wrong epoch and misstates mean-mask comparison", "replacement": "outputs/augmentation_probability_recovery_v2/FINAL_PROBABILITY_STUDY_REPORT_RECOVERED_RU.md"},
        {"v1_artifact": "outputs/augmentation_probability_study_v1/METRICS_ALL_ARMS_BY_SEED.csv historical comparator missing fields", "issue": "some historical cached-prediction threshold metrics were omitted", "replacement": "outputs/augmentation_probability_recovery_v2/METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv"},
    ])
    superseded.to_csv(OUT / "SUPERSEDED_ARTIFACTS.csv", index=False)

    # Record a compact final provenance summary, then hash the complete package.
    manifest_path = OUT / "RECOVERY_MANIFEST.json"
    file_rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        file_rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "package": "augmentation_probability_recovery_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FOR_REVIEW",
        "scope": {"new_training": False, "new_test_model_inference": False, "full_792_mask_inference": False, "models_recovered": 12, "new_model_metric_rows": 28656, "paired_bootstrap_intervals": 128},
        "recovery_protocol_sha256": protocol["recovery_protocol_sha256"],
        "comparator_supplement_sha256": supplement["supplement_sha256"],
        "input_files_hashed_in_primary_protocol": protocol["original_input_count"],
        "files": file_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary), "mask_summary_rows": len(mask_summary), "fixed_condition_summary_rows": len(fixed_rows), "missing_common_condition_cells": missing, "bootstrap_intervals": len(intervals), "manifest_files": len(file_rows), "manifest_sha256": sha256(manifest_path), "report": str(OUT / "FINAL_PROBABILITY_STUDY_REPORT_RECOVERED_RU.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
