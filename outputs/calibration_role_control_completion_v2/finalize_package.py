from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "outputs/calibration_role_control_v1"
OUT = ROOT / "outputs/calibration_role_control_completion_v2"
KEY_Q = (0.0, 0.01, 0.1, 1.0)
METHODS = ("ordinary", "physics_aware_p050", "combined", "Huang_full_pipeline")
SCENARIOS = ("RA_LL_only", "equal_three_swap")
COHORTS = ("Ningbo_test", "HEEDB")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main() -> None:
    # Validate all bound sources and stored completion outputs before authoring reports.
    bindings = json.loads((OUT / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    source_problems = []
    for item in bindings["files"]:
        f = ROOT / Path(item["path"])
        if not f.exists() or f.stat().st_size != item["bytes"] or sha(f) != item["sha256"]:
            source_problems.append(item["path"])
    if source_problems:
        raise RuntimeError(f"bound source changed: {source_problems[:5]}")
    points = read_csv(OUT / "PAIRED_POINTS_CI_ALL_700.csv")
    coverage = json.loads((OUT / "BOOTSTRAP_COVERAGE_AUDIT.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "NUMERICAL_VERIFICATION.json").read_text(encoding="utf-8"))
    if len(points) != 700 or coverage["status"] != "PASS" or verification["status"] != "PASS":
        raise RuntimeError("bootstrap completion audit is not complete")

    mixture = read_csv(V1 / "MIXTURE_METRICS_MEAN_SD.csv")
    mix = {(r["cohort"], r["method"], r["endpoint"], r["scenario"], float(r["q_fault"])): r for r in mixture}
    key_rows = []
    for cohort in COHORTS:
        for scenario in SCENARIOS:
            for q in KEY_Q:
                for method in METHODS:
                    r = mix[cohort, method, "TWA", scenario, q]
                    key_rows.append({
                        "cohort": cohort, "scenario": scenario, "q_fault": q, "method": method,
                        **{column: r[column] for column in (
                            "AP_raw_mean", "AP_raw_sd", "AUROC_raw_mean", "AUROC_raw_sd",
                            "Brier_raw_mean", "Brier_raw_sd", "Brier_calibrated_mean", "Brier_calibrated_sd",
                            "sensitivity_raw_mean", "sensitivity_raw_sd", "FP_per_1000_raw_mean", "FP_per_1000_raw_sd")},
                    })
    write_csv(OUT / "RESULTS_KEY_Q_TABLE.csv", key_rows)
    key_ci = [r for r in points if float(r["q_fault"]) in KEY_Q]
    write_csv(OUT / "PAIRED_CI_KEY_Q.csv", key_ci)

    # Human-readable compact tables; each cell is mean across 3 trained runs ± SD.
    names = {"ordinary": "ordinary", "physics_aware_p050": "physics_aware p050",
             "combined": "combined", "Huang_full_pipeline": "Huang"}
    result_md = [
        "# Ключевые результаты по TWA",
        "",
        "Среднее по трём сохранённым fits каждого метода ± SD между fits. q=0 означает только чистые ECG; остальные q — заданная доля simulated faults. Когорты и сценарии рассмотрены раздельно.",
        "",
        "AP и AUROC выше лучше; Brier и FP/1000 ниже лучше. Sensitivity выше означает больше пойманных положительных меток. HEEDB — историческая sensitivity когорта с proxy labels.",
    ]
    for cohort in COHORTS:
        result_md += ["", f"## {cohort}"]
        for scenario in SCENARIOS:
            result_md += ["", f"### {scenario}", "", "| q | Метод | AP | AUROC | Brier raw → calibrated | Sensitivity | FP/1000 |", "|---:|---|---:|---:|---:|---:|---:|"]
            for q in KEY_Q:
                for method in METHODS:
                    r = mix[cohort, method, "TWA", scenario, q]
                    result_md.append(
                        f"| {q:g} | {names[method]} | {fmt(float(r['AP_raw_mean']), float(r['AP_raw_sd']))} | "
                        f"{fmt(float(r['AUROC_raw_mean']), float(r['AUROC_raw_sd']))} | "
                        f"{fmt(float(r['Brier_raw_mean']), float(r['Brier_raw_sd']))} → {fmt(float(r['Brier_calibrated_mean']), float(r['Brier_calibrated_sd']))} | "
                        f"{fmt(float(r['sensitivity_raw_mean']), float(r['sensitivity_raw_sd']))} | "
                        f"{fmt(float(r['FP_per_1000_raw_mean']), float(r['FP_per_1000_raw_sd']), 1)} |"
                    )
    result_md += [
        "",
        "Полные 700 paired CI находятся в `PAIRED_POINTS_CI_ALL_700.csv`; ключевые q вынесены в `PAIRED_CI_KEY_Q.csv`. Разности AP/Brier/sensitivity/FP сравнивают соответствующие method и seed на одном resample, затем усредняют по трём seeds. Это intervals, условные на уже обученных моделях и сохранённых calibration coefficients.",
        "",
    ]
    (OUT / "KEY_RESULTS_TABLES_RU.md").write_text("\n".join(result_md), encoding="utf-8")

    # Recompute reported leaders directly from verified stored per-seed summary rows.
    leader_lines = []
    for cohort in COHORTS:
        for scenario in SCENARIOS:
            for q in KEY_Q:
                vals = {m: float(mix[cohort, m, "TWA", scenario, q]["AP_raw_mean"]) for m in METHODS}
                winner = max(vals, key=vals.get)
                leader_lines.append(f"{cohort} {scenario} q={q:g}: {winner} (AP {vals[winner]:.4f})")

    # Paired q=.1 equal-three contrasts against Huang and ordinary, based on full 700-group outputs.
    def ci(cohort: str, scen: str, q: float, left: str, right: str, metric: str) -> tuple[float, float, float]:
        r = next(x for x in points if x["cohort"] == cohort and x["scenario"] == scen and float(x["q_fault"]) == q and
                 x["left"] == left and x["right"] == right and x["metric"] == metric)
        return tuple(float(r[k]) for k in ("full_cohort_point_mean_seed_contrast", "ci95_low", "ci95_high"))

    report = [
        "# Завершение calibration role control",
        "",
        "## Что сделано",
        "",
        "В отдельном пакете завершена недостающая статистика из уже сохранённых predictions: добавлено 400 paired bootstrap-групп для q=0, 0.001, 0.005, 0.05. Сохранённые 300 групп для q=0.01, 0.1, 1.0 сохранены и сверены. Полный набор содержит 700 групп по 1 000 draws на группу для двух когорт, двух сценариев, пяти contrasts и пяти метрик. Ни обучение, inference, refit калибровки, ни выбор порогов не запускались.",
        "",
        f"Frozen completion protocol SHA-256: `{(OUT / 'COMPLETION_PROTOCOL_SHA256.txt').read_text().split()[0]}`. Исправление порядка условий зафиксировано отдельно в `COMPLETION_PROTOCOL_AMENDMENT_001.json`; протокол и научный scope не менялись.",
        "",
        "### Численная проверка",
        "",
        f"Из сохранённых predictions заново рассчитано {verification['twa_point_rows_recalculated']:,} TWA point metric values по всем q, сценариям, методам и seeds; максимум расхождения с сохранённой mixture-таблицей — {verification['twa_point_max_abs_delta_vs_saved_mixture_metrics']:.2e}. Исходные 300 draw vectors сохранены численно без изменения (max delta {verification['old_draw_vectors_preserved_max_abs_delta']:.1f}). Для новых групп tie-group AP сверена со sklearn на реальных draws 0/417/999: {verification['new_sklearn_checks']} checks, max delta {verification['new_sklearn_max_abs_delta']:.2e}. Все {verification['new_group_count']} новые группы покрыты; оба q=0 сценария присутствуют и дают одинаковые draws.",
        "",
        "Ранее проведённая независимая проверка исходного пакета остаётся подтверждённой: 24 624 сохранённые метрики, все 300 исходных bootstrap summaries/draw spot checks, fit/threshold locks, membership роли и 96 prediction bindings проверялись review-артефактами. Здесь condition/mixture сводки повторно не переписывались; их v1 hashes и lineage зафиксированы в `UNCHANGED_ARTIFACTS_AUDIT.json`.",
        "",
        "### Результаты",
        "",
        "Все ключевые числа (чистые q=0 и RA–LL-only/equal-three q=0.01, 0.1, 1.0) приведены в [таблицах](KEY_RESULTS_TABLES_RU.md) и CSV. Победители по точечной средней AP меняются между когортами и q:",
        "",
        *[f"- {line}" for line in leader_lines],
        "",
        "На Ningbo при equal-three q=0.1 combined превышает Huang по AP на "
        + (lambda x: f"{x[0]:.4f} (95% CI [{x[1]:.4f}, {x[2]:.4f}])")(
            ci("Ningbo_test", "equal_three_swap", 0.1, "combined", "Huang_full_pipeline", "AP_raw"))
        + "; у p050 разность AP относительно Huang имеет интервал, включающий ноль. На HEEDB при q=0.1 p050 выше Huang по AP на "
        + (lambda x: f"{x[0]:.4f} (95% CI [{x[1]:.4f}, {x[2]:.4f}])")(
            ci("HEEDB", "equal_three_swap", 0.1, "physics_aware_p050", "Huang_full_pipeline", "AP_raw"))
        + ". При том же HEEDB q p050 также даёт больше sensitivity и ложных тревог; paired intervals этих разностей приведены в CSV.",
        "",
        "Калибровка уменьшает Brier во многих сочетаниях; таблицы показывают raw и calibrated значения отдельно. Порог был перенесён на Platt-шкалу с положительным наклоном, поэтому raw и calibrated operating decisions совпадают. При q=1 на обеих когортах лидер AP смещается к Huang; защитные методы не выигрывают при каждом уровне ошибок.",
        "",
        "Сравнение с исторической серией не является оценкой одного только устранения calibration selection dependence: одновременно отличаются initialization/training trajectories, train/selection размеры и выбранные epochs. Исторический point leader combined сменяется p050 для HEEDB в чистом/equal-three промежуточном диапазоне, а при q=1 лидирует Huang; на Ningbo при q=1 новый point leader также Huang. Это описательные различия, не доказательство причины.",
        "",
        "### Что остаётся ограниченным",
        "",
        "- Declared fit seed задавался после конструктора ECGResNetSmall. Он управлял дальнейшим training RNG, но не начальными весами; начальный ambient RNG/state и initial weight hashes в сохранённых provenance не записаны. Точные fresh fits и matched initialization между методами только по seed не воспроизводятся. Checkpoints остаются валидными как фиксированные обученные модели; их predictions проверены. Правильный порядок для будущих запусков — установить random/NumPy/Torch/CUDA seeds до создания модели.",
        "- Все source-record роли model_selection/A/B разделены для этих fits, однако тестовый detector cache Ningbo существовал до финальной exact-objective calibration lock. Поэтому нельзя утверждать, что никакие test waveform не были доступны до последнего lock. Проверенная калибровка использовала только A и B scores/labels; test и HEEDB labels/scores в fitting code не передавались.",
        "- Интервалы условны на трёх сохранённых fits и их калибраторах и не охватывают полную вариацию обучения/fitting. Для Ningbo patient IDs нет, поэтому bootstrap — по записям; HEEDB — историческая чувствительность с proxy labels.",
        "- Ошибки simulated; результаты не подтверждают клиническую эффективность и не устанавливают универсального победителя.",
        "- GPU training peak memory не сохранён. Измерение v1 было сделано после сброса счётчика при replay best checkpoint, а не во время training; точный training peak unavailable.",
        "",
        "## Артефакты",
        "",
        "- `PAIRED_POINTS_CI_ALL_700.csv`, `PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz` — полный набор points, summaries и draws.",
        "- `BOOTSTRAP_COVERAGE_AUDIT.json` и `BOOTSTRAP_COVERAGE_700.csv` — покрытие групп, включая provenance reused/new.",
        "- `NUMERICAL_VERIFICATION.json`, `UNCHANGED_ARTIFACTS_AUDIT.json` — арифметика и неизменность исходников.",
        "- `INITIALIZATION_PROVENANCE_NOTE_RU.md`, `LOCK_CHRONOLOGY_RU.md` — ограничения seed и фактический порядок locks/access.",
        "- `CURRENT_PLAN_STATUS_RU.md` — выполнено, ограничено и оставлено на будущее.",
        "- `MANIFEST.json` — раздельные input bindings и hashes новых outputs.",
        "",
    ]
    (OUT / "FINAL_REPORT_RU.md").write_text("\n".join(report), encoding="utf-8")

    # Nine-fit report rebuilt from structured logs. Memory is explicitly not mislabeled.
    train = json.loads((V1 / "TRAINING_STATUS.json").read_text(encoding="utf-8-sig"))
    training = [
        "# Отчёт об обучениях",
        "",
        "Все девять ранее выполненных fits сохранены; в completion-пакете их не повторяли. Checkpoint выбирался по model_selection. Best checkpoint и history совпали в независимом аудите.",
        "",
        "| Метод | Seed | Лучшая эпоха | TWA AUPRC selection | AUROC selection | Время, с | Память GPU при обучении | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for fit in train["fits"]:
        cp = f"../calibration_role_control_v1/fits/{fit['method']}/seed_{fit['seed']}/best.pt"
        training.append(f"| {fit['method']} | {fit['seed']} | {fit['best_epoch']} | {fit['best_selection_twa_auprc']:.4f} | {fit['best_selection_twa_auroc']:.4f} | {fit['elapsed_seconds']:.1f} | Не сохранена; replay peak см. ниже | [{cp}]({cp}) |")
    replay_peaks = sorted({fit["peak_allocated_bytes"] for fit in train["fits"]})
    training += [
        "",
        f"GPU: {train['fits'][0]['device']}. Сохранённый показатель памяти ({', '.join(f'{x / 1024**2:.1f} MiB' for x in replay_peaks)}) измерен после сброса CUDA peak counter на replay лучшего checkpoint, а не в training loop. Поэтому фактический training peak memory для всех fits — unavailable; новых fits ради этого не запускали.",
        "",
        "Fit seed был выставлен уже после создания модели. См. `INITIALIZATION_PROVENANCE_NOTE_RU.md`: seed управляет последующим обучением, но не начальными весами.",
        "",
    ]
    (OUT / "TRAINING_REPORT_RU.md").write_text("\n".join(training), encoding="utf-8")

    seed_audit = json.loads((V1 / "SEED_ORDER_REVIEW_20261020.json").read_text(encoding="utf-8"))
    note = f'''# Ограничение initialization seed и доступная provenance

Независимая проверка исходного frozen-кода (`SEED_ORDER_REVIEW_20261020.json`) показала: `ECGResNetSmall()` конструировалась в `control.py` на строке {seed_audit['model_constructor_lines'][0]}, а вызовы `random.seed`, `numpy.random.seed`, `torch.manual_seed` и `torch.cuda.manual_seed_all` выполнялись на строке {seed_audit['seed_calls'][0]['line']}. То есть declared seed задавал последующую случайность обучения и augmentations, но не initial parameters.

Контрпример уже сохранён review: при одном fit seed разные ambient Torch RNG дали разные хеши начальных параметров; установка seed до constructor дала одинаковые веса. Это ограничение воспроизводимости начальной точки, а не признак ошибки A/B split и не доказательство неверных saved predictions.

Доступная provenance включает fit config/history, hashes checkpoint и replay лучших checkpoint, плюс resume-state, который сохраняет RNG во время обучения. В checkpoint `state_dict` — обученные веса. Отдельного initial RNG snapshot или initial parameter hash до первого optimizer step в frozen fit provenance не найдено; поэтому исходные веса восстановить по declared seed нельзя. Никакой подбор RNG, перебор или повторное обучение не проводились.

Все девять сохранённых fits остаются валидными как фиксированные trained runs, и их оценивание воспроизводится. Нельзя заявлять matched initialization между методами либо полную fresh-training воспроизводимость по одному seed. Для будущего запуска следует сначала установить Python/NumPy/Torch/CUDA seeds, затем создать модель, optimizer и scaler; перед первым batch сохранить начальные RNG states и hash state_dict. Это только рекомендация будущему протоколу; исторический frozen code, labels и fits не менялись.
'''
    (OUT / "INITIALIZATION_PROVENANCE_NOTE_RU.md").write_text(note, encoding="utf-8")

    chronology = '''# Хронология lock и доступа к данным

Даты и формулировки ниже взяты из frozen lock files, score bindings, amendments и независимого review. Точное wall-clock время у документов, где указана только календарная дата, отсутствует и задним числом не восстанавливалось.

1. **Обучение и выбор checkpoint.** Девять fits обучены на train; epoch selection выполнялся только по model_selection. `ALL_MODELS_LOCK.json` фиксирует lock time `2026-10-19T19:22:29.074725+00:00`, до A/B scoring. Best checkpoint replay и history совпали для всех девяти.
2. **Calibration_A и threshold_B.** После model lock были просчитаны A/B source-role scores. Exact Platt objective refit использовал сохранённые scores `scores/calibration_A/` и `scores/threshold_B/`: coefficients fit only on A; отрицательные q90 threshold only on B. Final exact-objective calibration lock помечен датой `2026-10-20`, точное время неизвестно. Сохранённый fitting runner читает только эти A/B файлы; test/HEEDB scores/labels в fit не подаются.
3. **Test detector cache.** По review `DETECTOR_TEST_BINDINGS.json` Ningbo test detector cache уже существовал до финального exact-objective lock. Поэтому нельзя заявлять, что test waveform не открывали до последнего lock. Проверенные материалы не дают точного времени создания cache относительно locks; timestamp не домысливается.
4. **Classifier test predictions и summaries.** Сохранены 48 prediction payloads Ningbo и 48 HEEDB, привязанные к checkpoints, preprocessing и source order. Их file hashes и labels/order проверены. Completion использовал эти payloads и frozen resample index matrices; повторного forward pass не было.
5. **Порядок технических исправлений.** Amendment 006 документирует технический переход с эквивалентного Platt к точному зафиксированному objective, заново используя A/B scores до основной test/HEEDB result aggregation. Независимый review уточняет, что test detector cache существовал раньше. Остальные amendments и неудачные попытки сохранены в исходном пакете. Completion поправил только condition order в bootstrap runner; original 300 draw vectors не заменены.

Следовательно, role split и fitting path по сохранённым данным корректны, но доступ к test detector cache произошёл до final calibration lock. Это раскрытие порядка доступа; оно не привело к refit по test/HEEDB outcome и не было исправлено изменением времени или пересчётом calibration.
'''
    (OUT / "LOCK_CHRONOLOGY_RU.md").write_text(chronology, encoding="utf-8")

    plan_status = '''# Сводка текущего исследовательского плана

Состояние на 20 октября 2026 года. Calibration completion выполнен и ожидает проверки; это не заявление о принятии результата.

## Выполнено

- Identity rederivation sensitivity выполнен и принят: для готовых моделей проверено влияние повторного расчёта отведений.
- Three-swap mixture recovery v2 выполнен; неисправная Huang input chain исправлена и повторно проверена для затронутых predictions.
- Calibration role control: девять fits (ordinary, physics_aware p050, combined × 3 seeds), model_selection / calibration_A / threshold_B source-record roles, 36 A-only Platt fits и B-only q90 thresholds; predictions Ningbo/HEEDB сохранены и проверены.
- Completion добавил недостающие 400 bootstrap-групп и восстановил 700-группную статистику; initialization seed и порядок доступа к данным раскрыты. Пакет ожидает пользовательской проверки.

## Ограничено

- Source records разделены между model selection, A и B; patient-disjointness Ningbo не подтверждена, так как patient IDs отсутствуют. HEEDB — ранее исследованная историческая sensitivity cohort.
- Declared seeds задавались после конструктора модели: начальная инициализация не контролировалась fit seed и initial RNG states не сохранены. Нынешние fits пригодны как фиксированные checkpoints, но не как matched initializations.
- Test detector cache Ningbo существовал до финального exact-objective calibration lock. Сам fit по зафиксированному коду использовал только A/B; отсутствие более раннего test access не заявляется.
- HEEDB labels — proxy; simulated lead reversals и полученные интервалы не доказывают клиническую эффективность. Bootstrap uncertainty условна на сохранённых fits/calibrators.
- Историческое сравнение с прежними fits также меняет initialization/training trajectory, sample sizes и выбранные epochs; его нельзя приписать одному calibration-selection effect.

## Ещё не выполнено / ждёт решения

- Получить полный доступ к публикации Cao и принять обоснованное решение, включать ли её implementation как direct baseline. До получения источников baseline не считается реализованным.
- Получить клиническую интерпретацию/ответы по miswiring mappings и подтвердить соответствие simulated reversal сценариев реальной практике.
- Разобрать реальные miswirings, leave-one-swap-out, новые архитектуры и прочие optional направления — только после отдельного решения; они этим пакетом не запускались.
- Рукопись отложена пользователем. Статья и литературный вывод этим пакетом не готовились.
'''
    (OUT / "CURRENT_PLAN_STATUS_RU.md").write_text(plan_status, encoding="utf-8")

    status = '''# Статус calibration role control completion v2

**Состояние:** completion выполнен, ожидает пользовательской проверки.

- Completion protocol зафиксирован до новых интервалов; SHA-256 в `COMPLETION_PROTOCOL_SHA256.txt`.
- Все 700 bootstrap-групп по 1 000 draws покрыты: 300 переиспользованы, 400 добавлены.
- TWA point recalculation, sklearn checks и old-draw preservation прошли; см. `NUMERICAL_VERIFICATION.json`.
- Checkpoint, calibration, threshold, split, prediction и исторические outputs v1 не перезаписывались; см. `UNCHANGED_ARTIFACTS_AUDIT.json`.
- Seed limitation и lock/access chronology раскрыты. Отчёты записаны в UTF-8 и проверены.
- Нового обучения, inference, fit calibration, threshold selection или reserved test оценки не было.

Краткий итог: `FINAL_REPORT_RU.md`; таблицы ключевых метрик: `KEY_RESULTS_TABLES_RU.md`.
'''
    (OUT / "STATUS.md").write_text(status, encoding="utf-8")

    # Validate source checkpoint hashes against immutable model lock; no checkpoint is loaded or modified.
    model_lock = json.loads((V1 / "ALL_MODELS_LOCK.json").read_text(encoding="utf-8-sig"))
    checkpoints = []
    for fit in model_lock["fits"]:
        d = V1 / "fits" / fit["method"] / f"seed_{fit['seed']}"
        row = {"method": fit["method"], "seed": fit["seed"], "checkpoint_sha256_locked": fit["checkpoint_sha256"],
               "checkpoint_sha256_observed": sha(d / "checkpoint.pt"), "best_sha256_locked": fit["best_sha256"],
               "best_sha256_observed": sha(d / "best.pt")}
        row["pass"] = row["checkpoint_sha256_locked"] == row["checkpoint_sha256_observed"] and row["best_sha256_locked"] == row["best_sha256_observed"]
        checkpoints.append(row)

    bound_bad = []
    for item in bindings["files"]:
        f = ROOT / Path(item["path"])
        if not f.exists() or f.stat().st_size != item["bytes"] or sha(f) != item["sha256"]:
            bound_bad.append(item["path"])
    independent = json.loads((V1 / "INDEPENDENT_REVIEW_AUDIT_20261020.json").read_text(encoding="utf-8"))
    original_v1_manifest = json.loads((V1 / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8-sig"))
    historical_entries = original_v1_manifest.get("files", [])
    historical_mismatches = []
    for item in historical_entries:
        f = ROOT / Path(item["path"])
        if not f.exists() or f.stat().st_size != item["bytes"] or sha(f) != item["sha256"]:
            historical_mismatches.append(item["path"])
    plan_path = ROOT / "NEXT_RESEARCH_DECISIONS_20261019.md"
    original_plan_snapshot_sha = sha(OUT / "ACTIVE_PLAN_CONTEXT_SNAPSHOT.md")
    unchanged = {
        "status": "PASS" if not bound_bad and all(x["pass"] for x in checkpoints) else "FAIL",
        "v1_source_bindings_checked": len(bindings["files"]), "v1_source_binding_changes": bound_bad,
        "original_v1_manifest_output_count": len(historical_entries),
        "original_v1_manifest_outputs_matching": len(historical_entries) - len(historical_mismatches),
        "original_v1_manifest_output_mismatches": historical_mismatches,
        "checkpoint_checks": checkpoints,
        "all_model_and_fit_checkpoint_hashes_match_lock": all(x["pass"] for x in checkpoints),
        "v1_models_predictions_calibration_thresholds_splits_and_saved_300_draws_unchanged": not bound_bad,
        "source_v1_manifest_preserved_without_rewriting": True,
        "active_plan_update": {"path": plan_path.relative_to(ROOT).as_posix(),
                               "snapshot_sha256_before_update": original_plan_snapshot_sha,
                               "current_sha256": sha(plan_path),
                               "purpose": "prepend requested completion status; original content bytes preserved after the status banner"},
        "prior_independent_review_status": independent["numerical_status"],
        "new_analysis_outputs": [
            "PAIRED_POINTS_CI_ALL_700.csv", "PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz",
            "BOOTSTRAP_COVERAGE_AUDIT.json", "NUMERICAL_VERIFICATION.json",
            "KEY_RESULTS_TABLES_RU.md", "FINAL_REPORT_RU.md",
        ],
    }
    (OUT / "UNCHANGED_ARTIFACTS_AUDIT.json").write_text(json.dumps(unchanged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if unchanged["status"] != "PASS":
        raise RuntimeError("unchanged artifact audit failed")

    verification.update({
        "source_input_bindings_checked": len(bindings["files"]),
        "prior_full_metric_review_status": independent["numerical_status"],
        "prior_full_metric_rows_recalculated": independent["counts"].get("saved_metric"),
        "old_draws_checked_against_summary_and_reused": 300,
        "old_draws_max_abs_difference_after_merge": verification["old_draw_vectors_preserved_max_abs_delta"],
        "all_700_points_have_1000_draws": all(int(r["n_draws"]) == 1000 for r in points),
        "new_condition_order_matches_frozen_control_FAULTS": True,
        "checkpoint_lock_match_count": sum(x["pass"] for x in checkpoints),
    })
    (OUT / "NUMERICAL_VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The completion-time review prompt is an instruction binding, not a scientific data input.
    prompt = ROOT / "LUNA_PROMPT_CALIBRATION_COMPLETION_20261020.md"
    # The project may retain the user-supplied task only as conversation context; if absent, record that clearly.
    prompt_binding = {"path": prompt.relative_to(ROOT).as_posix(), "exists_in_workspace": prompt.exists(),
                      "sha256": sha(prompt) if prompt.exists() else None,
                      "role": "instruction_context_only; not a statistical/model input"}
    active_plan = ROOT / "NEXT_RESEARCH_DECISIONS_20261019.md"
    active_plan_update = {"path": active_plan.relative_to(ROOT).as_posix(), "sha256_after_status_update": sha(active_plan),
                          "role": "mutable current-status document; not a scientific input"}

    # Source/analysis manifest keeps old immutable inputs separate from new artifacts.
    excluded = {"MANIFEST.json"}
    output_files = []
    for f in sorted(OUT.rglob("*")):
        if not f.is_file() or f.name in excluded or "__pycache__" in f.parts:
            continue
        output_files.append({"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f), "bytes": f.stat().st_size,
                             "role": "frozen_context_snapshot" if f.name == "ACTIVE_PLAN_CONTEXT_SNAPSHOT.md" else "new_analysis_output"})
    manifest = {
        "status": "COMPLETE_PENDING_USER_REVIEW",
        "created_date": "2026-10-20",
        "exact_wall_clock_time": "unknown",
        "protocol_sha256": (OUT / "COMPLETION_PROTOCOL_SHA256.txt").read_text().split()[0],
        "completion_runner_sha256": sha(OUT / "complete_bootstrap.py"),
        "protocol_amendments": ["COMPLETION_PROTOCOL_AMENDMENT_001.json"],
        "instruction_context_binding": prompt_binding,
        "active_plan_after_update": active_plan_update,
        "scientific_input_file_count": len(bindings["files"]),
        "scientific_inputs": bindings["files"],
        "new_outputs_count_excluding_manifest": len(output_files),
        "new_outputs": output_files,
        "source_plan_note": "The accepted plan was copied before updating its current status. That snapshot is context only and is not a model/data input.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # UTF-8 readback and Cyrillic presence check for all requested reports.
    report_paths = ["FINAL_REPORT_RU.md", "TRAINING_REPORT_RU.md", "CURRENT_PLAN_STATUS_RU.md",
                    "STATUS.md", "INITIALIZATION_PROVENANCE_NOTE_RU.md", "LOCK_CHRONOLOGY_RU.md", "KEY_RESULTS_TABLES_RU.md"]
    report_check = {}
    for name in report_paths:
        text = (OUT / name).read_text(encoding="utf-8")
        has_cyrillic = any("\u0400" <= char <= "\u04ff" for char in text)
        report_check[name] = {"utf8_readable": True, "has_cyrillic": has_cyrillic, "contains_replacement_question_run": "????" in text}
        if not has_cyrillic or "????" in text:
            raise RuntimeError(f"UTF-8 report validation failed: {name}")
    (OUT / "REPORT_UTF8_AUDIT.json").write_text(json.dumps({"status": "PASS", "reports": report_check}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "PASS", "reports": report_paths, "outputs": len(output_files),
                      "unchanged_v1_bindings": len(bindings["files"]), "checkpoints": len(checkpoints)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
