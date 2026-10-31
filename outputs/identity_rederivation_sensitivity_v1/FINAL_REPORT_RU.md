# Identity re-derivation sensitivity: завершённый пакет

**Статус:** полный пакет завершён; пост-access sensitivity analysis на 18 существующих моделях. Новое обучение не выполнялось. Protocol SHA-256: `c7652e1604ef445c89453cd84b2022ddf687c8e409c31ce2911828a4dcb8fc81`.

## Короткий ответ

Повторный вывод производных limb leads изменяет сами сигналы относительно исторического 12-channel clean. Далее проверялась величина изменения вероятностей и метрик, а не предполагалась безвредность по малой разнице сигналов. Полные оценки по каждой модели и endpoint сохранены в CSV ниже.

Сравнение двух clean controls показывает sensitivity к повторному вычислению; AP не раскладывается на причинные слагаемые. Циклы остаются отдельными физическими преобразованиями. Их ранее сохранённые предсказания переиспользованы и не изменялись.

Identity re-derivation слегка меняет probabilities, хотя сохраняемые I, II и V1–V6 тождественны исходным. По всем TWA моделям средний absolute score change был около 0.00044; максимальный TWA change — 0.0166, максимум по всем endpoints — 0.0352, а максимальный TWA p95 абсолютного сдвига — 0.00342. Для TWA/SB/STach максимальные доли смены бинарного решения при фиксированном validation threshold составили 0.146%, 0.049% и 0.098%. Это измеренные значения, а не утверждение клинической безвредности.

Для TWA AP точечные identity_clean−original_clean разности по шести cohort×arm группам находятся от −0.000010 до +0.000145; все соответствующие 95% CI включают ноль. Для TWA AUROC сдвиги также малы (до 0.000019 по абсолютной точечной разности). Raw Brier изменился на величину до примерно 0.000112; эти небольшие shifts имеют узкие conditional intervals, поэтому это не буквальная идентичность вероятностей. CI через ноль для AP/AUROC не доказывает эквивалентность.

В сравнении cycle с двумя clean controls численные оценки близки. Для physics-aware cycle 120 против original clean AP: Chapman −0.03894, Ningbo −0.02987; против identity-derived clean: Chapman −0.03897, Ningbo −0.02986. Контрасты physics-aware/combined против ordinary на обоих циклах совпали с прежними полными bootstrap draws до max difference <2e−16. Следовательно, чувствительность clean-path не меняет наблюдаемую трактовку циклических результатов или выигрыша обученных защит в этих данных.

## Итоги проверки

- Engineering: `PASS`; формулы проверены независимым вычислением; float32 identity transform идемпотентен. Signal units оставлены как raw, не названы микровольтами.
- Replay: `PASS` для 18 моделей × 3 условий на первых 32 записях; tolerance 0.0001. Новые полные predictions: 18 identity-rederived files; original clean и оба cycle условия переиспользованы после binding/replay checks.
- Replay history: первая попытка batch 32 превысила `1e-4` (max 0.0004883). Исторический inference работал batch 128, поэтому после сохранения неудачной попытки повторили обработку в batch 128 и сравнили те же первые 32 записи; все 54 checks прошли с прежним допуском. См. amendment 001 и `REPLAY_AUDIT_ATTEMPT_BATCH32.json`.
- По всем test сигналам максимальное отличие исторических 12 каналов от identity-derived было 3.5 raw units; combined RMSE — 0.8844 Chapman и 0.8840 Ningbo. Производные limb leads отличаются, а независимые I, II и V1–V6 совпали точно.
- Повторный расчёт сохранённых метрик и bootstrap summaries: `PASS`; внутренний audit, не внешний независимый review.

- Засечённое время: полный identity inference 10.59 s на 18 файлов; анализ, включая 2×1,000 draw, 25.44 s. Обучений — 0.

## TWA: mean по трём seed ± sample SD

В таблице приведены AP/AUPRC, AUROC и raw Brier отдельно для двух когорт и четырёх условий. Полные строки для TWA, SB и STach, включая sensitivity/specificity при исходном validation threshold, находятся в `CONDITION_RESULTS_MEAN_SD.csv`; значения по каждому seed — в `CONDITION_RESULTS_BY_SEED.csv`.

| Когорта | Режим | Условие | AP mean | AUROC mean | Raw Brier mean |
|---|---|---|---:|---:|---:|
| Chapman | ordinary | original_clean | 0.65314 ± 0.00703 | 0.91558 ± 0.00099 | 0.11514 ± 0.00901 |
| Chapman | ordinary | identity_rederived_clean | 0.65313 ± 0.00684 | 0.91556 ± 0.00098 | 0.11503 ± 0.00895 |
| Chapman | ordinary | cycle_slots_120 | 0.26606 ± 0.03825 | 0.62189 ± 0.04842 | 0.57348 ± 0.04641 |
| Chapman | ordinary | cycle_slots_201 | 0.39038 ± 0.02607 | 0.71921 ± 0.04326 | 0.32343 ± 0.04787 |
| Chapman | physics_aware | original_clean | 0.66406 ± 0.00922 | 0.91384 ± 0.00166 | 0.13586 ± 0.03332 |
| Chapman | physics_aware | identity_rederived_clean | 0.66409 ± 0.00934 | 0.91384 ± 0.00173 | 0.13575 ± 0.03326 |
| Chapman | physics_aware | cycle_slots_120 | 0.62512 ± 0.01074 | 0.89459 ± 0.00301 | 0.16584 ± 0.04547 |
| Chapman | physics_aware | cycle_slots_201 | 0.64528 ± 0.00547 | 0.89452 ± 0.00315 | 0.15045 ± 0.03475 |
| Chapman | combined | original_clean | 0.65864 ± 0.00920 | 0.91280 ± 0.00568 | 0.12238 ± 0.01538 |
| Chapman | combined | identity_rederived_clean | 0.65864 ± 0.00930 | 0.91281 ± 0.00571 | 0.12229 ± 0.01531 |
| Chapman | combined | cycle_slots_120 | 0.60986 ± 0.00633 | 0.89292 ± 0.00209 | 0.13543 ± 0.02164 |
| Chapman | combined | cycle_slots_201 | 0.62870 ± 0.01026 | 0.88628 ± 0.00662 | 0.13859 ± 0.02652 |
| Ningbo | ordinary | original_clean | 0.67612 ± 0.00279 | 0.92292 ± 0.00115 | 0.09896 ± 0.00923 |
| Ningbo | ordinary | identity_rederived_clean | 0.67627 ± 0.00261 | 0.92294 ± 0.00115 | 0.09903 ± 0.00924 |
| Ningbo | ordinary | cycle_slots_120 | 0.15102 ± 0.01540 | 0.49250 ± 0.03313 | 0.59214 ± 0.04603 |
| Ningbo | ordinary | cycle_slots_201 | 0.22593 ± 0.04318 | 0.59940 ± 0.04278 | 0.39965 ± 0.04155 |
| Ningbo | physics_aware | original_clean | 0.68688 ± 0.00986 | 0.92417 ± 0.00482 | 0.10771 ± 0.01606 |
| Ningbo | physics_aware | identity_rederived_clean | 0.68687 ± 0.00985 | 0.92419 ± 0.00481 | 0.10764 ± 0.01604 |
| Ningbo | physics_aware | cycle_slots_120 | 0.65701 ± 0.00732 | 0.90592 ± 0.00475 | 0.12182 ± 0.02275 |
| Ningbo | physics_aware | cycle_slots_201 | 0.64822 ± 0.01242 | 0.90428 ± 0.00717 | 0.12374 ± 0.02230 |
| Ningbo | combined | original_clean | 0.67676 ± 0.01151 | 0.92403 ± 0.00293 | 0.10243 ± 0.00515 |
| Ningbo | combined | identity_rederived_clean | 0.67680 ± 0.01164 | 0.92403 ± 0.00293 | 0.10245 ± 0.00518 |
| Ningbo | combined | cycle_slots_120 | 0.63323 ± 0.01539 | 0.89338 ± 0.01028 | 0.12719 ± 0.02024 |
| Ningbo | combined | cycle_slots_201 | 0.63021 ± 0.01282 | 0.89540 ± 0.00436 | 0.12621 ± 0.01328 |

## Сопоставления и трактовка циклов

Парные contrast таблицы содержат full-cohort point estimate отдельно от bootstrap mean и percentile 95% CI. Bootstrap использовал исходные 1,000 synchronized record-index draws, условия и seeds парны; три seeds усреднялись внутри каждой реплики. Пересечение CI с нулём не означает эквивалентность.

Для каждой модели рассчитаны identity-clean−original-clean, оба cycle−original-clean и оба cycle−identity-clean. Также проверены physics-aware−ordinary и combined−ordinary на каждом цикле. `HISTORICAL_CYCLE_CONTRAST_MATCH.json` сравнивает полные 1,000 draws новых повторных расчётов с ранее сохранёнными cycle contrast draws; циклические predictions и их point estimates не менялись.

`SCORE_CHANGES_BY_SEED.csv` показывает max/mean и квантили абсолютного изменения вероятностей, а также долю решений, изменившихся при неизменных validation thresholds. Не назначался произвольный клинический порог заметности.

## Ограничения и вывод

Это дополнительная sensitivity проверка на ранее просмотренных Chapman и Ningbo тестовых когортах, не независимая подтверждающая выборка. Bootstrap отражает вариацию записей при фиксированных моделях. Сравнение identity-derived и исторического clean оценивает чувствительность к пути представления; оно не устанавливает единственную правильную реализацию для всех исходных ECG устройств. Циклические swaps переиспользуют исторические cycle score files, поэтому их прежняя трактовка сохраняется как post-access результат.

Технически повторное вычисление работает детерминированно и в float32 идемпотентно, а source identity отличается от ранее сохранённых derived limb channels (до 3.5 raw units). Метрики показывают небольшую, но измеримую чувствительность вероятностей и raw Brier; TWA ranking changes малы относительно cycle effects. Для этой sensitivity-проверки это не указывает на необходимость переобучить или переписать модели, но способ формирования derived leads следует раскрыть и согласованно зафиксировать до следующей новой серии. Не заменять задним числом clean результаты без отдельного согласованного анализа.

## Артефакты

- `FROZEN_PROTOCOL.json`, `PROTOCOL_HASH.json` — scope, bindings и fingerprint до новых inference.
- `PROTOCOL_AMENDMENT_001.json`–`PROTOCOL_AMENDMENT_006.json` — технические уточнения replay, bootstrap, verifier и manifest finalization; scientific scope, условия, thresholds и данные не менялись.
- `ENGINEERING_CHECKS.json`, `SOURCE_BINDINGS.json`, `REPLAY_AUDIT.json` — формулы, signal differences, input hashes, replay.
- `PREDICTION_BINDINGS.json`, `predictions/` — новые identity predictions и bindings.
- `CONDITION_RESULTS_BY_SEED.csv`, `CONDITION_RESULTS_MEAN_SD.csv`, `SCORE_CHANGES_BY_SEED.csv` — endpoint metrics и score changes.
- `PAIRED_CONTRASTS.csv`, `BOOTSTRAP_REPLICATES.csv`, `HISTORICAL_CYCLE_CONTRAST_MATCH.json` — paired estimates, all draws and unchanged historical cycle comparison.
- `VERIFICATION_AUDIT.json`, `REPRODUCIBILITY_MANIFEST.json` — checks and hashes.

По согласованной очереди следующий этап — смесь трёх swaps. Он здесь не запускался; пакет остановлен для проверки.
