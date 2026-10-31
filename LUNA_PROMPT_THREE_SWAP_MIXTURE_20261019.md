Ты работаешь в проекте:
C:\Users\User\Documents\AI_research_tests\03_new_biomedical_research\projects\03_physical_ecg_failure_benchmark

Активный план — NEXT_RESEARCH_DECISIONS_20261019.md. Identity sensitivity завершён и принят: см. outputs/identity_rederivation_sensitivity_v1/REVIEW_NOTES_20261019.md. Выполни следующий ограниченный пакет — смесь трёх swaps среди преимущественно чистых ECG. Это выполнение эксперимента до проверенного результата, а не только план или написание скрипта. Новые обучения и calibration fitting не нужны.

Цель: раньше основной prevalence analysis смешивал clean и только RA–LL. Проверить, сохраняются ли выводы при условном равном сочетании RA–LL, RA–LA, LA–LL. Это exploratory extension на уже просмотренных данных, не новая независимая внешняя репликация и не оценка реальной частоты ошибок.

1. Прочитай применимые AGENTS.md, активный план, результаты identity control, а также фактические протоколы и проверенные источники:
- outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_PROTOCOL.json и FINAL_FOUR_METHOD_COMPARISON.md;
- outputs/ningbo_four_arm_replication_v1/ (готовые модели, scores, CALIBRATION_AND_PROVENANCE.json);
- outputs/heedb_four_arm_extension_v1/FROZEN_EXTENSION_PROTOCOL.json, PREDICTION_BINDINGS.json и FINAL_EXTERNAL_SIX_METHOD_REPORT.md;
- outputs/heedb_probability_extension_v1/REVIEW_NOTES_20261019.md, если файл существует, иначе актуальные review notes этого пакета;
- outputs/published_detector_pipeline_v1/ и outputs/heedb_huang_exploratory_v1/ — реальная интеграция Huang.
Найди актуальные paths и hashes; не принимай старые future-work тексты за текущие задания. Не исполняй старые workflow с побочными изменениями исторических пакетов.

2. Создай отдельный outputs/three_swap_population_mixture_v1/. Проведи preflight инвентаризации coverage, checkpoints, normalization, labels/IDs, scores, calibrators, thresholds и runtime. Сохрани INPUT_COVERAGE.csv и FROZEN_PROTOCOL.json с SHA-256 до получения новых inference outcomes. Затем продолжай без запроса повторного разрешения на согласованный объём.

Объём:
- Ningbo: прежние 6 981 test records, в прежнем порядке;
- HEEDB: та же frozen I0006 когорта 10 000 уникальных пациентов и прежние proxy labels, без расширения/перевыбора;
- методы ordinary ResNet, physics_aware p050, combined, Huang full pipeline;
- seeds 104729, 130363, 155921 для диагностических моделей;
- условия clean, RA_LL_reversal, RA_LA_reversal, LA_LL_reversal;
- q = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1].
Chapman, p010/p025, chest-only, masking-only, Founder и Transformer в этот ограниченный пакет не добавляй. Никаких новых train/validation splits, calibrator fit, threshold selection, hyperparameter tuning или reserved_test evaluation.

3. Переиспользуй подтверждённые старые clean/RA–LL scores, а также RA–LA/LA–LL, если уже есть подходящие файлы. Новый inference — только для отсутствующего покрытия. Исторические файлы не перезаписывай. Сохраняй новые probabilities с record IDs, labels, condition, checkpoint/preprocessing/protocol hashes; provenance переиспользованных comparator scores должен быть подтверждён.

Применяй прежнюю физическую реализацию каждого swap к исходному raw signal до замороженной normalization. Clean остаётся историческим clean, без задней замены на identity-derived. Проверка identity не оправдывает произвольные новые transforms. Проверь независимые формулы, channel order, сохранение V1–V6 и отсутствие случайных transforms.

Huang: официальный frozen detector, его прежняя preprocessing и predicted-class inverse correction, затем соответствующий ordinary диагностический checkpoint. Class map прежнего протокола: clean, RA_LA, RA_LL, LA_LL. Детектор вызывается и на clean, включая ошибочные исправления. Не использовать истинную fault class как oracle. Detector outputs переиспользуй между seeds; не запускай один и тот же detector три раза. Соблюдай существующие отдельные Python environments, не переустанавливай зависимости глобально. Для каждой fault покажи confusion matrix/false-correction behaviour, без клинических выводов. Если реальная интеграция не воспроизводится, сохрани точный блокер Huang и заверши доступные методы, не подменяй его oracle.

Сначала replay малого фиксированного блока Ningbo на прежних clean/RA–LL условиях, затем недостающий source inference и инженерные проверки. После успешной source проверки — replay и missing inference на прежней HEEDB. Сохрани прежние batch/precision/device paths; tolerance diagnostic probability 1e-4, detector 1e-5 согласно прежнему протоколу. Изменение batch может влиять на float16 numerics: не повышай tolerance молча. Reuse vs new inference и фактические runtimes записать отдельно.

4. Две фиксированные смеси:
A) основной equal-three-swap сценарий: каждой записи присвоены веса clean=1−q; каждому из трёх corrupted views=q/3;
B) RA–LL-only comparator: clean=1−q, RA–LL=q, остальные=0.
Это weighted views одного пациента/record с одинаковой исходной label. Общая масса на запись всегда 1. Равные частоты faults и их независимость от label здесь предположения сценария, не клинические факты. Для q=1 в A все записи faulty, но это смесь трёх faults, а не только RA–LL.

Не усредняй AP/AUROC отдельных условий для получения mixture AP/AUROC. Вычисляй их по совместным scores и sample weights. Brier и confusion counts должны совпадать с взвешенными condition values. FP/FN на 1000 нормируй по массе исходной когорты, не по 4N. Zero-weight views/tie groups корректно исключать; учесть исправленный ранее zero-weight bootstrap bug.

5. Calibrators и thresholds возьми без изменений из проверенных предыдущих пакетов для каждого метода/seed/endpoint. Для Huang — его собственный full-pipeline source-validation-B threshold, не исторический baseline threshold от ordinary классификатора. Покажи primary raw_validation_B и соответствующий calibrated operating point; калиброванный threshold — прежний transformed raw threshold. Legacy threshold, если нужен для сверки, только отдельно.

Сохрани прежние clipping/precision/Platt параметры. Ничего не подгонять под новые faults или HEEDB. Проверь ranking/tie changes и равенство решений raw/calibrated, объясни возможные clipping effects. Обязательно раскрыть прежнюю зависимость calibration данных от checkpoint selection: этот пакет её не исправляет; независимая calibration проверка стоит следующим отдельным этапом.

6. TWA primary, SB/STach secondary. Для каждой source×method×seed×scenario×q: raw/calibrated AP/AUPRC, AUROC, Brier, ECE15, sensitivity, specificity, predicted-positive fraction, FP и FN на 1000; затем mean±sample SD по seed-specific metrics, не probability ensemble. Таблицы single-condition metrics тоже сохранить.

Primary equal-three-swap paired contrasts: physics−ordinary, combined−ordinary, Huang−ordinary, physics−Huang, combined−Huang. Если Huang технически отсутствует, соответствующие сравнения явно missing. Для raw TWA AP, raw/calibrated TWA Brier, sensitivity и FP/1000 при primary validation-B operating point — 1 000 парных bootstrap draws для всех q. Raw/calibrated operating decisions при совпадении не дублировать как независимые результаты. Seeds усреднять внутри общего draw.

Ningbo — прежняя record-resampling схема с раскрытием ограничения; HEEDB — patient bootstrap. Все четыре views одного пациента/record получают одну и ту же bootstrap multiplicity, также общую между методами и seeds. Проверь идентичность bootstrap indices прежним. Не считай 4N views или три seeds независимыми пациентами. На q=0 новый расчёт обязан воспроизвести старые clean результаты; сценарий B на всей q-grid — прежний RA–LL-only анализ. Сначала подтвердить эти anchors, затем интерпретировать A. Доверительные интервалы условны на фиксированных моделях/calibrators; пересечение нуля не доказывает эквивалентность.

Не сортируй заново огромные scores внутри каждого bootstrap без необходимости: можно использовать проверенный weighted/tied-score подход, предварительно подтвердив его против sklearn на настоящих repeated-index samples. Не расширяй bootstrap на все secondary endpoints/режимы ради количества таблиц.

7. До завершения отдельно пересчитай метрики из saved predictions, weight identities, q=0 и RA–LL-only anchors, bootstrap summaries из saved draws; для настоящих draws 0/417/999 сравни с независимым sklearn sample-weight/repeated-record расчётом. Сохрани реальные counts и max errors. Не называй внутренний verifier независимым внешним review. При кодовом исправлении сохрани failed attempt/amendment, не меняй scope или критерии для красивого результата.

Минимальные deliverables: frozen protocol/hash; INPUT_COVERAGE.csv; prediction/detector bindings и новые predictions; engineering/replay audits; calibration provenance и decision checks; condition/mixture tables по seeds и mean/SD; RA_LL_ANCHOR_AUDIT.json; paired points, all draws, CI summaries; VERIFICATION_AUDIT.json; input/output SHA-256 manifest; STATUS.md; FINAL_REPORT_RU.md. Допустимы компактные NPZ вместо огромных CSV draws при полной документированной структуре. Сохранить 1–2 понятных графика TWA AP, calibrated Brier и FP/1000 против q, если это помогает сравнению.

В финальном отчёте простым языком ответь: изменился ли вывод по сравнению с RA–LL-only; при каких q какой метод лучше по конкретной метрике; сохраняется ли цена на clean и ложных тревогах; что переносится на HEEDB, с напоминанием о proxy labels. Не объявлять универсальную победу, клиническую эффективность, equivalence или новую независимую внешнюю репликацию. Укажи scientific limitations и technical completeness отдельно.

Обнови только текущий статус активного плана после фактического завершения, с пометкой «результат ожидает проверки», не заявляй принятие аудитором. После этого остановись. Следующий шаг — calibration preflight; его и новое обучение пока не запускать. Статью не писать, сообщения авторам не отправлять.
