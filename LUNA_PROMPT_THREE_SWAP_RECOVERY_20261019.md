Ты работаешь в проекте:
C:\Users\User\Documents\AI_research_tests\03_new_biomedical_research\projects\03_physical_ecg_failure_benchmark

Выполни точечное восстановление three-swap mixture. Это исправление обязательного текущего этапа, а не новое направление. К калибровочному preflight, обучению и статье пока не переходить.

Сначала прочитай применимые AGENTS.md, NEXT_RESEARCH_DECISIONS_20261019.md и:
- outputs/three_swap_population_mixture_v1/REVIEW_NOTES_20261019.md;
- outputs/three_swap_population_mixture_v1/INFERENCE_PATH_REVIEW_20261019.json;
- outputs/three_swap_population_mixture_v1/review_huang_input_path_20261019.py;
- исходное задание LUNA_PROMPT_THREE_SWAP_MIXTURE_20261019.md;
- v1 frozen protocol/amendments, run_inference.py и analyze_mixture.py.

Ошибка доказана: в infer_huang_classifier() для новых RA_LA/LA_LL вызван corrected_batch(original_raw, predicted_classes_for_faulty_ECG). Но detector_worker() подавал детектору реально переставленный ECG. Диагностическая ветка пропустила начальное создание faulty input. Все 6 ошибочных файлов воспроизводятся по неправильной цепочке с max probability difference=0.0; правильная цепочка на том же блоке отличается сильно. Численный PASS v1 проверял арифметику ошибочных probabilities и не является подтверждением корректной реализации метода.

1. Создай отдельный outputs/three_swap_population_mixture_recovery_v2/. Исторический v1, его scripts, predictions, manifest и failed attempts не перезаписывай. Сохрани recovery protocol и hash до нового full inference: исправление кода, ограниченный scope, точные frozen sources, checkpoints, transformations, precision, thresholds, calibrators, q, сценарии и bootstrap indices. Различай SHA-256 файла и canonical JSON fingerprint. Изменяемый активный план сохрани отдельным snapshot как контекст; не делай дальнейшее разрешённое обновление его статуса нарушением неизменности научных входов.

2. Новый full inference требуется только для Huang_full_pipeline на HEEDB, RA_LA_reversal и LA_LL_reversal, seeds 104729/130363/155921: всего 6 prediction файлов. Детектор заново не обучать/не запускать на всей когорте: его готовые probabilities корректны по найденной ошибке, переиспользовать после проверки provenance. Остальные 90 model×condition score наборов переиспользовать с явными bindings (48 Ningbo и 42 HEEDB). Повторное обучение, calibration fitting, threshold selection, новая когорта и reserved_test запрещены этим scope.

Правильная цепочка для каждой новой fault:
original raw ECG → incoming_faulty = electrical_reversal_batch(raw, condition) → frozen detector probabilities для этого же incoming_faulty → predicted_class=argmax → corrected_batch(incoming_faulty, predicted_class) → прежний ordinary diagnostic checkpoint с прежней normalization и stride.

Истинная condition нужна для симуляции incoming ECG, но НЕ для выбора correction. Случаи predicted clean и неправильных predicted swaps должны сохранять реальное поведение полного pipeline. Не заменять detector argmax истинным классом и не исправлять метод до идеальной коррекции. Clean и RA_LL использовать из исторических проверенных caches, не генерировать по неправильной новой ветке.

3. До полного inference сделай инженерный контроль фактического input handoff, а не только отдельно формул. На реальных записях и синтетических сигналах проверь для каждого swap и каждой predicted class 0/1/2/3, что:
- detector и correction получают версии одного incoming faulty ECG;
- predicted class 0 оставляет faulty input без inverse;
- правильный inverse инженерно возвращает identity-derived clean с обоснованным float32 tolerance;
- неверный inverse ведёт себя как соответствующая композиция;
- I/II/V1–V6 порядок, normalization, stride, dtype и batch не менялись;
- исходный clean buffer не мутирует.
Инженерный known-class inverse check отдели от оценивания метода. Отдельно реализованный ожидаемый сигнал должен проверять всю цепочку; не ограничиваться вызовом того же helper дважды.

Воспроизведи первые 32 записи из полного исторического batch128 для всех 6 единиц, затем по одному среднему и последнему историческому batch block. Сравни corrected pipeline с отдельной явной реализацией (тот же frozen model, preprocessing и detector classes), tolerance probabilities 1e-4. Сохрани также контроль обнаруженной неправильной цепочки; не выдавай совпадение с ошибочными scores за нужный PASS. Не требуй любой заранее заданной величины улучшения метрик: научный результат заранее неизвестен.

4. После успешного handoff/replay заново посчитай 6 полных файлов на тех же 10 000 HEEDB пациентах. Сохрани IDs, labels, file IDs, condition, checkpoint/preprocessing/detector/protocol hashes и runtimes. Корректные HEEDB detector probabilities должны быть общими для seeds; не считать detector повторно для каждого classifier.

5. Собери полный восстановленный анализ прежнего scope: Ningbo + HEEDB; ordinary, physics p050, combined, Huang; 3 seeds; clean и 3 swaps; equal_three_swap и RA_LL_only; q=[0,.001,.005,.01,.05,.1,1]; TWA primary, SB/STach secondary. Calibrators и собственные validation-B thresholds всех методов прежние, для Huang — full-pipeline Huang thresholds. Все ограничения прежней calibration зависимости от checkpoint selection сохраняются.

Пересчитай condition metrics, смеси, mean/SD, paired full points/CI и draws, затронутые 6 заменами. Допустим полный повтор арифметики, но не лишний inference. q=0, весь RA_LL_only, весь Ningbo, а также сравнения physics−ordinary/combined−ordinary в HEEDB должны сохраниться; проверь это явно по всем доступным значениям и draws. Для неизменных результатов можно переиспользовать подтверждённые draws после проверки. Для зависимых Huang equal-three-swap HEEDB q>0 — настоящий пересчёт при тех же 1000 patient bootstrap indices, парность всех views/методов/seeds сохранить. Не переносить туда старые неверные интервалы.

Сохрани все прежние anchor checks. Дополнительно проверь, что non-RA faults действительно участвовали в исправленном полном diagnostic path: RA_LL-only anchors эту ошибку не обнаруживали. Views одного пациента не становятся независимыми наблюдениями. AP/AUROC смеси считать по weighted scores, не усреднением condition AP/AUROC. Веса, 4N normalization, zero-weight/tie handling и seed aggregation — по прежнему корректному протоколу.

6. Исправь provenance в новом пакете:
- заполнить фактические input/final paths и hashes для всех 96 единиц, без placeholder NEW_INFERENCE_REQUIRED/CHECK_SOURCE_AUDIT в финальной coverage;
- ordinary HEEDB исходные scores не называть masking; для многомерных NPY указать оси/indices, checkpoint binding и content hash slice по необходимости;
- добавить Ningbo confusion/false-correction summary из существующих detector caches, сохранив HEEDB summary;
- времена прежних запусков Huang, которые не были записаны, указать unavailable; для текущего recovery измерить реальные времена;
- старый FINAL_REPORT_RU.md не считать верным сравнением с Huang.

7. Проверь независимо из saved predictions все метрики и means/SD; из saved draws пересчитай summaries. На фактических draws 0/417/999 перепроверь затронутые contrasts отдельным sklearn calculation. Сопоставь неизменные результаты с v1 с точными/обоснованными численными tolerances. Парсер CSV должен сохранять round-trip точность. Scientific metrics и full input-path verification — разные разделы аудита: нужны оба. Сохрани input/output hash manifests, новый код, execution summary, recovery status и failed attempts при необходимости.

Ожидаемые результаты: RECOVERY_PROTOCOL.json/hash, INPUT_COVERAGE.csv, 6 новых predictions с bindings, INPUT_HANDOFF_AUDIT.json, REPLAY_AUDIT.json, condition/mixture tables, paired points/draws/CI, UNAFFECTED_RESULTS_AUDIT.json, VERIFICATION_AUDIT.json, REPRODUCIBILITY_MANIFEST.json, STATUS.md и FINAL_RECOVERY_REPORT_RU.md. Другие вспомогательные файлы допустимы в новом пакете.

В отчёте простым русским языком объясни ошибку, что пришлось заменить и что осталось прежним; затем сравни восстановленные результаты с прежним RA_LL-only сценарием. Покажи Huang, physics и combined на HEEDB хотя бы при q=.01,.1,1, raw AP, calibrated Brier, sensitivity и FP/1000, вместе с relevant CI. Отличай лучший point estimate от статистически подтверждённой разницы. Не обещать победу нашей модели и не интерпретировать баг как слабость Huang.

В конце краткое резюме и пути к артефактам. Активный план можно обновить только до «recovery выполнен, ожидает проверки», не объявлять внешнее принятие. После этого остановись для нашей проверки. Calibration preflight, новые обучения и статью пока не запускать.
