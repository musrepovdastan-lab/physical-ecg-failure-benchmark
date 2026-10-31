Ты работаешь в проекте:
C:\Users\User\Documents\AI_research_tests\03_new_biomedical_research\projects\03_physical_ecg_failure_benchmark

Выполни следующий этап принятого плана — CALIBRATION PREFLIGHT. Это аудит доступных данных и конкретный проект ограниченного эксперимента, а не запуск обучения. Дойди до готового, проверяемого решения с membership и counts, а не остановись на общих рекомендациях.

Прочитай применимые AGENTS.md, NEXT_RESEARCH_DECISIONS_20261019.md и:
- outputs/three_swap_population_mixture_recovery_v2/REVIEW_NOTES_20261019.md и FINAL_RECOVERY_REPORT_RU.md — recovery принят;
- outputs/mixed_fault_calibration_v1/FROZEN_ANALYSIS_PROTOCOL.json и calibration_split_indices.npz;
- outputs/mixed_fault_calibration_v2/FROZEN_COMPLETION_PROTOCOL.json, CALIBRATION_AND_PROVENANCE.json, HUANG_CALIBRATION_AND_PROVENANCE.json;
- outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json, SPLIT_MEMBERSHIP.json, TRAIN_VALIDATION_MEMBERSHIP.json, SPLIT_LEAKAGE_PROOF.json и исходники обучения;
- outputs/ningbo_four_arm_replication_v1/FROZEN_NINGBO_FOUR_ARM_PROTOCOL.json, training timings, preprocessing/calibration provenance;
- исходные frozen p050/ordinary training configurations и actual history/checkpoint selection paths.
Актуальная mixture оценка — recovery_v2; v1 сравнения HEEDB Huang для двух новых faults ошибочны и не заменяют recovery.

Проблема: все 3 488 Ningbo validation записей использовались для выбора checkpoint до деления на calibration A/B. Старый calibration fit не использовал test/HEEDB, но не отделён от checkpoint selection. Нужно понять, каким минимальным контролем проверить устойчивость наших выводов к этой зависимости. Не утверждать заранее, что текущие результаты завышены или что эффект будет большим.

1. Создай outputs/calibration_independence_preflight_v1/. Работай read-only относительно исторических файлов; всё новое только в этом пакете. Не обучай модели, не запускай новые forward passes, не fit calibrators и не выбирай thresholds. Не загружай новые test/HEEDB scores ради выбора дизайна, не оценивай reserved_test. Для preflight достаточно исходных train/validation metadata, memberships, counts и сохранённой документации. Исходные label mappings и исключения не менять; исключённые конфликтные/некачественные записи не превращать в удобный новый holdout.

2. Подготовь DATA_USE_AUDIT.csv: для каждого реально доступного подходящего source-пула покажи exact membership/hash, число записей, историю использования в обучении, normalization/pos_weight, checkpoint selection, tuning, calibration, threshold selection и просмотре outcomes. Для известного использования укажи конкретный код/артефакт; неизвестное обозначай unknown, а не unused. Учти source normalization, прошлую разработку гиперпараметров и ранние версии моделей. Test/HEEDB уже использовались и не являются новым независимым calibration pool.

Отдельно выясни единицу идентичности: есть ли подтверждённый patient ID и повторные ECG одного пациента, или только record IDs и waveform hashes. Нельзя доказать patient independence отсутствием совпавших waveform bytes или назвать record ID пациентом. Проверь доступные metadata/headers/docs без сетевого скачивания новых данных. Объясни границу утверждения: patient-disjoint либо только record/waveform-disjoint. При неизвестных IDs не изобретай идентификаторы из возраста/пола и не обещай строгую patient independence.

3. Выбери один основной путь:
A) Если действительно существует пригодный source holdout, не использованный для fitting/selection/tuning, обоснуй его независимость от текущих моделей и возможность обойтись без новых обучений. Нельзя объявлять unused только потому, что нет calibration scores.
B) Если такого holdout нет, спроектируй новый ограниченный Ningbo контроль: неизменный исходный train, а прежний validation разделён на model_selection, calibration_A и threshold_B ДО новых обучений. В model_selection выбирается epoch; calibration_A только для Platt; threshold_B только для operating point. Ни A, ни B не участвуют в checkpoint selection. Нельзя исправить эту зависимость простым переименованием старых checkpoints или кэшей: для B нужны новые fits.

Начальный кандидат B — 50/25/25 от прежнего validation, один детерминированный split seed 20261019, с сохранением multi-label состава насколько возможно. Если есть patient groups, делить целые группы, учитывая их пересечение с исходным train/test. При конфликте объясни, можно ли сохранить train или нужно отдельное решение; не менять train автоматически. Если patient IDs недоступны, предложи честный record-disjoint вариант и обозначь остаточное ограничение.

Сохрани proposed membership/indices с labels и stable hashes, не записывая его в историческую папку. Split выбирать по metadata/labels и заранее описанному алгоритму, без прогнозов моделей и без перебора splits ради хорошего результата. Если 50/25/25 технически/статистически непригоден, опиши причину и одну ограниченную альтернативу; не открывай большой поиск.

4. Для каждой новой части и исходного train покажи n, positives/negatives и prevalence TWA/SB/STach, а также multi-label strata и доступные patient counts. Проверь пересечения IDs, patient groups при наличии и waveform hashes между всеми частями и существующим test, используя сохранённые metadata; test outcomes не анализировать.

Раздели техническую возможность fit (есть оба класса и конечные данные) и статистическую точность. По фактическим class counts оцени ожидаемую неопределённость negative 90th-percentile operating point и чувствительности (например, binomial/rank uncertainty), без обучения/fitting и без обещания универсального достаточного размера выборки. Укажи риски редких endpoints, separation и узкого calibration range, которые counts сами по себе не исключают. TWA primary; слабость secondary endpoint не маскировать усреднением.

5. Составь DRAFT_CONTROL_PROTOCOL.json и MINIMAL_EXECUTION_PLAN.md, но не исполняй их. Для пути B ограничение — ordinary, physics p050, combined по seeds 104729/130363/155921, максимум 9 новых обучений. Из существующих protocol/code выпиши exact recipes, hyperparameters, early stopping, checkpoint rules, normalization и pos_weight происхождение. Использовать одинаковый model_selection split и прозрачный сопоставимый training budget; не улучшать методы по test. Preprocessing на исходном неизменном train можно переиспользовать только при подтверждённых bindings.

Для будущей калибровки предложи прежний положительный-slope Platt и прежний negative q90 operating rule, если нет конкретного препятствия; coefficients и thresholds сейчас не вычислять. Опиши отдельные A/B действия, возможные numerical failures и заранее заданное поведение при них. Не подбирать augmentation probabilities заново. Будущий evaluation: прежние clean/three-swap q сценарии на source test и прежней HEEDB, без новой независимой внешней репликации и без нового label mapping. Не повторять Chapman, Founder, Transformer, p-grid, cycles и 792-mask panel ради этой проверки.

Huang включить как полный frozen detector → predicted inverse → соответствующий НОВЫЙ ordinary classifier, с собственными full-pipeline calibration_A/threshold_B scores. Старые диагностические Huang outputs нельзя считать outputs нового ordinary checkpoint. Detector classes можно переиспользовать после проверки input/source-order bindings; указать, какие source validation scores понадобятся и что придётся пересчитать. Corrected handoff из recovery сохранить и предусмотреть его regression check.

6. Опиши допустимые claims: новый split отделяет calibration от selection конкретных новых checkpoints, но протокол появился после прошлых outcomes и исторической настройки гиперпараметров. Он не стирает прошлое использование данных и не делает прежнюю HEEDB новой untouched confirmation. Не обещать полную независимость от всей исследовательской разработки. Cross-fitting обсуждать только как запасной вариант при конкретном препятствии, без выполнения и без неограниченного роста training budget.

7. Оцени бюджет по реально сохранённым временам похожих обучений/инференсов: число fits, максимальные epochs, файлы, ожидаемые дополнительные вычисления. Различай измерение и оценку; не сочинять runtime/memory. Составь список функций/скриптов, которые понадобится адаптировать, и ключевых regression checks (best checkpoint validation, отсутствие A/B в selection, source normalization, Huang faulty handoff, unchanged historical inputs).

8. Перепроверь counts, partition/exclusion/overlap checks, hashes и принадлежность исходным memberships независимо от генератора split. Сохрани PRELIGHT_AUDIT.json (или PREFLIGHT_AUDIT.json), INPUT_MANIFEST.json, SPLIT_COUNTS.csv, DATA_USE_AUDIT.csv, proposed membership, IDENTITY_AND_LEAKAGE_AUDIT.json, draft protocol и короткий PREFLIGHT_REPORT_RU.md/STATUS.md. Различай evidence, assumption и unknown. Статус — READY_FOR_REVIEW либо NEEDS_DESIGN_DECISION с конкретной причиной; недостаток данных нельзя скрывать формальным PASS.

В финале простым русским языком ответь: есть ли действительно отдельные данные; что старая процедура не разделяла; какой контроль рекомендуешь; какие данные куда пойдут; сколько новых обучений нужно; какие ограничения останутся. Обнови активный план только до «preflight выполнен, ожидает проверки». После этого остановись: обучения, calibration fitting, новый inference и статью пока не запускать.
