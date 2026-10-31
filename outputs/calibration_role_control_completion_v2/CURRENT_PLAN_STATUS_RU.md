# Сводка текущего исследовательского плана

Состояние на 20 октября 2026 года. Calibration completion принят после независимой проверки всех 700 групп points/summaries и выбранных настоящих draws; см. REVIEW_NOTES_20261020.md. Основная ограниченная вычислительная очередь завершена с явно сохранёнными ограничениями.

## Выполнено

- Identity rederivation sensitivity выполнен и принят: для готовых моделей проверено влияние повторного расчёта отведений.
- Three-swap mixture recovery v2 выполнен; неисправная Huang input chain исправлена и повторно проверена для затронутых predictions.
- Calibration role control: девять fits (ordinary, physics_aware p050, combined × 3 seeds), model_selection / calibration_A / threshold_B source-record roles, 36 A-only Platt fits и B-only q90 thresholds; predictions Ningbo/HEEDB сохранены и проверены.
- Completion добавил недостающие 400 bootstrap-групп и восстановил 700-группную статистику; initialization seed и порядок доступа к данным раскрыты. Пакет принят после review.

## Ограничено

- Source records разделены между model selection, A и B; patient-disjointness Ningbo не подтверждена, так как patient IDs отсутствуют. HEEDB — ранее исследованная историческая sensitivity cohort.
- Declared seeds задавались после конструктора модели: начальная инициализация не контролировалась fit seed и initial RNG states не сохранены. Нынешние fits пригодны как фиксированные checkpoints, но не как matched initializations.
- Test detector cache Ningbo существовал до финального exact-objective calibration lock. Сам fit по зафиксированному коду использовал только A/B; отсутствие более раннего test access не заявляется.
- HEEDB labels — proxy; simulated lead reversals и полученные интервалы не доказывают клиническую эффективность. Bootstrap uncertainty условна на сохранённых fits/calibrators.
- Историческое сравнение с прежними fits также меняет initialization/training trajectory, sample sizes и выбранные epochs; его нельзя приписать одному calibration-selection effect.

## Ещё не выполнено / ждёт решения

- Получить полный доступ к публикации Cao и принять обоснованное решение, включать ли её implementation как direct baseline. До получения источников baseline не считается реализованным.
- Получить ответы клинициста по значениям TWA/SB/STach, SNOMED/12SL proxy mappings и правилам negatives; уточнить цель — сохранение intended clean-ECG interpretation или underlying disease. Использовать существующий CLINICIAN_REVIEW_PACKET.md и документацию кодов. Ответы ещё не получены; подтверждённые реальные miswirings остаются отдельным optional усилением.
- Разобрать реальные miswirings, leave-one-swap-out, новые архитектуры и прочие optional направления — только после отдельного решения; они этим пакетом не запускались.
- Рукопись отложена пользователем. Статья и литературный вывод этим пакетом не готовились.
