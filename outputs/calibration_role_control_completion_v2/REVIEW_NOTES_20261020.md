# Completion принят: ограниченная вычислительная очередь завершена

Дата: 20 октября 2026. Полный calibration completion принят после независимой проверки. Новые обучения, inference и refit при review не выполнялись. Это закрытие конкретного ограниченного плана, а не исполнение каждой дополнительной идеи исходного большого списка и не готовность клинических claims.

## Независимая проверка

- Совпали hashes и размеры 29 исходных completion outputs, 125 scientific inputs и всех 293 исходных артефактов training/control v1. Frozen protocol и amended runner проверены отдельно.
- Полное множество ключей совпало с заранее заданными 700 комбинациями: 2 когорты × 2 сценария × 7 q × 5 contrasts × 5 metrics. У всех по 1 000 конечных draws.
- Все 700 full points пересчитаны по прежним predictions и frozen calibration/threshold values. Максимальная разница — 5.69e-14.
- Пересчитаны 2 100 bootstrap summary values и 2 100 настоящих draw contrasts (0/417/999), включая все 400 новых групп. Maximum spot difference — 1.14e-13.
- Все прежние 300 vectors сохранены точно, provenance reused300/new400 совпал. Оба сценария q=0 имеют одинаковые draws для каждой пары cohort/contrast/metric.
- Новые русские отчёты читаются как UTF-8, содержат кириллицу и не содержат утраченных фрагментов вида '????'. Модели, splits, коэффициенты и thresholds не менялись.

Подробности: independent_review_20261020.py и INDEPENDENT_REVIEW_AUDIT_20261020.json. Эти review файлы добавлены после исходного manifest, который не переписан. Неповторённые стадии inference/training опираются на ранее выполненный review и подтверждённую неизменность артефактов.

## Выводы

Защитные режимы сохраняют пользу на simulated faulty ECG, но конкретный лучший режим меняется между cohort/q и training series. В новом role control на HEEDB equal-three q=.1 p050 имеет AP около .5881 против .5742 у Huang (conditional interval разности исключает ноль); одновременно больше sensitivity и FP. На Ningbo при том же q combined имеет AP около .6799 против .6688 у Huang. При q=1 point leader AP — Huang в обеих когортах; point leader не всегда означает установленную разницу для каждой пары.

Это аргумент за исследование компромиссов и несовпадения видов устойчивости, а не универсальное превосходство собственной модели. Начальные веса не контролировались declared fit seed, A/B размеры и выбранные checkpoints изменились; old→new сдвиг нельзя интерпретировать как чистый эффект calibration independence. Сохранённые модели пригодны для воспроизводимого оценивания, но fresh-training по одному seed полностью не воспроизводится.

Остаются прежние границы: record/waveform split Ningbo, неизвестные patient IDs, previously accessed HEEDB с proxy labels, simulated faults, conditional/unadjusted CIs и отсутствие real-miswiring clinical validation. Ни одна из этих границ не исчезает от завершения статистики.

## Небольшие документальные поправки review

В FINAL_REPORT_RU.md уточняется, что исходный train не менялся: различаются selection/calibration/threshold sizes и trajectories, а не размер train. В CURRENT_PLAN_STATUS_RU.md задача врача уточняется до endpoints, SNOMED/12SL proxy mappings, правил negatives и intended clean interpretation vs underlying disease; создание/подтверждение miswiring датасета не является обязательным запросом врачу. Исходные версии сохранены в archive_before_review_20261020/.

Верх активного плана в корне был вновь повреждён при статусной записи. Восстановлен читаемый UTF-8 статус с сохранением содержательной очереди; исходный файл также сохранён в архиве. Это изменения документации после проверки исходных hashes.

## Дальше

Новые обучения автоматически не запускать. Следующий приоритет — согласовать точный научный вклад с ближайшей литературой, честно закрыть статус доступа Cao и принять окончательное решение о прямом baseline. Ранний CAO2020_FULLTEXT_REVIEW.md основан на indexed extraction при недоступном PDF, несмотря на заголовок; SOURCE_ACCESS_LOG.json отмечает непроверенные страницы/figures. Не считать это новым полученным полным PDF и не повторять уже сделанные тесты.

Параллельная открытая часть — ответы клинициста по существующему CLINICIAN_REVIEW_PACKET.md. Их получение не подтверждено. Real miswirings/LOSO/ещё архитектуры остаются optional. Рукопись пользователь отложил. Задание следующего невычислительного пакета — LUNA_PROMPT_NOVELTY_POSITIONING_20261020.md.
