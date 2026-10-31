# Calibration role control: вычисления проверены, пакет требует завершения

Дата: 20 октября 2026. Новые 9 fits и сохранённые 96 prediction наборов проверены как существующие модели/результаты. Полное исполнение плана пока НЕ принято: интервалы отсутствуют для 4 из 7 q; нарушена управляемость initialization seed; текстовые отчёты повреждены. Следующий шаг — ограниченное завершение анализа и исправление provenance, без автоматического нового обучения.

## Независимо подтверждено

- Все 293 исходных output hashes/размера, 40 input hashes и 3 frozen runner hashes совпали. File hash и canonical fingerprint протокола проверены отдельно.
- Модель выбиралась по model_selection. Все 9 histories и best epochs совпали с lock; selection caches имеют правильные IDs/labels/checkpoint bindings, их AP/AUROC воспроизведены.
- A/B имеют разные записи и правильные source roles. Все 36 Platt coefficients заново получены из A с точным ранее заданным objective и совпали; все 36 negative q90 порогов получены из B и совпали. Сохранённая калибровка не использует test/HEEDB labels.
- Восстановлены labels/IDs Ningbo test и HEEDB; проверены все 96 prediction payloads и checkpoint/preprocessing bindings.
- Пересчитаны все 24 624 значений condition/mixture метрик; все 300 имеющихся paired points, 900 summary values и 900 настоящих draws (0/417/999) воспроизведены. Максимальное расхождение — около 1.14e-13. Mean/SD таблицы дополнительно проверены из seed-specific строк.
- Независимый GPU replay по явно написанным NumPy physical equations: 72 проверки HEEDB. Первые исторические блоки всех четырёх методов, трёх runs и четырёх условий; дополнительно middle/last для Huang, включая clean false correction. Maximum probability difference 0.0. Никакого полного нового inference/обучения при review не выполнялось. Neural detector и Ningbo GPU inference при review заново не прогонялись.

## Незавершённый scope

В evaluate.py bootstrap() цикл q задан как (.01,.1,1), хотя frozen protocol и задание содержат [0,.001,.005,.01,.05,.1,1]. Имеются 300 групп intervals/draws вместо 700. Не хватает 400 групп для q=0,.001,.005,.05, по обоим сценариям/когортам/пяти contrasts/пяти метрикам. Point metrics для этих q уже имеются. Затронут только анализ готовых scores; новое обучение и inference не нужны.

FINAL_REPORT_RU.md, TRAINING_REPORT.md и часть STATUS.md содержат буквальные вопросительные знаки вместо русского текста. Это повреждение сохранённого содержимого, а не полезный отчёт в другой кодировке. Требуется заново сформировать русский UTF-8 текст из проверенных CSV и provenance, сохранив старый.

## Initialization seed: обнаруженное ограничение

В frozen control.py train_one() ECGResNetSmall создаётся на строке109, а random/NumPy/Torch/CUDA seed устанавливается на строке111. Следовательно, записанный fit seed не задаёт начальные веса. Небольшой независимый CPU counterexample подтвердил: одинаковый declared seed после конструктора при разных ambient Torch RNG даёт разные initial parameter hashes; установка seed до конструктора устраняет различие.

Это не свидетельство A/B leakage, не делает saved probabilities арифметически неверными и не требует выбросить 9 checkpoints. Но нельзя заявлять точное воспроизведение обучения по одному seed или matched initialization между методами. Доступные saved checkpoints воспроизводят оценивание. Начальное RNG/state provenance нужно явно описать; не выдумывать отсутствующие initial states и не менять старый frozen код задним числом. Будущую правильную последовательность можно документировать отдельно. Перезапуск всех9 fits не входит в следующий пакет.

## Порядок фиксации

Все модели были locked до A/B scoring. Exact-objective calibration refit был технической заменой первоначальной эквивалентной реализации, без новых memberships/outcome-driven fitting. Однако Ningbo test detector cache уже существовал до окончательного exact-objective lock. Нельзя расширять это до утверждения, что ни один test waveform не открывался до последнего lock. По проверенным fitting путям test/HEEDB scores/labels не служили входом калибровке. Этот порядок нужно честно отразить, а не исправлять timestamps задним числом.

## Что уже можно сказать по имеющимся результатам

В новом контроле защитные методы сохраняют ranking преимущество перед ordinary на повреждённых сценариях. Но прежний конкретный лидер не закрепился универсально: в equal-three-swap q=.1 на Ningbo combined имеет AP около .680, а на HEEDB p050 около .588 и выше Huang примерно на .0139, с условным CI примерно [.0024,.0240]. На HEEDB p050 даёт больше FP/1000 и более высокую чувствительность. Exact Brier comparisons и остальные q следует обсуждать после полного завершения таблиц.

Сдвиг относительно исторических результатов не является оценкой чистого эффекта устранения selection dependence: изменились размеры selection/A/B, выбранные checkpoints и training trajectories; initialization не полностью контролировалась declared seeds. Интервалы условны на этих фиксированных моделях/калибраторах, а не на всей вариации обучения.

## Дальше

Выполнить completion_v2 по LUNA_PROMPT_CALIBRATION_COMPLETION_20261020.md: добавить 400 недостающих interval groups, сохранить имеющиеся300 и все predictions, явно оформить seed/lock-order ограничения, восстановить UTF-8 отчёт, обновить сводку планов. Новое обучение, inference, новая архитектура и написание статьи этим заданием не запускаются.

Review artifacts: independent_review_20261020.py/INDEPENDENT_REVIEW_AUDIT_20261020.json; review_prediction_replay_20261020.py/INDEPENDENT_PREDICTION_REPLAY_20261020.json; review_seed_order_20261020.py/SEED_ORDER_REVIEW_20261020.json. Они добавлены после исходного manifest, который не переписан.
