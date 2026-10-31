# Завершение calibration role control

## Что сделано

В отдельном пакете завершена недостающая статистика из уже сохранённых predictions: добавлено 400 paired bootstrap-групп для q=0, 0.001, 0.005, 0.05. Сохранённые 300 групп для q=0.01, 0.1, 1.0 сохранены и сверены. Полный набор содержит 700 групп по 1 000 draws на группу для двух когорт, двух сценариев, пяти contrasts и пяти метрик. Ни обучение, inference, refit калибровки, ни выбор порогов не запускались.

Frozen completion protocol SHA-256: `fd86abdb3a3ab9e762a9245a369f521c4759f3c1cec0f1c5dc6c899ad7b2aece`. Исправление порядка условий зафиксировано отдельно в `COMPLETION_PROTOCOL_AMENDMENT_001.json`; протокол и научный scope не менялись.

### Численная проверка

Из сохранённых predictions заново рассчитано 2,016 TWA point metric values по всем q, сценариям, методам и seeds; максимум расхождения с сохранённой mixture-таблицей — 7.26e-14. Исходные 300 draw vectors сохранены численно без изменения (max delta 0.0). Для новых групп tie-group AP сверена со sklearn на реальных draws 0/417/999: 576 checks, max delta 7.55e-15. Все 400 новые группы покрыты; оба q=0 сценария присутствуют и дают одинаковые draws.

Ранее проведённая независимая проверка исходного пакета остаётся подтверждённой: 24 624 сохранённые метрики, все 300 исходных bootstrap summaries/draw spot checks, fit/threshold locks, membership роли и 96 prediction bindings проверялись review-артефактами. Здесь condition/mixture сводки повторно не переписывались; их v1 hashes и lineage зафиксированы в `UNCHANGED_ARTIFACTS_AUDIT.json`.

### Результаты

Все ключевые числа (чистые q=0 и RA–LL-only/equal-three q=0.01, 0.1, 1.0) приведены в [таблицах](KEY_RESULTS_TABLES_RU.md) и CSV. Победители по точечной средней AP меняются между когортами и q:

- Ningbo_test RA_LL_only q=0: combined (AP 0.6835)
- Ningbo_test RA_LL_only q=0.01: combined (AP 0.6831)
- Ningbo_test RA_LL_only q=0.1: combined (AP 0.6788)
- Ningbo_test RA_LL_only q=1: Huang_full_pipeline (AP 0.6560)
- Ningbo_test equal_three_swap q=0: combined (AP 0.6835)
- Ningbo_test equal_three_swap q=0.01: combined (AP 0.6832)
- Ningbo_test equal_three_swap q=0.1: combined (AP 0.6799)
- Ningbo_test equal_three_swap q=1: Huang_full_pipeline (AP 0.6616)
- HEEDB RA_LL_only q=0: physics_aware_p050 (AP 0.5902)
- HEEDB RA_LL_only q=0.01: physics_aware_p050 (AP 0.5898)
- HEEDB RA_LL_only q=0.1: physics_aware_p050 (AP 0.5855)
- HEEDB RA_LL_only q=1: Huang_full_pipeline (AP 0.5838)
- HEEDB equal_three_swap q=0: physics_aware_p050 (AP 0.5902)
- HEEDB equal_three_swap q=0.01: physics_aware_p050 (AP 0.5900)
- HEEDB equal_three_swap q=0.1: physics_aware_p050 (AP 0.5881)
- HEEDB equal_three_swap q=1: Huang_full_pipeline (AP 0.5842)

На Ningbo при equal-three q=0.1 combined превышает Huang по AP на 0.0111 (95% CI [0.0005, 0.0213]); у p050 разность AP относительно Huang имеет интервал, включающий ноль. На HEEDB при q=0.1 p050 выше Huang по AP на 0.0139 (95% CI [0.0024, 0.0240]). При том же HEEDB q p050 также даёт больше sensitivity и ложных тревог; paired intervals этих разностей приведены в CSV.

Калибровка уменьшает Brier во многих сочетаниях; таблицы показывают raw и calibrated значения отдельно. Порог был перенесён на Platt-шкалу с положительным наклоном, поэтому raw и calibrated operating decisions совпадают. При q=1 на обеих когортах лидер AP смещается к Huang; защитные методы не выигрывают при каждом уровне ошибок.

Сравнение с исторической серией не является оценкой одного только устранения calibration selection dependence: одновременно отличаются initialization/training trajectories, размеры model_selection/calibration_A/threshold_B и выбранные epochs. Исходный train из 24 430 записей не менялся. Исторический point leader combined сменяется p050 для HEEDB в чистом/equal-three промежуточном диапазоне, а при q=1 лидирует Huang; на Ningbo при q=1 новый point leader также Huang. Это описательные различия, не доказательство причины.

### Что остаётся ограниченным

- Declared fit seed задавался после конструктора ECGResNetSmall. Он управлял дальнейшим training RNG, но не начальными весами; начальный ambient RNG/state и initial weight hashes в сохранённых provenance не записаны. Точные fresh fits и matched initialization между методами только по seed не воспроизводятся. Checkpoints остаются валидными как фиксированные обученные модели; их predictions проверены. Правильный порядок для будущих запусков — установить random/NumPy/Torch/CUDA seeds до создания модели.
- Все source-record роли model_selection/A/B разделены для этих fits, однако тестовый detector cache Ningbo существовал до финальной exact-objective calibration lock. Поэтому нельзя утверждать, что никакие test waveform не были доступны до последнего lock. Проверенная калибровка использовала только A и B scores/labels; test и HEEDB labels/scores в fitting code не передавались.
- Интервалы условны на трёх сохранённых fits и их калибраторах и не охватывают полную вариацию обучения/fitting. Для Ningbo patient IDs нет, поэтому bootstrap — по записям; HEEDB — историческая чувствительность с proxy labels.
- Ошибки simulated; результаты не подтверждают клиническую эффективность и не устанавливают универсального победителя.
- GPU training peak memory не сохранён. Измерение v1 было сделано после сброса счётчика при replay best checkpoint, а не во время training; точный training peak unavailable.

## Артефакты

- `PAIRED_POINTS_CI_ALL_700.csv`, `PAIRED_BOOTSTRAP_DRAWS_ALL_700.npz` — полный набор points, summaries и draws.
- `BOOTSTRAP_COVERAGE_AUDIT.json` и `BOOTSTRAP_COVERAGE_700.csv` — покрытие групп, включая provenance reused/new.
- `NUMERICAL_VERIFICATION.json`, `UNCHANGED_ARTIFACTS_AUDIT.json` — арифметика и неизменность исходников.
- `INITIALIZATION_PROVENANCE_NOTE_RU.md`, `LOCK_CHRONOLOGY_RU.md` — ограничения seed и фактический порядок locks/access.
- `CURRENT_PLAN_STATUS_RU.md` — выполнено, ограничено и оставлено на будущее.
- `MANIFEST.json` — раздельные input bindings и hashes новых outputs.
