# Исправленная серия calibration-role control

**Статус:** все 9/9 запусков завершены; отбор checkpoint прошёл; калибровки, пороги, predictions и статистика завершены и прошли вычислительный аудит. Пакет ожидает пользовательской проверки.

## Запуски

| Метод | Seed | Лучшая эпоха | Validation TWA AUPRC | Validation TWA AUROC | Эпох выполнено | Остановка | Обучение, с | Пик выделенной GPU памяти |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| ordinary | 104729 | 11 | 0.661401 | 0.919287 | 16 | patience 5 | 542.5 | 317 MiB |
| ordinary | 130363 | 16 | 0.680863 | 0.925860 | 21 | patience 5 | 509.8 | 317 MiB |
| ordinary | 155921 | 16 | 0.660257 | 0.922577 | 21 | patience 5 | 508.4 | 317 MiB |
| physics_aware_p050 | 104729 | 14 | 0.666143 | 0.922010 | 19 | patience 5 | 454.2 | 317 MiB |
| physics_aware_p050 | 130363 | 14 | 0.659326 | 0.922033 | 19 | patience 5 | 484.4 | 317 MiB |
| physics_aware_p050 | 155921 | 12 | 0.657606 | 0.917551 | 17 | patience 5 | 406.1 | 317 MiB |
| combined | 104729 | 7 | 0.657930 | 0.917175 | 12 | patience 5 | 285.9 | 317 MiB |
| combined | 130363 | 26 | 0.671928 | 0.923797 | 30 | максимум 30 эпох | 725.1 | 317 MiB |
| combined | 155921 | 19 | 0.672578 | 0.922354 | 24 | patience 5 | 565.6 | 317 MiB |

Время обучения суммарно: **4,481.955 с (74.70 GPU-минуты)** на NVIDIA GeForce RTX 5070 Laptop GPU. Память — `torch.cuda.max_memory_allocated`, не полное потребление устройства.

| Метод | Seed | Best checkpoint | SHA-256 checkpoint | SHA-256 initialization |
|---|---:|---|---|---|
| ordinary | 104729 | `fits/ordinary/seed_104729/checkpoint.pt` | `dd5733ac239045ff8b6bd0a1e37914289a897e51a7e02ca0ab6d7ca7ba3becd0` | `705a6b3d82bf21251f61f92d620ef0c975b35ea0735d928522d136294c0c5429` |
| ordinary | 130363 | `fits/ordinary/seed_130363/checkpoint.pt` | `4d5e3e36daee890b69279bf3e7380cf393e02b62a12839645bfcbdc1df5790ba` | `70e7097dcafae812f49882ae1dcaddbc6039dfaeacbe7c93d3622e491d7975ce` |
| ordinary | 155921 | `fits/ordinary/seed_155921/checkpoint.pt` | `d3a041d49c315204a33f4251a7a1008c1cb0bd0157fb695d9f0b4c44204b3bbc` | `2f6da4e634c829392aba74d2f766fdcdab3085d9159d6c9a294ad11c1c3bcb63` |
| physics_aware_p050 | 104729 | `fits/physics_aware_p050/seed_104729/checkpoint.pt` | `cb2162e786d52de198921a9fe6d800adf93ced2e1942171bc09a044e54166584` | `705a6b3d82bf21251f61f92d620ef0c975b35ea0735d928522d136294c0c5429` |
| physics_aware_p050 | 130363 | `fits/physics_aware_p050/seed_130363/checkpoint.pt` | `530b9380a83ef5f01ac42ddaed6831e3cb31e928eadb5d1b507aa250974a3033` | `70e7097dcafae812f49882ae1dcaddbc6039dfaeacbe7c93d3622e491d7975ce` |
| physics_aware_p050 | 155921 | `fits/physics_aware_p050/seed_155921/checkpoint.pt` | `97d1c8d7db83a12e73b7acda8863a22208992e65cef6d616c7343e4f129d4245` | `2f6da4e634c829392aba74d2f766fdcdab3085d9159d6c9a294ad11c1c3bcb63` |
| combined | 104729 | `fits/combined/seed_104729/checkpoint.pt` | `25e864d30c7befbb44ca4b0523a6523858941be88f5333ae9ed7ea3e7d73d79b` | `705a6b3d82bf21251f61f92d620ef0c975b35ea0735d928522d136294c0c5429` |
| combined | 130363 | `fits/combined/seed_130363/checkpoint.pt` | `e99b1aa5a45a19310fa24f9cae7e35359f7fe85c1e4b041dfc61cb4856dde9b5` | `70e7097dcafae812f49882ae1dcaddbc6039dfaeacbe7c93d3622e491d7975ce` |
| combined | 155921 | `fits/combined/seed_155921/checkpoint.pt` | `06ec9874aeb8db6b6ba0cecc00890432d97aeacae2709ca6adb4d83372a40dd6` | `2f6da4e634c829392aba74d2f766fdcdab3085d9159d6c9a294ad11c1c3bcb63` |

`TRAINING_STATUS.json` содержит точные hashes всех checkpoints, histories и preprocessing для каждого запуска. Внутри одного seed начальные тензоры одинаковы во всех трёх методах, как и ожидается для одной архитектуры и одинакового seed; между разными seeds веса различаются.

## Порядок и проверка воспроизводимости

Все seeds Python, NumPy, Torch CPU и CUDA задаются до `ECGResNetSmall()`. До и после создания модели, а также перед обучением сохраняются RNG-состояния и hash начальных весов, рассчитанный по имени, dtype, форме и байтам каждого тензора. Проверки на 18 свежих процессах подтвердили одинаковую инициализацию между методами для одного seed, одинаковый результат при повторе процесса, различие seeds и независимость от RNG-состояния до явного seed. Отдельная двухэпоховая проверка подтвердила совпадение непрерывного и прерванного/возобновлённого состояния модели и генераторов. Она не является независимым полным повтором девяти научных траекторий.

Проверены: best-checkpoint replay для 9/9; 36/36 калибровочных и пороговых фиксаций; точный порядок/условия и hashes четырёх переиспользованных Huang detector cache; целостность 447 исторических файлов (0 несовпадений); 96 prediction-файлов по двум когортам, четырём методам, трём seed и четырём условиям; 700 bootstrap-групп по 1,000 draws; совпадение weighted AP со sklearn на выбранных фактических resamples (максимальная абсолютная разность `6.44e-15`); нулевая разность между сценариями при q=0.

## Фактическое время и воспроизведение

- Обучение: 4,481.955 с; checkpoints и журнал эпох находятся в `fits/`.
- Inference: Ningbo_test — 396 с, HEEDB — 378 с (из интервала времени создания первого и последнего prediction-файла каждой когорты; 96 файлов). Повторный статистический проход использовал их без повторного inference.
- Метрики и 700 bootstrap-групп: 331 с для полного прохода `evaluate_full.py`, включая повторную проверку сохранённых predictions; по cohort архивы draws записаны после вычисления.
- Поддерживаемая полная команда на свежей подготовленной копии: сначала `python outputs/calibration_role_control_seed_corrected_v1/train_all_fresh.py --check-inputs`, затем — только в копии без checkpoints — `python outputs/calibration_role_control_seed_corrected_v1/train_all_fresh.py`; после обучения `python outputs/calibration_role_control_seed_corrected_v1/postprocess.py` и `python outputs/calibration_role_control_seed_corrected_v1/evaluate_full.py`. Запускать с `.venv` проекта. Проверочный режим прошёл в текущем проекте; он не обучает. `train_all_fresh.py` намеренно отказывается перезаписывать существующие checkpoints. В wrapper `control_replay.py` сверяет все 65 frozen научных входов по исходным hashes; три текущих документа проверяются отдельным происхождением, включая точные проверенные исторические snapshots. Само обучение выполняется неизменёнными `control.train_one()` функциями из исходного frozen `control.py`, его hash совпадает с привязкой всех девяти checkpoints. Подробности: `INPUT_VALIDATION_AMENDMENT_001.md` и `AMENDMENT_003.md`.

Методный отпечаток frozen протокола: `ae1fc61c15eeffbdc0847c4ba69f78cf1a82ee24e5a0ca5c40e686c4ad40654a`; hash самого frozen протокола: `d1edfd3e55af95cebd2c93d596e43809ba5670c2ee8e2ec43bc568b90a0a8e14`.

Старая серия остаётся исторической; исходные начальные веса неизвестны. Поэтому различия старой и исправленной серий — анализ чувствительности на уже изученных данных, а не чистая причинная оценка ошибки seed. Ни новые пациенты, ни новая независимая внешняя валидация не добавлены.
