# Preflight принят с исправлениями описания

Дата: 19 октября 2026. Предложенный split принят для ограниченного нового контроля: исходный train 24 430 без изменений; model_selection 1 744; calibration_A 872; threshold_B 872. Новое обучение, inference и fitting при review не выполнялись.

## Независимая проверка

Все 67 исходных input hashes совпали до обновления активного плана. Для 34 899 canonical records проверены labels и waveform hashes; для всех 3 488 proposed records независимо проверены source membership/index, labels, waveform/record hashes и отсутствие придуманных patient IDs. Proposed roles точно разбивают прежний validation; IDs и waveform hashes не пересекаются между частями и исходными train/test. Шесть исключённых conflict records не возвращены.

Пересчитаны все 54 строки counts, role-membership fingerprints, 6 rank/биномиальных precision estimates и заявленная экстраполяция бюджета. В A/B по 129 положительных TWA, по 142 STach; SB 317/316 положительных. Числа позволяют выполнить запланированный fit, но не гарантируют точную калибровку или стабильную оценку каждого operating point. Ningbo patient identity отсутствует в проверенных metadata: доказано только разделение record/waveform, не пациентов.

## Что исправлено

1. В исходном preflight ошибочно говорилось, что HEEDB использовали для fitting калибровки/порогов. В проверенных предыдущих пакетах fit и q90 selection выполнялись на Ningbo, а к HEEDB переносились фиксированные параметры. HEEDB ранее просматривалась и не является новой untouched evaluation, но это не прямое fitting на ней. Исправлены PREFLIGHT_REPORT_RU.md, DATA_USE_AUDIT.csv и SOURCE_HOLDOUT_ELIGIBILITY.json.
2. STach был ошибочно назван главным ограничением precision. Меньше всего positives у TWA (129), меньше всего negatives у SB (555/556). Исправлен текст отчёта; численные counts/precision исходно правильные.
3. С учётом Huang будущих endpoint Platt fits и q90 selections будет по 36: 27 для трёх основных методов и ещё 9 для Huang. Новых обучений по-прежнему 9. Уточнён бюджет.
4. В draft добавлены конкретные прежние selection adequacy gates и численные параметры Platt/q90. Роли A/B и sample split не изменялись. Перед обучением нужно зафиксировать новый execution protocol в отдельном пакете, не считать preflight уже выполненным экспериментом.

Исходные шесть изменённых документов и исходный активный план сохранены в archive_before_review_20261019/. PROPOSED_MEMBERSHIP.csv и все научные исходные данные не менялись. build_preflight.py оставлен как исходный генератор для прослеживаемости: его не следует запускать поверх принятых документов, поскольку он вернёт старые ошибочные формулировки.

## Решение и границы

Рекомендуется выполнить 9 новых обучений (ordinary, p050, combined × 3 seeds) и дальнейшее A-only calibration/B-only threshold selection, включая Huang с новым ordinary classifier. Все checkpoints фиксируются до A/B inference, затем calibrators/thresholds фиксируются до внешнего оценивания.

Это проверка устойчивости прежних сравнительных выводов при разделении ролей данных для новых checkpoints. Она не делает всю историю разработки нетронутой и не доказывает patient independence. Сравнение с прежними числами меняет также размер selection/calibration частей и потенциально выбранные checkpoints, поэтому нельзя приписать всю разницу только устранению selection dependence или объявить её измерением прежнего bias.

Ориентир по прошлым сходным обучением: 2.77–5.10 часа на 9 fits, медианный ориентир 3.18 часа; это оценка, не обещанный runtime и без будущих calibration/inference/bootstrap затрат. Новое ограниченное задание — LUNA_PROMPT_CALIBRATION_CONTROL_20261019.md в корне проекта.

Аудит воспроизводится independent_review_20261019.py; результат INDEPENDENT_REVIEW_AUDIT_20261019.json. Review и исправления не являются новым обучением.
