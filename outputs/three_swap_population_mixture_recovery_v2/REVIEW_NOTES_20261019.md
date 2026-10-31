# Восстановление принято после проверки

Дата: 19 октября 2026. Recovery v2 принят. Ошибка Huang input path исправлена; прежние выводы v1 о резком ухудшении Huang на новой смеси недействительны. Новое обучение и полный inference при review не выполнялись; выполнен ограниченный независимый GPU replay.

## Независимая проверка

- Все 107 исходных output hashes recovery совпали; 143 frozen input hashes и соответствующий input manifest проверены. Canonical fingerprint и SHA-256 файла протокола воспроизведены отдельно.
- Coverage содержит 96 фактических prediction наборов с корректными input/final hashes. Все 90 незатронутых probability arrays совпадают с исходными точно; 6 новых файлов имеют правильные recovery bindings.
- Labels и IDs заново восстановлены из исходных Ningbo membership/canonical records и HEEDB physician codes. HEEDB содержит прежние 10 000 уникальных patient IDs.
- Проверены все 576 condition и 2 016 mixture rows: 33 696 metric values, 2 592 thresholds, 12 096 mean/SD values. Дополнительно пересчитаны 700 полных точек, 2 100 bootstrap summary значений и 2 100 настоящих bootstrap spots (draws 0/417/999). Максимальное расхождение spot calculations — 5.69e-14.
- Отдельная явная NumPy реализация физических swaps и predicted inverse, без импорта correction/adapter helpers, воспроизвела новые Huang scores на первых/средних/последних source-aligned блоках всех 3 seeds и 2 faults: 18 GPU replay checks, max probability difference 0.0. Использовались прежние detector probabilities и замороженные ordinary weights/normalization. Сам neural detector при review повторно не запускался; его сохранённый block replay и bindings проверены.
- Все 107 исходных файлов v1 сохранили hashes. Все 610 незатронутых bootstrap groups совпали в допуске 1e-12; maximum difference 3.33e-16.

Небольшое уточнение формулировки отчёта: неизменённые score arrays совпали точно, но bootstrap draws совпали до машинной точности, не все буквально побитово. На вывод это не влияет.

## Научный результат

Huang больше не показывает искусственного провала на смеси. На прежней HEEDB при equal-three-swap q=.01/.1 combined имеет более высокий TWA AP и меньший calibrated Brier, чем Huang; соответствующие условные интервалы разностей не пересекают ноль. Но combined даёт примерно 7.25/8.58 дополнительных FP на 1 000 записей и имеет более высокую чувствительность. Это компромисс, а не универсальное превосходство.

При q=1 Huang лучше combined по AP, calibrated Brier и FP/1000. Для p050 AP-разность с Huang не установлена по CI на показанных q=.01/.1/1; в q=1 точка Huang выше. Состав трёх swaps не вызвал прежнего кажущегося collapse; близость RA_LL-only и equal-three-swap — описательное наблюдение, не доказанная эквивалентность сценариев.

Все прежние ограничения остаются: previously accessed HEEDB, proxy labels, simulated faults, fixed checkpoints/calibrators, record bootstrap Ningbo и три seeds. Калибровка ещё не отделена от исторического checkpoint selection; recovery это не исправляет.

## Следующий этап

Calibraton preflight по активному плану: проверить пригодные source holdouts, историю их использования, class counts и возможность раздельных selection/calibration/threshold частей. Сейчас требуется проектирование и проверка данных, без обучения, fitting или нового test/HEEDB evaluation. Задание — LUNA_PROMPT_CALIBRATION_PREFLIGHT_20261019.md в корне проекта.

Review artifacts добавлены после исходного manifest: independent_review_20261019.py и INDEPENDENT_REVIEW_AUDIT_20261019.json; review_input_path_replay_20261019.py и INDEPENDENT_INPUT_PATH_REPLAY_20261019.json; review_unaffected_20261019.py и INDEPENDENT_UNAFFECTED_AUDIT_20261019.json. Исходный manifest не перезаписан.
