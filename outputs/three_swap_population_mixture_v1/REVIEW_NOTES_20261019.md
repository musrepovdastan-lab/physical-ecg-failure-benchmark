# Результат проверки: требуется точечное восстановление

**Пакет не принят: FAIL_HUANG_HEEDB_NEW_FAULT_INPUT_PATH.** Численные таблицы правильно описывают сохранённые predictions, но шесть новых prediction файлов Huang получены из неверного входного сигнала. Выводы о сравнении с Huang в HEEDB equal-three-swap сценарии при q>0 пока использовать нельзя.

## Ошибка и доказательство

В run_inference.py, infer_huang_classifier(), для новых RA_LA_reversal и LA_LL_reversal код делает:

`corrected_batch(original_raw_batch, detector_classes_for_reversed_batch)`

В detector_worker() классы при этом действительно получены для ECG после соответствующего swap. Перед predicted inverse в диагностической ветке initial swap пропущен. Правильная цепочка: original raw → simulated incoming fault → detector argmax → inverse applied to that same faulty ECG → ordinary diagnostic classifier. Истинный тип fault используется только для создания входа; выбирать исправление по нему нельзя.

Независимый focused GPU replay для всех 3 seeds и 2 новых условий воспроизвёл ошибочные сохранённые вероятности точно (max difference 0.0 на первых 32 записях исторического batch128). Прогон корректной цепочки на том же блоке дал maximum probability differences от 0.2646 до 0.9684 относительно ошибочного кэша. Это не эффект округления или batch size.

Два способа генерации incoming swap совпали точно. Инженерное применение правильного inverse к faulty ECG вернуло identity-derived clean с небольшим float32 rounding: max 0.0001221 raw units для RA_LA и около 3.46e-15 для LA_LL. Это только инженерный контроль; oracle correction не использовалась для новой оценки метода.

Доказательство: INFERENCE_PATH_REVIEW_20261019.json; воспроизводящий код: review_huang_input_path_20261019.py. Полный inference и новое обучение при проверке не выполнялись.

## Что проверено и что остаётся валидным

Независимо проверены все 107 исходных output hashes, canonical/file protocol hashes, 44 неизменяемых input hashes, 4 дополнительных source bindings и 32 waveform hashes. Активный план изменён после freeze по разрешённому статусному обновлению; это не научный вход, и его новое содержимое нельзя считать старым frozen hash.

Из исходной документации заново восстановлены labels и порядок Ningbo, а также HEEDB physician codes и 10 000 уникальных patient IDs. Все 576 condition rows и 2 016 mixture rows пересчитаны: 33 696 metric values совпали; 2 592 thresholds совпали при round-trip CSV parsing. Дополнительно проверены 12 096 mean/SD значений, 700 paired points, 2 100 summaries и 2 100 настоящих bootstrap spots (draws 0/417/999).

Этот PASS относится к арифметике сохранённых scores, а не к правильности inference pipeline. Поэтому он не отменяет выявленную ошибку.

По проверенному коду дефект относится к 6 новым HEEDB Huang files (2 faults × 3 seeds). Исторические Huang clean/RA_LL, весь Ningbo и остальные три метода этим дефектом не затронуты. Их не нужно переобучать или целиком пересчитывать inference. RA_LL-only anchors не могли поймать ошибку, поскольку они не используют два новых faulty условия.

## Дополнительные замечания

- INPUT_COVERAGE_COMPLETION.csv оставляет placeholders для Huang HEEDB; preflight artifact paths для ordinary местами ошибочно указывают masking. В recovery нужно заполнить фактические файлы/hashes, отдельно сохранив исторический preflight.
- Confusion/false-correction сводка в этом пакете содержит HEEDB, но не Ningbo. Можно добавить Ningbo из готовых опубликованных detector caches без нового inference.
- Полные времена Huang не записаны; не придумывать runtime. Общий прежний benchmark не заменяет отсутствующие timings этого запуска.
- Финальный текст о недостатках Huang и численных лидерах должен быть переписан по восстановленным данным. Предварительное кажущееся ухудшение Huang на новой смеси недостоверно из-за ошибки входа.

## Следующий шаг

Отдельный recovery_v2: переиспользовать проверенные detector probabilities и все незатронутые scores; заново получить только 6 Huang HEEDB диагностических predictions, пересчитать зависимые метрики/mixtures/contrasts/CI, проверить неизменность незатронутых результатов. К calibration preflight переходить после проверки recovery. Подробное поручение — LUNA_PROMPT_THREE_SWAP_RECOVERY_20261019.md в корне проекта.
