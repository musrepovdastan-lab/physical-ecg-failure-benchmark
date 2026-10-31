# Исследование вероятности физической аугментации: итоговый отчёт

## Краткие ответы

1. **10%, 25% и 50% swaps.** Новые `physics_p010` и `physics_p025` сравнили с историческим `physics_aware` (50% swaps), `ordinary` (0%) и `combined` (25% swaps + 25% masks). На clean у новых режимов TWA AUPRC близок к историческим режимам: различия малы и не дают основания выбирать вероятность. При физически нарушенных входах оба новых режима заметно превосходят ordinary. 50% режим сохраняет более сильные средние результаты на знакомой RA_LL перестановке и части cycle условий; преимущество по маскам зависит от когорты и от того, рассматривается конкретная маска или среднее по 792 маскам.
2. **Меньшая доля повреждений.** При 10% swaps польза относительно обычного обучения сохраняется на RA_LL, обоих циклах и точной маске в обеих когортах. Для средней метрики 792 масок прирост также положительный, но скромнее, особенно в Ningbo. Это показывает перенос на изученные типы нарушений, не доказывает перенос на новые пациенты или независимую когорту.
3. **25% swaps против combined.** Доля swaps одинакова, но combined заменяет оставшуюся четверть чистых примеров масками. `physics_p025` имел почти одинаковый clean AUPRC, но ниже AUPRC на RA_LL, циклах и масках в Ningbo; на Chapman разница также отрицательна для RA_LL, cycle_slots_120 и точной маски. По среднему 792 масок `physics_p025` выше combined на Chapman, но ниже на Ningbo. Поэтому эффекта «маски помогают/мешают» без оговорок нет.
4. **Условия теста.** Сводная таблица ниже показывает TWA AUPRC по clean, знакомой перестановке, двум проверенным циклам и точной маске. Ordinary резко проседает на нарушенных сигналах. p010/p025 улучшают устойчивость, но не достигают одинакового уровня во всех стресс-условиях. Для остальных endpoint/метрик (AUPRC, AUROC, Brier, ECE15 и фиксированные пороговые показатели) доступны полные таблицы по каждому seed; отдельные метрики не следует смешивать с TWA AUPRC.
5. **Chapman и Ningbo.** Направление сравнения p010 с ordinary совпало: практически нулевое изменение clean и положительное на RA_LL, обоих циклах и точной маске. p025 против combined совпал по направлению на clean, RA_LL, cycle_slots_120 и точной маске; cycle_slots_201 и средние 792 маски различались. Это согласованность части сигналов, а не независимое подтверждение: те же заранее исследованные наборы и три seed.
6. **Что остаётся неопределённым.** Только три seed; bootstrap интервалы условны на выбранных моделях и test когортах. Интервалы не измеряют неопределённость выбора моделей и не являются доказательством равенства при пересечении нуля. Не установлена оптимальная вероятность swaps; нет оценки HEEDB, reserved test или иных новых данных. Сравнение p025 с combined изолирует замену части clean примеров масками при фиксированной доле swaps, но не все различия в процессе обучения.

## Результаты по TWA AUPRC

Среднее ± SD по трём seed; здесь приведены значения из исходных seed-specific метрик. Для 792 масок отдельный bootstrap отчёт хранит среднюю метрику по маскам, а не смешивает её с конкретной маской.

### Chapman

| Условие | ordinary | physics_p010 | physics_p025 | physics_aware | combined |
| --- | --- | --- | --- | --- | --- |
| clean | — | 0.654 ± 0.005 | 0.664 ± 0.001 | — | — |
| RA_LL_reversal | — | 0.589 ± 0.011 | 0.618 ± 0.019 | — | — |
| cycle_slots_120 | — | 0.552 ± 0.020 | 0.588 ± 0.039 | — | — |
| cycle_slots_201 | — | 0.622 ± 0.025 | 0.637 ± 0.015 | — | — |
| mask5_0_1_2_3_5 | — | 0.629 ± 0.009 | 0.627 ± 0.017 | — | — |

### Ningbo

| Условие | ordinary | physics_p010 | physics_p025 | physics_aware | combined |
| --- | --- | --- | --- | --- | --- |
| clean | — | 0.674 ± 0.011 | 0.678 ± 0.009 | — | — |
| RA_LL_reversal | — | 0.601 ± 0.009 | 0.625 ± 0.023 | — | — |
| cycle_slots_120 | — | 0.571 ± 0.011 | 0.607 ± 0.035 | — | — |
| cycle_slots_201 | — | 0.590 ± 0.006 | 0.607 ± 0.028 | — | — |
| mask5_0_1_2_3_5 | — | 0.616 ± 0.012 | 0.616 ± 0.034 | — | — |

Для ключевых условий ниже приведены те же seed mean ± SD для AUROC и Brier. Более низкий Brier лучше; AUROC не заменяет AUPRC при дисбалансе классов.

### Chapman: AUROC / Brier

| Условие | ordinary | physics_p010 | physics_p025 | physics_aware | combined |
| --- | --- | --- | --- | --- | --- |
| clean | — | 0.913 ± 0.001 / 0.130 ± 0.022 | 0.915 ± 0.003 / 0.131 ± 0.016 | — | — |
| RA_LL_reversal | — | 0.869 ± 0.005 / 0.169 ± 0.037 | 0.886 ± 0.013 / 0.176 ± 0.067 | — | — |
| cycle_slots_120 | — | 0.860 ± 0.012 / 0.193 ± 0.053 | 0.869 ± 0.025 / 0.203 ± 0.088 | — | — |
| cycle_slots_201 | — | 0.877 ± 0.017 / 0.137 ± 0.033 | 0.890 ± 0.013 / 0.165 ± 0.066 | — | — |
| mask5_0_1_2_3_5 | — | 0.893 ± 0.003 / 0.176 ± 0.051 | 0.888 ± 0.007 / 0.212 ± 0.078 | — | — |

### Ningbo: AUROC / Brier

| Условие | ordinary | physics_p010 | physics_p025 | physics_aware | combined |
| --- | --- | --- | --- | --- | --- |
| clean | — | 0.922 ± 0.003 / 0.099 ± 0.008 | 0.924 ± 0.004 / 0.094 ± 0.004 | — | — |
| RA_LL_reversal | — | 0.885 ± 0.006 / 0.121 ± 0.021 | 0.896 ± 0.012 / 0.122 ± 0.011 | — | — |
| cycle_slots_120 | — | 0.868 ± 0.007 / 0.140 ± 0.029 | 0.877 ± 0.018 / 0.138 ± 0.022 | — | — |
| cycle_slots_201 | — | 0.877 ± 0.004 / 0.125 ± 0.020 | 0.889 ± 0.012 / 0.127 ± 0.016 | — | — |
| mask5_0_1_2_3_5 | — | 0.887 ± 0.009 / 0.167 ± 0.041 | 0.885 ± 0.021 / 0.165 ± 0.025 | — | — |

### Распределение по 792 маскам

Ниже средние по seed для распределительных характеристик mask AUPRC; SD по seed доступен в исходном файле. `mean_auprc` усредняет 792 отдельных mask condition, а `exact_mask_auprc` относится к одной заранее заданной маске.

| cohort | arm | mean_auprc | median_auprc | p05_auprc | p25_auprc | p75_auprc | p95_auprc | exact_mask_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | physics_p010 | 0.5919 | 0.6216 | 0.4263 | 0.5761 | 0.6388 | 0.6530 | 0.6287 |
| Chapman | physics_p025 | 0.5837 | 0.6214 | 0.3515 | 0.5731 | 0.6430 | 0.6595 | 0.6265 |
| Ningbo | physics_p010 | 0.5963 | 0.6255 | 0.4259 | 0.5774 | 0.6435 | 0.6596 | 0.6164 |
| Ningbo | physics_p025 | 0.6043 | 0.6296 | 0.4456 | 0.5901 | 0.6463 | 0.6617 | 0.6158 |

## Парные bootstrap интервалы

Первичная метрика — TWA AUPRC. Bootstrap синхронно пересэмплирует одни и те же test индексы для сравниваемых моделей и усредняет seed-specific различия внутри повтора. `full_cohort_point_difference` — оценка по полному test набору; интервал — 2.5–97.5 percentile из 1000 повторов.

| Когорта | Контраст (A−B) | Условие | Полный test | Bootstrap mean | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Chapman | physics_p025_minus_combined | clean | 0.0053 | 0.0054 | [-0.0046, 0.0145] |
| Chapman | physics_p025_minus_combined | RA_LL_reversal | -0.0131 | -0.0130 | [-0.0264, 0.0012] |
| Chapman | physics_p025_minus_combined | cycle_slots_120 | -0.0219 | -0.0219 | [-0.0353, -0.0076] |
| Chapman | physics_p025_minus_combined | cycle_slots_201 | 0.0084 | 0.0088 | [-0.0030, 0.0208] |
| Chapman | physics_p025_minus_combined | mask5_0_1_2_3_5 | -0.0146 | -0.0145 | [-0.0253, -0.0049] |
| Chapman | physics_p025_minus_combined | mean_792_masks | -0.0474 | -0.0474 | [-0.0585, -0.0372] |
| Chapman | physics_p025_minus_physics_aware | clean | -0.0002 | 0.0001 | [-0.0125, 0.0125] |
| Chapman | physics_p025_minus_physics_aware | RA_LL_reversal | -0.0189 | -0.0182 | [-0.0324, -0.0046] |
| Chapman | physics_p025_minus_physics_aware | cycle_slots_120 | -0.0371 | -0.0366 | [-0.0521, -0.0207] |
| Chapman | physics_p025_minus_physics_aware | cycle_slots_201 | -0.0082 | -0.0077 | [-0.0200, 0.0045] |
| Chapman | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | -0.0178 | -0.0176 | [-0.0282, -0.0073] |
| Chapman | physics_p025_minus_physics_aware | mean_792_masks | 0.0158 | 0.0159 | [0.0063, 0.0256] |
| Chapman | physics_p010_minus_ordinary | clean | 0.0013 | 0.0011 | [-0.0116, 0.0143] |
| Chapman | physics_p010_minus_ordinary | RA_LL_reversal | 0.2608 | 0.2600 | [0.2261, 0.2926] |
| Chapman | physics_p010_minus_ordinary | cycle_slots_120 | 0.2855 | 0.2862 | [0.2527, 0.3218] |
| Chapman | physics_p010_minus_ordinary | cycle_slots_201 | 0.2317 | 0.2320 | [0.1998, 0.2647] |
| Chapman | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | 0.1029 | 0.1022 | [0.0783, 0.1270] |
| Chapman | physics_p010_minus_ordinary | mean_792_masks | 0.0181 | 0.0177 | [0.0068, 0.0279] |
| Chapman | physics_p010_minus_physics_aware | clean | -0.0097 | -0.0097 | [-0.0237, 0.0052] |
| Chapman | physics_p010_minus_physics_aware | RA_LL_reversal | -0.0486 | -0.0486 | [-0.0691, -0.0283] |
| Chapman | physics_p010_minus_physics_aware | cycle_slots_120 | -0.0735 | -0.0726 | [-0.0962, -0.0502] |
| Chapman | physics_p010_minus_physics_aware | cycle_slots_201 | -0.0232 | -0.0231 | [-0.0423, -0.0040] |
| Chapman | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | -0.0156 | -0.0161 | [-0.0279, -0.0033] |
| Chapman | physics_p010_minus_physics_aware | mean_792_masks | 0.0239 | 0.0240 | [0.0145, 0.0340] |
| Ningbo | physics_p025_minus_combined | clean | 0.0008 | 0.0008 | [-0.0073, 0.0093] |
| Ningbo | physics_p025_minus_combined | RA_LL_reversal | -0.0201 | -0.0201 | [-0.0289, -0.0109] |
| Ningbo | physics_p025_minus_combined | cycle_slots_120 | -0.0266 | -0.0268 | [-0.0363, -0.0173] |
| Ningbo | physics_p025_minus_combined | cycle_slots_201 | -0.0236 | -0.0237 | [-0.0334, -0.0147] |
| Ningbo | physics_p025_minus_combined | mask5_0_1_2_3_5 | -0.0301 | -0.0301 | [-0.0406, -0.0194] |
| Ningbo | physics_p025_minus_combined | mean_792_masks | -0.0377 | -0.0377 | [-0.0449, -0.0304] |
| Ningbo | physics_p025_minus_physics_aware | clean | -0.0093 | -0.0093 | [-0.0179, -0.0015] |
| Ningbo | physics_p025_minus_physics_aware | RA_LL_reversal | -0.0419 | -0.0420 | [-0.0530, -0.0311] |
| Ningbo | physics_p025_minus_physics_aware | cycle_slots_120 | -0.0504 | -0.0506 | [-0.0609, -0.0400] |
| Ningbo | physics_p025_minus_physics_aware | cycle_slots_201 | -0.0416 | -0.0417 | [-0.0528, -0.0314] |
| Ningbo | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | -0.0102 | -0.0104 | [-0.0197, -0.0014] |
| Ningbo | physics_p025_minus_physics_aware | mean_792_masks | -0.0006 | -0.0008 | [-0.0068, 0.0048] |
| Ningbo | physics_p010_minus_ordinary | clean | -0.0019 | -0.0020 | [-0.0104, 0.0060] |
| Ningbo | physics_p010_minus_ordinary | RA_LL_reversal | 0.4185 | 0.4175 | [0.3925, 0.4419] |
| Ningbo | physics_p010_minus_ordinary | cycle_slots_120 | 0.4197 | 0.4190 | [0.3911, 0.4432] |
| Ningbo | physics_p010_minus_ordinary | cycle_slots_201 | 0.3642 | 0.3632 | [0.3403, 0.3868] |
| Ningbo | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | 0.1806 | 0.1805 | [0.1628, 0.1982] |
| Ningbo | physics_p010_minus_ordinary | mean_792_masks | 0.0100 | 0.0099 | [0.0033, 0.0162] |
| Ningbo | physics_p010_minus_physics_aware | clean | -0.0127 | -0.0128 | [-0.0239, -0.0023] |
| Ningbo | physics_p010_minus_physics_aware | RA_LL_reversal | -0.0658 | -0.0661 | [-0.0821, -0.0508] |
| Ningbo | physics_p010_minus_physics_aware | cycle_slots_120 | -0.0863 | -0.0862 | [-0.1021, -0.0718] |
| Ningbo | physics_p010_minus_physics_aware | cycle_slots_201 | -0.0581 | -0.0582 | [-0.0755, -0.0421] |
| Ningbo | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | -0.0096 | -0.0096 | [-0.0249, 0.0050] |
| Ningbo | physics_p010_minus_physics_aware | mean_792_masks | -0.0086 | -0.0089 | [-0.0169, -0.0006] |

## Ningbo смеси clean/RA_LL

Сохранена вся сетка `q_fault = 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1` для всех endpoint и пяти режимов в `MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv`. Таблица ниже показывает новые режимы для TWA, усреднение по seed; raw и calibrated значения разделены. Не выбирайте q по test как рабочий порог.

| arm | q | score_mode | auprc | auroc | brier | ece15 |
| --- | --- | --- | --- | --- | --- | --- |
| physics_p010 | 0.0000 | calibrated | 0.6742 | 0.9219 | 0.0782 | 0.0350 |
| physics_p010 | 0.0000 | raw | 0.6742 | 0.9219 | 0.0993 | 0.1011 |
| physics_p010 | 0.0010 | calibrated | 0.6741 | 0.9219 | 0.0782 | 0.0350 |
| physics_p010 | 0.0010 | raw | 0.6741 | 0.9219 | 0.0993 | 0.1011 |
| physics_p010 | 0.0050 | calibrated | 0.6738 | 0.9218 | 0.0783 | 0.0350 |
| physics_p010 | 0.0050 | raw | 0.6738 | 0.9218 | 0.0994 | 0.1012 |
| physics_p010 | 0.0100 | calibrated | 0.6734 | 0.9216 | 0.0783 | 0.0350 |
| physics_p010 | 0.0100 | raw | 0.6734 | 0.9216 | 0.0995 | 0.1013 |
| physics_p010 | 0.0500 | calibrated | 0.6705 | 0.9202 | 0.0787 | 0.0348 |
| physics_p010 | 0.0500 | raw | 0.6705 | 0.9202 | 0.1004 | 0.1024 |
| physics_p010 | 0.1000 | calibrated | 0.6669 | 0.9185 | 0.0793 | 0.0346 |
| physics_p010 | 0.1000 | raw | 0.6669 | 0.9185 | 0.1015 | 0.1038 |
| physics_p010 | 1.0000 | calibrated | 0.6007 | 0.8852 | 0.0885 | 0.0366 |
| physics_p010 | 1.0000 | raw | 0.6007 | 0.8852 | 0.1212 | 0.1321 |
| physics_p025 | 0.0000 | calibrated | 0.6776 | 0.9237 | 0.0808 | 0.0451 |
| physics_p025 | 0.0000 | raw | 0.6776 | 0.9237 | 0.0944 | 0.0957 |
| physics_p025 | 0.0010 | calibrated | 0.6775 | 0.9237 | 0.0808 | 0.0451 |
| physics_p025 | 0.0010 | raw | 0.6775 | 0.9237 | 0.0944 | 0.0957 |
| physics_p025 | 0.0050 | calibrated | 0.6773 | 0.9235 | 0.0808 | 0.0450 |
| physics_p025 | 0.0050 | raw | 0.6773 | 0.9235 | 0.0945 | 0.0960 |
| physics_p025 | 0.0100 | calibrated | 0.6770 | 0.9234 | 0.0809 | 0.0449 |
| physics_p025 | 0.0100 | raw | 0.6770 | 0.9234 | 0.0946 | 0.0962 |
| physics_p025 | 0.0500 | calibrated | 0.6744 | 0.9222 | 0.0811 | 0.0442 |
| physics_p025 | 0.0500 | raw | 0.6744 | 0.9222 | 0.0957 | 0.0985 |
| physics_p025 | 0.1000 | calibrated | 0.6714 | 0.9207 | 0.0814 | 0.0434 |
| physics_p025 | 0.1000 | raw | 0.6714 | 0.9207 | 0.0971 | 0.1014 |
| physics_p025 | 1.0000 | calibrated | 0.6246 | 0.8955 | 0.0862 | 0.0407 |
| physics_p025 | 1.0000 | raw | 0.6246 | 0.8955 | 0.1217 | 0.1533 |

Для новых моделей Platt calibrator fitted только на прежней Ningbo validation A; threshold определён только на прежней validation B. Параметры и происхождение по каждой модели записаны в `CALIBRATION_PROVENANCE_NEW.json`. Исторические arms используют сохранённые calibrators/thresholds соответствующей фазы; raw и calibrated оценки хранятся отдельно.

## Выполнение 12 запусков

Все шесть Chapman запусков прошли frozen validation/integrity gate, затем были оценены; после этого так же завершены шесть Ningbo. Остановка каждого обучения — `patience_5`. Значение памяти — пиковая CUDA allocation по gate, а не системная RAM.

| Когорта | Режим | Seed | Статус | Лучшая эпоха | Val TWA AUPRC | Val TWA AUROC | Остановка | Время, ч | Пик GPU, MiB | Условия inference | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | physics_p010 | 104729 | PASS | 9 | 0.7462 | 0.9329 | patience_5 | 0.1140 | 269.1392 | 796 | training/Chapman/physics_p010/seed_104729/best.pt |
| Chapman | physics_p010 | 130363 | PASS | 7 | 0.7375 | 0.9293 | patience_5 | 0.0944 | 270.8306 | 796 | training/Chapman/physics_p010/seed_130363/best.pt |
| Chapman | physics_p010 | 155921 | PASS | 13 | 0.7554 | 0.9311 | patience_5 | 0.1399 | 270.8306 | 796 | training/Chapman/physics_p010/seed_155921/best.pt |
| Chapman | physics_p025 | 104729 | PASS | 13 | 0.7615 | 0.9340 | patience_5 | 0.1501 | 270.8306 | 796 | training/Chapman/physics_p025/seed_104729/best.pt |
| Chapman | physics_p025 | 130363 | PASS | 15 | 0.7550 | 0.9352 | patience_5 | 0.1662 | 270.8306 | 796 | training/Chapman/physics_p025/seed_130363/best.pt |
| Chapman | physics_p025 | 155921 | PASS | 16 | 0.7629 | 0.9344 | patience_5 | 0.1635 | 270.8306 | 796 | training/Chapman/physics_p025/seed_155921/best.pt |
| Ningbo | physics_p010 | 104729 | PASS | 7 | 0.6472 | 0.9141 | patience_5 | 0.1850 | 270.8306 | 796 | training/Ningbo/physics_p010/seed_104729/best.pt |
| Ningbo | physics_p010 | 130363 | PASS | 16 | 0.6681 | 0.9221 | patience_5 | 0.2290 | 270.8306 | 796 | training/Ningbo/physics_p010/seed_130363/best.pt |
| Ningbo | physics_p010 | 155921 | PASS | 8 | 0.6626 | 0.9176 | patience_5 | 0.1425 | 270.8306 | 796 | training/Ningbo/physics_p010/seed_155921/best.pt |
| Ningbo | physics_p025 | 104729 | PASS | 8 | 0.6564 | 0.9115 | patience_5 | 0.1542 | 270.8306 | 796 | training/Ningbo/physics_p025/seed_104729/best.pt |
| Ningbo | physics_p025 | 130363 | PASS | 8 | 0.6585 | 0.9182 | patience_5 | 0.1537 | 270.8306 | 796 | training/Ningbo/physics_p025/seed_130363/best.pt |
| Ningbo | physics_p025 | 155921 | PASS | 21 | 0.6833 | 0.9246 | patience_5 | 0.3032 | 270.8306 | 796 | training/Ningbo/physics_p025/seed_155921/best.pt |

## Метод и ограничения сопоставимости

Использованы прежние ECGResNet-small, labels, splits, train-derived normalization и фиксированный recipe: AdamW, lr 0.001, weight decay 0.0001, weighted multilabel BCE с positive-weight cap 20, batch 64, максимум 30 эпох, AMP, выбор checkpoint по clean-validation TWA AUPRC, patience 5. Swap применялся к raw сигналу до normalization. Новые режимы содержали только clean или один из трёх reversal swaps RA_LA/RA_LL/LA_LL; маски и циклы в train augmentation не добавлялись.

Исторические результаты переиспользованы без перезапуска: ordinary, physics-aware 50%, combined и mask comparators взяты из проверенных сохранённых outputs, prediction caches и heldout-cycle пакета. Точные файлы-источники зафиксированы в `FROZEN_PROTOCOL.json`, `PROTOCOL_AMENDMENT_001..004.json`, `build_final_report.py` и `paired_bootstrap.py`. Поэтому это не полностью контролируемое повторное сравнение и не независимая репликация.

Во время первоначального запуска первой Chapman p010 единицы завершились train и validation эпохи 1, но запись validation артефакта сорвалась из-за конфликта аргумента функции; checkpoint epoch не был зафиксирован и test inference не запускался. Попытка сохранена в `INTERRUPTED_ATTEMPTS.json`, исправление документировано Amendment 004; unit чисто повторён с исходным seed и учтён один раз среди 12 завершённых запусков.

**Методический SHA256:** `49c3fdef06868f951aaf96803b42c31146e67362b0eb71290848501d6ec7e382`.

**Runner SHA256:** `c6052fe7e0b4f27e93b24a509736b2231574392a3e06944bfe1e1646690befb2`.

Engineering checks: PASS; validation gates: 12/12 PASS; test conditions: 796 на unit (4 фиксированных + 792 маски); mask records: 792 на unit. HEEDB/reserved test не оценивались; новые модели и подбор параметров не проводились.

## Артефакты для аудита

- Полные seed-specific метрики и исторические comparators: `METRICS_ALL_ARMS_BY_SEED.csv`.
- Сводки по условию/endpoint: `METRICS_CHAPMAN_SUMMARY.csv`, `METRICS_NINGBO_SUMMARY.csv`.
- Распределения 792 масок новых моделей: `MASK_DISTRIBUTION_CHAPMAN.csv`, `MASK_DISTRIBUTION_NINGBO.csv`.
- Парные разности по seed и 1000-repeat intervals: `PAIRED_CONTRASTS_POINT_BY_SEED.csv`, `PAIRED_BOOTSTRAP_TWA_AUPRC.csv`.
- Mixtures/calibration: `MIXTURE_RESULTS_ALL_ARMS_BY_SEED.csv`, `CALIBRATION_PROVENANCE_NEW.json`.
- Все checkpoints, validation outputs, per-condition test predictions и integrity manifests: подкаталоги `training/` и `predictions/`.
- Текущий file/hash inventory: `RUN_MANIFEST.json`.

Отчёт фиксирует результаты для проверки. Дальнейшая оценка данных или новые эксперименты не запускались.
