# Восстановленный отчёт: augmentation probability study

Пакет восстановления: `outputs/augmentation_probability_recovery_v2/`. Это постфактум-восстановление по уже обученным моделям и сохранённым test probabilities. Оно не было зарегистрировано до первоначального просмотра test. Исходные данные и артефакты v1 не перезаписывались.

## Объём и метод

- 12 уже обученных моделей: Chapman/Ningbo × physics_p010/physics_p025 × seeds 104729, 130363, 155921. Новых обучений и полных повторных прогонов 792 масок не было.
- Для каждого запуска восстановлена validation prediction по `best.pt`; входные predictions, y и record IDs записаны в `validation/`. Первичный протокол и SHA256 зафиксированы в `RECOVERY_PROTOCOL.json` (отпечаток протокола: `89e85ec4beba20297c7f202fdbda92299ab6717f3c819547f9ef03319e41a4f0`). Дополнительный frozen supplement исторических cached predictions: `d1f3175e1b6b8fd097f4c31d37816fa2b7cdb6f6b9fa6811bc6b46debf5c75ab`.
- Протокол явно помечен постфактум: первичный test review уже состоялся. Supplement создан позднее, до повторного заполнения исторических comparator-полей.
- Численный допуск для validation TWA AUPRC/AUROC заранее установлен как 1e-6. Применённый порядок записей, split, preprocessing и inference соответствуют сохранённым данным и контрольным артефактам.

## Исправление validation

Все 12 `best.pt` state_dict побитно совпали с `checkpoint.pt`, использованным для test inference. Восстановленные TWA AUPRC/AUROC совпали с `validation_gate.json` во всех 12 случаях; максимальная абсолютная разница 0. Старый `validation.npz` действительно содержал метрики последней эпохи, тогда как тестовые предсказания были получены из best checkpoint.

| Cohort | Arm | Seed | Best epoch | Last epoch | Старый AP | Восстановленный AP | Gate AP | Восстановленный AUROC | Gate AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cohort | arm | seed | best_epoch | last_epoch | old_saved_validation_twa_auprc | restored_best_checkpoint_twa_auprc | validation_gate_twa_auprc | restored_best_checkpoint_twa_auroc | validation_gate_twa_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | physics_p010 | 104729.000000 | 9.000000 | 14.000000 | 0.714255 | 0.746246 | 0.746246 | 0.932850 | 0.932850 |
| Chapman | physics_p010 | 130363.000000 | 7.000000 | 12.000000 | 0.730373 | 0.737525 | 0.737525 | 0.929312 | 0.929312 |
| Chapman | physics_p010 | 155921.000000 | 13.000000 | 18.000000 | 0.734713 | 0.755356 | 0.755356 | 0.931096 | 0.931096 |
| Chapman | physics_p025 | 104729.000000 | 13.000000 | 18.000000 | 0.746734 | 0.761468 | 0.761468 | 0.933955 | 0.933955 |
| Chapman | physics_p025 | 130363.000000 | 15.000000 | 20.000000 | 0.743767 | 0.755031 | 0.755031 | 0.935239 | 0.935239 |
| Chapman | physics_p025 | 155921.000000 | 16.000000 | 21.000000 | 0.752442 | 0.762905 | 0.762905 | 0.934412 | 0.934412 |
| Ningbo | physics_p010 | 104729.000000 | 7.000000 | 12.000000 | 0.623565 | 0.647218 | 0.647218 | 0.914070 | 0.914070 |
| Ningbo | physics_p010 | 130363.000000 | 16.000000 | 21.000000 | 0.655484 | 0.668099 | 0.668099 | 0.922081 | 0.922081 |
| Ningbo | physics_p010 | 155921.000000 | 8.000000 | 13.000000 | 0.660253 | 0.662625 | 0.662625 | 0.917563 | 0.917563 |
| Ningbo | physics_p025 | 104729.000000 | 8.000000 | 13.000000 | 0.652960 | 0.656365 | 0.656365 | 0.911488 | 0.911488 |
| Ningbo | physics_p025 | 130363.000000 | 8.000000 | 13.000000 | 0.654507 | 0.658541 | 0.658541 | 0.918203 | 0.918203 |
| Ningbo | physics_p025 | 155921.000000 | 21.000000 | 26.000000 | 0.651462 | 0.683343 | 0.683343 | 0.924642 | 0.924642 |

Проверка небольшого набора test probabilities для каждого checkpoint прошла: 24 сравнения (clean и RA_LL для 12 моделей) совпали в пределах фиксированного same-batch контекста. Первичная проба с отдельным batch из 16 записей показала разницу 0.00048828125 при допуске 1e-6; диагностика установила зависимость AMP вычислений от размера batch. Поэтому amendment сохранил заранее выбранные индексы и допуск, а повтор воспроизводит исходный batch размером 128; подробности — `RECOVERY_AMENDMENT_001.json`.

## Test-метрики и пороги

Для новых моделей пороги пересчитаны из отрицательных clean validation scores по правилу линейного 90-го процентиля, отдельно для каждой модели и endpoint. Пересчитаны sensitivity, specificity, predicted-positive fraction и accuracy для всех 796 условий × 3 endpoint × 12 моделей = 28 656 строк. Raw AUPRC/AUROC/Brier/ECE оставлены из v1 после проверки SHA256 всех файлов предсказаний и канонического порядка записей; максимальное отличие от v1 равно 0. Исторические пороги существующих comparator-моделей оставлены прежними, а отсутствовавшие comparator поля восстановлены из cached predictions без inference. Для всех общих фиксированных условий в таблице нет пропусков; 54 поля accuracy по двум дополнительным Ningbo условиям RA_LA/LA_LL остаются недоступны из-за отсутствия сохранённых исторических предсказаний для этих условий.

Полная summary-таблица по всем пяти arm, четырём фиксированным условиям и трём endpoint: `METRICS_SUMMARY_ALL_ARMS.csv` (mean и SD по seed). Полная строковая таблица по всем 796 условиям на каждый доступный arm: `METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv`. Файлы не содержат специально восстановленных/выдуманных значений для отсутствующих исторических предсказаний.

### TWA, clean

Пороговые показатели относятся к сохранённому порогу соответствующей модели: для новых arm — к восстановленному clean validation q90, для исторических comparator — к исходному native validation threshold.

| cohort | arm | auprc_mean | auprc_std | auroc_mean | auroc_std | brier_mean | ece15_mean | sensitivity_mean | specificity_mean | predicted_positive_fraction_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | ordinary | 0.6531 | 0.0070 | 0.9156 | 0.0010 | 0.1151 | 0.1137 | 0.7745 | 0.8826 | 0.2382 | 0.8628 |
| Chapman | physics_p010 | 0.6544 | 0.0047 | 0.9131 | 0.0014 | 0.1305 | 0.1480 | 0.7737 | 0.8725 | 0.2463 | 0.8543 |
| Chapman | physics_p025 | 0.6639 | 0.0007 | 0.9152 | 0.0026 | 0.1315 | 0.1518 | 0.7683 | 0.8715 | 0.2462 | 0.8525 |
| Chapman | physics_p050 | 0.6641 | 0.0092 | 0.9138 | 0.0017 | 0.1359 | 0.1628 | 0.7586 | 0.8795 | 0.2379 | 0.8572 |
| Chapman | combined | 0.6586 | 0.0092 | 0.9128 | 0.0057 | 0.1224 | 0.1379 | 0.7586 | 0.8775 | 0.2395 | 0.8556 |
| Ningbo | ordinary | 0.6761 | 0.0028 | 0.9229 | 0.0011 | 0.0990 | 0.0978 | 0.7328 | 0.9101 | 0.1851 | 0.8838 |
| Ningbo | physics_p010 | 0.6742 | 0.0112 | 0.9219 | 0.0027 | 0.0993 | 0.1011 | 0.7308 | 0.9072 | 0.1873 | 0.8811 |
| Ningbo | physics_p025 | 0.6776 | 0.0090 | 0.9237 | 0.0039 | 0.0944 | 0.0957 | 0.7340 | 0.9082 | 0.1869 | 0.8824 |
| Ningbo | physics_p050 | 0.6869 | 0.0099 | 0.9242 | 0.0048 | 0.1077 | 0.1247 | 0.7363 | 0.9056 | 0.1895 | 0.8805 |
| Ningbo | combined | 0.6768 | 0.0115 | 0.9240 | 0.0029 | 0.1024 | 0.1175 | 0.7350 | 0.9066 | 0.1884 | 0.8812 |

### TWA, RA_LL reversal

| cohort | arm | auprc_mean | auroc_mean | brier_mean | ece15_mean | sensitivity_mean | specificity_mean | predicted_positive_fraction_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | ordinary | 0.3278 | 0.6614 | 0.5209 | 0.5983 | 0.8753 | 0.2642 | 0.7615 | 0.3766 |
| Chapman | physics_p010 | 0.5886 | 0.8695 | 0.1690 | 0.1974 | 0.7498 | 0.8356 | 0.2720 | 0.8198 |
| Chapman | physics_p025 | 0.6183 | 0.8864 | 0.1764 | 0.2205 | 0.7781 | 0.8390 | 0.2745 | 0.8278 |
| Chapman | physics_p050 | 0.6372 | 0.8946 | 0.1618 | 0.2085 | 0.7630 | 0.8561 | 0.2577 | 0.8390 |
| Chapman | combined | 0.6314 | 0.8953 | 0.1329 | 0.1475 | 0.7445 | 0.8717 | 0.2416 | 0.8483 |
| Ningbo | ordinary | 0.1822 | 0.5625 | 0.5117 | 0.5758 | 0.7634 | 0.3197 | 0.6926 | 0.3854 |
| Ningbo | physics_p010 | 0.6007 | 0.8852 | 0.1212 | 0.1321 | 0.6631 | 0.8921 | 0.1901 | 0.8582 |
| Ningbo | physics_p025 | 0.6246 | 0.8955 | 0.1217 | 0.1533 | 0.7282 | 0.8770 | 0.2127 | 0.8549 |
| Ningbo | physics_p050 | 0.6665 | 0.9144 | 0.1157 | 0.1400 | 0.7273 | 0.9011 | 0.1919 | 0.8754 |
| Ningbo | combined | 0.6447 | 0.9048 | 0.1154 | 0.1374 | 0.7115 | 0.8969 | 0.1932 | 0.8694 |

## Среднее по 792 маскам

Значения — сначала среднее AP по 792 маскам внутри каждого seed, затем mean и SD по трём seeds. Полная таблица всех метрик и endpoint находится в `MEAN_792_MASK_METRICS_ALL_ARMS.csv`.

| cohort | arm | auprc_mean | auprc_std | auroc_mean | brier_mean | ece15_mean | sensitivity_mean | specificity_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | ordinary | 0.5738 | 0.0176 | 0.8680 | 0.1959 | 0.2291 | 0.8045 | 0.7702 | 0.7765 |
| Chapman | physics_p010 | 0.5919 | 0.0174 | 0.8745 | 0.2022 | 0.2457 | 0.8002 | 0.7804 | 0.7840 |
| Chapman | physics_p025 | 0.5837 | 0.0069 | 0.8659 | 0.2240 | 0.2737 | 0.8124 | 0.7493 | 0.7609 |
| Chapman | physics_p050 | 0.5679 | 0.0098 | 0.8523 | 0.2405 | 0.2938 | 0.7996 | 0.7406 | 0.7514 |
| Chapman | combined | 0.6312 | 0.0046 | 0.8982 | 0.1359 | 0.1593 | 0.7509 | 0.8630 | 0.8424 |
| Ningbo | ordinary | 0.5863 | 0.0018 | 0.8847 | 0.1596 | 0.1817 | 0.7475 | 0.8242 | 0.8129 |
| Ningbo | physics_p010 | 0.5963 | 0.0043 | 0.8876 | 0.1489 | 0.1689 | 0.7373 | 0.8361 | 0.8215 |
| Ningbo | physics_p025 | 0.6043 | 0.0140 | 0.8908 | 0.1381 | 0.1618 | 0.7466 | 0.8380 | 0.8245 |
| Ningbo | physics_p050 | 0.6049 | 0.0107 | 0.8856 | 0.1678 | 0.2060 | 0.7507 | 0.8222 | 0.8116 |
| Ningbo | combined | 0.6420 | 0.0077 | 0.9103 | 0.1156 | 0.1367 | 0.7266 | 0.8934 | 0.8687 |

Для TWA разница AUPRC `p025 − combined` составила −0.047445 на Chapman и −0.037691 на Ningbo. Парные 95% bootstrap CI соответственно [−0.058512, −0.037227] и [−0.044864, −0.030394]. Здесь отрицательное значение означает ниже AUPRC у p025. Это сравнение двух обученных стратегий по зафиксированному набору масок; оно само по себе не изолирует причинный эффект маски.

## Парные интервалы

Сохранены прежние 48 TWA AUPRC интервалов и добавлены 80 TWA AUROC/raw Brier интервалов (1000 синхронизированных bootstrap выборок; seeds усреднены внутри каждой реплики). Full-cohort point estimate приведён отдельно от bootstrap mean. Для разницы Brier `arm_a − arm_b` отрицательное значение означает ниже Brier у arm_a. Полный набор из 128 групп находится в `ALL_PAIRED_BOOTSTRAP_INTERVALS.csv`.

### AUPRC: clean и среднее по 792 маскам

| cohort | contrast | condition | full_cohort_point_difference | bootstrap_mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| Chapman | physics_p025_minus_combined | clean | 0.0053 | 0.0054 | -0.0046 | 0.0145 |
| Chapman | physics_p025_minus_combined | mean_792_masks | -0.0474 | -0.0474 | -0.0585 | -0.0372 |
| Chapman | physics_p025_minus_physics_aware | clean | -0.0002 | 0.0001 | -0.0125 | 0.0125 |
| Chapman | physics_p025_minus_physics_aware | mean_792_masks | 0.0158 | 0.0159 | 0.0063 | 0.0256 |
| Chapman | physics_p010_minus_ordinary | clean | 0.0013 | 0.0011 | -0.0116 | 0.0143 |
| Chapman | physics_p010_minus_ordinary | mean_792_masks | 0.0181 | 0.0177 | 0.0068 | 0.0279 |
| Chapman | physics_p010_minus_physics_aware | clean | -0.0097 | -0.0097 | -0.0237 | 0.0052 |
| Chapman | physics_p010_minus_physics_aware | mean_792_masks | 0.0239 | 0.0240 | 0.0145 | 0.0340 |
| Ningbo | physics_p025_minus_combined | clean | 0.0008 | 0.0008 | -0.0073 | 0.0093 |
| Ningbo | physics_p025_minus_combined | mean_792_masks | -0.0377 | -0.0377 | -0.0449 | -0.0304 |
| Ningbo | physics_p025_minus_physics_aware | clean | -0.0093 | -0.0093 | -0.0179 | -0.0015 |
| Ningbo | physics_p025_minus_physics_aware | mean_792_masks | -0.0006 | -0.0008 | -0.0068 | 0.0048 |
| Ningbo | physics_p010_minus_ordinary | clean | -0.0019 | -0.0020 | -0.0104 | 0.0060 |
| Ningbo | physics_p010_minus_ordinary | mean_792_masks | 0.0100 | 0.0099 | 0.0033 | 0.0162 |
| Ningbo | physics_p010_minus_physics_aware | clean | -0.0127 | -0.0128 | -0.0239 | -0.0023 |
| Ningbo | physics_p010_minus_physics_aware | mean_792_masks | -0.0086 | -0.0089 | -0.0169 | -0.0006 |

### AUROC и raw Brier: clean, RA_LL и одна заранее заданная маска

| cohort | contrast | condition | metric | full_cohort_point_difference | bootstrap_mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chapman | physics_p025_minus_combined | clean | AUROC | 0.0024 | 0.0025 | -0.0009 | 0.0060 |
| Chapman | physics_p025_minus_combined | clean | raw Brier | 0.0091 | 0.0090 | 0.0062 | 0.0118 |
| Chapman | physics_p025_minus_combined | RA_LL_reversal | AUROC | -0.0089 | -0.0088 | -0.0141 | -0.0036 |
| Chapman | physics_p025_minus_combined | RA_LL_reversal | raw Brier | 0.0435 | 0.0435 | 0.0397 | 0.0479 |
| Chapman | physics_p025_minus_combined | mask5_0_1_2_3_5 | AUROC | -0.0097 | -0.0096 | -0.0148 | -0.0045 |
| Chapman | physics_p025_minus_combined | mask5_0_1_2_3_5 | raw Brier | 0.0683 | 0.0682 | 0.0639 | 0.0720 |
| Chapman | physics_p025_minus_physics_aware | clean | AUROC | 0.0013 | 0.0015 | -0.0028 | 0.0059 |
| Chapman | physics_p025_minus_physics_aware | clean | raw Brier | -0.0044 | -0.0044 | -0.0074 | -0.0015 |
| Chapman | physics_p025_minus_physics_aware | RA_LL_reversal | AUROC | -0.0082 | -0.0080 | -0.0131 | -0.0027 |
| Chapman | physics_p025_minus_physics_aware | RA_LL_reversal | raw Brier | 0.0146 | 0.0145 | 0.0114 | 0.0174 |
| Chapman | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | AUROC | -0.0116 | -0.0115 | -0.0161 | -0.0071 |
| Chapman | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | raw Brier | 0.0479 | 0.0479 | 0.0446 | 0.0509 |
| Chapman | physics_p010_minus_ordinary | clean | AUROC | -0.0025 | -0.0025 | -0.0059 | 0.0008 |
| Chapman | physics_p010_minus_ordinary | clean | raw Brier | 0.0153 | 0.0153 | 0.0124 | 0.0181 |
| Chapman | physics_p010_minus_ordinary | RA_LL_reversal | AUROC | 0.2080 | 0.2079 | 0.1873 | 0.2290 |
| Chapman | physics_p010_minus_ordinary | RA_LL_reversal | raw Brier | -0.3519 | -0.3520 | -0.3656 | -0.3391 |
| Chapman | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | AUROC | 0.0584 | 0.0585 | 0.0481 | 0.0689 |
| Chapman | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | raw Brier | -0.1800 | -0.1801 | -0.1887 | -0.1717 |
| Chapman | physics_p010_minus_physics_aware | clean | AUROC | -0.0007 | -0.0006 | -0.0045 | 0.0034 |
| Chapman | physics_p010_minus_physics_aware | clean | raw Brier | -0.0054 | -0.0054 | -0.0083 | -0.0027 |
| Chapman | physics_p010_minus_physics_aware | RA_LL_reversal | AUROC | -0.0251 | -0.0251 | -0.0329 | -0.0173 |
| Chapman | physics_p010_minus_physics_aware | RA_LL_reversal | raw Brier | 0.0072 | 0.0071 | 0.0023 | 0.0118 |
| Chapman | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | AUROC | -0.0064 | -0.0064 | -0.0103 | -0.0022 |
| Chapman | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | raw Brier | 0.0120 | 0.0120 | 0.0095 | 0.0145 |
| Ningbo | physics_p025_minus_combined | clean | AUROC | -0.0003 | -0.0003 | -0.0024 | 0.0018 |
| Ningbo | physics_p025_minus_combined | clean | raw Brier | -0.0081 | -0.0081 | -0.0095 | -0.0066 |
| Ningbo | physics_p025_minus_combined | RA_LL_reversal | AUROC | -0.0093 | -0.0093 | -0.0124 | -0.0062 |
| Ningbo | physics_p025_minus_combined | RA_LL_reversal | raw Brier | 0.0063 | 0.0063 | 0.0050 | 0.0078 |
| Ningbo | physics_p025_minus_combined | mask5_0_1_2_3_5 | AUROC | -0.0176 | -0.0174 | -0.0215 | -0.0136 |
| Ningbo | physics_p025_minus_combined | mask5_0_1_2_3_5 | raw Brier | 0.0297 | 0.0297 | 0.0279 | 0.0317 |
| Ningbo | physics_p025_minus_physics_aware | clean | AUROC | -0.0005 | -0.0005 | -0.0027 | 0.0018 |
| Ningbo | physics_p025_minus_physics_aware | clean | raw Brier | -0.0134 | -0.0134 | -0.0149 | -0.0120 |
| Ningbo | physics_p025_minus_physics_aware | RA_LL_reversal | AUROC | -0.0189 | -0.0190 | -0.0225 | -0.0154 |
| Ningbo | physics_p025_minus_physics_aware | RA_LL_reversal | raw Brier | 0.0061 | 0.0061 | 0.0044 | 0.0077 |
| Ningbo | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | AUROC | -0.0013 | -0.0012 | -0.0046 | 0.0022 |
| Ningbo | physics_p025_minus_physics_aware | mask5_0_1_2_3_5 | raw Brier | -0.0119 | -0.0119 | -0.0136 | -0.0102 |
| Ningbo | physics_p010_minus_ordinary | clean | AUROC | -0.0010 | -0.0010 | -0.0030 | 0.0008 |
| Ningbo | physics_p010_minus_ordinary | clean | raw Brier | 0.0003 | 0.0003 | -0.0009 | 0.0015 |
| Ningbo | physics_p010_minus_ordinary | RA_LL_reversal | AUROC | 0.3227 | 0.3222 | 0.3060 | 0.3385 |
| Ningbo | physics_p010_minus_ordinary | RA_LL_reversal | raw Brier | -0.3905 | -0.3903 | -0.3986 | -0.3822 |
| Ningbo | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | AUROC | 0.0920 | 0.0919 | 0.0835 | 0.1006 |
| Ningbo | physics_p010_minus_ordinary | mask5_0_1_2_3_5 | raw Brier | -0.2265 | -0.2265 | -0.2319 | -0.2213 |
| Ningbo | physics_p010_minus_physics_aware | clean | AUROC | -0.0022 | -0.0023 | -0.0049 | 0.0001 |
| Ningbo | physics_p010_minus_physics_aware | clean | raw Brier | -0.0084 | -0.0084 | -0.0100 | -0.0068 |
| Ningbo | physics_p010_minus_physics_aware | RA_LL_reversal | AUROC | -0.0292 | -0.0293 | -0.0342 | -0.0249 |
| Ningbo | physics_p010_minus_physics_aware | RA_LL_reversal | raw Brier | 0.0056 | 0.0055 | 0.0032 | 0.0079 |
| Ningbo | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | AUROC | 0.0015 | 0.0014 | -0.0037 | 0.0065 |
| Ningbo | physics_p010_minus_physics_aware | mask5_0_1_2_3_5 | raw Brier | -0.0098 | -0.0099 | -0.0123 | -0.0076 |

Для p050 против новых p010/p025 на чистом Ningbo AUPRC интервалы разницы `p025 − p050` и `p010 − p050` не включают ноль: соответственно −0.009299 [−0.017904, −0.001550] и −0.012707 [−0.023890, −0.002316]. Это наблюдение по одной метрике и одной когорте; другие метрики не дают основания объявлять общую оптимальную вероятность или эквивалентность вариантов.

## Ningbo calibration и смеси

Platt-модели переоценены для шести новых Ningbo запусков и трёх endpoint на validation A; пороги рассчитаны по raw scores отрицательных примеров B, а calibrated threshold получен преобразованием raw threshold. Использованы прежние split indices, clipping и настройки L-BFGS-B. Тестовая сетка `q_fault`: 0, 0.001, 0.005, 0.01, 0.05, 0.1, 1; полный результат по raw/calibrated режимам — `NINGBO_MIXTURES_RECOVERED_BY_SEED.csv`.

Есть важное ограничение: полный validation ранее участвовал в выборе checkpoint до разделения A/B. Поэтому A/B не являются независимыми от отбора модели и калибровка имеет утечку через checkpoint selection. На clean и RA_LL raw/calibrated бинарные решения совпали во всех 18 проверках; ties и clipping отдельно перечислены в `NINGBO_CALIBRATION_PROVENANCE_RECOVERED.json`.

Изменение средних calibrated показателей относительно ошибочного v1 для q=0, 0.1 и 1:

| arm | q | delta_brier | delta_ece15 | delta_sensitivity | delta_specificity | delta_accuracy | delta_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| physics_p010 | 0.00000 | -0.00293 | -0.01264 | 0.04231 | 0.00199 | 0.00665 | -0.00114 |
| physics_p010 | 0.10000 | -0.00308 | -0.01320 | 0.04270 | 0.00174 | 0.00649 | -0.00114 |
| physics_p010 | 1.00000 | -0.00442 | -0.01378 | 0.04621 | -0.00045 | 0.00498 | -0.00114 |
| physics_p025 | 0.00000 | -0.00314 | -0.01253 | 0.06396 | 0.00131 | 0.00829 | 0.00566 |
| physics_p025 | 0.10000 | -0.00308 | -0.01221 | 0.06424 | 0.00069 | 0.00783 | 0.00566 |
| physics_p025 | 1.00000 | -0.00258 | -0.00610 | 0.06682 | -0.00481 | 0.00364 | 0.00566 |

Raw test ranking/probability metrics для новых arm не менялись; изменения calibrated Brier/ECE и пороговых метрик связаны с исправленным validation, Platt fit и порогом. Тест не использовался для настройки параметров.

## Ответы на вопросы

1. **Что из v1 подтвердилось?** Сами новые модели, test probabilities, порядок записей и raw AUPRC/AUROC/Brier/ECE: контрольные хеши предсказаний подтверждены, raw метрики совпали с v1 (максимальная разница 0). Восстановленные test sample probabilities совпали с кешем при исходном контексте batch.
2. **Что изменилось после исправления?** Validation перешла с последней эпохи на best checkpoint; из-за этого изменились новые validation-derived пороги, пороговые метрики и Ningbo Platt calibration/mixtures. Старые значения оставлены в v1 и перечислены в `SUPERSEDED_ARTIFACTS.csv`.
3. **Помогают ли маски при одинаковой доле перестановок?** В среднем по 792 маскам TWA AUPRC ниже у p025, чем combined, на 0.0474 для Chapman и 0.0377 для Ningbo; парные интервалы не пересекают ноль. Это поддерживает разницу между этими arms в данной оценке, но не является самостоятельным доказательством причинного эффекта маски.
4. **Есть ли выигрыш на clean от снижения вероятности augmentation?** Убедительного общего выигрыша p010/p025 над p050 по всем метрикам и когортам нет. На clean Ningbo TWA AUPRC p050 выше обоих новых arm с указанными выше интервалами; это не доказывает общего превосходства p050. Интервал, включающий ноль, не трактуется как равенство.
5. **Готов ли пакет к следующему архитектурному тесту?** Да, как восстановленный и проверяемый пакет для планирования следующего теста. Он не даёт основания заявлять клиническое превосходство или независимое подтверждение; calibration A/B ограничен отбором checkpoint по полному validation. Новые обучения, HEEDB, Founder, reserved_test и работа над статьёй не выполнялись.

## Происхождение и заменённые результаты

- `RECOVERY_PROTOCOL.json`, `RECOVERY_AMENDMENT_001.json`, `COMPARATOR_COMPLETENESS_SUPPLEMENT.json` фиксируют протокол, поправку по batch-dependent replay и дополнительный пересчёт исторических comparator.
- `RECOVERY_MANIFEST.json` содержит SHA256 и размер всех файлов пакета, кроме самого manifest.
- `SUPERSEDED_ARTIFACTS.csv` перечисляет ошибочные v1 артефакты и соответствующие замены в этом каталоге. Файлы v1 сохранены без изменений.
- `VALIDATION_RESTORATION_AUDIT.json`, `RAW_TEST_METRIC_REPRODUCTION_AUDIT.json`, `HISTORICAL_COMPARATOR_COMPLETENESS_AUDIT.json`, `BOOTSTRAP_RECOVERY_AUDIT.json` содержат машинно-читаемые проверки.
