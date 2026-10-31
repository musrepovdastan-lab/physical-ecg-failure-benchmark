# Проверка переноса на две циклические перестановки конечностных электродов

**Статус:** расчёты выполнены на существующих моделях; это post-access исследовательское расширение, а не новая независимая подтверждающая проверка. Все цифры относятся к симуляциям перестановки кабелей. Новое обучение не проводилось.

## Краткие ответы

1. **Помогло ли обучение с перестановками?** Да. На обоих циклах и обеих когортах physics-aware и combined превзошли ordinary по TWA AUPRC. Physics-aware−ordinary составило +0.359 и +0.255 на Chapman, +0.506 и +0.422 на Ningbo (сначала цикл 120, затем 201); все четыре 95% paired bootstrap интервала выше нуля. Для combined−ordinary оценки равны +0.344, +0.238, +0.482 и +0.404.
2. **Насколько сохранилось качество относительно clean и RA–LL?** Physics-aware потерял относительно clean 0.019–0.039 AUPRC на Chapman и 0.030–0.039 на Ningbo; combined — 0.030–0.049 и 0.044–0.047. Ordinary потерял 0.263–0.387 на Chapman и 0.450–0.525 на Ningbo. Качество на циклах было близко к RA–LL; cycle−RA–LL сопоставляет разные преобразования и само по себе не изолирует эффект знакомства.
3. **Как повели себя combined и chest-only?** Combined сохранил устойчивость и превзошёл ordinary, но уступил physics-aware по TWA AUPRC на каждом цикле примерно на 0.015–0.024. Chest-only дал ровно одинаковые clean и циклические вероятности на всех записях (max delta 0). Его TWA AUPRC составил 0.650 на Chapman и 0.655 на Ningbo.
4. **Совпало ли направление на Chapman и Ningbo?** Да: на обоих циклах physics-aware и combined лучше ordinary, chest-only неизменен. Это согласованность в двух ранее доступных когортах, а не независимая репликация.
5. **Что это позволяет сказать?** Для этих моделей физически допустимые комбинации двух swaps, изучавшихся по отдельности, сохраняют большую часть качества reversal-обученных моделей на ранее использованных когортах. Это не доказывает перенос на реальные miswirings, другие типы отказов, клиническую пользу или внешнюю независимую валидность. Интервалы условны на фиксированных моделях; критерий успеха постфактум не задавался.

Frozen protocol SHA-256: `a4ad541c3645eae04273a8edee30d7235f5192a52d70a3ac0ef7f8e7a8dec9d6`. Число моделей: 30; новых prediction-файлов: 60. Пороги — сохранённые per-model validation-пороги. Новая калибровка и threshold fitting не выполнялись.

## Метрики по моделям

Полные строки для каждого seed и endpoint: `METRICS_BY_MODEL_SEED.csv`. Средние и SD по трём seeds: `METRICS_MEAN_SD.csv`. Основной endpoint — TWA; SB и STach вторичные. Вероятности разных seeds не усреднялись в ансамбль.

Для каждого endpoint указаны raw AUPRC/AUROC, Brier, ECE15, sensitivity, specificity и predicted-positive fraction при сохранённом validation threshold. Средние — сначала по seed-specific метрикам; SD рассчитано между тремя seed.

| Когорта | Режим | Условие | TWA AUPRC mean ± SD | AUROC mean ± SD | Brier mean ± SD | Sensitivity | Specificity | Predicted-positive fraction |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Chapman | ordinary | clean | 0.6531 ± 0.0070 | 0.9156 ± 0.0010 | 0.1151 ± 0.0090 | 0.775 ± 0.015 | 0.883 ± 0.001 | 0.238 ± 0.002 |
| Chapman | ordinary | RA_LL_reversal | 0.3278 ± 0.0475 | 0.6614 ± 0.0540 | 0.5209 ± 0.0680 | 0.875 ± 0.005 | 0.264 ± 0.083 | 0.761 ± 0.069 |
| Chapman | ordinary | cycle_slots_120 | 0.2661 ± 0.0383 | 0.6219 ± 0.0484 | 0.5735 ± 0.0464 | 0.925 ± 0.036 | 0.164 ± 0.089 | 0.852 ± 0.079 |
| Chapman | ordinary | cycle_slots_201 | 0.3904 ± 0.0261 | 0.7192 ± 0.0433 | 0.3234 ± 0.0479 | 0.715 ± 0.067 | 0.618 ± 0.027 | 0.444 ± 0.029 |
| Chapman | masking | clean | 0.6525 ± 0.0168 | 0.9119 ± 0.0061 | 0.1284 ± 0.0327 | 0.773 ± 0.033 | 0.879 ± 0.004 | 0.241 ± 0.009 |
| Chapman | masking | RA_LL_reversal | 0.2797 ± 0.0800 | 0.5710 ± 0.0681 | 0.5482 ± 0.0538 | 0.836 ± 0.020 | 0.161 ± 0.031 | 0.839 ± 0.025 |
| Chapman | masking | cycle_slots_120 | 0.2556 ± 0.0662 | 0.5929 ± 0.0747 | 0.5491 ± 0.0652 | 0.920 ± 0.010 | 0.144 ± 0.051 | 0.867 ± 0.043 |
| Chapman | masking | cycle_slots_201 | 0.3300 ± 0.0888 | 0.6414 ± 0.0633 | 0.3959 ± 0.1039 | 0.737 ± 0.036 | 0.449 ± 0.124 | 0.585 ± 0.105 |
| Chapman | physics_aware | clean | 0.6641 ± 0.0092 | 0.9138 ± 0.0017 | 0.1359 ± 0.0333 | 0.759 ± 0.012 | 0.879 ± 0.003 | 0.238 ± 0.005 |
| Chapman | physics_aware | RA_LL_reversal | 0.6372 ± 0.0081 | 0.8946 ± 0.0023 | 0.1618 ± 0.0370 | 0.763 ± 0.020 | 0.856 ± 0.010 | 0.258 ± 0.012 |
| Chapman | physics_aware | cycle_slots_120 | 0.6251 ± 0.0107 | 0.8946 ± 0.0030 | 0.1658 ± 0.0455 | 0.771 ± 0.012 | 0.856 ± 0.006 | 0.259 ± 0.007 |
| Chapman | physics_aware | cycle_slots_201 | 0.6453 ± 0.0055 | 0.8945 ± 0.0031 | 0.1504 ± 0.0348 | 0.745 ± 0.024 | 0.875 ± 0.025 | 0.239 ± 0.025 |
| Chapman | combined | clean | 0.6586 ± 0.0092 | 0.9128 ± 0.0057 | 0.1224 ± 0.0154 | 0.759 ± 0.018 | 0.877 ± 0.004 | 0.240 ± 0.004 |
| Chapman | combined | RA_LL_reversal | 0.6314 ± 0.0073 | 0.8953 ± 0.0055 | 0.1329 ± 0.0242 | 0.744 ± 0.027 | 0.872 ± 0.019 | 0.242 ± 0.021 |
| Chapman | combined | cycle_slots_120 | 0.6099 ± 0.0063 | 0.8929 ± 0.0021 | 0.1354 ± 0.0216 | 0.739 ± 0.027 | 0.869 ± 0.016 | 0.243 ± 0.018 |
| Chapman | combined | cycle_slots_201 | 0.6287 ± 0.0103 | 0.8863 ± 0.0066 | 0.1386 ± 0.0265 | 0.739 ± 0.023 | 0.869 ± 0.019 | 0.243 ± 0.019 |
| Chapman | chest_only | clean | 0.6503 ± 0.0067 | 0.9019 ± 0.0013 | 0.1466 ± 0.0254 | 0.734 ± 0.019 | 0.885 ± 0.003 | 0.228 ± 0.006 |
| Chapman | chest_only | RA_LL_reversal | 0.6503 ± 0.0067 | 0.9019 ± 0.0013 | 0.1466 ± 0.0254 | 0.734 ± 0.019 | 0.885 ± 0.003 | 0.228 ± 0.006 |
| Chapman | chest_only | cycle_slots_120 | 0.6503 ± 0.0067 | 0.9019 ± 0.0013 | 0.1466 ± 0.0254 | 0.734 ± 0.019 | 0.885 ± 0.003 | 0.228 ± 0.006 |
| Chapman | chest_only | cycle_slots_201 | 0.6503 ± 0.0067 | 0.9019 ± 0.0013 | 0.1466 ± 0.0254 | 0.734 ± 0.019 | 0.885 ± 0.003 | 0.228 ± 0.006 |
| Ningbo | ordinary | clean | 0.6761 ± 0.0028 | 0.9229 ± 0.0011 | 0.0990 ± 0.0092 | 0.733 ± 0.007 | 0.910 ± 0.002 | 0.185 ± 0.002 |
| Ningbo | ordinary | RA_LL_reversal | 0.1822 ± 0.0053 | 0.5625 ± 0.0135 | 0.5117 ± 0.0073 | 0.763 ± 0.042 | 0.320 ± 0.083 | 0.693 ± 0.077 |
| Ningbo | ordinary | cycle_slots_120 | 0.1510 ± 0.0154 | 0.4925 ± 0.0331 | 0.5921 ± 0.0460 | 0.826 ± 0.022 | 0.184 ± 0.009 | 0.818 ± 0.011 |
| Ningbo | ordinary | cycle_slots_201 | 0.2259 ± 0.0432 | 0.5994 ± 0.0428 | 0.3996 ± 0.0416 | 0.650 ± 0.084 | 0.485 ± 0.141 | 0.535 ± 0.131 |
| Ningbo | masking | clean | 0.6740 ± 0.0077 | 0.9229 ± 0.0024 | 0.1079 ± 0.0082 | 0.725 ± 0.004 | 0.909 ± 0.003 | 0.185 ± 0.003 |
| Ningbo | masking | RA_LL_reversal | 0.1692 ± 0.0383 | 0.5019 ± 0.0614 | 0.5742 ± 0.0829 | 0.742 ± 0.044 | 0.251 ± 0.122 | 0.748 ± 0.110 |
| Ningbo | masking | cycle_slots_120 | 0.1628 ± 0.0383 | 0.4950 ± 0.0583 | 0.6124 ± 0.0374 | 0.834 ± 0.033 | 0.182 ± 0.053 | 0.820 ± 0.050 |
| Ningbo | masking | cycle_slots_201 | 0.2414 ± 0.0526 | 0.6348 ± 0.0410 | 0.3487 ± 0.0664 | 0.602 ± 0.007 | 0.614 ± 0.081 | 0.418 ± 0.070 |
| Ningbo | physics_aware | clean | 0.6869 ± 0.0099 | 0.9242 ± 0.0048 | 0.1077 ± 0.0161 | 0.736 ± 0.010 | 0.906 ± 0.003 | 0.189 ± 0.002 |
| Ningbo | physics_aware | RA_LL_reversal | 0.6665 ± 0.0068 | 0.9144 ± 0.0053 | 0.1157 ± 0.0189 | 0.727 ± 0.007 | 0.901 ± 0.009 | 0.192 ± 0.009 |
| Ningbo | physics_aware | cycle_slots_120 | 0.6570 ± 0.0073 | 0.9059 ± 0.0048 | 0.1218 ± 0.0227 | 0.721 ± 0.023 | 0.900 ± 0.009 | 0.192 ± 0.011 |
| Ningbo | physics_aware | cycle_slots_201 | 0.6482 ± 0.0124 | 0.9043 ± 0.0072 | 0.1237 ± 0.0223 | 0.733 ± 0.040 | 0.889 ± 0.013 | 0.203 ± 0.016 |
| Ningbo | combined | clean | 0.6768 ± 0.0115 | 0.9240 ± 0.0029 | 0.1024 ± 0.0051 | 0.735 ± 0.024 | 0.907 ± 0.006 | 0.188 ± 0.008 |
| Ningbo | combined | RA_LL_reversal | 0.6447 ± 0.0150 | 0.9048 ± 0.0056 | 0.1154 ± 0.0131 | 0.711 ± 0.041 | 0.897 ± 0.028 | 0.193 ± 0.030 |
| Ningbo | combined | cycle_slots_120 | 0.6332 ± 0.0154 | 0.8934 ± 0.0103 | 0.1272 ± 0.0202 | 0.711 ± 0.050 | 0.886 ± 0.043 | 0.203 ± 0.044 |
| Ningbo | combined | cycle_slots_201 | 0.6302 ± 0.0128 | 0.8954 ± 0.0044 | 0.1262 ± 0.0133 | 0.717 ± 0.020 | 0.886 ± 0.014 | 0.203 ± 0.015 |
| Ningbo | chest_only | clean | 0.6554 ± 0.0071 | 0.9081 ± 0.0047 | 0.1131 ± 0.0063 | 0.709 ± 0.003 | 0.908 ± 0.003 | 0.183 ± 0.002 |
| Ningbo | chest_only | RA_LL_reversal | 0.6554 ± 0.0071 | 0.9081 ± 0.0047 | 0.1131 ± 0.0063 | 0.709 ± 0.003 | 0.908 ± 0.003 | 0.183 ± 0.002 |
| Ningbo | chest_only | cycle_slots_120 | 0.6554 ± 0.0071 | 0.9081 ± 0.0047 | 0.1131 ± 0.0063 | 0.709 ± 0.003 | 0.908 ± 0.003 | 0.183 ± 0.002 |
| Ningbo | chest_only | cycle_slots_201 | 0.6554 ± 0.0071 | 0.9081 ± 0.0047 | 0.1131 ± 0.0063 | 0.709 ± 0.003 | 0.908 ± 0.003 | 0.183 ± 0.002 |

## Заранее заданные сравнения и интервалы

Полная когорта и bootstrap mean показаны раздельно. Парные 95% интервалы используют те же 1,000 bootstrap индексов для всех условий, режимов и seeds; в каждом повторе сначала вычисляются seed-specific метрики, затем усредняются три seed. Интервалы условны на фиксированных моделях; пересечение нуля не означает эквивалентность.

| Когорта | Цикл/условие | Контраст | Метрика | Point estimate | Bootstrap mean | 95% CI | Класс |
|---|---|---|---|---:|---:|---:|---|
| Chapman | cycle_slots_120 | physics_aware_minus_ordinary | auprc | +0.35906 | +0.35882 | [+0.31981, +0.39847] | primary |
| Chapman | cycle_slots_120 | physics_aware_minus_ordinary | auroc | +0.27271 | +0.27280 | [+0.24777, +0.29904] | primary |
| Chapman | cycle_slots_120 | physics_aware_minus_ordinary | brier | -0.40764 | -0.40771 | [-0.42179, -0.39224] | primary |
| Chapman | cycle_slots_120 | combined_minus_ordinary | auprc | +0.34379 | +0.34410 | [+0.30632, +0.38157] | primary |
| Chapman | cycle_slots_120 | combined_minus_ordinary | auroc | +0.27103 | +0.27123 | [+0.24623, +0.29678] | primary |
| Chapman | cycle_slots_120 | combined_minus_ordinary | brier | -0.43805 | -0.43814 | [-0.45300, -0.42167] | primary |
| Chapman | cycle_slots_120 | masking_minus_ordinary | auprc | -0.01045 | -0.00998 | [-0.02879, +0.00822] | secondary |
| Chapman | cycle_slots_120 | masking_minus_ordinary | auroc | -0.02902 | -0.02895 | [-0.05075, -0.00724] | secondary |
| Chapman | cycle_slots_120 | masking_minus_ordinary | brier | -0.02443 | -0.02447 | [-0.03052, -0.01874] | secondary |
| Chapman | cycle_slots_120 | combined_minus_physics_aware | auprc | -0.01526 | -0.01471 | [-0.02944, -0.00108] | secondary |
| Chapman | cycle_slots_120 | combined_minus_physics_aware | auroc | -0.00168 | -0.00158 | [-0.00634, +0.00321] | secondary |
| Chapman | cycle_slots_120 | combined_minus_physics_aware | brier | -0.03041 | -0.03043 | [-0.03413, -0.02682] | secondary |
| Chapman | cycle_slots_120 | physics_aware_minus_chest_only | auprc | -0.02522 | -0.02536 | [-0.04086, -0.01052] | secondary |
| Chapman | cycle_slots_120 | physics_aware_minus_chest_only | auroc | -0.00735 | -0.00741 | [-0.01164, -0.00324] | secondary |
| Chapman | cycle_slots_120 | physics_aware_minus_chest_only | brier | +0.01927 | +0.01933 | [+0.01600, +0.02226] | secondary |
| Chapman | cycle_slots_120 | combined_minus_chest_only | auprc | -0.04049 | -0.04007 | [-0.05714, -0.02327] | secondary |
| Chapman | cycle_slots_120 | combined_minus_chest_only | auroc | -0.00903 | -0.00899 | [-0.01484, -0.00296] | secondary |
| Chapman | cycle_slots_120 | combined_minus_chest_only | brier | -0.01114 | -0.01110 | [-0.01487, -0.00762] | secondary |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_clean | auprc | -0.38707 | -0.38677 | [-0.42500, -0.34575] | descriptive |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_clean | auroc | -0.29369 | -0.29389 | [-0.31953, -0.26816] | descriptive |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_clean | brier | +0.45835 | +0.45844 | [+0.44091, +0.47386] | descriptive |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_RA_LL | auprc | -0.06175 | -0.06200 | [-0.09437, -0.03098] | descriptive |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_RA_LL | auroc | -0.03956 | -0.03986 | [-0.06920, -0.01058] | descriptive |
| Chapman | cycle_slots_120 | ordinary_cycle_minus_RA_LL | brier | +0.05261 | +0.05252 | [+0.04293, +0.06215] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_clean | auprc | -0.39692 | -0.39614 | [-0.43642, -0.35481] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_clean | auroc | -0.31906 | -0.31915 | [-0.34741, -0.29079] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_clean | brier | +0.42064 | +0.42065 | [+0.40476, +0.43553] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_RA_LL | auprc | -0.02408 | -0.02368 | [-0.05282, +0.00395] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_RA_LL | auroc | +0.02187 | +0.02183 | [-0.00874, +0.05397] | descriptive |
| Chapman | cycle_slots_120 | masking_cycle_minus_RA_LL | brier | +0.00086 | +0.00071 | [-0.00819, +0.00874] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_clean | auprc | -0.03894 | -0.03871 | [-0.05515, -0.02400] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_clean | auroc | -0.01925 | -0.01926 | [-0.02576, -0.01342] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_clean | brier | +0.02998 | +0.02998 | [+0.02648, +0.03349] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | auprc | -0.01210 | -0.01175 | [-0.02373, -0.00020] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | auroc | +0.00001 | -0.00003 | [-0.00404, +0.00405] | descriptive |
| Chapman | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | brier | +0.00400 | +0.00398 | [+0.00093, +0.00679] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_clean | auprc | -0.04878 | -0.04811 | [-0.06835, -0.02894] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_clean | auroc | -0.01989 | -0.01979 | [-0.02878, -0.01147] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_clean | brier | +0.01305 | +0.01296 | [+0.00890, +0.01721] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_RA_LL | auprc | -0.02154 | -0.02122 | [-0.04089, -0.00311] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_RA_LL | auroc | -0.00241 | -0.00238 | [-0.00875, +0.00421] | descriptive |
| Chapman | cycle_slots_120 | combined_cycle_minus_RA_LL | brier | +0.00254 | +0.00249 | [-0.00138, +0.00611] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_clean | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_clean | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_clean | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_RA_LL | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_RA_LL | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_120 | chest_only_cycle_minus_RA_LL | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_minus_ordinary | auprc | +0.25490 | +0.25506 | [+0.21496, +0.29703] | primary |
| Chapman | cycle_slots_201 | physics_aware_minus_ordinary | auroc | +0.17531 | +0.17541 | [+0.15356, +0.19764] | primary |
| Chapman | cycle_slots_201 | physics_aware_minus_ordinary | brier | -0.17298 | -0.17310 | [-0.18397, -0.16201] | primary |
| Chapman | cycle_slots_201 | combined_minus_ordinary | auprc | +0.23832 | +0.23860 | [+0.19961, +0.27392] | primary |
| Chapman | cycle_slots_201 | combined_minus_ordinary | auroc | +0.16707 | +0.16710 | [+0.14684, +0.18788] | primary |
| Chapman | cycle_slots_201 | combined_minus_ordinary | brier | -0.18484 | -0.18489 | [-0.19497, -0.17398] | primary |
| Chapman | cycle_slots_201 | masking_minus_ordinary | auprc | -0.06037 | -0.06012 | [-0.08509, -0.03438] | secondary |
| Chapman | cycle_slots_201 | masking_minus_ordinary | auroc | -0.07780 | -0.07800 | [-0.09642, -0.05818] | secondary |
| Chapman | cycle_slots_201 | masking_minus_ordinary | brier | +0.07242 | +0.07251 | [+0.06347, +0.08148] | secondary |
| Chapman | cycle_slots_201 | combined_minus_physics_aware | auprc | -0.01658 | -0.01646 | [-0.03017, -0.00355] | secondary |
| Chapman | cycle_slots_201 | combined_minus_physics_aware | auroc | -0.00824 | -0.00831 | [-0.01396, -0.00280] | secondary |
| Chapman | cycle_slots_201 | combined_minus_physics_aware | brier | -0.01186 | -0.01180 | [-0.01497, -0.00855] | secondary |
| Chapman | cycle_slots_201 | physics_aware_minus_chest_only | auprc | -0.00506 | -0.00525 | [-0.01906, +0.00785] | secondary |
| Chapman | cycle_slots_201 | physics_aware_minus_chest_only | auroc | -0.00743 | -0.00737 | [-0.01370, -0.00146] | secondary |
| Chapman | cycle_slots_201 | physics_aware_minus_chest_only | brier | +0.00388 | +0.00394 | [+0.00067, +0.00750] | secondary |
| Chapman | cycle_slots_201 | combined_minus_chest_only | auprc | -0.02165 | -0.02171 | [-0.04130, -0.00228] | secondary |
| Chapman | cycle_slots_201 | combined_minus_chest_only | auroc | -0.01567 | -0.01568 | [-0.02471, -0.00645] | secondary |
| Chapman | cycle_slots_201 | combined_minus_chest_only | brier | -0.00799 | -0.00786 | [-0.01245, -0.00336] | secondary |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_clean | auprc | -0.26276 | -0.26290 | [-0.30510, -0.21927] | descriptive |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_clean | auroc | -0.19637 | -0.19645 | [-0.22111, -0.17296] | descriptive |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_clean | brier | +0.20829 | +0.20844 | [+0.19699, +0.21955] | descriptive |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_RA_LL | auprc | +0.06256 | +0.06186 | [+0.03683, +0.08750] | descriptive |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_RA_LL | auroc | +0.05777 | +0.05758 | [+0.04155, +0.07312] | descriptive |
| Chapman | cycle_slots_201 | ordinary_cycle_minus_RA_LL | brier | -0.19745 | -0.19748 | [-0.20759, -0.18711] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_clean | auprc | -0.32252 | -0.32241 | [-0.36510, -0.27543] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_clean | auroc | -0.27052 | -0.27075 | [-0.29969, -0.24184] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_clean | brier | +0.26743 | +0.26764 | [+0.25411, +0.28122] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_RA_LL | auprc | +0.05031 | +0.05005 | [+0.02933, +0.07018] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_RA_LL | auroc | +0.07041 | +0.07023 | [+0.05550, +0.08487] | descriptive |
| Chapman | cycle_slots_201 | masking_cycle_minus_RA_LL | brier | -0.15234 | -0.15230 | [-0.16018, -0.14487] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_clean | auprc | -0.01878 | -0.01860 | [-0.03416, -0.00485] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_clean | auroc | -0.01932 | -0.01921 | [-0.02803, -0.01083] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_clean | brier | +0.01459 | +0.01459 | [+0.01108, +0.01847] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | auprc | +0.00805 | +0.00835 | [-0.00232, +0.01861] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | auroc | -0.00006 | +0.00002 | [-0.00505, +0.00475] | descriptive |
| Chapman | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | brier | -0.01139 | -0.01141 | [-0.01398, -0.00866] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_clean | auprc | -0.02995 | -0.02975 | [-0.05275, -0.00775] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_clean | auroc | -0.02652 | -0.02648 | [-0.03728, -0.01542] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_clean | brier | +0.01621 | +0.01621 | [+0.01189, +0.02058] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_RA_LL | auprc | -0.00271 | -0.00286 | [-0.01717, +0.01084] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_RA_LL | auroc | -0.00904 | -0.00906 | [-0.01582, -0.00179] | descriptive |
| Chapman | cycle_slots_201 | combined_cycle_minus_RA_LL | brier | +0.00570 | +0.00574 | [+0.00243, +0.00883] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_clean | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_clean | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_clean | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_RA_LL | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_RA_LL | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Chapman | cycle_slots_201 | chest_only_cycle_minus_RA_LL | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_minus_ordinary | auprc | +0.50599 | +0.50514 | [+0.47783, +0.53046] | primary |
| Ningbo | cycle_slots_120 | physics_aware_minus_ordinary | auroc | +0.41342 | +0.41312 | [+0.39358, +0.43230] | primary |
| Ningbo | cycle_slots_120 | physics_aware_minus_ordinary | brier | -0.47032 | -0.47038 | [-0.47923, -0.46132] | primary |
| Ningbo | cycle_slots_120 | combined_minus_ordinary | auprc | +0.48221 | +0.48136 | [+0.45498, +0.50563] | primary |
| Ningbo | cycle_slots_120 | combined_minus_ordinary | auroc | +0.40088 | +0.40045 | [+0.38074, +0.41931] | primary |
| Ningbo | cycle_slots_120 | combined_minus_ordinary | brier | -0.46495 | -0.46496 | [-0.47381, -0.45606] | primary |
| Ningbo | cycle_slots_120 | masking_minus_ordinary | auprc | +0.01183 | +0.01194 | [+0.00533, +0.01865] | secondary |
| Ningbo | cycle_slots_120 | masking_minus_ordinary | auroc | +0.00249 | +0.00250 | [-0.00666, +0.01174] | secondary |
| Ningbo | cycle_slots_120 | masking_minus_ordinary | brier | +0.02024 | +0.02027 | [+0.01748, +0.02307] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_physics_aware | auprc | -0.02378 | -0.02377 | [-0.03349, -0.01475] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_physics_aware | auroc | -0.01254 | -0.01266 | [-0.01612, -0.00906] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_physics_aware | brier | +0.00537 | +0.00542 | [+0.00371, +0.00706] | secondary |
| Ningbo | cycle_slots_120 | physics_aware_minus_chest_only | auprc | +0.00161 | +0.00173 | [-0.01145, +0.01442] | secondary |
| Ningbo | cycle_slots_120 | physics_aware_minus_chest_only | auroc | -0.00217 | -0.00211 | [-0.00655, +0.00216] | secondary |
| Ningbo | cycle_slots_120 | physics_aware_minus_chest_only | brier | +0.00876 | +0.00875 | [+0.00669, +0.01090] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_chest_only | auprc | -0.02216 | -0.02205 | [-0.03843, -0.00682] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_chest_only | auroc | -0.01471 | -0.01477 | [-0.01979, -0.00931] | secondary |
| Ningbo | cycle_slots_120 | combined_minus_chest_only | brier | +0.01413 | +0.01417 | [+0.01177, +0.01657] | secondary |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_clean | auprc | -0.52510 | -0.52414 | [-0.55100, -0.49690] | descriptive |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_clean | auroc | -0.43042 | -0.43005 | [-0.44808, -0.41047] | descriptive |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_clean | brier | +0.49318 | +0.49321 | [+0.48356, +0.50235] | descriptive |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_RA_LL | auprc | -0.03115 | -0.03092 | [-0.04428, -0.01758] | descriptive |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_RA_LL | auroc | -0.06998 | -0.07007 | [-0.09148, -0.04779] | descriptive |
| Ningbo | cycle_slots_120 | ordinary_cycle_minus_RA_LL | brier | +0.08044 | +0.08065 | [+0.07327, +0.08814] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_clean | auprc | -0.51117 | -0.51007 | [-0.53675, -0.48226] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_clean | auroc | -0.42788 | -0.42751 | [-0.44558, -0.40849] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_clean | brier | +0.50452 | +0.50459 | [+0.49538, +0.51364] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_RA_LL | auprc | -0.00631 | -0.00610 | [-0.01987, +0.00749] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_RA_LL | auroc | -0.00686 | -0.00715 | [-0.02807, +0.01240] | descriptive |
| Ningbo | cycle_slots_120 | masking_cycle_minus_RA_LL | brier | +0.03817 | +0.03832 | [+0.03232, +0.04402] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_clean | auprc | -0.02987 | -0.02973 | [-0.03852, -0.02095] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_clean | auroc | -0.01824 | -0.01825 | [-0.02276, -0.01403] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_clean | brier | +0.01411 | +0.01408 | [+0.01222, +0.01587] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | auprc | -0.00951 | -0.00941 | [-0.01536, -0.00319] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | auroc | -0.00848 | -0.00843 | [-0.01157, -0.00520] | descriptive |
| Ningbo | cycle_slots_120 | physics_aware_cycle_minus_RA_LL | brier | +0.00616 | +0.00610 | [+0.00478, +0.00748] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_clean | auprc | -0.04352 | -0.04338 | [-0.05855, -0.02936] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_clean | auroc | -0.03065 | -0.03071 | [-0.03687, -0.02459] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_clean | brier | +0.02476 | +0.02479 | [+0.02224, +0.02735] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_RA_LL | auprc | -0.01150 | -0.01123 | [-0.02155, -0.00126] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_RA_LL | auroc | -0.01144 | -0.01139 | [-0.01529, -0.00710] | descriptive |
| Ningbo | cycle_slots_120 | combined_cycle_minus_RA_LL | brier | +0.01178 | +0.01177 | [+0.00969, +0.01361] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_clean | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_clean | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_clean | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_RA_LL | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_RA_LL | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_120 | chest_only_cycle_minus_RA_LL | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_minus_ordinary | auprc | +0.42230 | +0.42136 | [+0.39511, +0.44556] | primary |
| Ningbo | cycle_slots_201 | physics_aware_minus_ordinary | auroc | +0.30488 | +0.30445 | [+0.28859, +0.32163] | primary |
| Ningbo | cycle_slots_201 | physics_aware_minus_ordinary | brier | -0.27590 | -0.27572 | [-0.28301, -0.26858] | primary |
| Ningbo | cycle_slots_201 | combined_minus_ordinary | auprc | +0.40428 | +0.40336 | [+0.37999, +0.42584] | primary |
| Ningbo | cycle_slots_201 | combined_minus_ordinary | auroc | +0.29600 | +0.29550 | [+0.27953, +0.31204] | primary |
| Ningbo | cycle_slots_201 | combined_minus_ordinary | brier | -0.27344 | -0.27324 | [-0.27996, -0.26611] | primary |
| Ningbo | cycle_slots_201 | masking_minus_ordinary | auprc | +0.01543 | +0.01542 | [+0.00592, +0.02468] | secondary |
| Ningbo | cycle_slots_201 | masking_minus_ordinary | auroc | +0.03537 | +0.03531 | [+0.02557, +0.04567] | secondary |
| Ningbo | cycle_slots_201 | masking_minus_ordinary | brier | -0.05094 | -0.05080 | [-0.05596, -0.04578] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_physics_aware | auprc | -0.01801 | -0.01800 | [-0.02905, -0.00681] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_physics_aware | auroc | -0.00889 | -0.00895 | [-0.01226, -0.00609] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_physics_aware | brier | +0.00246 | +0.00248 | [+0.00084, +0.00420] | secondary |
| Ningbo | cycle_slots_201 | physics_aware_minus_chest_only | auprc | -0.00718 | -0.00723 | [-0.02128, +0.00550] | secondary |
| Ningbo | cycle_slots_201 | physics_aware_minus_chest_only | auroc | -0.00381 | -0.00384 | [-0.00799, +0.00031] | secondary |
| Ningbo | cycle_slots_201 | physics_aware_minus_chest_only | brier | +0.01068 | +0.01074 | [+0.00865, +0.01288] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_chest_only | auprc | -0.02519 | -0.02523 | [-0.04218, -0.00908] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_chest_only | auroc | -0.01269 | -0.01279 | [-0.01756, -0.00833] | secondary |
| Ningbo | cycle_slots_201 | combined_minus_chest_only | brier | +0.01315 | +0.01322 | [+0.01089, +0.01565] | secondary |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_clean | auprc | -0.45020 | -0.44932 | [-0.47614, -0.42198] | descriptive |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_clean | auroc | -0.32352 | -0.32312 | [-0.34293, -0.30573] | descriptive |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_clean | brier | +0.30069 | +0.30054 | [+0.29237, +0.30844] | descriptive |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_RA_LL | auprc | +0.04375 | +0.04390 | [+0.03426, +0.05326] | descriptive |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_RA_LL | auroc | +0.03692 | +0.03687 | [+0.02719, +0.04707] | descriptive |
| Ningbo | cycle_slots_201 | ordinary_cycle_minus_RA_LL | brier | -0.11205 | -0.11202 | [-0.11673, -0.10750] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_clean | auprc | -0.43267 | -0.43176 | [-0.45880, -0.40496] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_clean | auroc | -0.28809 | -0.28776 | [-0.30443, -0.27152] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_clean | brier | +0.24084 | +0.24085 | [+0.23348, +0.24875] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_RA_LL | auprc | +0.07220 | +0.07221 | [+0.06089, +0.08434] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_RA_LL | auroc | +0.13292 | +0.13260 | [+0.12004, +0.14574] | descriptive |
| Ningbo | cycle_slots_201 | masking_cycle_minus_RA_LL | brier | -0.22551 | -0.22542 | [-0.23225, -0.21890] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_clean | auprc | -0.03866 | -0.03869 | [-0.04985, -0.02798] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_clean | auroc | -0.01988 | -0.01998 | [-0.02485, -0.01543] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_clean | brier | +0.01604 | +0.01607 | [+0.01411, +0.01806] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | auprc | -0.01830 | -0.01837 | [-0.02717, -0.00973] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | auroc | -0.01012 | -0.01016 | [-0.01355, -0.00675] | descriptive |
| Ningbo | cycle_slots_201 | physics_aware_cycle_minus_RA_LL | brier | +0.00808 | +0.00809 | [+0.00660, +0.00955] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_clean | auprc | -0.04655 | -0.04657 | [-0.06273, -0.03046] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_clean | auroc | -0.02863 | -0.02873 | [-0.03489, -0.02304] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_clean | brier | +0.02378 | +0.02384 | [+0.02133, +0.02650] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_RA_LL | auprc | -0.01452 | -0.01442 | [-0.02445, -0.00492] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_RA_LL | auroc | -0.00943 | -0.00941 | [-0.01306, -0.00583] | descriptive |
| Ningbo | cycle_slots_201 | combined_cycle_minus_RA_LL | brier | +0.01080 | +0.01081 | [+0.00919, +0.01247] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_clean | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_clean | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_clean | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_RA_LL | auprc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_RA_LL | auroc | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |
| Ningbo | cycle_slots_201 | chest_only_cycle_minus_RA_LL | brier | +0.00000 | +0.00000 | [+0.00000, +0.00000] | descriptive |

## Инженерная проверка и ограничения

Физические формулы, взаимная обратимость, три применения, композиции известных swaps, инвариантность V1–V6, channel order и identity reconstruction записаны в `ENGINEERING_CHECKS.json`. Исторический clean-вход не корректировался под повторно выведенные отведения; наблюдаемое отличие производных сигналов от реально сохранённых каналов показано числом в инженерном аудите.

Числа являются raw score метриками по ранее определённым proxy labels. Не делалась новая калибровка. Данные и результаты Chapman/Ningbo уже имели предшествующий test access. Сами циклы встречались в литературе; эксперимент не утверждает новизну перестановок.

## Файлы для проверки

- `FROZEN_PROTOCOL.json`, `PROTOCOL_HASH.json` — протокол до новых метрик и его fingerprint.
- `MODEL_AND_CACHE_AUDIT.json` — 30 checkpoint/preprocessing/gate/cache bindings.
- `TRAINING_TRANSFORM_AUDIT.md` — происхождение обучающих fault transforms и обоснование unseen-комбинаций.
- `ENGINEERING_CHECKS.json`, `REPLAY_AUDIT.json`, `CHEST_ONLY_INVARIANCE_AUDIT.json` — физические проверки, replay и полнокогортная проверка chest-only.
- `INFERENCE_MANIFEST.json`, `predictions/` — 60 новых сохранённых вероятностных файлов с ID/метками/bindings.
- `METRICS_BY_MODEL_SEED.csv`, `METRICS_MEAN_SD.csv`, `PAIRED_CONTRASTS_BOOTSTRAP.csv`, `BOOTSTRAP_CONTRAST_DRAWS.csv` — анализ и все повторные контрасты.
- `STATISTIC_IMPLEMENTATION_AUDIT.json` — сверка кода метрик с обычной реализацией на реальных bootstrap выборках.
- `TWA_AUPRC_BY_CYCLE.png`, `PLOT_DATA_TWA.csv`, `FILE_HASH_MANIFEST.json` — график, использованные агрегаты и хеши.

Этот отчёт не объявляет автоматический критерий успеха. Следующие эксперименты не запускались.
