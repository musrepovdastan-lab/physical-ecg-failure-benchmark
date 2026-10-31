# Результаты эксперимента ECGFounder mitigation

**Статус: завершён.** Инференс на test, расчёт метрик и парный bootstrap завершены.

## Дизайн и границы вывода

Сравнивались две линейные головы — clean_head и physics_head — поверх одного официального замороженного encoder ECGFounder с 1024-мерными признаками. Encoder не дообучался. Матрица: 2 когорты × 2 режима × 3 заранее заданных seed = 12 завершённых обучений. Это проверка переноса физической аугментации на голову при фиксированном encoder; она не проверяет универсальность архитектуры и не является полным дообучением ECGFounder.

Научный протокол, данные, панель масок и параметры обучения не менялись. Checkpoint выбирался по максимальному TWA AUPRC на clean validation; early stopping использовал patience 5. Порог каждого endpoint зафиксирован на 90-м процентиле вероятностей соответствующей головы для отрицательных примеров clean validation. Пороги не настраивались на test.

Основное сравнение — разность TWA AUPRC physics_head минус clean_head, усреднённая по трём seed. 95% интервалы получены на 1 000 синхронизированных парных bootstrap-повторах записей, с усреднением разностей seed внутри каждого повтора. Для оценки по 32 маскам результаты масок тоже усреднялись внутри повтора. Интервалы отражают повторную выборку записей при зафиксированных обученных головах, а не разброс между возможными seed.

## 12 обучений

Validation accuracy, balanced accuracy и macro-F1 рассчитаны для TWA с порогом, полученным на validation. Время обучения взято из результата каждого запуска. Пиковая память GPU для отдельных запусков не измерялась и в сохранённых артефактах отсутствует; завершённые головы не перезапускались ради этого замера.

| Когорта | Seed | Статус | Голова | Лучшая/последняя эпоха | Val TWA AUPRC | Val TWA AUROC | Accuracy | Balanced accuracy | Macro-F1 | Порог TWA | Причина остановки | Время, с | Память GPU | Checkpoint |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| Chapman | 104729 | COMPLETE | clean_head | 6/11 | 0.7037 | 0.9123 | 0.8707 | 0.8207 | 0.7978 | 0.7404 | early_stopping_patience_5 | 3.65 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_104729/clean_head/best.pt) |
| Chapman | 104729 | COMPLETE | physics_head | 6/11 | 0.7023 | 0.9048 | 0.8737 | 0.8287 | 0.8035 | 0.7571 | early_stopping_patience_5 | 3.38 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_104729/physics_head/best.pt) |
| Chapman | 130363 | COMPLETE | clean_head | 7/12 | 0.6929 | 0.9126 | 0.8658 | 0.8072 | 0.7882 | 0.7501 | early_stopping_patience_5 | 3.24 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_130363/clean_head/best.pt) |
| Chapman | 130363 | COMPLETE | physics_head | 7/12 | 0.6852 | 0.9040 | 0.8717 | 0.8234 | 0.7997 | 0.7540 | early_stopping_patience_5 | 3.72 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_130363/physics_head/best.pt) |
| Chapman | 155921 | COMPLETE | clean_head | 6/11 | 0.6996 | 0.9122 | 0.8678 | 0.8126 | 0.7921 | 0.7411 | early_stopping_patience_5 | 2.95 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_155921/clean_head/best.pt) |
| Chapman | 155921 | COMPLETE | physics_head | 8/13 | 0.6895 | 0.9028 | 0.8678 | 0.8126 | 0.7921 | 0.7518 | early_stopping_patience_5 | 3.93 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/chapman/seed_155921/physics_head/best.pt) |
| Ningbo | 104729 | COMPLETE | clean_head | 10/15 | 0.6030 | 0.8960 | 0.8655 | 0.7842 | 0.7574 | 0.7618 | early_stopping_patience_5 | 12.27 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_104729/clean_head/best.pt) |
| Ningbo | 104729 | COMPLETE | physics_head | 14/19 | 0.5931 | 0.8878 | 0.8638 | 0.7784 | 0.7532 | 0.7258 | early_stopping_patience_5 | 17.65 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_104729/physics_head/best.pt) |
| Ningbo | 130363 | COMPLETE | clean_head | 11/16 | 0.5972 | 0.8952 | 0.8650 | 0.7822 | 0.7560 | 0.7743 | early_stopping_patience_5 | 13.40 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_130363/clean_head/best.pt) |
| Ningbo | 130363 | COMPLETE | physics_head | 17/22 | 0.5920 | 0.8896 | 0.8624 | 0.7735 | 0.7498 | 0.7410 | early_stopping_patience_5 | 20.99 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_130363/physics_head/best.pt) |
| Ningbo | 155921 | COMPLETE | clean_head | 12/17 | 0.6054 | 0.8951 | 0.8653 | 0.7832 | 0.7567 | 0.7561 | early_stopping_patience_5 | 14.31 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_155921/clean_head/best.pt) |
| Ningbo | 155921 | COMPLETE | physics_head | 9/14 | 0.5890 | 0.8913 | 0.8632 | 0.7764 | 0.7518 | 0.7690 | early_stopping_patience_5 | 13.16 | не измерялась | [best.pt](outputs/ecgfounder_mitigation_v1/heads/ningbo/seed_155921/physics_head/best.pt) |

Все 12 запусков завершены; каждый остановился по early_stopping_patience_5. Checkpoint и validation predictions связаны с отпечатками frozen protocol и scaler для своей когорты.

## Качество на чистом test по seed

| Когорта | Голова | Seed | TWA AUPRC | TWA AUROC | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Chapman | clean_head | 104729 | 0.6551 | 0.8961 | 0.8522 | 0.8180 | 0.7806 |
| Chapman | clean_head | 130363 | 0.6549 | 0.8948 | 0.8522 | 0.8067 | 0.7765 |
| Chapman | clean_head | 155921 | 0.6572 | 0.8963 | 0.8517 | 0.8105 | 0.7775 |
| Chapman | physics_head | 104729 | 0.6586 | 0.8933 | 0.8546 | 0.8072 | 0.7789 |
| Chapman | physics_head | 130363 | 0.6527 | 0.8923 | 0.8537 | 0.8097 | 0.7789 |
| Chapman | physics_head | 155921 | 0.6495 | 0.8902 | 0.8561 | 0.8112 | 0.7817 |
| Ningbo | clean_head | 104729 | 0.6328 | 0.9060 | 0.8788 | 0.8034 | 0.7787 |
| Ningbo | clean_head | 130363 | 0.6356 | 0.9069 | 0.8808 | 0.8026 | 0.7805 |
| Ningbo | clean_head | 155921 | 0.6321 | 0.9059 | 0.8812 | 0.8041 | 0.7816 |
| Ningbo | physics_head | 104729 | 0.6272 | 0.9015 | 0.8738 | 0.8025 | 0.7732 |
| Ningbo | physics_head | 130363 | 0.6158 | 0.9010 | 0.8732 | 0.7910 | 0.7680 |
| Ningbo | physics_head | 155921 | 0.6226 | 0.9027 | 0.8767 | 0.7974 | 0.7741 |

Средние значения по трём seed:

| Cohort | Head | TWA AUPRC | TWA AUROC | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Chapman | clean_head | 0.6557 | 0.8957 | 0.8520 | 0.8117 | 0.7782 |
| Chapman | physics_head | 0.6536 | 0.8919 | 0.8548 | 0.8093 | 0.7798 |
| Ningbo | clean_head | 0.6335 | 0.9063 | 0.8803 | 0.8034 | 0.7803 |
| Ningbo | physics_head | 0.6219 | 0.9018 | 0.8746 | 0.7969 | 0.7717 |

На чистом test разность TWA AUPRC physics_head − clean_head составила −0.0022 в Chapman (paired bootstrap 95% CI: −0.0173…+0.0119, включает ноль) и −0.0117 в Ningbo (−0.0222…−0.0007). Улучшение на возмущённых условиях не сопровождалось улучшением чистого качества, особенно в Ningbo.

## Перенос: разность TWA AUPRC по seed

Значения — physics_head минус clean_head для seed 104729, 130363 и 155921. Интервал — paired bootstrap 95% по записям с усреднением seed внутри каждого повтора; для панели масок усредняются все 32 заранее заданные маски.

| Когорта | Условие | Разности по seed | Средняя разность | Paired bootstrap 95% CI |
|---|---|---|---:|---:|
| Chapman | RA-LA reversal | +0.0633, +0.0648, +0.0512 | 0.0598 | [0.0337, 0.0839] |
| Chapman | RA-LL reversal | +0.1908, +0.1895, +0.1700 | 0.1834 | [0.1414, 0.2235] |
| Chapman | LA-LL reversal | +0.0219, +0.0214, +0.0216 | 0.0216 | [0.0041, 0.0387] |
| Chapman | Цикл 120 | +0.2244, +0.2323, +0.2220 | 0.2262 | [0.1841, 0.2627] |
| Chapman | Цикл 201 | +0.0605, +0.0782, +0.0602 | 0.0663 | [0.0335, 0.0953] |
| Chapman | Точная маска пяти отведений | +0.0226, +0.0265, +0.0152 | 0.0214 | [0.0047, 0.0382] |
| Chapman | Среднее по 32 маскам | +0.0115, +0.0134, +0.0014 | 0.0088 | [-0.0025, 0.0185] |
| Ningbo | RA-LA reversal | +0.1613, +0.1422, +0.1421 | 0.1485 | [0.1256, 0.1715] |
| Ningbo | RA-LL reversal | +0.2163, +0.1987, +0.2225 | 0.2125 | [0.1909, 0.2349] |
| Ningbo | LA-LL reversal | -0.0018, -0.0103, +0.0090 | -0.0010 | [-0.0122, 0.0101] |
| Ningbo | Цикл 120 | +0.2189, +0.2016, +0.2273 | 0.2159 | [0.1942, 0.2386] |
| Ningbo | Цикл 201 | +0.1425, +0.1406, +0.1421 | 0.1417 | [0.1203, 0.1631] |
| Ningbo | Точная маска пяти отведений | +0.0744, +0.0372, +0.0850 | 0.0655 | [0.0485, 0.0826] |
| Ningbo | Среднее по 32 маскам | +0.0005, -0.0188, +0.0103 | -0.0027 | [-0.0098, 0.0047] |

**Интерпретация.** Оба held-out terminal cycle дали положительный перенос в обеих когортах: Chapman +0.066…+0.226 AUPRC, Ningbo +0.142…+0.216; bootstrap интервалы выше нуля. Инверсия RA–LL также дала крупный прирост (+0.183 в Chapman; +0.212 в Ningbo). Инверсия RA–LA положительна в обеих когортах (+0.060; +0.149). Для LA–LL эффект небольшой в Chapman (+0.022, CI выше нуля) и около нуля в Ningbo (−0.001, CI включает ноль).

Точная заранее заданная маска пяти отведений улучшила AUPRC в обеих когортах: Chapman +0.021 (CI +0.005…+0.038), Ningbo +0.066 (+0.048…+0.083). Средний эффект по всей панели из 32 масок слабее и неопределён: Chapman +0.009 (−0.002…+0.018), Ningbo −0.003 (−0.010…+0.005). Это не подтверждает общую устойчивость к произвольному выпадению каналов.

## Ранее зафиксированный результат Transformer

В frozen protocol сохранён предыдущий смешанный результат Transformer M3: скорректированный Chapman M3 gate не пройден, поскольку у seed 155921 средний по маскам TWA AUPRC снизился на 0.1126; прирост на RA–LL не компенсировал провал gate. Этап Ningbo M3 не запускался. Текущий эксперимент ECGFounder с обучением только головы — отдельная проверка и не отменяет этот смешанный результат.

## Проверки, данные и происхождение

- Аудит эквивалентности оптимизированного извлечения 32 масок только на train/validation: PASS для обеих когорт и всех 32 масок; допуск признаков 1e−6. Аудит scaler и пути предсказания: PASS; стандартизация выполняется один раз с mean/std clean train своей когорты.
- Создано 76 пакетов test-предсказаний (38 условий × 2 когорты). Payload и metadata сверены с протоколом, каноническим порядком записей, SHA feature cache и scaler, порогами и порядком шести моделей.
- Размер test split: Chapman n=2 050; Ningbo n=6 981. Согласно документации, ECGFounder предобучен на HEEDB. Сопоставления пациентов в пакете нет, поэтому пересечение с любой из когорт неизвестно и не исключено.
- CSV содержат метрики для всех трёх endpoints, каждого seed и всех 38 условий. В отчёте основной акцент на TWA AUPRC — primary outcome протокола. Reserved_test не оценивался.

## Время и артефакты

Event-log reconstruction gives approximately 7,904 s feature preparation (2 h 11 min 44 s), 126.6 s head training, and 204.6 s analysis/bootstrap. Recorded pipeline wall time was 8,249.1 s (2 h 17 min 29 s). These are end-to-end workflow times, не измерялась GPU-active time. Peak GPU memory was not recorded.

Полный состав файлов и SHA-256 для исходников, всех checkpoint, feature cache и test prediction bundles приведён в RESULTS_MANIFEST.json. Артефакты метрик: TEST_METRICS_BY_SEED.csv (1 368 строк), TEST_METRICS_MEAN_SD.csv, TEST_METRIC_DELTAS_FROM_CLEAN.csv, TWA_PAIRED_BOOTSTRAP.csv (44 сводки), CONTINUATION_AMENDMENT.json и CONTINUATION_EVENTS.jsonl.

### Отпечатки

- Frozen method protocol: 8ca374850607497506bb702bed0633fa1a540749c0903043425023a85e85ff4c
- Continuation amendment: 83f5a1915baad121c50d631d9445db51821dfac9d94fc30d2d4e824a9f25c92b
- Continuation runner: 39ecdac719797f07e1ffdc5cf6ddb188b3901c197fa0c45435e06edaf108f85f
- Официальный замороженный checkpoint ECGFounder: ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997
