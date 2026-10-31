# Минимальный план calibration control

Статус: проект проверен с уточнениями review; перед запуском зафиксировать отдельный execution protocol в новом пакете. Не перезапускать build_preflight.py: исходный генератор сохранён для прослеживаемости и содержит исправленные при review формулировки.

1. Зафиксировать membership и hashes. Train 24 430; selection 1 744; A 872; B 872. Только record/waveform-disjoint, Ningbo patient IDs отсутствуют.
2. Обучить ordinary, physics-aware p050, combined по seeds 104729/130363/155921: максимум 9 fit × 30 эпох. Точные параметры в DRAFT_CONTROL_PROTOCOL.json.
3. Checkpoint и early stop выбирать только по selection TWA AUPRC. A/B закрыты selector. Gate failure закрывает test.
4. После фиксации всех 9 checkpoints: Platt fit только на clean A; negative raw q90 только на clean B. Не подбирать параметры заново. С Huang всего 36 endpoint fits и 36 selections. Численные параметры и gates уточнены в DRAFT_CONTROL_PROTOCOL.json.
5. Huang: frozen detector -> predicted inverse на incoming damaged signal -> новый ordinary checkpoint; собственные pipeline A/B scores и recovery-v2 handoff regression.
6. Открыть прежние Ningbo test и HEEDB только после lock; HEEDB — stability check, не независимая репликация.
7. Сохранить реальные runtime/memory и provenance.

Бюджет 9 fits, ≤270 эпох. По измеренным похожим Ningbo runs: 2.77–5.10 ч, медианный ориентир 3.18 ч; измеренный пик allocated 0.264 GiB/run. Это экстраполяция.
