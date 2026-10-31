# Статус calibration role control completion v2

**Состояние:** completion выполнен, ожидает пользовательской проверки.

- Completion protocol зафиксирован до новых интервалов; SHA-256 в `COMPLETION_PROTOCOL_SHA256.txt`.
- Все 700 bootstrap-групп по 1 000 draws покрыты: 300 переиспользованы, 400 добавлены.
- TWA point recalculation, sklearn checks и old-draw preservation прошли; см. `NUMERICAL_VERIFICATION.json`.
- Checkpoint, calibration, threshold, split, prediction и исторические outputs v1 не перезаписывались; см. `UNCHANGED_ARTIFACTS_AUDIT.json`.
- Seed limitation и lock/access chronology раскрыты. Отчёты записаны в UTF-8 и проверены.
- Нового обучения, inference, fit calibration, threshold selection или reserved test оценки не было.

Краткий итог: `FINAL_REPORT_RU.md`; таблицы ключевых метрик: `KEY_RESULTS_TABLES_RU.md`.
