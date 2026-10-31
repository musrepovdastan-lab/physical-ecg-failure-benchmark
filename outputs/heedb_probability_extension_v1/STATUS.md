# Статус пакета HEEDB probability extension v1

Статус: **ЗАВЕРШЕНО**. Ограниченное внешнее расширение p010/p025 выполнено на уже анализировавшейся когорте HEEDB (10 000 пациентов); результат не является новой слепой независимой подтверждающей выборкой.

- До нового inference заморожен протокол; поправки 001–002 внесены до replay, 003–005 документируют только пути и аудитные/анализные реализации. Исходные варианты протокола сохранены.
- Bounded replay: 18/18 PASS; максимальное различие 0 при допуске 0.0001.
- HEEDB inference: 12/12 prediction files, 2 arms × 3 seeds × 2 conditions; обучение отсутствовало.
- Анализ: 1 071 seed-specific строки с отдельными score_mode; 357 строк средних; 48 предписанных bootstrap-интервалов и 48 draws.
- Calibration decision audit: 144 строки, 0 несовпадений; calibration provenance и checkpoint/preprocessing hashes сохранены.
- Reserved test не оценивался. Founder и новые модели/настройки не добавлялись.

Итог: `RESULTS_REPORT_RU.md`. Протокол: `FROZEN_EXTENSION_PROTOCOL.json`. Машиночитаемый индекс артефактов: `REPRODUCIBILITY_MANIFEST.json`.
