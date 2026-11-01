# Краткий отчёт по технической проверке

**Результат: PASS.**

- Исправлена сборка общего bootstrap-архива. Предварительно проверяются ровно 700 ожидаемых ключей, отсутствие дубликатов и конечные массивы длины 1,000.
- На существующих архивах Ningbo и HEEDB проверена фактическая функция записи: все 700 групп и все draws в точности совпали с текущим `PAIRED_BOOTSTRAP_DRAWS_ALL.npz`. Полная статистика не пересчитывалась.
- Проверен поддерживаемый путь воспроизведения через `train_all_fresh.py --check-inputs`: PASS для 65 научных входов, original frozen protocol/runner hashes и трёх пар исторических/текущих документных hashes. Исторические документы извлечены из Git `HEAD`, hashes точно совпали с frozen значениями.
- `control.py`, исходный frozen protocol, checkpoints, predictions, calibration/threshold locks и научные таблицы не менялись. Обучение, inference и refit не запускались.
- Manifest v2 построен по явному списку из 300 требуемых файлов; pycache, временные, failed-payload и review материалы исключены. Все перечисленные hashes прошли проверку.

Технические детали и поддерживаемые команды: [AMENDMENT_003.md](AMENDMENT_003.md), [SUPPORTED_REPRODUCTION.md](SUPPORTED_REPRODUCTION.md), [INPUT_VALIDATION_AMENDMENT_001.md](INPUT_VALIDATION_AMENDMENT_001.md).
