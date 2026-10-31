# Отчёт об обучениях

Все девять ранее выполненных fits сохранены; в completion-пакете их не повторяли. Checkpoint выбирался по model_selection. Best checkpoint и history совпали в независимом аудите.

| Метод | Seed | Лучшая эпоха | TWA AUPRC selection | AUROC selection | Время, с | Память GPU при обучении | Checkpoint |
|---|---:|---:|---:|---:|---:|---|---|
| ordinary | 104729 | 10 | 0.6519 | 0.9236 | 456.9 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/ordinary/seed_104729/best.pt](../calibration_role_control_v1/fits/ordinary/seed_104729/best.pt) |
| ordinary | 130363 | 15 | 0.6625 | 0.9181 | 457.3 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/ordinary/seed_130363/best.pt](../calibration_role_control_v1/fits/ordinary/seed_130363/best.pt) |
| ordinary | 155921 | 11 | 0.6808 | 0.9258 | 365.5 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/ordinary/seed_155921/best.pt](../calibration_role_control_v1/fits/ordinary/seed_155921/best.pt) |
| physics_aware_p050 | 104729 | 20 | 0.6677 | 0.9221 | 581.3 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/physics_aware_p050/seed_104729/best.pt](../calibration_role_control_v1/fits/physics_aware_p050/seed_104729/best.pt) |
| physics_aware_p050 | 130363 | 20 | 0.6852 | 0.9233 | 569.6 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/physics_aware_p050/seed_130363/best.pt](../calibration_role_control_v1/fits/physics_aware_p050/seed_130363/best.pt) |
| physics_aware_p050 | 155921 | 11 | 0.6512 | 0.9189 | 363.2 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/physics_aware_p050/seed_155921/best.pt](../calibration_role_control_v1/fits/physics_aware_p050/seed_155921/best.pt) |
| combined | 104729 | 12 | 0.6545 | 0.9207 | 384.4 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/combined/seed_104729/best.pt](../calibration_role_control_v1/fits/combined/seed_104729/best.pt) |
| combined | 130363 | 17 | 0.6782 | 0.9209 | 497.2 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/combined/seed_130363/best.pt](../calibration_role_control_v1/fits/combined/seed_130363/best.pt) |
| combined | 155921 | 19 | 0.6753 | 0.9247 | 540.8 | Не сохранена; replay peak см. ниже | [../calibration_role_control_v1/fits/combined/seed_155921/best.pt](../calibration_role_control_v1/fits/combined/seed_155921/best.pt) |

GPU: NVIDIA GeForce RTX 5070 Laptop GPU. Сохранённый показатель памяти (112.5 MiB, 112.8 MiB) измерен после сброса CUDA peak counter на replay лучшего checkpoint, а не в training loop. Поэтому фактический training peak memory для всех fits — unavailable; новых fits ради этого не запускали.

Fit seed был выставлен уже после создания модели. См. `INITIALIZATION_PROVENANCE_NOTE_RU.md`: seed управляет последующим обучением, но не начальными весами.
