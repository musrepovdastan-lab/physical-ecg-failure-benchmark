Ты работаешь в проекте:
C:\Users\User\Documents\AI_research_tests\03_new_biomedical_research\projects\03_physical_ecg_failure_benchmark

Выполни следующий согласованный этап до результата — ограниченный calibration-role control. Preflight проверен и принят. Это разрешённые 9 новых обучений и следующий анализ, а не ещё один план. Не останавливайся после freeze/написания кода; продолжай автономно до завершения пакета, кроме реального технического препятствия или заранее заданного gate failure.

Прочитай применимые AGENTS.md, NEXT_RESEARCH_DECISIONS_20261019.md, а также:
- outputs/calibration_independence_preflight_v1/REVIEW_NOTES_20261019.md;
- там же DRAFT_CONTROL_PROTOCOL.json, PROPOSED_MEMBERSHIP.csv, SPLIT_COUNTS.csv, IDENTITY_AND_LEAKAGE_AUDIT.json, MINIMAL_EXECUTION_PLAN.md;
- outputs/three_swap_population_mixture_recovery_v2/REVIEW_NOTES_20261019.md и восстановленный input-handoff код;
- предыдущие frozen training configs обычного Ningbo ResNet, p050 и combined, их фактические train loops и normalization provenance;
- проверенные Platt/mixture/threshold реализации прошлых пакетов.
build_preflight.py поверх принятых документов не запускать: исходный генератор содержит исправленные при review неточные формулировки. Старые запрещения предыдущих протоколов относятся к их историческому scope; текущее явное задание разрешает этот новый ограниченный контроль в отдельном пакете.

Цель: проверить, сохраняются ли сравнительные выводы о защите, clean cost и mostly-clean populations, когда checkpoint selection, calibration fitting и threshold selection для новых моделей выполняются на разных записях. Старый fit был source-only, не на HEEDB; проблема — прежнее пересечение его calibration pool с checkpoint selection. Не обещать заранее определённый исход и не интерпретировать весь old→new сдвиг как измеренный calibration bias: меняются также размер частей и выбор checkpoint.

1. Отдельная папка outputs/calibration_role_control_v1/. Исторические результаты не перезаписывать. Сначала зафиксируй execution protocol/hash, код, exact memberships и immutable bindings; изменяемый активный план сохранить отдельным context snapshot. Различать canonical fingerprint и file SHA-256. До новых outcomes уточни только необходимые инженерные детали из старых frozen sources, не выбирай рецепт по test. После freeze сразу выполняй пакет, не запрашивая повторного подтверждения уже согласованного scope.

Данные:
- исходный Ningbo train 24 430 записей не менять;
- model_selection 1 744, calibration_A 872, threshold_B 872 — ровно из принятого PROPOSED_MEMBERSHIP.csv, seed разделения 20261019;
- прежний Ningbo test 6 981 и та же HEEDB I0006 когорта 10 000 пациентов, прежние endpoints/proxy mappings;
- patient IDs Ningbo не установлены: record/waveform-disjoint, не patient-disjoint.
Не генерировать новое разделение, не возвращать excluded records и не подбирать split заново. Preprocessing/pos_weight только из неизменного train; прежние train-derived stats можно переиспользовать при exact provenance. Не путать 100 Hz source cache с 500 Hz HEEDB raw и stride5.

2. Всего 9 новых fits: ordinary, physics_aware_p050, combined × seeds 104729/130363/155921. Старые checkpoints не заменяют новые fits. ECGResNet-small, weighted multilabel BCE с train-only pos_weight cap20; AdamW lr=.001, weight_decay=.0001; batch64; максимум30 эпох, CUDA autocast/GradScaler; selection TWA AUPRC early stopping patience5; best checkpoint по строгому улучшению, earliest tie. Никакого tuning.

Рецепты до train normalization:
- ordinary: clean100%;
- p050: clean50% + один равновероятный swap RA_LA/RA_LL/LA_LL50%;
- combined: clean50% + uniform five-displayed-lead mask25% + один uniform swap25%; режимы взаимоисключающие.
Маска удаляет именно выбранные5 из12, остальные7 сохраняет, sentinel — train lead mean. Использовать исторические проверенные определения и seed/epoch/record sampling без смены рецепта. Проверь raw transforms/семантику masks перед обучением, но не запускай 792-mask evaluation.

Только model_selection используется в epoch scoring, ранней остановке и adequacy gate. A/B не должны участвовать в gradients, loss, checkpoint/epoch selection, normalization/pos_weight или любом критерии продолжения. Знание их labels/counts для заранее выполненной стратификации допустимо; их scores при обучении не вычислять. Проверить это не только описанием: distinct loaders, membership assertions и provenance/traces оценивания.

3. До длинного запуска выполни небольшой engineering smoke на train и selection. Зафиксируй baseline adequacy gate на model_selection: TWA AUROC>=.75, AUPRC>=.45; finite losses/predictions и checkpoint/membership integrity для всех9. Это прежние технические критерии пригодности, не критерии научной победы. При gate failure не открывать test/HEEDB и не менять пороги/архитектуру ради PASS; сохранить конкретный результат. Не отбрасывать неудобный seed.

Обучать последовательно с сохранением текущего состояния. Resume должен восстанавливать optimizer/scaler/RNG, best epoch/metric, patience и next_epoch, не повторять завершённую эпоху и не выдавать last epoch за best. Для завершённых fits проверять hashes/config/membership перед reuse; не повторять fit без причины. Сохранять timings, device, histories и измеренную память, progress после каждого fit. Ориентир прошлых запусков 3–5 часов — оценка, не повод обрывать корректную работу по произвольному таймеру. Не добавлять десятый fit или новую tuning сетку.

4. После lock всех9 checkpoints заново загрузить именно best веса и проверить selection score против выбранной epoch/history. Затем получить clean scores A и B. Не переиспользовать старые diagnostic scores от других весов. Модель/нормализация после lock не меняются.

Для каждого метода×seed×endpoint: Platt fit только на clean A, отрицательные raw scores clean B — только для q90 operating threshold. Положительный slope a=exp(theta0), p clipping[1e-6,1−1e-6]; unweighted logistic loss; начальные theta=[0,0], L-BFGS-B bounds theta0[-12,12], b[-30,30], maxiter2000, ftol1e-13, gtol1e-8, maxls50; применение sigmoid с прежней численной защитой. Quantile numpy linear. Зафиксировать score/threshold precision до вычисления и проверить cast/ties; не округлять удобным образом после просмотра результатов.

Huang: frozen detector → predicted-class inverse к тому же incoming ECG → соответствующий НОВЫЙ ordinary classifier. Три новых ordinary models переиспользуются, дополнительных Huang обучений нет. Для Huang нужны собственные full-pipeline clean A/B diagnostic scores и собственные Platt/q90. Detector outputs можно переиспользовать только с проверенными input/order/checkpoint/class bindings; старые Huang diagnostic probabilities от старого classifier — нельзя. Никакой oracle коррекции; ошибочные исправления clean включать.

Всего 36 endpoint Platt fits и 36 q90 selections: 27 core +9 Huang. Проверить finite coefficients/convergence и обе classes; ошибки не замалчивать и не выбирать fallback по test. При secondary fit failure — явно missing endpoint; при primary TWA failure test gate закрыт. После этого сохранить immutable calibration/threshold lock до test/HEEDB scoring. Primary calls: raw_score>=raw_B_q90; calibrated threshold — его mapping, различия из-за clipping/saturation явно аудитировать, не менять threshold ради совпадения. Calibration не даёт нового operating point сама по себе.

5. До полного Huang scoring отдельно проверить исправленный handoff: simulated fault → detector/input → predicted inverse на том же faulty signal → classifier. Независимые явные формулы, predicted classes0..3, отсутствие mutation; regression на прежнюю v1 missing-fault ошибку. После нового inference — source-aligned first/middle/last batch replay, прежние tolerances diagnostic1e-4/detector1e-5 и batch/precision paths. Не ограничиваться пересчётом метрик неверных scores.

6. Только после всех locks/gates оцени новые модели и Huang на прежних Ningbo test и HEEDB. Условия clean, RA_LL, RA_LA, LA_LL; q=[0,.001,.005,.01,.05,.1,1]; сценарии RA_LL_only и equal_three_swap. Во втором веса[1−q,q/3,q/3,q/3]; это условный состав faults, не реальная распространённость. Один исходный record/patient, несколько связанных views; total weight1 на единицу. AP/AUROC считать по joint weighted scores, не mean condition AP.

По каждому seed и mean±sample SD: raw/calibrated AP, AUROC, Brier, ECE15; sensitivity/specificity, predicted-positive fraction, FP/FN на1000. TWA primary, SB/STach secondary. Не усреднять probabilities вместо seed metrics. Clean raw Brier/AUROC cost показать явно, даже если source calibration улучшает calibrated Brier. HEEDB proxy labels не становятся доказанными вероятностями underlying disease.

Для TWA: прежние1000 paired synchronized bootstrap indices, Ningbo record unit и HEEDB patient unit. Все views/методы/seeds разделяют multiplicity. Primary contrasts physics−ordinary, combined−ordinary, Huang−ordinary, physics−Huang, combined−Huang; raw AP, raw/calibrated Brier и primary operating sensitivity/FP1000 по прежнему scope. Сохранить full points отдельно от bootstrap means и CI/draws. Не считать3seeds независимыми пациентами. Интервалы conditional на выбранных моделях/fits; zero crossing не equivalence. Не расширять эксперимент на Chapman, Founder, Transformer, p-grid, cycles, 792 masks или новые endpoints.

7. Сравни с принятым recovery_v2 и прежними калибровочными результатами по одинаковым cohorts/q/metrics. Главный вопрос — сохраняются ли clean-cost и tradeoffs методов в новом record-role control. Все old→new differences описывать с учётом новых checkpoint selection и уменьшенных A/B sizes, а не приписывать одной причине. Прежняя HEEDB уже изучалась; это sensitivity control с исторически разработанными гиперпараметрами, не новая независимая внешняя репликация. Формулировка patient independence запрещена без фактических IDs.

8. До завершения независимо пересчитать метрики из saved predictions, summaries из draws и actual draws0/417/999 отдельным sklearn расчётом; проверить A-only/B-only/selection-only использование, best-checkpoint replay, q0/weights/FP denominator, positive-slope mapping/ties и input path. Отдельные AUDIT для scientific arithmetic и pipeline/data roles — один не заменяет другой. Сохранить input/output hashes и корректные IDs/bindings во всех score файлах. При исправлениях сохранять failed attempts/amendments; не менять scientific scope после outcomes.

Deliverables в новом пакете: FROZEN_EXECUTION_PROTOCOL.json/hash, immutable split/context snapshots; per-fit configs/checkpoints/history/resume/provenance/timings; selection/A/B scores с roles; TRAINING_STATUS.json; ALL_MODELS_LOCK.json; CALIBRATION_AND_THRESHOLD_LOCK.json; detector/prediction bindings; engineering/role/handoff/replay audits; condition/mixture/mean tables; paired points/draws/CI; historical comparison; VERIFICATION_AUDIT.json, REPRODUCIBILITY_MANIFEST.json, STATUS.md и FINAL_REPORT_RU.md. Имена вспомогательных файлов можно адаптировать, сохранив необходимую проверяемость.

В финальном отчёте простым русским языком: все ли9 fits завершены; что изменилось на чистых и corrupted ECG; сохраняются ли преимущества/недостатки combined/p050 относительно Huang при q=.01/.1/1; calibration/Brier и ложные тревоги отдельно; что нельзя заключить. Не объявлять универсальную победу и не скрывать отрицательный результат. Обновить активный план до «контроль выполнен, ожидает проверки», не объявлять принятие. После пакета остановиться. Статью и новые исследования не начинать.
