Ты работаешь в проекте:
C:\Users\User\Documents\AI_research_tests\03_new_biomedical_research\projects\03_physical_ecg_failure_benchmark

Активный согласованный план — NEXT_RESEARCH_DECISIONS_20261019.md. Выполни только ближайший вычислительный этап: identity-rederivation sensitivity контроль. Работай до завершения этого пакета, без промежуточного запроса подтверждения рутинных решений. Новые модели не обучай, статью не пиши, к смеси трёх swaps и калибровочному эксперименту пока не переходи. Сообщения авторам не отправляй.

Зачем: в outputs/heldout_limb_cycles_v1 циклы получают из I, II, V1–V6, повторно вычисляя остальные limb leads. Исторический clean содержит исходные сохранённые 12 каналов. ENGINEERING_CHECKS.json зафиксировал identity reconstruction max abs 3.5 raw units и RMSE около 0.89 в обеих когортах. Влияние на модель пока не проверяли. Маленькая разница сигналов сама по себе не доказывает маленькую разницу результатов.

1. Прочитай применимые AGENTS.md, активный план и файлы:
- outputs/heldout_limb_cycles_v1/cycle_adapter.py
- outputs/heldout_limb_cycles_v1/run_heldout_cycles.py
- outputs/heldout_limb_cycles_v1/FROZEN_PROTOCOL.json
- outputs/heldout_limb_cycles_v1/ENGINEERING_CHECKS.json
- outputs/heldout_limb_cycles_v1/INFERENCE_MANIFEST.json
- outputs/heldout_limb_cycles_v1/REVIEW_NOTES_20261018.md
Проверь фактические bindings и актуальные источники. Старые future-work разделы исторических отчётов не являются текущими инструкциями. Для вычислений используй существующее окружение проекта.

2. Создай отдельный пакет outputs/identity_rederivation_sensitivity_v1/. Сначала сохрани короткий FROZEN_PROTOCOL.json и его SHA-256, затем выполняй inference. Предыдущие outcomes уже просматривались: это дополнительная проверка после анализа, не независимая подтверждающая выборка. Старые исходники, checkpoints, preprocessing, кэши, протоколы и отчёты не перезаписывай. Избегай импортов с побочными записями в исторические пакеты.

Объём фиксирован: Chapman и Ningbo; ordinary, physics_aware p050 и combined; seeds 104729, 130363, 155921. Всего 18 существующих model units. Ровно те test records, labels, порядок, best checkpoints, training preprocessing и validation-derived thresholds, которые связаны с этими единицами в cycle package. Никакого нового split, подбора threshold/calibrator, обучения, HEEDB, Founder, Transformer или панели 792 масок.

3. Сравни условия:
- original_clean: исторический сохранённый сигнал;
- identity_rederived_clean: извлечь I, II, V1–V6 и derive_display без перестановки, через ту же физическую процедуру, что использовалась для циклов; преобразование до training normalization;
- cycle_slots_120 и cycle_slots_201: готовые проверенные предсказания из старого пакета.
Новый полный inference нужен только для identity_rederived_clean — 18 файлов. Исторические comparator scores переиспользуй после проверки provenance. Если original clean cache не подтверждается replay/bindings, не продолжай молча: опиши причину и границы; недостающий comparator можно заново вычислить в новом пакете, сохранив старый.

4. Перед полным inference проверь независимыми формулами identity: III=II−I, aVR=−(I+II)/2, aVL=I−II/2, aVF=II−I/2; I, II и V1–V6 сохраняются с учётом явно описанного dtype. Проверь повторную identity на идемпотентность в выбранном dtype. Покажи signal differences по каналам и когортам; не называй raw units микровольтами без подтверждения. На фиксированных первых 32 записях каждой когорты для каждой model unit воспроизведи original_clean и оба цикла; используй прежнюю replay tolerance 1e-4 для вероятностей, опиши device/batch. Если tolerance нарушена, выясни причину до анализа. Preprocessing не переоценивай; проверь eval mode и отсутствие случайных transforms.

5. Для TWA (основной endpoint), SB и STach покажи AP/AUPRC, AUROC, raw Brier и sensitivity/specificity при прежних фиксированных порогах. Отдельно по seed и mean±SD; изменения score max/mean abs, квантили abs change, долю изменившихся решений. Сохрани новые probabilities, IDs/labels и bindings.

Основные парные сравнения:
a) identity_clean − original_clean;
b) cycle − original_clean и cycle − identity_clean для каждого цикла;
c) physics/combined − ordinary на каждом цикле: подтвердить сохранение прежнего выигрыша, поскольку cycle scores здесь не меняются. Не выдавай такую неизменность за независимое подтверждение.
Используй прежние 1000 synchronized bootstrap indices с проверкой порядка записей и исходной единицы ресэмплинга. Сохраняй парность условий и синхронизацию seeds; seeds не считать независимыми пациентами. Для TWA AP, AUROC и Brier сохрани полные точки, draws и 95% CI, отдельно от bootstrap mean. Для вторичных endpoints достаточно таблиц по seed/средних и score/decision changes; не расширяй bootstrap без необходимости.

Не назначай произвольную клиническую границу безвредности. Покажи абсолютную величину identity effect относительно измеренных cycle effects и ширину CI. Интервал, пересекающий ноль, не доказывает эквивалентность. Раздели техническую корректность пакета и научный вывод: влияние мало по измеренным числам / заметно меняет трактовку / остаётся неопределённым. Не складывай и не вычитай AP как будто это линейное разложение причин; сравнение двух clean controls — sensitivity analysis.

6. Проверь все метрики независимым повторным расчётом из сохранённых файлов, bootstrap summaries — из draws; фактические draws 0/417/999 перепроверь отдельным расчётом. Сохрани input/output hash manifest и audit с числом реально выполненных проверок. Отдельно укажи reused/new inference, elapsed time и объём. Не называй собственную проверку внешним независимым аудитом. При техническом препятствии сохрани конкретный partial status вместо выдуманного результата.

Минимальные результаты пакета: FROZEN_PROTOCOL.json, prediction bindings и новые predictions, ENGINEERING_CHECKS.json, REPLAY_AUDIT.json, CONDITION_RESULTS_BY_SEED.csv, PAIRED_CONTRASTS.csv, BOOTSTRAP_REPLICATES.csv, BOOTSTRAP_SUMMARY.csv, VERIFICATION_AUDIT.json, REPRODUCIBILITY_MANIFEST.json, FINAL_REPORT_RU.md и STATUS.md. Структуру CSV можно адаптировать, сохранив все необходимые значения.

В FINAL_REPORT_RU.md простым языком ответь: влияет ли повторное вычисление отведений на предсказания; меняет ли оно оценку потерь от циклов и выводы о защите; нужна ли дальнейшая правка обработки или достаточно раскрыть ограничение. Исторические результаты не подменять. В конце дай краткое резюме, пути к артефактам и следующий шаг по согласованному плану. После этого остановись для нашей проверки.
