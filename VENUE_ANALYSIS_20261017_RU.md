# Куда подавать Physical ECG Failure Benchmark

Проверено 17 октября 2026 года. Горизонт: оставшаяся часть 2026 года и цикл 2027 года. Даты ниже — даты подачи новой работы, если явно не указано иное. «Не объявлено» не означает, что дедлайн обязательно будет в обычный для площадки месяц.

## Решение для проекта

Рекомендуемый маршрут: **короткая работа в AAAI-27 Deployable AI в ноябре → полная статья в CHIL 2027 к 3 февраля**. Making Sense — более узкая альтернативная аудитория по ECG/биосигналам; ASTAD — альтернативная аудитория по временным аномалиям. Для двух последних сначала установить публикационный статус. Среди самостоятельных инженерных конференций особенно подходят EMBC, EUSIPCO, Computing in Cardiology и MLSP. Для более амбициозной программы — NeurIPS Evaluations & Datasets 2027, когда появится CFP.

Это оценка соответствия темы, готовности доказательств, времени и возможности последующей публикации. Она не является оценкой вероятности принятия. Воркшоп при известной конференции и основной трек конференции — разные результаты публикации.

## 1. Что именно представляет собой наша работа

Основа оценки — [план после замечаний ментора от 16 октября](RESEARCH_PLAN_AFTER_MENTOR_20261016.md), [текущий черновик](outputs/submission_evidence_v1/MANUSCRIPT_DRAFT.md), [HEEDB confirmation](outputs/heedb_independent_confirmation_v1/FINAL_REPORT.md), [Huang comparison](outputs/heedb_huang_exploratory_v1/FINAL_COMPARISON.md), таблицы M1 и статус M3. README и старый readiness PASS не использовались как окончательное подтверждение сегодняшней готовности. Новое обучение и повторная проверка всех чисел не выполнялись.

**Наиболее убедительная история:** физически согласованные перестановки терминалов создают структурированный сдвиг входа ЭКГ; стандартная проверка отсутствующих отведений может не описывать такую уязвимость; разные способы защиты имеют разные компромиссы по ранжированию, калибровке, пороговым ошибкам и стоимости применения.

Сильные стороны существующих материалов:

- Электрически обоснованные преобразования RA–LA, RA–LL, LA–LL вместо произвольной порчи каналов.
- Chapman и Ningbo, три seed, сохранённые протоколы и предсказания.
- В отчёте независимой HEEDB-группы: 10 000 пациентов, RA–LL TWA AUPRC 0,1647 у baseline и 0,5683 у physics-aware; парная разница +0,4036, 95% CI 0,3785–0,4255. Это результат сохранённого отчёта, а не новый пересчёт.
- Наличие сильного опубликованного detect-and-correct baseline и отрицательных для универсального превосходства augmentation результатов.
- На Chapman имеются clean/physics-aware Transformer checkpoints; есть frozen-backbone ECGFounder с обученной линейной головой.

Что ограничивает подачу:

1. **Ошибка направления маски.** План от 16 октября указывает инверсию в Ningbo/M2/M3 evaluator. Проверки «убраны нужные пять отведений», exact-mask и затронутые сравнения надо восстановить. Полный перебор комбинаций не исправляет различие между удалением пяти и семи каналов. Старый manuscript содержит соответствующие числа, поэтому его нельзя просто перенести в шаблон конференции. Чистые и reversal-ветки непосредственно этой ошибкой не затронуты, но итоговые claims о missingness требуют аудита.
2. **Новизна.** Cao, Shi, Wu, ICASSP 2020, “Reversal No Longer Matters…” уже изучает reversal-robust arrhythmia detection. Нельзя заявлять первую такую диагностику. Наш вклад следует связывать с оценкой, границами missingness-тестов и сравнением защит. Полный литературный разбор остаётся задачей, отмеченной локальным планом.
3. **Простой сильный контроль.** Нужен chest-only V1–V6, иначе эффект защиты может объясняться отказом от limb leads.
4. **Широта моделей.** M3 пока нельзя считать завершённой репликацией; статус workflow содержит незавершённую статистику. Frozen Founder с линейной головой не равен полному fine-tuning foundation model.
5. **Сильный конкурент.** Huang pipeline имеет RA–LL AUPRC 0,5811 против 0,5683; CI разницы включает ноль. Он лучше по некоторым другим метрикам. Это post-hoc exploratory comparison, не доказательство превосходства, эквивалентности или универсальной практической выгоды augmentation.
6. **Mostly-clean применение.** В локальном плане показан exploratory Brier crossover около 4,08% RA–LL faults для augmentation против baseline. Это сценарий смеси при заданных предположениях, не измеренная распространённость ошибок. Для deployment-статьи нужны кривые по q_fault, validation-only calibration и operating points.
7. **Клинические границы.** HEEDB labels — proxy-коды; отдельная группа пациентов не означает ещё один внешний центр. Faults симулированы. Улучшение AUPRC не означает улучшения чувствительности или пользы пациентам.

**Вывод:** тема подходит для evaluation/robustness/health/biosignal venues уже по сути исследования. Готовность конкретного текста зависит прежде всего от корректности сравнений. Новая архитектура сама по себе не закрывает этот вопрос.

## 2. Ближайшие воркшопы: разбор каждого содержательного кандидата

Оценки соответствия — экспертное суждение применительно к нынешнему проекту. «Высокое» не означает низкий порог рецензирования.

| Воркшоп | Дедлайн новой работы | Формат и публикация | Соответствие и решение |
|---|---|---|---|
| [AAAI-27 Deployable AI](https://dai-workshop.github.io/) | **20.11.2026 AoE** | 4 или 7 страниц без references/appendix; явно non-archival | **Высокое; лучший маршрут к последующей полной статье.** Упор на структурированные ошибки сенсора, calibration, q_fault и сравнение augmentation / chest-only / correction. Не заявлять готовность клинического продукта. |
| [AAAI-27 Making Sense](https://making-sense-workshop.github.io/) | **20.11.2026 23:59 AoE** | Short ≤4, long ≤7 страниц; research и demo/dataset tracks. Архивный статус на прочитанной странице не указан | **Высокое по биосигналам.** ECG, missing modalities и sensor evaluation явно входят в темы. С учётом акцента на foundation models полезен аккуратный ECGFounder-анализ. Называть frozen-feature pipeline точно. |
| [AAAI-27 ASTAD](https://ismart.ece.mcgill.ca/astad4/) | **20.11.2026**; точное время уточнить | До 7 страниц +2 только references; double-blind. Архивный статус не подтверждён | **Хорошее, условное.** Подходит structured sensor anomalies / robustness benchmark. Наша основная задача — сохранение диагностических предсказаний, поэтому не выдавать классификатор диагнозов за новый детектор неисправностей. |
| [AAAI-27 AI4TS: упоминание организатора](https://sites.google.com/site/qingsongwen8/services) | **CFP 2027 не удалось подтвердить** | Организатор перечисляет AAAI’27-AI4TS; связанная страница не прочиталась | **Потенциально очень хорошее.** Включить в список проверки, но не переносить сроки и правила AAAI-26 на AAAI-27. |
| [WACV-27 FABRIC](https://sites.google.com/view/fabric-wacv) | **20.10.2026**, продление; часовой пояс не указан | Архивные WACV workshop proceedings | **Умеренное.** Biomedical foundation-model evaluation в scope, но нынешнее ядро работы — ResNet/Transformer robustness. Три дня недостаточны для закрытия известных научных пробелов. Не основной выбор. |
| [WACV-27 SENSE](https://sensewacv.github.io/Website/) | **21.10.2026**, точное время не указано | Архивные proceedings / IEEE Xplore | **Умеренное–низкое.** Healthcare sensing подходит, центральная тема agentic reasoning у нас отсутствует. Четыре дня и известный аудит делают спешную подачу неразумной. |
| [WACV-27 Data4VFM](https://data4vfm-wacv2027workshop.github.io/index.html) | **23.11.2026 11:59 UTC** | 4 страницы; явно non-archival | **Низкое–умеренное.** Время есть, но visual/multimodal foundation data — другая основная аудитория. Чистая 1D ECG robustness работа слабее соответствует теме. Перевод ECG в картинку сам по себе проблему не решает. |
| [WACV-27 Vision for Biomedical Discovery](https://cvpath-vision4neuro.github.io/wacv2027/) | Regular **16.10.2026 23:59 AoE**; у short/abstract отдельный срок | Regular 8 страниц, archival; другие вклады имеют отдельные правила | **Низкое.** Два направления: pathology и clinical neuroscience/vision. Наш ECG-проект не совпадает с ними. На момент начала поиска regular-срок ещё не истёк: в UTC+5 это 17 октября 16:59. Техническое окно не делает площадку подходящей. |

AAAI workshops проходят 22–23 февраля 2027 в Монреале; [общая политика](https://aaai.org/conference/aaai/aaai-27/workshops-call/) требует очного представления соавтором. Важны регистрация, поездка и возможность получить визу. На этапе поиска авторов и организаторов никто не контактировал.

**Менее подходящие AAAI-кандидаты, проверенные отдельно:**

| Площадка | Срок / статус | Почему не выбирать сейчас |
|---|---|---|
| [SAFE](https://www.safeworkshop.org/aaai-2027/) | 02.11.2026 AoE | Основной предмет — synthetic media, deepfakes, forensics и adversarial attacks. Случайная перестановка ECG-кабеля — другая проблема. |
| [AICS](https://aics.site/AICS2027/cfp.html) | На странице 20.11.2026; есть ошибки в других датах | Кибербезопасность не является нашим исследовательским вопросом. Не добавлять искусственный adversary ради соответствия CFP. |
| [Effective and Efficient Evaluation for AI Scientists](https://eval-ai-scientists.github.io/aaai27/) | 20.11.2026 **06:00 UTC** | Оценивает автономные исследовательские системы. Сам факт использования ИИ в подготовке исследования не превращает нашу работу в evaluation AI scientists. |
| [CALE-AI](https://www.vprover.org/cfp/CALE-AI2027) | Проверен scope; срок не использован | Изучает долгосрочное воздействие AI на людей и поведение. У нас нет такого исследования. |

Для WACV просмотрен весь [официальный список из 26 воркшопов](https://wacv.thecvf.com/Conferences/2027/Workshops). Помимо пяти кандидатов выше, остальные ориентированы на конкретные визуальные задачи, документы, видео, биометрию, геопространственные данные, транспорт и embodied systems; самостоятельного подходящего места для нынешнего ECG benchmark среди них не обнаружено.

## 3. NeurIPS 2026: что подходит по теме и что уже закрыто

Просмотрен [официальный список 102 принятых воркшопов](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/). Ниже — содержательные кандидаты и близкие по названию варианты. Декабрьское проведение не означает возможность октябрьской подачи. Более поздний camera-ready относится к уже принятым работам.

| Воркшоп | Дедлайн / статус | Разбор для нашего проекта |
|---|---|---|
| [Trust-AI-Eval / Can We Trust AI Evaluation?](https://tai-eval.github.io/) | **29.08.2026; закрыт** | Один из лучших тематических fit: надёжность стресс-тестов, construct validity и неверные выводы из benchmark. Первая цель для аналогичного будущего CFP. |
| [Foundation Models for the Brain and Body](https://brainbodyfm-workshop.github.io/) | Papers **05.09**, demos **19.09.2026; прошли** | Прямо включает ECG и biosignals. Для нашей работы особенно уместен аудит foundation-model pipeline, но нельзя обещать новое предобучение. |
| [TS-LIMITS](https://ts-limits.github.io/) | **05.09.2026**, по индексированному CFP; даты названы tentative | Хорош для latency / memory / time-series generalization, если deployment-cost анализ становится полноценным вкладом. Нынешний сравнительный runtime надо переносить осторожно между архитектурами. |
| [ICBINB-BIO](https://icbinb-bio.github.io/) | Продлён до **02.09.2026; закрыт** | Хорош по failure analysis и неожиданно сильным простым baseline. Аудит missingness и конкурентный chest-only/Huang дают честную историю; ECG менее центрально, чем молекулярная биология. |
| [FMTS](https://fmts-workshop.github.io/) | **15.09.2026 AoE; прошёл** | Есть temporal reliability, но workshop преимущественно о foundation/world modeling. Текущий вклад уже и менее непосредственно соответствует теме. |
| [Responsible Communication of ML Research in Biomedicine](https://translatingmlresearch.github.io/RCMLR/) | **29.08.2026; прошёл** | Подходит отдельному case study о различии AUPRC, sensitivity, proxy labels и клинических claims. Слабее как место для основной экспериментальной статьи. |
| [Sim2Science](https://www.sim2science.com/) | Продлён до **02.09.2026; прошёл** | Центральная проблема — несовершенство scientific simulators. Наш электрический transform точен при предположениях; реальный sim-to-real gap пока не измерен. Соответствие условное. |
| [Trustworthy AI for Good](https://trustworthy-ai-for-good.github.io/) | **01.09.2026; прошёл** | Допускает reliability/evaluation, но повестка заметно шире и более социальная/агентная. По теме ниже Trust-AI-Eval. |
| [GenAI4Health](https://genai4health.github.io/2026-NeurIPS/) | **09.09.2026; прошёл** | Наш benchmark не посвящён generative/agentic AI. Возможен только содержательный foundation-model компонент, не переименование augmentation в GenAI. |
| [NeurReps](https://neurreps.org/) | На странице прямо **submissions closed** | Перестановки образуют математическую структуру, но использования transforms в augmentation недостаточно для статьи о symmetry/geometry. Нужны явная групповая формализация и содержательный метод/анализ. |
| [Physical World AI](https://physworld-org.github.io/physworld.github.io/) | **09.09.2026; прошёл** | «Physical» в названии проекта недостаточно: программа преимущественно о 3D/robotics/physical sensing. |
| [EvoRobust](https://liuj-2023.github.io/EvoRobust-NeurIPS2026/) | Приём завершён, опубликованы camera-ready инструкции | Self-evolving diversity-driven search отсутствует в нашей работе. Обычная robustness-тема не делает этот workshop приоритетом. |

Остальные элементы списка отсеяны по предмету: языковые/агентные системы, биомолекулы, робототехника, отдельные естественнонаучные приложения или инфраструктура, которую проект не исследует. Проверка ближайших тематических CFP и поиск late-breaking возможностей не дали подтверждённого открытого подходящего приёма на NeurIPS 2026. Это не утверждение об отсутствии любых неиндексированных специальных приглашений.

**ML4H — отдельный симпозиум:** [ML4H 2026](https://ml4h.ahli.cc/) проходит 6–7 декабря в Сиднее; paper и demo submissions закрыты, основной срок был 10 сентября. Не учитывать его как ещё один открытый NeurIPS workshop. Следующее издание — хороший кандидат после появления CFP.

## 4. Самостоятельные конференции и большие треки

| Площадка | Подтверждённое окно | Оценка применительно к проекту |
|---|---|---|
| [CHIL 2027](https://chil.ahli.cc/submit/call-for-papers/) | **03.02.2027**, accounts к 15.01; решение 12.04 | **Основная цель полной статьи.** CFP явно допускает applied/evaluative health ML. Формат 8–10 страниц. Есть время для исправлений и сильных сравнений. |
| [MLHC 2027](https://mlhc.org/) | 11–13.08.2027, ETH Zurich; срок подачи **не объявлен** | **Очень хорошая аудитория.** Полезны клинически проверенные endpoints, внешняя проверка и сравнение стратегий. Не использовать CFP 2026 как календарь 2027. |
| [IEEE EMBC 2027](https://embc.embs.org/2027/papers/) | **24.01.2027 AoE**, открытие в декабре; решение 16.04 | **Сильный инженерный вариант.** 4 страницы технического текста + пятая только references, ≤20 ссылок. Большую матрицу придётся сократить до одного законченного вопроса. [Проведение](https://embc.embs.org/2027/about/faqs/): 11–15 июля, Сингапур. |
| [EUSIPCO 2027](https://eusipco2027.de/) | **06.02.2027**; часовой пояс на прочитанной странице не установлен | **Сильный signal-processing вариант.** Electrical fault model и diagnostic robustness естественны. [CFP](https://eusipco2027.de/conference-papers/) допускает biomedical signal analytics; пять страниц. |
| [Computing in Cardiology 2027](https://www.cinc2027.org/) | 19–22.09.2027, Auckland; abstract deadline **TBA** | **Один из лучших предметных вариантов.** Для ECG error benchmark аудитория очень точная. Формат, дедлайн и публикационные правила следующего CFP надо дождаться. |
| [IEEE MLSP 2027](https://signalprocessingsociety.org/event-names/mlsp) | 13–16.09.2027, Uppsala; paper deadline не подтверждён | **Хороший ML+signals вариант.** Это самостоятельный IEEE workshop, а не короткая секция при NeurIPS. Не путать deadline заявки на проведение MLSP с подачей научной статьи. |
| [AIME 2027](https://aime27.aimedicine.info/call-for-papers/) | Abstract **31.01**, full **07.02.2027 AoE**; решение 19.04 | **Хороший альтернативный маршрут.** Medical AI, evaluation и trustworthy systems; full ≤10, short/demo ≤5 страниц вместе со ссылками, архивный LNAI. На странице есть дублирующийся блок с нарушенным порядком дат; использована согласованная верхняя таблица и явное повторение сроков. Перед подачей сверить портал. |
| [IJCNN 2027](https://ijcnn.org/2027/) | **31.01.2027**, открытие 01.12; решение 15.03 | **Хороший широкий инженерный резерв.** Neural-network robustness подходит; нужен ясный вклад сверх стандартного augmentation. Не обязательно изобретать новую архитектуру. |
| [EMBEC 2027](https://www.embec2027.eu/instructions-for-authors/) | **01.12.2026**; решение после revisions 01.02.2027 | **Хороший предметный резерв.** Springer IFMBE, Ischia 14–18 июня; short 6–11 / full 12–15+ страниц с таблицами и ссылками. Архивный выбор меняет стратегию следующей подачи. |
| [BIOSIGNALS 2027](https://biosignals.scitevents.org/ImportantDates.aspx) | Второе окно **22.10.2026 AoE**; abstract **03.12.2026** | **По теме подходит, по ближайшему времени неудобно.** Пять дней на научное исправление и текст — плохой расчёт. [Abstract не публикуется в proceedings](https://biosignals.scitevents.org/CallForPapers.aspx?y=2027). Оценивать отдельно ценность полноценной статьи и только выступления. |
| [IEEE ICHI 2027](https://ichi2027.github.io/ICHI2027/) | Early-bird **14.10.2026 прошёл**; полный календарь TBA | **Хороший вторичный вариант** по healthcare analytics. Дальнейшую возможность подачи нельзя выводить из прошедшего раннего окна. |
| [NeurIPS E&D — будущий цикл 2027](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) | **CFP/дедлайн 2027 не подтверждены** | **Самая подходящая амбициозная общая ML-цель.** Правила 2026 поддерживают evaluation audits, stress tests и отрицательные результаты без обязательной новой модели. На 2027 это ориентир, не гарантия неизменных правил. Понадобятся широкий benchmark, сильные baselines и воспроизводимый инструмент. |
| [IJCAI 2027](https://2027.ijcai.org/) | Abstract **04.01**, full **11.01.2027 AoE** | **Амбициозно; ниже CHIL по fit.** К этому сроку нужно показать общий вклад в AI evaluation или механизм устойчивости. Сам по себе один ECG-кейс с augmentation — уязвимое позиционирование. |
| [KDD 2027](https://kdd2027.kdd.org/research-track-call-for-papers/) | Первый цикл **26.07.2026 прошёл**; следующий срок здесь не подтверждён | Research / datasets-benchmarks потенциально подходят после усиления. Не назначать февральский дедлайн по памяти. [MiLeTS](https://kdd-milets.github.io/milets2026/) — более подходящая workshop-серия; издание 2027 пока не подтверждено. |
| [ICML](https://icml.cc/) | CFP 2027 не подтверждён в проверенных материалах | Возможная амбициозная цель после существенного методического/оценочного обобщения. Точный январский день не назначать. |
| [IEEE BHI](https://bhi.embs.org/2026/) | Издание/CFP 2027 не подтверждены | Тематически полезный список ожидания по biomedical informatics; ссылка ведёт на проверенное издание 2026. |
| [AMIA Annual Symposium 2027](https://amia.org/education-events/annual-symposium) | 06–10.11.2027, San Diego; CFP не подтверждён | Вторичный вариант после более сильной клинической/информатической трактовки. Это не [Amplify 2027](https://amia.org/education-events/2027-amplify-informatics-conference/summit-proposals), чей срок 03.09.2026 уже прошёл. |

Нельзя рассматривать перечисленные архивные площадки как последовательные подачи одного текста без проверки правил. Например, EMBC принимает решения позже дедлайна CHIL. Если подать substantially similar full paper в EMBC, автоматически подавать его же в CHIL 3 февраля нельзя.

## 5. Уже закрытые основные треки и слабое соответствие

| Площадка | Состояние на 17.10.2026 | Вывод |
|---|---|---|
| [NeurIPS 2026 main / E&D](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) | Full paper был 06.05.2026 | Текущий цикл закрыт. |
| [AAAI-27 main](https://aaai.org/conference/aaai/aaai-27/areas-and-topics/) | Full paper 28.07.2026 | Ноябрьский workshop deadline не открывает основной трек. |
| [WACV-27 main](https://wacv.thecvf.com/Conferences/2027/Dates) | Round 2 full paper 28.08.2026 | Ноябрьская дата — camera-ready. Кроме того, чистые ECG-сигналы менее естественны для CV-аудитории. |
| [ICLR 2027 main](https://iclr.cc/) | Full paper 25.09.2026 | Доступно только будущее workshop-окно при подтверждении конкретных workshops. |
| [AISTATS 2027 main](https://aistats.org/aistats2027/dates.html) | Full paper 06.10.2026 | Закрыт; текущая работа также не имеет центрального нового статистического вклада. |
| [ICASSP 2027 main](https://2027.ieeeicassp.org/call-for-papers/) | Full paper 23.09.2026 | Предметно хорош, но закрыт. 27 января — final paper, не новая подача. |
| [CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) | Registration 10.11, paper 16.11.2026 | Открыт по времени, **не приоритет по теме**. Перерисовка сигналов в 2D не создаёт самостоятельного vision-вклада. |
| [MIDL 2027](https://2027.midl.io/call-for-papers) | Есть main и clinical-translation tracks | Основная область — biomedical imaging; наличие clinical track не делает любую health ML работу подходящей. |
| [CHI 2027](https://chi2027.acm.org/authors/papers/) | Full paper 10.09.2026 | Нет human-computer interaction study; эта аудитория не соответствует нынешнему проекту. |

### Все ICASSP satellite workshops

Просмотрен [официальный список всех 11 workshops](https://2027.ieeeicassp.org/satellite-workshops/). Общий [workshop paper deadline — 11.11.2026](https://2027.ieeeicassp.org/call-for-satellite-workshops/), но он не является общей возможностью подать любую signal-processing работу. Ни один workshop не оказался прямым кандидатом для нынешней статьи:

- Mental healthcare и speech biomarkers посвящены другим клиническим задачам/модальностям.
- Trustworthy driving, speech/audio language modeling и voice agents — другим системам.
- Wireless/6G/ambient/distributed sensing — другим измерениям и задачам.
- Cybersecurity, quantum ML и школьная outreach-программа не соответствуют исследовательскому вопросу.

Нет причины менять задачу проекта ради общего позднего срока ICASSP workshops.

## 6. Будущие workshop-окна: что действительно известно

| Семейство | Что подтверждено | Что искать для нашей работы |
|---|---|---|
| [ICLR 2027 workshops](https://iclr.cc/Conferences/2027/CallForWorkshops) | Решения организаторам **29.11.2026**; рекомендованный paper deadline **01.02.2027**; это не обязательный срок каждого workshop | Health time series, biosignals, evaluation validity, robust ML. Предварительный сайт с «ICLR 2027» ещё не доказывает принятие workshop. |
| [AISTATS 2027 workshops](https://virtual.aistats.org/Conferences/2027/CallForWorkshops) | Решения mid-December; рекомендованные submissions late January | Возможен calibration / evaluation / distribution shift workshop. Названия пока не утверждать. 30 октября — proposal deadline организаторам. |
| ICLR health/robustness серии | [TS4H 2026](https://timeseries4health.github.io/iclr2026/index.html), [Trustworthy AI 2026](https://trustworthy-ai-workshop.github.io/iclr2026/), [CAO 2026](https://sites.google.com/view/iclr-2026-workshop-cao/home) | Хорошие тематические ориентиры; продолжение в 2027 не подтверждено. CAO сильнее подходит при добавлении мониторинга/адаптации. |
| [ICML Structured Data for Health](https://structureddata4health.github.io/) | Проверено издание 2026 | Очень хорошая аудитория для physiological time series; 2027 ждать отдельно. |
| [KDD MiLeTS](https://kdd-milets.github.io/milets2026/) | Проверено издание 2026 | Multivariate time series, measurement noise, robust classification; не объявлять издание и сроки 2027 установленными. |
| NeurIPS 2027 workshops | Список и сроки не подтверждены | Приоритет типов: evaluation → biosignals → time series → failure analysis. Не предполагать, что все 102 названия 2026 повторятся. |
| [CVPR 2027 workshops](https://cvpr.thecvf.com/Conferences/2027/Dates) | Решения по proposals **23.12.2026** | Рассматривать только workshop, явно допускающий физиологические сигналы или общую sensor reliability. Medical imaging само по себе недостаточно. |
| [IJCNN workshops / special sessions](https://ijcnn.org/2027/) | Решения proposals **15.12.2026** | Biosignals / dependable neural models; следить за индивидуальными CFP. |
| [EMBC workshops](https://embc.embs.org/2027/papers/) | Proposal deadline **24.01.2027** | Это заявки на организацию. Их нельзя записывать в календарь как срок подачи нашей статьи в готовый workshop. |

Отдельная ловушка: [ICBINB на ICLR 2026](https://sites.google.com/view/icbinb-2026) посвящён LLM limitations. Хорошее соответствие ICBINB-BIO не означает автоматического соответствия любому workshop серии ICBINB.

## 7. Реалистичный календарь подготовки

Это плановая оценка нагрузки, не измеренная длительность новых запусков. Исследование venue не включало бенчмаркинг вычислительного бюджета.

| Период | Работа | Результат, позволяющий двигаться дальше |
|---|---|---|
| 17–24 октября | Provenance mask audit, фиксация исправленного evaluator, оценка стоимости inference; полный Cao audit | Понятно, какие claims и таблицы сохраняются; есть корректный минимальный пересчёт |
| 25 октября – 7 ноября | Восстановить основные mask contrasts; chest-only Chapman; закончить M3 Chapman и парные интервалы | Один законченный результат, выдерживающий сравнение с простыми baseline |
| 8–15 ноября | Short-paper текст и рисунки, clinician/coauthor review; выбрать один workshop | Подаваемая версия без неподтверждённых выводов; подтверждён publication status |
| До 18 ноября | Внутренний срок финализации | Запас до 20 ноября на портал и формат |
| Конец ноября – декабрь | Ningbo Transformer; masking+combined; chest-only Ningbo; q_fault/calibration | Полная сравнительная программа вместо одной демонстрации уязвимости |
| Январь | Литература, внешние проверки, reproducibility, full-paper red-team собственных claims | Выбор **одного** основного archival venue |
| 3 февраля | CHIL при завершённых доказательствах | Основная рекомендованная подача |
| После решения / при смене стратегии | MLHC / CinC / MLSP / NeurIPS E&D по реально опубликованным CFP | Переподача или существенно иной вклад с учётом действующих правил |

**Минимум для ноябрьской workshop-версии:** исправленные основные сравнения; корректная novelty; честная интерпретация Huang; сильный простой baseline либо существенное ограничение claims при его отсутствии; клинически проверяемые определения меток; аккуратное указание предварительного статуса незавершённых веток. Не обязательно до ноября выполнять весь план L0–L9.

**Предпочтительно для CHIL:** две существенно разные архитектуры с clean/physics-aware сравнением; Chapman/Ningbo replication; chest-only; четырёхрежимный ResNet на Ningbo; корректные mask5/exact-mask; q_fault/calibration; ограниченный внешний HEEDB анализ и прозрачный публично воспроизводимый компонент.

**Для NeurIPS E&D 2027:** кроме перечисленного, доказать более широкую ценность протокола и инструмента; расширить условия/модели/faults без размывания физической корректности, проверить held-out reversals и документировать доступность restricted/public данных. Реальные подтверждённые wiring errors усилили бы работу, но не обязательны для честно ограниченного simulation benchmark.

## 8. Публикационная стратегия и время по UTC+5

**Самый ясный маршрут:** Deployable AI short non-archival → расширенная CHIL submission. [CHIL допускает предыдущие non-archival workshop versions](https://chil.ahli.cc/submit/call-for-papers/), но запрещает substantially similar работу на рассмотрении другого archival venue. Перед конкретной следующей подачей прочитать обе действующие политики. Making Sense и ASTAD пока не имеют подтверждённого в этом аудите non-archival статуса.

Резервные архивные варианты — **альтернативы**, а не пакет одновременных подач: EMBC / EUSIPCO / AIME / IJCNN. Workshop proceedings WACV также могут быть архивной публикацией и влиять на дальнейшую стратегию. Название «workshop» само по себе ничего не гарантирует.

Для **23:59 AoE** местное время в Asia/Qyzylorda (UTC+5) — **16:59 следующего календарного дня**:

| Срок источника | Местное время UTC+5 |
|---|---|
| 16.10.2026 23:59 AoE | 17.10.2026 16:59 |
| 20.11.2026 23:59 AoE, AAAI DAI / Making Sense | **21.11.2026 16:59** |
| 23.11.2026 11:59 UTC, Data4VFM | **23.11.2026 16:59** — не следующий день |
| 24.01.2027 23:59 AoE, EMBC | **25.01.2027 16:59** |

У CHIL, ASTAD, FABRIC, SENSE, IJCNN и EUSIPCO не присваивался точный местный час без подтверждения timezone соответствующего срока. Для планирования достаточно даты; для загрузки нужен точный portal deadline.

## 9. Границы полноты поиска

Проведён широкий целевой обзор, а не доказанная перепись всех мировых мероприятий. Полностью просмотрены опубликованные списки **NeurIPS 2026 (102), WACV 2027 (26) и ICASSP 2027 satellite workshops (11)**, затем отдельно изучены содержательные кандидаты. Не все нерелевантные сайты открывались: отбор по названию и предмету полного официального списка предшествовал подробному чтению CFP.

AAAI сообщает о **47 workshops** в [программе 2027](https://aaai.org/conference/aaai/aaai-27/aaai-27-workshop-program/), но подробная общая страница не загрузилась напрямую; индексированная версия дала общую информацию. Индивидуально проверены найденные подходящие и близкие workshops. Поэтому **полная проверка всех 47 AAAI workshops не подтверждена**. AI4TS отмечен отдельно как упоминание организатора без прочитанного CFP; Health Intelligence/W3PHIAI-27 не внесён в подтверждённый открытый список, поскольку нашлось только предыдущее издание.

Не подтверждены будущие конкретные CFP ICLR/ICML/NeurIPS 2027 workshops, сроки MLHC/CinC/MLSP 2027, следующий цикл KDD и издание BHI 2027. Эти пробелы явно оставлены открытыми. Числа acceptance rate, индексации будущих выпусков и шансы принятия не выдумывались. Автоматический мониторинг не создавался.

Приоритет следующей проверки источников: полный AAAI-27 workshop list и AI4TS; затем 29 ноября — подтверждённые ICLR workshops; после середины декабря — AISTATS/IJCNN workshops; далее CFP MLHC, CinC, MLSP и NeurIPS E&D 2027.
