"""Document-only independent review; preserve the agent's original manifest/files."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = OUT / "archive_before_review_20261020"


def digest(data):
    return hashlib.sha256(data).hexdigest()


manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8-sig"))
checks = []
for section in ("inputs", "local_full_pdfs", "outputs"):
    for row in manifest[section]:
        data = (ROOT / row["path"]).read_bytes()
        checks.append({"section": section, "path": row["path"],
                       "size_ok": len(data) == row["bytes"],
                       "sha_ok": digest(data) == row["sha256"]})
assert all(c["size_ok"] and c["sha_ok"] for c in checks)
rows = list(csv.DictReader((OUT / "PRIOR_WORK_MATRIX.csv").open(encoding="utf-8-sig", newline="")))
assert len(rows) == 10
changes = []


def revise(path, transform, reason):
    original = path.read_bytes()
    revised = transform(original.decode("utf-8-sig")).replace("\r\n", "\n").encode("utf-8")
    archive = ARCHIVE / path.relative_to(ROOT)
    assert not archive.exists(), archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(original)
    path.write_bytes(revised)
    changes.append({"path": path.relative_to(ROOT).as_posix(),
                    "archive": archive.relative_to(ROOT).as_posix(),
                    "before_sha256": digest(original), "after_sha256": digest(revised),
                    "reason": reason})


def positioning(text):
    old = "Cao et al. прямо описали robustness к нескольким swap/cycle вариантам в rhythm/noise классификации с чистым обучением."
    new = "В проверенных индексированных фрагментах Cao et al. описаны rhythm/noise классификация, обучение на исходном train и проверки нескольких swap/cycle вариантов на преобразованном test. Полный текст всех страниц пока не прочитан."
    assert old in text
    text = text.replace(old, new, 1)
    old = "несколько долей/смесей ошибок и исходы на двух ранее исследованных cohort settings."
    new = "несколько долей/смесей ошибок. В общей программе использовались Chapman, Ningbo и внешний HEEDB; завершённый контроль calibration roles и population mixtures охватывает Ningbo и HEEDB. HEEDB уже исследовался ранее, поэтому это не новая нетронутая внешняя выборка."
    assert old in text
    text = text.replace(old, new, 1)
    marker = "- **ECGFounder:**"
    assert marker in text
    negative = "- **TimePatchTransformer — смешанный результат:** на Chapman защита улучшила RA–LL во всех трёх запусках, но один запуск потерял 0.1126 TWA AP на среднем по 792 маскам при заранее установленном допустимом снижении 0.04. Поэтому общий M3-критерий не пройден, Ningbo-этап по протоколу не запускался. Это ограничение переноса вмешательства с сохранением качества, а не доказательство непригодности всех Transformer. Источник: `outputs/architecture_replication_m3_recovery_v2/FINAL_M3_DECISION.md`.\n"
    return text.replace(marker, negative + marker, 1)


revise(OUT / "RESEARCH_POSITIONING_RU.md", positioning,
       "Restore negative Transformer result; qualify partial Cao evidence and distinguish three project cohorts from two-cohort role control.")


def remaining(text):
    old = "Предыдущий запрос автору и маршрут через ментора/библиотеку остаются человеческими шагами; в этой задаче повторное сообщение не отправлялось."
    new = "Ранее подготовлен черновик запроса автору; его фактическая отправка и получение ответа не подтверждены файлами проекта. Помощь ментора/библиотеки остаётся следующим маршрутом доступа. Внешние сообщения этим пакетом не отправлялись."
    assert old in text
    return text.replace(old, new, 1)


revise(OUT / "REMAINING_DECISIONS_RU.md", remaining,
       "Do not imply that a prepared author-request draft was actually sent.")


def cao(text):
    marker = "**Рекомендацию пересмотреть**"
    assert marker in text
    qualification = "Отсутствие готового кода или точной репликации само по себе не доказывает, что сравнительный baseline не нужен. Если главным тезисом станет преимущество нашего способа защиты над существующими подходами, релевантная адаптация Cao может быть необходима и без исходных весов. Сейчас решение отложить её — выбор узкого вопроса исследования, а не подтверждение достаточной новизны для конкретной конференции.\n\n"
    return text.replace(marker, qualification + marker, 1)


revise(OUT / "CAO_BASELINE_DECISION_RU.md", cao,
       "Make optional-baseline recommendation conditional on claims; inaccessible exact reproduction is not a general excuse for omitting a relevant comparison.")


def source_log(text):
    data = json.loads(text)
    data["independent_review_20261020"] = {
        "result": "Consistent with a limited empirical positioning; not a completed full-text Cao review or proof of priority.",
        "cao": {
            "official_record_reopened": True,
            "mirror_direct_open": "Web reader returned Internal Error; no complete readable PDF obtained.",
            "indexed_primary_rechecked": "Printed p953 sections3.1-3.3 and p954 Table2 returned by targeted search; training described on original train and fault sets from original test. Table2 clean73.68 and RA/LL73.17 F1 agree with report.",
            "queries": [
                "\"Reversal No Longer Matters\" Cao Shi Wu pdf",
                "\"Reversal No Longer Matters\" \"training\" \"3.2\" [domain dihana.cps.unizar.es]",
                "site:dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000951.pdf \"73.68\" \"73.17\"",
                "\"Reversal No Longer Matters\" Cao github code"
            ],
            "all_pages_readable_full_text_obtained": False,
            "code_weights": "Not verified in inspected public routes; not a claim that no implementation exists."
        },
        "other_primary_checks": [
            {"url": "https://proceedings.mlr.press/v158/gopal21a.html", "scope": "Official abstract: physiological spatial/temporal augmentation; 23-class downstream task."},
            {"url": "https://proceedings.mlr.press/v174/oh22a.html", "scope": "Official abstract: random lead masking, arbitrary lead sets; arrhythmia and identification tasks."},
            {"url": "https://github.com/wjcai/ECG_lead_check", "scope": "Author README and visible file listing: detector code, saved_model, DOI, pipeline/preprocessing description."},
            {"url": "https://www.nature.com/articles/s41467-023-39472-8", "scope": "Methods pre-filtration and augmentation: wearing-error detection and channel mask; distinct diagnostic endpoints."},
            {"url": "https://link.springer.com/article/10.1007/s13534-024-00455-3", "scope": "Official article metadata/abstract: precordial positional augmentation."},
            {"url": "https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2026.1856211/full", "scope": "Methods/Table2: 4094-subset benchmark; Transformer covered32 representative combinations, unlike exhaustive linear/CNN coverage."},
            {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12346549/", "scope": "Direct reader hit verification page; primary indexed result/MDPI indexed result supports12 reversal scenarios. Full-text review remains inherited from prior project audit, not newly repeated here."}
        ],
        "prior_pdf_reviews": "Alves and Rafiei local PDF hashes verified against manifest; not reread in full in this document-only review."
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


revise(OUT / "SOURCE_ACCESS_AND_EVIDENCE.json", source_log,
       "Record independent primary-source checks and their actual access limits.")


def status(text):
    text = text.replace("# Статус: позиционирование подготовлено, ожидает проверки",
                        "# Статус: разбор проверен с поправками; научные решения остаются открытыми", 1)
    text = text.replace("Ожидает проверки пользователем/авторами.",
                        "Независимый review выполнен: все 23 исходные manifest-записи совпали; сравнительная матрица содержит 10 работ. В итог возвращён отрицательный Transformer M3 результат, уточнены cohort scope, условность решения о Cao baseline и отсутствие подтверждения отправки запроса автору. См. `REVIEW_NOTES_20261020.md`. Это принятие разбора, не окончательная верификация новизны или достаточности для выбранной площадки.", 1)
    return text


revise(OUT / "STATUS.md", status, "Record reviewed status and remaining human decisions.")


def plan(text):
    lines = text.splitlines()
    assert lines[2].startswith("**Статус на 20 октября 2026:")
    lines[2] = "**Статус на 20 октября 2026: ограниченная вычислительная очередь завершена; пакет позиционирования проверен с документальными поправками.** См. `outputs/novelty_positioning_after_controls_v1/REVIEW_NOTES_20261020.md`. Приняты identity sensitivity, three-swap recovery v2 и calibration role control completion v2 (700 bootstrap groups). Отрицательный Transformer M3 и ограничения Founder/seed/patient/proxy/previous-test-access сохраняются. Cao остаётся PARTIAL_SOURCE_ACCESS; прямой Cao baseline не выполнялся. Рекомендация пока не запускать adaptation условна на узком сравнительном вкладе и должна быть обсуждена с ментором; она не доказывает достаточность новизны для площадки. Следующие действия: согласовать вклад/масштаб с ментором, получить полный Cao через автора/библиотеку по возможности, получить клинические ответы по endpoints/mappings. Новая автоматическая очередь обучений и написание рукописи не запланированы."
    lines.insert(4, "Разделы ниже сохраняют исходный план и его обоснования; актуальное состояние — в статусе выше и принятых review notes. Запрос автору ранее подготовлен как черновик; отправка и ответ не подтверждены.")
    return "\n".join(lines) + "\n"


revise(ROOT / "NEXT_RESEARCH_DECISIONS_20261019.md", plan,
       "Replace stale next-stage wording with reviewed status and concrete human follow-up; preserve historical plan.")

notes = """# Независимая проверка позиционирования — 20 октября 2026

Пакет принят как внутренний ограниченный разбор с указанными ниже поправками. Это не полная сверка всех страниц Cao, не доказательство приоритета и не гарантия достаточной новизны для выбранной конференции.

## Что проверено

- Все 23 записи исходного MANIFEST.json: существование, размер и SHA-256 совпали до поправок. В матрице 10 ближайших работ.
- Повторно открыты официальная запись Cao, PMLR 3KG/Oh, авторский репозиторий Huang, Nature Lai, Springer по positional augmentation и Frontiers по reduced-lead benchmark. По Cao индексированные первичные разделы3.2–3.3/таблица2 согласуются с описанием и F1 числами. Весь PDF получить не удалось; доступ остаётся PARTIAL_SOURCE_ACCESS. Подробности и границы повторной проверки — SOURCE_ACCESS_AND_EVIDENCE.json.
- Числа mixture/calibration сопоставлены с уже принятым REVIEW_NOTES_20261020.md в calibration_role_control_completion_v2; в этой задаче статистику заново не рассчитывали. Полные проверки этой статистики относятся к прежнему независимому аудиту.
- Сохранены ограничения chest-only, frozen-Founder/32 masks, simulated faults, ранее исследованного HEEDB, proxy labels и initialization-seed. Новых тренировок, inference, fitting или внешних сообщений не было.

## Исправления

1. Возвращён пропущенный Transformer M3: RA–LL benefit есть, но один запуск теряет0.1126 AP на масках при лимите0.04. Общая проверка не пройдена, Ningbo-этап не запускался. Это нельзя скрывать за положительным Founder результатом.
2. Различены три когорты всей программы и две когорты нового role-control анализа. По Cao полное чтение не приписывается индексированным фрагментам.
3. Подготовленный черновик запроса автору не назван отправленным письмом.
4. Рекомендация отложить Cao baseline признана условной. Отсутствие исходного кода не доказывает ненужность адаптации, если заявляется сравнительное преимущество защиты. Для узкого исследования компромиссов её сейчас можно отложить; достаточность этого масштаба нужно обсудить с ментором.
5. В актуальном плане убрана уже устаревшая команда снова делать positioning. Открыты полное уточнение Cao, клинические ответы и авторское решение о масштабе вклада; новая вычислительная очередь автоматически не добавлена.

## Что это означает простыми словами

Мы лучше подтвердили, как разные способы защиты ведут себя в разных условиях. Мы не установили, что придумали новый общий принцип или лучшую архитектуру. Главный кандидат на вклад — честное сравнение устойчивости к пропавшим каналам и перестановкам электродов, способов защиты и цены защиты на чистых/смешанных данных. Значимость этого сравнительного вывода не растёт автоматически от количества тестов.

Дальше разумно обсуждать этот конкретный вклад с ментором и передать врачу готовые вопросы по смыслу меток. Полный Cao желательно получить для окончательного сопоставления; ответ автора не является разрешением на исследование. Настоящие подтверждённые ошибки подключения остаются отдельным возможным усилением simulation study, а не обязательным следующим сбором данных.

Исходные документы сохранены в archive_before_review_20261020/. Исходный MANIFEST.json не переписан; изменения учтены в DOCUMENT_REVIEW_CHANGES_20261020.json. Скрипт review_positioning_20261020.py фиксирует операции этого разового review и не предназначен для повторного перезаписывания архива.
"""
(OUT / "REVIEW_NOTES_20261020.md").write_text(notes, encoding="utf-8")
(OUT / "DOCUMENT_REVIEW_CHANGES_20261020.json").write_text(
    json.dumps({"date": "2026-10-20", "changes": changes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "INDEPENDENT_REVIEW_AUDIT_20261020.json").write_text(
    json.dumps({"status": "PASS_WITH_DOCUMENT_CORRECTIONS_AND_OPEN_SCIENTIFIC_DECISIONS",
                "original_manifest_checks": checks, "matrix_rows": len(rows),
                "changes": changes, "new_training_inference_fitting": False,
                "full_cao_access": False, "baseline_run": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"original_manifest_entries": len(checks), "all_original_hashes_match": True,
                  "matrix_rows": len(rows), "document_changes": len(changes)}, ensure_ascii=True))
