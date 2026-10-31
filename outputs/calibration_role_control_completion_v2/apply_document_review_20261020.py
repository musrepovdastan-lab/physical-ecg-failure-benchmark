"""Repair only mutable plan text and record post-audit document corrections."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
p=ROOT/'NEXT_RESEARCH_DECISIONS_20261019.md';text=p.read_text(encoding='utf-8')
body=text.split('\n---\n',1)[1].lstrip() if '\n---\n' in text and text.startswith('# ?????') else text
assert sum('\u0400'<=c<='\u04ff' for c in body)>1000 and '\ufffd' not in body,'Plan source is not intact UTF-8; preserve and investigate rather than overwrite.'
start=body.index('**Статус');end=body.index('\n\n',start)
status='**Статус на 20 октября 2026: ограниченная вычислительная очередь завершена и принята с раскрытыми ограничениями.** Identity sensitivity, three-swap recovery v2 и calibration role control completion v2 проверены. Все700 bootstrap groups готовы; 9 fits и predictions сохранены. Initialization seed задавался после создания модели; воспроизводимость свежего обучения только по seed неполная. Остальные patient/proxy/previous-test-access ограничения сохраняются. См. `outputs/calibration_role_control_completion_v2/REVIEW_NOTES_20261020.md`. Следующий этап — научное позиционирование и решение о Cao baseline по `LUNA_PROMPT_NOVELTY_POSITIONING_20261020.md`; клинические ответы по endpoints/mappings остаются открытыми. Новые обучения и рукопись автоматически не запускать.'
body=body[:start]+status+body[end:];p.write_text(body,encoding='utf-8')
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
changes=[]
for current,arch in [(OUT/'FINAL_REPORT_RU.md',OUT/'archive_before_review_20261020/FINAL_REPORT_RU.md'),(OUT/'CURRENT_PLAN_STATUS_RU.md',OUT/'archive_before_review_20261020/CURRENT_PLAN_STATUS_RU.md'),(p,OUT/'archive_before_review_20261020/ACTIVE_PLAN_BEFORE_REVIEW.md')]:
 changes.append({'path':str(current.relative_to(ROOT)).replace('\\','/'),'original_archived_path':str(arch.relative_to(ROOT)).replace('\\','/'),'original_sha256':sha(arch),'reviewed_sha256':sha(current),'scope':'Text/status correction only; predictions, draws, locks and original manifest unchanged.'})
(OUT/'DOCUMENT_REVIEW_CHANGES_20261020.json').write_text(json.dumps({'changes':changes},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'UTF8_PLAN_REPAIRED','plan_cyrillic_characters':sum('\u0400'<=c<='\u04ff' for c in body),'document_changes':len(changes)}))
