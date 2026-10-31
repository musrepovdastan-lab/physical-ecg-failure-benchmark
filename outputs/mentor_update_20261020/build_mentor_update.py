from pathlib import Path
import csv
import hashlib
import json
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'ECG_Research_Update_After_Mentor_Feedback_2026_10_20.docx'
DOC = Document()
S = DOC.sections[0]
S.page_width = Inches(8.5)
S.page_height = Inches(11)
S.top_margin = S.bottom_margin = Inches(.65)
S.left_margin = S.right_margin = Inches(.75)
S.footer_distance = Inches(.28)
for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption'):
    style = DOC.styles[name]
    style.font.name = 'Calibri'
    style.font.color.rgb = RGBColor(0, 0, 0)
normal = DOC.styles['Normal']
normal.font.size = Pt(11)
normal.paragraph_format.line_spacing = 1.04
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.widow_control = True
DOC.styles['Title'].font.size = Pt(22)
DOC.styles['Title'].paragraph_format.space_after = Pt(5)
for name, size in [('Heading 1', 16), ('Heading 2', 12)]:
    style = DOC.styles[name]
    style.font.size = Pt(size)
    style.font.bold = True
    style.paragraph_format.space_before = Pt(9)
    style.paragraph_format.space_after = Pt(5)
    style.paragraph_format.keep_with_next = True
DOC.styles['Caption'].font.size = Pt(9.5)
DOC.styles['Caption'].font.bold = False
DOC.styles['Caption'].paragraph_format.space_after = Pt(6)
for node in list(DOC.styles.element.iter(qn('w:pBdr'))):
    node.getparent().remove(node)
footer = S.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run('ECG research update | 20 October 2026 | ')
run.font.size = Pt(9)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
DOC.core_properties.title = 'ECG reversal research after mentor feedback'
DOC.core_properties.author = 'Dastan Musrepov'
DOC.core_properties.subject = 'Completed controls, numerical findings, novelty and remaining decisions'
CONTENT = []


def p(text, style=None):
    CONTENT.append(text)
    return DOC.add_paragraph(text, style)


def h(text):
    return p(text, 'Heading 2')


def page(title):
    heading = p(title, 'Heading 1')
    heading.paragraph_format.page_break_before = True


def table(headers, rows, widths):
    assert abs(sum(widths) - 7) < .01
    t = DOC.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for i, value in enumerate(headers):
        t.rows[0].cells[i].text = str(value)
    for values in rows:
        for cell, value in zip(t.add_row().cells, values):
            cell.text = str(value)
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
        node = OxmlElement('w:' + side)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            node.set(qn('w:' + key), value)
        borders.append(node)
    t._tbl.tblPr.append(borders)
    for ri, row in enumerate(t.rows):
        prop = row._tr.get_or_add_trPr()
        prop.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            prop.append(OxmlElement('w:tblHeader'))
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp = cell._tc.get_or_add_tcPr()
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), '263F50' if ri == 0 else ('F2F5F7' if ri % 2 == 0 else 'FFFFFF'))
            cp.append(shade)
            margin = OxmlElement('w:tcMar')
            for side in ('top', 'bottom', 'left', 'right'):
                node = OxmlElement('w:' + side)
                node.set(qn('w:w'), '75')
                node.set(qn('w:type'), 'dxa')
                margin.append(node)
            cp.append(margin)
            for par in cell.paragraphs:
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                par.paragraph_format.keep_with_next = False
                par.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in par.runs:
                    r.font.size = Pt(10)
                    r.bold = ri == 0
                    r.font.color.rgb = RGBColor(255, 255, 255) if ri == 0 else RGBColor(0, 0, 0)
    CONTENT.append(' | '.join(headers))
    CONTENT.extend(' | '.join(map(str, row)) for row in rows)
    spacer = DOC.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(3)
    spacer.paragraph_format.keep_with_next = True
    spacer.add_run().font.size = Pt(3)
    return t


def read(rel):
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def reference(label, url):
    par = DOC.add_paragraph()
    par.paragraph_format.space_after = Pt(3)
    run = par.add_run(label + ' ')
    run.font.size = Pt(9)
    rel = par.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), rel)
    rr = OxmlElement('w:r')
    prop = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '175174')
    prop.append(color)
    size = OxmlElement('w:sz')
    size.set(qn('w:val'), '18')
    prop.append(size)
    rr.append(prop)
    value = OxmlElement('w:t')
    value.text = url
    rr.append(value)
    link.append(rr)
    par._p.append(link)
    CONTENT.append(label + ' ' + url)


p('ECG reversal research after mentor feedback', 'Title')
h('Main conclusion')
p('We have completed the bounded computational follow-up to your feedback. The results support a comparative study of ECG robustness: missing-lead testing can miss substantial diagnostic failure under electrode reversal, and mitigation choices involve different tradeoffs across fault types, cohorts and operating points. The evidence does not establish a new general augmentation principle, a universally superior architecture or clinical effectiveness.')
h('Starting evidence and the questions raised')
p('Before the feedback, a small ResNet trained on clean ECGs lost substantial T-wave-abnormality (TWA) ranking under a simulated right-arm/left-leg reversal (RA-LL). Training with three valid limb-electrode swaps recovered much of that loss. Four training arms existed on Chapman; Ningbo had ordinary and swap-augmented training. ECGFounder had a vulnerability evaluation, and the Transformer branch lacked a final decision.')
table(['Dataset', 'Ordinary clean', 'Ordinary RA-LL', 'p50 clean', 'p50 RA-LL'], [
    ['Chapman', '.6531', '.3278', '.6641', '.6372'],
    ['Ningbo', '.6761', '.1822', '.6869', '.6665'],
    ['HEEDB', '.5726', '.1647', '.5760', '.5683'],
], [1.32, 1.42, 1.42, 1.42, 1.42])
p('Table 1. Historical TWA AP, averaged across three fitted models. The clean and RA-LL values remain supported; affected historical mask values were subsequently replaced.', 'Caption')
p('Your feedback identified the key gaps: direct prior art, an untested chest-only explanation, incomplete affected-lead controls and Ningbo training arms, limited architecture evidence, a competitive correction pipeline, clean-data calibration costs, and a lack of evaluation in mostly clean populations. Clinical interpretation of endpoints and external proxy labels also remained unresolved.')
h('Data and interpretation')
p('Chapman and Ningbo test sets contain 2,050 and 6,981 canonical recordings. HEEDB results below use the previously evaluated 10,000-patient Emory confirmation cohort, including 1,456 core-TWA positives. Its original evaluation followed a separate earlier 10,000-patient cohort; later extensions reuse the confirmation patients. Saved Ningbo models were transferred without HEEDB fitting or threshold tuning.')
p('All faults are software simulations on recorded ECGs. TWA is primary; sinus bradycardia (SB) and sinus tachycardia (STach) are secondary. HEEDB physician-overread codes are proxy endpoints, not established equivalents of the public-data SNOMED labels. AP denotes average precision/AUPRC, not accuracy or sensitivity; higher AP/AUROC and lower Brier are preferable. Reported CIs are 95% paired bootstrap intervals conditional on fitted models; including zero does not establish equivalence.')

page('Missingness controls and simpler alternatives')
h('Corrected masks and the exact affected leads')
p('An audit found that several evaluation branches retained the chosen five leads and suppressed the other seven, reversing the intended mask. We regenerated all 792 five-lead masks for six existing Ningbo models, producing 4,752 corrected prediction sets. Clean and RA-LL predictions were verified and reused. The corrected ordinary mean-mask AP is .5863, replacing the historical .5139; affected older mask results are superseded.')
p('We then removed exactly the leads changed by RA-LL: I, II, III, aVR and aVF. Ordinary AP was .4358 for this mask versus .1822 for reversal: a gap of +.2536 (95% CI .2316 to .2778). Reversal was worse than every one of the 792 masks in each ordinary-model run. Across runs, mask 5th percentiles were .3925 to .4378 and 95th percentiles .6523 to .6608. Thus the result survives removal of the same affected leads and inspection of the mask distribution.')
h('Four training arms replicated on Ningbo')
p('We added three masking and three combined fits. Ordinary training uses clean inputs; masking uses 50% clean and 50% masked inputs; p50 uses 50% clean and 50% swaps; combined uses 50% clean, 25% masks and 25% swaps as exclusive choices.')
table(['Training', 'Clean AP', 'RA-LL AP', 'Mean 792 masks', 'Exact mask'], [
    ['Ordinary', '.6761', '.1822', '.5863', '.4358'],
    ['Masking', '.6740', '.1692', '.6461', '.6178'],
    ['Swap p50', '.6869', '.6665', '.6049', '.6260'],
    ['Combined', '.6768', '.6447', '.6420', '.6459'],
], [1.42, 1.22, 1.22, 1.72, 1.42])
p('Masking improved mean-mask AP by .0598 but reduced RA-LL AP by .0130 (CI -.0192 to -.0060). Combined gained .0371 over p50 on masks while losing .0218 on RA-LL and .0101 on clean input. Missingness robustness therefore did not substitute for reversal robustness. Secondary Ningbo RA-LL AP also improved with swap training: SB .9568 to .9956 and STach .7020 to .9843.')
h('Chest only baseline')
p('Six new ResNets used V1-V6 only. Those inputs and scores are exactly unchanged by the three tested limb swaps. Under RA-LL, the comparisons were:')
table(['Dataset', 'p50 AP', 'Chest AP', 'p50 minus chest and 95% CI'], [
    ['Chapman', '.6372', '.6503', '-.0131 [-.0282, +.0008]'],
    ['Ningbo', '.6665', '.6554', '+.0111 [-.0014, +.0231]'],
    ['HEEDB', '.5683', '.5367', '+.0316 [+.0202, +.0434]'],
], [1.25, 1.1, 1.1, 3.55])
p('An AP advantage over chest-only is unresolved locally and supported in the evaluated HEEDB proxy cohort. Chest-only remains a strong simple comparator. This experiment does not identify which features the augmented 12-lead model actually uses.')

page('Augmentation dose and architecture evidence')
h('How much swap augmentation is needed')
p('Twelve additional fits added p10 and p25 on both public sources, with three runs per setting. All probabilities refer to training presentations, not the percentage of patients permanently assigned a fault.')
table(['Swap probability', 'Chapman clean', 'Chapman RA-LL', 'Ningbo clean', 'Ningbo RA-LL'], [
    ['0%', '.6531', '.3278', '.6761', '.1822'],
    ['10%', '.6544', '.5886', '.6742', '.6007'],
    ['25%', '.6639', '.6183', '.6776', '.6246'],
    ['50%', '.6641', '.6372', '.6869', '.6665'],
], [1.35, 1.41, 1.42, 1.41, 1.41])
p('Even p10 provided substantial protection. However, no dose was best across ranking and probability error: Ningbo clean Brier was .0990/.0993/.0944/.1077 for p0/p10/p25/p50. On historical HEEDB mixtures with 1% RA-LL, p10/p25 AP was .5852/.5849 versus p50 .5759; at 100% RA-LL it was .5367/.5288 versus p50 .5683 and Huang .5811. The preferred dose depends on the population scenario and metric.')
p('To address unequal reversal exposure, we compared p25 with combined, both using 25% swaps. Mean-mask AP favored combined: Chapman .6312 versus .5837, a +.0474 gap (CI .0372 to .0585); Ningbo .6420 versus .6043, +.0377 (.0304 to .0449). This removes the nominal swap-frequency difference but does not isolate masking causally: clean exposure, historical implementations and random sequences also differ.')
h('Transformer result and stopping rule')
p('We completed corrected evaluation of six existing Chapman TimePatchTransformer models. RA-LL AP improved from .2183 to .6045; clean AP changed from .6369 to .6293. All three runs gained under reversal. One run nevertheless lost .1126 mean-mask AP (.5594 to .4469), exceeding the prespecified .04 retention limit. The overall M3 gate failed, so conditional Ningbo/HEEDB follow-up was not run. This is a completed mixed result, not evidence that Transformers generally fail.')
h('Mitigation on frozen ECGFounder features')
p('We trained 12 linear heads over one frozen official 1024-dimensional encoder: clean versus swap training, two sources and three runs. This tests the intervention on another representation without full backbone fine-tuning.')
table(['Dataset', 'Clean head clean', 'Swap head clean', 'Clean head RA-LL', 'Swap head RA-LL'], [
    ['Chapman', '.6557', '.6536', '.3354', '.5189'],
    ['Ningbo', '.6335', '.6219', '.2474', '.4598'],
], [1.08, 1.48, 1.48, 1.48, 1.48])
p('RA-LL gains were +.1834 (CI .1414 to .2235) and +.2125 (.1909 to .2349). Ningbo clean AP fell .0117 (CI -.0222 to -.0007). Mean changes over the fixed 32-mask panel were +.0088 and -.0027, both unresolved. Founder supports transfer beyond ResNet, with a clean-data cost; it does not cancel the Transformer result. Its pretraining used HEEDB, and possible cohort overlap was not excluded.')

page('Unseen compositions and practical tradeoffs')
h('Two cycles absent from training')
p('Existing models were tested on two three-electrode rotations, which were absent from training as complete transformations. Both remain compositions within the simulated limb-permutation family.')
table(['Dataset and cycle', 'Ordinary AP', 'p50 AP', 'Combined AP'], [
    ['Chapman 120', '.2661', '.6251', '.6099'],
    ['Chapman 201', '.3904', '.6453', '.6287'],
    ['Ningbo 120', '.1510', '.6570', '.6332'],
    ['Ningbo 201', '.2259', '.6482', '.6302'],
], [2.2, 1.6, 1.6, 1.6])
p('The four p50 gains were +.3591/+.2549/+.5060/+.4223, all with intervals above zero. Founder cycle gains were +.2262/+.0663 on Chapman and +.2159/+.1417 on Ningbo. A separate identity re-derivation control on 18 existing ResNets found TWA AP changes from -.000010 to +.000145; it did not change the interpretation of the cycle gains. These results support the two held-out compositions, not arbitrary unseen faults.')
h('Correction remains competitive')
p('In the original external RA-LL evaluation, the complete Huang detector followed by predicted inverse correction and an ordinary classifier remained competitive:')
table(['Pipeline', 'AP', 'AUROC', 'Raw Brier'], [
    ['Unprotected', '.1647', '.5383', '.4917'],
    ['Swap p50', '.5683', '.8901', '.1195'],
    ['Huang correction', '.5811', '.9066', '.0985'],
], [2.35, 1.55, 1.55, 1.55])
p('The p50-minus-Huang AP difference was -.0128 (CI -.0280 to +.0019), establishing neither superiority nor equivalence. Our integrated warm CPU median was 10.59 ms versus 31.51 ms for Huang, a 2.97-fold difference in the measured Windows implementation, not a universal deployment speedup.')
h('Clean quality and sensitivity costs')
p('On the original clean HEEDB evaluation, ordinary to p50 AP was .5726 to .5760, while AUROC worsened .9031 to .8927 and Brier .0994 to .1152. Preserved AP therefore did not mean preserved overall quality. Under reversal, broad-TWA sensitivity changed 69.69% to 68.66%, while specificity improved 35.99% to 89.08%; the unprotected model flagged 64.83% of records versus 19.33% for p50. Ranking recovery mainly removed a severe false-positive problem at that threshold.')
p('For the stricter TWA proxy, AP rose .1004 to .3156 but sensitivity fell 3.39 percentage points (CI -.0662 to -.0023 on the fractional scale). We retain this unfavorable result. AP recovery cannot be described as detecting more patients.')
h('Mixture extension and implementation recovery')
p('We extended evaluation to equal proportions of RA-LL, RA-LA and LA-LL among faulty records. An audit found six HEEDB Huang predictions where correction had been applied to the clean rather than faulty signal. These were replaced and the complete input path replayed. Historical 1%-fault AP was then .5914 combined versus .5721 Huang, but at 100% faults .5639 versus .5800. The new training series on the next page tests whether these comparative conclusions persist under separated calibration roles.')

page('Separated calibration roles and population mixtures')
p('The historical validation set selected checkpoints before being divided for calibration and threshold fitting. We therefore trained nine new Ningbo models: ordinary, p50 and combined, with three runs each. Train remained 24,430 records. Validation was separated before training into 1,744 selection records, 872 calibration-A records and 872 threshold-B records; each A/B subset contained 129 TWA positives. We fitted positive-slope Platt calibration on A and set thresholds at the 90th percentile of negative scores on B, then transported them unchanged. Huang used the corresponding new ordinary classifier, its frozen detector and its own source-derived calibration and threshold.')
p('We evaluated 0%, 0.1%, 0.5%, 1%, 5%, 10% and 100% faults for RA-LL-only and equal-three-swap mixtures. The completion package contains 700 paired comparison groups with 1,000 resamples each. The table shows raw TWA AP for the equal-three scenario. Calibration and threshold fitting used Ningbo validation data only.')
latest = read('outputs/calibration_role_control_completion_v2/RESULTS_KEY_Q_TABLE.csv')
rows = []
methods = ['ordinary', 'physics_aware_p050', 'combined', 'Huang_full_pipeline']
for cohort, label in [('Ningbo_test', 'Ningbo'), ('HEEDB', 'HEEDB')]:
    for q in (0, .01, .1, 1):
        selected = {x['method']: x for x in latest if x['cohort'] == cohort and x['scenario'] == 'equal_three_swap' and float(x['q_fault']) == q}
        rows.append([f'{label} {q*100:g}%'] + [f"{float(selected[m]['AP_raw_mean']):.4f}" for m in methods])
table(['Dataset and faults', 'Ordinary', 'p50', 'Combined', 'Huang'], rows, [2.0, 1.25, 1.25, 1.25, 1.25])
p('At 10% faults, combined exceeded Huang on Ningbo by .0111 AP (CI .0005 to .0213); p50 exceeded Huang on HEEDB by .0139 (.0024 to .0240). At 100% faults, Huang had the highest mean AP in both cohorts, although a point leader does not establish every pairwise difference.')
table(['HEEDB at 10% faults', 'Sensitivity', 'False positives per 1000 ECGs', 'Brier raw to calibrated'], [
    ['p50', '67.23%', '78.3', '.1121 to .0812'],
    ['Huang', '64.65%', '75.2', '.1142 to .0822'],
], [1.75, 1.25, 2.0, 2.0])
p('Higher p50 sensitivity accompanied more false positives. Calibration improved probability error; its positive monotone map and transformed threshold preserved the operating decisions. It did not itself increase the number of detected positives.')
p('The historical HEEDB AP leader at low fault fractions was combined; the new series favors p50. This cannot be attributed solely to separated calibration: selection/calibration sample sizes, initial weights, training trajectories and selected checkpoints also changed. The ranking of mitigation recipes is therefore conditional on both the population scenario and training series.')

page('Novelty assessment and remaining decisions')
h('Relationship to prior work')
p('Cao et al. [1] already studied reversal-robust diagnosis with an attention-based multiscale model. Indexed primary methods describe original-data training and reversed test sets; Table 2 reports macro-F1 73.68% clean versus 73.17% RA-LL. These values are not comparable to our AP. We have verified relevant indexed sections and tables, but have not obtained a complete readable five-page PDF. I have emailed Cao to request the paper; no reply has arrived, and a response is uncertain.')
p('Physiological augmentation [2], lead masking [3], published misplacement detection [4] and diagnostic reversal evaluations [5] also precede this study. We should position the contribution as comparative evidence: controls removing the affected leads reveal a different failure from reversal; mitigation changes the robustness profile; chest-only and correction provide competitive alternatives; and calibration, clean-data costs and fault proportions change the practical interpretation. The targeted review does not establish that the complete design is unprecedented.')
h('Boundaries that remain')
p('The separated-role fits retain an initialization-provenance defect: fit seeds were set after model construction, so initial weights cannot be regenerated from those seeds alone. Saved checkpoints and inference are verified, but matched initialization and full training reproducibility are not established. Ningbo lacks patient identifiers; separation and bootstrap are by canonical record, not proven patient. HEEDB has been repeatedly analyzed. Intervals condition on fitted models/calibrators, exclude full training variation and have no family-wide multiplicity adjustment.')
p('Founder is a frozen-encoder head experiment, and Transformer transfer to Ningbo was stopped by its gate. Simulated faults, proxy labels and preserved threshold operating points do not establish clinical benefit or real-miswiring performance. Test detector caches existed before the final calibration lock, although reviewed fitting used source A/B only; the control is not an untouched prospective evaluation.')
h('Recommended decisions')
p('1. Agree on the empirical contribution and submission scope. A direct Cao model has not been tested. Deferring an adaptation is defensible for the narrow evaluation claim; unavailable exact reproduction alone is not a reason to omit a relevant comparator if we claim superiority over published mitigation approaches.')
p('2. Complete the Cao comparison when the full text becomes available. Obtain clinician and source-documentation review of TWA/SB/STach meanings, SNOMED to HEEDB code mappings, ambiguous codes and whether code absence is a valid negative. Clarify whether the target is preserving the intended clean-ECG interpretation or detecting underlying disease. The clinician packet is prepared; clinical answers remain outstanding.')
p('3. Keep further experiments hypothesis-driven. Leave-one-swap-out, combined noise/contact errors and confirmed recorded miswirings are optional extensions. Real miswirings would strengthen external validity but are not a prerequisite for a clearly bounded simulation study. We have completed 45 additional fits in the follow-up blocks, including 12 small Founder heads; further model count alone will not establish novelty.')
h('Selected primary references')
reference('[1] Cao et al. ICASSP 2020. Reversal No Longer Matters.', 'https://doi.org/10.1109/ICASSP40776.2020.9053307')
reference('[2] Gopal et al. 3KG. ML4H 2021.', 'https://proceedings.mlr.press/v158/gopal21a.html')
reference('[3] Oh et al. Lead-agnostic Self-supervised Learning. CHIL 2022.', 'https://proceedings.mlr.press/v174/oh22a.html')
reference('[4] Huang et al. Physiological Measurement 2024. Author code and weights.', 'https://github.com/wjcai/ECG_lead_check')
reference('[5] Lee et al. Transparent and Robust AI-Driven ECG Model for LVSD. 2025.', 'https://doi.org/10.3390/diagnostics15151837')

DOC.save(OUT)
(HERE / 'REPORT_TEXT_FOR_REVIEW.txt').write_text('\n\n'.join(CONTENT) + '\n', encoding='utf-8')
sources = [
    'outputs/mentor_report_20261007/ECG_RESEARCH_COMPLETE_REPORT_READABLE.md',
    'outputs/mask_semantics_recovery_v2/FINAL_RECOVERY_REPORT.md',
    'outputs/mask_semantics_recovery_v2/MASK_DISTRIBUTION_SUMMARY.csv',
    'outputs/ningbo_four_arm_replication_v1/FINAL_FOUR_ARM_REPORT.md',
    'outputs/chest_only_independent_recheck_v1/RESULT_SUMMARY.md',
    'outputs/augmentation_probability_recovery_v2/FINAL_PROBABILITY_STUDY_REPORT_RECOVERED_RU.md',
    'outputs/architecture_replication_m3_recovery_v2/CHAPMAN_REPORT.md',
    'outputs/ecgfounder_mitigation_v1/RESULTS_REPORT_RU.md',
    'outputs/identity_rederivation_sensitivity_v1/FINAL_REPORT_RU.md',
    'outputs/heedb_probability_extension_v1/RESULTS_REPORT_RU.md',
    'outputs/three_swap_population_mixture_recovery_v2/FINAL_RECOVERY_REPORT_RU.md',
    'outputs/calibration_role_control_completion_v2/FINAL_REPORT_RU.md',
    'outputs/calibration_role_control_completion_v2/RESULTS_KEY_Q_TABLE.csv',
    'outputs/novelty_positioning_after_controls_v1/REVIEW_NOTES_20261020.md',
    'outputs/novelty_positioning_after_controls_v1/PRIOR_WORK_MATRIX.csv',
]
inventory = {'date': '2026-10-20',
             'author_contact_update': 'User reports having emailed Cao; no reply received. Sending details and delivery were not independently verified.',
             'sources': [{'path': x, 'sha256': hashlib.sha256((ROOT / x).read_bytes()).hexdigest()} for x in sources],
             'document': {'path': OUT.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()},
             'word_count_including_tables_and_references': len(re.findall(r'\S+', '\n'.join(CONTENT)))}
(HERE / 'REPORT_SOURCE_INVENTORY.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'docx': str(OUT), 'words': inventory['word_count_including_tables_and_references'], 'planned_pages': 6}))
