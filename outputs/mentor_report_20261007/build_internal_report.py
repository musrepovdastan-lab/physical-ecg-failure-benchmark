from pathlib import Path
import csv
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.docx'
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(.65)
sec.bottom_margin = Inches(.65)
sec.left_margin = sec.right_margin = Inches(.75)
sec.footer_distance = Inches(.25)
for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Caption']:
    st = doc.styles[name]
    st.font.name = 'Calibri'
    st.font.color.rgb = RGBColor(0, 0, 0)
    pf = st.paragraph_format
    pf.left_indent = pf.right_indent = pf.first_line_indent = Inches(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.05
for node in list(doc.styles.element.iter(qn('w:pBdr'))):
    node.getparent().remove(node)
doc.styles['Normal'].font.size = Pt(11)
doc.styles['Title'].font.size = Pt(22)
doc.styles['Title'].paragraph_format.space_after = Pt(7)
for name, size in [('Heading 1', 16), ('Heading 2', 12)]:
    st = doc.styles[name]
    st.font.size = Pt(size)
    st.font.bold = True
    st.paragraph_format.space_before = Pt(8)
    st.paragraph_format.keep_with_next = True
doc.core_properties.title = 'ECG cable reversal research report'
doc.core_properties.author = 'Dastan Musrepov'
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Dastan Musrepov | Research update | ').font.size = Pt(9)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)

def p(text, style=None):
    par = doc.add_paragraph(text, style)
    par.paragraph_format.left_indent = Inches(0)
    par.paragraph_format.right_indent = Inches(0)
    par.paragraph_format.first_line_indent = Inches(0)
    return par

def h(text):
    return p(text, 'Heading 2')

def page(title):
    doc.add_page_break()
    return p(title, 'Heading 1')

def table(headers, rows, widths=None):
    widths = widths or [1] * len(headers)
    widths = [6.95 * v / sum(widths) for v in widths]
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tw = t._tbl.tblPr.find(qn('w:tblW'))
    tw.set(qn('w:w'), str(round(6.95 * 1440)))
    tw.set(qn('w:type'), 'dxa')
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
        t.rows[0].cells[i].text = headers[i]
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    borders = OxmlElement('w:tblBorders')
    for side in ['top', 'bottom', 'left', 'right', 'insideH', 'insideV']:
        e = OxmlElement('w:' + side)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            e.set(qn('w:' + key), value)
        borders.append(e)
    t._tbl.tblPr.append(borders)
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            fill = OxmlElement('w:shd')
            fill.set(qn('w:fill'), '243E50' if ri == 0 else ('F3F6F8' if ri % 2 == 0 else 'FFFFFF'))
            tcpr.append(fill)
            mar = OxmlElement('w:tcMar')
            for side in ['top', 'bottom', 'left', 'right']:
                e = OxmlElement('w:' + side)
                e.set(qn('w:w'), '85')
                e.set(qn('w:type'), 'dxa')
                mar.append(e)
            tcpr.append(mar)
            for par in cell.paragraphs:
                pf = par.paragraph_format
                pf.space_after = Pt(0)
                pf.line_spacing = 1.02
                pf.left_indent = pf.right_indent = Inches(0)
                pf.keep_with_next = False
                par.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in par.runs:
                    run.font.size = Pt(10)
                    run.bold = ri == 0
                    run.font.color.rgb = RGBColor(255, 255, 255) if ri == 0 else RGBColor(0, 0, 0)
    spacer = p('')
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(3)

def reference(number, title, url, description):
    par = p(f'[{number}] ')
    rel = par.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    hl = OxmlElement('w:hyperlink')
    hl.set(qn('r:id'), rel)
    run = OxmlElement('w:r')
    prop = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '154C79')
    prop.append(color)
    run.append(prop)
    txt = OxmlElement('w:t')
    txt.text = title
    run.append(txt)
    hl.append(run)
    par._p.append(hl)
    par.add_run('. ' + description)
    return par

# 1. Formulation and data.
title = p('ECG cable reversal research report', 'Title')
pb = OxmlElement('w:pBdr')
bottom = OxmlElement('w:bottom')
bottom.set(qn('w:val'), 'nil')
pb.append(bottom)
title._p.get_or_add_pPr().append(pb)
p('Dastan Musrepov | For Sarvesh and the team | 7 October 2026')
h('The question and the current answer')
p('I tested whether an ECG classifier that appears robust to missing leads can still fail when electrode cables are swapped. I then tested whether adding electrically correct swaps during training repairs that failure, and compared it with a published detector that identifies and corrects the swap before diagnosis.')
p('The main finding is a large T-wave-abnormality ranking failure under simulated right-arm/left-leg reversal. Physics-aware training recovered most of the lost performance on Chapman, Ningbo and an external HEEDB sample. A published detector-correction pipeline also worked well. Our method did not establish better diagnostic performance than that pipeline, but was faster in our controlled CPU implementation.')
h('Why this is a different problem from missing leads')
p('An ECG records the heart\'s electrical activity. Ten body electrodes produce twelve displayed leads. A missing lead removes evidence; a cable swap produces complete signals with different lead meanings. Here, I simulate limb-cable swaps, not electrodes physically displaced on the chest. The electrical transformations come from established ECG physics [1].')
p('A useful research outcome is a more realistic robustness benchmark and a tested training intervention. We have not measured clinical outcomes, staff time saved or money saved. Our tests use actual recorded ECGs with faults introduced in software.')
h('Datasets and populations')
table(['Source', 'After audit', 'Train', 'Validation', 'Test'], [
    ['Chapman', '10,234', '7,163', '1,021', '2,050'],
    ['Ningbo', '34,899', '24,430', '3,488', '6,981'],
    ['HEEDB first sample', '10,000 patients', 'None', 'None', '10,000'],
    ['HEEDB new sample', '10,000 patients', 'None', 'None', '10,000'],
], [2, 1.5, 1, 1.15, 1])
p('Chapman and Ningbo came from PhysioNet Challenge 2021 [2,3]. The source contained 10,247 and 34,905 records; duplicate auditing removed or canonicalized 13 and 6 respectively before splitting. Public-source inference uses canonical records because independent patient IDs were not available. HEEDB [4] used one recording per identified patient; the two samples and 23 engineering-readiness patients were disjoint.')
p('The targets were T-wave abnormality (TWA), sinus bradycardia (SB) and sinus tachycardia (STach). Chapman test positives were 377/780/314; Ningbo positives were 1,034/2,534/1,138. HEEDB physician-overread codes define related proxy labels, not guaranteed exact copies of the public SNOMED labels. I did not evaluate every heart disease.')

# 2. Processing, physics, training and uncertainty.
page('How the experiment works')
h('From files to model input')
p('The public .mat file contains numerical waveforms; .hea provides lead names, sampling frequency and diagnoses. I checked lead order, audited duplicates and split the records before computing normalization. A 10-second ECG at 500 Hz has 5,000 samples per lead. The small ResNet uses every fifth sample, giving a 12 by 1,000 input at 100 Hz. This is decimation, not a claim of lossless or anti-aliased resampling.')
p('All faults are applied before training-set normalization. Masking replaces an entire lead with its training mean, so it becomes zero after normalization. Swaps instead rearrange and negate the original signals. HEEDB waveforms were converted through an audited unit, sampling and lead-order adapter to the same ResNet input. ECGFounder and Huang use their own published preprocessing, described below.')
h('The exact cable transformations')
table(['Lead', 'RA-LA', 'RA-LL', 'LA-LL'], [
    ['I', '-I', '-III', 'II'], ['II', 'III', '-II', 'I'], ['III', 'II', '-I', '-III'],
    ['aVR', 'aVL', 'aVF', 'aVR'], ['aVL', 'aVR', 'aVL', 'aVF'], ['aVF', 'aVF', 'aVR', 'aVL'],
    ['V1 to V6', 'Unchanged', 'Unchanged', 'Unchanged'],
], [1.7, 1, 1, 1])
p('RA-LL changes five displayed leads. I therefore compared it with all 792 ways of masking five of twelve leads. This matches the number of affected model inputs, not the amount of information or fault severity. The unchanged chest leads follow from preservation of the average limb potential for these three swaps. Tests checked the transformations and lead identities.')
h('The model and four training setups')
p('The small 1D ResNet has four residual blocks with 32/64/128/256 channels and three outputs. Training used weighted multilabel binary cross-entropy, AdamW (learning rate 0.001, weight decay 0.0001), batch size 64, at most 30 epochs, and early stopping after five epochs without improved validation TWA AUPRC.')
table(['Training arm', 'Clean', 'Mask five leads', 'Random exact swap'], [
    ['Baseline', '100%', '0%', '0%'], ['Masking', '50%', '50%', '0%'],
    ['Physics aware', '50%', '0%', '50%'], ['Combined', '50%', '25%', '25%'],
], [2.2, 1, 1.4, 1.6])
p('The swap is uniformly selected from RA-LA, RA-LL and LA-LL. Combined training does not mask and swap the same example. All four arms were completed on Chapman; only baseline and the locked physics-aware follow-up were completed on Ningbo. HEEDB reused Ningbo checkpoints without retraining.')
h('What the scores mean')
p('AUPRC (computed as average precision) measures positive-case ranking, not accuracy; prevalence affects it. AUROC measures discrimination, sensitivity counts detected positives, and specificity counts correctly rejected negatives. Thresholds came from clean validation negatives, targeting approximately 90% specificity, and stayed fixed. Brier and ECE15 measure probability calibration; lower is better.')

# 3. Public results and mitigation.
page('Results on Chapman and Ningbo')
p('Principal ResNet comparisons used three seeds and 1,000 synchronized paired bootstrap resamples. We averaged seed-level metrics, not predictions. Public analyses resampled canonical records; HEEDB resampled patients. An interval including zero does not prove equivalence. Secondary intervals were not corrected for all multiple comparisons.')
h('The clean trained model under faults')
table(['Source and label', 'Clean AP', 'Mean mask AP', 'RA-LL AP', 'Mask minus swap 95% CI'], [
    ['Chapman TWA', '.6531', '.5738', '.3278', '.246 [.210, .280]'],
    ['Chapman SB', '.9993', '.9988', '.9938', '.005 [.003, .007]'],
    ['Chapman STach', '.9856', '.9833', '.7557', '.227 [.187, .273]'],
    ['Ningbo TWA', '.6761', '.5139', '.1822', '.331 [.311, .350]'],
    ['Ningbo SB', '.9960', '.9945', '.9568', '.038 [.032, .045]'],
    ['Ningbo STach', '.9856', '.9740', '.7020', '.271 [.250, .294]'],
], [1.8, .9, 1.1, .9, 2])
p('RA-LL was worse than average masking by 24.6 AUPRC points for Chapman TWA and 33.1 points for Ningbo TWA. All three seeds showed the direction. These are absolute AUPRC points, not percentages of additional accuracy. The mean-mask score is the average of 792 separately computed condition scores.')
p('STach also lost substantial ranking, while SB was less affected. A cable swap preserves heartbeat timing, so morphology or lead-dependent cues are plausible explanations for the STach loss. We have not identified the model\'s actual causal features through attribution or representation experiments.')
h('The four training arms on Chapman')
table(['Training arm', 'Clean TWA AP', 'RA-LL TWA AP', 'Mean mask TWA AP'], [
    ['Baseline', '.6531', '.3278', '.5738'], ['Masking', '.6525', '.2797', '.6340'],
    ['Physics aware', '.6641', '.6372', '.5679'], ['Combined', '.6586', '.6314', '.6312'],
], [1.6, 1.2, 1.2, 1.5])
p('Masking training helped missing-input tolerance but did not fix the swap. Physics-aware training repaired reversal ranking; combined training protected both conditions in these point estimates. We did not confirm the combined arm on Ningbo or establish it as the best method.')
h('The locked Ningbo follow up')
table(['Ningbo TWA', 'Baseline', 'Physics aware', 'Change'], [
    ['Clean AUPRC', '.6761', '.6869', '+.0108'],
    ['RA-LL AUPRC', '.1822', '.6665', '+.4843'],
    ['Mean mask AUPRC', '.5139', '.5375', '+.0236'],
], [2, 1, 1, 1])
p('The reversal gain was +.3094 on Chapman and +.4843 on Ningbo. The reduction in the mask-minus-reversal gap was .3145 [.2771, .3508] and .4598 [.4348, .4839], respectively. This difference of contrasts is not the same as the direct reversal gain. Both frozen mitigation gates passed.')
p('On Ningbo, the matched-gap reduction was .0393 for SB and .2785 for STach. The later three-fault evaluation found physics-aware worst-fault AP of .9954 for SB and .9843 for STach. The recovery is therefore not restricted to TWA, but it remains model- and task-dependent.')

# 4. Foundation model adaptation.
page('What ECGFounder showed')
h('A published model rather than another small ResNet')
p('ECGFounder [5] is an open-weight model pretrained on more than ten million ECGs. I tested whether the vulnerability also appears in this stronger representation pipeline. Its 150-class vocabulary has no exact TWA match, so I did not substitute NONSPECIFIC T WAVE ABNORMALITY for our target.')
p('I froze the pretrained encoder, extracted 1,024-dimensional features and trained three binary logistic-regression heads on the existing public training splits. Regularization was chosen from C = .01/.1/1 using clean validation AUPRC. This is linear probing with exact benchmark labels, not full fine-tuning or a reproduction of the authors\' best model.')
h('Exact label heads compared with the clean ResNet')
table(['Source and label', 'Founder clean', 'Founder RA-LL', 'ResNet clean', 'ResNet RA-LL'], [
    ['Chapman TWA', '.626', '.316', '.653', '.328'],
    ['Chapman SB', '1.000', '1.000', '.999', '.994'],
    ['Chapman STach', '.995', '.902', '.986', '.756'],
    ['Ningbo TWA', '.629', '.209', '.676', '.182'],
    ['Ningbo SB', '.998', '.965', '.996', '.957'],
    ['Ningbo STach', '.991', '.971', '.986', '.702'],
], [1.8, 1.2, 1.2, 1.2, 1.2])
p('TWA AUPRC loss was .311 [.256, .365] on Chapman and .420 [.391, .448] on Ningbo. ECGFounder was much more stable for STach, especially on Ningbo (.991 to .971), while the ResNet fell from .986 to .702. This shows why I cannot generalize one failure to every diagnosis or architecture.')
h('Separate direct named outputs')
p('The official sinus-bradycardia and sinus-tachycardia outputs were also evaluated without training new heads. Their names match at the name level, not necessarily at the full clinical-definition level. These results must not be mixed with the table above.')
table(['Source and output', 'Clean AUPRC', 'RA-LL AUPRC'], [
    ['Chapman SB', '1.000', '1.000'], ['Chapman STach', '.968', '.946'],
    ['Ningbo SB', '.991', '.983'], ['Ningbo STach', '.940', '.953'],
], [2, 1, 1])
h('Preprocessing and comparison limits')
p('Founder received physical millivolts, twelve canonical leads, 500 Hz and ten seconds. Its pinned preprocessing used NaN-to-zero, the swap before filtering, a 50 Hz notch, a .67-40 Hz bandpass, median baseline removal and global z-scoring. All 45,133 analytical public records resolved; missing-sample and header-path cases were audited rather than silently excluded.')
p('The test cohorts had already been examined in our earlier work, so this is exploratory. Public pretraining overlap was not independently ruled out. I did not train a physics-aware Founder, compare its strongest fine-tuning configuration or evaluate it on HEEDB. These results show vulnerability of this particular adapter, not SOTA superiority of our ResNet.')

# 5. Ningbo operational comparator and deployment.
page('Published correction baseline and runtime')
h('How the Huang comparison was constructed')
p('Huang et al. [6] released a lead-error detector. I used its unchanged weights to predict a fault, inverted the ECG according to that prediction, and ran our saved clean ResNet for diagnosis. The known simulated fault was never supplied to the deployed correction policy. A true-fault oracle was only an implementation upper bound.')
p('The detector uses leads III/aVR/V6 after its own filtering, resampling to 1,200 samples and per-lead normalization. Diagnosis uses the restored twelve leads through the original ResNet preprocessing. We compare our integration of published detector weights, not diagnostic scores reported in Huang\'s paper.')
h('All three cable swaps on Ningbo')
table(['TWA condition', 'Unprotected AP', 'Physics AP', 'Huang chain AP', 'Physics minus Huang 95% CI'], [
    ['RA-LA', '.2613', '.6694', '.6652', '[-.0113, +.0204]'],
    ['RA-LL', '.1822', '.6665', '.6636', '[-.0133, +.0187]'],
    ['LA-LL', '.6338', '.6691', '.6735', '[-.0194, +.0098]'],
    ['Worst fault', '.1822', '.6665', '.6636', '[-.0129, +.0169]'],
], [1.3, 1.1, 1.0, 1.15, 2.15])
p('Every interval includes zero: neither superiority nor equivalence is established. The point estimates do not all favor the same method. Worst-fault SB/STach AP was .9954/.9843 for physics-aware and .9960/.9780 for Huang. The detector left 98.51% of clean ECGs untouched and falsely corrected 1.49%; clean TWA AP changes were +.0108 for physics-aware and -.0007 for Huang relative to baseline.')
h('End to end CPU measurement')
table(['Measurement', 'Physics aware', 'Huang chain'], [
    ['Warm request median', '10.59 ms', '31.51 ms'],
    ['Warm request p95', '12.42 ms', '34.16 ms'],
    ['Loading to ready', '5.82 s', '10.98 s'],
    ['Maximum sampled working set', '612.2 MiB', '953.6 MiB'],
    ['Model artifact size', '3.70 MiB', '3.81 MiB'],
], [2.5, 1.1, 1.1])
p('The single-model pipeline was 2.97 times faster by warm median and used about 36% less measured process-tree memory. These were direct request-to-response measurements, not sums of separately timed components. Seven runs used a balanced 128-ECG fixture on an AMD Ryzen AI 7 350 CPU, with one thread and logical CPU 0 for both frameworks.')
p('Before timing, decisions agreed 100% with frozen caches for both pipelines and detector classes agreed 100%; maximum probability differences were .0004/.0003. The speed ratio is specific to this Python integration and hardware. It is not an optimized-production comparison, a clinical time saving or an estimate of money saved. The fixture does not estimate real fault prevalence.')

# 6. HEEDB confirmation and calibration.
page('External confirmation on HEEDB')
h('Why there were two samples')
p('The first external implementation was invalid: it passed 5,000 samples to the ResNet trained on 1,000, and used wrong aVR/aVF swap equations. A corrected protocol reused the original 10,000 patients with the intended processing. Because that correction followed observed results, I then confirmed it on another 10,000 disjoint patients with the pipeline fixed before outcomes were examined.')
table(['Core TWA', 'Corrected original 10,000', 'New disjoint 10,000'], [
    ['Positive patients', '1,524', '1,456'],
    ['Baseline clean AUPRC', '.5770', '.5726'],
    ['Physics clean AUPRC', '.5715', '.5760'],
    ['Baseline RA-LL AUPRC', '.1774', '.1647'],
    ['Physics RA-LL AUPRC', '.5664', '.5683'],
    ['RA-LL AUPRC gain', '+.3890 [.3666, .4126]', '+.4036 [.3785, .4255]'],
], [2.2, 2, 2])
p('The cohorts were not pooled. The original corrected cohort included only a 24-mask subset: mean mask AP .5094 baseline versus .5135 physics-aware. That is not an exhaustive 792-mask HEEDB test. The independent primary confirmation compared clean with exact RA-LL. No model fitting, calibration or threshold tuning used HEEDB.')
h('Ranking and decisions on the independent sample')
table(['Arm and condition', 'AUPRC', 'AUROC', 'Sensitivity', 'Specificity', 'Flagged'], [
    ['Baseline clean', '.5726', '.9031', '63.62%', '91.40%', '16.61%'],
    ['Physics clean', '.5760', '.8927', '66.46%', '89.93%', '18.28%'],
    ['Baseline RA-LL', '.1647', '.5383', '69.69%', '35.99%', '64.83%'],
    ['Physics RA-LL', '.5683', '.8901', '68.66%', '89.08%', '19.33%'],
], [2.0, .9, .9, 1.2, 1.2, 1.0])
p('All three seeds improved reversed TWA AUPRC, by +.4527, +.3718 and +.3863. The mean gain was +.4036 [.3785, .4255]. Clean AP changed by +.0034 [-.0081, +.0148]. The useful recovery is ranking and rejection of false positives: reversal specificity improved by +.5309 [.5211, .5403].')
p('Reversal sensitivity did not improve: the change was -.0103 [-.0355, +.0113]. The unprotected model flags almost two-thirds of the cohort, which explains how it can retain sensitivity while losing discrimination. AUPRC recovery cannot be described as more patients detected at the frozen threshold.')
h('Probability calibration')
table(['Condition', 'Baseline Brier / ECE15', 'Physics Brier / ECE15'], [
    ['Clean', '.0994 / .0768', '.1152 / .1120'],
    ['RA-LL', '.4917 / .5477', '.1195 / .1253'],
], [1, 2, 2])
p('Lower is better. Physics-aware training greatly improved calibration under reversal, but clean calibration was worse. Similar clean AUPRC does not mean that all aspects of clean performance were preserved. The frozen thresholds came from Ningbo, not from a matched HEEDB operating point.')

# 7. Secondary outcomes and external comparator.
page('Other outcomes and the HEEDB comparator')
h('Secondary outcomes and stricter definitions')
p('Labels use physician-overread codes only. Core TWA uses 1140/1145/1150/1151/1160/1161/1170/1171/1172/1180/1181; SB uses 21/24; STach uses 23. Strict TWA uses only 1140 and strict SB only 21. Software codes were not merged. These constructs need clinical review before claiming exact diagnostic equivalence to public labels.')
table(['Outcome and positives', 'Baseline clean', 'Physics clean', 'Baseline RA-LL', 'Physics RA-LL'], [
    ['Core SB 1,637', '.9765', '.9751', '.8853', '.9757'],
    ['Core STach 615', '.9520', '.9502', '.5953', '.9485'],
    ['Strict TWA 727', '.3179', '.3209', '.1004', '.3156'],
    ['Strict SB 1,453', '.8838', '.8700', '.7773', '.8742'],
], [1.8, 1.2, 1.2, 1.2, 1.2])
p('Reversal AUPRC gains were +.0904 [.0781, .1021] for core SB, +.3532 [.3208, .3886] for core STach, +.2152 [.1912, .2401] for strict TWA, and +.0969 [.0788, .1137] for strict SB. Strict TWA sensitivity fell from 79.05% to 75.65%, a change of -.0339 [-.0662, -.0023]. This unfavorable result must remain visible.')
h('Huang on the same independent patients')
p('This comparison was planned after viewing HEEDB confirmation outcomes, so it is exploratory. The unchanged detector predicted the correction; the same clean ResNets then diagnosed the ECG. No new patients or thresholds were selected. Here, only TWA was analyzed.')
table(['RA-LL TWA', 'Unprotected', 'Physics aware', 'Huang chain'], [
    ['AUPRC', '.1647', '.5683', '.5811'], ['AUROC', '.5383', '.8901', '.9066'],
    ['Sensitivity', '69.69%', '68.66%', '65.48%'], ['Specificity', '35.99%', '89.08%', '91.17%'],
    ['Brier', '.4917', '.1195', '.0985'], ['ECE15', '.5477', '.1253', '.0769'],
], [1.7, 1.1, 1.1, 1.1])
p('Physics minus Huang was -.0128 AUPRC [-.0280, +.0019], -.0166 AUROC [-.0204, -.0124], +.0318 sensitivity [.0189, .0457], and -.0209 specificity [-.0246, -.0172]. Huang had better calibration. Different fixed thresholds yield different tradeoffs; these results do not establish an overall winner or clinical equivalence. Huang clean AP was .5721, close to baseline .5726.')
h('Detector mistakes and downstream effects')
p('Huang correctly left 9,701/10,000 clean ECGs unchanged and classified 9,919/10,000 simulated RA-LL ECGs correctly. It made 299 false clean corrections and 81 wrong calls under reversal. In those selected error subsets, 14.5% and 28.8% of seed-record TWA decisions differed from the clean reference. These are descriptive subset results, not overall patient error rates.')
p('Both approaches repair the observed failure. Our model avoids a separate correction stage but does not identify the faulty cable. Detection offers an explicit fault label and often better calibration, but its correction can be wrong. We have not evaluated abstention, confidence-based correction or a clinically optimized hybrid policy.')

# 8. Project history, reproducibility and code.
page('Project history and what remains')
h('Completed and unfinished branches')
p('The initial V1-V4 missingness hypothesis did not survive the signed-contrast audit: an absolute-value decision had hidden the direction, and 70 nominal masks produced only 50 distinct prediction outcomes. Removing V1-V4 was less damaging than its average comparator. I retained this as mixed/negative evidence and moved to cable reversals.')
p('Chapman reversal discovery was exploratory. The single-seed TWA scores were clean .6472, RA-LA .4894, RA-LL .2753 and LA-LL .6292. RA-LL was selected before the Ningbo confirmation protocol. Chapman three-seed stress testing, Ningbo confirmation, Chapman four-arm mitigation, locked Ningbo mitigation, Founder adaptation, Huang comparisons and the integrated runtime benchmark were completed.')
p('The additional TimePatchTransformer branch retained six validation units and condition caches, but no completed bootstrap or final scientific decision. It is not confirmed architecture replication. Early failed ingestion and the invalid first HEEDB run were preserved, not silently substituted. The corrected original HEEDB sample and new disjoint confirmation have distinct inferential roles.')
h('What was checked')
p('The evidence audit on 26 September passed: protocol bindings, duplicate handling, cohort separation, cached metric arithmetic, synchronized bootstrap intervals, detector counts and aggregate figure values matched. This report reorganizes those completed results; it does not add experiments. Restricted HEEDB records, predictions and identifiers stay outside shared report artifacts.')
h('Key code for mentor review')
p('src/ecg_physics.py and tests/test_ecg_physics.py implement and test the exact swaps. src/chapman_phase1.py defines the small ResNet. src/mitigation_m1_v1.py and src/mitigation_m2_ningbo_v1.py implement augmentation and the locked follow-up. src/ecgfounder_baseline_v2.py implements frozen features and trained heads.')
p('src/published_detector_multifault_v2.py runs the Ningbo correction comparison. src/integrated_deployment_v4.py measures full request latency and memory. src/heedb_external_corrected_v2.py and src/heedb_independent_confirmation_v1.py run external transfer. src/heedb_huang_exploratory_v1.py runs the external comparator. src/submission_evidence_audit_v1.py and src/submission_package_check_v1.py verify results and figures.')
h('Contribution and limitations')
p('The contribution is the matched missingness-versus-physical-fault comparison, tested augmentation across the stated sources, and an operational comparison with released detector weights. Exact cable physics and lead-reversal augmentation are not claimed as inventions. Earlier work already tested diagnostic models under reversals [7], so we cannot claim to be first.')
p('The study does not establish robustness to measured real miswirings, chest displacement, noise, different devices or every diagnosis. Our training includes the same swap family tested later. Public patient identity, label alignment, calibration and previously examined exploratory cohorts limit broader claims. Neither an AUPRC tie nor a CPU speed advantage proves SOTA or clinical superiority.')
h('Next steps before submission')
p('Write a focused robustness paper, obtain clinical review of HEEDB code definitions, and complete reference and coauthor checks. Keep secondary results and unfavorable findings available. Choose a suitable workshop with the mentor. Real-fault data or a completed additional architecture study would support stronger claims, but are not completed results we can report now.')

# 9. References and evidence navigation.
page('Relevant papers and where to check results')
p('The following sources explain the fault physics, motivate baselines and define the novelty boundary. Only ECGFounder and the Huang detector were implemented as published-model comparisons. The other methods are literature context, not experimental baselines.')
reference(1, 'Simulating Arbitrary Electrode Reversals in Standard 12 Lead ECG, Sensors 2019', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6651562/', 'Source of the physically valid terminal transformations.')
reference(2, 'PhysioNet Computing in Cardiology Challenge 2021, release 1.0.3', 'https://physionet.org/content/challenge-2021/1.0.3/', 'Official public source used for Chapman and Ningbo.')
reference(3, 'Zheng et al. A 12 lead electrocardiogram database for arrhythmia research, Scientific Data 2020', 'https://doi.org/10.1038/s41597-020-0386-x', 'Describes the Chapman data.')
reference(4, 'Harvard Emory ECG Database version 5.0, BDSP', 'https://www.bdsp.io/content/heedb/5.0/', 'Controlled-access external ECGs and physician-overread codes.')
reference(5, 'Li et al. An Electrocardiogram Foundation Model Built on over 10 Million Recordings, NEJM AI 2025', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/', 'ECGFounder supplies the pretrained encoder; our adaptation is linear probing.')
reference(6, 'Huang et al. A lightweight deep learning approach for detecting electrocardiographic lead misplacement, Physiological Measurement 2024', 'https://pubmed.ncbi.nlm.nih.gov/38663434/', 'Closest implemented detector. We added a predicted-correction and diagnosis chain.')
reference(7, 'Transparent and Robust Artificial Intelligence Driven Electrocardiogram Model for Left Ventricular Systolic Dysfunction, 2025', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12346549/', 'Already examines a diagnostic model under simulated lead reversal. Its endpoint differs from ours.')
reference(8, 'Oh et al. Lead agnostic Self supervised Learning for Local and Global Representations of Electrocardiogram, CHIL 2022', 'https://proceedings.mlr.press/v174/oh22a.html', 'Random lead masking and arbitrary lead subsets. Our supervised masking arm is not a reproduction of their full algorithm.')
reference(9, 'Bo et al. Pretraining ECG Data with Adversarial Masking Improves Model Generalizability for Data Scarce Tasks, ML4H 2022', 'https://arxiv.org/abs/2211.07889', 'Adversarial selection of signal regions to hide during pretraining.')
reference(10, 'Raghu et al. Data Augmentation for Electrocardiograms, CHIL 2022', 'https://proceedings.mlr.press/v174/raghu22a.html', 'Task-dependent augmentation; motivates checking several diagnoses.')
reference(11, 'Zhu et al. GeoECG, Machine Learning for Healthcare 2022', 'https://proceedings.mlr.press/v182/zhu22a.html', 'Wasserstein geodesic perturbations, a different augmentation mechanism.')
reference(12, 'Rjoob et al. Reliable Deep Learning Based Detection of Misplaced Chest Electrodes, JMIR Medical Informatics 2021', 'https://medinform.jmir.org/2021/4/e25347/', 'Chest-location error detection, not the same simulator or endpoint as our limb-cable tests.')
reference(13, 'McKeen et al. ECG FM an open electrocardiogram foundation model, JAMIA Open 2025', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/', 'Potential additional baseline, not tested here; public pretraining overlap needs acknowledgment.')
h('Detailed results in the project')
p('Public primary results: outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv. Four-arm results: outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv. Founder, multifault and runtime reports are in outputs/ecgfounder_baseline_v2, published_detector_multifault_v2 and integrated_deployment_v4.')
p('HEEDB summaries are in outputs/heedb_external_corrected_v2, heedb_independent_confirmation_v1 and heedb_huang_exploratory_v1. Final evidence checks and three aggregate figures are in outputs/submission_evidence_v1. These files retain per-seed results, additional metric tables and confidence intervals not repeated row by row here.')

doc.save(OUT)

# Check representative primary numbers against the unchanged scientific CSV.
with (ROOT / 'outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv').open(encoding='utf-8-sig', newline='') as f:
    results = list(csv.DictReader(f))
for dataset, arm, key, expected in [
    ('Chapman', 'cached_clean_baseline', 'ra_ll_twa_auprc', .3278),
    ('Chapman', 'physics_aware', 'ra_ll_twa_auprc', .6372),
    ('Ningbo', 'cached_clean_baseline', 'ra_ll_twa_auprc', .1822),
    ('Ningbo', 'physics_aware', 'ra_ll_twa_auprc', .6665),
]:
    vals = [float(r[key]) for r in results if r['dataset'] == dataset and r['arm'] == arm]
    assert len(vals) == 3
    assert abs(sum(vals) / len(vals) - expected) < .00006
assert len(doc.tables) == 15, len(doc.tables)
print('Created:', OUT)
print('Tables:', len(doc.tables), '| Intended pages: 9 | References: 13')
