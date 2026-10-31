from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'ECG_MENTOR_UPDATE_SHORT_2026_10_07.docx'
doc=Document();s=doc.sections[0]
s.page_width=Inches(8.5);s.page_height=Inches(11)
s.top_margin=Inches(.7);s.bottom_margin=Inches(.7)
s.left_margin=Inches(.85);s.right_margin=Inches(.85)
s.footer_distance=Inches(.3)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
    st=doc.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
for node in list(doc.styles.element.iter(qn('w:pBdr'))):node.getparent().remove(node)
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.line_spacing=1.08
doc.styles['Normal'].paragraph_format.space_after=Pt(7)
doc.styles['Title'].font.size=Pt(23);doc.styles['Title'].paragraph_format.space_after=Pt(6)
for n,size in [('Heading 1',16),('Heading 2',12)]:
    st=doc.styles[n];st.font.size=Pt(size);st.font.bold=True
    st.paragraph_format.space_before=Pt(11);st.paragraph_format.space_after=Pt(5);st.paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].font.bold=False
foot=s.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=foot.add_run('Dastan Musrepov | Internal research update | ');r.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
doc.core_properties.title='ECG cable reversal research update'
doc.core_properties.author='Dastan Musrepov'

def p(text,style=None):return doc.add_paragraph(text,style)
def h(text):return p(text,'Heading 2')
def table(headers,rows,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for i,w in enumerate(widths):t.columns[i].width=Inches(w)
    for i,x in enumerate(headers):t.rows[0].cells[i].text=x
    for row in rows:
        cs=t.add_row().cells
        for i,x in enumerate(row):cs[i].text=x
    borders=OxmlElement('w:tblBorders')
    for side in ['top','bottom','left','right','insideH','insideV']:
        z=OxmlElement('w:'+side);z.set(qn('w:val'),'single');z.set(qn('w:sz'),'4');z.set(qn('w:color'),'D9D9D9');borders.append(z)
    t._tbl.tblPr.append(borders)
    for ri,row in enumerate(t.rows):
        rp=row._tr.get_or_add_trPr();rp.append(OxmlElement('w:cantSplit'))
        if not ri:rp.append(OxmlElement('w:tblHeader'))
        for ci,c in enumerate(row.cells):
            c.width=Inches(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'243E50' if ri==0 else ('F3F6F8' if ri%2==0 else 'FFFFFF'));cp.append(sh)
            margins=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                z=OxmlElement('w:'+side);z.set(qn('w:w'),'95');z.set(qn('w:type'),'dxa');margins.append(z)
            cp.append(margins)
            for par in c.paragraphs:
                par.paragraph_format.space_after=Pt(0);par.paragraph_format.line_spacing=1.05
                par.paragraph_format.keep_with_next=(ri<len(t.rows)-1)
                par.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in par.runs:
                    r.font.size=Pt(10);r.bold=(ri==0);r.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(0,0,0)
    a=p('');a.paragraph_format.space_after=Pt(1);a.paragraph_format.line_spacing=.3

def link(text,url):
    par=doc.add_paragraph()
    rel=par.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    hh=OxmlElement('w:hyperlink');hh.set(qn('r:id'),rel)
    rr=OxmlElement('w:r');pr=OxmlElement('w:rPr');co=OxmlElement('w:color');co.set(qn('w:val'),'154C79');pr.append(co);rr.append(pr)
    tt=OxmlElement('w:t');tt.text=text;rr.append(tt);hh.append(rr);par._p.append(hh)
    par.paragraph_format.space_after=Pt(4)

# Page 1: formulation, data and method.
title=p('ECG cable reversal research update','Title')
bd=OxmlElement('w:pBdr');bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'nil');bd.append(bottom);title._p.get_or_add_pPr().append(bd)
p('Dastan Musrepov | For Sarvesh and the team | 7 October 2026')
h('The question')
p('Can an ECG model handle missing channels but still fail when electrode cables are swapped? Can training with physically correct cable swaps protect its diagnosis, and how does that compare with detecting and correcting the swap first?')
p('A missing channel removes information. A cable swap leaves a complete, realistic-looking ECG but changes what several leads mean. Our main finding is that this difference matters: masking tests missed a large failure in T-wave-abnormality prediction, and reversal augmentation recovered most of the lost performance.')
h('Data and task')
table(['Dataset','Data used','Role'],[
 ['Chapman','2,050 test ECGs','Initial experiments and training comparisons'],
 ['Ningbo','6,981 test ECGs','Separate-source replication'],
 ['HEEDB Emory','Two disjoint samples of 10,000 patients','External evaluation without retraining']], [1.1,2.2,3.5])
p('Chapman and Ningbo came from the public PhysioNet Challenge 2021 release. After duplicate auditing, I used separate training, validation and test splits. HEEDB was accessed through BDSP after approval. Its physician-overread labels are related proxies rather than exact matches for the public datasets.')
p('The main endpoint is T-wave abnormality (TWA). I also tested sinus bradycardia (SB) and sinus tachycardia (STach). These are three target labels, not an evaluation of every cardiac disease.')
h('What I did')
p('I used a small 1D ResNet on 12-lead, 10-second ECGs. The public-data input was reduced from 5,000 to 1,000 samples per lead and normalized using training-set statistics. Principal ResNet comparisons used three random seeds and 1,000 paired bootstrap resamples.')
p('For the stress test, I compared clean ECGs, all 792 combinations of five masked leads, and an exact RA-LL cable swap. RA-LL means exchanging the right-arm and left-leg cables: I becomes -III, II becomes -II, III becomes -I, and aVR/aVF exchange. aVL and V1-V6 stay unchanged.')
p('On Chapman I compared clean training, random masking, physics-aware training and combined training. Physics-aware training used 50% clean ECGs and 50% ECGs with RA-LA, RA-LL or LA-LL swaps. I then locked that method for Ningbo and transferred the saved Ningbo models to HEEDB.')

# Page 2: the headline finding and independent external confirmation.
doc.add_page_break();p('Results and what they mean','Heading 1')
p('AUPRC measures how well a model ranks positive cases. It is not accuracy. All main numbers below are the mean of three seed-level metrics; the cable errors were simulated on real recorded ECGs.')
table(['TWA dataset','Baseline clean','Baseline masks','Baseline RA-LL','Physics RA-LL'],[
 ['Chapman','0.653','0.574','0.328','0.637'],
 ['Ningbo','0.676','0.514','0.182','0.667'],
 ['HEEDB new cohort','0.573','Not exhaustive','0.165','0.568']], [1.6,1.3,1.3,1.3,1.3])
p('The baseline reversal score was below the average five-lead-mask score by 24.6 AUPRC points on Chapman (95% CI 21.0-28.0) and 33.1 points on Ningbo (31.1-35.0). Training on cable swaps increased reversal AUPRC by 30.9 points on Chapman and 48.4 points on Ningbo, while clean AUPRC remained similar.')
pic=p('');pic.alignment=WD_ALIGN_PARAGRAPH.CENTER;pic.paragraph_format.keep_with_next=True
shape=pic.add_run().add_picture(str(ROOT/'outputs/submission_evidence_v1/figures/figure1_cross_dataset_ranking.png'),width=Inches(6.8))
shape._inline.docPr.set('descr','TWA AUPRC on Chapman and Ningbo for clean ECGs, average masking and exact simulated RA-LL reversal.')
p('The grey baseline falls sharply under reversal; the blue physics-aware model retains most of its clean ranking performance.','Caption')
h('What HEEDB added')
p('On a new, patient-disjoint 10,000-patient sample, the same trained models reproduced the result: reversal AUPRC improved from 0.165 to 0.568, a gain of 40.4 points (95% CI 37.9-42.6). Clean AUPRC was 0.573 versus 0.576. There was no HEEDB model fitting, calibration or threshold tuning.')
p('The improvement was mainly in ranking and fewer false positives. Under reversal, specificity rose from 36.0% to 89.1%, and the fraction flagged positive fell from 64.8% to 19.3%. Sensitivity was 69.7% versus 68.7%, so I cannot claim improved sensitivity. Clean probability calibration was also worse for the physics-aware model.')
p('SB and STach also improved under reversal on HEEDB: AUPRC 0.885 to 0.976 and 0.595 to 0.949. A stricter TWA label definition still improved ranking but reduced sensitivity. Those label-dependent tradeoffs remain part of the result.')

# Page 3: published algorithms and operational tradeoff.
doc.add_page_break();p('Comparison with published approaches','Heading 1')
h('ECGFounder')
p('I used ECGFounder\'s released pretrained weights with a frozen encoder and diagnostic heads trained on our public training splits. Its output vocabulary has no exact TWA match, so I did not treat a similarly named output as the same diagnosis.')
table(['ECGFounder TWA','Clean AUPRC','RA-LL AUPRC'],[
 ['Chapman','0.626','0.316'],['Ningbo','0.629','0.209']], [2.8,2.0,2.0])
p('This adapted foundation-model pipeline also lost substantial TWA ranking under reversal. STach was more stable than in our small ResNet, so vulnerability depends on the task and model. This was an exploratory comparison on previously used cohorts; I did not train a physics-aware ECGFounder or reproduce its best full-finetuning setup.')
h('Huang detector followed by correction and diagnosis')
p('Huang et al. published a cable/lead-error detector. I used its unchanged weights to predict the error, correct the ECG according to that prediction, and then run our clean ResNet. The correction did not use the known simulated fault.')
table(['RA-LL TWA comparison','Physics aware','Huang correction chain'],[
 ['Ningbo AUPRC','0.667','0.664'],
 ['HEEDB AUPRC','0.568','0.581'],
 ['HEEDB sensitivity','68.7%','65.5%'],
 ['HEEDB specificity','89.1%','91.2%']], [3.0,1.7,2.1])
p('The HEEDB AUPRC difference was -0.013 for our method relative to Huang (95% CI -0.028 to +0.002). We did not establish superiority or equivalence. Huang had better AUROC, specificity and calibration; our model had higher sensitivity at its own fixed threshold. The HEEDB Huang comparison was planned after viewing the external results, so it is exploratory.')
p('On HEEDB, the detector made 299 false corrections of clean ECGs and 81 incorrect calls on reversed ECGs, out of 10,000 in each condition. These mistakes can affect diagnosis. Our model avoids a separate correction stage but does not tell the operator which cable was wrong.')
h('Practical difference')
p('In a controlled integrated Ningbo CPU benchmark, our single-model pipeline took 10.59 ms per warm request versus 31.51 ms for the Huang chain: 2.97 times faster, with about 36% lower measured memory. This applies to our implementation on one machine; it is not a clinical time-saving or financial estimate.')

# Page 4: history, limits, next actions and a small reading/code list.
doc.add_page_break();p('Current interpretation and next steps','Heading 1')
h('What changed during the project')
p('The first structured-missingness idea, removing V1-V4, did not support the intended hypothesis after the signed comparison was audited. We moved to exact cable reversals. Duplicate records were handled under documented amendments before splitting.')
p('The first HEEDB run had two implementation errors: 5,000 samples were passed to a model trained on 1,000, and aVR/aVF reversal equations were incorrect. I preserved that run, corrected the processing, and then confirmed the finding on another 10,000 patients. A later audit verified the final metrics, bootstrap intervals, cohort separation and figure values.')
p('Random masking alone improved masking robustness on Chapman but did not repair RA-LL (AUPRC 0.280). Combined training protected both masking and reversal (0.631 for each), but has no completed Ningbo follow-up. The additional Transformer branch has no final completed statistical report and is not counted as confirmed architecture replication.')
h('The contribution and its limits')
p('The contribution is the matched masking-versus-physical-fault benchmark, evidence that reversal augmentation repairs the observed ranking failure, and a comparison with a released detector-correction baseline. The electrical equations are established physics; our work does not introduce a new universal ECG architecture.')
p('We used actual clinical recordings with simulated cable faults. We have not measured verified real miswirings, patient outcomes or money saved. HEEDB labels need clinical review, and public-source patient identity is less certain than HEEDB\'s patient-linked selection. Our results support a focused robustness study, not an overall SOTA-superiority claim.')
h('Next steps')
p('My next step is to write the paper around this bounded finding, have the HEEDB label mapping reviewed by a clinician, and finalize the figures, citations and coauthor review for a suitable workshop. Real-fault data or another completed model-family test would strengthen broader claims, but are not results we already have.')
h('Main papers and data')
link('Oh et al. 2022, CHIL: random lead masking and lead-agnostic learning','https://proceedings.mlr.press/v174/oh22a.html')
link('Huang et al. 2024: published lead-misplacement detector','https://pubmed.ncbi.nlm.nih.gov/38663434/')
link('Li et al. 2025, NEJM AI: ECGFounder','https://pubmed.ncbi.nlm.nih.gov/40771651/')
link('Exact electrical simulation of electrode reversals, Sensors 2019','https://pmc.ncbi.nlm.nih.gov/articles/PMC6651562/')
link('PhysioNet Challenge 2021: Chapman and Ningbo source','https://physionet.org/content/challenge-2021/1.0.3/')
link('HEEDB version 5.0: external data and physician-overread labels','https://www.bdsp.io/content/heedb/5.0/')
h('Code to show')
p('In src: ecg_physics.py implements swaps; chapman_phase1.py defines the ResNet; mitigation_m1_v1.py implements augmentation; heedb_external_corrected_v2.py and heedb_independent_confirmation_v1.py evaluate transfer; published_detector_multifault_v2.py and heedb_huang_exploratory_v1.py implement the comparator.')

doc.save(OUT)
check=Document(OUT)
assert len(check.tables)==4 and len(check.inline_shapes)==1
print('Created short report:',OUT)
