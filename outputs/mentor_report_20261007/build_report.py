from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'ECG_RESEARCH_COMPLETE_REPORT.md'
OUTPUT = HERE / 'ECG_RESEARCH_COMPLETE_REPORT_2026_10_07.docx'

SOURCES = [
 ('S1','Cross dataset baseline metrics','outputs/paper_readiness_v1/CROSS_DATASET_RESULTS.csv'),
 ('S2','Chapman ingestion and multiseed decision','outputs/phase1_chapman_seed1_v4/DATA_INGESTION_AUDIT_v4.json'),
 ('S3','Ningbo ingestion and confirmation','outputs/ningbo_confirmation_v2/FINAL_NINGBO_CONFIRMATION_DECISION.md'),
 ('S4','Chapman mitigation arms','outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv'),
 ('S5','Locked Ningbo mitigation and audit','outputs/post_m2_paper_readiness_v2/FINAL_POST_M2_SCIENTIFIC_AUDIT.md'),
 ('S6','ECGFounder frozen feature comparison','outputs/ecgfounder_baseline_v2/FINAL_BASELINE_REPORT.md'),
 ('S7','HEEDB corrected external report','outputs/heedb_external_corrected_v2/FINAL_REPORT.md'),
 ('S8','HEEDB independent confirmation','outputs/heedb_independent_confirmation_v1/FINAL_REPORT.md'),
 ('S9','Ningbo published multifault comparator','outputs/published_detector_multifault_v2/FINAL_MULTIFAULT_DECISION.md'),
 ('S10','HEEDB exploratory Huang comparison','outputs/heedb_huang_exploratory_v1/FINAL_COMPARISON.md'),
 ('S11','Integrated latency benchmark','outputs/integrated_deployment_v4/FINAL_DECISION.md'),
 ('S12','Independent scientific evidence check','outputs/submission_evidence_v1/VERIFICATION_REPORT.md'),
 ('S13','Exact terminal physics implementation','src/ecg_physics.py'),
 ('S14','Early masking interpretation audit','outputs/phase1_interpretation_audit_v5/FINAL_INTERPRETATION_DECISION.md'),
 ('S15','Exploratory cable reversal development','outputs/phase1b_matched_reversal_development_v6/FINAL_PHASE1B_DECISION.md'),
 ('S16','Mitigation recovery and telemetry audit','outputs/post_m2_paper_readiness_v2/INCIDENTS_AND_TELEMETRY_AUDIT.md'),
 ('S17','HEEDB failed implementation audit','outputs/heedb_failure_audit_v1/FINAL_AUDIT.md'),
 ('S18','Manuscript readiness and remaining reviews','outputs/submission_evidence_v1/READINESS_DECISION.md'),
]

def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def mean(values):
    return sum(values)/len(values)

def fmt(x):
    return f'{float(x):.4f}'

baseline = rows('outputs/paper_readiness_v1/CROSS_DATASET_RESULTS.csv')
primary = rows('outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv')
m1 = rows('outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv')

def primary_mean(dataset, arm, key):
    return mean([float(r[key]) for r in primary if r['dataset']==dataset and r['arm']==arm])

tables={}
main=[]
for dataset in ['Chapman','Ningbo']:
    main.append([dataset,fmt(primary_mean(dataset,'cached_clean_baseline','clean_twa_auprc')),
                 fmt(primary_mean(dataset,'cached_clean_baseline','mean_five_lead_mask_twa_auprc')),
                 fmt(primary_mean(dataset,'cached_clean_baseline','ra_ll_twa_auprc')),
                 fmt(primary_mean(dataset,'physics_aware','ra_ll_twa_auprc'))])
main.append(['HEEDB independent','0.5726','Not exhaustive','0.1647','0.5683'])
tables['TABLE_MAIN']=(['Cohort','Baseline clean','Baseline masks','Baseline RA LL','Physics RA LL'],main,[1.55,1.18,1.23,1.18,1.36])
tables['TABLE_DATA']=(['Source','Source records','Analytical records','Train','Validation','Test'],[
 ['Chapman','10,247','10,234','7,163','1,021','2,050'],
 ['Ningbo','34,905','34,899','24,430','3,488','6,981'],
 ['HEEDB first cohort','Selected subset','10,000 patients','None','None','10,000'],
 ['HEEDB confirmation','New disjoint subset','10,000 patients','None','None','10,000']], [1.55,1.12,1.27,.81,.85,.90])
tables['TABLE_PUBLIC_LABEL_COUNTS']=(['Test source','TWA positive','SB positive','STach positive','Total'],[
 ['Chapman','377 (18.39%)','780 (38.05%)','314 (15.32%)','2,050'],
 ['Ningbo','1,034 (14.81%)','2,534 (36.30%)','1,138 (16.30%)','6,981']], [1.22,1.41,1.41,1.51,.95])
tables['TABLE_HEEDB_CODES']=(['Outcome','Physician codes','First 10000','New 10000'],[
 ['Core TWA','1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181','1,524','1,456'],
 ['Core SB','21 and 24','1,665','1,637'],['Core STach','23','673','615'],
 ['Strict TWA','1140','770','727'],['Strict SB','21','1,448','1,453']], [1.12,3.08,1.15,1.15])
tables['TABLE_SWAPS']=(['Displayed lead','RA LA swap','RA LL swap','LA LL swap'],[
 ['I','-I','-III','II'],['II','III','-II','I'],['III','II','-I','-III'],
 ['aVR','aVL','aVF','aVR'],['aVL','aVR','aVL','aVF'],['aVF','aVF','aVR','aVL'],
 ['V1 through V6','Unchanged','Unchanged','Unchanged']], [1.6,1.63,1.63,1.64])
tables['TABLE_TRAIN_ARMS']=(['Setup','Clean ECG','Five lead mask','Exact reversal','Where completed'],[
 ['Baseline','100%','0%','0%','Chapman and Ningbo'],
 ['Random masking','50%','50%','0%','Chapman'],
 ['Physics aware','50%','0%','50%','Chapman and Ningbo'],
 ['Combined','50%','25%','25%','Chapman']], [1.25,.9,1.05,1.0,2.3])
arm_labels={'cached_clean_baseline':'Clean baseline','masking':'Random masking','physics_aware':'Physics aware','combined':'Combined'}
m1_table=[]
for arm in ['cached_clean_baseline','masking','physics_aware','combined']:
    rr=[r for r in m1 if r['arm']==arm and r['endpoint']=='TWA']
    assert len({r['seed'] for r in rr})==3, arm
    vals=[]
    for condition in ['clean','RA_LL_reversal','mask']:
        selected=[float(r['auprc']) for r in rr if (r['condition'].startswith('mask5_') if condition=='mask' else r['condition']==condition)]
        assert len(selected)==(2376 if condition=='mask' else 3),(arm,condition,len(selected))
        vals.append(fmt(mean(selected)))
    m1_table.append([arm_labels[arm],*vals])
tables['TABLE_M1_ARMS']=(['Chapman training','Clean TWA','RA LL TWA','Mean masks TWA'],m1_table,[2.15,1.4,1.4,1.55])
baseline_table=[]
for dataset in ['Chapman','Ningbo']:
    for endpoint in ['TWA','SB','STach']:
        rr=[r for r in baseline if r['dataset']==dataset and r['endpoint']==endpoint]
        baseline_table.append([dataset,endpoint,*[fmt(mean([float(r[k]) for r in rr])) for k in ['clean_auprc','mean_792_mask_auprc','ra_ll_reversal_auprc']],
                               f"{float(rr[0]['bootstrap_mean_contrast']):.3f} [{float(rr[0]['bootstrap_lower95']):.3f}, {float(rr[0]['bootstrap_upper95']):.3f}]"])
tables['TABLE_BASELINE_ENDPOINTS']=(['Source','Label','Clean AP','Mask AP','RA LL AP','Mask minus swap 95% CI'],baseline_table,[1.0,.7,.85,.85,.85,2.25])
tables['TABLE_FOUNDER']=(['Source','Label','Founder clean','Founder RA LL','ResNet clean','ResNet RA LL'],[
 ['Chapman','TWA','0.626','0.316','0.653','0.328'],['Chapman','SB','1.000','1.000','0.999','0.994'],
 ['Chapman','STach','0.995','0.902','0.986','0.756'],['Ningbo','TWA','0.629','0.209','0.676','0.182'],
 ['Ningbo','SB','0.998','0.965','0.996','0.957'],['Ningbo','STach','0.991','0.971','0.986','0.702']], [1.07,.70,1.2,1.2,1.16,1.17])
tables['TABLE_MULTIFAULT']=(['Ningbo TWA','Baseline AP','Physics AP','Huang chain AP','Physics minus Huang CI'],[
 ['RA LA','0.2613','0.6694','0.6652','[-0.0113, +0.0204]'],['RA LL','0.1822','0.6665','0.6636','[-0.0133, +0.0187]'],
 ['LA LL','0.6338','0.6691','0.6735','[-0.0194, +0.0098]'],['Worst fault','0.1822','0.6665','0.6636','[-0.0129, +0.0169]']], [1.2,1.05,1.05,1.25,1.95])
tables['TABLE_LATENCY']=(['Integrated CPU measurement','Physics aware','Huang chain'],[
 ['Warm request median','10.59 ms','31.51 ms'],['Warm request p95','12.42 ms','34.16 ms'],
 ['Loading to ready','5.82 s','10.98 s'],['Maximum sampled working set','612.2 MiB','953.6 MiB'],
 ['Model artifact size','3.70 MiB','3.81 MiB']], [3.3,1.6,1.6])
tables['TABLE_HEEDB_MAIN']=(['Arm and condition','AUPRC','AUROC','Sensitivity','Specificity','Flagged','ECE15'],[
 ['Baseline clean','.5726','.9031','.6362','.9140','16.61%','.0768'],
 ['Physics clean','.5760','.8927','.6646','.8993','18.28%','.1120'],
 ['Baseline RA LL','.1647','.5383','.6969','.3599','64.83%','.5477'],
 ['Physics RA LL','.5683','.8901','.6866','.8908','19.33%','.1253']], [1.65,.74,.74,.95,.95,.77,.7])
tables['TABLE_HEEDB_SECONDARY']=(['Proxy outcome','Positives','Baseline clean','Physics clean','Baseline RA LL','Physics RA LL'],[
 ['Core SB','1,637','.9765','.9751','.8853','.9757'],['Core STach','615','.9520','.9502','.5953','.9485'],
 ['Strict TWA','727','.3179','.3209','.1004','.3156'],['Strict SB','1,453','.8838','.8700','.7773','.8742']], [1.15,.78,1.15,1.1,1.16,1.16])
tables['TABLE_HEEDB_HUANG']=(['RA LL TWA pipeline','AUPRC','AUROC','Sensitivity','Specificity','Brier','ECE15'],[
 ['Unprotected','.1647','.5383','.6969','.3599','.4917','.5477'],
 ['Physics aware','.5683','.8901','.6866','.8908','.1195','.1253'],
 ['Huang correction','.5811','.9066','.6548','.9117','.0985','.0769']], [1.6,.76,.76,.95,.95,.74,.74])

code_map=[
 ('src/ecg_physics.py','Common terminal permutation and twelve-lead derivation; source of all valid cable transforms.'),
 ('tests/test_ecg_physics.py','Tests electrical identities and exact transformations using synthetic inputs.'),
 ('src/audit_challenge2021_headers.py','Audits public source headers and labels before analysis.'),
 ('src/chapman_stage23.py and src/chapman_v4.py','Public-data ingestion, duplicate handling and canonical split preparation.'),
 ('src/chapman_phase1.py and src/chapman_train_v4.py','Small ResNet definition, raw loader, normalization, loss and initial training.'),
 ('src/chapman_evaluate_v4.py and src/interpret_v5.py','Original missingness evaluation and corrected signed-contrast interpretation.'),
 ('src/matched_reversal_v6.py','Exploratory three-swap and exhaustive five-lead-mask analysis.'),
 ('src/chapman_multiseed_v7.py','Three-seed clean training, checkpoint gate, RA-LL evaluation and bootstrap.'),
 ('src/chapman_multiseed_workflow_v7.py and recovery_v7.py','Supervises and safely resumes the Chapman multiseed workflow.'),
 ('src/ningbo_acquisition_v1.py','Resumable official waveform/header acquisition and checksums.'),
 ('src/ningbo_confirmation_v1.py and v2.py','Ningbo ingestion, split, clean training and confirmation; v2 applies the explicit duplicate amendment.'),
 ('src/mitigation_m1_v1.py and finalize_v1.py','Chapman physics, masking and combined training; saved-score paired bootstrap.'),
 ('src/mitigation_m2_ningbo_v1.py and finalize_v1.py','Locked physics-aware Ningbo follow-up and final analysis.'),
 ('src/paper_readiness_v1.py and post_m2_paper_readiness_v2.py','Aggregate result audits, mechanism summaries and claim boundaries.'),
 ('src/m3_transformer.py and architecture_m3_workflow_v1.py','Additional transformer architecture branch; no completed scientific decision retained.'),
 ('src/ecgfounder_audit_v1.py and preflight_v2.py','Pretrained-weight/label audit, path resolution and missing-sample checks.'),
 ('src/freeze_ecgfounder_baseline_v1.py and v2.py','Pins the separate exploratory Founder protocols and adapters.'),
 ('src/ecgfounder_baseline_v2.py and analyze_v2.py','Frozen feature extraction, diagnostic heads, evaluation and paired bootstrap.'),
 ('src/ecgfounder_report_v2.py and job_supervisor.py','Reports and supervises the completed adapted Founder comparison.'),
 ('src/huang_detector_infer_v1.py and v2.py','Published detector loading, preprocessing and fault-class inference.'),
 ('src/published_detector_pipeline_v1.py','Predicted fault correction followed by saved clean diagnostic ResNets.'),
 ('src/published_detector_multifault_v2.py','Ningbo clean plus three-fault comparison and paired contrasts.'),
 ('src/deployment_tradeoff_v3.py and deployment_latency_*.py','Initial component latency, memory and detector-error diagnostic audit.'),
 ('src/integrated_pipeline_worker_v4.py and integrated_huang_detector_worker_v4.py','Request-response workers for complete diagnostic pipelines.'),
 ('src/integrated_deployment_v4.py','Integrated output-agreement check, warm latency and process-tree memory benchmark.'),
 ('src/heedb_readiness_audit_v1.py and waveform_*_v1.py','Restricted table audit, small readiness sample, fetch and waveform verification.'),
 ('src/freeze_heedb_external_v1.py and heedb_external_v1.py','First frozen external cohort/evaluator; contains the documented invalid implementation.'),
 ('src/heedb_failure_audit_v1.py','Checks labels, units, time axis, algebra and cached outcomes after the first failure.'),
 ('src/heedb_external_corrected_v2.py','Same-cohort correction with exact input shape and common physics library.'),
 ('src/heedb_independent_confirmation_v1.py','New patient-disjoint selection, fixed model transfer and patient bootstrap.'),
 ('src/heedb_confirmation_transfer_amendment_v1a.py','Resumes the final transfer object and verifies completeness without changing science.'),
 ('src/heedb_huang_adapter_v1.py and detector_worker_v1.py','HEEDB-compatible published detector adapter and isolated detector runtime.'),
 ('src/heedb_huang_exploratory_v1.py','Same-patient exploratory correction-chain comparison and detector-error audit.'),
 ('src/submission_evidence_audit_v1.py','Independent arithmetic, protocol, cohort and result verification.'),
 ('src/submission_figures_v1.py and submission_package_check_v1.py','Aggregate figures and deterministic source/value/hash checks.'),
]
tables['TABLE_CODE_MAP']=(['Program or program family','Role in the research'],[list(x) for x in code_map],[2.5,4.0])
seed_table=[]
for r in primary:
    seed_table.append([r['dataset'],'Baseline' if r['arm']=='cached_clean_baseline' else 'Physics',r['seed'],
                       fmt(r['clean_twa_auprc']),fmt(r['ra_ll_twa_auprc']),fmt(r['mean_five_lead_mask_twa_auprc'])])
tables['TABLE_PRIMARY_SEEDS']=(['Source','Arm','Seed','Clean AP','RA LL AP','Mask AP'],seed_table,[1.05,1.1,1.0,1.12,1.12,1.11])
stage_rows=[
 ['Phase 0 feasibility','Amended design','Patient-ID requirement blocked original source design.'],
 ['Chapman v3 ingestion','Stopped','Exact duplicate exports identified before training.'],
 ['Chapman v4 and v5','Mixed or negative','V1-V4 direction did not support hypothesis; signed gate and equivalent masks audited.'],
 ['Reversal development v6','Exploratory completed','Three swaps characterized; RA-LL selected before Ningbo confirmation.'],
 ['Chapman v7','Exploratory completed','Three-seed RA-LL gap with exhaustive masks.'],
 ['Ningbo v1 ingestion','Stopped','Three metadata-conflicting duplicate waveform groups.'],
 ['Ningbo v2 baseline','Completed','Six records excluded before split; reversal gap confirmed.'],
 ['Chapman M1','Completed','Four training setups evaluated; primary physics-aware gate passed.'],
 ['Ningbo M2','Completed','Locked augmentation follow-up passed; no full four-arm retraining.'],
 ['Transformer M3','Unfinished','Six Chapman validation units and condition caches retained; no final bootstrap/decision.'],
 ['ECGFounder v1 to v2','Exploratory completed','Path/NaN adapter resolved; frozen-feature heads evaluated.'],
 ['Huang Ningbo v1 and v2','Completed comparator','Predicted correction and three faults; diagnostic difference unresolved.'],
 ['Deployment v3 to v4','Completed benchmark','Integrated end-to-end timing replaced component sums.'],
 ['HEEDB readiness','Completed','Tables and 23 excluded engineering ECGs audited.'],
 ['HEEDB external v1','Invalid intended test','Wrong temporal length and aVR/aVF transform; artifacts preserved.'],
 ['HEEDB corrected v2','Post-hoc correction','Same 10,000 patients; strong ranking recovered.'],
 ['HEEDB new confirmation','Completed confirmation','Another 10,000 patients; fixed pipeline result reproduced.'],
 ['HEEDB Huang comparison','Exploratory completed','Planned after new-cohort results; serious comparator, no established AP superiority.'],
 ['Submission evidence','Checked 26 September','Metrics, intervals, cohort separation and three aggregate figures verified.'],
]
tables['TABLE_STAGES']=(['Scientific branch','Retained status','Meaning'],stage_rows,[1.75,1.25,3.5])
tables['TABLE_SOURCES']=(['ID','Evidence','Project relative path'],[list(x) for x in SOURCES],[.45,2.0,4.05])

def set_run_font(run, size=None, bold=None):
    run.font.name='Calibri'
    run.font.color.rgb=RGBColor(0,0,0)
    if size: run.font.size=Pt(size)
    if bold is not None: run.bold=bold

def hyperlink(paragraph,url,label=None):
    rid=paragraph.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    h=OxmlElement('w:hyperlink');h.set(qn('r:id'),rid)
    run=OxmlElement('w:r');props=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'154C79');props.append(color)
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'single');props.append(underline)
    run.append(props);t=OxmlElement('w:t');t.text=label or url;run.append(t);h.append(run);paragraph._p.append(h)

def add_text(paragraph,text):
    # Keep URLs clickable in the shareable Word document and its PDF export.
    for part in re.split(r'(https?://[^\s]+)',text):
        if part.startswith(('https://','http://')):
            hyperlink(paragraph,part.rstrip('.,;'),part.rstrip('.,;'))
        elif part:
            paragraph.add_run(part)

def add_table(headers,data,widths):
    assert abs(sum(widths)-6.5)<.02
    table=doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
    for i,w in enumerate(widths):
        table.columns[i].width=Inches(w)
    for i,h in enumerate(headers):table.rows[0].cells[i].text=h
    for values in data:
        assert len(values)==len(headers)
        cells=table.add_row().cells
        for i,v in enumerate(values):cells[i].text=str(v)
    borders=OxmlElement('w:tblBorders')
    for tag in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+tag);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    table._tbl.tblPr.append(borders)
    for ri,row in enumerate(table.rows):
        trpr=row._tr.get_or_add_trPr()
        if ri==0:
            repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
        else:
            no_split=OxmlElement('w:cantSplit');trpr.append(no_split)
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr();margins=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                m=OxmlElement('w:'+side);m.set(qn('w:w'),'90' if side in ['top','bottom'] else '100');m.set(qn('w:type'),'dxa');margins.append(m)
            tcpr.append(margins)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'233D50' if ri==0 else ('F3F6F8' if ri%2==0 else 'FFFFFF'));tcpr.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.04
                p.paragraph_format.keep_with_next=(ri==0 or (len(data)<=8 and ri<len(table.rows)-1))
                # Numeric columns centered; narrative columns left aligned.
                numeric=ci>0 and widths[ci]<=1.65 and not (headers[ci] in ['Label','Arm','Seed','Retained status'])
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if numeric else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    set_run_font(run,9.5,ri==0)
                    if ri==0:run.font.color.rgb=RGBColor(255,255,255)
    after=doc.add_paragraph();after.paragraph_format.space_after=Pt(2);after.paragraph_format.space_before=Pt(0);after.paragraph_format.line_spacing=.4
    after.add_run().font.size=Pt(3)

doc=Document();section=doc.sections[0]
section.page_width=Inches(8.5);section.page_height=Inches(11)
section.top_margin=Inches(.7);section.bottom_margin=Inches(.7);section.left_margin=Inches(1);section.right_margin=Inches(1)
section.header_distance=Inches(.3);section.footer_distance=Inches(.3)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.line_spacing=1.09
doc.styles['Normal'].paragraph_format.space_after=Pt(7)
doc.styles['Title'].font.size=Pt(23)
doc.styles['Title'].paragraph_format.space_after=Pt(8)
if 'TOC 1' not in doc.styles:
    doc.styles.add_style('TOC 1', WD_STYLE_TYPE.PARAGRAPH)
doc.styles['TOC 1'].font.size=Pt(10.5)
doc.styles['TOC 1'].paragraph_format.space_after=Pt(3)
doc.styles['TOC 1'].paragraph_format.line_spacing=1.0
for name,size in [('Heading 1',15),('Heading 2',12.5)]:
    st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.space_before=Pt(14);st.paragraph_format.space_after=Pt(6);st.paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(9.5);doc.styles['Caption'].paragraph_format.space_after=Pt(9)
footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=footer.add_run('ECG research report  |  ');set_run_font(run,9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
doc.core_properties.title='ECG cable reversal robustness research report'
doc.core_properties.author='Dastan Musrepov'
doc.core_properties.subject='Complete mentor update through the September 2026 evaluation'
doc.core_properties.keywords='ECG; RA-LL; robustness; HEEDB; Huang; ECGFounder'

lines=SOURCE.read_text(encoding='utf-8').splitlines()
for line in lines:
    line=line.strip()
    if not line:continue
    if line.startswith('# '):doc.add_paragraph(line[2:],'Title')
    elif line.startswith('## '):
        if line.startswith('## 2 '):
            doc.add_page_break();doc.add_paragraph('Contents','Heading 1')
            toc=doc.add_paragraph();f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'TOC \\o "1-1" \\h \\z \\u');toc._p.append(f)
            doc.add_page_break()
        doc.add_paragraph(line[3:],'Heading 1')
    elif line.startswith('### '):doc.add_paragraph(line[4:],'Heading 2')
    elif line.startswith('{{TABLE_'):
        key=line[2:-2];add_table(*tables[key])
    elif line.startswith('{{FIGURE_'):
        num=line.split('_')[-1].strip('}')
        files={'1':'figure1_cross_dataset_ranking.png','2':'figure2_heedb_tradeoffs.png','3':'figure3_heedb_paired_contrasts.png'}
        path=ROOT/'outputs/submission_evidence_v1/figures'/files[num]
        with Image.open(path) as im:w,h=im.size
        width=min(6.5,4.6*w/h)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        picture=p.add_run().add_picture(str(path),width=Inches(width))
        picture._inline.docPr.set('descr',f'Figure {num}. Aggregate ECG robustness results described in the following caption.')
    else:
        p=doc.add_paragraph(style='Caption' if line.startswith('Figure ') else 'Normal');add_text(p,line)

doc.save(OUTPUT)

# A complete text copy expands every table, so future chats can recover the report without Word.
full=[]
for line in lines:
    if line.strip().startswith('{{TABLE_'):
        key=line.strip()[2:-2];heads,data,_=tables[key]
        full+=['| '+' | '.join(heads)+' |','| '+' | '.join(['---']*len(heads))+' |']
        full+=['| '+' | '.join(map(str,r))+' |' for r in data]
    elif line.strip().startswith('{{FIGURE_'):
        num=line.split('_')[-1].strip('}')
        full.append(f'[Aggregate Figure {num} in the Word and PDF report]')
    else:full.append(line)
(HERE/'ECG_RESEARCH_COMPLETE_REPORT_READABLE.md').write_text('\n'.join(full)+'\n',encoding='utf-8')

extra=[
 'configs/FROZEN_MITIGATION_M1_PROTOCOL_v1.json','configs/FROZEN_MITIGATION_M2_NINGBO_PROTOCOL_v1.json',
 'configs/FROZEN_NINGBO_CONFIRMATION_PROTOCOL_v2.json','configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json',
 'configs/FROZEN_HEEDB_INDEPENDENT_CONFIRMATION_v1.json',
 'outputs/heedb_huang_exploratory_v1/FROZEN_PROTOCOL.json',
 'outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv',
 'outputs/ningbo_confirmation_v2/DATA_INGESTION_AUDIT.json',
 'outputs/submission_evidence_v1/FIGURE_DATA.csv',
 'outputs/submission_evidence_v1/FIGURE_MANIFEST.json',
 'src/chapman_phase1.py',
]
manifest=[]
for rel in list(dict.fromkeys([x[2] for x in SOURCES]+extra)):
    path=ROOT/rel
    if not path.is_file():raise FileNotFoundError(path)
    manifest.append({'path':rel,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
(HERE/'REPORT_SOURCE_INVENTORY.json').write_text(json.dumps({'report_date':'2026-10-07','scientific_results_through':'2026-09-26','new_scientific_experiments_for_report':False,'sources':manifest,'reference_urls_checked':'2026-10-07','output_docx':OUTPUT.name},indent=2),encoding='utf-8')

# Lightweight structural and numerical checks, without inference or restricted-record export.
check=Document(OUTPUT)
assert len(check.tables)==len(tables),(len(check.tables),len(tables))
assert all('{{' not in p.text for p in check.paragraphs)
assert abs(primary_mean('Ningbo','physics_aware','ra_ll_twa_auprc')-.6665206242195928)<1e-9
assert abs(primary_mean('Chapman','cached_clean_baseline','clean_twa_auprc')-.6531360643308987)<1e-9
print(json.dumps({'docx':str(OUTPUT),'tables':len(check.tables),'figures':len(check.inline_shapes),'paragraphs':len(check.paragraphs),'m1_arm_means':m1_table,'structural_and_primary_numeric_checks':'PASS'},indent=2))
