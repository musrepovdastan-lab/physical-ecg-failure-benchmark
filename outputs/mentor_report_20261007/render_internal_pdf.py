from pathlib import Path
from html import escape
from docx import Document
from docx.oxml.ns import qn
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.docx'
OUT = HERE / 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.pdf'
pdfmetrics.registerFont(TTFont('Calibri', 'C:/Windows/Fonts/calibri.ttf'))
pdfmetrics.registerFont(TTFont('CalibriBold', 'C:/Windows/Fonts/calibrib.ttf'))
pdfmetrics.registerFontFamily('Calibri', normal='Calibri', bold='CalibriBold', italic='Calibri', boldItalic='CalibriBold')
base = dict(fontName='Calibri', fontSize=11, leading=12.3, spaceAfter=6, textColor=colors.black)
styles = {
    'Normal': ParagraphStyle('body', **base),
    'Title': ParagraphStyle('title', fontName='CalibriBold', fontSize=22, leading=25, spaceAfter=9),
    'Heading 1': ParagraphStyle('section', fontName='CalibriBold', fontSize=16, leading=19, spaceAfter=9, keepWithNext=True),
    'Heading 2': ParagraphStyle('subsection', fontName='CalibriBold', fontSize=12, leading=14, spaceBefore=7, spaceAfter=5, keepWithNext=True),
}
cell = ParagraphStyle('cell', fontName='Calibri', fontSize=10, leading=11, alignment=TA_CENTER)
cell_left = ParagraphStyle('cellleft', parent=cell, alignment=TA_LEFT)
cell_head = ParagraphStyle('headcell', parent=cell, fontName='CalibriBold', textColor=colors.white)
cell_head_left = ParagraphStyle('headleft', parent=cell_head, alignment=TA_LEFT)
d = Document(SOURCE)
story = []
# Remove only the paragraph blocks identified in the user's PDF comments.
excluded_prefixes = (
    'Dastan Musrepov | For Sarvesh and the team | 7 October 2026',
    'Key code for mentor review',
    'src/ecg_physics.py and tests/test_ecg_physics.py implement',
    'src/published_detector_multifault_v2.py runs',
    'Next steps before submission',
    'Write a focused robustness paper, obtain clinical review',
    'Detailed results in the project',
    'Public primary results: outputs/post_m2_paper_readiness_v2/',
    'HEEDB summaries are in outputs/heedb_external_corrected_v2,',
)
excluded_blocks = 0

def inline_xml(par):
    pieces = []
    for child in par:
        if child.tag == qn('w:r'):
            text = ''.join(child.itertext()) if False else ''.join(t.text or '' for t in child.iter(qn('w:t')))
            pieces.append(escape(text))
        elif child.tag == qn('w:hyperlink'):
            text = ''.join(t.text or '' for t in child.iter(qn('w:t')))
            rel_id = child.get(qn('r:id'))
            target = d.part.rels[rel_id].target_ref if rel_id else ''
            pieces.append(f'<link href="{escape(target, quote=True)}" color="#154C79">{escape(text)}</link>')
    return ''.join(pieces)

for block in d.element.body:
    if block.tag == qn('w:p'):
        plain_text = ''.join(t.text or '' for t in block.iter(qn('w:t')))
        if any(plain_text.startswith(prefix) for prefix in excluded_prefixes):
            excluded_blocks += 1
            continue
        if any(br.get(qn('w:type')) == 'page' for br in block.iter(qn('w:br'))):
            story.append(PageBreak())
            continue
        style_id = None
        pr = block.find(qn('w:pPr'))
        if pr is not None:
            ps = pr.find(qn('w:pStyle'))
            style_id = ps.get(qn('w:val')) if ps is not None else None
        name = d.styles[style_id].name if style_id else 'Normal'
        text = inline_xml(block)
        if text:
            story.append(Paragraph(text, styles.get(name, styles['Normal'])))
        else:
            story.append(Spacer(1, 3))
    elif block.tag == qn('w:tbl'):
        rows = []
        grid = block.find(qn('w:tblGrid'))
        widths = [int(c.get(qn('w:w'))) / 20 for c in grid]
        widths = [500.4 * w / sum(widths) for w in widths]
        for ri, tr in enumerate(block.findall(qn('w:tr'))):
            cells = []
            for ci, tc in enumerate(tr.findall(qn('w:tc'))):
                text = ' '.join(t.text or '' for t in tc.iter(qn('w:t')))
                st = cell_head_left if ri == 0 and ci == 0 else cell_head if ri == 0 else cell_left if ci == 0 else cell
                cells.append(Paragraph(escape(text), st))
            rows.append(cells)
        tbl = Table(rows, colWidths=widths, repeatRows=1, hAlign='CENTER')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#243E50')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F6F8')]),
            ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#D9D9D9')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4.2), ('RIGHTPADDING', (0, 0), (-1, -1), 4.2),
            ('TOPPADDING', (0, 0), (-1, -1), 4.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 4.2),
        ]))
        story.append(tbl)

def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Calibri', 9)
    canvas.setFillColor(colors.HexColor('#444444'))
    canvas.drawRightString(558, 22, str(document.page))
    canvas.restoreState()

pdf = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=54, leftMargin=54,
                        topMargin=46.8, bottomMargin=46.8,
                        title='ECG cable reversal research report', author='Dastan Musrepov')
assert excluded_blocks == 9, f'Expected nine selected blocks, got {excluded_blocks}'
pdf.build(story, onFirstPage=footer, onLaterPages=footer)
r = PdfReader(OUT)
print('PDF pages:', len(r.pages))
for i, pg in enumerate(r.pages, 1):
    print(i, len(pg.extract_text()), pg.extract_text().splitlines()[1:3])
