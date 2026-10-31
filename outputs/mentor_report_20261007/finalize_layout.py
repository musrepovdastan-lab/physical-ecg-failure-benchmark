from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

path=Path(__file__).resolve().parent/'ECG_RESEARCH_COMPLETE_REPORT_2026_10_07.docx'
doc=Document(path)
for root in [doc._element,doc.styles.element]:
    for border in list(root.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
title=doc.paragraphs[0]
# An explicit nil border also suppresses Word's inherited title decoration.
border=OxmlElement('w:pBdr')
for side in ['top','bottom','left','right','between','bar']:
    item=OxmlElement('w:'+side);item.set(qn('w:val'),'nil');border.append(item)
title._p.get_or_add_pPr().append(border)
for run in title.runs:run.underline=False
for p in doc.paragraphs:
    if p.text=='Contents':
        p.style=doc.styles['Normal']
        p.paragraph_format.keep_with_next=True
        p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.space_before=Pt(0)
        for run in p.runs:run.bold=True;run.font.size=Pt(15)
        break
doc.save(path)
print('Removed title decoration and the self-referencing contents entry.')
