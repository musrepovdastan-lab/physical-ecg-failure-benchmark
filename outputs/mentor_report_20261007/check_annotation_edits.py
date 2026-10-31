from pathlib import Path
from collections import Counter
import re
import runpy
import shutil
from pypdf import PdfReader
from docx import Document

here = Path(__file__).resolve().parent
target = here / 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.pdf'
backup = here / 'ECG_INTERNAL_RESEARCH_REPORT_BEFORE_ANNOTATION_EDITS.pdf'
if not backup.exists():
    shutil.copy2(target, backup)
before = PdfReader(backup)
source = Document(here / 'ECG_INTERNAL_RESEARCH_REPORT_2026_10_07.docx')
prefixes = (
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
removed = [p.text for p in source.paragraphs if any(p.text.startswith(x) for x in prefixes)]
assert len(removed) == 9
runpy.run_path(str(here / 'render_internal_pdf.py'), run_name='__main__')
after = PdfReader(target)
assert len(before.pages) == len(after.pages) == 9

def words(text):
    return Counter(re.findall(r'\S+', text))

for i, (old, new) in enumerate(zip(before.pages, after.pages), 1):
    expected = words(old.extract_text())
    expected.subtract(words('Dastan Musrepov | Research update |'))
    if i == 1:
        expected.subtract(words(removed[0]))
    if i == 8:
        for text in removed[1:6]:
            expected.subtract(words(text))
    if i == 9:
        for text in removed[6:]:
            expected.subtract(words(text))
    expected = +expected
    actual = words(new.extract_text())
    assert expected == actual, (i, expected - actual, actual - expected)
    assert new.extract_text().splitlines()[0].strip() == str(i)
text = '\n'.join(p.extract_text() for p in after.pages)
for forbidden in ['Key code for mentor review', 'Next steps before submission', 'For Sarvesh and the team', 'Research update |', 'Detailed results in the project']:
    assert forbidden not in text
print('PASS: only the nine selected blocks and repeated footer text were removed. All other page text preserved.')
