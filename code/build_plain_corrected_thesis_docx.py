"""Offline, versioned conversion of the corrected Chapters 1--4 to plain Word.

The accepted LaTeX, analytical outputs and previous Word file are preserved.
Non-numeric TeX math is rendered to local PNGs, never to Word equation fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz
from PIL import Image, ImageChops
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.text.run import Run

import build_thesis_docx as legacy

ROOT = legacy.ROOT
THESIS = legacy.THESIS
WORK = ROOT / 'analysis/maintenance/corrected_plain_docx_20260907_v1'
INPUT = WORK / 'build_input'
OUTPUT = WORK / legacy.FINAL_DOCX.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def walk(node, transform):
    if isinstance(node, list):
        return [walk(x, transform) for x in node]
    if isinstance(node, dict):
        node = {k: walk(v, transform) for k, v in node.items()}
        return transform(node)
    return node


def prepare():
    if INPUT.exists():
        raise RuntimeError('Prepared input already exists; use --build or --render.')
    WORK.mkdir(parents=True, exist_ok=True)
    before = WORK / 'before'
    before.mkdir(exist_ok=True)
    shutil.copy2(legacy.FINAL_DOCX, before / legacy.FINAL_DOCX.name)
    protected = [p for p in THESIS.rglob('*') if p.is_file() and p.suffix in {'.tex', '.bib', '.pdf', '.docx'}]
    save_json(WORK / 'source_hashes.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in protected})
    (WORK / 'user_instruction.txt').write_text(
        'Synchronize the original DOCX with the corrected thesis. All text uses the plainest formatting; formulas preferably as images.\n'
        'Scope: Chapters 1--4 and references. Offline conversion only.\n', encoding='utf-8')
    legacy.WORK, legacy.BUILD_INPUT = WORK, INPUT
    source = legacy.prepare_pandoc_input()
    corrected = 'Figure_3.1A_inventory_construction_corrected'
    figure = fitz.open(THESIS / 'figures/results' / (corrected + '.pdf'))
    figure[0].get_pixmap(dpi=300).save(INPUT / 'figures/results' / (corrected + '.png'))
    figure.close()
    # Bind Word cross-references to the successfully compiled current manuscript.
    aux = (THESIS / 'introduction_literature_methods_results_final_figures.aux').read_text(encoding='utf-8')
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{(?:\\th@\w+\s+)?([\d.]+)\}', aux))
    save_json(WORK / 'reference_bindings.json', labels)
    for p in [*INPUT.glob('chapters/*.tex'), *INPUT.glob('tables/*.tex')]:
        text = p.read_text(encoding='utf-8').replace(corrected + '.pdf', corrected + '.png')
        text = re.sub(r'\\ref\{([^}]+)\}', lambda m: labels[m[1]], text)
        # Ordinary signed numbers are text, not formulas.
        text = re.sub(r'\$(-?\d+(?:\.\d+)?)\$', lambda m: m[1].replace('-', '−'), text)
        p.write_text(text, encoding='utf-8')
    subprocess.run([str(legacy.PANDOC), source.name, '--from=latex', '--to=json', '--standalone',
                    '--citeproc', '--bibliography=references.bib',
                    '--metadata=reference-section-title:References',
                    '-o', str(WORK / 'source_ast.json')], cwd=INPUT, check=True)


def render_math(ast):
    equations = []
    def collect(node):
        if node.get('t') == 'Math':
            key = node['c'][0]['t'], node['c'][1]
            if key not in equations:
                equations.append(key)
        return node
    walk(ast, collect)
    folder = INPUT / 'formulas'
    folder.mkdir(exist_ok=True)
    body = []
    display_number = 0
    for kind, tex in equations:
        tex = re.sub(r'\\(?:begin|end)\{equation\*?\}', '', tex).strip()
        tex = tex.replace(r'\mathit{name}', r'\mathrm{name}').replace(r'\mathit{address}', r'\mathrm{address}').replace(r'\mathit{geography}', r'\mathrm{geography}')
        if kind == 'DisplayMath':
            display_number += 1
            body.append(r'\noindent\makebox[\textwidth]{\hfill$\displaystyle ' + tex + r'$\hfill(3.' + str(display_number) + ')}')
        else:
            body.append(r'\noindent$' + tex + '$')
    source = (r'\documentclass[11pt]{article}' '\n'
              r'\usepackage{amsmath}\usepackage[paperwidth=8.5in,paperheight=3in,margin=1in]{geometry}' '\n'
              r'\pagestyle{empty}\begin{document}' '\n' + '\n\\newpage\n'.join(body) + '\n\\end{document}\n')
    (folder / 'formulas.tex').write_text(source, encoding='utf-8')
    subprocess.run([str(THESIS / 'tectonic.exe'), '--only-cached', '--keep-logs', 'formulas.tex'], cwd=folder, check=True)
    pdf = fitz.open(folder / 'formulas.pdf')
    assert len(pdf) == len(equations)
    records = []
    mapping = {}
    for i, (kind, tex) in enumerate(equations):
        pix = pdf[i].get_pixmap(dpi=360)
        im = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        bbox = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
        assert bbox
        left, top, right, bottom = bbox
        im = im.crop((max(0, left-4), max(0, top-4), right+4, bottom+4))
        path = folder / f'equation_{i+1:02d}.png'
        im.save(path, dpi=(360, 360))
        width, height = im.width / 360, im.height / 360
        record = {'kind': kind, 'tex': tex, 'image': path.relative_to(INPUT).as_posix(),
                  'width_inches': width, 'height_inches': height, 'sha256': sha(path)}
        records.append(record)
        mapping[(kind, tex)] = record
    pdf.close()
    save_json(WORK / 'formula_provenance.json', records)
    def replace(node):
        if node.get('t') == 'Math':
            rec = mapping[(node['c'][0]['t'], node['c'][1])]
            return {'t': 'Image', 'c': [['', [], [['width', f"{rec['width_inches']}in"], ['height', f"{rec['height_inches']}in"]]],
                                       [{'t': 'Str', 'c': rec['tex']}], [rec['image'], '']]}
        return node
    return walk(ast, replace)


def plain_paragraph(p, table=False):
    p.style = 'Normal'
    pf = p.paragraph_format
    pf.left_indent = pf.right_indent = pf.first_line_indent = Pt(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(3 if table else 6)
    pf.line_spacing = 1.0 if table else 1.15
    pf.widow_control = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # Run styles inside hyperlinks otherwise survive p.runs-based cleaning.
    for elem in p._p.xpath('.//w:r'):
        run = Run(elem, p)
        if elem.rPr is not None:
            elem.remove(elem.rPr)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.bold = run.font.italic = run.font.underline = False
        run.font.color.rgb = RGBColor(0, 0, 0)


def build():
    ast = json.loads((WORK / 'source_ast.json').read_text(encoding='utf-8'))
    def separate_display_math(node):
        if node.get('t') not in {'Para', 'Plain'}:
            return node
        if not any(x.get('t') == 'Math' and x['c'][0]['t'] == 'DisplayMath' for x in node['c']):
            return node
        blocks, pending = [], []
        for inline in node['c']:
            if inline.get('t') == 'Math' and inline['c'][0]['t'] == 'DisplayMath':
                if pending:
                    blocks.append({'t': 'Para', 'c': pending})
                blocks.append({'t': 'Para', 'c': [inline]})
                pending = []
            else:
                pending.append(inline)
        if pending:
            blocks.append({'t': 'Para', 'c': pending})
        return {'t': 'Div', 'c': [['', [], []], blocks]}
    ast = render_math(walk(ast, separate_display_math))
    save_json(WORK / 'image_math_ast.json', ast)
    raw = WORK / 'plain_pandoc.docx'
    subprocess.run([str(legacy.PANDOC), str(WORK / 'image_math_ast.json'), '--from=json', '--to=docx',
                    '--standalone', '--resource-path=.;chapters;tables;figures;figures/results',
                    '-o', str(raw)], cwd=INPUT, check=True)
    doc = Document(raw)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Times New Roman', Pt(11)
    normal.font.bold = normal.font.italic = False
    normal.font.color.rgb = RGBColor(0, 0, 0)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    chapters = sections = subsections = tables = figures = 0
    for p in list(doc.paragraphs):
        style, text = p.style.name, p.text.strip()
        if style == 'Heading 1':
            if text != 'References':
                chapters += 1
                sections = subsections = 0
                p.text = f'Chapter {chapters}. {text}'
            p.paragraph_format.page_break_before = True
        elif style == 'Heading 2':
            sections += 1
            subsections = 0
            p.text = f'{chapters}.{sections} {text}'
        elif style == 'Heading 3':
            subsections += 1
            p.text = f'{chapters}.{sections}.{subsections} {text}'
        elif style == 'Table Caption':
            tables += 1
            p.text = f'Table 4.{tables}. {text}'
            p.paragraph_format.keep_with_next = True
        elif style == 'Image Caption':
            figures += 1
            if figures == 1:
                p.text = 'Figure 3.1. ' + text
            elif figures > 2:
                p.text = f'Figure 4.{figures-2}. ' + text
            p.paragraph_format.keep_together = True
        plain_paragraph(p)
        if style.startswith('Heading'):
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(10)
        if p._p.xpath('.//w:drawing') and not p.text.strip():
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_together = True
    assert chapters == 4 and tables == 6 and figures == 5, (chapters, tables, figures)
    assert len(doc.tables) == 6
    weights = [[1.3, 3.05, .9, 1.0], [2.95,.65,1,.9,.9], [2.8,1.35,1.35,.7],
               [.8,2.15,.65,.65,.75,1.35], [.75,.8,.8,1.1,1,.8,.8], [.86,.64,.28,1.13,1.13,.83,.815,.815]]
    for i, table in enumerate(doc.tables):
        widths = [w * 6.5 / sum(weights[i]) for w in weights[i]]
        table.autofit = False
        legacy.set_table_fixed_layout(table)
        legacy.apply_table_widths(table, widths)
        twips = [round(w * 1440) for w in widths]
        twips[-1] = 9360 - sum(twips[:-1])
        for col, width in zip(table._tbl.tblGrid.gridCol_lst, twips):
            col.set(qn('w:w'), str(width))
        for row in table.rows:
            col_index = 0
            for tc in row._tr.tc_lst:
                span = tc.grid_span
                tc.tcPr.find(qn('w:tcW')).set(qn('w:w'), str(sum(twips[col_index:col_index+span])))
                col_index += span
        table._tbl.tblPr.find(qn('w:tblW')).set(qn('w:type'), 'dxa')
        table._tbl.tblPr.find(qn('w:tblW')).set(qn('w:w'), '9360')
        legacy.set_table_borders(table)
        legacy.repeat_header(table.rows[0])
        for row_index, row in enumerate(table.rows):
            legacy.prevent_row_split(row)
            for cell in row.cells:
                legacy.set_cell_margins(cell, top=45, start=60, bottom=45, end=60)
                for p in cell.paragraphs:
                    plain_paragraph(p, table=True)
                    p.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
    # Strip theme decoration, field codes, and link behavior; visible text remains.
    for hyperlink in list(doc._element.xpath('.//w:hyperlink')):
        parent = hyperlink.getparent()
        index = parent.index(hyperlink)
        for child in list(hyperlink):
            parent.insert(index, child)
            index += 1
        parent.remove(hyperlink)
    # Fit page images at their natural aspect ratio; leave formula sizes untouched.
    image_records = []
    formula_by_hash = {r['sha256']: r for r in json.loads((WORK / 'formula_provenance.json').read_text(encoding='utf-8'))}
    for shape in doc.inline_shapes:
        rel = shape._inline.xpath('.//a:blip')[0].get(qn('r:embed'))
        blob = doc.part.related_parts[rel].blob
        import io
        im = Image.open(io.BytesIO(blob))
        is_figure = im.width > 1500 and im.height > 500
        if is_figure:
            ratio = im.height / im.width
            width = min(6.5, 6.9 / ratio)
            shape.width, shape.height = Inches(width), Inches(width * ratio)
        formula = formula_by_hash.get(hashlib.sha256(blob).hexdigest())
        if formula:
            if formula['kind'] == 'DisplayMath':
                paragraph = Paragraph(shape._inline.getparent().getparent().getparent(), doc)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_before = Pt(6)
                paragraph.paragraph_format.space_after = Pt(6)
            else:
                run = shape._inline.getparent().getparent()
                position = OxmlElement('w:position')
                position.set(qn('w:val'), '-4')
                run.get_or_add_rPr().append(position)
        image_records.append({'relationship': rel, 'sha256': hashlib.sha256(blob).hexdigest(),
                              'pixels': list(im.size), 'figure': is_figure})
    assert sum(x['figure'] for x in image_records) == 5
    doc.core_properties.title = legacy.TITLE.replace('\n', ' ')
    doc.core_properties.author = 'Congyuan (Othello) Zheng'
    doc.core_properties.subject = 'Corrected Introduction through Results; plain text styling and formula images'
    doc.core_properties.comments = 'Offline conversion of the accepted corrected manuscript. Formulas are locally rendered images.'
    doc.save(OUTPUT)
    save_json(WORK / 'image_bindings.json', image_records)
    print(f'Built {OUTPUT}; {len(doc.inline_shapes)} images; six tables')


def render():
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx('Word.Application')
    app.Visible = False
    app.DisplayAlerts = 0
    document = None
    try:
        document = app.Documents.Open(str(OUTPUT), ReadOnly=True, AddToRecentFiles=False)
        document.Repaginate()
        document.ExportAsFixedFormat(str(WORK / 'plain_docx_review.pdf'), 17)
        print(f'Word rendered {document.ComputeStatistics(2)} pages')
    finally:
        if document is not None:
            document.Close(False)
        app.Quit()
        pythoncom.CoUninitialize()
    folder = WORK / 'pages'
    folder.mkdir(exist_ok=True)
    pdf = fitz.open(WORK / 'plain_docx_review.pdf')
    for i, page in enumerate(pdf):
        page.get_pixmap(dpi=110).save(folder / f'page_{i+1:02d}.png')
    print(f'Rendered {len(pdf)} page images')


def verify_and_deliver():
    raw, doc = Document(WORK / 'plain_pandoc.docx'), Document(OUTPUT)
    normalize = lambda s: re.sub(r'\s+', ' ', s).strip()
    final_texts = [normalize(p.text) for p in doc.paragraphs]
    missing = [p.text for p in raw.paragraphs if p.text.strip() and not any(normalize(p.text) in s for s in final_texts)]
    cells = lambda d: [[[c.text for c in r.cells] for r in t.rows] for t in d.tables]
    protected = json.loads((WORK / 'source_hashes.json').read_text(encoding='utf-8'))
    target = legacy.FINAL_DOCX.relative_to(ROOT).as_posix()
    unexpected = [p for p, h in protected.items() if p != target and sha(ROOT / p) != h]
    backup = WORK / 'before' / legacy.FINAL_DOCX.name
    assert sha(backup) == protected[target]
    assert sha(legacy.FINAL_DOCX) in {protected[target], sha(OUTPUT)}, 'Original DOCX changed since backup'
    runs = doc._element.xpath('.//w:r[w:t]')
    nonplain = []
    for r in runs:
        font = Run(r, doc).font
        if font.name != 'Times New Roman' or font.size != Pt(11) or font.bold or font.italic or font.underline or font.color.rgb != RGBColor(0, 0, 0):
            nonplain.append(r.xml[:150])
    pdf = fitz.open(WORK / 'plain_docx_review.pdf')
    overflow = [(i+1, list(b[:4])) for i, p in enumerate(pdf) for b in p.get_text('blocks')
                if b[0] < 65 or b[2] > 548 or b[1] < 60 or b[3] > 733]
    checks = {
        'all_converted_source_paragraphs_preserved': not missing,
        'all_six_tables_exact_text_preserved': len(doc.tables) == 6 and cells(raw) == cells(doc),
        'all_text_runs_plain_black_11pt_times': not nonplain,
        'all_paragraph_styles_normal': all(p.style.name == 'Normal' for p in doc.paragraphs),
        'no_word_math_objects': not doc._element.xpath('.//m:oMath | .//m:oMathPara'),
        'no_field_codes': not doc._element.xpath('.//w:instrText | .//w:fldSimple'),
        'five_figures_and_fourteen_formula_images': len(doc.inline_shapes) == 19,
        'source_manuscripts_pdfs_bibliography_other_docx_unchanged': not unexpected,
        'rendered_text_within_page_margins': not overflow,
        'rendered_pages': len(pdf) == 27,
    }
    assert all(checks.values()), checks
    report = {'checks': checks, 'source_paragraphs': sum(bool(p.text.strip()) for p in raw.paragraphs),
              'references': sum(p.style.name == 'Bibliography' for p in raw.paragraphs),
              'pages': len(pdf), 'backup_sha256': sha(backup), 'final_docx_sha256': sha(OUTPUT),
              'review_pdf_sha256': sha(WORK / 'plain_docx_review.pdf'),
              'builder_sha256': sha(Path(__file__)), 'legacy_preparation_helper_sha256': sha(Path(legacy.__file__)),
              'manual_visual_review': 'All 27 rendered pages inspected; final pagination changes re-inspected on pages 18, 19, 20, 21, 23, 24. Other page PNG hashes unchanged.',
              'reviewed_page_sha256': {p.name: sha(p) for p in sorted((WORK / 'pages').glob('page_*.png'))},
              'user_format_override': 'All editable text is Normal style, black 11 pt Times New Roman, without bold, italics, underlines, or heading styles. Figures retain accepted graphics; formulas are images. No automatic TOC or fields.',
              'destination': target, 'network_or_provider_calls': 0}
    save_json(WORK / 'verification.json', report)
    pending = legacy.FINAL_DOCX.with_suffix('.pending.docx')
    assert not pending.exists()
    shutil.copyfile(OUTPUT, pending)
    pending.replace(legacy.FINAL_DOCX)
    assert sha(legacy.FINAL_DOCX) == report['final_docx_sha256']
    (WORK / 'REVIEW_REPORT.md').write_text(
        '# Corrected plain DOCX delivery\n\n'
        'Updated the original Introduction-through-Results DOCX from the accepted corrected manuscript. '
        'All editable text uses Normal style, black 11 pt Times New Roman, with no bold/italic/underline or heading styles. '
        'Preserved all six editable tables, five accepted figure panels, 35 references, and locally rendered 14 mathematical images '
        '(two display equations and twelve inline expressions). Signed numerical results remain ordinary text.\n\n'
        'Microsoft Word rendered 27 pages locally. All pages were visually inspected; table pagination was repaired and the affected pages re-inspected. '
        'All source paragraphs and table cells passed conversion-preservation checks. The original DOCX is preserved in before/. '
        'The current LaTeX, bibliography, PDFs, other DOCX files, and scientific outputs were not modified. '
        'No providers, model calls, empirical reruns, or Git commands were used.\n\n'
        'Reproduction: the versioned source_ast.json and build_input preserve conversion inputs; '
        'run code/build_plain_corrected_thesis_docx.py --build, then --render. '
        'The Windows renderer requires the installed Word COM session. Run --deliver only after reviewing any rebuilt pages.\n', encoding='utf-8')
    save_json(WORK / 'delivery_manifest.json', {p.relative_to(WORK).as_posix(): sha(p)
              for p in sorted(WORK.rglob('*')) if p.is_file() and p.name != 'delivery_manifest.json'})
    print(json.dumps({'delivered': str(legacy.FINAL_DOCX), 'sha256': sha(legacy.FINAL_DOCX), 'pages': len(pdf), 'checks': checks}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--deliver', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.build:
        build()
    if args.render:
        render()
    if args.deliver:
        verify_and_deliver()
