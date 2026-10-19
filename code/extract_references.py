"""Extract text from all reference PDFs in References/ to References/_parsed/.

Each PDF -> a .txt with page markers. Lets the assistant read text via the
Read tool instead of needing pdftoppm.
"""
from pathlib import Path

import fitz  # PyMuPDF

SRC = Path("References")
OUT = SRC / "_parsed"
OUT.mkdir(exist_ok=True)

for pdf in sorted(SRC.glob("*.pdf")):
    target = OUT / (pdf.stem + ".txt")
    if target.exists() and target.stat().st_size > 1000:
        print(f"skip (exists): {pdf.name}")
        continue
    try:
        doc = fitz.open(pdf)
        chunks = []
        for i, page in enumerate(doc, 1):
            chunks.append(f"\n===== PAGE {i} =====\n")
            chunks.append(page.get_text("text"))
        target.write_text("".join(chunks), encoding="utf-8")
        size_kb = target.stat().st_size / 1024
        print(f"OK [{size_kb:.0f} KB, {len(doc)} pages]: {pdf.name}")
        doc.close()
    except Exception as e:
        print(f"FAIL: {pdf.name}: {e}")

print(f"\nWrote .txt extractions to {OUT}")
