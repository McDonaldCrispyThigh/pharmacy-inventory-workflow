"""Render and audit the temporary thesis-class figure review document.

This script performs presentation QA only. It reads the compiled review PDF and
the preserved compiled thesis, renders fixed pages, and creates accessibility and
same-physical-scale contact sheets. It does not read or recompute scientific data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/v2/final_results_20260902T012636Z_figure_rebuild_v1"
PAGE_DIR = OUT / "page_mockups"
REVIEW_PDF = PAGE_DIR / "review_pages.pdf"
BASELINE_PDF = ROOT / "analysis/v2/final_results_20260902T012636Z_visual_redesign_v3/compiled_thesis/introduction_literature_methods_results_draft.pdf"
RENDER_DPI = 170
CONTACT_DPI = 105

REVIEW_LABELS = [
    "Figure 1A — two panels",
    "Figure 1B — grouped forest",
    "Figure 2A — neutral histogram",
    "Figure 2B — signed-color histogram",
    "Figure 2 diagnostic — ECDF",
    "Figure 3A — no roads",
    "Figure 3B — primary highways",
    "Optional appendix — 2 miles",
]

BASELINE_PAGES = {
    "Figure 1 — rejected": 34,
    "Figure 2 — rejected": 39,
    "Figure 3 — rejected": 37,
}

CVD_MATRICES = {
    "deuteranopia": np.array(
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
    ),
    "protanopia": np.array(
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
    for candidate in candidates:
        path = Path(r"C:\Windows\Fonts") / candidate
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render_page(document: fitz.Document, index: int, dpi: int) -> Image.Image:
    page = document[index]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def transform(image: Image.Image, mode: str) -> Image.Image:
    base = image.convert("RGB")
    if mode == "grayscale":
        return ImageOps.grayscale(base).convert("RGB")
    array = np.asarray(base, dtype=float) / 255.0
    transformed = np.clip(array @ CVD_MATRICES[mode].T, 0, 1)
    return Image.fromarray(np.round(transformed * 255).astype(np.uint8), mode="RGB")


def save_review_pages() -> list[dict]:
    target = PAGE_DIR / "rendered_pages"
    target.mkdir(parents=True, exist_ok=True)
    accessibility_root = PAGE_DIR / "accessibility"
    with fitz.open(REVIEW_PDF) as document:
        if document.page_count != 8:
            raise AssertionError(f"Expected 8 review pages, found {document.page_count}")
        records = []
        for index, label in enumerate(REVIEW_LABELS):
            page = document[index]
            expected_landscape = index in {5, 6}
            actual_landscape = page.rect.width > page.rect.height
            if actual_landscape != expected_landscape:
                raise AssertionError(f"Unexpected orientation on review page {index + 1}")
            image = render_page(document, index, RENDER_DPI)
            path = target / f"page_{index + 1:02d}.png"
            image.save(path, dpi=(RENDER_DPI, RENDER_DPI))
            span_sizes = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            span_sizes.append(float(span["size"]))
            accessibility_paths = {}
            for mode in ["grayscale", *CVD_MATRICES]:
                mode_dir = accessibility_root / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                mode_path = mode_dir / path.name
                transform(image, mode).save(mode_path, dpi=(RENDER_DPI, RENDER_DPI))
                accessibility_paths[mode] = mode_path.relative_to(ROOT).as_posix()
            records.append(
                {
                    "page": index + 1,
                    "label": label,
                    "orientation": "landscape" if actual_landscape else "portrait",
                    "page_inches": [round(page.rect.width / 72, 3), round(page.rect.height / 72, 3)],
                    "render_dpi": RENDER_DPI,
                    "render_pixels": list(image.size),
                    "render_path": path.relative_to(ROOT).as_posix(),
                    "render_sha256": sha256(path),
                    "minimum_extracted_text_pt": round(min(span_sizes), 3) if span_sizes else None,
                    "accessibility": accessibility_paths,
                }
            )
    return records


def save_baseline_pages() -> dict[str, Path]:
    target = PAGE_DIR / "baseline_pages"
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with fitz.open(BASELINE_PDF) as document:
        for label, one_based_page in BASELINE_PAGES.items():
            image = render_page(document, one_based_page - 1, RENDER_DPI)
            path = target / f"page_{one_based_page:02d}.png"
            image.save(path, dpi=(RENDER_DPI, RENDER_DPI))
            outputs[label] = path
    return outputs


def card(image: Image.Image, label: str, label_height: int = 34) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + label_height), "white")
    result.paste(image, (0, label_height))
    ImageDraw.Draw(result).text((8, 7), label, fill="#222222", font=font(18, bold=True))
    return result


def contact_sheet(rows: list[list[tuple[str, Image.Image]]], output: Path) -> None:
    gap = 22
    cards_by_row = [[card(image, label) for label, image in row] for row in rows]
    row_widths = [sum(item.width for item in row) + gap * (len(row) - 1) for row in cards_by_row]
    row_heights = [max(item.height for item in row) for row in cards_by_row]
    canvas = Image.new("RGB", (max(row_widths), sum(row_heights) + gap * (len(rows) - 1)), "#D9D9D9")
    y = 0
    for items, height in zip(cards_by_row, row_heights):
        x = 0
        for item in items:
            canvas.paste(item, (x, y))
            x += item.width + gap
        y += height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(CONTACT_DPI, CONTACT_DPI))


def page_at_contact_scale(document: fitz.Document, one_based: int) -> Image.Image:
    return render_page(document, one_based - 1, CONTACT_DPI)


def build_contacts() -> dict[str, str]:
    outputs = {}
    with fitz.open(BASELINE_PDF) as baseline, fitz.open(REVIEW_PDF) as review:
        comparison_rows = [
            [
                ("Rejected thesis page", page_at_contact_scale(baseline, 34)),
                ("Candidate A page", page_at_contact_scale(review, 1)),
                ("Candidate B page", page_at_contact_scale(review, 2)),
            ],
            [
                ("Rejected thesis page", page_at_contact_scale(baseline, 39)),
                ("Candidate A page", page_at_contact_scale(review, 3)),
                ("Candidate B page", page_at_contact_scale(review, 4)),
                ("ECDF diagnostic page", page_at_contact_scale(review, 5)),
            ],
            [
                ("Rejected thesis page", page_at_contact_scale(baseline, 37)),
                ("Candidate A page", page_at_contact_scale(review, 6)),
                ("Candidate B page", page_at_contact_scale(review, 7)),
                ("2-mile appendix page", page_at_contact_scale(review, 8)),
            ],
        ]
        path = OUT / "before_after_page_contact_sheet.png"
        contact_sheet(comparison_rows, path)
        outputs["before_after"] = path.relative_to(ROOT).as_posix()

        full_color = [(label, page_at_contact_scale(review, index + 1)) for index, label in enumerate(REVIEW_LABELS)]
        for mode in ["color", "grayscale", *CVD_MATRICES]:
            cards = []
            for label, page_image in full_color:
                cards.append((label, page_image if mode == "color" else transform(page_image, mode)))
            rows = [cards[:4], cards[4:]]
            mode_path = PAGE_DIR / f"review_page_contact_{mode}.png"
            contact_sheet(rows, mode_path)
            outputs[mode] = mode_path.relative_to(ROOT).as_posix()
    return outputs


def main() -> None:
    for required in [REVIEW_PDF, BASELINE_PDF]:
        if not required.is_file():
            raise FileNotFoundError(required)
    pages = save_review_pages()
    baselines = save_baseline_pages()
    contacts = build_contacts()
    manifest = {
        "schema_version": "nets-ai-figure-review-package-v1",
        "scope": "presentation QA only; no scientific recomputation; no thesis modification",
        "review_pdf": {
            "path": REVIEW_PDF.relative_to(ROOT).as_posix(),
            "sha256": sha256(REVIEW_PDF),
            "bytes": REVIEW_PDF.stat().st_size,
        },
        "baseline_pdf": {
            "path": BASELINE_PDF.relative_to(ROOT).as_posix(),
            "sha256": sha256(BASELINE_PDF),
            "bytes": BASELINE_PDF.stat().st_size,
            "pages_used": BASELINE_PAGES,
        },
        "review_pages": pages,
        "baseline_page_renders": {label: path.relative_to(ROOT).as_posix() for label, path in baselines.items()},
        "contact_sheets": contacts,
        "checks": {
            "review_page_count_8": len(pages) == 8,
            "portrait_pages_1_to_5_and_8": all(pages[index]["orientation"] == "portrait" for index in [0, 1, 2, 3, 4, 7]),
            "landscape_pages_6_and_7": all(pages[index]["orientation"] == "landscape" for index in [5, 6]),
            "same_contact_render_dpi": CONTACT_DPI,
            "color_and_three_accessibility_modes_created": set(contacts) == {"before_after", "color", "grayscale", "deuteranopia", "protanopia"},
        },
    }
    output = OUT / "review_package_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rendered {len(pages)} review pages and accessibility/contact-sheet QA")
    print(output)


if __name__ == "__main__":
    main()
