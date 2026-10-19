"""Prepare thesis-safe workflow figure assets from the user-supplied PDFs.

The original PDFs are preserved. Panel A receives two narrowly scoped wording
corrections so the displayed method matches the frozen forced-choice schema and
the cautious request wording in the thesis. Both panels are rasterized at the
same physical DPI for DOCX conversion; LaTeX continues to use PDF sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "thesis" / "figures" / "results"
PANEL_A_SOURCE = FIGURES / "Figure_3.1A_inventory_construction_cropped.pdf"
PANEL_B_SOURCE = FIGURES / "Figure_3.1B_correspondence_proximity_matched_scale.pdf"
PANEL_A_THESIS = FIGURES / "Figure_3.1A_inventory_construction_thesis.pdf"
PANEL_A_PNG = FIGURES / "Figure_3.1A_inventory_construction_thesis.png"
PANEL_B_PNG = FIGURES / "Figure_3.1B_correspondence_proximity_matched_scale.png"
MANIFEST = FIGURES / "Figure_3.1_workflow_asset_manifest.json"
ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")

EXPECTED_SOURCE_HASHES = {
    PANEL_A_SOURCE.name: "64952f866c4e76a3bf25662b4bdfe934f065482940ce03f035e5cb550a680201",
    PANEL_B_SOURCE.name: "d6a7ba01b177c49d86eb24abc2422c616be67bc46c665e7fc5fb2cb738bd8413",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    for path in (PANEL_A_SOURCE, PANEL_B_SOURCE):
        actual = sha256(path)
        expected = EXPECTED_SOURCE_HASHES[path.name]
        if actual != expected:
            raise RuntimeError(
                f"Refusing unexpected workflow source {path.name}: {actual}"
            )
    if not ARIAL.exists():
        raise FileNotFoundError(f"Required font not found: {ARIAL}")


def replace_line(page: pymupdf.Page, old: str, new: str, font_size: float) -> None:
    matches = page.search_for(old)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}; found {len(matches)}")
    rect = matches[0]
    redaction = pymupdf.Rect(rect.x0 - 1.0, rect.y0 - 1.0, rect.x1 + 2.0, rect.y1 + 1.0)
    page.add_redact_annot(redaction, fill=(1, 1, 1))
    page.apply_redactions()
    baseline = rect.y1 - 1.9
    color = (0x24 / 255, 0x37 / 255, 0x46 / 255)
    page.insert_text(
        (rect.x0, baseline),
        new,
        fontsize=font_size,
        fontname="ArialWorkflow",
        fontfile=str(ARIAL),
        color=color,
        overlay=True,
    )


def build_corrected_panel_a() -> None:
    document = pymupdf.open(PANEL_A_SOURCE)
    if document.page_count != 1:
        raise RuntimeError("Panel A must contain exactly one page")
    page = document[0]
    replace_line(page, "• One request per", "• Requested for every", 9.04)
    replace_line(page, "• NAICS code or null.", "• Target-construct status.", 9.18)
    document.save(PANEL_A_THESIS, garbage=4, deflate=True)
    document.close()

    check = pymupdf.open(PANEL_A_THESIS)
    text = (
        check[0].get_text().replace("\u00a0", " ").replace("\u00ad", "-")
    )
    check.close()
    required = ("Requested for every", "Target-construct status.")
    forbidden = ("One request per", "NAICS code or null")
    if any(value not in text for value in required) or any(
        value in text for value in forbidden
    ):
        raise RuntimeError("Panel A wording verification failed")


def render_png(source: Path, output: Path, dpi: int = 300) -> tuple[int, int]:
    document = pymupdf.open(source)
    if document.page_count != 1:
        raise RuntimeError(f"Expected one-page figure: {source}")
    scale = dpi / 72
    pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pixmap.save(output)
    dimensions = (pixmap.width, pixmap.height)
    document.close()
    return dimensions


def main() -> None:
    verify_sources()
    build_corrected_panel_a()
    dimensions = {
        PANEL_A_PNG.name: render_png(PANEL_A_THESIS, PANEL_A_PNG),
        PANEL_B_PNG.name: render_png(PANEL_B_SOURCE, PANEL_B_PNG),
    }
    manifest = {
        "schema_version": "nets-ai-workflow-figure-assets-v1",
        "source_files_preserved": True,
        "panel_a_text_corrections": {
            "• One request per": "• Requested for every",
            "• NAICS code or null.": "• Target-construct status.",
        },
        "docx_raster_dpi": 300,
        "pixel_dimensions": {key: list(value) for key, value in dimensions.items()},
        "sha256": {
            path.name: sha256(path)
            for path in (
                PANEL_A_SOURCE,
                PANEL_B_SOURCE,
                PANEL_A_THESIS,
                PANEL_A_PNG,
                PANEL_B_PNG,
            )
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
