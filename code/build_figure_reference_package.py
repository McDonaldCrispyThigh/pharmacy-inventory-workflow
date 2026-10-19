"""Build the local visual-reference package for the v2 figure rebuild.

This script does not read thesis result data and does not draw thesis figures. It
copies selected public-gallery previews and renders selected pages from locally
retained scholarly PDFs for visual review. The resulting images are review-only
and must not be redistributed as thesis artwork.
"""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/v2/final_results_20260902T012636Z_figure_rebuild_v1/references"
GALLERY = Path(r"C:\Users\16067\AgentFigureGallery\assets\packs\full-public\previews")


GALLERY_PREVIEWS = {
    "forest": [
        "bar_chart/BAR-50C1EC25F9.png",
        "bar_chart/BAR-A5B841DCB8.png",
        "bar_chart/BAR-EBD0FCA21C.png",
        "benchmark_performance/BENCH-379D83BC9F.png",
        "benchmark_performance/BENCH-9A4BE0F490.png",
        "benchmark_performance/BENCH-E677614A03.png",
    ],
    "histogram": [
        "bar_chart/BAR-6FB7EEAFF2.png",
        "bar_chart/BAR-1BD575C850.png",
        "bar_chart/BAR-6CC5400E3F.png",
        "benchmark_performance/BENCH-DB2D3A3011.png",
        "benchmark_performance/BENCH-EC475288A3.png",
        "box_violin_distribution/BOX-7BEC3EF320.png",
    ],
    "map": [
        "spatial_map/SPAT-11448E0216.png",
        "spatial_map/SPAT-E164436E3E.png",
        "spatial_map/SPAT-A91F9C7BB5.png",
    ],
}


PAPER_PAGES = {
    "finlay2026": (
        ROOT
        / "References/Alzheimer s   Dementia - 2026 - Finlay - Measuring  third places   Comparing neighborhood data for cognitive health.pdf",
        [11],
    ),
    "wittenauer2024": (
        ROOT / "References/Locations and characteristics of pharmacy deserts in the United States.pdf",
        [2, 3, 4],
    ),
    "qato2017": (
        ROOT / "References/Geographic access to pharmacies in the United States.pdf",
        list(range(1, 9)),
    ),
    "bader2010": (
        ROOT
        / "References/Economic Geography - 2010 - Bader - Disparities in Neighborhood Food Environments  Implications of Measurement Strategies.pdf",
        [9, 10],
    ),
    "catalano2025": (
        ROOT
        / "References/Distribution of pharmacy deserts and its association with digital divide and residential redlining across the United States.pdf",
        list(range(1, 7)),
    ),
}


PAPER_FIGURES = {
    "F1-R1_finlay2026_figure2": (PAPER_PAGES["finlay2026"][0], 11),
    "F3-R1_wittenauer2024_figure1": (PAPER_PAGES["wittenauer2024"][0], 4),
    "F3-R5_bader2010_figure1": (PAPER_PAGES["bader2010"][0], 10),
    "F3-R4_catalano2025_figure1": (PAPER_PAGES["catalano2025"][0], 5),
}


SHORTLIST_CARDS = {
    "figure1": [
        ("F1-R1", "Finlay et al. 2026, Fig. 2", "paper_figures/F1-R1_finlay2026_figure2.jpeg"),
        ("F1-R2", "AgentFigureGallery BAR-A5B841DCB8", "gallery/forest/BAR-A5B841DCB8.png"),
        ("F1-R3", "AgentFigureGallery BENCH-379D83BC9F", "gallery/forest/BENCH-379D83BC9F.png"),
        ("F1-R4", "AgentFigureGallery BAR-50C1EC25F9", "gallery/forest/BAR-50C1EC25F9.png"),
        ("F1-R5", "AgentFigureGallery BENCH-9A4BE0F490", "gallery/forest/BENCH-9A4BE0F490.png"),
        ("F1-R6", "AgentFigureGallery BENCH-E677614A03", "gallery/forest/BENCH-E677614A03.png"),
        ("F1-R7", "Pocock et al. 2008, Fig. 5\nForest-plot conventions", None),
        ("F1-R8", "Rücker & Schwarzer 2014\nForest/rainforest comparison", None),
    ],
    "figure2": [
        ("F2-R1", "AgentFigureGallery BAR-1BD575C850", "gallery/histogram/BAR-1BD575C850.png"),
        ("F2-R2", "AgentFigureGallery BAR-6FB7EEAFF2", "gallery/histogram/BAR-6FB7EEAFF2.png"),
        ("F2-R3", "AgentFigureGallery BENCH-EC475288A3", "gallery/histogram/BENCH-EC475288A3.png"),
        ("F2-R4", "AgentFigureGallery BENCH-DB2D3A3011", "gallery/histogram/BENCH-DB2D3A3011.png"),
        ("F2-R5", "AgentFigureGallery BAR-6CC5400E3F", "gallery/histogram/BAR-6CC5400E3F.png"),
        ("F2-R6", "AgentFigureGallery BOX-7BEC3EF320\nDuplicate route of F2-R1", "gallery/histogram/BOX-7BEC3EF320.png"),
        ("F2-R7", "Bland & Altman 1986\nSigned paired-difference semantics", None),
        ("F2-R8", "Weissgerber et al. 2015, Fig. 1\nShow the distribution", None),
    ],
    "figure3": [
        ("F3-R1", "Wittenauer et al. 2024, Fig. 1", "paper_figures/F3-R1_wittenauer2024_figure1.jpeg"),
        ("F3-R2", "Qato et al. 2017, Fig. 2\nCategorical county map", None),
        ("F3-R3", "Mathis et al. 2025, Fig. 3\nComparable small-multiple maps", None),
        ("F3-R4", "Catalano et al. 2025, Fig. 1", "paper_figures/F3-R4_catalano2025_figure1.jpeg"),
        ("F3-R5", "Bader et al. 2010, Fig. 1", "paper_figures/F3-R5_bader2010_figure1.jpeg"),
        ("F3-R6", "AgentFigureGallery SPAT-11448E0216", "gallery/map/SPAT-11448E0216.png"),
        ("F3-R7", "AgentFigureGallery SPAT-E164436E3E\nMisclassified boxplot", "gallery/map/SPAT-E164436E3E.png"),
        ("F3-R8", "AgentFigureGallery SPAT-A91F9C7BB5\nMisclassified embedding", "gallery/map/SPAT-A91F9C7BB5.png"),
    ],
}


def copy_gallery_previews() -> None:
    for group, rel_paths in GALLERY_PREVIEWS.items():
        group_dir = OUT / "gallery" / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for rel in rel_paths:
            src = GALLERY / rel
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, group_dir / src.name)


def render_paper_pages(dpi: int = 120) -> None:
    paper_dir = OUT / "papers"
    paper_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    for key, (pdf_path, page_numbers) in PAPER_PAGES.items():
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        with fitz.open(pdf_path) as doc:
            for page_number in page_numbers:
                if not 1 <= page_number <= len(doc):
                    continue
                page = doc[page_number - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pix.save(paper_dir / f"{key}_page{page_number:02d}.png")


def extract_embedded_paper_figures() -> None:
    figure_dir = OUT / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key, (pdf_path, page_number) in PAPER_FIGURES.items():
        with fitz.open(pdf_path) as doc:
            page = doc[page_number - 1]
            images = page.get_images(full=True)
            if not images:
                raise RuntimeError(f"No embedded image found for {key}")
            largest = max(images, key=lambda image: image[2] * image[3])
            payload = doc.extract_image(largest[0])
            output = figure_dir / f"{key}.{payload['ext']}"
            output.write_bytes(payload["image"])


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def make_contact_sheet(images: list[Path], output: Path, title: str, columns: int = 3) -> None:
    if not images:
        return
    thumb_w, thumb_h = 420, 320
    label_h, title_h, gutter = 42, 58, 18
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            gutter + columns * (thumb_w + gutter),
            title_h + gutter + rows * (thumb_h + label_h + gutter),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 14), title, fill="#222222", font=_font(22))
    for index, path in enumerate(images):
        row, col = divmod(index, columns)
        x = gutter + col * (thumb_w + gutter)
        y = title_h + gutter + row * (thumb_h + label_h + gutter)
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile_x = x + (thumb_w - tile.width) // 2
            tile_y = y + (thumb_h - tile.height) // 2
            canvas.paste(tile, (tile_x, tile_y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#B8B8B8", width=1)
        draw.text((x, y + thumb_h + 8), path.stem, fill="#333333", font=_font(15))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(120, 120))


def make_scouting_sheets() -> None:
    scout_dir = OUT / "scouting_contact_sheets"
    for group in GALLERY_PREVIEWS:
        images = sorted((OUT / "gallery" / group).glob("*.png"))
        make_contact_sheet(images, scout_dir / f"gallery_{group}.png", f"AgentFigureGallery: {group}")
    for key in PAPER_PAGES:
        images = sorted((OUT / "papers").glob(f"{key}_page*.png"))
        make_contact_sheet(images, scout_dir / f"paper_{key}.png", f"Paper page scan: {key}")


def make_shortlist_sheets() -> None:
    shortlist_dir = OUT / "shortlist_contact_sheets"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    for figure_key, cards in SHORTLIST_CARDS.items():
        columns, thumb_w, thumb_h = 4, 360, 245
        label_h, title_h, gutter = 62, 62, 16
        rows = (len(cards) + columns - 1) // columns
        canvas = Image.new(
            "RGB",
            (
                gutter + columns * (thumb_w + gutter),
                title_h + gutter + rows * (thumb_h + label_h + gutter),
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((gutter, 16), f"Shortlisted visual references: {figure_key}", fill="#222222", font=_font(22))
        for index, (reference_id, label, rel_path) in enumerate(cards):
            row, col = divmod(index, columns)
            x = gutter + col * (thumb_w + gutter)
            y = title_h + gutter + row * (thumb_h + label_h + gutter)
            draw.rectangle((x, y, x + thumb_w, y + thumb_h), fill="#F8F8F8", outline="#AFAFAF", width=1)
            if rel_path:
                image_path = OUT / rel_path
                with Image.open(image_path) as source:
                    tile = source.convert("RGB")
                    tile.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
                    canvas.paste(tile, (x + (thumb_w - tile.width) // 2, y + (thumb_h - tile.height) // 2))
            else:
                draw.text((x + 18, y + 20), reference_id, fill="#315D7C", font=_font(24))
                wrapped = "\n".join(textwrap.wrap(label.replace("\n", " — "), width=32))
                draw.multiline_text((x + 18, y + 62), wrapped, fill="#333333", font=_font(18), spacing=7)
            label_text = f"{reference_id}  {label}"
            wrapped_label = "\n".join(textwrap.wrap(label_text.replace("\n", " — "), width=43))
            draw.multiline_text((x + 4, y + thumb_h + 7), wrapped_label, fill="#333333", font=_font(14), spacing=3)
        canvas.save(shortlist_dir / f"{figure_key}_reference_contact_sheet.png", dpi=(120, 120))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    copy_gallery_previews()
    render_paper_pages(dpi=args.dpi)
    extract_embedded_paper_figures()
    make_scouting_sheets()
    make_shortlist_sheets()
    print(f"Reference images written to {OUT}")


if __name__ == "__main__":
    main()
