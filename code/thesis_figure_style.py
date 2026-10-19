"""Central visual and export contract for the thesis Results figures.

The module contains presentation rules only. It does not calculate or alter any
scientific result. All canvases use their intended final LaTeX size so typography
is never rescued by down-scaling.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


THESIS_TEXT_WIDTH_IN = 360.0 / 72.27
THESIS_TEXT_HEIGHT_IN = 595.0 / 72.27
# Measured from the compiled pdflscape page: the landscape line width is
# 539.427 PDF points (the v2 audit exposed a 0.91 down-scaling of an 8.233-in
# source canvas). Author at the actual width so 8.5-pt labels stay 8.5 pt.
THESIS_LANDSCAPE_WIDTH_IN = 539.427161 / 72.0

FIGURE_SIZES = {
    "figure01_record_flow": (THESIS_TEXT_WIDTH_IN, 3.20),
    "figure02_core_decisions_confidence": (THESIS_TEXT_WIDTH_IN, 5.20),
    "figure03_geography_correspondence": (THESIS_TEXT_WIDTH_IN, 4.15),
    "figure04_name_recurrence": (THESIS_TEXT_WIDTH_IN, 4.15),
    "figure05_spatial_disagreement_maps": (THESIS_LANDSCAPE_WIDTH_IN, 3.85),
    "figure05_vertical_candidate": (THESIS_TEXT_WIDTH_IN, 8.05),
    "figure06_nearest_distance_difference": (THESIS_TEXT_WIDTH_IN, 3.55),
    "figure06_ecdf_candidate": (THESIS_TEXT_WIDTH_IN, 3.55),
}

FONT_FAMILY = "DejaVu Sans"
FONT_SIZES = {
    "annotation": 8.5,
    "tick": 8.5,
    "axis": 9.25,
    "legend": 8.5,
    "panel": 10.25,
    "panel_label": 10.5,
}

# Paul Tol high-contrast blue/amber anchor the two source semantics. The map's
# two neutral agreement states are deliberately separated in luminance.
COLORS = {
    "workflow": "#004488",
    "workflow_light": "#8DB7D3",
    "board": "#DDAA33",
    "board_dark": "#8C6508",
    "agreement_within": "#F1F1EE",
    "agreement_beyond": "#6D6D6D",
    "secondary": "#A7A7A7",
    "secondary_light": "#D9D9D9",
    "grid": "#E3E3E3",
    "text": "#333333",
    "annotation": "#555555",
    "water": "#DCECF2",
    "context": "#9A9A9A",
    "context_fill": "#FAFAF8",
    "white": "#FFFFFF",
    "confidence_high": "#24476B",
    "confidence_moderate": "#87A6C4",
    "confidence_low": "#E1E8EF",
}

LINE_WIDTHS = {
    "data": 1.35,
    "interval": 1.25,
    "grid": 0.55,
    "tract": 0.22,
    "context": 0.55,
    "frame": 1.05,
}

MARKER_SIZES = {"dot": 34, "forest": 38, "process": 34}
EXPORT_DPI = 500


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def thesis_figure_context():
    """Apply one scoped, publication-oriented Matplotlib style."""

    with mpl.rc_context():
        sns.set_theme(font_scale=1.0, style="whitegrid", font=FONT_FAMILY)
        mpl.rcParams.update(
            {
                "font.family": FONT_FAMILY,
                "font.size": FONT_SIZES["annotation"],
                "axes.titlesize": FONT_SIZES["panel"],
                "axes.titleweight": "semibold",
                "axes.labelsize": FONT_SIZES["axis"],
                "xtick.labelsize": FONT_SIZES["tick"],
                "ytick.labelsize": FONT_SIZES["tick"],
                "legend.fontsize": FONT_SIZES["legend"],
                "figure.titlesize": FONT_SIZES["panel"],
                "text.color": COLORS["text"],
                "axes.labelcolor": COLORS["text"],
                "axes.edgecolor": COLORS["text"],
                "xtick.color": COLORS["annotation"],
                "ytick.color": COLORS["annotation"],
                "axes.linewidth": 0.75,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.axisbelow": True,
                "grid.color": COLORS["grid"],
                "grid.linewidth": LINE_WIDTHS["grid"],
                "grid.alpha": 1.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "figure.facecolor": COLORS["white"],
                "axes.facecolor": COLORS["white"],
                "savefig.facecolor": COLORS["white"],
            }
        )
        yield


def clean_axis(ax, *, grid_axis=None, left_spine=False, bottom_spine=True):
    """Apply the shared low-ink statistical-axis treatment."""

    sns.despine(ax=ax, left=not left_spine, bottom=not bottom_spine)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=LINE_WIDTHS["grid"])
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelcolor=COLORS["annotation"],
    )


def legend_kwargs(**overrides):
    """Return the consistent unobtrusive legend treatment."""

    values = {
        "frameon": True,
        "facecolor": "white",
        "framealpha": 0.92,
        "edgecolor": "lightgrey",
        "labelcolor": COLORS["annotation"],
        "fontsize": FONT_SIZES["legend"],
    }
    values.update(overrides)
    return values


def _assert_canvas_contains_visible_artists(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tolerance = 3.0
    offenders = []
    suppressed_axis_artists = set()
    for axes in fig.axes:
        if not axes.axison:
            suppressed_axis_artists.update(axes.get_xticklabels())
            suppressed_axis_artists.update(axes.get_yticklabels())
            suppressed_axis_artists.add(axes.xaxis.label)
            suppressed_axis_artists.add(axes.yaxis.label)
    for artist in fig.findobj():
        if not artist.get_visible() or not hasattr(artist, "get_window_extent"):
            continue
        if artist in suppressed_axis_artists:
            continue
        # Matplotlib tick-label artists remain individually ``visible`` even
        # when their parent axes has been turned off. They are not rendered.
        parent_axes = getattr(artist, "axes", None)
        if parent_axes is not None and hasattr(parent_axes, "axison") and not parent_axes.axison:
            continue
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            continue
        if not np.isfinite(bbox.extents).all() or bbox.width == 0 or bbox.height == 0:
            continue
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            label = type(artist).__name__
            if hasattr(artist, "get_text"):
                label += f"({artist.get_text()!r}; bbox={tuple(round(v, 1) for v in bbox.extents)})"
            offenders.append(label)
    if offenders:
        raise ValueError(f"visible artists extend beyond fixed canvas: {sorted(set(offenders))}")


def _inspect_pdf(path, expected_size):
    document = fitz.open(path)
    if document.page_count != 1:
        raise ValueError(f"figure PDF must have one page: {path}")
    page = document[0]
    actual = (page.rect.width / 72.0, page.rect.height / 72.0)
    if any(abs(a - e) > 0.015 for a, e in zip(actual, expected_size)):
        raise ValueError(f"PDF canvas mismatch for {path}: {actual} != {expected_size}")
    fonts = page.get_fonts(full=True)
    font_names = sorted({str(item[3]) for item in fonts})
    if not any("DejaVuSans" in name.replace("-", "") for name in font_names):
        raise ValueError(f"DejaVu Sans not represented in PDF font resources: {font_names}")
    embedded = 0
    for item in fonts:
        xref = int(item[0])
        if xref <= 0:
            continue
        try:
            extracted = document.extract_font(xref)
        except Exception:
            continue
        if extracted and len(extracted) >= 4 and extracted[3]:
            embedded += 1
    text_dict = page.get_text("dict")
    visible_sizes = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if str(span.get("text", "")).strip():
                    visible_sizes.append(float(span["size"]))
    min_text = min(visible_sizes) if visible_sizes else None
    if min_text is not None and min_text < 8.45:
        raise ValueError(f"PDF contains visible text below 8.5 pt: {min_text:.3f} in {path}")
    document.close()
    return {
        "page_inches": {"width": actual[0], "height": actual[1]},
        "font_resources": font_names,
        "embedded_font_streams": embedded,
        "minimum_extracted_text_pt": min_text,
    }


def save_publication_figure(fig, stem, *, expected_size, dpi=EXPORT_DPI):
    """Save PDF, SVG, and opaque RGB PNG, then reopen and verify each file."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing figure: {path}")
    size = tuple(float(value) for value in fig.get_size_inches())
    if any(abs(a - e) > 1e-6 for a, e in zip(size, expected_size)):
        raise ValueError(f"authored canvas {size} does not match expected {expected_size}")
    _assert_canvas_contains_visible_artists(fig)
    # Plot constructors leave the scoped style context before reaching this
    # exporter. Reassert the backend-specific font contract at save time.
    previous_svg_fonttype = mpl.rcParams["svg.fonttype"]
    previous_pdf_fonttype = mpl.rcParams["pdf.fonttype"]
    previous_ps_fonttype = mpl.rcParams["ps.fonttype"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    metadata = {
        "Title": stem.name,
        "Author": "NETS-AI thesis workflow",
        "Subject": "Aggregate Results figure; frozen local artifacts only",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        outputs["pdf"],
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        metadata=metadata,
    )
    fig.savefig(
        outputs["svg"],
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        metadata={"Title": stem.name, "Creator": "NETS-AI thesis workflow"},
    )
    fig.savefig(
        outputs["png"],
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    mpl.rcParams["svg.fonttype"] = previous_svg_fonttype
    mpl.rcParams["pdf.fonttype"] = previous_pdf_fonttype
    mpl.rcParams["ps.fonttype"] = previous_ps_fonttype
    plt.close(fig)

    # Matplotlib's Agg backend truncates non-integer physical pixels.
    expected_pixels = tuple(int(value * dpi) for value in expected_size)
    with Image.open(outputs["png"]) as image:
        if image.mode == "RGBA":
            if image.getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"PNG contains transparency: {outputs['png']}")
            converted = image.convert("RGB")
            converted.save(outputs["png"], dpi=(dpi, dpi), optimize=True)
        elif image.mode != "RGB":
            converted = image.convert("RGB")
            converted.save(outputs["png"], dpi=(dpi, dpi), optimize=True)
    with Image.open(outputs["png"]) as image:
        actual_dpi = image.info.get("dpi", (0.0, 0.0))
        if image.mode != "RGB" or image.size != expected_pixels:
            raise ValueError(
                f"PNG contract mismatch: mode={image.mode}, size={image.size}, expected={expected_pixels}"
            )
        if any(abs(float(value) - dpi) > 1.0 for value in actual_dpi):
            raise ValueError(f"PNG resolution metadata mismatch: {actual_dpi}")

    svg_text = outputs["svg"].read_text(encoding="utf-8")
    if "<svg" not in svg_text or "DejaVu Sans" not in svg_text:
        raise ValueError(f"SVG does not preserve expected vector text: {outputs['svg']}")
    if any(path.stat().st_size < 1_000 for path in outputs.values()):
        raise ValueError(f"empty or implausibly small export for {stem}")
    pdf_report = _inspect_pdf(outputs["pdf"], expected_size)
    return {
        "canvas_inches": {"width": expected_size[0], "height": expected_size[1]},
        "formats": {
            key: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
        "png": {
            "dpi": dpi,
            "mode": "RGB",
            "pixels": {"width": expected_pixels[0], "height": expected_pixels[1]},
            "opaque": True,
        },
        "pdf": pdf_report,
        "bbox_inches": None,
        "fixed_canvas": True,
        "source_data_changed": False,
        "scientific_estimand_changed": False,
    }


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
