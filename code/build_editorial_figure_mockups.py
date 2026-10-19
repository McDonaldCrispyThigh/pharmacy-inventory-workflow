"""Build three low-resolution editorial mockups from frozen Results artifacts.

This script does not rerun an analysis and does not edit the thesis. It reads the
preserved figure-data exports from the visual-redesign v3 package and local Census
geometry, then writes PNG-only draft concepts at their intended physical sizes.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

from study_design import TARGET_ZIPS


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260902T012636Z"
SOURCE = (
    ROOT
    / "analysis"
    / "v2"
    / f"final_results_{RUN_ID}_visual_redesign_v3"
)
OUTPUT = (
    ROOT
    / "analysis"
    / "v2"
    / f"final_results_{RUN_ID}_editorial_mockups_v3"
)
MOCKUPS = OUTPUT / "mockups"

DRAFT_DPI = 150
CONTACT_PPI = 100
FOREST_SIZE = (4.981, 2.35)
MAP_SIZE = (7.492, 4.60)
DISTANCE_SIZE = (4.981, 2.65)

COLORS = {
    "workflow": "#335F7D",
    "board": "#B8782E",
    "agreement": "#E5E4E0",
    "secondary": "#7B7F82",
    "text": "#222222",
    "annotation": "#4F5153",
    "grid": "#D9D9D6",
    "context": "#8A8C8B",
    "context_fill": "#FAFAF8",
    "water": "#DCEBF1",
    "iqr": "#D6D6D2",
}


def require(condition, message):
    """Raise a clear error when a frozen-data or output contract fails."""
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    """Return a lowercase SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_thesis_font():
    """Register the thesis's Latin Modern Roman font with a stable fallback."""
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    roots = [
        local / "TectonicProject" / "Tectonic" / "bundles" / "data",
        local / "TectonicProject" / "Tectonic" / "cache" / "bundles" / "data",
    ]
    registered = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in (
            "lmroman10-regular.otf",
            "lmroman10-bold.otf",
            "lmroman10-italic.otf",
        ):
            matches = list(root.rglob(name))
            if matches:
                font_manager.fontManager.addfont(str(matches[0]))
                registered.append(matches[0])
        if registered:
            break
    family = "Latin Modern Roman" if registered else "STIXGeneral"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "STIXGeneral", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family, registered


def save_mockup(fig, name, size):
    """Save one exact-size, low-resolution PNG without tight-box resizing."""
    path = MOCKUPS / name
    require(not path.exists(), f"refusing to overwrite {path}")
    observed = tuple(float(value) for value in fig.get_size_inches())
    require(
        all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(observed, size)),
        f"unexpected canvas size for {name}: {observed}",
    )
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=DRAFT_DPI,
        bbox_inches=None,
        facecolor="white",
        metadata={"Title": "Draft editorial mockup; not final artwork"},
    )
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rendered.convert("RGB").save(
            path,
            format="PNG",
            dpi=(DRAFT_DPI, DRAFT_DPI),
            optimize=True,
        )
    buffer.close()
    with Image.open(path) as image:
        require(image.mode in {"RGB", "RGBA"}, f"unexpected image mode {name}")
        expected = tuple(int(round(value * DRAFT_DPI)) for value in size)
        require(
            all(abs(observed - target) <= 1 for observed, target in zip(image.size, expected)),
            f"unexpected raster size {name}: {image.size}",
        )
        actual_pixels = image.size
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "canvas_inches": [size[0], size[1]],
        "pixels": list(actual_pixels),
        "dpi": DRAFT_DPI,
    }


def load_frozen_tables():
    """Read and validate the three preserved display-data files."""
    data = SOURCE / "figure_data"
    paths = {
        "recurrence": data / "figure04_name_recurrence.csv",
        "states": data / "figure05_tract_states.csv",
        "distance": data / "figure06_signed_distances.csv",
    }
    for path in paths.values():
        require(path.is_file(), f"missing frozen figure data: {path}")
    recurrence = pd.read_csv(paths["recurrence"])
    states = pd.read_csv(paths["states"], dtype={"GEOID": str})
    distance = pd.read_csv(paths["distance"], dtype={"GEOID": str})

    require(len(states) == 327 and states["GEOID"].nunique() == 327, "map frame must contain 327 unique tracts")
    require(len(distance) == 327 and distance["GEOID"].nunique() == 327, "distance frame must contain 327 unique tracts")
    require(set(states["GEOID"]) == set(distance["GEOID"]), "map and distance tract frames differ")

    expected_states = {
        "discordance_type_0_5mi": {
            "neither_beyond": 93,
            "both_beyond": 188,
            "board_only_beyond": 28,
            "google_only_beyond": 18,
        },
        "discordance_type_1_0mi": {
            "neither_beyond": 233,
            "both_beyond": 52,
            "board_only_beyond": 32,
            "google_only_beyond": 10,
        },
        "discordance_type_2_0mi": {
            "neither_beyond": 322,
            "both_beyond": 1,
            "board_only_beyond": 4,
            "google_only_beyond": 0,
        },
    }
    for column, expected in expected_states.items():
        observed = states[column].value_counts().to_dict()
        for key in expected:
            observed.setdefault(key, 0)
        require(
            {key: int(observed[key]) for key in expected} == expected,
            f"frozen map counts changed for {column}",
        )

    values = distance["workflow_minus_board_miles"].astype(float)
    require(round(float(values.median()), 3) == -0.009, "frozen median changed")
    require(round(float(values.quantile(0.25)), 3) == -0.078, "frozen first quartile changed")
    require(round(float(values.quantile(0.75)), 3) == 0.022, "frozen third quartile changed")
    require(round(float(values.min()), 3) == -1.909, "frozen minimum changed")
    require(round(float(values.max()), 3) == 1.044, "frozen maximum changed")
    require(int(values.lt(0).sum()) == 192 and int(values.gt(0).sum()) == 135, "frozen distance directions changed")

    return recurrence, states, distance, paths


def recurrence_mockup(frame):
    """Draw a compact, single-axis four-row forest plot."""
    subset = frame.loc[
        frame["recurrence_threshold"].eq(2)
        & ~frame["group"].eq("unresolved")
    ].copy()
    order = [
        ("Workflow inventory locations", "repeated-name component"),
        ("Workflow inventory locations", "single-occurrence name component"),
        ("Board keyword-screen reference", "repeated-name component"),
        ("Board keyword-screen reference", "single-occurrence name component"),
    ]
    labels = [
        "Workflow — repeated-name",
        "Workflow — single-occurrence",
        "Board — repeated-name",
        "Board — single-occurrence",
    ]
    positions = [3.15, 2.15, 0.75, -0.25]
    expected = [54.50, 21.55, 93.13, 51.06]

    rows = []
    for source, group in order:
        row = subset.loc[
            subset["source"].eq(source) & subset["group"].eq(group)
        ]
        require(len(row) == 1, f"missing unique recurrence row: {source}/{group}")
        rows.append(row.iloc[0])
    require(
        [round(float(row["linked_pct"]), 2) for row in rows] == expected,
        "frozen recurrence estimates changed",
    )

    fig = plt.figure(figsize=FOREST_SIZE)
    ax = fig.add_axes([0.405, 0.23, 0.555, 0.69])
    for index, (row, y) in enumerate(zip(rows, positions)):
        value = float(row["linked_pct"])
        low = float(row["cluster_bootstrap_95_low_pct"])
        high = float(row["cluster_bootstrap_95_high_pct"])
        is_workflow = index < 2
        color = COLORS["workflow"] if is_workflow else COLORS["board"]
        marker = "o" if is_workflow else "s"
        ax.hlines(y, low, high, color=color, linewidth=1.35, zorder=2)
        ax.vlines([low, high], y - 0.075, y + 0.075, color=color, linewidth=0.9, zorder=2)
        ax.scatter(
            value,
            y,
            s=30,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        if value > 88:
            x, align = value - 2.2, "right"
        else:
            x, align = value + 2.2, "left"
        ax.text(
            x,
            y + 0.18,
            f"{value:.1f}%",
            ha=align,
            va="bottom",
            fontsize=8.4,
            color=COLORS["annotation"],
        )

    ax.axhline(1.45, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 3.65)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Correspondence (%)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", which="both", length=0, colors=COLORS["annotation"])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["secondary"])
    ax.spines["bottom"].set_linewidth(0.65)
    return fig


def load_map_layers(states):
    """Load frozen tract states and local cartographic context in EPSG:26915."""
    tract_path = ROOT / "data" / "tl_2023_27_tract.zip"
    zcta_path = ROOT / "data" / "tl_2023_us_zcta520.zip"
    context = ROOT / "data" / "cartographic_context" / "tigerline_2023"
    place_path = context / "tl_2023_27_place.zip"
    water_paths = [
        context / "tl_2023_27053_areawater.zip",
        context / "tl_2023_27123_areawater.zip",
    ]
    for path in [tract_path, zcta_path, place_path, *water_paths]:
        require(path.is_file(), f"missing cartographic source: {path}")

    tracts = gpd.read_file(f"zip://{tract_path.as_posix()}")
    require(tracts.crs is not None and tracts.crs.to_epsg() == 4269, "tract CRS must be EPSG:4269")
    require(tracts.geometry.notna().all() and (~tracts.geometry.is_empty).all(), "tract geometry missing or empty")
    require(tracts.geometry.is_valid.all(), "invalid tract geometry")
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    mapped = tracts[["GEOID", "COUNTYFP", "geometry"]].merge(
        states, on="GEOID", how="inner", validate="one_to_one"
    )
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=tracts.crs).to_crs(26915)
    require(len(mapped) == 327 and mapped.geometry.is_valid.all(), "projected map frame invalid")

    county = tracts.loc[
        tracts["COUNTYFP"].isin(["053", "123"]), ["COUNTYFP", "geometry"]
    ].to_crs(26915)
    county = county.dissolve(by="COUNTYFP", as_index=False)
    county_union = county.geometry.union_all()

    zctas = gpd.read_file(f"zip://{zcta_path.as_posix()}")
    require(zctas.crs is not None and zctas.crs.to_epsg() == 4269, "ZCTA CRS must be EPSG:4269")
    zctas["ZCTA5CE20"] = zctas["ZCTA5CE20"].astype(str).str.zfill(5)
    primary_zctas = zctas.loc[
        zctas["ZCTA5CE20"].isin(TARGET_ZIPS), ["ZCTA5CE20", "geometry"]
    ].to_crs(26915)
    require(len(primary_zctas) == 60, "all 60 predefined ZCTAs must be present")
    frame_geometry = primary_zctas.geometry.union_all().intersection(county_union)
    require(not frame_geometry.is_empty and frame_geometry.is_valid, "study-frame geometry invalid")
    study_frame = gpd.GeoDataFrame(
        {"frame": ["60 target ZCTAs clipped to Hennepin/Ramsey"]},
        geometry=[frame_geometry],
        crs=26915,
    )

    places = gpd.read_file(f"zip://{place_path.as_posix()}")
    require(places.crs is not None and places.crs.to_epsg() == 4269, "place CRS must be EPSG:4269")
    cities = places.loc[
        places["NAME"].isin(["Minneapolis", "St. Paul"]),
        ["NAME", "geometry"],
    ].to_crs(26915)
    require(set(cities["NAME"]) == {"Minneapolis", "St. Paul"}, "city context incomplete")

    water_parts = []
    for path in water_paths:
        layer = gpd.read_file(f"zip://{path.as_posix()}")
        require(layer.crs is not None and layer.crs.to_epsg() == 4269, f"water CRS changed: {path.name}")
        require(layer.geometry.is_valid.all(), f"invalid water geometry: {path.name}")
        water_parts.append(layer[["FULLNAME", "geometry"]])
    water = gpd.GeoDataFrame(
        pd.concat(water_parts, ignore_index=True), geometry="geometry", crs=4269
    ).to_crs(26915)
    study_buffer = mapped.geometry.union_all().buffer(5000)
    water = water.loc[water.geometry.intersects(study_buffer)].copy()
    water["area_m2"] = water.geometry.area
    water = water.loc[
        water["FULLNAME"].fillna("").str.contains("Mississippi Riv", case=False)
        | water["area_m2"].ge(500_000)
    ].copy()
    require(len(water) == 38, "cartographic water selection changed")

    bounds = study_frame.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    extent = (
        bounds[0] - 0.035 * width,
        bounds[2] + 0.035 * width,
        bounds[1] - 0.055 * height,
        bounds[3] + 0.035 * height,
    )
    return {
        "mapped": mapped,
        "county": county,
        "study_frame": study_frame,
        "cities": cities,
        "water": water,
        "extent": extent,
        "source_paths": [tract_path, zcta_path, place_path, *water_paths],
    }


def draw_scale_bar(ax, extent):
    """Add one five-mile scale bar in projected metres."""
    length = 5 * 1609.344
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    x0 = extent[0] + 0.055 * width
    y0 = extent[2] + 0.032 * height
    line = ax.plot(
        [x0, x0 + length],
        [y0, y0],
        color=COLORS["text"],
        linewidth=1.6,
        solid_capstyle="butt",
        zorder=12,
    )[0]
    line.set_path_effects(
        [patheffects.Stroke(linewidth=3.0, foreground="white"), patheffects.Normal()]
    )
    for x in (x0, x0 + length):
        ax.plot(
            [x, x],
            [y0 - 0.008 * height, y0 + 0.008 * height],
            color=COLORS["text"],
            linewidth=0.75,
            zorder=12,
        )
    label = ax.text(
        x0 + length / 2,
        y0 + 0.013 * height,
        "5 mi",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=COLORS["text"],
        zorder=12,
    )
    label.set_path_effects(
        [patheffects.Stroke(linewidth=2.2, foreground="white"), patheffects.Normal()]
    )


def draw_north_arrow(ax):
    """Add one restrained grid-north arrow shared by the common-orientation maps."""
    ax.annotate(
        "",
        xy=(0.94, 0.91),
        xytext=(0.94, 0.80),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["text"],
            "linewidth": 0.9,
            "mutation_scale": 8,
        },
        zorder=12,
    )
    ax.text(
        0.94,
        0.94,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["text"],
        zorder=12,
    )


def draw_map_panel(ax, layers, column, label, *, scale=False, north=False, city_labels=True):
    """Draw one common-extent categorical disagreement map."""
    mapped = layers["mapped"].copy()
    mapped["display_state"] = np.where(
        mapped[column].isin(["neither_beyond", "both_beyond"]),
        "agreement",
        np.where(mapped[column].eq("board_only_beyond"), "workflow", "board"),
    )
    require(not mapped["display_state"].isna().any(), f"missing display state: {column}")

    layers["county"].plot(
        ax=ax,
        facecolor=COLORS["context_fill"],
        edgecolor=COLORS["context"],
        linewidth=0.45,
        zorder=0,
    )
    for state in ("agreement", "workflow", "board"):
        subset = mapped.loc[mapped["display_state"].eq(state)]
        if not subset.empty:
            subset.plot(
                ax=ax,
                facecolor=COLORS[state],
                edgecolor="white",
                linewidth=0.24,
                zorder=2,
            )
    layers["water"].plot(
        ax=ax,
        facecolor=COLORS["water"],
        edgecolor="none",
        zorder=3,
    )
    layers["cities"].boundary.plot(
        ax=ax,
        color="#777777",
        linewidth=0.45,
        linestyle=(0, (2, 2)),
        zorder=5,
    )
    layers["study_frame"].boundary.plot(
        ax=ax, color=COLORS["text"], linewidth=0.72, zorder=6
    )
    layers["county"].boundary.plot(
        ax=ax, color=COLORS["context"], linewidth=0.52, zorder=6
    )

    if city_labels:
        for _, row in layers["cities"].iterrows():
            anchor = row.geometry.representative_point()
            text = ax.text(
                anchor.x,
                anchor.y,
                row["NAME"],
                ha="center",
                va="center",
                fontsize=7.8,
                color=COLORS["annotation"],
                zorder=8,
            )
            text.set_path_effects(
                [patheffects.Stroke(linewidth=2.3, foreground="white"), patheffects.Normal()]
            )

    extent = layers["extent"]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(label, loc="left", pad=3, fontsize=9.2, fontweight="normal")
    if scale:
        draw_scale_bar(ax, extent)
    if north:
        draw_north_arrow(ax)


def spatial_mockup(layers):
    """Draw two large threshold maps and a smaller two-mile inset."""
    fig = plt.figure(figsize=MAP_SIZE)
    axes = [
        fig.add_axes([0.018, 0.16, 0.355, 0.76]),
        fig.add_axes([0.365, 0.16, 0.355, 0.76]),
        fig.add_axes([0.755, 0.31, 0.215, 0.48]),
    ]
    draw_map_panel(
        axes[0],
        layers,
        "discordance_type_0_5mi",
        "(a)  0.5 mile",
        scale=True,
    )
    draw_map_panel(
        axes[1],
        layers,
        "discordance_type_1_0mi",
        "(b)  1 mile",
        north=True,
    )
    draw_map_panel(
        axes[2],
        layers,
        "discordance_type_2_0mi",
        "(c)  2 miles",
        city_labels=False,
    )
    handles = [
        Patch(facecolor=COLORS["agreement"], edgecolor="#A8A8A4", linewidth=0.5, label="Same classification"),
        Patch(facecolor=COLORS["workflow"], edgecolor=COLORS["workflow"], label="Within only: workflow"),
        Patch(facecolor=COLORS["board"], edgecolor=COLORS["board"], label="Within only: Board"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.035),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.35,
        labelcolor=COLORS["annotation"],
    )
    return fig


def distance_mockup(frame):
    """Draw a restrained one-panel signed-distance histogram."""
    values = frame["workflow_minus_board_miles"].astype(float)
    lower = math.floor(float(values.min()) * 10) / 10
    upper = math.ceil(float(values.max()) * 10) / 10 + 0.1
    bins = np.arange(lower, upper + 1e-9, 0.1)
    q1, median, q3 = values.quantile([0.25, 0.50, 0.75]).tolist()

    fig = plt.figure(figsize=DISTANCE_SIZE)
    ax = fig.add_axes([0.135, 0.24, 0.825, 0.68])
    ax.axvspan(q1, q3, color=COLORS["iqr"], alpha=0.72, zorder=0)
    ax.hist(
        values,
        bins=bins,
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=0.32,
        zorder=2,
    )
    ax.axvline(
        0,
        color=COLORS["text"],
        linewidth=0.95,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    ax.axvline(median, color=COLORS["workflow"], linewidth=1.25, zorder=5)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Workflow minus Board nearest distance (miles)")
    ax.set_ylabel("Tract origins")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="both", length=0, colors=COLORS["annotation"])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["secondary"])
    ax.spines["bottom"].set_linewidth(0.65)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["workflow"], linewidth=1.25, label="Median"),
            Line2D([0], [0], color=COLORS["text"], linewidth=0.95, linestyle=(0, (3, 2)), label="Zero"),
        ],
        loc="upper left",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
        labelcolor=COLORS["annotation"],
    )
    return fig


def pillow_font_paths():
    """Return regular and bold fonts for Letter-page mockups."""
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    root = local / "TectonicProject" / "Tectonic" / "bundles" / "data"
    regular = list(root.rglob("lmroman10-regular.otf")) if root.is_dir() else []
    bold = list(root.rglob("lmroman10-bold.otf")) if root.is_dir() else []
    if regular and bold:
        return regular[0], bold[0]
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    return windows / "times.ttf", windows / "timesbd.ttf"


def draw_wrapped(draw, text, box, font, fill, line_spacing=1.18):
    """Draw wrapped prose within a pixel box and return the next y position."""
    x, y, width = box
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_height = int(round(font.size * line_spacing))
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def paste_at_inches(page, figure_path, x, y, size):
    """Paste a figure at a common 100-pixel-per-inch physical scale."""
    target = tuple(int(round(value * CONTACT_PPI)) for value in size)
    with Image.open(figure_path) as image:
        scaled = image.convert("RGB").resize(target, Image.Resampling.LANCZOS)
    page.paste(scaled, (int(round(x * CONTACT_PPI)), int(round(y * CONTACT_PPI))))


def page_fonts():
    """Create thesis-like Pillow fonts at the contact sheet's physical scale."""
    regular_path, bold_path = pillow_font_paths()
    px = lambda pt: max(1, int(round(pt * CONTACT_PPI / 72)))
    return {
        "body": ImageFont.truetype(str(regular_path), px(10.5)),
        "caption": ImageFont.truetype(str(regular_path), px(9.2)),
        "heading": ImageFont.truetype(str(bold_path), px(14.0)),
        "small": ImageFont.truetype(str(regular_path), px(8.5)),
    }


def portrait_recurrence_page(path):
    """Compose the forest plot at intended size on a Letter portrait page."""
    page = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(page)
    fonts = page_fonts()
    x, width = 176, 498
    y = 92
    draw.text((x, y), "Name recurrence", font=fonts["heading"], fill=COLORS["text"])
    y += 42
    y = draw_wrapped(
        draw,
        "Correspondence was higher for repeated-name components than for single-occurrence components in both inventories.",
        (x, y, width),
        fonts["body"],
        COLORS["text"],
    )
    y += 18
    paste_at_inches(page, path, 1.76, y / CONTACT_PPI, FOREST_SIZE)
    y += int(round(FOREST_SIZE[1] * CONTACT_PPI)) + 18
    y = draw_wrapped(
        draw,
        "Figure 4.4. Primary correspondence by normalized-name recurrence. Points show workflow linked share or Board-reference coverage; horizontal lines show 95% component-level percentile bootstrap intervals from 5,000 resamples. One recurrence-unresolved workflow row is not plotted. Name recurrence does not establish ownership.",
        (x, y, width),
        fonts["caption"],
        COLORS["text"],
        line_spacing=1.16,
    )
    y += 24
    draw_wrapped(
        draw,
        "The predefined recurrence cutoffs produced the same ordering at the three-address and five-address sensitivities.",
        (x, y, width),
        fonts["body"],
        COLORS["text"],
    )
    draw.text((421, 1035), "33", font=fonts["small"], fill=COLORS["text"])
    return page


def landscape_map_page(path):
    """Compose the map on a dedicated Letter landscape page."""
    page = Image.new("RGB", (1100, 850), "white")
    draw = ImageDraw.Draw(page)
    fonts = page_fonts()
    x, width = 176, 749
    paste_at_inches(page, path, 1.755, 0.76, MAP_SIZE)
    y = 0.76 * CONTACT_PPI + MAP_SIZE[1] * CONTACT_PPI + 18
    draw_wrapped(
        draw,
        "Figure 4.5. Tracts whose 0.5-, 1-, or 2-mile potential-proximity classification differs between inventories. Gray tracts have the same classification; blue tracts are within the threshold only under the workflow inventory, and orange tracts only under the Board inventory. Panels use the same extent and EPSG:26915 projection. Boundaries and water: U.S. Census Bureau, 2023 TIGER/Line.",
        (x, int(y), width),
        fonts["caption"],
        COLORS["text"],
        line_spacing=1.16,
    )
    draw.text((546, 800), "36", font=fonts["small"], fill=COLORS["text"])
    return page


def portrait_distance_page(path):
    """Compose the histogram at intended size on a Letter portrait page."""
    page = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(page)
    fonts = page_fonts()
    x, width = 176, 498
    y = 92
    draw.text(
        (x, y),
        "Nearest distances and paired classifications",
        font=fonts["heading"],
        fill=COLORS["text"],
    )
    y += 44
    y = draw_wrapped(
        draw,
        "Across 327 tract origins, the signed workflow-minus-Board nearest-distance difference was concentrated near zero with observations on both sides.",
        (x, y, width),
        fonts["body"],
        COLORS["text"],
    )
    y += 18
    paste_at_inches(page, path, 1.76, y / CONTACT_PPI, DISTANCE_SIZE)
    y += int(round(DISTANCE_SIZE[1] * CONTACT_PPI)) + 18
    y = draw_wrapped(
        draw,
        "Figure 4.6. Distribution of workflow-minus-Board nearest-distance differences across 327 tract origins. Bars use fixed 0.1-mile bins. The solid blue line marks the median (−0.009 miles), the gray band the interquartile range (−0.078 to 0.022), and the dashed line zero; the observed range was −1.909 to 1.044 miles.",
        (x, y, width),
        fonts["caption"],
        COLORS["text"],
        line_spacing=1.16,
    )
    y += 26
    draw.text((x, y), "Neighborhood-quartile patterns", font=fonts["heading"], fill=COLORS["text"])
    y += 42
    draw_wrapped(
        draw,
        "Mean nearest distances and classification-change rates varied across the predefined income and non-White quartiles.",
        (x, y, width),
        fonts["body"],
        COLORS["text"],
    )
    draw.text((421, 1035), "38", font=fonts["small"], fill=COLORS["text"])
    return page


def contact_sheet(figure_paths):
    """Place three Letter pages on one sheet at one common physical scale."""
    recurrence_page = portrait_recurrence_page(figure_paths["recurrence"])
    map_page = landscape_map_page(figure_paths["map"])
    distance_page = portrait_distance_page(figure_paths["distance"])
    gap = 24
    top = 40
    width = 850 + 1100 + 850 + gap * 4
    height = 1100 + top + 30
    sheet = Image.new("RGB", (width, height), "#D9D9D7")
    draw = ImageDraw.Draw(sheet)
    regular_path, _ = pillow_font_paths()
    label_font = ImageFont.truetype(str(regular_path), 15)
    positions = [
        (recurrence_page, gap, top, "Portrait: recurrence"),
        (map_page, 850 + gap * 2, top + 125, "Landscape: spatial centerpiece"),
        (distance_page, 850 + 1100 + gap * 3, top, "Portrait: distance distribution"),
    ]
    for page, x, y, label in positions:
        draw.text((x, 10), label, font=label_font, fill=COLORS["text"])
        sheet.paste(page, (x, y))
    path = OUTPUT / "letter_page_contact_sheet.png"
    require(not path.exists(), f"refusing to overwrite {path}")
    sheet.save(path, dpi=(CONTACT_PPI, CONTACT_PPI), optimize=True)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "pixels": list(sheet.size),
        "pixels_per_inch": CONTACT_PPI,
        "scale_rule": "Every page and embedded figure uses 100 contact-sheet pixels per physical inch.",
    }


def main():
    """Validate frozen display data and build the three requested draft concepts."""
    require(OUTPUT.is_dir(), f"output directory must already exist: {OUTPUT}")
    require(MOCKUPS.is_dir(), f"mockup directory must already exist: {MOCKUPS}")
    require((OUTPUT / "figure_triage.md").is_file(), "figure triage must precede drawing")
    require((OUTPUT / "paper_figure_benchmark.md").is_file(), "paper benchmark must precede drawing")

    family, registered_fonts = configure_thesis_font()
    recurrence, states, distance, data_paths = load_frozen_tables()
    layers = load_map_layers(states)

    records = {}
    records["recurrence"] = save_mockup(
        recurrence_mockup(recurrence),
        "name_recurrence_forest_mockup.png",
        FOREST_SIZE,
    )
    records["map"] = save_mockup(
        spatial_mockup(layers),
        "spatial_disagreement_mockup.png",
        MAP_SIZE,
    )
    records["distance"] = save_mockup(
        distance_mockup(distance),
        "signed_distance_distribution_mockup.png",
        DISTANCE_SIZE,
    )
    contact = contact_sheet(
        {key: ROOT / record["path"] for key, record in records.items()}
    )

    manifest = {
        "schema_version": "nets-ai-editorial-mockups-v1",
        "status": "draft_mockups_for_user_approval",
        "canonical_results_run": RUN_ID,
        "scientific_results_frozen": True,
        "analysis_rerun": False,
        "provider_or_model_calls": False,
        "thesis_integrated": False,
        "thesis_files_modified": False,
        "draft_dpi": DRAFT_DPI,
        "figure_font_family": family,
        "registered_latin_modern_files": [str(path) for path in registered_fonts],
        "semantic_palette": COLORS,
        "source_data": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in data_paths.items()
        },
        "cartographic_sources": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in layers["source_paths"]
        ],
        "map_contract": {
            "display_crs": "EPSG:26915 (NAD83 / UTM zone 15N; metres)",
            "common_extent_m": [float(value) for value in layers["extent"]],
            "equal_aspect": True,
            "eligible_tracts": 327,
            "agreement_states_merged_for_display": ["both within", "both beyond"],
            "scale_bar_miles": 5,
            "north_arrow": "one shared grid-north cue",
            "target_zcta_count": 60,
            "water_feature_count": 38,
            "roads": "omitted; no vetted local road layer was needed for orientation",
            "remote_tiles": False,
        },
        "mockups": records,
        "contact_sheet": contact,
        "approval_gate": "Stop before thesis integration or final-resolution export.",
    }
    manifest_path = OUTPUT / "mockup_manifest.json"
    require(not manifest_path.exists(), f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mockups": records, "contact_sheet": contact}, indent=2))


if __name__ == "__main__":
    main()
