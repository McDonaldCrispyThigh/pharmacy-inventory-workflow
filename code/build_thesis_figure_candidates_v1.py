"""Build frozen-data figure candidates for review without editing the thesis.

Scientific inputs are read only from the retained visual-redesign figure-data
exports and checked against the final forced-choice Results audit before drawing.
No provider, model, classification, linkage, or spatial-analysis workflow is run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import fitz
import geopandas as gpd
import matplotlib
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/v2/final_results_20260902T012636Z_visual_redesign_v3"
AUDIT_PATH = ROOT / "analysis/v2/results_audit_20260902T012636Z_forced_choice_final/results_audit.json"
OUT = ROOT / "analysis/v2/final_results_20260902T012636Z_figure_rebuild_v1"
CANDIDATES = OUT / "candidates"
APPENDIX = OUT / "appendix_candidates"
QA = OUT / "qa"
TARGET_CRS = "EPSG:26915"
PNG_DPI = 300

COLORS = {
    "workflow": "#315F7A",
    "board": "#C9822B",
    "neutral_bar": "#777C7E",
    "neutral_mark": "#4F5557",
    "text": "#292B2C",
    "muted_text": "#55595B",
    "axis": "#4A4D4E",
    "grid": "#DDDFDF",
    "agreement": "#FAFAF7",
    "outside": "#E5E7E5",
    "tract_edge": "#C7C9C8",
    "county_edge": "#686C6D",
    "city_edge": "#777B7C",
    "water": "#D9ECF2",
    "road": "#9A9D9D",
    "white": "#FFFFFF",
}

CANVAS_INCHES = {
    "figure1_version_a_two_panel": (6.20, 2.35),
    "figure1_version_b_grouped_forest": (6.20, 2.50),
    "figure2_version_a_neutral_histogram": (6.20, 2.90),
    "figure2_version_b_signed_histogram": (6.20, 2.90),
    "figure2_diagnostic_ecdf": (6.20, 2.90),
    "figure3_version_a_no_roads": (8.58, 4.51),
    "figure3_version_b_primary_highways": (8.58, 4.51),
    "figure3_appendix_2_miles": (6.20, 4.65),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.25,
            "ytick.labelsize": 8.25,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_and_assert_frozen_values() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    recurrence_path = SOURCE / "figure_data/figure04_name_recurrence.csv"
    states_path = SOURCE / "figure_data/figure05_tract_states.csv"
    distances_path = SOURCE / "figure_data/figure06_signed_distances.csv"
    for path in [recurrence_path, states_path, distances_path, AUDIT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    recurrence = pd.read_csv(recurrence_path)
    states = pd.read_csv(states_path, dtype={"GEOID": str})
    distances = pd.read_csv(distances_path, dtype={"GEOID": str})
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    if audit["run_id"] != "20260902T012636Z":
        raise AssertionError(f"Unexpected Results audit run: {audit['run_id']}")

    compare_columns = [
        "source",
        "group_code",
        "records",
        "linked_records",
        "linked_pct",
        "cluster_bootstrap_95_low_pct",
        "cluster_bootstrap_95_high_pct",
    ]
    csv_primary = recurrence.loc[
        (recurrence["recurrence_threshold"] == 2)
        & recurrence["group_code"].isin(["repeated", "below_threshold"]),
        compare_columns,
    ].sort_values(["source", "group_code"]).reset_index(drop=True)
    audit_primary = pd.DataFrame(audit["rq2_name_recurrence"]["primary_rows"])
    audit_primary = audit_primary.loc[
        (audit_primary["recurrence_threshold"] == 2)
        & audit_primary["group_code"].isin(["repeated", "below_threshold"]),
        compare_columns,
    ].sort_values(["source", "group_code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(csv_primary, audit_primary, check_dtype=False)

    expected_recurrence = {
        ("Workflow inventory locations", "repeated"): (222, 121, 54.50, 38.97, 66.96),
        ("Workflow inventory locations", "below_threshold"): (116, 25, 21.55, 14.53, 29.31),
        ("Board keyword-screen reference", "repeated"): (131, 122, 93.13, 90.36, 94.52),
        ("Board keyword-screen reference", "below_threshold"): (47, 24, 51.06, 36.17, 65.96),
    }
    for key, expected in expected_recurrence.items():
        row = csv_primary.loc[(csv_primary["source"] == key[0]) & (csv_primary["group_code"] == key[1])].iloc[0]
        actual = tuple(float(row[column]) for column in compare_columns[2:])
        if not np.allclose(actual, expected, atol=1e-9):
            raise AssertionError(f"Recurrence mismatch for {key}: {actual} != {expected}")

    unresolved = recurrence.loc[
        (recurrence["recurrence_threshold"] == 2) & (recurrence["group_code"] == "unresolved")
    ]
    workflow_unresolved = int(
        unresolved.loc[unresolved["source"] == "Workflow inventory locations", "records"].sum()
    )
    board_unresolved = int(
        unresolved.loc[unresolved["source"] == "Board keyword-screen reference", "records"].sum()
    )
    if (workflow_unresolved, board_unresolved) != (1, 0):
        raise AssertionError("Unexpected recurrence-unresolved counts")

    values = distances["workflow_minus_board_miles"].to_numpy(float)
    summary = {
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "median": round(float(np.median(values)), 3),
        "q1": round(float(np.quantile(values, 0.25)), 3),
        "q3": round(float(np.quantile(values, 0.75)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
        "google_closer": int((values < 0).sum()),
        "google_farther": int((values > 0).sum()),
        "equal": int((values == 0).sum()),
    }
    audit_distance = audit["rq3_spatial"]["signed_google_minus_board_miles"]
    expected_distance = {
        "n": audit_distance["n"],
        "mean": audit_distance["mean"],
        "median": audit_distance["median"],
        "q1": audit_distance["q1"],
        "q3": audit_distance["q3"],
        "min": audit_distance["min"],
        "max": audit_distance["max"],
        "google_closer": audit_distance["sign_distribution"]["google_closer"]["count"],
        "google_farther": audit_distance["sign_distribution"]["google_farther"]["count"],
        "equal": audit_distance["sign_distribution"]["equal"]["count"],
    }
    if summary != expected_distance:
        raise AssertionError(f"Distance summary mismatch: {summary} != {expected_distance}")

    threshold_columns = {"0.5": "discordance_type_0_5mi", "1.0": "discordance_type_1_0mi", "2.0": "discordance_type_2_0mi"}
    threshold_assertions = {}
    for threshold, column in threshold_columns.items():
        scenario = audit["rq3_spatial"]["thresholds"][threshold]["scenarios"][
            "primary_google_include_vs_board_screened"
        ]
        audit_counts = {str(key): int(value["count"]) for key, value in scenario["states"].items()}
        counts = {key: int(states[column].eq(key).sum()) for key in audit_counts}
        if counts != audit_counts:
            raise AssertionError(f"State mismatch at {threshold}: {counts} != {audit_counts}")
        changed = counts.get("board_only_beyond", 0) + counts.get("google_only_beyond", 0)
        expected_changed = int(scenario["classification_changed"]["count"])
        if changed != expected_changed:
            raise AssertionError(f"Changed-count mismatch at {threshold}")
        threshold_assertions[threshold] = {
            "counts": counts,
            "changed_count": changed,
            "changed_pct": float(scenario["classification_changed"]["pct"]),
        }

    if len(states) != 327 or not states["GEOID"].is_unique:
        raise AssertionError("Frozen tract-state export is not 327 unique GEOIDs")
    if len(distances) != 327 or not distances["GEOID"].is_unique:
        raise AssertionError("Frozen signed-distance export is not 327 unique GEOIDs")
    if set(states["GEOID"]) != set(distances["GEOID"]):
        raise AssertionError("Tract GEOID frames differ between frozen exports")

    assertion_report = {
        "status": "pass",
        "scope": "frozen figure-data values checked against final Results audit before plotting",
        "audit_path": AUDIT_PATH.relative_to(ROOT).as_posix(),
        "audit_sha256": sha256(AUDIT_PATH),
        "inputs": {
            recurrence_path.relative_to(ROOT).as_posix(): sha256(recurrence_path),
            states_path.relative_to(ROOT).as_posix(): sha256(states_path),
            distances_path.relative_to(ROOT).as_posix(): sha256(distances_path),
        },
        "recurrence": {
            "rows": {f"{key[0]} | {key[1]}": list(value) for key, value in expected_recurrence.items()},
            "workflow_unresolved": workflow_unresolved,
            "board_unresolved": board_unresolved,
        },
        "signed_distance": summary,
        "thresholds": threshold_assertions,
        "provider_calls": False,
        "model_calls": False,
        "scientific_values_changed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_value_assertions.json").write_text(
        json.dumps(assertion_report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return recurrence, states, distances, assertion_report


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def export_figure(fig: plt.Figure, output_dir: Path, name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output_dir / f"{name}.{suffix}" for suffix in ["svg", "pdf", "png"]}
    fig.savefig(paths["svg"], format="svg", facecolor="white")
    fig.savefig(paths["pdf"], format="pdf", facecolor="white")
    fig.savefig(paths["png"], format="png", dpi=PNG_DPI, facecolor="white")
    plt.close(fig)

    expected_w, expected_h = CANVAS_INCHES[name]
    with Image.open(paths["png"]) as image:
        if image.mode != "RGB":
            rgb = image.convert("RGB")
            rgb.save(paths["png"], dpi=(PNG_DPI, PNG_DPI))
        pixels = [image.width, image.height]
    expected_pixels = [round(expected_w * PNG_DPI), round(expected_h * PNG_DPI)]
    if pixels != expected_pixels:
        raise AssertionError(f"Unexpected PNG canvas for {name}: {pixels} != {expected_pixels}")
    with fitz.open(paths["pdf"]) as doc:
        rect = doc[0].rect
        pdf_inches = [rect.width / 72.0, rect.height / 72.0]
        pdf_text = doc[0].get_text()
    if not np.allclose(pdf_inches, [expected_w, expected_h], atol=0.002):
        raise AssertionError(f"Unexpected PDF canvas for {name}: {pdf_inches}")
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise AssertionError(f"SVG text is not editable for {name}")
    if not pdf_text.strip():
        raise AssertionError(f"PDF text extraction failed for {name}")
    return {
        "name": name,
        "canvas_inches": {"width": expected_w, "height": expected_h},
        "png_dpi": PNG_DPI,
        "png_pixels": pixels,
        "formats": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
        "svg_editable_text": True,
        "pdf_text_extractable": True,
    }


def _recurrence_rows(recurrence: pd.DataFrame, source: str) -> pd.DataFrame:
    order = pd.CategoricalDtype(["repeated", "below_threshold"], ordered=True)
    rows = recurrence.loc[
        (recurrence["recurrence_threshold"] == 2)
        & (recurrence["source"] == source)
        & recurrence["group_code"].isin(["repeated", "below_threshold"])
    ].copy()
    rows["group_code"] = rows["group_code"].astype(order)
    return rows.sort_values("group_code")


def _draw_interval_rows(ax: plt.Axes, rows: pd.DataFrame, color: str, show_ylabels: bool) -> None:
    y = np.array([1.0, 0.0])
    x = rows["linked_pct"].to_numpy(float)
    low = rows["cluster_bootstrap_95_low_pct"].to_numpy(float)
    high = rows["cluster_bootstrap_95_high_pct"].to_numpy(float)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - low, high - x]),
        fmt="o",
        markersize=5.0,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        ecolor=color,
        elinewidth=1.25,
        capsize=3.1,
        capthick=1.1,
        zorder=3,
    )
    for point, ypos in zip(x, y):
        offset = -1.6 if point > 89 else 1.8
        align = "right" if point > 89 else "left"
        ax.text(point + offset, ypos + 0.18, f"{point:.1f}%", ha=align, va="bottom", fontsize=8.25, color=COLORS["muted_text"])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(["Repeated name", "Single occurrence"])
    else:
        ax.tick_params(axis="y", labelleft=False)
    clean_axis(ax)


def figure1_version_a(recurrence: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=CANVAS_INCHES["figure1_version_a_two_panel"],
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.17, "left": 0.19, "right": 0.985, "top": 0.80, "bottom": 0.25},
    )
    specs = [
        ("Workflow inventory locations", COLORS["neutral_mark"], "(a)  Workflow linked share"),
        ("Board keyword-screen reference", COLORS["neutral_mark"], "(b)  Board-reference coverage"),
    ]
    for index, (ax, (source, color, title)) in enumerate(zip(axes, specs)):
        _draw_interval_rows(ax, _recurrence_rows(recurrence, source), color, show_ylabels=index == 0)
        ax.set_title(title, loc="left", pad=9, fontsize=9.5, fontweight="semibold")
    fig.supxlabel("Locations (%)", fontsize=9.0, y=0.055)
    return fig


def figure1_version_b(recurrence: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=CANVAS_INCHES["figure1_version_b_grouped_forest"])
    fig.subplots_adjust(left=0.35, right=0.985, top=0.91, bottom=0.20)
    groups = [
        ("Workflow inventory locations", COLORS["workflow"], [3.15, 2.25], "Workflow linked share", 3.72),
        ("Board keyword-screen reference", COLORS["board"], [0.95, 0.05], "Board-reference coverage", 1.52),
    ]
    y_ticks, y_labels = [], []
    for source, color, y, heading, heading_y in groups:
        rows = _recurrence_rows(recurrence, source)
        x = rows["linked_pct"].to_numpy(float)
        low = rows["cluster_bootstrap_95_low_pct"].to_numpy(float)
        high = rows["cluster_bootstrap_95_high_pct"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            xerr=np.vstack([x - low, high - x]),
            fmt="o",
            markersize=5.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            ecolor=color,
            elinewidth=1.25,
            capsize=3.1,
            capthick=1.1,
            zorder=3,
        )
        for point, ypos in zip(x, y):
            offset = -1.6 if point > 89 else 1.8
            align = "right" if point > 89 else "left"
            ax.text(point + offset, ypos + 0.16, f"{point:.1f}%", ha=align, va="bottom", fontsize=8.25, color=COLORS["muted_text"])
        ax.text(-0.31, heading_y, heading, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.0, fontweight="semibold", color=color)
        y_ticks.extend(y)
        y_labels.extend(["Repeated name", "Single occurrence"])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.45, 4.0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Locations (%)")
    clean_axis(ax)
    return fig


def _histogram(distances: pd.DataFrame, signed_color: bool) -> plt.Figure:
    name = "figure2_version_b_signed_histogram" if signed_color else "figure2_version_a_neutral_histogram"
    fig, ax = plt.subplots(figsize=CANVAS_INCHES[name])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.93, bottom=0.25)
    values = distances["workflow_minus_board_miles"].to_numpy(float)
    start = math.floor(values.min() * 10) / 10
    stop = math.ceil(values.max() * 10) / 10
    bins = np.arange(start, stop + 0.10001, 0.1)
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    if signed_color:
        colors = [COLORS["workflow"] if (left + right) / 2 < 0 else COLORS["board"] for left, right in zip(edges[:-1], edges[1:])]
    else:
        colors = [COLORS["neutral_bar"]] * len(counts)
    ax.bar(edges[:-1], counts, width=widths, align="edge", color=colors, edgecolor="white", linewidth=0.45)
    ax.axvline(0, color=COLORS["text"], linewidth=0.7, zorder=4)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(counts) * 1.08)
    ax.set_xlabel("Difference in nearest-pharmacy distance (workflow − Board), miles", labelpad=7)
    ax.set_ylabel("Number of tracts", labelpad=7)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def figure2_ecdf(distances: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=CANVAS_INCHES["figure2_diagnostic_ecdf"])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.93, bottom=0.25)
    values = np.sort(distances["workflow_minus_board_miles"].to_numpy(float))
    y = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, y, where="post", color=COLORS["neutral_bar"], linewidth=1.25)
    ax.axvline(0, color=COLORS["text"], linewidth=0.7)
    ax.set_xlim(math.floor(values.min() * 10) / 10, math.ceil(values.max() * 10) / 10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Difference in nearest-pharmacy distance (workflow − Board), miles", labelpad=7)
    ax.set_ylabel("Cumulative share of tracts", labelpad=7)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def load_map_context(states: pd.DataFrame) -> dict:
    tracts = gpd.read_file(ROOT / "data/tl_2023_27_tract.zip")
    tracts = tracts.loc[(tracts["STATEFP"] == "27") & tracts["COUNTYFP"].isin(["053", "123"])].copy().to_crs(TARGET_CRS)
    if len(tracts) != 472:
        raise AssertionError(f"Unexpected two-county tract count: {len(tracts)}")
    state_gdf = tracts.merge(states, on="GEOID", how="left", validate="one_to_one", indicator=True)
    if int((state_gdf["_merge"] == "both").sum()) != 327:
        raise AssertionError("Not all 327 frozen tract states joined exactly once")
    state_gdf["analysis_frame"] = state_gdf["_merge"].eq("both")
    state_gdf = state_gdf.drop(columns="_merge")

    county_outlines = tracts.dissolve(by="COUNTYFP")
    places = gpd.read_file(ROOT / "data/cartographic_context/tigerline_2023/tl_2023_27_place.zip").to_crs(TARGET_CRS)
    places = places.loc[places["NAME"].isin(["Minneapolis", "St. Paul"])].copy()
    if set(places["NAME"]) != {"Minneapolis", "St. Paul"}:
        raise AssertionError("Expected Minneapolis and St. Paul place features")

    water_frames = [
        gpd.read_file(ROOT / "data/cartographic_context/tigerline_2023/tl_2023_27053_areawater.zip"),
        gpd.read_file(ROOT / "data/cartographic_context/tigerline_2023/tl_2023_27123_areawater.zip"),
    ]
    water = gpd.GeoDataFrame(pd.concat(water_frames, ignore_index=True), geometry="geometry", crs="EPSG:4269").to_crs(TARGET_CRS)
    water["area_km2"] = water.geometry.area / 1_000_000
    water_names = water["FULLNAME"].fillna("")
    water = water.loc[(water["area_km2"] >= 0.20) | water_names.str.contains("Mississippi", case=False)].copy()
    county_union = county_outlines.geometry.union_all()
    water = gpd.clip(water, county_union)

    roads = gpd.read_file(ROOT / "data/cartographic_context/tigerline_2023/tl_2023_27_prisecroads.zip").to_crs(TARGET_CRS)
    roads = roads.loc[roads["MTFCC"] == "S1100"].copy()
    roads = gpd.clip(roads, county_union)
    roads = roads.loc[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    minx, miny, maxx, maxy = tracts.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.025, (maxy - miny) * 0.035
    extent = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)
    return {
        "tracts": state_gdf,
        "counties": county_outlines,
        "places": places,
        "water": water,
        "roads": roads,
        "extent": extent,
    }


def _map_display_class(series: pd.Series, analysis_frame: pd.Series) -> pd.Series:
    display = pd.Series("Outside analyzed frame", index=series.index, dtype="object")
    display.loc[analysis_frame & series.isin(["both_beyond", "neither_beyond"])] = "Same classification"
    display.loc[analysis_frame & series.eq("board_only_beyond")] = "Within threshold only: workflow"
    display.loc[analysis_frame & series.eq("google_only_beyond")] = "Within threshold only: Board"
    return display


def _plot_map_panel(ax: plt.Axes, context: dict, column: str, title: str, roads: bool, scale_bar: bool) -> None:
    tracts = context["tracts"].copy()
    tracts["display_class"] = _map_display_class(tracts[column], tracts["analysis_frame"])
    order = [
        "Outside analyzed frame",
        "Same classification",
        "Within threshold only: workflow",
        "Within threshold only: Board",
    ]
    fills = {
        "Outside analyzed frame": COLORS["outside"],
        "Same classification": COLORS["agreement"],
        "Within threshold only: workflow": COLORS["workflow"],
        "Within threshold only: Board": COLORS["board"],
    }
    for category in order:
        subset = tracts.loc[tracts["display_class"] == category]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            facecolor=fills[category],
            edgecolor=COLORS["tract_edge"],
            linewidth=0.18 if category == "Outside analyzed frame" else 0.24,
            zorder=1,
        )
    context["water"].plot(ax=ax, facecolor=COLORS["water"], edgecolor="none", zorder=2)
    if roads:
        context["roads"].plot(ax=ax, color=COLORS["road"], linewidth=0.28, alpha=0.48, zorder=3)
    context["counties"].boundary.plot(ax=ax, color=COLORS["county_edge"], linewidth=0.72, zorder=5)
    manual_label_positions = {
        # Chosen by inspection inside agreement tracts at both displayed thresholds.
        "Minneapolis": (478_000, 4_984_000),
        "St. Paul": (493_500, 4_979_000),
    }
    for name, (label_x, label_y) in manual_label_positions.items():
        ax.text(
            label_x,
            label_y,
            name,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="semibold",
            color=COLORS["text"],
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
            zorder=6,
        )
    minx, maxx, miny, maxy = context["extent"]
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=6, fontsize=9.5, fontweight="semibold", linespacing=1.25)
    if scale_bar:
        length = 5 * 1609.344
        x0 = minx + 0.055 * (maxx - minx)
        y0 = miny + 0.052 * (maxy - miny)
        ax.plot([x0, x0 + length], [y0, y0], color=COLORS["text"], linewidth=1.25, zorder=7)
        cap = 0.009 * (maxy - miny)
        ax.plot([x0, x0], [y0 - cap, y0 + cap], color=COLORS["text"], linewidth=0.9, zorder=7)
        ax.plot([x0 + length, x0 + length], [y0 - cap, y0 + cap], color=COLORS["text"], linewidth=0.9, zorder=7)
        ax.text(x0 + length / 2, y0 + 1.7 * cap, "5 miles", ha="center", va="bottom", fontsize=8.0, color=COLORS["text"], zorder=7)


def _map_legend(fig: plt.Figure, y: float) -> None:
    handles = [
        Patch(facecolor=COLORS["agreement"], edgecolor=COLORS["tract_edge"], linewidth=0.6, label="Same classification"),
        Patch(facecolor=COLORS["workflow"], edgecolor="none", label="Within threshold only: workflow"),
        Patch(facecolor=COLORS["board"], edgecolor="none", label="Within threshold only: Board"),
        Patch(facecolor=COLORS["outside"], edgecolor=COLORS["tract_edge"], linewidth=0.5, label="Outside analyzed frame"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, frameon=False, handlelength=1.25, handleheight=0.8, columnspacing=1.35, fontsize=8.0)


def figure3_two_panel(context: dict, roads: bool) -> plt.Figure:
    name = "figure3_version_b_primary_highways" if roads else "figure3_version_a_no_roads"
    fig, axes = plt.subplots(1, 2, figsize=CANVAS_INCHES[name])
    fig.subplots_adjust(left=0.018, right=0.988, top=0.90, bottom=0.13, wspace=0.045)
    _plot_map_panel(axes[0], context, "discordance_type_0_5mi", "(a)  0.5-mile threshold\n14.1% changed (46 tracts)", roads=roads, scale_bar=True)
    _plot_map_panel(axes[1], context, "discordance_type_1_0mi", "(b)  1-mile threshold\n12.8% changed (42 tracts)", roads=roads, scale_bar=False)
    _map_legend(fig, y=0.025)
    return fig


def figure3_appendix(context: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=CANVAS_INCHES["figure3_appendix_2_miles"])
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.16)
    _plot_map_panel(ax, context, "discordance_type_2_0mi", "2-mile threshold\n1.2% changed (4 tracts)", roads=False, scale_bar=True)
    _map_legend(fig, y=0.035)
    return fig


def simulate_accessibility(candidate_records: list[dict]) -> list[dict]:
    matrices = {
        "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
        "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    }
    outputs = []
    for record in candidate_records:
        source = ROOT / record["formats"]["png"]["path"]
        with Image.open(source) as opened:
            base = opened.convert("RGB")
            for mode in ["grayscale", *matrices]:
                mode_dir = OUT / "accessibility" / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                output = mode_dir / source.name
                if mode == "grayscale":
                    transformed = ImageOps.grayscale(base).convert("RGB")
                else:
                    array = np.asarray(base, dtype=float) / 255.0
                    transformed_array = np.clip(array @ matrices[mode].T, 0, 1)
                    transformed = Image.fromarray(np.round(transformed_array * 255).astype(np.uint8), mode="RGB")
                transformed.save(output, dpi=(PNG_DPI, PNG_DPI))
                outputs.append({"candidate": record["name"], "mode": mode, "path": output.relative_to(ROOT).as_posix(), "sha256": sha256(output)})
    return outputs


def _pil_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _physical_tile(path: Path, inches: tuple[float, float], ppi: int) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    target = (round(inches[0] * ppi), round(inches[1] * ppi))
    return image.resize(target, Image.Resampling.LANCZOS)


def build_figure_contact_sheet() -> Path:
    ppi = 100
    old_root = ROOT / "analysis/v2/final_results_20260902T012636Z_editorial_mockups_v3/mockups"
    rows = [
        (
            "Figure 1 — recurrence",
            [
                ("Rejected mockup", old_root / "name_recurrence_forest_mockup.png", (4.981, 2.35)),
                ("Version A", CANDIDATES / "figure1_version_a_two_panel.png", CANVAS_INCHES["figure1_version_a_two_panel"]),
                ("Version B", CANDIDATES / "figure1_version_b_grouped_forest.png", CANVAS_INCHES["figure1_version_b_grouped_forest"]),
            ],
        ),
        (
            "Figure 2 — signed distance",
            [
                ("Rejected mockup", old_root / "signed_distance_distribution_mockup.png", (4.981, 2.65)),
                ("Version A", CANDIDATES / "figure2_version_a_neutral_histogram.png", CANVAS_INCHES["figure2_version_a_neutral_histogram"]),
                ("Version B", CANDIDATES / "figure2_version_b_signed_histogram.png", CANVAS_INCHES["figure2_version_b_signed_histogram"]),
                ("ECDF diagnostic", CANDIDATES / "figure2_diagnostic_ecdf.png", CANVAS_INCHES["figure2_diagnostic_ecdf"]),
            ],
        ),
        (
            "Figure 3 — spatial disagreement",
            [
                ("Rejected mockup", old_root / "spatial_disagreement_mockup.png", (7.492, 4.60)),
                ("Version A", CANDIDATES / "figure3_version_a_no_roads.png", CANVAS_INCHES["figure3_version_a_no_roads"]),
                ("Version B", CANDIDATES / "figure3_version_b_primary_highways.png", CANVAS_INCHES["figure3_version_b_primary_highways"]),
                ("2-mile appendix", APPENDIX / "figure3_appendix_2_miles.png", CANVAS_INCHES["figure3_appendix_2_miles"]),
            ],
        ),
    ]
    gutter, label_h, row_title_h = 22, 28, 34
    rendered_rows = []
    max_width = 0
    for row_title, items in rows:
        tiles = []
        for label, path, inches in items:
            tile = _physical_tile(path, inches, ppi)
            card = Image.new("RGB", (tile.width, tile.height + label_h), "white")
            card.paste(tile, (0, label_h))
            ImageDraw.Draw(card).text((4, 4), label, fill="#222222", font=_pil_font(17, bold=True))
            tiles.append(card)
        width = gutter + sum(tile.width + gutter for tile in tiles)
        height = row_title_h + max(tile.height for tile in tiles) + gutter
        strip = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(strip)
        draw.text((gutter, 4), row_title, fill="#222222", font=_pil_font(20, bold=True))
        x = gutter
        for tile in tiles:
            strip.paste(tile, (x, row_title_h))
            draw.rectangle((x, row_title_h + label_h, x + tile.width - 1, row_title_h + tile.height - 1), outline="#B5B7B7", width=1)
            x += tile.width + gutter
        rendered_rows.append(strip)
        max_width = max(max_width, width)
    canvas = Image.new("RGB", (max_width, sum(row.height for row in rendered_rows)), "white")
    y = 0
    for row in rendered_rows:
        canvas.paste(row, (0, y))
        y += row.height
    QA.mkdir(parents=True, exist_ok=True)
    output = QA / "figure_candidate_contact_sheet_equal_physical_scale.png"
    canvas.save(output, dpi=(ppi, ppi))
    return output


def main() -> None:
    set_style()
    recurrence, states, distances, assertion_report = load_and_assert_frozen_values()
    records = []
    records.append(export_figure(figure1_version_a(recurrence), CANDIDATES, "figure1_version_a_two_panel"))
    records.append(export_figure(figure1_version_b(recurrence), CANDIDATES, "figure1_version_b_grouped_forest"))
    records.append(export_figure(_histogram(distances, signed_color=False), CANDIDATES, "figure2_version_a_neutral_histogram"))
    records.append(export_figure(_histogram(distances, signed_color=True), CANDIDATES, "figure2_version_b_signed_histogram"))
    records.append(export_figure(figure2_ecdf(distances), CANDIDATES, "figure2_diagnostic_ecdf"))
    context = load_map_context(states)
    records.append(export_figure(figure3_two_panel(context, roads=False), CANDIDATES, "figure3_version_a_no_roads"))
    records.append(export_figure(figure3_two_panel(context, roads=True), CANDIDATES, "figure3_version_b_primary_highways"))
    records.append(export_figure(figure3_appendix(context), APPENDIX, "figure3_appendix_2_miles"))
    accessibility = simulate_accessibility(records)
    contact = build_figure_contact_sheet()
    manifest = {
        "schema_version": "nets-ai-figure-rebuild-candidates-v1",
        "scope": "review candidates only; frozen data; no thesis integration",
        "frozen_assertions": assertion_report,
        "style": {"font_stack": plt.rcParams["font.sans-serif"], "png_dpi": PNG_DPI, "colors": COLORS},
        "candidates": records,
        "accessibility": accessibility,
        "equal_physical_scale_contact_sheet": {"path": contact.relative_to(ROOT).as_posix(), "sha256": sha256(contact), "pixels_per_inch": 100},
        "map": {
            "crs": TARGET_CRS,
            "two_county_tracts": 472,
            "analyzed_tracts": 327,
            "outside_analyzed_frame": 145,
            "water_features": int(len(context["water"])),
            "version_b_primary_road_segments": int(len(context["roads"])),
            "remote_tiles": False,
            "north_arrow": False,
            "scale_bar_miles": 5,
            "same_extent": list(context["extent"]),
        },
        "scientific_values_changed": False,
    }
    (OUT / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Built {len(records)} review candidates; frozen assertions: {assertion_report['status']}")
    print(f"Manifest: {OUT / 'candidate_manifest.json'}")


if __name__ == "__main__":
    main()
