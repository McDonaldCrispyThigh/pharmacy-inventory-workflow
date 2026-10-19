"""Build and verify the three approved final Results figures.

This script is intentionally downstream-only.  It reads the frozen figure-data
exports, asserts them against the final forced-choice Results audit, and draws the
three user-approved figures.  It does not call a provider/model or rerun any
classification, linkage, Board-screen, or spatial-estimation workflow.
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
from PIL import Image, ImageOps
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/v2/final_results_20260902T012636Z_visual_redesign_v3"
AUDIT_PATH = (
    ROOT
    / "analysis/v2/results_audit_20260902T012636Z_forced_choice_final/results_audit.json"
)
OUT = ROOT / "analysis/v2/final_results_20260902T012636Z_final_figure_integration_v1"
FIGURES = OUT / "figures"
TARGET_CRS = "EPSG:26915"
PNG_DPI = 500
MILES_TO_METRES = 1609.344
MAP_BUFFER_MILES = 5.0

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
    "water": "#D9ECF2",
    "white": "#FFFFFF",
}

CANVAS_INCHES = {
    "figure01_name_recurrence": (6.20, 2.35),
    "figure02_spatial_disagreement": (8.90, 4.72),
    "figure03_signed_distance_histogram": (6.20, 2.90),
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
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.25,
            "axes.titleweight": "normal",
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
            "savefig.facecolor": COLORS["white"],
            "figure.facecolor": COLORS["white"],
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
        row = csv_primary.loc[
            (csv_primary["source"] == key[0]) & (csv_primary["group_code"] == key[1])
        ].iloc[0]
        actual = tuple(float(row[column]) for column in compare_columns[2:])
        if not np.allclose(actual, expected, atol=1e-9):
            raise AssertionError(f"Recurrence mismatch for {key}: {actual} != {expected}")

    unresolved = recurrence.loc[
        (recurrence["recurrence_threshold"] == 2)
        & recurrence["group_code"].eq("unresolved")
    ]
    workflow_unresolved = int(
        unresolved.loc[
            unresolved["source"].eq("Workflow inventory locations"), "records"
        ].sum()
    )
    board_unresolved = int(
        unresolved.loc[
            unresolved["source"].eq("Board keyword-screen reference"), "records"
        ].sum()
    )
    if (workflow_unresolved, board_unresolved) != (1, 0):
        raise AssertionError("Unexpected recurrence-unresolved counts")

    values = distances["workflow_minus_board_miles"].to_numpy(float)
    distance_summary = {
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
    if distance_summary != expected_distance:
        raise AssertionError(
            f"Distance summary mismatch: {distance_summary} != {expected_distance}"
        )

    threshold_columns = {
        "0.5": "discordance_type_0_5mi",
        "1.0": "discordance_type_1_0mi",
        "2.0": "discordance_type_2_0mi",
    }
    threshold_assertions: dict[str, dict] = {}
    for threshold, column in threshold_columns.items():
        scenario = audit["rq3_spatial"]["thresholds"][threshold]["scenarios"][
            "primary_google_include_vs_board_screened"
        ]
        audit_counts = {
            str(key): int(value["count"]) for key, value in scenario["states"].items()
        }
        counts = {key: int(states[column].eq(key).sum()) for key in audit_counts}
        if counts != audit_counts:
            raise AssertionError(f"State mismatch at {threshold}: {counts} != {audit_counts}")
        changed = counts.get("board_only_beyond", 0) + counts.get("google_only_beyond", 0)
        if changed != int(scenario["classification_changed"]["count"]):
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

    report = {
        "status": "pass",
        "scope": "final approved figures checked against frozen Results audit before plotting",
        "audit_path": AUDIT_PATH.relative_to(ROOT).as_posix(),
        "audit_sha256": sha256(AUDIT_PATH),
        "inputs": {
            recurrence_path.relative_to(ROOT).as_posix(): sha256(recurrence_path),
            states_path.relative_to(ROOT).as_posix(): sha256(states_path),
            distances_path.relative_to(ROOT).as_posix(): sha256(distances_path),
        },
        "recurrence": {
            "rows": {
                f"{key[0]} | {key[1]}": list(value)
                for key, value in expected_recurrence.items()
            },
            "workflow_unresolved": workflow_unresolved,
            "board_unresolved": board_unresolved,
        },
        "signed_distance": distance_summary,
        "thresholds": threshold_assertions,
        "provider_calls": False,
        "model_calls": False,
        "scientific_values_changed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_value_assertions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return recurrence, states, distances, report


def clean_axis(ax: plt.Axes, *, hide_left_spine: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def export_figure(fig: plt.Figure, name: str) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {suffix: FIGURES / f"{name}.{suffix}" for suffix in ["svg", "pdf", "png"]}
    fig.savefig(paths["svg"], format="svg", facecolor=COLORS["white"], transparent=False)
    fig.savefig(paths["pdf"], format="pdf", facecolor=COLORS["white"], transparent=False)
    fig.savefig(
        paths["png"],
        format="png",
        dpi=PNG_DPI,
        facecolor=COLORS["white"],
        transparent=False,
    )
    plt.close(fig)

    expected_w, expected_h = CANVAS_INCHES[name]
    with Image.open(paths["png"]) as opened:
        if opened.mode != "RGB":
            rgb = opened.convert("RGB")
            rgb.save(paths["png"], dpi=(PNG_DPI, PNG_DPI))
        pixels = [opened.width, opened.height]
    expected_pixels = [round(expected_w * PNG_DPI), round(expected_h * PNG_DPI)]
    if pixels != expected_pixels:
        raise AssertionError(f"Unexpected PNG canvas for {name}: {pixels} != {expected_pixels}")

    with fitz.open(paths["pdf"]) as document:
        rect = document[0].rect
        pdf_inches = [rect.width / 72.0, rect.height / 72.0]
        pdf_text = document[0].get_text()
        pdf_fonts = sorted({font[3] for font in document.get_page_fonts(0, full=True)})
    if not np.allclose(pdf_inches, [expected_w, expected_h], atol=0.002):
        raise AssertionError(f"Unexpected PDF canvas for {name}: {pdf_inches}")
    if not pdf_text.strip():
        raise AssertionError(f"PDF text extraction failed for {name}")
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise AssertionError(f"SVG text is not editable for {name}")

    return {
        "name": name,
        "canvas_inches": {"width": expected_w, "height": expected_h},
        "png_dpi": PNG_DPI,
        "png_pixels": pixels,
        "png_mode": "RGB",
        "pdf_fonts": pdf_fonts,
        "svg_editable_text": True,
        "pdf_text_extractable": True,
        "formats": {
            suffix: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for suffix, path in paths.items()
        },
    }


def recurrence_rows(recurrence: pd.DataFrame, source: str) -> pd.DataFrame:
    order = pd.CategoricalDtype(["repeated", "below_threshold"], ordered=True)
    rows = recurrence.loc[
        (recurrence["recurrence_threshold"] == 2)
        & recurrence["source"].eq(source)
        & recurrence["group_code"].isin(["repeated", "below_threshold"])
    ].copy()
    rows["group_code"] = rows["group_code"].astype(order)
    return rows.sort_values("group_code")


def draw_interval_rows(ax: plt.Axes, rows: pd.DataFrame, *, show_ylabels: bool) -> None:
    y = np.array([1.0, 0.0])
    point = rows["linked_pct"].to_numpy(float)
    low = rows["cluster_bootstrap_95_low_pct"].to_numpy(float)
    high = rows["cluster_bootstrap_95_high_pct"].to_numpy(float)
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        markersize=5.1,
        markerfacecolor=COLORS["neutral_mark"],
        markeredgecolor=COLORS["white"],
        markeredgewidth=0.55,
        ecolor=COLORS["neutral_mark"],
        elinewidth=1.25,
        capsize=3.1,
        capthick=1.1,
        zorder=3,
    )
    for value, ypos in zip(point, y):
        # Put the 93.1% label on the inside of the point so it cannot crowd 100%.
        if value > 89:
            ax.annotate(
                f"{value:.1f}%",
                (value, ypos),
                xytext=(-7, 6),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8.25,
                color=COLORS["muted_text"],
            )
        else:
            ax.annotate(
                f"{value:.1f}%",
                (value, ypos),
                xytext=(7, 6),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8.25,
                color=COLORS["muted_text"],
            )
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(["Repeated name", "Single occurrence"])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    clean_axis(ax, hide_left_spine=True)


def figure_name_recurrence(recurrence: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=CANVAS_INCHES["figure01_name_recurrence"], sharey=True
    )
    fig.subplots_adjust(left=0.185, right=0.985, top=0.81, bottom=0.27, wspace=0.16)
    specs = [
        ("Workflow inventory locations", "(a)", "Workflow linked share"),
        ("Board keyword-screen reference", "(b)", "Board-reference coverage"),
    ]
    for index, (ax, (source, label, heading)) in enumerate(zip(axes, specs)):
        draw_interval_rows(ax, recurrence_rows(recurrence, source), show_ylabels=index == 0)
        ax.text(
            0.0,
            1.10,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="normal",
        )
        ax.text(
            0.105,
            1.10,
            heading,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="normal",
        )
    fig.supxlabel("Correspondence (%)", fontsize=9.0, y=0.055)
    return fig


def figure_signed_histogram(distances: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=CANVAS_INCHES["figure03_signed_distance_histogram"])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.93, bottom=0.25)
    values = distances["workflow_minus_board_miles"].to_numpy(float)
    start = math.floor(values.min() * 10) / 10
    stop = math.ceil(values.max() * 10) / 10
    bins = np.arange(start, stop + 0.10001, 0.1)
    counts, edges = np.histogram(values, bins=bins)
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=COLORS["neutral_bar"],
        edgecolor=COLORS["white"],
        linewidth=0.45,
    )
    ax.axvline(0, color=COLORS["text"], linewidth=0.65, zorder=4)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(counts) * 1.08)
    ax.set_xlabel("Workflow − Board nearest distance (miles)", labelpad=7)
    ax.set_ylabel("Number of tracts", labelpad=7)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    clean_axis(ax)
    return fig


def load_map_context(states: pd.DataFrame) -> dict:
    tracts = gpd.read_file(ROOT / "data/tl_2023_27_tract.zip")
    tracts = tracts.loc[
        tracts["STATEFP"].eq("27") & tracts["COUNTYFP"].isin(["053", "123"])
    ].copy()
    if tracts.crs is None:
        raise AssertionError("TIGER/Line tract CRS metadata is missing")
    tracts = tracts.to_crs(TARGET_CRS)
    if len(tracts) != 472:
        raise AssertionError(f"Unexpected two-county tract count: {len(tracts)}")
    if tracts.geometry.isna().any() or tracts.geometry.is_empty.any():
        raise AssertionError("TIGER/Line tract layer contains null or empty geometry")
    if not tracts.geometry.is_valid.all():
        raise AssertionError("TIGER/Line tract layer contains invalid geometry")

    state_gdf = tracts.merge(states, on="GEOID", how="left", validate="one_to_one", indicator=True)
    if int(state_gdf["_merge"].eq("both").sum()) != 327:
        raise AssertionError("Not all 327 frozen tract states joined exactly once")
    state_gdf["analysis_frame"] = state_gdf["_merge"].eq("both")
    state_gdf = state_gdf.drop(columns="_merge")
    analyzed = state_gdf.loc[state_gdf["analysis_frame"]].copy()

    pad = MAP_BUFFER_MILES * MILES_TO_METRES
    analyzed_minx, analyzed_miny, analyzed_maxx, analyzed_maxy = analyzed.total_bounds
    extent = (
        float(analyzed_minx - pad),
        float(analyzed_maxx + pad),
        float(analyzed_miny - pad),
        float(analyzed_maxy + pad),
    )
    clip_box = box(extent[0], extent[2], extent[1], extent[3])
    if not analyzed.geometry.within(clip_box).all():
        raise AssertionError("An analyzed tract falls outside the buffered map rectangle")

    contextual_tracts = gpd.clip(state_gdf, clip_box, keep_geom_type=True)
    if set(analyzed["GEOID"]) - set(contextual_tracts["GEOID"]):
        raise AssertionError("Buffered clipping removed an analyzed tract")
    county_outlines = gpd.clip(tracts.dissolve(by="COUNTYFP"), clip_box, keep_geom_type=True)

    places = gpd.read_file(
        ROOT / "data/cartographic_context/tigerline_2023/tl_2023_27_place.zip"
    ).to_crs(TARGET_CRS)
    places = places.loc[places["NAME"].isin(["Minneapolis", "St. Paul"])].copy()
    if set(places["NAME"]) != {"Minneapolis", "St. Paul"}:
        raise AssertionError("Expected Minneapolis and St. Paul place features")

    water_frames = [
        gpd.read_file(
            ROOT
            / "data/cartographic_context/tigerline_2023/tl_2023_27053_areawater.zip"
        ),
        gpd.read_file(
            ROOT
            / "data/cartographic_context/tigerline_2023/tl_2023_27123_areawater.zip"
        ),
    ]
    water = gpd.GeoDataFrame(
        pd.concat(water_frames, ignore_index=True), geometry="geometry", crs="EPSG:4269"
    ).to_crs(TARGET_CRS)
    water["area_km2"] = water.geometry.area / 1_000_000
    water_names = water["FULLNAME"].fillna("")
    water = water.loc[
        water["area_km2"].ge(0.20)
        | water_names.str.contains("Mississippi", case=False)
    ].copy()
    water = gpd.clip(water, clip_box, keep_geom_type=True)

    highlighted = analyzed.loc[
        analyzed["discordance_type_0_5mi"].isin(
            ["board_only_beyond", "google_only_beyond"]
        )
        | analyzed["discordance_type_1_0mi"].isin(
            ["board_only_beyond", "google_only_beyond"]
        )
    ]
    if not highlighted.geometry.within(clip_box).all():
        raise AssertionError("A highlighted disagreement tract would be clipped")

    return {
        "tracts": contextual_tracts,
        "counties": county_outlines,
        "places": places,
        "water": water,
        "extent": extent,
        "analyzed_bounds": [
            float(analyzed_minx),
            float(analyzed_maxx),
            float(analyzed_miny),
            float(analyzed_maxy),
        ],
        "contextual_tract_count": int(len(contextual_tracts)),
        "water_feature_count": int(len(water)),
        "highlighted_geoid_count": int(len(highlighted)),
    }


def map_display_class(series: pd.Series, analysis_frame: pd.Series) -> pd.Series:
    display = pd.Series("Outside analysis frame", index=series.index, dtype="object")
    display.loc[
        analysis_frame & series.isin(["both_beyond", "neither_beyond"])
    ] = "Same classification"
    # board_only_beyond means Board is beyond the threshold and workflow is within.
    display.loc[
        analysis_frame & series.eq("board_only_beyond")
    ] = "Workflow only"
    # google_only_beyond means workflow is beyond the threshold and Board is within.
    display.loc[
        analysis_frame & series.eq("google_only_beyond")
    ] = "Board only"
    return display


def draw_map_panel(
    ax: plt.Axes,
    context: dict,
    column: str,
    panel_label: str,
    threshold_heading: str,
    changed_heading: str,
    *,
    draw_scale_bar: bool,
) -> None:
    tracts = context["tracts"].copy()
    tracts["display_class"] = map_display_class(tracts[column], tracts["analysis_frame"])
    fills = {
        "Outside analysis frame": COLORS["outside"],
        "Same classification": COLORS["agreement"],
        "Workflow only": COLORS["workflow"],
        "Board only": COLORS["board"],
    }
    for category in [
        "Outside analysis frame",
        "Same classification",
        "Workflow only",
        "Board only",
    ]:
        subset = tracts.loc[tracts["display_class"].eq(category)]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            facecolor=fills[category],
            edgecolor=COLORS["tract_edge"],
            linewidth=0.18 if category == "Outside analysis frame" else 0.24,
            zorder=1,
        )
    if not context["water"].empty:
        context["water"].plot(
            ax=ax, facecolor=COLORS["water"], edgecolor="none", zorder=2
        )
    context["counties"].boundary.plot(
        ax=ax, color=COLORS["county_edge"], linewidth=0.75, zorder=4
    )

    manual_label_positions = {
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
            fontsize=8.2,
            fontweight="normal",
            color=COLORS["text"],
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
            zorder=5,
        )

    minx, maxx, miny, maxy = context["extent"]
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(
        0.0,
        1.045,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.25,
        fontweight="normal",
    )
    ax.text(
        0.08,
        1.045,
        threshold_heading,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.25,
        fontweight="normal",
    )
    ax.text(
        0.08,
        1.005,
        changed_heading,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="normal",
        color=COLORS["muted_text"],
    )

    if draw_scale_bar:
        length = 5 * MILES_TO_METRES
        x0 = minx + 0.055 * (maxx - minx)
        y0 = miny + 0.050 * (maxy - miny)
        ax.plot(
            [x0, x0 + length],
            [y0, y0],
            color=COLORS["text"],
            linewidth=1.2,
            zorder=6,
        )
        cap = 0.009 * (maxy - miny)
        ax.plot([x0, x0], [y0 - cap, y0 + cap], color=COLORS["text"], linewidth=0.9, zorder=6)
        ax.plot(
            [x0 + length, x0 + length],
            [y0 - cap, y0 + cap],
            color=COLORS["text"],
            linewidth=0.9,
            zorder=6,
        )
        ax.text(
            x0 + length / 2,
            y0 + 1.7 * cap,
            "5 miles",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COLORS["text"],
            zorder=6,
        )


def figure_spatial_disagreement(context: dict) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=CANVAS_INCHES["figure02_spatial_disagreement"]
    )
    fig.subplots_adjust(left=0.018, right=0.988, top=0.90, bottom=0.13, wspace=0.045)
    draw_map_panel(
        axes[0],
        context,
        "discordance_type_0_5mi",
        "(a)",
        "0.5-mile threshold",
        "14.1% changed",
        draw_scale_bar=True,
    )
    draw_map_panel(
        axes[1],
        context,
        "discordance_type_1_0mi",
        "(b)",
        "1-mile threshold",
        "12.8% changed",
        draw_scale_bar=False,
    )
    handles = [
        Patch(
            facecolor=COLORS["agreement"],
            edgecolor=COLORS["tract_edge"],
            linewidth=0.6,
            label="Same classification",
        ),
        Patch(facecolor=COLORS["workflow"], edgecolor="none", label="Workflow only"),
        Patch(facecolor=COLORS["board"], edgecolor="none", label="Board only"),
        Patch(
            facecolor=COLORS["outside"],
            edgecolor=COLORS["tract_edge"],
            linewidth=0.5,
            label="Outside analysis frame",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        handlelength=1.25,
        handleheight=0.8,
        columnspacing=1.35,
        fontsize=8.0,
    )
    return fig


def simulate_accessibility(records: list[dict]) -> list[dict]:
    matrices = {
        "deuteranopia": np.array(
            [
                [0.367322, 0.860646, -0.227968],
                [0.280085, 0.672501, 0.047413],
                [-0.011820, 0.042940, 0.968881],
            ]
        ),
        "protanopia": np.array(
            [
                [0.152286, 1.052583, -0.204868],
                [0.114503, 0.786281, 0.099216],
                [-0.003882, -0.048116, 1.051998],
            ]
        ),
    }
    outputs = []
    for record in records:
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
                transformed = Image.fromarray(
                    np.round(transformed_array * 255).astype(np.uint8), mode="RGB"
                )
            transformed.save(output, dpi=(PNG_DPI, PNG_DPI))
            outputs.append(
                {
                    "figure": record["name"],
                    "mode": mode,
                    "path": output.relative_to(ROOT).as_posix(),
                    "sha256": sha256(output),
                }
            )
    return outputs


def main() -> None:
    set_style()
    recurrence, states, distances, assertions = load_and_assert_frozen_values()
    records = [
        export_figure(figure_name_recurrence(recurrence), "figure01_name_recurrence"),
    ]
    map_context = load_map_context(states)
    records.append(
        export_figure(
            figure_spatial_disagreement(map_context),
            "figure02_spatial_disagreement",
        )
    )
    records.append(
        export_figure(
            figure_signed_histogram(distances),
            "figure03_signed_distance_histogram",
        )
    )
    accessibility = simulate_accessibility(records)

    map_minx, map_maxx, map_miny, map_maxy = map_context["extent"]
    manifest = {
        "schema_version": "nets-ai-final-results-figures-v1",
        "scope": "three approved Results figures; frozen data; final thesis integration",
        "selection": {
            "name_recurrence": "approved Candidate A, refined",
            "signed_distance": "approved Candidate A, refined",
            "spatial_disagreement": "approved Candidate A without roads, refined",
            "rejected": ["all Candidate B versions", "ECDF diagnostic", "2-mile map"],
        },
        "frozen_assertions": assertions,
        "style": {
            "font_stack": plt.rcParams["font.sans-serif"],
            "font_point_range": [8.0, 9.25],
            "png_dpi": PNG_DPI,
            "colors": COLORS,
        },
        "figures": records,
        "map": {
            "crs": TARGET_CRS,
            "extent_rule": "total bounds of 327 analyzed tract geometries plus a 5-mile rectangular buffer on every side",
            "buffer_miles": MAP_BUFFER_MILES,
            "extent_epsg_26915_metres": [map_minx, map_maxx, map_miny, map_maxy],
            "extent_width_metres": map_maxx - map_minx,
            "extent_height_metres": map_maxy - map_miny,
            "analyzed_bounds_epsg_26915_metres": map_context["analyzed_bounds"],
            "analyzed_tracts": 327,
            "contextual_tract_features_after_clip": map_context[
                "contextual_tract_count"
            ],
            "water_features_after_clip": map_context["water_feature_count"],
            "highlighted_geoid_count_across_two_thresholds": map_context[
                "highlighted_geoid_count"
            ],
            "same_extent": True,
            "equal_aspect": True,
            "remote_tiles": False,
            "roads": False,
            "north_arrow": False,
            "scale_bar_miles": 5,
        },
        "accessibility_outputs": accessibility,
        "provider_calls": False,
        "model_calls": False,
        "scientific_values_changed": False,
    }
    manifest_path = OUT / "final_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Built {len(records)} final figures; frozen assertions: {assertions['status']}")
    print(f"Map extent: {manifest['map']['extent_epsg_26915_metres']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
