"""Build the frozen Results figure visual redesign without scientific changes.

The command reads only authenticated local Results artifacts and 2023 Census
geometry. It writes a new versioned figure package and refuses to overwrite any
existing figure export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import fitz
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from study_design import TARGET_ZIPS
from thesis_figure_style import (
    COLORS,
    EXPORT_DPI,
    FIGURE_SIZES,
    FONT_FAMILY,
    FONT_SIZES,
    LINE_WIDTHS,
    MARKER_SIZES,
    THESIS_TEXT_WIDTH_IN,
    clean_axis,
    legend_kwargs,
    save_publication_figure,
    sha256_file,
    thesis_figure_context,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260902T012636Z"
SOURCE = ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_publication_ready"
DELIVERY = ROOT / "data" / "v2" / RUN_ID / "delivery"
DEFAULT_OUTPUT = (
    ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_visual_redesign_v3"
)
PREFLIGHT_SOURCE = (
    ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_visual_redesign_v2"
)
BASELINE_PDF = ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.pdf"
MILES_PER_METER = 1.0 / 1609.344


def require(condition, label):
    if not condition:
        raise ValueError(f"frozen figure assertion failed: {label}")


def load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object {path}")
    return value


def authenticate_source_package():
    manifest = load_json(SOURCE / "package_manifest.json")
    for relative, expected in manifest["artifacts"].items():
        path = SOURCE / relative
        require(path.is_file(), f"source artifact exists: {relative}")
        require(sha256_file(path) == expected, f"source artifact hash: {relative}")
    payload = json.dumps(
        manifest["artifacts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    require(
        hashlib.sha256(payload).hexdigest() == manifest["semantic_manifest_sha256"],
        "source package semantic manifest",
    )
    return manifest


def read_frozen_inputs():
    audit = load_json(SOURCE / "final_results_audit.json")
    tables = {
        "flow": pd.read_csv(SOURCE / "tables" / "table1_candidate_flow.csv"),
        "confidence": pd.read_csv(SOURCE / "tables" / "table1_confidence.csv"),
        "geography": pd.read_csv(
            SOURCE / "tables" / "table2_geography_sensitivity.csv"
        ),
        "recurrence": pd.read_csv(
            SOURCE / "tables" / "table3_name_recurrence.csv"
        ),
        "paired": pd.read_csv(
            SOURCE / "tables" / "table4_paired_spatial_classification.csv"
        ),
        "tract": pd.read_csv(
            DELIVERY / "tract_spatial.csv", dtype={"GEOID": str, "target_zcta": str}
        ),
    }
    require(audit["run_id"] == RUN_ID, "audit run ID")
    require(all(audit["verification_checks"].values()), "source Results verification")
    return audit, tables


def row_value(frame, section, measure, field):
    row = frame.loc[frame["section"].eq(section) & frame["measure"].eq(measure)]
    require(len(row) == 1, f"unique {section}/{measure}")
    return row.iloc[0][field]


def assert_frozen_values(audit, tables):
    flow = tables["flow"]
    expected_flow = {
        ("eligibility", "include"): (498, 40.62),
        ("eligibility", "exclude"): (728, 59.38),
        ("eligibility", "unresolved"): (0, 0.0),
        ("target_construct_status", "target_pharmacy"): (498, 40.62),
        ("target_construct_status", "non_target"): (728, 59.38),
        ("business_status", "active"): (1224, 99.84),
        ("business_status", "inactive"): (2, 0.16),
        ("candidate_plausibility", "plausible"): (938, 76.51),
        ("candidate_plausibility", "implausible"): (288, 23.49),
        ("identity_consistency", "consistent"): (1191, 97.15),
        ("identity_consistency", "inconsistent"): (35, 2.85),
    }
    for (section, measure), (count, percent) in expected_flow.items():
        require(int(row_value(flow, section, measure, "count")) == count, f"{section}/{measure} count")
        require(
            math.isclose(float(row_value(flow, section, measure, "percent")), percent, abs_tol=0.005),
            f"{section}/{measure} percent",
        )
        require(int(row_value(flow, section, measure, "denominator")) == 1226, f"{section}/{measure} denominator")

    confidence = tables["confidence"]
    expected_confidence = {
        "target_construct_status": {"High": 1028, "Moderate": 184, "Low": 14},
        "business_status": {"High": 1154, "Moderate": 71, "Low": 1},
        "candidate_plausibility": {"High": 1108, "Moderate": 113, "Low": 5},
        "identity_consistency": {"High": 1143, "Moderate": 83, "Low": 0},
    }
    for field, expected in expected_confidence.items():
        subset = confidence.loc[confidence["field"].eq(field)].set_index("confidence")
        observed = {level: int(subset.loc[level, "count"]) if level in subset.index else 0 for level in expected}
        require(observed == expected, f"confidence counts {field}")
        require(sum(observed.values()) == 1226, f"confidence total {field}")

    geography = tables["geography"]
    expected_geography = [
        ("primary_60_zip", 60, 339, 245, 178, 146, 193, 32, 43.07, 82.02),
        ("v1_observed_101_zip", 101, 495, 476, 375, 242, 253, 133, 48.89, 64.53),
        ("historical_v1_union_106_zip", 106, 496, 477, 376, 242, 254, 134, 48.79, 64.36),
    ]
    for expected, (_, row) in zip(expected_geography, geography.iterrows()):
        observed = (
            row["frame"], int(row["zip_count"]), int(row["workflow_locations"]),
            int(row["active_board_locations"]), int(row["screened_board_locations"]),
            int(row["linked_pairs"]), int(row["unmatched_workflow"]), int(row["unmatched_board"]),
            round(float(row["workflow_linked_share_pct"]), 2),
            round(float(row["board_reference_coverage_pct"]), 2),
        )
        require(observed == expected, f"geography row {expected[0]}")

    recurrence = tables["recurrence"]
    primary = recurrence.loc[
        recurrence["recurrence_threshold"].eq(2)
        & ~recurrence["group"].eq("unresolved")
    ]
    expected_recurrence = {
        ("Workflow inventory locations", "repeated-name component"): (222, 121, 54.50, 38.97, 66.96),
        ("Workflow inventory locations", "single-occurrence name component"): (116, 25, 21.55, 14.53, 29.31),
        ("Board keyword-screen reference", "repeated-name component"): (131, 122, 93.13, 90.36, 94.52),
        ("Board keyword-screen reference", "single-occurrence name component"): (47, 24, 51.06, 36.17, 65.96),
    }
    for key, expected in expected_recurrence.items():
        row = primary.loc[primary["source"].eq(key[0]) & primary["group"].eq(key[1])]
        require(len(row) == 1, f"recurrence row {key}")
        item = row.iloc[0]
        observed = (
            int(item["records"]), int(item["linked_records"]), round(float(item["linked_pct"]), 2),
            round(float(item["cluster_bootstrap_95_low_pct"]), 2),
            round(float(item["cluster_bootstrap_95_high_pct"]), 2),
        )
        require(observed == expected, f"recurrence values {key}")
    unresolved = recurrence.loc[
        recurrence["recurrence_threshold"].eq(2)
        & recurrence["source"].eq("Workflow inventory locations")
        & recurrence["group"].eq("unresolved")
    ]
    require(len(unresolved) == 1 and int(unresolved.iloc[0]["records"]) == 1, "one recurrence-unresolved workflow row")

    paired = tables["paired"]
    expected_paired = [
        (0.5, 93, 188, 28, 18, 46, 14.07, 327),
        (1.0, 233, 52, 32, 10, 42, 12.84, 327),
        (2.0, 322, 1, 4, 0, 4, 1.22, 327),
    ]
    for expected, (_, row) in zip(expected_paired, paired.iterrows()):
        observed = (
            float(row["threshold_miles"]), int(row["both_within"]), int(row["both_beyond"]),
            int(row["workflow_only_within"]), int(row["board_only_within"]),
            int(row["changed_tracts"]), round(float(row["changed_percent"]), 2), int(row["denominator"]),
        )
        require(observed == expected, f"paired spatial row {expected[0]}")

    tract = tables["tract"]
    require(len(tract) == 327 and tract["GEOID"].nunique() == 327, "327 unique tract origins")
    signed = (
        tract["google_include_nearest_m"] - tract["board_screened_nearest_m"]
    ) * MILES_PER_METER
    summary = {
        "workflow_mean": (tract["google_include_nearest_m"] * MILES_PER_METER).mean(),
        "board_mean": (tract["board_screened_nearest_m"] * MILES_PER_METER).mean(),
        "signed_mean": signed.mean(),
        "median": signed.median(),
        "q1": signed.quantile(0.25),
        "q3": signed.quantile(0.75),
        "min": signed.min(),
        "max": signed.max(),
        "workflow_nearer": int(signed.lt(0).sum()),
        "board_nearer": int(signed.gt(0).sum()),
    }
    expected_summary = {
        "workflow_mean": 0.677,
        "board_mean": 0.749,
        "signed_mean": -0.072,
        "median": -0.009,
        "q1": -0.078,
        "q3": 0.022,
        "min": -1.909,
        "max": 1.044,
    }
    for key, expected in expected_summary.items():
        require(round(float(summary[key]), 3) == expected, f"distance summary {key}")
    require(summary["workflow_nearer"] == 192 and summary["board_nearer"] == 135, "distance direction counts")

    for threshold, suffix, expected in (
        (0.5, "0_5", {"neither_beyond": 93, "both_beyond": 188, "board_only_beyond": 28, "google_only_beyond": 18}),
        (1.0, "1_0", {"neither_beyond": 233, "both_beyond": 52, "board_only_beyond": 32, "google_only_beyond": 10}),
        (2.0, "2_0", {"neither_beyond": 322, "both_beyond": 1, "board_only_beyond": 4, "google_only_beyond": 0}),
    ):
        counts = tract[f"discordance_type_{suffix}mi"].value_counts().to_dict()
        for key in expected:
            counts.setdefault(key, 0)
        require({key: int(counts[key]) for key in expected} == expected, f"map state counts {threshold}")

    return summary


def write_figure_data(tables, output, signed):
    data_dir = output / "figure_data"
    data_dir.mkdir()
    geography = tables["geography"]
    primary = geography.iloc[0]
    record_flow = pd.DataFrame(
        [
            ("Workflow", 1, "Candidates", 1226, "", 0),
            ("Workflow", 2, "Included", 498, "Excluded", 728),
            ("Workflow", 3, "In primary frame", 339, "Outside primary frame", 159),
            ("Workflow", 4, "Linked", 146, "Not linked", 193),
            ("Board", 1, "Active in primary frame", 245, "", 0),
            ("Board", 2, "Screened", 178, "Excluded by screen", 67),
            ("Board", 3, "Covered", 146, "Not covered", 32),
        ],
        columns=["source", "stage_order", "stage", "stage_count", "loss", "loss_count"],
    )
    require(int(primary["active_board_locations"]) == 245, "record flow Board active")
    record_flow.to_csv(data_dir / "figure01_record_flow.csv", index=False, lineterminator="\n")
    tables["flow"].to_csv(data_dir / "figure02_core_decisions.csv", index=False, lineterminator="\n")
    tables["confidence"].to_csv(data_dir / "figure02_confidence.csv", index=False, lineterminator="\n")
    tables["geography"].to_csv(data_dir / "figure03_geography.csv", index=False, lineterminator="\n")
    tables["recurrence"].loc[tables["recurrence"]["recurrence_threshold"].eq(2)].to_csv(
        data_dir / "figure04_name_recurrence.csv", index=False, lineterminator="\n"
    )
    state_columns = ["GEOID", "discordance_type_0_5mi", "discordance_type_1_0mi", "discordance_type_2_0mi"]
    tables["tract"][state_columns].to_csv(data_dir / "figure05_tract_states.csv", index=False, lineterminator="\n")
    pd.DataFrame({"GEOID": tables["tract"]["GEOID"], "workflow_minus_board_miles": signed}).to_csv(
        data_dir / "figure06_signed_distances.csv", index=False, lineterminator="\n"
    )


def render_baseline_pages(output):
    require(BASELINE_PDF.is_file(), "baseline compiled PDF exists")
    target = output / "qa" / "baseline_compiled_pages"
    target.mkdir(parents=True)
    document = fitz.open(BASELINE_PDF)
    pages = [27, 29, 31, 33, 36, 37]
    rendered = []
    for page_number in pages:
        pixmap = document[page_number - 1].get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        path = target / f"page{page_number:02d}.png"
        pixmap.save(path)
        rendered.append({"page": page_number, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "pixels": [pixmap.width, pixmap.height]})
    document.close()
    return {
        "source_pdf": BASELINE_PDF.relative_to(ROOT).as_posix(),
        "source_pdf_sha256": sha256_file(BASELINE_PDF),
        "render_scale": 2.5,
        "pages": rendered,
    }


def plot_record_flow(stem):
    workflow = [
        (1226, "Candidates"),
        (498, "Included"),
        (339, "In primary\nframe"),
        (146, "Linked"),
    ]
    board = [(245, "Active in\nprimary frame"), (178, "Screened"), (146, "Covered")]
    losses = {
        "Workflow": ["728\nexcluded", "159\noutside frame", "193\nnot linked"],
        "Board": ["67 excluded by screen", "32 not covered"],
    }
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure01_record_flow"])
        ax = fig.add_axes([0.06, 0.08, 0.92, 0.86])
        # Reserve explicit left-side room for the two source labels so the
        # fixed-canvas export never relies on bbox_inches="tight".
        ax.set_xlim(-0.72, 3.57)
        ax.set_ylim(-0.12, 1.92)
        ax.axis("off")

        def path(source, stages, y, color, positions):
            ax.text(-0.38, y, source, ha="left", va="center", fontsize=FONT_SIZES["panel"], weight="semibold", color=color)
            for index, ((count, label), x) in enumerate(zip(stages, positions)):
                if index < len(stages) - 1:
                    next_x = positions[index + 1]
                    ax.add_patch(
                        FancyArrowPatch(
                            (x + 0.08, y), (next_x - 0.08, y),
                            arrowstyle="-|>", mutation_scale=8.5,
                            linewidth=1.25, color=color, shrinkA=0, shrinkB=0,
                        )
                    )
                    midpoint = (x + next_x) / 2
                    loss_offset = 0.28 if source == "Workflow" else 0.43
                    ax.text(
                        midpoint, y - loss_offset, losses[source][index],
                        ha="center", va="top", fontsize=FONT_SIZES["annotation"],
                        color=COLORS["annotation"],
                    )
                ax.scatter(
                    [x], [y], s=MARKER_SIZES["process"], color=color,
                    edgecolor="white", linewidth=0.75, zorder=4,
                )
                ax.text(x, y + 0.13, f"{count:,}", ha="center", va="bottom", fontsize=9.6, weight="semibold", color=COLORS["text"])
                ax.text(x, y - 0.10, label, ha="center", va="top", fontsize=FONT_SIZES["annotation"], color=COLORS["annotation"])

        path("Workflow", workflow, 1.40, COLORS["workflow"], [0.42, 1.40, 2.38, 3.36])
        path("Board", board, 0.46, COLORS["board"], [0.42, 1.89, 3.36])
    return save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure01_record_flow"])


def plot_core_decisions(tables, stem):
    flow = tables["flow"]
    confidence = tables["confidence"].copy()
    fields = [
        ("target_construct_status", "Target construct", "target_pharmacy"),
        ("business_status", "Operating status", "active"),
        ("candidate_plausibility", "Candidate plausibility", "plausible"),
        ("identity_consistency", "Identity consistency", "consistent"),
        ("eligibility", "Final eligibility", "include"),
    ]
    values = [float(row_value(flow, section, measure, "percent")) for section, _, measure in fields]
    labels = [label for _, label, _ in fields]
    confidence_fields = [item[0] for item in fields[:4]]
    confidence_labels = labels[:4]
    levels = ["High", "Moderate", "Low"]
    confidence_colors = [COLORS["confidence_high"], COLORS["confidence_moderate"], COLORS["confidence_low"]]
    confidence_matrix = []
    for field in confidence_fields:
        subset = confidence.loc[confidence["field"].eq(field)].set_index("confidence")
        counts = np.array([float(subset.loc[level, "count"]) if level in subset.index else 0.0 for level in levels])
        confidence_matrix.append(counts / counts.sum() * 100.0)
    confidence_matrix = np.asarray(confidence_matrix)

    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure02_core_decisions_confidence"])
        grid = fig.add_gridspec(2, 1, left=0.34, right=0.98, top=0.96, bottom=0.19, hspace=0.62)
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[1, 0])
        y_a = np.arange(len(labels))
        ax_a.hlines(y_a, 0, values, color=COLORS["secondary_light"], linewidth=2.0, zorder=1)
        ax_a.scatter(values, y_a, s=MARKER_SIZES["dot"], color=COLORS["workflow"], marker="o", edgecolor="white", linewidth=0.65, zorder=3)
        for y, value in zip(y_a, values):
            if value > 93:
                x, ha = value - 2.2, "right"
            else:
                x, ha = value + 2.2, "left"
            ax_a.text(x, y, f"{value:.1f}%", ha=ha, va="center", fontsize=FONT_SIZES["annotation"], weight="semibold", color=COLORS["annotation"])
        ax_a.set(yticks=y_a, yticklabels=labels, xlim=(0, 100), xlabel="Affirmative / included candidates (%)")
        ax_a.set_xticks([0, 25, 50, 75, 100])
        ax_a.set_ylim(4.5, -0.5)
        ax_a.set_title("A. Core decisions", loc="left", pad=7)
        clean_axis(ax_a, grid_axis="x", bottom_spine=True)

        y_b = np.arange(4)
        left = np.zeros(4)
        for index, (level, color) in enumerate(zip(levels, confidence_colors)):
            widths = confidence_matrix[:, index]
            ax_b.barh(y_b, widths, left=left, height=0.52, color=color, edgecolor="white", linewidth=0.55, label=level)
            left += widths
        for y, high in zip(y_b, confidence_matrix[:, 0]):
            ax_b.text(high - 2.0, y, f"{high:.1f}%", ha="right", va="center", fontsize=FONT_SIZES["annotation"], weight="semibold", color="white")
        ax_b.set(yticks=y_b, yticklabels=confidence_labels, xlim=(0, 100), xlabel="Candidates (%)")
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.set_ylim(4.5, -0.5)
        ax_b.set_title("B. Field-specific confidence", loc="left", pad=7)
        clean_axis(ax_b, grid_axis=None, bottom_spine=True)
        handles, legend_labels = ax_b.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.66, 0.018), ncol=3, handlelength=1.6, columnspacing=1.1, **legend_kwargs())
    return save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure02_core_decisions_confidence"])


def plot_geography(frame, stem):
    # Reassign explicitly in UTF-8; an earlier Windows console round-trip
    # damaged only the display dash in the draft label string below.
    row_labels = ["60 ZIP — Primary", "101 ZIP — Diagnostic", "106 ZIP — Historical sensitivity"]
    row_labels = ["60 ZIP — Primary", "101 ZIP — Diagnostic", "106 ZIP — Historical sensitivity"]
    row_labels = ["60 ZIP - Primary", "101 ZIP - Diagnostic", "106 ZIP - Historical sensitivity"]
    y = np.arange(3)
    panels = [
        ("workflow_linked_share_pct", "A. Workflow linked share", COLORS["workflow"], "o"),
        ("board_reference_coverage_pct", "B. Board-reference coverage", COLORS["board"], "s"),
    ]
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure03_geography_correspondence"])
        grid = fig.add_gridspec(2, 1, left=0.43, right=0.98, top=0.95, bottom=0.13, hspace=0.78)
        axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
        for ax, (column, title, primary_color, marker) in zip(axes, panels):
            values = frame[column].astype(float).to_numpy()
            colors = [primary_color, COLORS["secondary"], COLORS["secondary"]]
            edgecolors = [COLORS["text"], COLORS["text"], COLORS["text"]]
            ax.scatter(values, y, s=[48, 34, 34], c=colors, marker=marker, edgecolors=edgecolors, linewidths=0.55, zorder=3)
            for position, value in enumerate(values):
                if value > 88:
                    x, ha = value - 2.0, "right"
                else:
                    x, ha = value + 2.0, "left"
                ax.text(x, position, f"{value:.1f}%", ha=ha, va="center", fontsize=FONT_SIZES["annotation"], weight="semibold" if position == 0 else "normal", color=COLORS["annotation"])
            ax.set(xlim=(0, 100), xlabel="Locations (%)", yticks=y, yticklabels=row_labels)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(2.55, -0.55)
            ax.set_title(title, loc="left", pad=7)
            ax.get_yticklabels()[0].set_weight("semibold")
            clean_axis(ax, grid_axis="x", bottom_spine=True)
    return save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure03_geography_correspondence"])


def plot_recurrence(frame, stem):
    subset = frame.loc[frame["recurrence_threshold"].eq(2) & ~frame["group"].eq("unresolved")].copy()
    order = ["repeated-name component", "single-occurrence name component"]
    panel_specs = [
        ("Workflow inventory locations", "A. Workflow linked share", COLORS["workflow"]),
        ("Board keyword-screen reference", "B. Board-reference coverage", COLORS["board"]),
    ]
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure04_name_recurrence"])
        grid = fig.add_gridspec(2, 1, left=0.31, right=0.98, top=0.95, bottom=0.13, hspace=0.82)
        axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
        for ax, (source, title, color) in zip(axes, panel_specs):
            data = subset.loc[subset["source"].eq(source)].set_index("group").loc[order]
            values = data["linked_pct"].astype(float).to_numpy()
            low = data["cluster_bootstrap_95_low_pct"].astype(float).to_numpy()
            high = data["cluster_bootstrap_95_high_pct"].astype(float).to_numpy()
            y = np.arange(2)
            for yy, value, lo, hi in zip(y, values, low, high):
                ax.hlines(yy, lo, hi, color=color, linewidth=LINE_WIDTHS["interval"], zorder=2)
                ax.vlines([lo, hi], yy - 0.065, yy + 0.065, color=color, linewidth=0.9, zorder=2)
                ax.scatter([value], [yy], s=MARKER_SIZES["forest"], color=color, marker="o", edgecolor="white", linewidth=0.65, zorder=3)
                if value > 88:
                    x, ha = value - 2.0, "right"
                else:
                    x, ha = value + 2.0, "left"
                ax.text(x, yy - 0.14, f"{value:.1f}%", ha=ha, va="bottom", fontsize=FONT_SIZES["annotation"], weight="semibold", color=COLORS["annotation"])
            ax.set(yticks=y, yticklabels=["Repeated-name", "Single-occurrence"], xlim=(0, 100), xlabel="Locations (%)")
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(1.55, -0.55)
            ax.set_title(title, loc="left", pad=7)
            clean_axis(ax, grid_axis="x", bottom_spine=True)
    return save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure04_name_recurrence"])


def load_map_layers(tract_table):
    tract_path = ROOT / "data" / "tl_2023_27_tract.zip"
    zcta_path = ROOT / "data" / "tl_2023_us_zcta520.zip"
    place_path = ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27_place.zip"
    water_paths = [
        ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27053_areawater.zip",
        ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27123_areawater.zip",
    ]
    for path in [tract_path, zcta_path, place_path, *water_paths]:
        require(path.is_file(), f"cartographic input exists: {path}")
    tracts = gpd.read_file(f"zip://{tract_path.as_posix()}")
    require(tracts.crs is not None and tracts.crs.to_epsg() == 4269, "tract CRS EPSG:4269")
    require(not tracts.geometry.isna().any() and not tracts.geometry.is_empty.any() and tracts.geometry.is_valid.all(), "valid tract geometry")
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    mapped = tracts[["GEOID", "COUNTYFP", "geometry"]].merge(tract_table, on="GEOID", how="inner", validate="one_to_one")
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=tracts.crs).to_crs(26915)
    require(len(mapped) == 327 and mapped.crs.to_epsg() == 26915, "mapped tract frame")
    require(mapped.geometry.is_valid.all(), "valid projected mapped geometry")

    county = tracts.loc[tracts["COUNTYFP"].isin(["053", "123"]), ["COUNTYFP", "geometry"]].to_crs(26915)
    county = county.dissolve(by="COUNTYFP", as_index=False)
    county_union = county.geometry.union_all()
    zctas = gpd.read_file(f"zip://{zcta_path.as_posix()}")
    require(zctas.crs is not None and zctas.crs.to_epsg() == 4269, "ZCTA CRS EPSG:4269")
    zctas["ZCTA5CE20"] = zctas["ZCTA5CE20"].astype(str).str.zfill(5)
    primary_zctas = zctas.loc[zctas["ZCTA5CE20"].isin(TARGET_ZIPS), ["ZCTA5CE20", "geometry"]].to_crs(26915)
    require(len(primary_zctas) == 60 and set(primary_zctas["ZCTA5CE20"]) == set(TARGET_ZIPS), "all fixed 60 target ZCTAs")
    primary_geometry = primary_zctas.geometry.union_all().intersection(county_union)
    require(not primary_geometry.is_empty and primary_geometry.is_valid, "valid clipped 60-ZCTA frame")
    primary_frame = gpd.GeoDataFrame({"frame": ["60 target ZCTAs clipped to Hennepin/Ramsey"]}, geometry=[primary_geometry], crs=26915)

    places = gpd.read_file(f"zip://{place_path.as_posix()}")
    require(places.crs is not None and places.crs.to_epsg() == 4269, "place CRS EPSG:4269")
    cities = places.loc[places["NAME"].isin(["Minneapolis", "St. Paul"]), ["NAME", "GEOID", "geometry"]].to_crs(26915)
    require(set(cities["NAME"]) == {"Minneapolis", "St. Paul"} and len(cities) == 2, "Minneapolis and St. Paul place polygons")

    waters = []
    for path in water_paths:
        layer = gpd.read_file(f"zip://{path.as_posix()}")
        require(layer.crs is not None and layer.crs.to_epsg() == 4269, f"water CRS {path.name}")
        require(layer.geometry.is_valid.all(), f"valid water geometry {path.name}")
        waters.append(layer[["FULLNAME", "MTFCC", "geometry"]])
    water = gpd.GeoDataFrame(pd.concat(waters, ignore_index=True), geometry="geometry", crs=4269).to_crs(26915)
    study_buffer = mapped.geometry.union_all().buffer(5000)
    water = water.loc[water.geometry.intersects(study_buffer)].copy()
    water["area_m2"] = water.geometry.area
    water = water.loc[
        water["FULLNAME"].fillna("").str.contains("Mississippi Riv", case=False)
        | water["area_m2"].ge(500_000)
    ].copy()
    require(len(water) == 38, "fixed context water selection count")

    frame_bounds = primary_frame.total_bounds
    width = frame_bounds[2] - frame_bounds[0]
    height = frame_bounds[3] - frame_bounds[1]
    extent = (
        frame_bounds[0] - 0.035 * width,
        frame_bounds[2] + 0.035 * width,
        frame_bounds[1] - 0.055 * height,
        frame_bounds[3] + 0.035 * height,
    )
    require(extent[0] < extent[1] and extent[2] < extent[3], "common map extent")
    return {
        "mapped": mapped,
        "county": county,
        "primary_frame": primary_frame,
        "cities": cities,
        "water": water,
        "state": tracts.to_crs(26915),
        "extent": extent,
        "paths": [tract_path, zcta_path, place_path, *water_paths],
    }


MAP_COLORS = {
    "neither_beyond": COLORS["agreement_within"],
    "both_beyond": COLORS["agreement_beyond"],
    "board_only_beyond": COLORS["workflow"],
    "google_only_beyond": COLORS["board"],
}
MAP_LABELS = {
    "neither_beyond": "Both within",
    "both_beyond": "Both beyond",
    "board_only_beyond": "Workflow only within",
    "google_only_beyond": "Board only within",
}
MAP_ORDER = ["neither_beyond", "both_beyond", "board_only_beyond", "google_only_beyond"]


def draw_map_panel(ax, layers, column, title, subtitle, *, add_scale=False, add_north=False):
    county = layers["county"]
    mapped = layers["mapped"]
    county.plot(ax=ax, facecolor=COLORS["context_fill"], edgecolor=COLORS["context"], linewidth=LINE_WIDTHS["context"], zorder=0)
    for state in MAP_ORDER:
        subset = mapped.loc[mapped[column].eq(state)]
        if not subset.empty:
            subset.plot(ax=ax, facecolor=MAP_COLORS[state], edgecolor="white", linewidth=LINE_WIDTHS["tract"], zorder=2)
    layers["water"].plot(ax=ax, facecolor=COLORS["water"], edgecolor="none", alpha=0.98, zorder=3)
    mapped.boundary.plot(ax=ax, color="white", linewidth=LINE_WIDTHS["tract"], zorder=4)
    layers["cities"].boundary.plot(ax=ax, color="#666666", linewidth=0.55, linestyle=(0, (2, 2)), zorder=5)
    layers["primary_frame"].boundary.plot(ax=ax, color=COLORS["text"], linewidth=LINE_WIDTHS["frame"], zorder=6)
    county.boundary.plot(ax=ax, color=COLORS["context"], linewidth=0.65, zorder=6)

    for _, row in layers["cities"].iterrows():
        anchor = row.geometry.representative_point()
        label = ax.text(anchor.x, anchor.y, row["NAME"], ha="center", va="center", fontsize=8.5, weight="semibold", color="#555555", zorder=8)
        label.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()])
    extent = layers["extent"]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{title}\n{subtitle}", loc="left", pad=5, fontsize=FONT_SIZES["panel"])

    if add_scale:
        length_m = 5 * 1609.344
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        x0 = extent[0] + 0.055 * width
        y0 = extent[2] + 0.032 * height
        line = ax.plot([x0, x0 + length_m], [y0, y0], color=COLORS["text"], linewidth=2.0, solid_capstyle="butt", zorder=12)[0]
        line.set_path_effects([patheffects.Stroke(linewidth=3.8, foreground="white"), patheffects.Normal()])
        for xx in (x0, x0 + length_m):
            tick = ax.plot([xx, xx], [y0 - 0.009 * height, y0 + 0.009 * height], color=COLORS["text"], linewidth=0.9, zorder=12)[0]
            tick.set_path_effects([patheffects.Stroke(linewidth=2.3, foreground="white"), patheffects.Normal()])
        text = ax.text(x0 + length_m / 2, y0 + 0.014 * height, "5 mi", ha="center", va="bottom", fontsize=8.5, color=COLORS["text"], zorder=12)
        text.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()])
    if add_north:
        ax.annotate("", xy=(0.94, 0.90), xytext=(0.94, 0.79), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": COLORS["text"], "linewidth": 1.0, "mutation_scale": 9}, zorder=12)
        ax.text(0.94, 0.93, "N", transform=ax.transAxes, ha="center", va="center", fontsize=9.0, weight="semibold", color=COLORS["text"], zorder=12)


def map_legend_handles():
    return [Patch(facecolor=MAP_COLORS[state], edgecolor=COLORS["text"] if state != "neither_beyond" else COLORS["secondary"], linewidth=0.55, label=MAP_LABELS[state]) for state in MAP_ORDER]


def draw_locator(fig, layers, rect):
    ax = fig.add_axes(rect)
    state_outline = gpd.GeoSeries([layers["state"].geometry.union_all()], crs=26915).simplify(2500)
    state_outline.boundary.plot(ax=ax, color=COLORS["context"], linewidth=0.6)
    min_x, max_x, min_y, max_y = layers["extent"]
    rectangle = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, facecolor=COLORS["workflow"], edgecolor="white", linewidth=0.4, zorder=3)
    ax.add_patch(rectangle)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.5, -0.06, "Minnesota", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=COLORS["annotation"])
    return ax


def plot_spatial_maps(layers, stem, *, vertical=False):
    specs = [
        ("A. 0.5-mile threshold", "14.1% changed (46 tracts)", "discordance_type_0_5mi"),
        ("B. 1-mile threshold", "12.8% changed (42 tracts)", "discordance_type_1_0mi"),
        ("C. 2-mile threshold", "1.2% changed (4 tracts)", "discordance_type_2_0mi"),
    ]
    key = "figure05_vertical_candidate" if vertical else "figure05_spatial_disagreement_maps"
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES[key])
        if vertical:
            grid = fig.add_gridspec(3, 1, left=0.07, right=0.93, top=0.91, bottom=0.16, hspace=0.38)
            axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
            for index, (ax, (title, subtitle, column)) in enumerate(zip(axes, specs)):
                draw_map_panel(ax, layers, column, title, subtitle, add_scale=index == 1, add_north=index == 2)
            fig.legend(handles=map_legend_handles(), loc="lower center", bbox_to_anchor=(0.48, 0.035), ncol=2, handlelength=1.8, columnspacing=1.1, **legend_kwargs())
            draw_locator(fig, layers, [0.82, 0.015, 0.13, 0.105])
        else:
            grid = fig.add_gridspec(1, 3, left=0.025, right=0.985, top=0.865, bottom=0.18, wspace=0.055)
            axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
            for index, (ax, (title, subtitle, column)) in enumerate(zip(axes, specs)):
                draw_map_panel(ax, layers, column, title, subtitle, add_scale=index == 1, add_north=index == 2)
            fig.legend(handles=map_legend_handles(), loc="lower left", bbox_to_anchor=(0.04, 0.025), ncol=4, handlelength=1.8, columnspacing=1.15, **legend_kwargs())
            draw_locator(fig, layers, [0.875, 0.060, 0.095, 0.13])
    result = save_publication_figure(fig, stem, expected_size=FIGURE_SIZES[key])
    result["layout"] = "vertical candidate" if vertical else "full-page landscape triptych"
    result["map"] = {
        "display_crs": "EPSG:26915",
        "display_crs_name": layers["mapped"].crs.name,
        "linear_units": "metre",
        "common_extent_m": [float(value) for value in layers["extent"]],
        "equal_aspect": True,
        "scale_bar_miles": 5,
        "north_arrow": "one shared north-up orientation cue",
        "eligible_tracts": len(layers["mapped"]),
        "missing_classification_count": 0,
        "target_zcta_count": 60,
        "county_count": len(layers["county"]),
        "city_context": sorted(layers["cities"]["NAME"].tolist()),
        "water_selection": "Hennepin/Ramsey 2023 TIGER/Line area-water polygons intersecting a 5-km eligible-tract buffer and either named Mississippi River or area >= 500,000 m2",
        "water_feature_count": len(layers["water"]),
        "remote_tiles": False,
    }
    return result


def histogram_figure(values, stem):
    lower = math.floor(float(values.min()) * 10) / 10
    upper = math.ceil(float(values.max()) * 10) / 10 + 0.1
    bins = np.arange(lower, upper + 1e-9, 0.1)
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [COLORS["workflow_light"] if center < 0 else "#E8C979" for center in centers]
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure06_nearest_distance_difference"])
        ax = fig.add_axes([0.12, 0.25, 0.85, 0.65])
        ax.axvspan(q1, q3, color=COLORS["secondary_light"], alpha=0.72, zorder=0)
        ax.bar(centers, counts, width=0.095, color=colors, edgecolor="white", linewidth=0.45, zorder=2)
        ax.axvline(0, color=COLORS["text"], linewidth=1.15, linestyle=(0, (3, 2)), zorder=4)
        ax.axvline(median, color="#111111", linewidth=1.25, zorder=5)
        ymax = max(counts) * 1.18
        ax.set_ylim(0, ymax)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("Workflow nearest distance − Board nearest distance (miles)")
        ax.set_xlabel("Workflow minus Board nearest distance (miles)")
        ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_ylabel("Tract origins (count)")
        clean_axis(ax, grid_axis="y", left_spine=False, bottom_spine=True)
        summary = (
            f"Range  {values.min():.3f} to {values.max():.3f} mi\n"
            f"IQR  {q1:.3f} to {q3:.3f} mi\n"
            f"Median  {median:.3f} mi"
        )
        ax.text(0.01, 0.96, summary, transform=ax.transAxes, ha="left", va="top", linespacing=1.35, fontsize=FONT_SIZES["annotation"], color=COLORS["annotation"])
        fig.text(0.12, 0.07, "Negative (192): workflow nearer", ha="left", va="center", fontsize=FONT_SIZES["annotation"], color=COLORS["workflow"])
        fig.text(0.97, 0.07, "Positive (135): Board nearer", ha="right", va="center", fontsize=FONT_SIZES["annotation"], color=COLORS["board_dark"])
    result = save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure06_nearest_distance_difference"])
    result["encoding"] = "histogram"
    result["bin_width_miles"] = 0.1
    return result


def ecdf_figure(values, stem):
    ordered = np.sort(values.to_numpy(dtype=float))
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100.0
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
    with thesis_figure_context():
        fig = plt.figure(figsize=FIGURE_SIZES["figure06_ecdf_candidate"])
        ax = fig.add_axes([0.12, 0.22, 0.85, 0.69])
        ax.axvspan(q1, q3, color=COLORS["secondary_light"], alpha=0.72, zorder=0)
        ax.step(ordered, cumulative, where="post", color=COLORS["workflow"], linewidth=1.5, zorder=3)
        ax.axvline(0, color=COLORS["text"], linewidth=1.15, linestyle=(0, (3, 2)), zorder=4)
        ax.axvline(median, color="#111111", linewidth=1.25, zorder=5)
        ax.set(xlim=(float(values.min()), float(values.max())), ylim=(0, 100), xlabel="Workflow nearest distance − Board nearest distance (miles)", ylabel="Cumulative tract origins (%)")
        ax.set_xlabel("Workflow minus Board nearest distance (miles)")
        ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([0, 25, 50, 75, 100])
        clean_axis(ax, grid_axis="both", bottom_spine=True)
        ax.text(0.01, 0.96, f"Observed range  {values.min():.3f} to {values.max():.3f} mi", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZES["annotation"], color=COLORS["annotation"])
        ax.text(0.98, 0.16, f"Median  {median:.3f} mi\nIQR  {q1:.3f} to {q3:.3f} mi", transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_SIZES["annotation"], color=COLORS["annotation"])
        fig.text(0.12, 0.055, "Negative: workflow location nearer", ha="left", va="center", fontsize=FONT_SIZES["annotation"], color=COLORS["workflow"])
        fig.text(0.97, 0.055, "Positive: Board location nearer", ha="right", va="center", fontsize=FONT_SIZES["annotation"], color=COLORS["board_dark"])
    result = save_publication_figure(fig, stem, expected_size=FIGURE_SIZES["figure06_ecdf_candidate"])
    result["encoding"] = "empirical cumulative distribution"
    return result


def simulate_color_vision(image, matrix):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    transformed = np.clip(linear @ np.asarray(matrix, dtype=np.float32).T, 0.0, 1.0)
    srgb = np.where(transformed <= 0.0031308, transformed * 12.92, 1.055 * transformed ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.uint8(np.clip(srgb, 0.0, 1.0) * 255.0), mode="RGB")


def accessibility_outputs(figures, output):
    matrices = {
        "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
        "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    }
    root = output / "accessibility"
    records = []
    for mode in ["grayscale", *matrices]:
        (root / mode).mkdir(parents=True)
    for name in figures:
        source = output / "figures" / f"{name}.png"
        with Image.open(source) as image:
            base = image.convert("RGB")
            gray = ImageOps.grayscale(base).convert("RGB")
            gray_path = root / "grayscale" / source.name
            gray.save(gray_path, dpi=(EXPORT_DPI, EXPORT_DPI), optimize=True)
            records.append({"figure": name, "simulation": "grayscale", "path": gray_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(gray_path)})
            for mode, matrix in matrices.items():
                simulated = simulate_color_vision(base, matrix)
                path = root / mode / source.name
                simulated.save(path, dpi=(EXPORT_DPI, EXPORT_DPI), optimize=True)
                records.append({"figure": name, "simulation": mode, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})
    return records


def contact_sheet(output, figure_names):
    old_manifest = load_json(SOURCE / "figure_manifest.json")["figures"]
    ppi = 100
    label_height = 32
    gap = 24
    rows = []
    font_path = Path(plt.matplotlib.font_manager.findfont(FONT_FAMILY))
    font = ImageFont.truetype(str(font_path), 16)
    for index, name in enumerate(figure_names, start=1):
        old_path = ROOT / old_manifest[name]["png"]["path"]
        new_path = output / "figures" / f"{name}.png"
        old_canvas = old_manifest[name]["canvas_inches"]
        old_effective_width = THESIS_TEXT_WIDTH_IN
        old_scale = old_effective_width / float(old_canvas["width"])
        old_size = (int(round(old_canvas["width"] * old_scale * ppi)), int(round(old_canvas["height"] * old_scale * ppi)))
        new_size_inches = FIGURE_SIZES[name]
        new_size = (int(round(new_size_inches[0] * ppi)), int(round(new_size_inches[1] * ppi)))
        with Image.open(old_path) as image:
            old = image.convert("RGB").resize(old_size, Image.Resampling.LANCZOS)
        with Image.open(new_path) as image:
            new = image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
        row_width = old.width + gap + new.width
        row_height = max(old.height, new.height) + label_height
        row = Image.new("RGB", (row_width, row_height), "white")
        draw = ImageDraw.Draw(row)
        draw.text((0, 3), f"Figure 4.{index} — before ({old_effective_width:.3f} in embedded)", font=font, fill=COLORS["text"])
        draw.text((old.width + gap, 3), f"Figure 4.{index} — after ({new_size_inches[0]:.3f} in embedded)", font=font, fill=COLORS["text"])
        row.paste(old, (0, label_height))
        row.paste(new, (old.width + gap, label_height))
        rows.append(row)
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    path = output / "qa" / "before_after_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, dpi=(ppi, ppi), optimize=True)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "pixels_per_inch": ppi, "comparison_rule": "one contact-sheet pixel scale per physical inch; old figures shown at prior LaTeX embedded size and new figures at intended embedded size"}


def vector_asset_inventory(output):
    specifications = [
        (ROOT / "data" / "holc_twin_cities.geojson", "historical HOLC polygons", "no: unrelated to this comparison"),
        (ROOT / "data" / "north_mpls_zcta.geojson", "two local ZCTAs", "no: incomplete extent"),
        (ROOT / "data" / "tl_2023_27_bg.zip", "Minnesota block groups", "no: unnecessary detail"),
        (ROOT / "data" / "tl_2023_27_tract.zip", "eligible tracts, county/state context", "yes"),
        (ROOT / "data" / "tl_2023_us_zcta520.zip", "fixed 60-ZCTA outline", "yes"),
        (ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27_place.zip", "Minneapolis/St. Paul outlines", "yes"),
        (ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27053_areawater.zip", "Hennepin water context", "yes, selected/clipped"),
        (ROOT / "data" / "cartographic_context" / "tigerline_2023" / "tl_2023_27123_areawater.zip", "Ramsey water context", "yes, selected/clipped"),
    ]
    rows = []
    study_shapes = gpd.read_file(f"zip://{(ROOT / 'data' / 'tl_2023_27_tract.zip').as_posix()}")
    tract_ids = set(pd.read_csv(DELIVERY / "tract_spatial.csv", dtype={"GEOID": str})["GEOID"])
    study_union = study_shapes.loc[study_shapes["GEOID"].astype(str).isin(tract_ids)].to_crs(26915).geometry.union_all()
    for path, role, suitability in specifications:
        layer = gpd.read_file(f"zip://{path.as_posix()}" if path.suffix.lower() == ".zip" else path)
        projected = layer.to_crs(26915)
        rows.append({
            "path": path.relative_to(ROOT).as_posix(), "format": path.suffix.lower(),
            "geometry_type": "|".join(sorted(layer.geom_type.dropna().unique())), "feature_count": len(layer),
            "crs": str(layer.crs), "bbox": "|".join(f"{value:.6f}" for value in layer.total_bounds),
            "raster_resolution": "not applicable", "probable_role": role,
            "overlaps_study_area": bool(projected.geometry.intersects(study_union).any()),
            "publication_suitability": suitability, "sha256": sha256_file(path),
            "null_geometry": int(layer.geometry.isna().sum()), "empty_geometry": int(layer.geometry.is_empty.sum()),
            "invalid_geometry": int((~layer.geometry.is_valid & layer.geometry.notna()).sum()),
        })
    frame = pd.DataFrame(rows)
    path = output / "repo_spatial_asset_inventory.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "asset_count": len(frame)}


def cartographic_metadata(layers, output):
    context_root = ROOT / "data" / "cartographic_context" / "tigerline_2023"
    sources = [
        ("tl_2023_27_place.zip", "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_27_place.zip"),
        ("tl_2023_27053_areawater.zip", "https://www2.census.gov/geo/tiger/TIGER2023/AREAWATER/tl_2023_27053_areawater.zip"),
        ("tl_2023_27123_areawater.zip", "https://www2.census.gov/geo/tiger/TIGER2023/AREAWATER/tl_2023_27123_areawater.zip"),
    ]
    metadata = {
        "schema_version": "nets-ai-cartographic-context-v1",
        "access_date": "2026-09-02",
        "source": "U.S. Census Bureau Geography Division, 2023 TIGER/Line Shapefiles",
        "attribution": "U.S. Census Bureau, 2023 TIGER/Line",
        "license_status": "U.S. federal-government geographic data; Census attribution retained",
        "downloaded_files": [{"path": (context_root / name).relative_to(ROOT).as_posix(), "url": url, "sha256": sha256_file(context_root / name)} for name, url in sources],
        "preexisting_files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for path in [ROOT / "data" / "tl_2023_27_tract.zip", ROOT / "data" / "tl_2023_us_zcta520.zip"]],
        "display_crs": "EPSG:26915 (NAD83 / UTM zone 15N; metre)",
        "common_extent_m": [float(value) for value in layers["extent"]],
        "target_boundary": "union of the 60 predeclared target ZCTAs, intersected with the Hennepin/Ramsey county union; display context only",
        "water_selection": "intersects a 5-km eligible-tract buffer AND (FULLNAME contains Mississippi Riv OR projected area >= 500,000 m2)",
        "water_feature_count": len(layers["water"]),
        "municipal_context": ["Minneapolis", "St. Paul"],
        "remote_tiles": False,
        "individual_pharmacy_points_plotted": False,
    }
    path = output / "cartographic_data_sources.json"
    write_json(path, metadata)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


def normalize_paths(value):
    if isinstance(value, dict):
        return {key: normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(ROOT):
                return path.relative_to(ROOT).as_posix()
        except (OSError, ValueError):
            pass
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed_name = output.name == DEFAULT_OUTPUT.name or output.name.startswith(
        DEFAULT_OUTPUT.name + "_round"
    )
    require(
        output.parent == DEFAULT_OUTPUT.resolve().parent and allowed_name,
        "build target is the frozen visual-redesign-v2 package or a versioned QA round",
    )
    if output == DEFAULT_OUTPUT.resolve():
        if not output.exists():
            output.mkdir()
            for name in (
                "figure_design_preflight.md",
                "figure_design_audit.md",
                "repo_spatial_asset_inventory.md",
            ):
                shutil.copy2(PREFLIGHT_SOURCE / name, output / name)
        require(output.is_dir(), "preflight output directory exists")
        require((output / "figure_design_preflight.md").is_file(), "preflight checklist exists")
    else:
        require(not output.exists(), "versioned QA-round target is new")
        output.mkdir()
        shutil.copy2(
            DEFAULT_OUTPUT / "figure_design_preflight.md",
            output / "figure_design_preflight.md",
        )
    for name in ("figures", "figure_data", "accessibility", "layout_candidates"):
        require(not (output / name).exists(), f"new build target absent: {name}")
    source_manifest = authenticate_source_package()
    audit, tables = read_frozen_inputs()
    distance_summary = assert_frozen_values(audit, tables)
    signed = (
        tables["tract"]["google_include_nearest_m"]
        - tables["tract"]["board_screened_nearest_m"]
    ) * MILES_PER_METER

    (output / "figures").mkdir()
    (output / "layout_candidates").mkdir()
    (output / "qa").mkdir()
    write_figure_data(tables, output, signed)
    baseline = render_baseline_pages(output)
    inventory = vector_asset_inventory(output)
    layers = load_map_layers(tables["tract"])
    cartography = cartographic_metadata(layers, output)

    figures = {}
    figures["figure01_record_flow"] = plot_record_flow(output / "figures" / "figure01_record_flow")
    figures["figure02_core_decisions_confidence"] = plot_core_decisions(tables, output / "figures" / "figure02_core_decisions_confidence")
    figures["figure03_geography_correspondence"] = plot_geography(tables["geography"], output / "figures" / "figure03_geography_correspondence")
    figures["figure04_name_recurrence"] = plot_recurrence(tables["recurrence"], output / "figures" / "figure04_name_recurrence")
    figures["figure05_spatial_disagreement_maps"] = plot_spatial_maps(layers, output / "figures" / "figure05_spatial_disagreement_maps", vertical=False)
    layout_candidates = {
        "figure05_vertical_candidate": plot_spatial_maps(layers, output / "layout_candidates" / "figure05_vertical_candidate", vertical=True),
        "figure06_ecdf_candidate": ecdf_figure(signed, output / "layout_candidates" / "figure06_ecdf_candidate"),
    }
    figures["figure06_nearest_distance_difference"] = histogram_figure(signed, output / "figures" / "figure06_nearest_distance_difference")

    figure_names = list(figures)
    accessibility = accessibility_outputs(figure_names, output)
    comparison = contact_sheet(output, figure_names)
    manifest = {
        "schema_version": "nets-ai-thesis-results-visual-redesign-v2",
        "run_id": RUN_ID,
        "scope": "visual redesign only; frozen local Results artifacts; no provider or model calls",
        "source_package": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "semantic_manifest_sha256": source_manifest["semantic_manifest_sha256"],
            "final_results_audit_sha256": sha256_file(SOURCE / "final_results_audit.json"),
        },
        "scientific_assertions": {
            "all_fixed_values_pass": True,
            "distance_summary": distance_summary,
            "candidate_classifications_changed": 0,
            "board_screen_changed": False,
            "linkage_changed": False,
            "spatial_estimate_changed": False,
            "denominator_changed": False,
        },
        "style": {
            "font_family": FONT_FAMILY,
            "font_sizes_pt": FONT_SIZES,
            "minimum_final_embedded_text_pt": 8.5,
            "portrait_text_width_inches": THESIS_TEXT_WIDTH_IN,
            "semantic_colors": COLORS,
            "png_dpi": EXPORT_DPI,
            "primary_format": "vector PDF",
            "secondary_formats": ["SVG with editable text", "opaque RGB PNG"],
        },
        "figures": normalize_paths(figures),
        "layout_candidates": normalize_paths(layout_candidates),
        "layout_selection": {
            "figure05": "full-page landscape triptych selected; the vertical candidate makes each common-scale map materially smaller",
            "figure06": "histogram selected; fixed-width bins show the near-zero concentration and both tails more directly than the ECDF candidate",
        },
        "baseline_compiled_pages": baseline,
        "spatial_inventory": inventory,
        "cartographic_sources": cartography,
        "accessibility_outputs": accessibility,
        "before_after_contact_sheet": comparison,
        "thesis_source_switched": False,
    }
    write_json(output / "figure_manifest.json", manifest)
    print(json.dumps({"status": "built", "output": output.as_posix(), "figures": len(figures), "candidates": len(layout_candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
