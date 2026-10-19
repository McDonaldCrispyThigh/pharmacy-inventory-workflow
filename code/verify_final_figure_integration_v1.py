"""Verify the approved three-figure Results integration and render its pages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "analysis/v2/final_results_20260902T012636Z_final_figure_integration_v1"
FIGURES = PACKAGE / "figures"
COMPILED = PACKAGE / "compiled_thesis"
PDF = COMPILED / "introduction_literature_methods_results_final_figures.pdf"
LOG = COMPILED / "introduction_literature_methods_results_final_figures.log"
LOF = COMPILED / "introduction_literature_methods_results_final_figures.lof"
MANIFEST = PACKAGE / "final_figure_manifest.json"
RESULTS = ROOT / "docs/thesis/chapters/04_results.tex"
METHODS = ROOT / "docs/thesis/chapters/03_methodology.tex"
RECURRENCE_TABLE = ROOT / "docs/thesis/tables/table3_name_recurrence.tex"
SPATIAL_TABLE = ROOT / "docs/thesis/tables/table4_paired_spatial.tex"
WRAPPER = ROOT / "docs/thesis/introduction_literature_methods_results_final_figures.tex"
QA = PACKAGE / "qa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def page_for_caption(document: pymupdf.Document, caption_start: str) -> int:
    matches = [
        index
        for index, page in enumerate(document)
        if caption_start in page.get_text()
    ]
    require(len(matches) == 1, f"Expected one page for {caption_start!r}; found {matches}")
    return matches[0]


def render_page(page: pymupdf.Page, output: Path, dpi: int = 300) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    pixmap.save(output)
    with Image.open(output) as opened:
        require(opened.mode == "RGB", f"Rendered page is not RGB: {output}")
        expected = (round(page.rect.width / 72 * dpi), round(page.rect.height / 72 * dpi))
        require(opened.size == expected, f"Rendered page dimensions differ: {opened.size} != {expected}")


def main() -> None:
    for path in [
        PDF,
        LOG,
        LOF,
        MANIFEST,
        METHODS,
        RESULTS,
        RECURRENCE_TABLE,
        SPATIAL_TABLE,
        WRAPPER,
    ]:
        require(path.is_file(), f"Missing required file: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    methods = METHODS.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")
    recurrence_table = RECURRENCE_TABLE.read_text(encoding="utf-8")
    spatial_table = SPATIAL_TABLE.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    lof = LOF.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8", errors="replace")

    require(manifest["frozen_assertions"]["status"] == "pass", "Frozen assertions did not pass")
    require(not manifest["scientific_values_changed"], "Manifest reports changed scientific values")
    require(not manifest["provider_calls"] and not manifest["model_calls"], "Unexpected upstream call")
    require(results.count("\\begin{figure}") == 3, "Results does not contain exactly three figures")

    old_tokens = [
        "fig:record_flow",
        "fig:core_decisions",
        "fig:geography_correspondence",
        "figure01_record_flow.pdf",
        "figure02_core_decisions_confidence.pdf",
        "figure03_geography_correspondence.pdf",
        "figure04_name_recurrence.pdf",
        "figure05_spatial_disagreement_maps.pdf",
        "figure06_nearest_distance_difference.pdf",
    ]
    require(not any(token in results for token in old_tokens), "An old figure or label remains integrated")
    final_labels = [
        "fig:name_recurrence_final",
        "fig:spatial_disagreement_final",
        "fig:signed_distance_final",
    ]
    label_positions = [results.index(f"\\label{{{label}}}") for label in final_labels]
    require(label_positions == sorted(label_positions), "Final figure labels are not sequential in source")
    require(
        "all four classification changes were workflow only within" in results,
        "Required 2-mile prose statement is missing",
    )
    require("2-mile map" not in results and "appendix map" not in results, "A 2-mile map reference remains")

    paired_text = "\n".join([methods, results, spatial_table])
    paired_flat = re.sub(r"\s+", " ", paired_text)
    require(
        not any(
            obsolete.lower() in paired_flat.lower()
            for obsolete in ["Google only beyond", "Board only beyond"]
        ),
        "Obsolete paired-state terminology remains",
    )
    require(
        all(
            term.lower() in paired_flat.lower()
            for term in [
                "both within",
                "both beyond",
                "workflow only within",
                "Board only within",
            ]
        ),
        "Standard paired-state terminology is incomplete",
    )
    require("Google-minus-Board" not in methods, "Methods still names the final inventory Google")
    require(
        not any(
            re.search(r"\bmetres?\b", path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
            for path in (ROOT / "docs/thesis").rglob("*.tex")
        ),
        "British metre spelling remains in thesis source",
    )
    require(
        results.lower().count("does not establish ownership") == 1
        and "ownership" not in recurrence_table.lower(),
        "Figure 4.1 page repeats the ownership caveat",
    )

    require("\\documentclass[defaultstyle,11pt]{thesis}" in wrapper, "Integration wrapper does not use thesis.cls")
    require("\\input{chapters/04_results}" in wrapper, "Results is not in integration wrapper")
    require("05_discussion" not in wrapper, "Discussion is present in integration wrapper")
    require("06_conclusion" not in wrapper, "Conclusion is present in integration wrapper")
    require("\\abstract{" not in wrapper, "Abstract is present in integration wrapper")
    require("\\listoffigures" in wrapper, "List of Figures is missing")

    expected_lof = [
        "4.1",
        "Primary correspondence by normalized-name recurrence",
        "4.2",
        "Paired spatial disagreement at 0.5- and 1-mile thresholds",
        "4.3",
        "Workflow-minus-Board nearest-distance differences",
    ]
    require(all(item in lof for item in expected_lof), "Final List of Figures entries are incomplete")
    require(lof.count("\\contentsline {figure}") == 3, "List of Figures does not contain three entries")
    require(
        not any(text in lof for text in ["Candidate and Board-reference record flow", "Core decisions", "geographic frames"]),
        "A deleted figure remains in the List of Figures",
    )

    forbidden_log_patterns = [
        "Overfull \\hbox",
        "Overfull \\vbox",
        "Underfull \\hbox",
        "Underfull \\vbox",
        "undefined references",
        "multiply defined",
        "LaTeX Warning:",
        "! Undefined control sequence",
        "! LaTeX Error",
    ]
    require(
        not any(pattern.lower() in log.lower() for pattern in forbidden_log_patterns),
        "Compilation log contains a layout/reference warning",
    )

    expected_figures = {
        "figure01_name_recurrence": (6.20, 2.35, 3100, 1175),
        "figure02_spatial_disagreement": (8.90, 4.72, 4450, 2360),
        "figure03_signed_distance_histogram": (6.20, 2.90, 3100, 1450),
    }
    manifest_hashes = {
        record["name"]: {
            suffix: record["formats"][suffix]["sha256"]
            for suffix in ["pdf", "svg", "png"]
        }
        for record in manifest["figures"]
    }
    standalone_checks = {}
    for stem, (width, height, pixels_w, pixels_h) in expected_figures.items():
        paths = {suffix: FIGURES / f"{stem}.{suffix}" for suffix in ["pdf", "svg", "png"]}
        require(all(path.is_file() for path in paths.values()), f"Missing export for {stem}")
        require(
            {suffix: sha256(path) for suffix, path in paths.items()} == manifest_hashes[stem],
            f"Approved figure bytes changed for {stem}",
        )
        with Image.open(paths["png"]) as opened:
            require(opened.mode == "RGB", f"PNG is not opaque RGB for {stem}")
            require(opened.size == (pixels_w, pixels_h), f"PNG canvas mismatch for {stem}")
            dpi = opened.info.get("dpi", (0, 0))
            require(all(abs(value - 500) < 1 for value in dpi), f"PNG DPI mismatch for {stem}: {dpi}")
        svg = paths["svg"].read_text(encoding="utf-8")
        require("<text" in svg, f"SVG text is not editable for {stem}")
        with pymupdf.open(paths["pdf"]) as figure_pdf:
            require(len(figure_pdf) == 1, f"Standalone PDF is not one page for {stem}")
            page = figure_pdf[0]
            require(abs(page.rect.width / 72 - width) < 0.002, f"PDF width mismatch for {stem}")
            require(abs(page.rect.height / 72 - height) < 0.002, f"PDF height mismatch for {stem}")
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
            ]
            fonts = sorted({span["font"] for span in spans})
            sizes = [float(span["size"]) for span in spans]
            font_resources = [font[2] for font in page.get_fonts(full=True)]
            require(fonts == ["ArialMT"], f"Unexpected figure fonts for {stem}: {fonts}")
            require("Type3" not in font_resources, f"Type3 font found in {stem}")
            standalone_checks[stem] = {
                "canvas_inches": [width, height],
                "png_pixels": [pixels_w, pixels_h],
                "png_dpi": 500,
                "font_names": fonts,
                "font_point_range": [min(sizes), max(sizes)],
                "hashes": {suffix: sha256(path) for suffix, path in paths.items()},
            }
            if stem == "figure01_name_recurrence":
                label_spans = [span for span in spans if span["text"] == "93.1%"]
                require(len(label_spans) == 1, "Could not locate 93.1% label")
                right_margin_points = page.rect.width - label_spans[0]["bbox"][2]
                require(right_margin_points >= 18, "The 93.1% label lacks right-side margin")
                standalone_checks[stem]["label_93_1_right_margin_points"] = right_margin_points

    with pymupdf.open(PDF) as document:
        compiled_font_types = sorted(
            {font[2] for page in document for font in page.get_fonts(full=True)}
        )
        type3_pages = [
            index + 1
            for index, page in enumerate(document)
            if any(font[2] == "Type3" for font in page.get_fonts(full=True))
        ]
        require(not type3_pages, f"Compiled thesis contains Type3 fonts on pages {type3_pages}")
        captions = {
            "Figure 4.1": "Figure 4.1: Primary correspondence by normalized-name recurrence.",
            "Figure 4.2": "Figure 4.2: Paired spatial disagreement among 327 tract origins.",
            "Figure 4.3": "Figure 4.3: Workflow-minus-Board nearest-distance differences",
        }
        pages = {label: page_for_caption(document, caption) for label, caption in captions.items()}
        require(pages["Figure 4.1"] < pages["Figure 4.2"] < pages["Figure 4.3"], "Figure order differs")
        require(document[pages["Figure 4.1"]].rect.height > document[pages["Figure 4.1"]].rect.width, "Figure 4.1 page is not portrait")
        require(document[pages["Figure 4.2"]].rect.width > document[pages["Figure 4.2"]].rect.height, "Figure 4.2 page is not landscape")
        require(document[pages["Figure 4.3"]].rect.height > document[pages["Figure 4.3"]].rect.width, "Figure 4.3 page is not portrait")

        recurrence_page_text = document[pages["Figure 4.1"]].get_text()
        require("Table 4.3:" in recurrence_page_text, "Figure 4.1 is isolated from its table/prose")
        histogram_page_text = document[pages["Figure 4.3"]].get_text()
        require("Table 4.5:" not in histogram_page_text, "Table 4.5 shares the histogram page")
        require("Neighborhood-quartile patterns" in histogram_page_text, "Histogram lacks surrounding prose")

        render_dir = QA / "compiled_figure_pages_300dpi"
        rendered = {}
        for label, index in pages.items():
            slug = label.lower().replace(" ", "_").replace(".", "_")
            output = render_dir / f"{slug}_physical_{index + 1}.png"
            render_page(document[index], output, dpi=300)
            first_line = document[index].get_text().splitlines()[0]
            require(first_line.isdigit(), f"Could not identify printed page number for {label}")
            rendered[label] = {
                "printed_page": int(first_line),
                "physical_pdf_page": index + 1,
                "orientation": "landscape" if document[index].rect.width > document[index].rect.height else "portrait",
                "render_path": output.relative_to(ROOT).as_posix(),
                "render_sha256": sha256(output),
                "render_dpi": 300,
            }

    old_review = ROOT / "analysis/v2/final_results_20260902T012636Z_figure_rebuild_v1"
    require((old_review / "candidate_manifest.json").is_file(), "Prior candidate package is missing")
    require((old_review / "candidates/figure1_version_b_grouped_forest.pdf").is_file(), "Rejected candidate was not preserved")
    require((old_review / "appendix_candidates/figure3_appendix_2_miles.pdf").is_file(), "2-mile appendix candidate was not preserved")

    output = {
        "schema_version": "nets-ai-final-figure-integration-qa-v1",
        "status": "pass",
        "compiled_pdf": PDF.relative_to(ROOT).as_posix(),
        "compiled_pdf_sha256": sha256(PDF),
        "compiled_pdf_pages": len(pymupdf.open(PDF)),
        "results_source": RESULTS.relative_to(ROOT).as_posix(),
        "results_source_sha256": sha256(RESULTS),
        "integration_wrapper": WRAPPER.relative_to(ROOT).as_posix(),
        "integration_wrapper_sha256": sha256(WRAPPER),
        "standalone_figures": standalone_checks,
        "compiled_figure_pages": rendered,
        "list_of_figures_count": 3,
        "deleted_figure_references_present": False,
        "paired_state_terminology_consistent": True,
        "american_english_meters_consistent": True,
        "figure_4_1_ownership_caveat_reduced": True,
        "approved_figure_files_byte_identical": True,
        "cross_references_resolved": True,
        "layout_warnings": 0,
        "compiled_pdf_font_types": compiled_font_types,
        "compiled_pdf_type3_pages": [],
        "table_4_5_shares_histogram_page": False,
        "old_candidate_package_preserved": True,
        "abstract_integration": False,
        "discussion_integration": False,
        "conclusion_integration": False,
        "discussion_source_sha256": sha256(ROOT / "docs/thesis/chapters/05_discussion.tex"),
        "conclusion_source_sha256": sha256(ROOT / "docs/thesis/chapters/06_conclusion.tex"),
        "numerical_assertions": "pass",
        "scientific_values_changed": False,
    }
    QA.mkdir(parents=True, exist_ok=True)
    destination = QA / "final_integration_qa.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Final integration QA: {output['status']}")
    print(f"Compiled PDF pages: {output['compiled_pdf_pages']}")
    print(f"QA report: {destination}")


if __name__ == "__main__":
    main()
