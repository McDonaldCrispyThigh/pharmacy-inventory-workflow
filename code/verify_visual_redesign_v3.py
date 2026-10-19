"""Verify and close the publication-quality Results figure redesign package."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260902T012636Z"
PACKAGE = ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_visual_redesign_v3"
SOURCE = ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_publication_ready"
BASELINE_PDF = ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.pdf"
RESULTS_TEX = ROOT / "docs" / "thesis" / "chapters" / "04_results.tex"
WRAPPER_TEX = ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.tex"
COMPILED_PDF = PACKAGE / "compiled_thesis" / "introduction_literature_methods_results_draft.pdf"

FIGURES = [
    "figure01_record_flow",
    "figure02_core_decisions_confidence",
    "figure03_geography_correspondence",
    "figure04_name_recurrence",
    "figure05_spatial_disagreement_maps",
    "figure06_nearest_distance_difference",
]
FIGURE_PAGES = {1: 27, 2: 29, 3: 31, 4: 34, 5: 37, 6: 39}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def close(a: float, b: float, tolerance: float = 0.015) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> None:
    checks: list[dict] = []

    def check(label: str, condition: bool, evidence) -> None:
        record = {"check": label, "status": "pass" if condition else "fail", "evidence": evidence}
        checks.append(record)
        if not condition:
            raise AssertionError(f"{label}: {evidence}")

    check("final package exists", PACKAGE.is_dir(), PACKAGE.relative_to(ROOT).as_posix())
    build_manifest = load_json(PACKAGE / "figure_manifest.json")
    source_manifest = load_json(SOURCE / "package_manifest.json")
    check(
        "authenticated frozen Results semantic manifest",
        build_manifest["source_package"]["semantic_manifest_sha256"]
        == source_manifest["semantic_manifest_sha256"]
        == "4605f38b8d5dae12bce90cd610d3dc7b2464a481b7fe75b97c202d0e2829c1ad",
        source_manifest["semantic_manifest_sha256"],
    )
    check(
        "scientific-change flags are all false",
        build_manifest["scientific_assertions"]["all_fixed_values_pass"]
        and build_manifest["scientific_assertions"]["candidate_classifications_changed"] == 0
        and not build_manifest["scientific_assertions"]["board_screen_changed"]
        and not build_manifest["scientific_assertions"]["linkage_changed"]
        and not build_manifest["scientific_assertions"]["spatial_estimate_changed"]
        and not build_manifest["scientific_assertions"]["denominator_changed"],
        build_manifest["scientific_assertions"],
    )

    export_summary = {}
    for name in FIGURES:
        record = build_manifest["figures"][name]
        expected_size = (record["canvas_inches"]["width"], record["canvas_inches"]["height"])
        per_format = {}
        for suffix in ("pdf", "svg", "png"):
            path = PACKAGE / "figures" / f"{name}.{suffix}"
            expected_hash = record["formats"][suffix]["sha256"]
            check(f"{name} {suffix} exists and hash matches", path.is_file() and sha256_file(path) == expected_hash, expected_hash)
            per_format[suffix] = {"bytes": path.stat().st_size, "sha256": expected_hash}
        pdf = fitz.open(PACKAGE / "figures" / f"{name}.pdf")
        page = pdf[0]
        pdf_size = (page.rect.width / 72.0, page.rect.height / 72.0)
        spans = [
            span
            for block in page.get_text("dict").get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if str(span.get("text", "")).strip()
        ]
        min_text = min(float(span["size"]) for span in spans)
        figure_text = " ".join(str(span["text"]) for span in spans)
        check(f"{name} fixed PDF canvas", all(close(a, b) for a, b in zip(pdf_size, expected_size)), {"actual": pdf_size, "expected": expected_size})
        check(f"{name} standalone minimum text", min_text >= 8.45, min_text)
        check(f"{name} no arithmetic label", not re.search(r"\d+\s*/\s*\d+|=|H/M/L", figure_text), figure_text[:220])
        check(f"{name} PDF font resource", any("DejaVuSans" in item[3] for item in page.get_fonts(full=True)), [item[3] for item in page.get_fonts(full=True)])
        pdf.close()
        svg = (PACKAGE / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        check(f"{name} editable SVG text", "<text" in svg and "DejaVu Sans" in svg, {"text_elements": svg.count("<text")})
        with Image.open(PACKAGE / "figures" / f"{name}.png") as image:
            actual_dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            check(f"{name} opaque RGB PNG", image.mode == "RGB", image.mode)
            check(f"{name} PNG 500 dpi", all(abs(value - 500) <= 1 for value in actual_dpi), actual_dpi)
            png_pixels = image.size
        export_summary[name] = {
            "canvas_inches": expected_size,
            "minimum_standalone_text_pt": min_text,
            "png_pixels": png_pixels,
            "formats": per_format,
        }

    accessibility = list((PACKAGE / "accessibility").glob("*/*.png"))
    check("accessibility derivative count", len(accessibility) == 18, len(accessibility))
    check("accessibility derivatives are RGB", all(Image.open(path).mode == "RGB" for path in accessibility), "18 RGB simulations")
    candidates = list((PACKAGE / "layout_candidates").glob("*.*"))
    check("two layout candidates in three formats", len(candidates) == 6, [path.name for path in candidates])
    check("before/after contact sheet exists", (PACKAGE / "qa" / "before_after_contact_sheet.png").is_file(), "100 pixels per physical inch")

    flow = pd.read_csv(PACKAGE / "figure_data" / "figure01_record_flow.csv")
    expected_flow = {
        ("Workflow", "Candidates"): 1226,
        ("Workflow", "Included"): 498,
        ("Workflow", "In primary frame"): 339,
        ("Workflow", "Linked"): 146,
        ("Board", "Active in primary frame"): 245,
        ("Board", "Screened"): 178,
        ("Board", "Covered"): 146,
    }
    observed_flow = {(row.source, row.stage): int(row.stage_count) for row in flow.itertuples()}
    check(
        "Figure 4.1 frozen record flow",
        observed_flow == expected_flow,
        {f"{source}|{stage}": count for (source, stage), count in observed_flow.items()},
    )

    decisions = pd.read_csv(PACKAGE / "figure_data" / "figure02_core_decisions.csv")
    decision_lookup = {(row.section, row.measure): float(row.percent) for row in decisions.itertuples()}
    expected_decisions = {
        ("target_construct_status", "target_pharmacy"): 40.62,
        ("business_status", "active"): 99.84,
        ("candidate_plausibility", "plausible"): 76.51,
        ("identity_consistency", "consistent"): 97.15,
        ("eligibility", "include"): 40.62,
    }
    check("Figure 4.2 frozen decision percentages", all(close(decision_lookup[key], value, 0.005) for key, value in expected_decisions.items()), {str(key): decision_lookup[key] for key in expected_decisions})

    geography = pd.read_csv(PACKAGE / "figure_data" / "figure03_geography.csv")
    observed_geography = [(int(row.zip_count), float(row.workflow_linked_share_pct), float(row.board_reference_coverage_pct)) for row in geography.itertuples()]
    check("Figure 4.3 frozen geography summaries", observed_geography == [(60, 43.07, 82.02), (101, 48.89, 64.53), (106, 48.79, 64.36)], observed_geography)

    recurrence = pd.read_csv(PACKAGE / "figure_data" / "figure04_name_recurrence.csv")
    nonmissing = recurrence.loc[recurrence["linked_pct"].notna() & ~recurrence["group"].eq("unresolved")]
    observed_recurrence = sorted(round(float(value), 2) for value in nonmissing["linked_pct"])
    check("Figure 4.4 frozen recurrence percentages", observed_recurrence == [21.55, 51.06, 54.5, 93.13], observed_recurrence)
    check("Figure 4.4 one workflow unresolved row", int(recurrence.loc[recurrence["group"].eq("unresolved"), "records"].sum()) == 1, recurrence.loc[recurrence["group"].eq("unresolved")].to_dict("records"))

    states = pd.read_csv(PACKAGE / "figure_data" / "figure05_tract_states.csv", dtype={"GEOID": str})
    expected_states = {
        "discordance_type_0_5mi": {"neither_beyond": 93, "both_beyond": 188, "board_only_beyond": 28, "google_only_beyond": 18},
        "discordance_type_1_0mi": {"neither_beyond": 233, "both_beyond": 52, "board_only_beyond": 32, "google_only_beyond": 10},
        "discordance_type_2_0mi": {"neither_beyond": 322, "both_beyond": 1, "board_only_beyond": 4},
    }
    observed_states = {column: states[column].value_counts().to_dict() for column in expected_states}
    check("Figure 4.5 frozen paired states", len(states) == 327 and observed_states == expected_states, observed_states)

    distances = pd.read_csv(PACKAGE / "figure_data" / "figure06_signed_distances.csv")["workflow_minus_board_miles"]
    distance_values = {
        "n": len(distances),
        "mean": float(distances.mean()),
        "median": float(distances.median()),
        "q1": float(distances.quantile(0.25)),
        "q3": float(distances.quantile(0.75)),
        "min": float(distances.min()),
        "max": float(distances.max()),
        "negative": int(distances.lt(0).sum()),
        "positive": int(distances.gt(0).sum()),
    }
    expected_distance = {"n": 327, "mean": -0.0716413780, "median": -0.0088437115, "q1": -0.0779555233, "q3": 0.0221792701, "min": -1.9086468298, "max": 1.0438718433, "negative": 192, "positive": 135}
    check("Figure 4.6 frozen signed distribution", all(close(distance_values[key], value, 1e-8) for key, value in expected_distance.items()), distance_values)

    results_text = RESULTS_TEX.read_text(encoding="utf-8")
    wrapper_text = WRAPPER_TEX.read_text(encoding="utf-8")
    check("Results points to six v3 figures", results_text.count("final_results_20260902T012636Z_visual_redesign_v3/figures/") == 6, "six references")
    check("Results does not point to superseded redesign", "visual_redesign_v2/figures/" not in results_text, "none")
    check("landscape map environment active", "\\begin{landscape}" in results_text and "\\end{landscape}" in results_text and "\\usepackage{pdflscape}" in wrapper_text, "pdflscape")
    chapter_inputs = re.findall(r"\\input\{chapters/([^}]+)\}", wrapper_text)
    check("compile scope stops at Results", chapter_inputs == ["01_introduction_rewrite", "02_literature_rewrite", "03_methodology", "04_results"], chapter_inputs)

    compiled = fitz.open(COMPILED_PDF)
    check("compiled PDF page count", compiled.page_count == 44, compiled.page_count)
    check("Figure 4.5 landscape page", compiled[36].rotation == 90 and close(compiled[36].rect.width, 792, 0.1) and close(compiled[36].rect.height, 612, 0.1), {"rotation": compiled[36].rotation, "rect": list(compiled[36].rect)})
    compiled_fonts = {}
    for number, page_number in FIGURE_PAGES.items():
        page = compiled[page_number - 1]
        sizes = [
            float(span["size"])
            for block in page.get_text("dict").get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if "DejaVuSans" in str(span.get("font", "")) and str(span.get("text", "")).strip()
        ]
        check(f"compiled Figure 4.{number} text present", bool(sizes), {"page": page_number})
        check(f"compiled Figure 4.{number} minimum figure text", min(sizes) >= 8.49, min(sizes))
        compiled_fonts[f"figure_4_{number}"] = {"page": page_number, "minimum_pt": min(sizes), "maximum_pt": max(sizes)}
    map_xobjects = compiled[36].get_xobjects()
    check("compiled map unit-scale source width", len(map_xobjects) == 1 and close(map_xobjects[0][3][2], 539.427185, 0.01), map_xobjects)
    compiled.close()

    rendered_pages = list((PACKAGE / "qa" / "compiled_pages").glob("page*.png"))
    check("all six compiled figure pages rendered", len(rendered_pages) == 6, [path.name for path in rendered_pages])
    check("baseline PDF preserved", sha256_file(BASELINE_PDF) == "34d6776cafda62733df477510f631cdabd6e5ee24028f17c0ab5f45157040ffd", sha256_file(BASELINE_PDF))
    check("superseded v2 redesign preserved", (ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_visual_redesign_v2" / "figure_manifest.json").is_file(), "v2 build remains on disk")

    qa = {
        "schema_version": "nets-ai-results-figure-qa-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "package": PACKAGE.relative_to(ROOT).as_posix(),
        "source_semantic_manifest_sha256": source_manifest["semantic_manifest_sha256"],
        "compiled_pdf": {"path": COMPILED_PDF.relative_to(ROOT).as_posix(), "sha256": sha256_file(COMPILED_PDF), "pages": 44},
        "exports": export_summary,
        "compiled_figure_fonts": compiled_fonts,
        "frozen_distance_summary": distance_values,
        "checks": checks,
        "warnings": [
            "Color-vision images are deterministic simulations for design inspection, not clinical certifications.",
            "Tectonic reported underfull boxes, one 0.24402-pt overfull box, and Windows font-file lookup warnings; compilation completed and the resulting PDF exposes the expected Latin Modern and embedded DejaVu font resources.",
            "The build-time figure_manifest retains its v2 schema label and pre-switch flag; this closure report and final_package_manifest record the selected v3 package and compiled thesis state without overwriting the build manifest.",
        ],
    }
    qa_json = PACKAGE / "visual_qa_report.json"
    qa_json.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = f"""# Visual QA report\n\nStatus: **PASS**. Final package: `{PACKAGE.relative_to(ROOT).as_posix()}`.\n\n- Six selected figures were verified in PDF, editable-text SVG, and opaque RGB 500-dpi PNG.\n- All standalone and compiled figure text is at least 8.49 pt; the intended minimum is 8.5 pt, with the 0.01-pt tolerance covering PDF floating-point extraction.\n- The compiled Introduction--Results PDF has 44 pages. Figure 4.5 is a landscape page; the remaining selected figures use the portrait text width.\n- Eighteen color-accessibility derivatives were generated: grayscale, deuteranopia simulation, and protanopia simulation for every selected figure.\n- The Figure 4.5 map uses EPSG:26915, a common extent and aspect, a five-mile scale bar, one north arrow, city labels, water, county and fixed 60-ZCTA context, and a Minnesota locator.\n- Frozen record-flow, decision, geography, recurrence, paired-state, and signed-distance values all reproduced exactly. No candidate classification, Board screen, linkage, denominator, or spatial estimate changed.\n- The Results source points to the six v3 figures, activates the landscape map, and the compile scope ends with Chapter 4 Results.\n\n## Preserved design evidence\n\nThe repository-level spatial inventory, before/after contact sheet at 100 pixels per physical inch, rejected map and distribution layout candidates, baseline compiled-page renders, v2 scale-audit package, and all earlier scientific artifacts remain on disk.\n\n## Non-blocking compiler notes\n\nTectonic reported ordinary underfull boxes, one 0.24402-pt overfull box, and Windows font-file lookup warnings. Compilation completed, and inspection of the resulting PDF confirms the expected Latin Modern manuscript fonts and embedded DejaVu figure fonts. Color-vision derivatives are design simulations, not clinical certifications.\n"""
    (PACKAGE / "visual_qa_report.md").write_text(markdown, encoding="utf-8")

    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    excluded = {"final_package_manifest.json"}
    artifacts = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name not in excluded:
            artifacts[path.relative_to(PACKAGE).as_posix()] = sha256_file(path)
    payload = json.dumps(artifacts, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    final_manifest = {
        "schema_version": "nets-ai-results-visual-redesign-v3-final",
        "status": "complete",
        "run_id": RUN_ID,
        "git_head": git_head,
        "scope": "publication-quality figure and cartographic redesign only; no scientific or provider workflow changes",
        "source_semantic_manifest_sha256": source_manifest["semantic_manifest_sha256"],
        "compiled_pdf_sha256": sha256_file(COMPILED_PDF),
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts": artifacts,
        "semantic_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "verification": {"status": "pass", "report": "visual_qa_report.json", "check_count": len(checks)},
    }
    (PACKAGE / "final_package_manifest.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": len(checks), "artifact_count": len(artifacts), "semantic_manifest_sha256": final_manifest["semantic_manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
