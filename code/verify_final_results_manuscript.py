"""Offline verification for the frozen forced-choice Results package and manuscript."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260902T012636Z"
PACKAGE = ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}_publication_ready"
CLOSURE = ROOT / "analysis" / "v2" / f"results_closure_audit_{RUN_ID}"
RESULTS = ROOT / "docs" / "thesis" / "chapters" / "04_results.tex"
METHODS = ROOT / "docs" / "thesis" / "chapters" / "03_methodology.tex"
TABLE2 = ROOT / "docs" / "thesis" / "tables" / "table2_board_correspondence.tex"
WRAPPER = (
    ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.tex"
)
PDF = ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(f"verification failed: {label}")


def main() -> int:
    package_manifest = load_json(PACKAGE / "package_manifest.json")
    for relative, expected in package_manifest["artifacts"].items():
        path = PACKAGE / relative
        require(path.is_file() and sha256(path) == expected, f"package hash {relative}")
    semantic_payload = json.dumps(
        package_manifest["artifacts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    require(
        hashlib.sha256(semantic_payload).hexdigest()
        == package_manifest["semantic_manifest_sha256"],
        "package semantic manifest",
    )

    audit = load_json(PACKAGE / "final_results_audit.json")
    require(audit["run_id"] == RUN_ID, "audit run ID")
    require(all(audit["verification_checks"].values()), "audit verification checks")
    rq1 = audit["base_results_audit"]["rq1_candidate_flow"]
    require(
        [rq1["eligibility"][key]["count"] for key in ("include", "exclude", "unresolved")]
        == [498, 728, 0],
        "RQ1 flow",
    )
    primary = audit["rq2"]["geography_sensitivity"][0]
    require(
        [
            primary["workflow_locations"],
            primary["screened_board_locations"],
            primary["linked_pairs"],
            primary["unmatched_workflow"],
            primary["unmatched_board"],
        ]
        == [339, 178, 146, 193, 32],
        "RQ2 primary flow",
    )
    require(
        [
            audit["base_results_audit"]["rq3_spatial"]["thresholds"][threshold]
            ["scenarios"]["primary_google_include_vs_board_screened"]
            ["classification_changed"]["count"]
            for threshold in ("0.5", "1.0", "2.0")
        ]
        == [46, 42, 4],
        "RQ3 primary changes",
    )
    figure_manifest = load_json(PACKAGE / "figure_manifest.json")
    require(
        figure_manifest["schema_version"] == "nets-ai-thesis-results-figures-v2"
        and len(figure_manifest["figures"]) == 6,
        "publication figure manifest",
    )
    for name, item in figure_manifest["figures"].items():
        require(
            item["fixed_canvas"] is True
            and item["opaque_background"] is True
            and item["canvas_inches"]["width"] == 6.5
            and item["png_expected_pixels"]["width"] == 2600,
            f"fixed publication canvas {name}",
        )
        png_path = ROOT / item["png"]["path"]
        with Image.open(png_path) as rendered:
            require(
                rendered.mode == "RGB"
                and rendered.size
                == (
                    item["png_expected_pixels"]["width"],
                    item["png_expected_pixels"]["height"],
                ),
                f"RGB raster fallback {name}",
            )
    map_spec = figure_manifest["figures"]["figure05_spatial_disagreement_maps"][
        "map_specification"
    ]
    require(
        map_spec["display_crs"] == "EPSG:26915"
        and map_spec["linear_units"] == "metre"
        and map_spec["scale_bar_km"] == 10
        and map_spec["north_arrow"] == "grid north"
        and map_spec["missing_classification_count"] == 0
        and map_spec["remote_tiles"] is False,
        "publication map specification",
    )

    closure_manifest = load_json(CLOSURE / "manifest.json")
    for relative, expected in closure_manifest["artifacts"].items():
        path = CLOSURE / relative
        require(path.is_file() and sha256(path) == expected, f"closure hash {relative}")
    closure_payload = json.dumps(
        closure_manifest["artifacts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    require(
        hashlib.sha256(closure_payload).hexdigest()
        == closure_manifest["semantic_manifest_sha256"],
        "closure semantic manifest",
    )
    closure = load_json(CLOSURE / "closure_audit.json")
    collapsed = closure["collapsed_location_sensitivity"]
    require(
        [
            collapsed["collapsed_outlet_locations"]["workflow_locations"],
            collapsed["collapsed_outlet_locations"]["linked_pairs"],
            collapsed["collapsed_outlet_locations"]["unmatched_workflow_locations"],
            collapsed["collapsed_outlet_locations"]["unmatched_board_records"],
        ]
        == [239, 146, 93, 32],
        "collapsed location flow",
    )
    require(
        collapsed["collapsed_outlet_locations"]["place_ids_per_location_distribution"]
        == {"1": 150, "2": 78, "3": 11},
        "Place IDs per collapsed location",
    )
    require(
        collapsed["original_193_unmatched_rows"]
        ["disappear_by_collapsing_into_location_with_primary_linked_member"]
        == 90,
        "unmatched rows absorbed by a linked location",
    )
    require(
        collapsed["location_to_board_pair_comparison"]["identical_pair_sets"] is True,
        "identical location-to-Board pair sets",
    )
    field = closure["forced_choice_field_dependence"]
    require(
        field["target_pharmacy"] == {"total": 498, "included": 498, "excluded": 0}
        and field["non_target"] == {"total": 728, "included": 0, "excluded": 728}
        and field["realized_eligibility_equals_target_indicator"] is True
        and field["recorded_state_equals_independent_four_field_reduction"] is True
        and field["recorded_reason_codes_equal_independent_reduction"] is True,
        "forced-choice field dependence",
    )
    pilot = closure["evidence_ablation_boundary"]
    require(
        pilot["validation_passed"] == 24
        and pilot["technical_failures"] == 0
        and pilot["eligibility_changes"] == 0
        and pilot["unresolved_state_changes"] == 0,
        "A/B/C closure boundary",
    )
    figure_audit = closure["figure_readability_audit"]
    require(
        len(figure_audit["compiled_pdf_sha256"]) == 64
        and figure_audit["figures_retained"] == 6
        and figure_audit["scientific_content_changed"] is False
        and figure_audit["observed_clipping_or_overlap_after_revision"] is False,
        "preserved pre-redraw closure figure audit",
    )

    results = RESULTS.read_text(encoding="utf-8")
    require(results.count("\\input{tables/") == 5, "five Results tables")
    require(results.count("\\includegraphics") == 6, "six Results figures")
    require(
        results.count(f"final_results_{RUN_ID}_publication_ready/figures/") == 6,
        "six publication-ready figure paths",
    )
    for fragment in (
        "498 of 1,226 candidates",
        "728 (59.38\\%)",
        "146 of 339 workflow rows",
        "43.07\\%",
        "82.02\\%",
        "193 workflow",
        "32 Board",
        "339 primary rows formed 239 outlet-location units",
        r"workflow linked share was 61.09\%",
        "90 collapsed into a location",
        "no non-target record was included",
        "0.677 miles",
        "0.749 miles",
        "46 tracts (14.07\\%)",
        "42 (12.84\\%)",
        "4 (1.22\\%)",
    ):
        require(fragment in results, f"Results fragment {fragment}")
    require(
        re.search(
            r"\b(precision|recall|accuracy|specificity|false positive|false negative)\b",
            results,
            flags=re.IGNORECASE,
        )
        is None,
        "forbidden Results metric/error terminology",
    )
    methods = METHODS.read_text(encoding="utf-8")
    require(
        "not population accuracy or whether an evidence" in methods
        and "post hoc Results-closure sensitivity" in methods,
        "Methods closure boundaries",
    )
    table2 = TABLE2.read_text(encoding="utf-8")
    require(
        "Address--ZIP location collapse (post hoc) & 146 & 93 / 32 & 61.09 & 82.02"
        in table2,
        "Table 2 collapsed sensitivity",
    )

    wrapper = WRAPPER.read_text(encoding="utf-8")
    require("\\input{chapters/04_results}" in wrapper, "Results included")
    require(
        all(
            token not in wrapper
            for token in (
                "chapters/05_discussion",
                "chapters/06_conclusion",
                "\\begin{abstract}",
            )
        ),
        "later chapters remain excluded",
    )
    require(PDF.is_file() and PDF.stat().st_size > 1_000_000, "compiled PDF")

    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": RUN_ID,
                "package_semantic_manifest_sha256": package_manifest[
                    "semantic_manifest_sha256"
                ],
                "closure_semantic_manifest_sha256": closure_manifest[
                    "semantic_manifest_sha256"
                ],
                "tables": 5,
                "figures": 6,
                "compiled_pdf_sha256": sha256(PDF),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
