"""Deterministic closure audit for the frozen forced-choice Results package.

The audit reads authenticated local artifacts for run ``20260902T012636Z``.  It
does not call providers, alter completed candidates, or replace the primary
Place-ID linkage.  Its address-collapse calculation is a post hoc sensitivity
requested for Results closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

import geo_match
from audit_forced_choice_results import CONDITIONS, CORE_FIELDS, build_audit
from finalize_forced_choice_results import (
    PRIMARY_STRATUM,
    PRIMARY_THRESHOLD,
    ROOT,
    RUN_ID,
    build_google_frame,
    load_board,
    load_completed_units,
    pct,
    sha256_file,
)
from study_design import TARGET_ZIPS, normalized_location_key
from v2_board_adapter import prepare_v2_board_frames


OUTPUT_ROOT = ROOT / "analysis" / "v2" / f"results_closure_audit_{RUN_ID}"
THESIS_PDF = (
    ROOT / "docs" / "thesis" / "introduction_literature_methods_results_draft.pdf"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def _safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_reason_codes(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("invalid eligibility reason-code encoding")
    return parsed


def _nested_crosstab(frame: pd.DataFrame, row: str, column: str) -> dict[str, Any]:
    table = pd.crosstab(frame[row], frame[column], dropna=False)
    return {
        "row_field": row,
        "column_field": column,
        "row_levels": [str(value) for value in table.index],
        "column_levels": [str(value) for value in table.columns],
        "counts": {
            str(row_level): {
                str(column_level): int(table.loc[row_level, column_level])
                for column_level in table.columns
            }
            for row_level in table.index
        },
    }


def field_dependence_audit(
    units: pd.DataFrame, delivery: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Authenticate record-level fields and report every pairwise crosstab."""
    eligibility = pd.read_csv(delivery / "eligibility.csv", dtype=str).fillna("")
    if len(eligibility) != 1226 or eligibility["place_id"].duplicated().any():
        raise ValueError("eligibility frame is incomplete")
    fields = [*CORE_FIELDS, "eligibility_state"]
    joined = eligibility.merge(
        units[["place_id", "candidate_id", *CORE_FIELDS]],
        on=["place_id", "candidate_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_delivery", "_unit"),
    )
    field_bindings = {
        field: bool(joined[f"{field}_delivery"].eq(joined[f"{field}_unit"]).all())
        for field in CORE_FIELDS
    }
    frame = pd.DataFrame(
        {
            field: joined[f"{field}_delivery"] for field in CORE_FIELDS
        }
    )
    frame["eligibility_state"] = eligibility["eligibility_state"]
    frame["reason_codes"] = eligibility["reason_codes"]

    expected_state = np.where(
        frame["target_construct_status"].eq("target_pharmacy")
        & frame["business_status"].eq("active")
        & frame["candidate_plausibility"].eq("plausible")
        & frame["identity_consistency"].eq("consistent"),
        "include",
        "exclude",
    )
    expected_reasons: list[list[str]] = []
    for row in frame.itertuples(index=False):
        reasons = []
        if row.target_construct_status == "non_target":
            reasons.append("non_target")
        if row.business_status == "inactive":
            reasons.append("inactive")
        if row.candidate_plausibility == "implausible":
            reasons.append("implausible")
        if row.identity_consistency == "inconsistent":
            reasons.append("identity_inconsistent")
        expected_reasons.append(reasons or ["all_four_core_judgments_positive"])
    recorded_reasons = [_parse_reason_codes(value) for value in frame["reason_codes"]]

    target = frame["target_construct_status"].eq("target_pharmacy")
    included = frame["eligibility_state"].eq("include")
    crosstabs = {
        f"{left}__by__{right}": _nested_crosstab(frame, left, right)
        for left, right in combinations(fields, 2)
    }
    key_rows = []
    for field in CORE_FIELDS:
        grouped = (
            frame.groupby([field, "eligibility_state"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        key_rows.extend(grouped.to_dict(orient="records"))
    key_table = pd.DataFrame(key_rows)
    audit = {
        "candidate_denominator": len(frame),
        "record_level_completed_unit_bindings": field_bindings,
        "recorded_state_equals_independent_four_field_reduction": bool(
            np.array_equal(frame["eligibility_state"].to_numpy(), expected_state)
        ),
        "recorded_reason_codes_equal_independent_reduction": recorded_reasons
        == expected_reasons,
        "target_pharmacy": {
            "total": int(target.sum()),
            "included": int((target & included).sum()),
            "excluded": int((target & ~included).sum()),
        },
        "non_target": {
            "total": int((~target).sum()),
            "included": int((~target & included).sum()),
            "excluded": int((~target & ~included).sum()),
        },
        "realized_eligibility_equals_target_indicator": bool(target.equals(included)),
        "realized_positive_field_nesting": {
            "every_target_is_active": bool(
                frame.loc[target, "business_status"].eq("active").all()
            ),
            "every_target_is_plausible": bool(
                frame.loc[target, "candidate_plausibility"].eq("plausible").all()
            ),
            "every_target_is_identity_consistent": bool(
                frame.loc[target, "identity_consistency"].eq("consistent").all()
            ),
        },
        "interpretation": (
            "The reducer remains a four-field conjunction. In this realized candidate "
            "frame, target-construct status perfectly partitions include versus exclude "
            "because every target-pharmacy row is also active, plausible, and identity-"
            "consistent, and every non-target row is excluded."
        ),
        "pairwise_crosstabs": crosstabs,
    }
    checks = [
        *field_bindings.values(),
        audit["recorded_state_equals_independent_four_field_reduction"],
        audit["recorded_reason_codes_equal_independent_reduction"],
    ]
    if not all(checks):
        raise ValueError("forced-choice field/reducer audit failed")
    return audit, key_table


def _group_distribution(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(int(item) for item in values).items())
    }


def collapsed_location_sensitivity(
    google_primary: pd.DataFrame,
    board_primary: pd.DataFrame,
    delivery: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Collapse exact address-plus-ZIP groups and rerun the primary assignment."""
    primary = google_primary.copy()
    primary["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(primary["Street_Address"], primary["Zip_Code"])
    ]
    if primary["location_key"].eq("|").any():
        raise ValueError("blank normalized location key cannot define this sensitivity")

    rows: list[dict[str, Any]] = []
    group_members: dict[str, pd.DataFrame] = {}
    for location_key, members in primary.groupby("location_key", sort=True):
        ordered = members.sort_values("place_id").reset_index(drop=True)
        parsed_addresses = {
            geo_match.parse_address(str(value)) for value in ordered["Street_Address"]
        }
        zip_values = {str(value)[:5] for value in ordered["Zip_Code"]}
        if len(parsed_addresses) != 1 or len(zip_values) != 1:
            raise ValueError("a collapsed group does not retain one linkage address and ZIP")
        names = tuple(sorted({str(value) for value in ordered["Company"] if str(value).strip()}))
        if not names:
            raise ValueError("collapsed group lacks a public-name representation")
        group_members[location_key] = ordered
        representative = ordered.iloc[0]
        rows.append(
            {
                "location_key": location_key,
                "location_group_sha256": _safe_hash(location_key),
                "Company": names[0],
                "candidate_names": names,
                "Street_Address": representative["Street_Address"],
                "Zip_Code": str(representative["Zip_Code"])[:5],
                "Latitude": representative["Latitude"],
                "Longitude": representative["Longitude"],
                "place_id_count": len(ordered),
                "public_name_variant_count": len(names),
            }
        )
    collapsed = pd.DataFrame(rows).sort_values("location_key").reset_index(drop=True)
    matrix = geo_match.score_matrix(
        collapsed, board_primary, {"w_name": 1.0, "w_addr": 0.8}
    )
    pairs = geo_match.assign_hungarian(
        matrix, PRIMARY_THRESHOLD, objective="max_weight"
    )
    linked_group_indices = {left for left, _, _ in pairs}
    linked_board_indices = {right for _, right, _ in pairs}

    links = pd.read_csv(delivery / "board_links.csv", dtype=str).fillna("")
    primary_links = links.loc[links["stratum"].eq(PRIMARY_STRATUM)].copy()
    if len(primary_links) != 146 or primary_links["place_id"].duplicated().any():
        raise ValueError("primary linkage artifact changed")
    primary_linked_ids = set(primary_links["place_id"])
    primary_linked_board = set(primary_links["license_nbr"])
    id_to_key = dict(zip(primary["place_id"], primary["location_key"]))
    primary_linked_keys = {id_to_key[value] for value in primary_linked_ids}
    primary_location_board_pairs = {
        (id_to_key[row.place_id], str(row.license_nbr))
        for row in primary_links.itertuples(index=False)
    }
    primary_unmatched = primary.loc[~primary["place_id"].isin(primary_linked_ids)]
    if len(primary_unmatched) != 193:
        raise ValueError("primary unmatched workflow denominator changed")
    collapsed_linked_keys = set(collapsed.iloc[list(linked_group_indices)]["location_key"])
    collapsed_linked_board = {
        str(board_primary.iloc[index]["license_nbr"]) for index in linked_board_indices
    }
    collapsed_location_board_pairs = {
        (
            str(collapsed.iloc[left]["location_key"]),
            str(board_primary.iloc[right]["license_nbr"]),
        )
        for left, right, _ in pairs
    }

    detail_rows = []
    for index, row in collapsed.iterrows():
        members = group_members[row["location_key"]]
        primary_linked_members = int(members["place_id"].isin(primary_linked_ids).sum())
        original_unmatched_members = len(members) - primary_linked_members
        assigned = next((pair for pair in pairs if pair[0] == index), None)
        detail_rows.append(
            {
                "location_group_sha256": row["location_group_sha256"],
                "place_id_count": int(row["place_id_count"]),
                "public_name_variant_count": int(row["public_name_variant_count"]),
                "primary_linked_member_count": primary_linked_members,
                "original_unmatched_member_count": original_unmatched_members,
                "collapsed_linked": assigned is not None,
                "collapsed_score": round(float(assigned[2]), 6) if assigned else "",
            }
        )
    detail = pd.DataFrame(detail_rows)

    disappearing_into_already_linked = int(
        primary_unmatched["location_key"].isin(primary_linked_keys).sum()
    )
    original_unmatched_in_collapsed_linked = int(
        primary_unmatched["location_key"].isin(collapsed_linked_keys).sum()
    )
    multi = collapsed["place_id_count"].gt(1)
    audit = {
        "role": "post hoc outlet-location granularity sensitivity; primary remains Place ID",
        "frame": "pre-specified 60 target ZIPs",
        "collapse_key": "exact normalized full returned address plus five-digit returned ZIP",
        "linkage": {
            "components": "same name-plus-address scoring",
            "threshold": PRIMARY_THRESHOLD,
            "assignment": "Hungarian one-to-one",
            "objective": "max_weight",
            "name_handling": (
                "all distinct public-name variants from constituent Place IDs retained as "
                "aliases; the strongest compatible name score contributes for each Board edge"
            ),
        },
        "primary_place_id_reference": {
            "workflow_rows": len(primary),
            "screened_board_records": len(board_primary),
            "linked_pairs": len(primary_links),
            "unmatched_workflow_rows": len(primary) - len(primary_links),
            "unmatched_board_records": len(board_primary) - len(primary_links),
            "workflow_linked_share_pct": pct(len(primary_links), len(primary)),
            "board_reference_coverage_pct": pct(len(primary_links), len(board_primary)),
        },
        "collapsed_outlet_locations": {
            "workflow_locations": len(collapsed),
            "screened_board_records": len(board_primary),
            "linked_pairs": len(pairs),
            "unmatched_workflow_locations": len(collapsed) - len(pairs),
            "unmatched_board_records": len(board_primary) - len(pairs),
            "workflow_linked_share_pct": pct(len(pairs), len(collapsed)),
            "board_reference_coverage_pct": pct(len(pairs), len(board_primary)),
            "multi_place_id_locations": int(multi.sum()),
            "single_place_id_locations": int((~multi).sum()),
            "place_ids_per_location_distribution": _group_distribution(
                collapsed["place_id_count"]
            ),
            "public_name_variants_per_location_distribution": _group_distribution(
                collapsed["public_name_variant_count"]
            ),
            "maximum_place_ids_at_one_location": int(collapsed["place_id_count"].max()),
        },
        "original_193_unmatched_rows": {
            "disappear_by_collapsing_into_location_with_primary_linked_member": disappearing_into_already_linked,
            "belong_to_a_location_linked_in_collapsed_sensitivity": original_unmatched_in_collapsed_linked,
            "remain_in_locations_unlinked_in_collapsed_sensitivity": len(primary_unmatched)
            - original_unmatched_in_collapsed_linked,
        },
        "board_coverage_set_comparison": {
            "covered_in_both": len(primary_linked_board & collapsed_linked_board),
            "covered_only_in_primary_place_id_assignment": len(
                primary_linked_board - collapsed_linked_board
            ),
            "covered_only_in_collapsed_assignment": len(
                collapsed_linked_board - primary_linked_board
            ),
        },
        "location_to_board_pair_comparison": {
            "primary_unique_location_board_pairs": len(primary_location_board_pairs),
            "collapsed_location_board_pairs": len(collapsed_location_board_pairs),
            "identical_pair_sets": primary_location_board_pairs
            == collapsed_location_board_pairs,
            "pairs_only_in_primary": len(
                primary_location_board_pairs - collapsed_location_board_pairs
            ),
            "pairs_only_in_collapsed": len(
                collapsed_location_board_pairs - primary_location_board_pairs
            ),
        },
        "score_summary": {
            "minimum": round(min(score for _, _, score in pairs), 4),
            "median": round(float(np.median([score for _, _, score in pairs])), 4),
            "mean": round(float(np.mean([score for _, _, score in pairs])), 4),
            "maximum": round(max(score for _, _, score in pairs), 4),
        },
        "privacy": (
            "Constituent Place IDs and public names were used from authenticated local "
            "inputs but are not duplicated in this derived report; group identifiers are hashes."
        ),
    }
    if len(primary) != 339 or len(board_primary) != 178:
        raise ValueError("primary sensitivity frame changed")
    return audit, detail


def pilot_boundary_audit(base: Mapping[str, Any]) -> dict[str, Any]:
    pilot = base["formal_pilot"]
    validation = pilot["validation"]
    validation_ok = (
        validation["rows"] == 24
        and validation["successful_outputs"] == 24
        and validation["validation_passes"] == 24
        and validation["candidate_binding_passes"] == 24
        and validation["technical_failures"] == 0
        and validation["board_separation_passed"] is True
        and validation["promotion_passed"] is True
    )
    if not validation_ok:
        raise ValueError("formal A/B/C pilot audit failed")
    field_changes = pilot["pairwise_changes"]
    changed_counts = [
        detail["changed_candidates"]
        for comparison in field_changes.values()
        for key, detail in comparison.items()
        if key != "any_core_field_changed_candidates"
    ]
    any_core_counts = [
        comparison["any_core_field_changed_candidates"]
        for comparison in field_changes.values()
    ]
    if any(changed_counts) or any(any_core_counts):
        raise ValueError("formal pilot now contains a field-level change")
    return {
        "candidates": 8,
        "conditions": list(CONDITIONS),
        "validated_outputs": 24,
        "validation_passed": 24,
        "technical_failures": 0,
        "field_level_changes": field_changes,
        "eligibility_changes": 0,
        "unresolved_state_changes": 0,
        "supported_interpretation": (
            "The purposive pilot verifies executable A/B/C evidence separation and shows "
            "that the added evidence did not change recorded decisions for these eight "
            "candidates. It does not estimate population accuracy or demonstrate that an "
            "evidence layer improves classification performance."
        ),
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    collapsed = audit["collapsed_location_sensitivity"]
    primary = collapsed["primary_place_id_reference"]
    location = collapsed["collapsed_outlet_locations"]
    field = audit["forced_choice_field_dependence"]
    pilot = audit["evidence_ablation_boundary"]
    figures = audit["figure_readability_audit"]
    return "\n".join(
        [
            "# Final Results closure audit",
            "",
            f"Frozen forced-choice run: `{audit['run_id']}`. This report uses local authenticated artifacts only.",
            "",
            "## Place-ID and outlet-location correspondence",
            "",
            "| Unit | Workflow denominator | Links | Unmatched workflow | Board denominator | Unmatched Board | Linked share | Board coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Primary Place ID | {primary['workflow_rows']} | {primary['linked_pairs']} | {primary['unmatched_workflow_rows']} | {primary['screened_board_records']} | {primary['unmatched_board_records']} | {primary['workflow_linked_share_pct']:.2f}% | {primary['board_reference_coverage_pct']:.2f}% |",
            f"| Collapsed address + ZIP sensitivity | {location['workflow_locations']} | {location['linked_pairs']} | {location['unmatched_workflow_locations']} | {location['screened_board_records']} | {location['unmatched_board_records']} | {location['workflow_linked_share_pct']:.2f}% | {location['board_reference_coverage_pct']:.2f}% |",
            "",
            f"The collapsed frame contains {location['multi_place_id_locations']} multi-Place-ID locations. Of the primary analysis's 193 unmatched rows, {collapsed['original_193_unmatched_rows']['disappear_by_collapsing_into_location_with_primary_linked_member']} collapse into a location that already contained a primary linked Place ID.",
            "",
            "## Forced-choice field dependence",
            "",
            f"All {field['target_pharmacy']['total']} target-pharmacy rows were included; target exclusions = {field['target_pharmacy']['excluded']}; non-target inclusions = {field['non_target']['included']}. The recorded eligibility and reason codes reproduce the independent four-field reduction exactly.",
            "",
            "The reducer is a four-field conjunction. In this realized frame, target-construct status perfectly partitions eligibility because all target rows also have positive operating-status, plausibility, and identity judgments.",
            "",
            "## Evidence-ablation boundary",
            "",
            f"All {pilot['validation_passed']} of {pilot['validated_outputs']} pilot outputs passed deterministic validation; field, eligibility, and unresolved-state changes were all zero. {pilot['supported_interpretation']}",
            "",
            "## Figure readability",
            "",
            f"All {figures['figures_retained']} Results figures were inspected in the compiled PDF. Figures 4.1, 4.3, and 4.4 received layout-only enlargement; scientific content and values were unchanged.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"closure audit is frozen to {RUN_ID}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("closure-audit output directory already exists")

    run_root = ROOT / "data" / "v2" / RUN_ID
    delivery = run_root / "delivery"
    base = build_audit(RUN_ID)
    units = load_completed_units(run_root)
    google_all = build_google_frame(units, delivery)
    google_primary = (
        google_all.loc[
            google_all["eligibility_state"].eq("include")
            & google_all["Zip_Code"].isin(TARGET_ZIPS)
        ]
        .sort_values("place_id")
        .reset_index(drop=True)
    )
    board_all, _ = load_board()
    board_primary = (
        prepare_v2_board_frames(board_all, TARGET_ZIPS)
        .screened_target.sort_values("license_nbr")
        .reset_index(drop=True)
    )
    fields, field_table = field_dependence_audit(units, delivery)
    collapsed, collapsed_detail = collapsed_location_sensitivity(
        google_primary, board_primary, delivery
    )
    pilot = pilot_boundary_audit(base)
    if not THESIS_PDF.is_file():
        raise FileNotFoundError("compiled thesis PDF is unavailable for closure provenance")

    output.mkdir(parents=True)
    field_table.to_csv(output / "field_by_eligibility.csv", index=False, lineterminator="\n")
    collapsed_detail.to_csv(
        output / "collapsed_location_derived_detail.csv", index=False, lineterminator="\n"
    )
    report = {
        "schema_version": "nets-ai-results-closure-audit-v1",
        "run_id": RUN_ID,
        "scope": "local deterministic closure audit; no provider or model calls",
        "provenance": {
            "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
            "delivery_manifest_sha256": sha256_file(delivery / "run_manifest.json"),
            "forced_choice_classification_sha256": sha256_file(
                run_root / "classification_forced_choice.jsonl"
            ),
            "provider_calls": False,
            "classification_changes": 0,
            "primary_linkage_replaced": False,
        },
        "primary_place_id_correspondence": collapsed["primary_place_id_reference"],
        "collapsed_location_sensitivity": collapsed,
        "forced_choice_field_dependence": fields,
        "evidence_ablation_boundary": pilot,
        "figure_readability_audit": {
            "review_method": "manual visual inspection of locally rasterized compiled-PDF pages",
            "compiled_pdf_sha256": sha256_file(THESIS_PDF),
            "figures_retained": 6,
            "figures_reviewed": ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"],
            "layout_only_changes": {
                "4.1": "taller portrait-page canvas, larger labels, and wrapped box text",
                "4.3": "taller full-width canvas and separated percentage labels",
                "4.4": "stacked source panels and labels placed beyond interval bars",
            },
            "unchanged_figures": ["4.2", "4.5", "4.6"],
            "scientific_content_changed": False,
            "figure_count_changed": False,
            "observed_clipping_or_overlap_after_revision": False,
        },
    }
    _write_json(output / "closure_audit.json", report)
    (output / "closure_audit.md").write_text(
        render_markdown(report), encoding="utf-8", newline=""
    )
    artifacts = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "nets-ai-results-closure-audit-manifest-v1",
            "run_id": RUN_ID,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "semantic_manifest_sha256": _safe_hash(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":"))
            ),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
