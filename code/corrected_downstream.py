"""Isolated, provider-free downstream phases for the corrected evidence run.

The caller authenticates the fixed September 2 Place-ID set and the complete new
classifier packets before entering these functions.  No historical pilot or
classification is used.  Existing scoring, bootstrap and proximity algorithms
are reused; their old run-specific entry points are deliberately not invoked.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_forced_choice_results as audit
import audit_stage_specific_board_linkage as stage
import finalize_forced_choice_results as finalizer
import forced_choice_delivery as delivery
from forced_choice_models import CORE_FIELDS
from study_design import TARGET_ZIPS
from v2_board_adapter import LinkageConfigV2, prepare_v2_board_frames, run_v2_board_linkage
from v2_delivery import (
    _json_domain,
    _safe_tract_frame,
    _validated_spatial_summary,
    _write_atomic_package,
    _write_csv,
)
from v2_spatial_adapter import SpatialContractV2, run_v2_spatial_analysis


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "nets-ai-corrected-downstream-v1"
FROZEN_CANDIDATE_COUNT = 1226
FROZEN_ORIGIN_COUNT = 327
HISTORICAL_RUN_ID = "20260902T012636Z"
V1_GEOGRAPHY_SOURCE = "data/Minneapolis-StPaul_pharmacy_20260413_201606.csv"
IMPLEMENTATION_FILES = (
    "code/corrected_downstream.py",
    "code/forced_choice_delivery.py",
    "code/forced_choice_models.py",
    "code/forced_choice_eligibility.py",
    "code/forced_choice_classifier.py",
    "code/v2_board_adapter.py",
    "code/v2_spatial_adapter.py",
    "code/v2_delivery.py",
    "code/v2_downstream_models.py",
    "code/board_google_comparison.py",
    "code/geo_match.py",
    "code/study_design.py",
    "code/audit_forced_choice_results.py",
    "code/audit_stage_specific_board_linkage.py",
    "code/finalize_forced_choice_results.py",
    "analysis/v2/results_audit_20260831T230500Z/recompute_results_audit.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY = "include_screened_primary"
_RAW = "all_candidates_screened_retrieval"


def _validate_inputs(candidates, context, target_zips, output_root):
    candidates = tuple(candidates)
    if len(candidates) != FROZEN_CANDIDATE_COUNT:
        raise ValueError("corrected downstream requires all 1,226 fixed candidates")
    if any(not isinstance(item, delivery.ForcedChoiceDeliveryCandidate) for item in candidates):
        raise TypeError("corrected downstream requires typed delivery candidates")
    if len({item.location.place_id for item in candidates}) != len(candidates):
        raise ValueError("corrected downstream Place IDs are not unique")
    if len({item.location.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("corrected downstream candidate IDs are not unique")
    if any(item.decision is None or item.eligibility.eligibility_state == "unresolved" for item in candidates):
        raise ValueError("technical-unresolved candidates must be reported before downstream promotion")
    if not isinstance(context, delivery.ForcedChoiceLocalContext):
        raise TypeError("corrected downstream requires authenticated local context")
    if len(context.origins) != FROZEN_ORIGIN_COUNT or context.origins["GEOID"].duplicated().any():
        raise ValueError("fixed 327 tract-origin frame changed")
    if set(target_zips) != set(TARGET_ZIPS) or len(target_zips) != len(TARGET_ZIPS):
        raise ValueError("fixed 60 target ZIP definitions changed")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    resolved = output_root.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("corrected output must remain within the project")
    if HISTORICAL_RUN_ID in str(resolved):
        raise ValueError("historical September 2 artifacts must remain untouched")
    if resolved.exists():
        raise FileExistsError("corrected downstream output already exists")
    return tuple(sorted(candidates, key=lambda item: item.location.place_id))


def implementation_hashes() -> dict[str, str]:
    """Files the phase runner must pin before empirical execution/resumption."""
    return {relative: delivery._sha256_file(ROOT / relative) for relative in IMPLEMENTATION_FILES}


def _source_hashes(context, supplied, *, geography=False):
    combined = dict(context.source_hashes)
    for relative in ("analysis/input_manifest.json", *delivery._LOCAL_INPUT_PATHS):
        if combined.get(relative) != delivery._sha256_file(ROOT / relative):
            raise ValueError(f"authenticated local context source changed: {relative}")
    for key, digest in dict(supplied).items():
        if key in combined and combined[key] != digest:
            raise ValueError(f"conflicting source hash: {key}")
        combined[key] = digest
    for key, digest in implementation_hashes().items():
        if key in combined and combined[key] != digest:
            raise ValueError(f"downstream implementation changed: {key}")
        combined[key] = digest
    if geography:
        manifest_path = ROOT / "analysis/input_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {row["path"]: row for row in manifest["inputs"]}
        entry = entries[V1_GEOGRAPHY_SOURCE]
        digest = delivery._sha256_file(ROOT / V1_GEOGRAPHY_SOURCE)
        if digest != entry["sha256"]:
            raise ValueError("frozen v1 geography source failed authentication")
        combined[V1_GEOGRAPHY_SOURCE] = digest
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in combined.values()):
        raise ValueError("source hashes must be SHA-256 digests")
    return dict(sorted(combined.items()))


def _manifest(candidates, context, metadata, hashes, phase):
    contract = SpatialContractV2()
    return {
        "schema_version": CONTRACT_VERSION,
        "phase": phase,
        "status": "linkage_phase_complete" if phase == 3 else "spatial_phase_complete",
        "candidate_count": len(candidates),
        "origin_count": len(context.origins),
        "eligibility_counts": dict(sorted(Counter(item.eligibility.eligibility_state for item in candidates).items())),
        "technical_unresolved_count": 0,
        "runtime_failure_count": 0,
        "run_metadata": _json_domain(dict(metadata), "run metadata"),
        "source_hashes": hashes,
        "candidate_lineage_sha256": delivery._canonical_sha256([item.lineage_record() for item in candidates]),
        "contracts": {
            "classifier_contract_version": delivery.FORCED_CHOICE_CONTRACT_VERSION,
            "classifier_contract_sha256": delivery.FORCED_CHOICE_CONTRACT_SHA256,
            "classifier_schema_version": delivery.FORCED_CHOICE_SCHEMA_VERSION,
            "prompt_template_version": delivery.PROMPT_VERSION,
            "prompt_template_sha256": delivery.PROMPT_TEMPLATE_SHA256,
            "eligibility_contract_version": delivery.ELIGIBILITY_CONTRACT_VERSION,
            "eligibility_contract_sha256": delivery.ELIGIBILITY_CONTRACT_SHA256,
            "board_adapter_version": delivery.LINKAGE_ADAPTER_VERSION,
            "spatial_adapter_version": delivery.SPATIAL_ADAPTER_VERSION,
            "spatial_contract_sha256": contract.contract_sha256,
            "spatial_thresholds_miles": list(contract.thresholds_miles),
        },
        "board_separation": {
            "board_enters_classifier": False,
            "board_enters_eligibility_reducer": False,
            "board_first_stage": "phase_3_board_preparation",
        },
        "privacy_contract": {
            "scope": "this derived downstream package only",
            "raw_google_or_archived_text_retained": False,
            "public_name_address_coordinates_retained": False,
            "normalized_name_address_keys_hashed": True,
            "complete_classifier_packets_retained_by_upstream_run": True,
        },
        "pilot_requirement": "none; complete corrected single-model frame authenticated upstream",
        "name_recurrence_contract": {"ownership_inference": False, "bootstrap_draws": 5000, "thresholds": [2, 3, 5]},
    }


def _google_and_boards(candidates, context, target_zips):
    google, _ = delivery._prepare_google_frame(candidates)
    labels = delivery._eligibility_frame(candidates)
    columns = ["place_id", *CORE_FIELDS, *(f"{field}_confidence" for field in CORE_FIELDS)]
    google = google.merge(labels[columns], on="place_id", validate="one_to_one", sort=False)
    boards = prepare_v2_board_frames(
        context.board, target_zips,
        observed_google_zips=tuple(google.loc[google["Zip_Code"].ne(""), "Zip_Code"]),
    )
    return google, boards


def _safe_recurrence_detail(frame):
    result = frame.copy()
    for column in ("name_signature", "name_component_id", "location_key"):
        result[f"{column}_sha256"] = result[column].map(
            lambda value: hashlib.sha256(f"{column}\0{value}".encode("utf-8")).hexdigest()
        )
        result = result.drop(columns=column)
    return result


def _finite_recurrence_summary(summary):
    """Represent empty-group gaps without changing the inherited estimates."""
    result = dict(summary)
    if "repeated_minus_below_threshold_gaps" not in result:
        return _json_domain(result, "name recurrence summary")
    key_fields = ("source", "linkage_strategy", "recurrence_threshold")
    denominators = {
        (*tuple(row[field] for field in key_fields), row["group_code"]): row["records"]
        for row in result["rows"]
    }
    gaps = []
    for original in result["repeated_minus_below_threshold_gaps"]:
        gap = dict(original)
        value = gap["repeated_minus_below_threshold_percentage_points"]
        if value is None or not np.isfinite(value):
            key = tuple(gap[field] for field in key_fields)
            counts = {group: denominators[(*key, group)] for group in ("repeated", "below_threshold")}
            if any(count < 0 for count in counts.values()) or not any(count == 0 for count in counts.values()):
                raise ValueError("undefined gap for nonempty recurrence groups")
            gap.update({
                "repeated_minus_below_threshold_percentage_points": None,
                "estimable": False,
                "non_estimable_reason": "empty_repeated_or_below_threshold_group",
                "group_denominators": counts,
            })
        gaps.append(gap)
    result["repeated_minus_below_threshold_gaps"] = gaps
    return _json_domain(result, "name recurrence summary")


def _collapsed(frame):
    if len(frame):
        return stage.collapse_locations(frame)
    return (
        pd.DataFrame(columns=["_location_key", "location_group_sha256", "Company", "candidate_names", "Street_Address", "Zip_Code", "Latitude", "Longitude", "constituent_place_ids", "place_id_count", "public_name_variant_count"]),
        pd.DataFrame(columns=["location_group_sha256", "constituent_place_ids", "place_id_count", "public_name_variant_count", "public_name_variant_sha256s"]),
    )


def _candidate_cross_tab(raw, matrix, board, included_pairs):
    qualified = (matrix >= stage.PRIMARY_THRESHOLD).any(axis=1)
    excluded_qualified = qualified & raw["eligibility_state"].eq("exclude").to_numpy()
    if excluded_qualified.any():
        return stage.candidate_cross_tab_and_excluded_detail(raw, matrix, board, included_pairs)
    # The historical helper assumes at least one excluded qualified candidate.
    # An empty group is a valid corrected result, with an explicit zero denominator.
    rows = []
    for has_edge in (True, False):
        for state_value in ("include", "exclude"):
            count = int(((qualified == has_edge) & raw["eligibility_state"].eq(state_value).to_numpy()).sum())
            rows.append({
                "candidate_category": f"{'any' if has_edge else 'no'} qualified Board edge + LLM/reducer {state_value}",
                "count": count,
                "pct_of_raw_in_frame_candidates": stage.pct(count, len(raw)),
            })
    detail = pd.DataFrame(columns=["place_id", "candidate_id", *CORE_FIELDS, *(f"{field}_confidence" for field in CORE_FIELDS), "maximum_board_linkage_score", "maximum_score_board_license_nbr", "qualified_board_record_count", "qualified_board_license_nbrs", "shares_normalized_address_zip_with_included_candidate", "maximum_score_board_covered_by_included_assignment", "any_qualified_board_covered_by_included_assignment", "all_qualified_boards_covered_by_included_assignment"])
    return pd.DataFrame(rows), detail, {"excluded_candidates_with_any_qualified_board_edge": 0, "denominator": 0, "interpretation": "No excluded candidate has a qualified Board edge under the declared rule."}


def _stage_for_units(raw, included, board, raw_matrix, included_matrix, *, unit):
    raw_pairs, raw_records = stage.assignment_records(raw, board, raw_matrix)
    included_pairs, included_records = stage.assignment_records(included, board, included_matrix)
    stages, details, core = stage.board_stage_decomposition(
        board, raw_matrix, included_matrix, raw, included, raw_pairs, included_pairs,
        unit_term="candidate" if unit == "place_id" else "location",
    )
    for row in stages:
        row["pct_of_board_records"] = row.pop("pct_of_178_board_records")
    identity, identity_detail = stage.board_assignment_identity_comparison(board, raw_pairs, included_pairs)
    endpoint, endpoint_detail = stage.linked_board_endpoint_stability(
        board, raw, included, raw_pairs, included_pairs,
        identifier_column="place_id" if unit == "place_id" else "location_group_sha256",
    )
    raw_assigned = {right for _, right, _ in raw_pairs}
    included_assigned = {right for _, right, _ in included_pairs}
    retained = len(raw_assigned & included_assigned)
    core["board_referenced_assignment_retention"] = {
        "numerator": retained,
        "denominator": len(raw_assigned),
        "pct": stage.pct(retained, len(raw_assigned)),
        "definition": "Board identities covered by both assignments divided by identities covered by raw assignment",
    }
    core["board_referenced_assignment_attrition"] = {
        "numerator": len(raw_assigned - included_assigned),
        "denominator": len(raw_assigned),
        "pct": stage.pct(len(raw_assigned - included_assigned), len(raw_assigned)),
    }
    summaries = [
        stage.assignment_summary(f"raw_{unit}", raw, board, raw_pairs),
        stage.assignment_summary(f"included_{unit}", included, board, included_pairs),
    ]
    report = {"assignments": summaries, "board_stages": stages, "core_metrics": core, "board_assignment_identities": identity, "endpoint_stability": endpoint}
    frames = {
        f"{unit}_assignment_summary.csv": pd.DataFrame(stage.flat_assignment_rows(summaries)),
        f"{unit}_board_stages.csv": pd.DataFrame(stages),
        f"{unit}_board_stage_detail.csv": details,
        f"{unit}_board_assignment_identities.csv": identity_detail,
        f"{unit}_changed_assignment_endpoints.csv": endpoint_detail,
        f"{unit}_raw_assignments.csv": pd.DataFrame(raw_records, columns=["left_identifier", "license_nbr", "composite_score"]),
        f"{unit}_included_assignments.csv": pd.DataFrame(included_records, columns=["left_identifier", "license_nbr", "composite_score"]),
    }
    return report, frames, raw_pairs, included_pairs, raw_records, included_records


def stage_specific_diagnostic(google, board, links, target_zips):
    """Recompute raw/filter diagnostics, including changed assignment identities."""
    google = google.sort_values("place_id").reset_index(drop=True)
    board = board.sort_values("license_nbr").reset_index(drop=True)
    if board.empty or board["license_nbr"].duplicated().any():
        raise ValueError("screened Board reference must be nonempty and unique")
    if not set(google["eligibility_state"]) <= {"include", "exclude"}:
        raise ValueError("stage diagnostic requires reconciled substantive decisions")
    raw = google.loc[google["Zip_Code"].isin(target_zips)].copy().reset_index(drop=True)
    included = raw.loc[raw["eligibility_state"].eq("include")].copy().reset_index(drop=True)
    all_matrix, _, _, _ = stage.score_with_components(google, board)
    position = {value: index for index, value in enumerate(google["place_id"])}
    raw_matrix = all_matrix[[position[value] for value in raw["place_id"]], :]
    included_matrix = all_matrix[[position[value] for value in included["place_id"]], :]
    report, frames, raw_pairs, included_pairs, raw_records, included_records = _stage_for_units(
        raw, included, board, raw_matrix, included_matrix, unit="place_id",
    )
    checks = {}
    for stratum, records in ((_RAW, raw_records), (_PRIMARY, included_records)):
        observed = {(row["left_identifier"], row["license_nbr"]): round(row["composite_score"], 4) for row in records}
        expected = {(str(row.place_id), str(row.license_nbr)): round(float(row.composite_score), 4) for row in links.loc[links["stratum"].eq(stratum)].itertuples(index=False)}
        checks[f"{stratum}_assignments_match_delivery"] = observed == expected
    if not all(checks.values()):
        raise ValueError("stage diagnostic assignments differ from corrected linkage")
    cross, excluded, excluded_summary = _candidate_cross_tab(raw, raw_matrix, board, included_pairs)
    frames.update({"candidate_qualified_edge_cross_tab.csv": cross, "excluded_qualified_candidate_detail.csv": excluded})
    raw_locations, raw_safe = _collapsed(raw)
    included_locations, included_safe = _collapsed(included)
    raw_location_matrix, _, _, _ = stage.score_with_components(raw_locations, board)
    included_location_matrix, _, _, _ = stage.score_with_components(included_locations, board)
    location_report, location_frames, *_ = _stage_for_units(
        raw_locations, included_locations, board, raw_location_matrix, included_location_matrix, unit="address_zip_location",
    )
    frames.update(location_frames)
    frames["raw_location_groups.csv"] = raw_safe
    frames["included_location_groups.csv"] = included_safe
    return {
        "candidate_count_scored": len(google),
        "raw_in_frame_count": len(raw),
        "outside_or_missing_returned_zip_count": len(google) - len(raw),
        "included_in_frame_count": len(included),
        "board_reference_count": len(board),
        "place_id_diagnostic": report,
        "address_zip_location_diagnostic": location_report,
        "excluded_qualified_candidate_diagnostic": excluded_summary,
        "checks": checks,
        "interpretation": "Descriptive retrieval, filtering and assignment diagnostics against the operational Board reference; not classification accuracy or a causal effect.",
    }, frames


def run_linkage_phase(
    candidates: Sequence[delivery.ForcedChoiceDeliveryCandidate], *,
    context: delivery.ForcedChoiceLocalContext,
    target_zips: Sequence[str],
    run_metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_root: Path,
) -> dict[str, Any]:
    """Phase 3: RQ1, Board linkage, recurrence and all declared diagnostics."""
    candidates = _validate_inputs(candidates, context, target_zips, output_root)
    hashes = _source_hashes(context, source_hashes, geography=True)
    google, boards = _google_and_boards(candidates, context, target_zips)
    config = LinkageConfigV2()
    linkage = run_v2_board_linkage(google, boards, config)
    recurrence_detail, recurrence_summary = delivery._name_recurrence(google, boards, config)
    recurrence_summary = _finite_recurrence_summary(recurrence_summary)
    eligibility = delivery._eligibility_frame(candidates)
    confidence_frame, confidence_summary = delivery._confidence_correspondence(candidates, linkage.links)
    board_primary = boards.screened_target.sort_values("license_nbr").reset_index(drop=True)
    google_primary = google.loc[google["eligibility_state"].eq("include") & google["Zip_Code"].isin(target_zips)].copy().reset_index(drop=True)
    sensitivity = finalizer.run_linkage_sensitivities(google_primary, board_primary)
    v1 = pd.read_csv(ROOT / V1_GEOGRAPHY_SOURCE, dtype={"Zip_Code": str}, low_memory=False)
    geography, geography_rows = finalizer.geography_sensitivity(google, context.board, v1)
    stage_report, stage_frames = stage_specific_diagnostic(google, board_primary, linkage.links, target_zips)
    artifacts = {
        "candidate_inventory.csv": ("csv", delivery._candidate_inventory(candidates)),
        "eligibility.csv": ("csv", eligibility),
        "unresolved_cases.csv": ("csv", eligibility.loc[eligibility["eligibility_state"].eq("unresolved")]),
        "board_links.csv": ("csv", linkage.links),
        "unmatched_google.csv": ("csv", linkage.unmatched_google),
        "unmatched_board.csv": ("csv", linkage.unmatched_board),
        "performance_metrics.json": ("json", linkage.summary),
        "name_recurrence_detail.csv": ("csv", _safe_recurrence_detail(recurrence_detail)),
        "name_recurrence_summary.json": ("json", recurrence_summary),
        "confidence_board_correspondence.csv": ("csv", confidence_frame),
        "confidence_board_correspondence.json": ("json", confidence_summary),
        "linkage_sensitivities.csv": ("csv", pd.DataFrame(sensitivity)),
        "geography_sensitivities.csv": ("csv", pd.DataFrame(geography_rows)),
        "geography_set_audit.json": ("json", geography),
        "stage_specific_diagnostic.json": ("json", stage_report),
        **{f"stage_specific/{name}": ("csv", frame) for name, frame in stage_frames.items()},
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corrected-rq1-", dir=output_root.parent) as directory:
        temporary = Path(directory)
        for name in ("eligibility.csv", "candidate_inventory.csv", "unresolved_cases.csv"):
            _write_csv(temporary / name, artifacts[name][1])
        rq1 = audit.audit_candidate_flow(temporary)
    artifacts["rq1_summary.json"] = ("json", rq1)
    manifest = _manifest(candidates, context, run_metadata, hashes, 3)
    manifest["checks"] = {
        "rq1_states_reconcile": rq1["checks"]["states_sum_to_total"],
        **stage_report["checks"],
        "recurrence_bootstrap_draws": 5000,
        "geography_sizes": [row["zip_count"] for row in geography_rows],
    }
    return _write_atomic_package(output_root, artifacts, manifest)


def run_spatial_phase(
    candidates: Sequence[delivery.ForcedChoiceDeliveryCandidate], *,
    context: delivery.ForcedChoiceLocalContext,
    target_zips: Sequence[str],
    run_metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_root: Path,
) -> dict[str, Any]:
    """Phase 4: unchanged 327-origin paired proximity and all sensitivities."""
    candidates = _validate_inputs(candidates, context, target_zips, output_root)
    hashes = _source_hashes(context, source_hashes)
    google, boards = _google_and_boards(candidates, context, target_zips)
    contract = SpatialContractV2()
    spatial = run_v2_spatial_analysis(google, boards, context.origins, contract, analysis_scope="full")
    summary = _validated_spatial_summary(spatial, contract)
    safe_tracts = _safe_tract_frame(spatial.tracts)
    # The frozen audit's pure spatial entry point reads only these staged files.
    # A fresh module instance avoids changing another caller's delivery globals.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corrected-rq3-", dir=output_root.parent) as directory:
        temporary = Path(directory)
        _write_csv(temporary / "tract_spatial.csv", safe_tracts)
        (temporary / "spatial_summary.json").write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
        base = audit._base_module()
        base.DELIVERY = temporary
        rq3 = base.audit_spatial()
        enrichment = finalizer.spatial_enrichment(temporary, rq3)
        extension = finalizer.spatial_sensitivity_extension(google, context.board, temporary)
    checks = [value for group in rq3["row_to_summary_checks"].values() for value in group.values()]
    if not all(checks) or extension["primary_delivery_recomputed_exactly"] is not True:
        raise ValueError("spatial audits do not reconcile with corrected delivery")
    signed = pd.DataFrame({
        "GEOID": safe_tracts["GEOID"],
        "workflow_minus_board_miles": (safe_tracts["google_include_nearest_m"] - safe_tracts["board_screened_nearest_m"]) / contract.meters_per_mile,
    })
    artifacts = {
        "tract_spatial.csv": ("csv", safe_tracts),
        "coordinate_qa.csv": ("csv", spatial.coordinate_qa),
        "coordinate_ledger.csv": ("csv", spatial.coordinate_ledger),
        "spatial_summary.json": ("json", summary),
        "rq3_summary.json": ("json", rq3),
        "spatial_enrichment.json": ("json", enrichment),
        "spatial_sensitivity_extension.json": ("json", extension),
        "signed_distances.csv": ("csv", signed),
    }
    manifest = _manifest(candidates, context, run_metadata, hashes, 4)
    manifest["origin_sha256"] = summary["origin_sha256"]
    manifest["checks"] = {"row_to_summary": rq3["row_to_summary_checks"], "primary_spatial_recomputed_exactly": True}
    return _write_atomic_package(output_root, artifacts, manifest)


__all__ = ["CONTRACT_VERSION", "IMPLEMENTATION_FILES", "implementation_hashes", "run_linkage_phase", "run_spatial_phase", "stage_specific_diagnostic"]
