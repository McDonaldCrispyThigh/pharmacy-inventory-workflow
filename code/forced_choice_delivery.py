"""Downstream delivery for the fixed-frame forced-choice inventory.

This module performs no provider calls.  It applies the new Board-blind reducer and
then reuses the existing Board screen, deterministic linkage, name-recurrence, and
paired straight-line spatial specifications.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from board_google_comparison import (
    _assign_exact_name_components,
    signature_analysis,
)
from forced_choice_classifier import (
    PROMPT_TEMPLATE_SHA256,
    PROMPT_VERSION,
)
from forced_choice_eligibility import (
    ELIGIBILITY_CONTRACT_SHA256,
    ELIGIBILITY_CONTRACT_VERSION,
    ForcedChoiceEligibilityDecision,
    reduce_forced_choice_eligibility,
)
from forced_choice_models import (
    CORE_FIELDS,
    FORCED_CHOICE_CONTRACT_SHA256,
    FORCED_CHOICE_CONTRACT_VERSION,
    FORCED_CHOICE_SCHEMA_VERSION,
    ForcedChoiceDecision,
)
from study_design import (
    normalize_legal_signature,
    normalize_name_signature,
    normalized_location_key,
)
from v2_board_adapter import (
    LINKAGE_ADAPTER_VERSION,
    LinkageConfigV2,
    _pairs,
    merge_v2_board_sources,
    prepare_v2_board_frames,
    run_v2_board_linkage,
)
from v2_delivery import (
    _safe_tract_frame,
    _validated_spatial_summary,
    _write_atomic_package,
)
from v2_downstream_models import CandidateLocationV2
from v2_spatial_adapter import (
    SPATIAL_ADAPTER_VERSION,
    SpatialContractV2,
    prepare_fixed_tract_origins,
    run_v2_spatial_analysis,
)


DELIVERY_CONTRACT_VERSION = "nets-ai-v2.2-forced-choice-delivery-v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_INPUT_PATHS = (
    "data/board_geocoded.csv",
    "data/Board/F.1  Pharmacy (PH).csv",
    "data/tl_2023_27_tract.zip",
    "data/tl_2023_us_zcta520.zip",
    "data/acs_tracts_2023.csv",
    "data/acs_tracts_vehicles_2023.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class ForcedChoiceDeliveryCandidate:
    location: CandidateLocationV2
    decision: ForcedChoiceDecision | None
    technical_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.location, CandidateLocationV2):
            raise TypeError("delivery candidate requires a typed location")
        if self.decision is None:
            if not isinstance(self.technical_status, str) or not self.technical_status:
                raise ValueError("missing decision requires a technical status")
        elif (
            not isinstance(self.decision, ForcedChoiceDecision)
            or self.decision.candidate_id != self.location.candidate_id
            or self.technical_status is not None
        ):
            raise ValueError("forced-choice delivery candidate binding is invalid")

    @property
    def eligibility(self) -> ForcedChoiceEligibilityDecision:
        return reduce_forced_choice_eligibility(
            self.decision,
            place_id=self.location.place_id,
            candidate_id=self.location.candidate_id,
        )

    def lineage_record(self) -> dict[str, Any]:
        return {
            "place_id": self.location.place_id,
            "candidate_id": self.location.candidate_id,
            "candidate_snapshot_sha256": self.location.candidate_snapshot_sha256,
            "classification_decision_sha256": (
                self.decision.decision_sha256 if self.decision is not None else None
            ),
        }


@dataclass(frozen=True, slots=True, repr=False)
class ForcedChoiceLocalContext:
    board: pd.DataFrame
    origins: gpd.GeoDataFrame
    source_hashes: Mapping[str, str]


def load_forced_choice_local_context(
    input_manifest_path: Path,
    target_zips: Sequence[str],
    *,
    project_root: Path = _PROJECT_ROOT,
) -> ForcedChoiceLocalContext:
    root = project_root.resolve()
    manifest_path = input_manifest_path.resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("input manifest escaped the project root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("inputs")
    if manifest.get("schema_version") != 1 or type(entries) is not list:
        raise ValueError("invalid input manifest")
    by_path = {entry.get("path"): entry for entry in entries if type(entry) is dict}
    paths: dict[str, Path] = {}
    hashes = {"analysis/input_manifest.json": _sha256_file(manifest_path)}
    for relative in _LOCAL_INPUT_PATHS:
        entry = by_path.get(relative)
        path = (root / relative).resolve()
        if (
            type(entry) is not dict
            or not path.is_relative_to(root)
            or not path.is_file()
            or entry.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"local delivery source failed authentication: {relative}")
        paths[relative] = path
        hashes[relative] = entry["sha256"]
    board_raw = pd.read_csv(
        paths["data/Board/F.1  Pharmacy (PH).csv"],
        encoding="utf-8-sig",
        skiprows=3,
        low_memory=False,
    )
    board_cache = pd.read_csv(paths["data/board_geocoded.csv"], low_memory=False)
    board = merge_v2_board_sources(board_raw, board_cache)
    tracts = gpd.read_file(f"zip://{paths['data/tl_2023_27_tract.zip'].as_posix()}")
    zctas = gpd.read_file(f"zip://{paths['data/tl_2023_us_zcta520.zip'].as_posix()}")
    acs = pd.read_csv(paths["data/acs_tracts_2023.csv"], dtype={"GEOID": str})
    vehicles = pd.read_csv(
        paths["data/acs_tracts_vehicles_2023.csv"], dtype={"GEOID": str}
    )
    origins = prepare_fixed_tract_origins(tracts, acs, vehicles, zctas, target_zips)
    if len(origins) != 327:
        raise ValueError("fixed tract-origin frame changed")
    return ForcedChoiceLocalContext(board, origins, hashes)


def _prepare_google_frame(
    candidates: Sequence[ForcedChoiceDeliveryCandidate],
) -> tuple[pd.DataFrame, list[ForcedChoiceEligibilityDecision]]:
    rows = []
    eligibility = []
    seen: set[str] = set()
    for item in candidates:
        location = item.location
        if location.place_id in seen:
            raise ValueError("delivery candidate Place IDs are not unique")
        seen.add(location.place_id)
        reduced = item.eligibility
        eligibility.append(reduced)
        rows.append(
            {
                "place_id": location.place_id,
                "candidate_id": location.candidate_id,
                "Company": location.name,
                "Street_Address": location.formatted_address,
                "Zip_Code": location.returned_postal_code or "",
                "zip5": location.returned_postal_code or "",
                "query_zip": location.query_provenance.source_zip,
                "query_definition_id": location.query_provenance.query_definition_id,
                "Latitude": location.latitude,
                "Longitude": location.longitude,
                "retrieved_at": location.retrieved_at.isoformat().replace("+00:00", "Z"),
                "eligibility_state": reduced.eligibility_state,
                "eligibility_reason_codes": reduced.reason_codes,
                "classification_decision_sha256": reduced.classification_decision_sha256,
                "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
                "eligibility_contract_version": reduced.contract_version,
                "eligibility_contract_sha256": reduced.contract_sha256,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("place_id").reset_index(drop=True),
        eligibility,
    )


def _city_from_address(value: object) -> str:
    parts = [piece.strip() for piece in str(value).split(",")]
    return parts[1] if len(parts) >= 3 else ""


def _name_recurrence(
    google: pd.DataFrame,
    boards: Any,
    linkage_config: LinkageConfigV2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ai = google.loc[
        google["eligibility_state"].eq("include")
        & google["Zip_Code"].isin(boards.target_zips)
    ].copy().sort_values("place_id").reset_index(drop=True)
    ai["ai_row_id"] = np.arange(len(ai), dtype=int)
    ai["signature_city"] = ai["Street_Address"].map(_city_from_address)
    ai["operating_signature"] = [
        normalize_name_signature(name, city)
        for name, city in zip(ai["Company"], ai["signature_city"])
    ]
    ai["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(ai["Street_Address"], ai["Zip_Code"])
    ]
    ai = _assign_exact_name_components(ai, operating_column="operating_signature")

    criterion = boards.screened_target.copy().sort_values("license_nbr").reset_index(drop=True)
    criterion["operating_signature"] = [
        normalize_name_signature(name, city)
        for name, city in zip(criterion["facility_name"], criterion["city"])
    ]
    criterion["legal_signature"] = criterion["facility_name"].map(
        normalize_legal_signature
    )
    criterion["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(criterion["address_line1"], criterion["zip5"])
    ]
    criterion = _assign_exact_name_components(
        criterion,
        operating_column="operating_signature",
        legal_column="legal_signature",
    )

    strategies: dict[str, list[tuple[int, int, float]]] = {}
    for label, threshold, objective in (
        ("name_address_hungarian_t75", 75.0, linkage_config.primary_objective),
        ("name_address_hungarian_t70", 70.0, linkage_config.primary_objective),
        ("name_address_hungarian_t80", 80.0, linkage_config.primary_objective),
        (
            "name_address_max_cardinality_then_weight_t75",
            75.0,
            linkage_config.sensitivity_objective,
        ),
    ):
        _, pairs = _pairs(
            ai,
            criterion,
            linkage_config,
            threshold=threshold,
            objective=objective,
        )
        strategies[label] = pairs
    detail, summary = signature_analysis(ai, criterion, strategies)
    old_source = "Google exact-address clusters"
    new_source = "Workflow inventory locations"
    detail["source"] = detail["source"].replace(old_source, new_source)
    for row in summary["rows"]:
        if row["source"] == old_source:
            row["source"] = new_source
    for row in summary["repeated_minus_below_threshold_gaps"]:
        if row["source"] == old_source:
            row["source"] = new_source
    summary["ownership_inference"] = False
    summary["primary_strategy"] = "name_address_hungarian_t75"
    return detail, summary


def _eligibility_frame(
    candidates: Sequence[ForcedChoiceDeliveryCandidate],
) -> pd.DataFrame:
    rows = []
    for item in candidates:
        reduced = item.eligibility
        row = reduced.to_public_dict()
        row["classification_status"] = (
            "success" if item.decision is not None else item.technical_status
        )
        for field in CORE_FIELDS:
            row[f"{field}_confidence"] = (
                item.decision.claim_confidence[field]
                if item.decision is not None else None
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("place_id").reset_index(drop=True)


def _candidate_inventory(
    candidates: Sequence[ForcedChoiceDeliveryCandidate],
) -> pd.DataFrame:
    rows = []
    for item in candidates:
        location = item.location
        reduced = item.eligibility
        rows.append(
            {
                "place_id": location.place_id,
                "candidate_id": location.candidate_id,
                "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
                "classification_decision_sha256": reduced.classification_decision_sha256,
                "query_definition_id": location.query_provenance.query_definition_id,
                "query_zip": location.query_provenance.source_zip,
                "returned_postal_code": location.returned_postal_code,
                "retrieved_at": location.retrieved_at.isoformat().replace("+00:00", "Z"),
                "coordinate_state": (
                    "complete" if location.latitude is not None else "missing"
                ),
                "classification_status": (
                    "success" if item.decision is not None else item.technical_status
                ),
                "eligibility_state": reduced.eligibility_state,
                "eligibility_reason_codes": list(reduced.reason_codes),
            }
        )
    return pd.DataFrame(rows).sort_values("place_id").reset_index(drop=True)


def _confidence_correspondence(
    candidates: Sequence[ForcedChoiceDeliveryCandidate],
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = links.loc[links["stratum"].eq("include_screened_primary")]
    matched = set(primary["place_id"].astype(str))
    rows = []
    for item in candidates:
        if item.decision is None or item.eligibility.eligibility_state != "include":
            continue
        row = {
            "place_id": item.location.place_id,
            "matched_to_screened_board_reference": item.location.place_id in matched,
        }
        for field in CORE_FIELDS:
            row[f"{field}_confidence"] = item.decision.claim_confidence[field]
        rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    if not frame.empty:
        for field in CORE_FIELDS:
            column = f"{field}_confidence"
            for confidence, group in frame.groupby(column, dropna=False):
                summaries.append(
                    {
                        "field": field,
                        "confidence": str(confidence),
                        "workflow_inventory_locations": len(group),
                        "matched_locations": int(
                            group["matched_to_screened_board_reference"].sum()
                        ),
                        "matched_share_pct": round(
                            100.0
                            * float(group["matched_to_screened_board_reference"].mean()),
                            2,
                        ),
                    }
                )
    return frame, {
        "scope": "included workflow inventory locations under primary linkage only",
        "interpretation": (
            "descriptive Board correspondence by model confidence; not an accuracy, "
            "calibration, precision, or recall estimate"
        ),
        "rows": summaries,
    }


def run_forced_choice_delivery(
    candidates: Sequence[ForcedChoiceDeliveryCandidate],
    *,
    context: ForcedChoiceLocalContext,
    target_zips: Sequence[str],
    ablation_rows: Sequence[Mapping[str, Any]],
    ablation_summary: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_root: Path,
) -> dict[str, Any]:
    candidates = tuple(candidates)
    if len(candidates) != 1226:
        raise ValueError("forced-choice delivery requires all 1,226 fixed candidates")
    if ablation_summary.get("promotion_passed") is not True:
        raise ValueError("forced-choice delivery requires a clean 24-output pilot")
    google, eligibility = _prepare_google_frame(candidates)
    returned_zips = tuple(
        item.location.returned_postal_code
        for item in candidates
        if item.location.returned_postal_code is not None
    )
    linkage_config = LinkageConfigV2()
    spatial_contract = SpatialContractV2()
    boards = prepare_v2_board_frames(
        context.board, target_zips, observed_google_zips=returned_zips
    )
    linkage = run_v2_board_linkage(google, boards, linkage_config)
    recurrence_detail, recurrence_summary = _name_recurrence(
        google, boards, linkage_config
    )
    spatial = run_v2_spatial_analysis(
        google,
        boards,
        context.origins,
        spatial_contract,
        analysis_scope="full",
    )
    spatial_summary = _validated_spatial_summary(spatial, spatial_contract)
    eligibility_frame = _eligibility_frame(candidates)
    unresolved = eligibility_frame.loc[
        eligibility_frame["eligibility_state"].eq("unresolved")
    ].reset_index(drop=True)
    confidence_frame, confidence_summary = _confidence_correspondence(
        candidates, linkage.links
    )
    artifacts = {
        "candidate_inventory.csv": ("csv", _candidate_inventory(candidates)),
        "eligibility.csv": ("csv", eligibility_frame),
        "unresolved_cases.csv": ("csv", unresolved),
        "board_links.csv": ("csv", linkage.links),
        "unmatched_google.csv": ("csv", linkage.unmatched_google),
        "unmatched_board.csv": ("csv", linkage.unmatched_board),
        "performance_metrics.json": ("json", linkage.summary),
        "name_recurrence_detail.csv": ("csv", recurrence_detail),
        "name_recurrence_summary.json": ("json", recurrence_summary),
        "confidence_board_correspondence.csv": ("csv", confidence_frame),
        "confidence_board_correspondence.json": ("json", confidence_summary),
        "ablation_results.jsonl": ("jsonl", tuple(dict(row) for row in ablation_rows)),
        "ablation_summary.json": ("json", dict(ablation_summary)),
        "tract_spatial.csv": ("csv", _safe_tract_frame(spatial.tracts)),
        "coordinate_qa.csv": ("csv", spatial.coordinate_qa),
        "coordinate_ledger.csv": ("csv", spatial.coordinate_ledger),
        "spatial_summary.json": ("json", spatial_summary),
    }
    lineage = [item.lineage_record() for item in candidates]
    manifest_base = {
        "schema_version": DELIVERY_CONTRACT_VERSION,
        "status": "full_delivery_complete",
        "candidate_count": len(candidates),
        "origin_count": len(spatial.tracts),
        "eligibility_counts": {
            str(key): int(value)
            for key, value in eligibility_frame["eligibility_state"].value_counts().items()
        },
        "technical_unresolved_count": len(unresolved),
        "runtime_failure_count": 0,
        "run_metadata": dict(run_metadata),
        "source_hashes": dict(sorted({**context.source_hashes, **source_hashes}.items())),
        "candidate_lineage_sha256": _canonical_sha256(
            sorted(lineage, key=lambda row: row["place_id"])
        ),
        "contracts": {
            "classifier_contract_version": FORCED_CHOICE_CONTRACT_VERSION,
            "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
            "classifier_schema_version": FORCED_CHOICE_SCHEMA_VERSION,
            "prompt_template_version": PROMPT_VERSION,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
            "board_adapter_version": LINKAGE_ADAPTER_VERSION,
            "spatial_adapter_version": SPATIAL_ADAPTER_VERSION,
            "spatial_contract_sha256": spatial_contract.contract_sha256,
            "spatial_thresholds_miles": list(spatial_contract.thresholds_miles),
        },
        "board_separation": {
            "board_enters_classifier": False,
            "board_enters_eligibility_reducer": False,
            "board_first_stage": "board_preparation",
        },
        "privacy_contract": {
            "prompt_retained": False,
            "raw_model_output_retained": False,
            "review_text_retained": False,
            "public_name_address_coordinates_retained": False,
        },
        "name_recurrence_contract": {
            "ownership_inference": False,
            "bootstrap_draws": 5000,
            "thresholds": [2, 3, 5],
        },
    }
    return _write_atomic_package(output_root, artifacts, manifest_base)


__all__ = [
    "DELIVERY_CONTRACT_VERSION",
    "ForcedChoiceDeliveryCandidate",
    "ForcedChoiceLocalContext",
    "load_forced_choice_local_context",
    "run_forced_choice_delivery",
]
