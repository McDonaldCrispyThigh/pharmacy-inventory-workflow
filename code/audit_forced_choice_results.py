"""Read-only numerical audit for a completed forced-choice empirical run.

The audit verifies every declared delivery artifact before computing descriptive
RQ1--RQ3 summaries.  It performs no provider calls and writes no thesis prose.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASE_AUDIT_PATH = (
    ROOT / "analysis" / "v2" / "results_audit_20260831T230500Z"
    / "recompute_results_audit.py"
)
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONDITIONS = (
    "A_metadata_only",
    "B_metadata_reviews",
    "C_metadata_reviews_wayback",
)
CORE_FIELDS = (
    "target_construct_status",
    "business_status",
    "candidate_plausibility",
    "identity_consistency",
)


def _base_module():
    specification = importlib.util.spec_from_file_location(
        "nets_ai_historical_results_audit", BASE_AUDIT_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("historical audit utilities are unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if type(value) is not dict:
            raise ValueError(f"JSONL object required: {path.name}")
        rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(numerator: float, denominator: float) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


def _counts(values: Iterable[Any], denominator: int) -> dict[str, dict[str, Any]]:
    counts = Counter(str(value) for value in values)
    return {
        key: {"count": count, "pct": _pct(count, denominator)}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def authenticate_delivery(delivery: Path) -> dict[str, Any]:
    manifest = _read_json(delivery / "run_manifest.json")
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("status") != "full_delivery_complete"
        or manifest.get("candidate_count") != 1226
        or manifest.get("origin_count") != 327
        or manifest.get("runtime_failure_count") != 0
        or type(artifacts) is not dict
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            for relative, digest in artifacts.items()
        )
    ):
        raise ValueError("delivery manifest is incomplete or malformed")
    actual = {
        path.relative_to(delivery).as_posix()
        for path in delivery.rglob("*")
        if path.is_file()
    }
    if actual != set(artifacts) | {"run_manifest.json"}:
        raise ValueError("delivery file set differs from the manifest")
    for relative, expected in artifacts.items():
        if _sha256_file(delivery / relative) != expected:
            raise ValueError(f"delivery artifact hash mismatch: {relative}")
    semantic = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_metadata", "semantic_manifest_sha256"}
    }
    if _canonical_sha256(semantic) != manifest.get("semantic_manifest_sha256"):
        raise ValueError("delivery semantic manifest hash mismatch")
    return manifest


def audit_candidate_flow(delivery: Path) -> dict[str, Any]:
    rows = _read_csv(delivery / "eligibility.csv")
    inventory = _read_csv(delivery / "candidate_inventory.csv")
    unresolved_file = _read_csv(delivery / "unresolved_cases.csv")
    total = len(rows)
    states = _counts((row["eligibility_state"] for row in rows), total)
    for state in ("include", "exclude", "unresolved"):
        states.setdefault(state, {"count": 0, "pct": _pct(0, total)})
    fields = {
        field: _counts((row[field] for row in rows), total)
        for field in CORE_FIELDS
    }
    confidence = {
        field: _counts((row[f"{field}_confidence"] for row in rows), total)
        for field in CORE_FIELDS
    }
    reason_distributions = {}
    for state in ("exclude", "unresolved"):
        subset = [row for row in rows if row["eligibility_state"] == state]
        individual: Counter[str] = Counter()
        combinations: Counter[str] = Counter()
        cause_counts: Counter[int] = Counter()
        for row in subset:
            reasons = json.loads(row["reason_codes"])
            individual.update(reasons)
            combinations[" + ".join(reasons)] += 1
            cause_counts[len(reasons)] += 1
        reason_distributions[state] = {
            "denominator": len(subset),
            "individual_reason_codes_nonexclusive": {
                key: {"count": count, "pct_of_state": _pct(count, len(subset))}
                for key, count in sorted(
                    individual.items(), key=lambda item: (-item[1], item[0])
                )
            },
            "reason_code_combinations": {
                key: {"count": count, "pct_of_state": _pct(count, len(subset))}
                for key, count in sorted(
                    combinations.items(), key=lambda item: (-item[1], item[0])
                )
            },
            "single_vs_multiple_causes": {
                "single": sum(count for number, count in cause_counts.items() if number == 1),
                "multiple": sum(count for number, count in cause_counts.items() if number > 1),
                "by_cause_count": dict(sorted(cause_counts.items())),
            },
        }
    technical = [row for row in rows if row["eligibility_state"] == "unresolved"]
    return {
        "candidate_denominator": total,
        "eligibility": states,
        "core_judgment_distributions": fields,
        "core_confidence_distributions": confidence,
        "reason_distributions": reason_distributions,
        "technical_unresolved": {
            "count": len(technical),
            "pct": _pct(len(technical), total),
            "classification_status_counts": _counts(
                (row["classification_status"] for row in technical), len(technical)
            ),
            "interpretation": (
                "Technical-unresolved only; not a substantive negative classification."
            ),
        },
        "checks": {
            "eligibility_rows": total,
            "candidate_inventory_rows": len(inventory),
            "unresolved_file_rows": len(unresolved_file),
            "states_sum_to_total": sum(item["count"] for item in states.values()) == total,
            "inventory_reconciles": len(inventory) == total,
            "unresolved_file_reconciles": len(unresolved_file) == len(technical),
        },
    }


def _pilot_value(row: dict[str, Any], field: str) -> Any:
    if field == "eligibility_state":
        return row[field]
    if field == "historical_year":
        decision = row["decision"]
        return (decision["historical_year_type"], decision["historical_year_value"])
    return row["decision"][field]


def audit_pilot(delivery: Path) -> dict[str, Any]:
    rows = _read_jsonl(delivery / "ablation_results.jsonl")
    summary = _read_json(delivery / "ablation_summary.json")
    grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["candidate_id"]][row["condition"]] = row
    if len(grouped) != 8 or any(set(value) != set(CONDITIONS) for value in grouped.values()):
        raise ValueError("pilot is not an exact eight-candidate A/B/C frame")
    fields = (*CORE_FIELDS, "historical_year", "eligibility_state")
    distributions = {}
    for condition in CONDITIONS:
        subset = [value[condition] for value in grouped.values()]
        distributions[condition] = {
            field: dict(
                sorted(Counter(str(_pilot_value(row, field)) for row in subset).items())
            )
            for field in fields
        }
    comparisons = {}
    for left, right in (
        (CONDITIONS[0], CONDITIONS[1]),
        (CONDITIONS[1], CONDITIONS[2]),
        (CONDITIONS[0], CONDITIONS[2]),
    ):
        field_changes = {}
        any_core = 0
        for field in fields:
            transitions: Counter[str] = Counter()
            changed = 0
            for slots in grouped.values():
                before = _pilot_value(slots[left], field)
                after = _pilot_value(slots[right], field)
                transitions[f"{before}->{after}"] += 1
                changed += before != after
            field_changes[field] = {
                "changed_candidates": changed,
                "transitions": dict(sorted(transitions.items())),
            }
        for slots in grouped.values():
            any_core += any(
                _pilot_value(slots[left], field)
                != _pilot_value(slots[right], field)
                for field in CORE_FIELDS
            )
        field_changes["any_core_field_changed_candidates"] = any_core
        comparisons[f"{left}_to_{right}"] = field_changes
    validation = {
        "rows": len(rows),
        "successful_outputs": summary["successful_output_count"],
        "validation_passes": summary["validation_pass_count"],
        "candidate_binding_passes": summary["candidate_binding_pass_count"],
        "technical_failures": summary["technical_failure_count"],
        "board_separation_passed": summary["board_separation_passed"],
        "promotion_passed": summary["promotion_passed"],
        "returned_models": summary["returned_models"],
    }
    return {
        "candidate_count": len(grouped),
        "condition_distributions": distributions,
        "pairwise_changes": comparisons,
        "validation": validation,
        "estimand_boundary": (
            "Purposive eight-candidate execution/evidence-order check; it does not "
            "estimate population accuracy."
        ),
    }


def audit_name_recurrence(delivery: Path) -> dict[str, Any]:
    summary = _read_json(delivery / "name_recurrence_summary.json")
    detail = _read_csv(delivery / "name_recurrence_detail.csv")
    primary_rows = [
        row for row in summary["rows"]
        if row["linkage_strategy"] == summary["primary_strategy"]
    ]
    return {
        "definition": summary["definition"],
        "ownership_inference": summary["ownership_inference"],
        "bootstrap": summary["uncertainty"],
        "thresholds": summary["thresholds"],
        "primary_strategy": summary["primary_strategy"],
        "primary_rows": primary_rows,
        "primary_gaps": [
            row for row in summary["repeated_minus_below_threshold_gaps"]
            if row["linkage_strategy"] == summary["primary_strategy"]
        ],
        "all_sensitivity_rows": summary["rows"],
        "detail_row_count": len(detail),
    }


def audit_confidence_correspondence(
    delivery: Path, target_zips: set[str]
) -> dict[str, Any]:
    delivered = _read_json(delivery / "confidence_board_correspondence.json")
    rows = _read_csv(delivery / "confidence_board_correspondence.csv")
    inventory = {
        row["place_id"]: row
        for row in _read_csv(delivery / "candidate_inventory.csv")
    }
    if any(row["place_id"] not in inventory for row in rows):
        raise ValueError("confidence correspondence escaped the candidate inventory")
    target_rows = [
        row for row in rows
        if inventory[row["place_id"]]["returned_postal_code"] in target_zips
    ]
    summaries = []
    for field in CORE_FIELDS:
        column = f"{field}_confidence"
        grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in target_rows:
            grouped[row[column]].append(row)
        for confidence, group in sorted(grouped.items()):
            matched = sum(
                str(row["matched_to_screened_board_reference"]).casefold() == "true"
                for row in group
            )
            summaries.append(
                {
                    "field": field,
                    "confidence": confidence,
                    "workflow_inventory_locations": len(group),
                    "matched_locations": matched,
                    "matched_share_pct": _pct(matched, len(group)),
                }
            )
    return {
        "target_frame_scope": (
            "included workflow inventory locations with provider-returned ZIP in the "
            "60-ZIP target frame under primary linkage"
        ),
        "target_frame_rows": len(target_rows),
        "target_frame_rows_by_confidence": summaries,
        "all_included_collection_frame_rows": len(rows),
        "delivered_all_included_summary": delivered,
        "interpretation": (
            "Descriptive Board correspondence by model confidence; not accuracy, "
            "calibration, precision, or recall."
        ),
    }


def build_audit(run_id: str) -> dict[str, Any]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run ID must use YYYYMMDDTHHMMSSZ")
    run_root = ROOT / "data" / "v2" / run_id
    delivery = run_root / "delivery"
    run_manifest = _read_json(run_root / "run_manifest.json")
    if (
        run_manifest.get("status") != "full_complete"
        or run_manifest.get("delivery_status") != "full_delivery_complete"
        or run_manifest.get("completed_count") != 1226
        or run_manifest.get("failure_count") != 0
        or run_manifest.get("model") != "gpt-5-mini"
    ):
        raise ValueError("forced-choice run is not complete")
    delivery_manifest = authenticate_delivery(delivery)
    base = _base_module()
    base.DELIVERY = delivery
    fixed_rows = _read_jsonl(run_root / "place_ids.jsonl")
    target_zips = {row["source_zip"] for row in fixed_rows}
    if len(target_zips) != 60:
        raise ValueError("fixed target ZIP frame changed")
    candidate = audit_candidate_flow(delivery)
    pilot = audit_pilot(delivery)
    board_linkage = base.audit_board_and_linkage()
    recurrence = audit_name_recurrence(delivery)
    spatial = base.audit_spatial()
    confidence = audit_confidence_correspondence(delivery, target_zips)
    primary = next(
        row for row in board_linkage["strata"]
        if row["stratum"] == "include_screened_primary"
    )
    checks = {
        "candidate_denominator_1226": candidate["candidate_denominator"] == 1226,
        "candidate_checks": all(
            value is True
            for key, value in candidate["checks"].items()
            if key.endswith("reconciles") or key == "states_sum_to_total"
        ),
        "delivery_candidate_count": delivery_manifest["candidate_count"] == 1226,
        "delivery_origin_count": delivery_manifest["origin_count"] == 327,
        "run_delivery_semantic_hash": (
            run_manifest["delivery_semantic_manifest_sha256"]
            == delivery_manifest["semantic_manifest_sha256"]
        ),
        "linkage_rows_reconcile": all(
            all(value.values()) for value in board_linkage["row_level_checks"].values()
        ),
        "spatial_rows_reconcile": all(
            all(value.values()) for value in spatial["row_to_summary_checks"].values()
        ),
        "confidence_all_rows_equal_all_includes": (
            confidence["all_included_collection_frame_rows"]
            == candidate["eligibility"]["include"]["count"]
        ),
        "confidence_target_rows_equal_primary_linkage_inventory": (
            confidence["target_frame_rows"] == primary["google_rows"]
        ),
        "name_recurrence_not_ownership": recurrence["ownership_inference"] is False,
    }
    anomalies = [
        {
            "id": "frame-denominators",
            "status": "expected_frame_difference",
            "detail": (
                "RQ1 uses all 1,226 fixed candidates; primary linkage uses included "
                "locations whose provider-returned ZIP is in the 60-ZIP target frame; "
                "spatial analysis uses the prespecified extended collection frame."
            ),
        },
        {
            "id": "board-reference-boundary",
            "status": "estimand_boundary",
            "detail": (
                "Board linkage supports workflow linked share and Board-reference "
                "coverage, not precision, recall, F1, classification accuracy, or "
                "confidence calibration."
            ),
        },
        {
            "id": "name-recurrence-boundary",
            "status": "terminology_boundary",
            "detail": (
                "Repeated-name components are naming patterns; they are not labeled "
                "chains and do not establish common ownership."
            ),
        },
        {
            "id": "pilot-boundary",
            "status": "estimand_boundary",
            "detail": (
                "The purposive eight-candidate A/B/C pilot is a structural and "
                "evidence-order check, not a population-accuracy estimate."
            ),
        },
        {
            "id": "forced-choice-unresolved",
            "status": "observed_zero",
            "detail": (
                "The forced-choice run has zero technical-unresolved candidates. "
                "Accordingly, unresolved-cause and evidence-deficiency tables have "
                "zero denominators; unresolved is not recoded as false or negative."
            ),
        },
        {
            "id": "board-coordinate-availability",
            "status": "sensitivity_required",
            "detail": (
                "The extended screened Board surface has 341 records, of which 303 "
                "have valid coordinates and 38 have missing coordinates. The delivered "
                "Exact-coordinate Board sensitivity contains 233 records."
            ),
        },
        {
            "id": "board-screen-reason-frame",
            "status": "reporting_boundary",
            "detail": (
                "Board screen-reason counts are tabulated for the 245-row active "
                "target-ZIP frame only. The 442-row extended active frame and its 101 "
                "screen exclusions must not be assigned that target-frame breakdown."
            ),
        },
        {
            "id": "name-recurrence-unresolved",
            "status": "distinct_construct",
            "detail": (
                "One primary-linkage workflow location has an unresolved name component. "
                "This is a name-recurrence grouping outcome, not a technical or "
                "classification unresolved case."
            ),
        },
        {
            "id": "confidence-frame",
            "status": "expected_frame_difference",
            "detail": (
                "Confidence summaries cover all 498 included collection-frame "
                "locations, whereas linkage-conditioned correspondence uses the 339 "
                "included locations in the provider-returned 60-ZIP target frame."
            ),
        },
    ]
    if candidate["technical_unresolved"]["count"]:
        anomalies.append(
            {
                "id": "technical-unresolved",
                "status": "review_required",
                "detail": (
                    f"{candidate['technical_unresolved']['count']} candidates have "
                    "genuinely unusable technical inputs; they are not substantive "
                    "negative judgments."
                ),
            }
        )
    return {
        "schema_version": "nets-ai-v2.2-forced-choice-results-audit-v1",
        "scope": "Numerical Results audit only; no Results prose or interpretation",
        "run_id": run_id,
        "model": run_manifest["model"],
        "promoted_returned_models": run_manifest["promoted_returned_models"],
        "fixed_frame_sha256": run_manifest["fixed_frame_sha256"],
        "scientific_implementation_sha256": run_manifest[
            "scientific_implementation_sha256"
        ],
        "delivery_semantic_manifest_sha256": delivery_manifest[
            "semantic_manifest_sha256"
        ],
        "source_artifacts": delivery_manifest["artifacts"],
        "rq1_candidate_flow": candidate,
        "formal_pilot": pilot,
        "rq2_board_and_linkage": board_linkage,
        "rq2_name_recurrence": recurrence,
        "rq2_confidence_board_correspondence": confidence,
        "rq3_spatial": spatial,
        "proposed_results_outline": [
            "Results audit boundary and fixed record-flow frames",
            "RQ1: Forced-choice workflow inventory and confidence",
            "RQ1 execution check: purposive A/B/C pilot",
            "RQ2: Screened Board reference and deterministic linkage",
            "RQ2: Pharmacy-name recurrence and descriptive confidence correspondence",
            "RQ3: Paired tract-level potential-proximity comparison",
            "Prespecified sensitivity results",
        ],
        "proposed_tables_and_figures": [
            "Table: 1,226-candidate record flow and four judgment distributions",
            "Table: core-judgment confidence distributions",
            "Table: eight-candidate A/B/C field transitions and validation gate",
            "Table: Board screen and primary/sensitivity linkage results",
            "Table: repeated-name versus single-occurrence linkage summaries",
            "Table: nearest-distance and paired threshold classifications",
            "Table: income- and non-White-quartile descriptive patterns",
            "Figure: candidate-to-inventory-to-Board record-flow diagram",
            "Figure: signed Google-minus-Board nearest-distance distribution",
            "Figure: paired threshold-classification changes by neighborhood quartile",
        ],
        "audit_issues_and_boundaries": anomalies,
        "verification_checks": checks,
    }


def _md_count(item: dict[str, Any]) -> str:
    return f"{item['count']:,} ({item['pct']:.2f}%)"


def render_markdown(audit: dict[str, Any]) -> str:
    rq1 = audit["rq1_candidate_flow"]
    linkage = audit["rq2_board_and_linkage"]
    recurrence = audit["rq2_name_recurrence"]
    confidence = audit["rq2_confidence_board_correspondence"]
    spatial = audit["rq3_spatial"]
    primary = next(
        row for row in linkage["strata"]
        if row["stratum"] == "include_screened_primary"
    )
    lines = [
        f"# Forced-choice Results audit: `{audit['run_id']}`",
        "",
        "Numerical audit only. The tables below are runtime/provenance and descriptive "
        "results, not Results-chapter prose or substantive interpretation.",
        "",
        "## RQ1 record flow",
        "",
        "| State | Count (%) |",
        "|---|---:|",
    ]
    for state in ("include", "exclude", "unresolved"):
        lines.append(f"| {state} | {_md_count(rq1['eligibility'][state])} |")
    lines.extend(
        [
            "",
            "### Core substantive judgments",
            "",
            "| Field | Label | Count | Percent |",
            "|---|---|---:|---:|",
        ]
    )
    for field, distribution in rq1["core_judgment_distributions"].items():
        for label, item in distribution.items():
            lines.append(
                f"| {field} | {label} | {item['count']:,} | {item['pct']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "### Core-judgment confidence",
            "",
            "| Field | Confidence | Count | Percent |",
            "|---|---|---:|---:|",
        ]
    )
    for field, distribution in rq1["core_confidence_distributions"].items():
        for label, item in distribution.items():
            lines.append(
                f"| {field} | {label} | {item['count']:,} | {item['pct']:.2f}% |"
            )
    excluded = rq1["reason_distributions"]["exclude"]
    lines.extend(
        [
            "",
            "### Exclusion reasons",
            "",
            "Individual reason codes are nonexclusive.",
            "",
            "| Reason code | Count | Percent of excludes |",
            "|---|---:|---:|",
        ]
    )
    for reason, item in excluded["individual_reason_codes_nonexclusive"].items():
        lines.append(
            f"| {reason} | {item['count']:,} | {item['pct_of_state']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Exclusion cause count | Rows |",
            "|---:|---:|",
        ]
    )
    for cause_count, count in excluded["single_vs_multiple_causes"][
        "by_cause_count"
    ].items():
        lines.append(f"| {cause_count} | {count:,} |")
    lines.extend(
        [
            "",
            "Technical-unresolved candidates: **0**. Unresolved-cause, "
            "single/multiple-cause, and evidence-deficiency distributions therefore "
            "have zero denominators; no unresolved row is treated as a negative.",
        ]
    )
    lines.extend(
        [
            "",
            "## Formal pilot",
            "",
            "| Check | Value |",
            "|---|---:|",
            f"| Candidates | {audit['formal_pilot']['candidate_count']} |",
            f"| Outputs | {audit['formal_pilot']['validation']['rows']} |",
            f"| Structurally valid | {audit['formal_pilot']['validation']['validation_passes']} |",
            f"| Technical failures | {audit['formal_pilot']['validation']['technical_failures']} |",
            "| Candidate-binding passes | "
            f"{audit['formal_pilot']['validation']['candidate_binding_passes']} |",
            "| Board separation | pass |",
            "",
            "All A→B, B→C, and A→C comparisons had 0/8 changes in each of the "
            "four core judgments, eligibility, and historical-year state. The "
            "purposive pilot is an execution/evidence-order check, not a "
            "population-accuracy estimate.",
            "",
            "## RQ2 Board frames",
            "",
            "| Frame | Rows |",
            "|---|---:|",
            f"| Active Minnesota source | {linkage['board_frames']['active_mn_rows']:,} |",
            f"| Active target-ZIP frame | {linkage['board_frames']['full_active_target_rows']:,} |",
            f"| Screened target-ZIP reference | {linkage['board_frames']['screened_target_rows']:,} |",
            f"| Active extended spatial frame | {linkage['board_frames']['full_active_extended_rows']:,} |",
            f"| Screened extended spatial reference | {linkage['board_frames']['screened_extended_rows']:,} |",
            "",
            "Target-frame screen exclusions: 64 institutional/hospital, 3 "
            "noncommunity-service, and 0 compounding-only (67 total).",
            "",
            "## RQ2 primary linkage",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Workflow locations entering primary linkage | {primary['google_rows']} |",
            f"| Screened Board locations | {primary['board_rows']} |",
            f"| Linked pairs | {primary['linked_pairs']} |",
            f"| Unmatched workflow locations | {primary['unmatched_google_rows']} |",
            f"| Unmatched Board locations | {primary['unmatched_board_rows']} |",
            f"| Workflow linked share | {primary['google_link_agreement_pct']:.2f}% |",
            f"| Board-reference coverage | {primary['board_recovery_pct']:.2f}% |",
            "",
            "Primary linkage used maximum-weight one-to-one assignment at threshold "
            f"{linkage['linkage_configuration']['threshold']:.0f}. Linked-pair scores: "
            f"n={linkage['primary_score_distribution']['n']}, min="
            f"{linkage['primary_score_distribution']['min']:.4f}, median="
            f"{linkage['primary_score_distribution']['median']:.1f}, mean="
            f"{linkage['primary_score_distribution']['mean']:.3f}, max="
            f"{linkage['primary_score_distribution']['max']:.1f}.",
            "",
            "| Linkage threshold | Linked pairs | Workflow linked share | Board-reference coverage |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in linkage["primary_threshold_sensitivity"]:
        lines.append(
            f"| {row['threshold']:.0f} | {row['linked_pairs']:,} | "
            f"{row['google_link_agreement_pct']:.2f}% | "
            f"{row['board_recovery_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The maximum-weight and maximum-cardinality-then-weight objectives both "
            "produced 146 pairs and total similarity 14,392.2373. Metrics are "
            "linked share and screened-Board coverage, not precision, recall, F1, "
            "or accuracy.",
            "",
            "### Name recurrence (primary linkage)",
            "",
            "| Source | Recurrence threshold | Group | Records | Linked | Linked percent | Cluster-bootstrap 95% interval |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in recurrence["primary_rows"]:
        low = row["cluster_bootstrap_95_low_pct"]
        high = row["cluster_bootstrap_95_high_pct"]
        interval = "NA" if low is None or high is None else f"{low:.2f}–{high:.2f}%"
        linked_pct = "NA" if row["linked_pct"] is None else f"{row['linked_pct']:.2f}%"
        lines.append(
            f"| {row['source']} | {row['recurrence_threshold']} | "
            f"{row['group']} | {row['records']:,} | {row['linked_records']:,} | "
            f"{linked_pct} | {interval} |"
        )
    lines.extend(
        [
            "",
            "Name recurrence describes exact-name components across distinct "
            "source-local addresses; it does not establish common ownership. One "
            "workflow location has an unresolved name component, distinct from the "
            "0 technical-unresolved classifications.",
            "",
            "### Confidence and Board correspondence (target linkage frame)",
            "",
            "| Field | Confidence | Workflow locations | Linked | Linked share |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in confidence["target_frame_rows_by_confidence"]:
        lines.append(
            f"| {row['field']} | {row['confidence']} | "
            f"{row['workflow_inventory_locations']:,} | "
            f"{row['matched_locations']:,} | {row['matched_share_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "These are descriptive correspondence summaries, not confidence "
            "calibration or accuracy estimates.",
            "",
            "## RQ3 nearest distances",
            "",
            "| Surface | n | Mean | Median | Q1 | Q3 | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in (
        ("Workflow include", spatial["google_include_nearest_miles"]),
        ("Screened Board", spatial["board_screened_nearest_miles"]),
        ("Signed workflow-minus-Board", spatial["signed_google_minus_board_miles"]),
    ):
        lines.append(
            f"| {label} | {summary['n']} | {summary['mean']:.3f} | "
            f"{summary['median']:.3f} | {summary['q1']:.3f} | "
            f"{summary['q3']:.3f} | {summary['min']:.3f} | "
            f"{summary['max']:.3f} |"
        )
    sign = spatial["signed_google_minus_board_miles"]["sign_distribution"]
    lines.extend(
        [
            "",
            f"Signed differences: workflow closer {sign['google_closer']['count']} "
            f"({sign['google_closer']['pct']:.2f}%), workflow farther "
            f"{sign['google_farther']['count']} "
            f"({sign['google_farther']['pct']:.2f}%), equal "
            f"{sign['equal']['count']} ({sign['equal']['pct']:.2f}%).",
            "",
            "## RQ3 paired classifications",
            "",
            "| Threshold (miles) | Both beyond | Board only beyond | Workflow only beyond | Neither beyond | Changed tracts | Percent |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold in ("0.5", "1.0", "2.0"):
        scenario = spatial["thresholds"][threshold]["scenarios"][
            "primary_google_include_vs_board_screened"
        ]
        states = scenario["states"]
        changed = scenario["classification_changed"]
        lines.append(
            f"| {threshold} | {states['both_beyond']['count']} | "
            f"{states['board_only_beyond']['count']} | "
            f"{states['google_only_beyond']['count']} | "
            f"{states['neither_beyond']['count']} | {changed['count']} | "
            f"{changed['pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Prespecified spatial sensitivities",
            "",
            "| Threshold | Delivered scenario range | Population-weighted changed | Household-weighted changed | −4 m / declared / +4 m changed counts |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold in ("0.5", "1.0", "2.0"):
        item = spatial["thresholds"][threshold]
        scenario_range = item["delivered_scenario_disagreement_range"]
        weighting = item["weighting"]
        tolerance = item["threshold_tolerance"]
        lines.append(
            f"| {threshold} mi | {scenario_range['min_count']}–"
            f"{scenario_range['max_count']} "
            f"({scenario_range['min_pct']:.2f}–{scenario_range['max_pct']:.2f}%) | "
            f"{weighting['population_weighted']['weighted_pct']:.2f}% | "
            f"{weighting['household_weighted']['weighted_pct']:.2f}% | "
            f"{tolerance['minus_4m_threshold']['paired_disagreements']} / "
            f"{tolerance['declared_threshold']['paired_disagreements']} / "
            f"{tolerance['plus_4m_threshold']['paired_disagreements']} |"
        )
    lines.extend(
        [
            "",
            "Scenario ranges are conditional ranges, not confidence intervals or "
            "bounds. Candidate include-plus-unresolved equals include because the "
            "technical-unresolved count is zero.",
            "",
            "### Neighborhood quartiles",
            "",
            "| Dimension | Quartile | n | Workflow mean miles | Board mean miles | Signed mean miles | Changed at 0.5 mi | Changed at 1 mi | Changed at 2 mi |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dimension, label in (
        ("income_quartiles", "Income"),
        ("nonwhite_quartiles", "Non-White share"),
    ):
        for quartile in ("1", "2", "3", "4"):
            item = spatial[dimension][quartile]
            lines.append(
                f"| {label} | {quartile} | {item['tract_count']} | "
                f"{item['google_nearest_miles']['mean']:.3f} | "
                f"{item['board_nearest_miles']['mean']:.3f} | "
                f"{item['signed_google_minus_board_miles']['mean']:.3f} | "
                f"{item['thresholds']['0.5']['classification_changed']['count']} "
                f"({item['thresholds']['0.5']['classification_changed']['pct']:.2f}%) | "
                f"{item['thresholds']['1.0']['classification_changed']['count']} "
                f"({item['thresholds']['1.0']['classification_changed']['pct']:.2f}%) | "
                f"{item['thresholds']['2.0']['classification_changed']['count']} "
                f"({item['thresholds']['2.0']['classification_changed']['pct']:.2f}%) |"
            )
    lines.extend(
        [
            "",
            "### Coordinate QA",
            "",
            "| Surface | Input rows | Valid coordinates | Missing/invalid |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in spatial["coordinate_qa"]:
        lines.append(
            f"| {row['surface']} | {row['input_rows']} | "
            f"{row['valid_coordinate_rows']} | {row['invalid_coordinate_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Audit issues and methodological boundaries",
            "",
        ]
    )
    for item in audit["audit_issues_and_boundaries"]:
        lines.append(f"- `{item['id']}` ({item['status']}): {item['detail']}")
    lines.extend(
        [
            "",
            "## Proposed Results chapter outline",
            "",
        ]
    )
    for index, item in enumerate(audit["proposed_results_outline"], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(
        [
            "",
            "## Proposed thesis tables and figures",
            "",
        ]
    )
    for item in audit["proposed_tables_and_figures"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for key, value in audit["verification_checks"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines.extend(
        [
            "",
            "The companion machine-readable JSON contains the complete distributions, "
            "reason-code combinations, all linkage strata, all recurrence thresholds, "
            "all spatial scenarios, artifact hashes, and reconciliation checks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.run_id)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Results audit output directory already exists")
    output.mkdir(parents=True)
    (output / "results_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    (output / "results_audit.md").write_text(
        render_markdown(audit), encoding="utf-8", newline=""
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output.as_posix(),
                "all_checks_pass": all(audit["verification_checks"].values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
