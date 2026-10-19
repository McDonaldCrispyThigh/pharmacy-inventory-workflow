"""Read-only, fail-closed verification for a completed v2.1 delivery package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = frozenset(
    {
        "candidate_inventory.csv",
        "eligibility.csv",
        "unresolved_cases.csv",
        "board_links.csv",
        "unmatched_google.csv",
        "unmatched_board.csv",
        "performance_metrics.json",
        "ablation_results.jsonl",
        "ablation_summary.json",
        "tract_spatial.csv",
        "coordinate_qa.csv",
        "coordinate_ledger.csv",
        "spatial_summary.json",
    }
)
ELIGIBILITY_STATES = frozenset({"include", "exclude", "unresolved"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(ValueError):
    """The package cannot support a completed-delivery claim."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is invalid") from exc
    if type(value) is not dict:
        raise VerificationError(f"{label} must be an object")
    return value


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise VerificationError(f"{label} has an invalid header")
            rows = list(reader)
            return list(reader.fieldnames), rows
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise VerificationError(f"{label} is invalid") from exc


def _ids(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        place_id = row.get("place_id", "")
        candidate_id = row.get("candidate_id", "")
        if not place_id or not candidate_id or place_id in result:
            raise VerificationError(f"{label} candidate keys are invalid")
        result[place_id] = row
    return result


def verify_v2_delivery(
    package_root: Path,
    *,
    expect_empirical: bool,
) -> dict[str, Any]:
    """Authenticate one complete full/offline delivery without changing it."""
    if not isinstance(package_root, Path) or type(expect_empirical) is not bool:
        raise TypeError("verification requires a Path and an explicit empirical flag")
    root = package_root.resolve()
    if not root.is_dir():
        raise VerificationError("delivery package directory is unavailable")
    manifest_path = root / "run_manifest.json"
    manifest = _read_json(manifest_path, "delivery manifest")

    observed_empirical = manifest.get("empirical_v2_results_exist")
    if observed_empirical is not expect_empirical:
        raise VerificationError("empirical result label does not match expectation")
    expected_status = (
        "full_delivery_complete" if expect_empirical else "offline_validation_complete"
    )
    if manifest.get("status") != expected_status or manifest.get("failure_count") != 0:
        raise VerificationError("delivery manifest is not complete")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != REQUIRED_ARTIFACTS:
        raise VerificationError("delivery artifact declaration is incomplete")
    if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in artifacts.values()):
        raise VerificationError("delivery artifact hash declaration is invalid")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    declared_files = set(artifacts) | {"run_manifest.json"}
    if actual_files != declared_files:
        raise VerificationError("delivery package contains undeclared or missing files")
    for relative, expected_hash in artifacts.items():
        if _sha256_file(root / relative) != expected_hash:
            raise VerificationError(f"artifact hash mismatch: {relative}")

    semantic = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_metadata", "semantic_manifest_sha256"}
    }
    observed_semantic_hash = manifest.get("semantic_manifest_sha256")
    if (
        not isinstance(observed_semantic_hash, str)
        or SHA256_RE.fullmatch(observed_semantic_hash) is None
        or hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
        != observed_semantic_hash
    ):
        raise VerificationError("semantic manifest hash mismatch")

    privacy = manifest.get("privacy_contract")
    if type(privacy) is not dict or any(
        privacy.get(key) is not False
        for key in (
            "public_name_address_coordinates_retained",
            "review_text_retained",
            "prompt_retained",
            "raw_model_output_retained",
        )
    ):
        raise VerificationError("privacy contract does not fail closed")
    separation = manifest.get("board_separation")
    if (
        type(separation) is not dict
        or separation.get("board_enters_classifier") is not False
        or separation.get("board_enters_eligibility_reducer") is not False
    ):
        raise VerificationError("Board separation contract is invalid")

    inventory_columns, inventory_rows = _read_csv(
        root / "candidate_inventory.csv", "candidate inventory"
    )
    restricted_columns = {
        "company",
        "street_address",
        "name",
        "formatted_address",
        "latitude",
        "longitude",
        "review_text",
        "prompt",
        "raw_model_output",
    }
    if restricted_columns & {column.casefold() for column in inventory_columns}:
        raise VerificationError("candidate inventory contains restricted public columns")
    inventory = _ids(inventory_rows, "candidate inventory")

    _, eligibility_rows = _read_csv(root / "eligibility.csv", "eligibility ledger")
    eligibility = _ids(eligibility_rows, "eligibility ledger")
    if set(inventory) != set(eligibility):
        raise VerificationError("candidate and eligibility ledgers disagree")
    if any(row.get("eligibility_state") not in ELIGIBILITY_STATES for row in eligibility_rows):
        raise VerificationError("eligibility ledger contains an invalid state")
    for place_id, row in eligibility.items():
        if inventory[place_id].get("candidate_id") != row.get("candidate_id"):
            raise VerificationError("candidate lineage disagrees across ledgers")

    _, unresolved_rows = _read_csv(
        root / "unresolved_cases.csv", "unresolved ledger"
    )
    unresolved = _ids(unresolved_rows, "unresolved ledger") if unresolved_rows else {}
    expected_unresolved = {
        place_id
        for place_id, row in eligibility.items()
        if row.get("eligibility_state") == "unresolved"
    }
    if set(unresolved) != expected_unresolved:
        raise VerificationError("unresolved ledger is not the exact eligibility subset")

    tract_columns, tract_rows = _read_csv(root / "tract_spatial.csv", "tract spatial output")
    if restricted_columns & {column.casefold() for column in tract_columns}:
        raise VerificationError("tract output contains restricted coordinate/content columns")
    ledger_columns, coordinate_rows = _read_csv(
        root / "coordinate_ledger.csv", "coordinate ledger"
    )
    if set(ledger_columns) != {
        "surface",
        "record_id",
        "coordinate_state",
        "enters_surface",
    }:
        raise VerificationError("coordinate ledger schema is invalid")
    if any(not row.get("surface") or not row.get("record_id") for row in coordinate_rows):
        raise VerificationError("coordinate ledger contains an untraceable record")

    spatial = _read_json(root / "spatial_summary.json", "spatial summary")
    surfaces = {
        "google_include", "google_include_unresolved", "board_screened",
        "board_screened_exact", "board_all_active",
    }
    surface_definitions = {
        "google_include": "Google candidates reduced to include",
        "google_include_unresolved":
        "conditional sensitivity: Google include plus unresolved",
        "board_screened":
        "automatically screened active Board operational reference",
        "board_screened_exact":
        "screened Board records with Exact cached coordinates only",
        "board_all_active": "all active Board records in the extended frame",
    }
    contract = spatial.get("spatial_contract")
    contract_hash = (
        hashlib.sha256(_canonical_bytes(contract)).hexdigest()
        if type(contract) is dict else None
    )
    if (
        spatial.get("analysis_scope") != "full"
        or spatial.get("estimand")
        != "projected straight-line potential proximity"
        or spatial.get("distance_units") != "meter"
        or spatial.get("origin_count") != len(tract_rows)
        or type(contract) is not dict
        or contract_hash != spatial.get("spatial_contract_sha256")
        or spatial.get("distance_crs") != contract.get("distance_crs")
        or spatial.get("supply_input_crs") != contract.get("supply_crs")
        or spatial.get("origin_id_column") != contract.get("origin_id_column")
        or spatial.get("origin_status")
        != ("ok" if tract_rows else "empty_origin_frame")
        or spatial.get("spatial_contract_sha256")
        != manifest.get("contracts", {}).get("spatial_contract_sha256")
        or type(spatial.get("thresholds")) is not list
        or type(spatial.get("coordinate_qa")) is not list
        or spatial.get("surface_definitions") != surface_definitions
    ):
        raise VerificationError("spatial summary provenance is invalid")
    qa_by_surface = {row.get("surface"): row for row in spatial["coordinate_qa"]
                     if type(row) is dict}
    if set(qa_by_surface) != surfaces:
        raise VerificationError("spatial coordinate summary is incomplete")
    ledger_counts = {surface: 0 for surface in surfaces}
    for row in coordinate_rows:
        if row.get("surface") not in surfaces:
            raise VerificationError("spatial coordinate ledger has an unknown surface")
        ledger_counts[row["surface"]] += 1
    if any(
        qa_by_surface[surface].get("input_rows") != ledger_counts[surface]
        for surface in surfaces
    ):
        raise VerificationError("spatial coordinate summary does not reconcile")
    expected_thresholds = manifest.get("contracts", {}).get(
        "spatial_thresholds_miles"
    )
    if expected_thresholds is None:
        expected_thresholds = [0.5, 1.0, 2.0]
    observed_miles = []

    def boolean_column(column: str) -> list[bool | None]:
        values: list[bool | None] = []
        for row in tract_rows:
            value = row[column]
            if value == "":
                values.append(None)
            elif value in {"True", "False"}:
                values.append(value == "True")
            else:
                raise VerificationError("spatial threshold column is invalid")
        return values

    def paired(board: list[bool | None], google: list[bool | None]) -> dict[str, int]:
        pairs = [(left, right) for left, right in zip(board, google)
                 if left is not None and right is not None]
        return {
            "paired_origins": len(pairs),
            "both_beyond": sum(left and right for left, right in pairs),
            "board_only_beyond": sum(left and not right for left, right in pairs),
            "google_only_beyond": sum(not left and right for left, right in pairs),
            "neither_beyond": sum(not left and not right for left, right in pairs),
            "paired_disagreements": sum(left != right for left, right in pairs),
        }

    for threshold in spatial["thresholds"]:
        if (
            type(threshold) is not dict
            or threshold.get("origin_count") != len(tract_rows)
            or type(threshold.get("surfaces")) is not dict
            or set(threshold["surfaces"]) != surfaces
        ):
            raise VerificationError("spatial threshold summary is invalid")
        miles = threshold.get("threshold_miles")
        if type(miles) not in {int, float}:
            raise VerificationError("spatial threshold is invalid")
        observed_miles.append(float(miles))
        if threshold.get("threshold_meters") != round(
            float(miles) * float(contract.get("meters_per_mile", 0)), 3
        ):
            raise VerificationError("spatial threshold meters do not reconcile")
        suffix = str(float(miles)).replace(".", "_") + "mi"
        columns: dict[str, list[bool | None]] = {}
        for surface in surfaces:
            column = f"{surface}_beyond_{suffix}"
            if column not in tract_columns:
                raise VerificationError("spatial threshold column is missing")
            values = boolean_column(column)
            columns[surface] = values
            known = [value for value in values if value is not None]
            observed = threshold["surfaces"][surface]
            if (
                type(observed) is not dict
                or observed.get("known_origins") != len(known)
                or observed.get("beyond_count") != sum(
                    bool(value) for value in known
                )
                or observed.get("beyond_pct") != (
                    round(100.0 * sum(bool(value) for value in known) / len(known), 2)
                    if known else None
                )
            ):
                raise VerificationError("spatial threshold summary does not reconcile")
        board = columns["board_screened"]
        if (
            threshold.get("primary_google_vs_board")
            != paired(board, columns["google_include"])
            or threshold.get("conditional_google_vs_board")
            != paired(board, columns["google_include_unresolved"])
        ):
            raise VerificationError("spatial paired summary does not reconcile")
    if observed_miles != [float(value) for value in expected_thresholds]:
        raise VerificationError("spatial thresholds differ from the contract")

    summary = _read_json(root / "ablation_summary.json", "ablation summary")
    if (
        summary.get("privacy_check_passed") is not True
        or summary.get("board_separation_passed") is not True
        or summary.get("unsafe_content_detected") is not False
        or summary.get("full_run_recommendation") != "proceed"
    ):
        raise VerificationError("ablation safety/promotion checks did not pass")

    candidate_count = manifest.get("candidate_count")
    origin_count = manifest.get("origin_count")
    unresolved_count = manifest.get("unresolved_count")
    if candidate_count != len(inventory_rows):
        raise VerificationError("candidate count does not reconcile")
    if origin_count != len(tract_rows):
        raise VerificationError("origin count does not reconcile")
    if unresolved_count != len(unresolved_rows):
        raise VerificationError("unresolved count does not reconcile")

    return {
        "schema_version": "nets-ai-v2.1-delivery-verification-v1",
        "status": "pass",
        "package_root": str(root),
        "empirical_v2_results_exist": observed_empirical,
        "artifact_count": len(artifacts),
        "candidate_count": candidate_count,
        "unresolved_count": unresolved_count,
        "origin_count": origin_count,
        "semantic_manifest_sha256": observed_semantic_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect-empirical", action="store_true")
    group.add_argument("--expect-offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_v2_delivery(
            args.package_root,
            expect_empirical=bool(args.expect_empirical),
        )
    except (OSError, TypeError, VerificationError):
        print(json.dumps({"status": "fail"}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
