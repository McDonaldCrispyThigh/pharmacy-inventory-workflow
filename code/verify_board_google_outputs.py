"""Strict, read-only verification for the canonical Board--Google delivery.

Verification is intentionally ordered. The summary is read first, then the
declared code and input bytes are authenticated before any generated output hash
or analytical invariant is trusted. The verifier writes no files.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from study_design import (  # noqa: E402
    TARGET_ZIPS,
    normalize_legal_signature,
    normalize_name_signature,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = ROOT / "analysis" / "delivery"
ANALYSIS_ROOT = ROOT / "analysis"
M_PER_MILE = 1609.344
BOARD_SOURCE = "Board keyword-screen reference"
GOOGLE_SOURCE = "Google exact-address clusters"

CANONICAL_RELATIVE_OUTPUTS = (
    Path("private/board_keyword_screen_fixed.csv"),
    Path("private/board_keyword_screen_extended.csv"),
    Path("private/board_google_record_links.csv"),
    Path("private/board_google_source_status.csv"),
    Path("private/board_google_name_signature_groups.csv"),
    Path("results/board_google_tract_proximity.csv"),
    Path("results/linkage_strategy_comparison.csv"),
    Path("results/proximity_threshold_comparison.csv"),
    Path("results/name_signature_summary.json"),
    Path("figures/board_google_proximity_comparison.png"),
    Path("figures/board_google_proximity_comparison.pdf"),
    Path("figures/name_signature_coverage.png"),
    Path("figures/name_signature_coverage.pdf"),
)
CORE_RELATIVE_OUTPUTS = CANONICAL_RELATIVE_OUTPUTS[:6] + (
    Path("results/name_signature_summary.json"),
)
STRICT_REQUIRED_CODE = (
    Path("code/study_design.py"),
    Path("code/geo_match.py"),
    Path("code/board_google_comparison.py"),
    Path("code/verify_board_google_outputs.py"),
)
STRICT_REQUIRED_INPUTS = (
    Path("data/Minneapolis-StPaul_pharmacy_20260413_201606.csv"),
    Path("data/board_geocoded.csv"),
    Path("data/Board/F.1  Pharmacy (PH).csv"),
    Path("data/tl_2023_27_tract.zip"),
    Path("data/tl_2023_us_zcta520.zip"),
    Path("data/acs_tracts_2023.csv"),
    Path("data/acs_tracts_vehicles_2023.csv"),
    Path("data/audit3_fn_institutional_20260414_172017.csv"),
    Path("data/board_fn_20260414_170601.csv"),
)
STRICT_REQUIRED_PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "rapidfuzz",
    "geopandas",
    "matplotlib",
    "shapely",
    "pyproj",
    "pyogrio",
)
REQUIREMENTS_ANALYSIS_PATH = ROOT / "requirements-analysis.txt"
TRACT_SOURCE_PATH = ROOT / "data" / "tl_2023_27_tract.zip"
ZCTA_SOURCE_PATH = ROOT / "data" / "tl_2023_us_zcta520.zip"
ACS_SOURCE_PATH = ROOT / "data" / "acs_tracts_2023.csv"
VEHICLES_SOURCE_PATH = ROOT / "data" / "acs_tracts_vehicles_2023.csv"
SPATIAL_COUNTY_FRAME = (("053", "Hennepin"), ("123", "Ramsey"))
SUPPLY_INPUT_CRS_ASSUMPTION = "EPSG:4326"


@dataclass
class VerificationError(Exception):
    """A fail-closed verification error with machine-readable context."""

    stage: str
    check: str
    message: str
    path: str | None = None
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "FAIL",
            "stage": self.stage,
            "check": self.check,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        return result


def _fail(
    stage: str,
    check: str,
    message: str,
    *,
    path: Path | str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> None:
    if isinstance(path, Path):
        try:
            display_path: str | None = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            display_path = str(path)
    else:
        display_path = path
    raise VerificationError(
        stage=stage,
        check=check,
        message=message,
        path=display_path,
        expected=expected,
        actual=actual,
    )


def _require(
    condition: bool,
    stage: str,
    check: str,
    message: str,
    **context: Any,
) -> None:
    if not bool(condition):
        _fail(stage, check, message, **context)


def _mapping(value: Any, stage: str, check: str) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping),
        stage,
        check,
        f"Expected an object for {check}",
        actual=type(value).__name__,
    )
    return value


def _sequence(value: Any, stage: str, check: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        stage,
        check,
        f"Expected an array for {check}",
        actual=type(value).__name__,
    )
    return value


def _nested(mapping: Mapping[str, Any], keys: Sequence[str], stage: str) -> Any:
    current: Any = mapping
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        _require(
            isinstance(current, Mapping) and key in current,
            stage,
            "summary_schema",
            f"Summary is missing {'.'.join(traversed)}",
        )
        current = current[key]
    return current


def _read_json(path: Path, stage: str) -> dict[str, Any]:
    _require(path.is_file(), stage, "file_exists", "Required JSON file is missing", path=path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(stage, "json_parse", f"Could not parse JSON: {exc}", path=path)
    return dict(_mapping(payload, stage, "json_root"))


def _read_csv(path: Path, *, dtype: Mapping[str, Any] | None = None) -> pd.DataFrame:
    _require(path.is_file(), "outputs", "file_exists", "Required CSV file is missing", path=path)
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=False, keep_default_na=False)
    except Exception as exc:  # pandas raises several parser/type subclasses
        _fail("outputs", "csv_parse", f"Could not parse CSV: {exc}", path=path)


def _resolve_output_root(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    resolved = candidate.resolve()
    _require(
        resolved != ANALYSIS_ROOT.resolve()
        and ANALYSIS_ROOT.resolve() in resolved.parents,
        "configuration",
        "output_root_scope",
        "--output-root must be a dedicated staging/delivery directory under repository/analysis",
        path=resolved,
    )
    return resolved


def _repo_path(relative: Any, manifest_name: str) -> tuple[str, Path]:
    _require(
        isinstance(relative, str) and bool(relative.strip()),
        manifest_name,
        "relative_path",
        "Manifest paths must be nonempty strings relative to the repository root",
        actual=relative,
    )
    rel = Path(relative)
    _require(
        not rel.is_absolute(),
        manifest_name,
        "relative_path",
        "Manifest path must not be absolute",
        path=relative,
    )
    resolved = (ROOT / rel).resolve()
    _require(
        resolved != ROOT and ROOT in resolved.parents,
        manifest_name,
        "path_scope",
        "Manifest path escapes the repository root",
        path=relative,
    )
    return resolved.relative_to(ROOT).as_posix(), resolved


def _manifest_entries(summary: Mapping[str, Any], name: str) -> dict[str, tuple[Path, str]]:
    manifest = _mapping(_nested(summary, [name], name), name, name)
    _require(bool(manifest), name, "manifest_nonempty", f"{name} must not be empty")
    result: dict[str, tuple[Path, str]] = {}
    for raw_path, expected_hash in manifest.items():
        normalized, resolved = _repo_path(raw_path, name)
        _require(
            normalized not in result,
            name,
            "path_collision",
            "Manifest contains paths that normalize to the same file",
            path=normalized,
        )
        _require(
            isinstance(expected_hash, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is not None,
            name,
            "sha256_format",
            "Manifest SHA-256 must contain exactly 64 hexadecimal characters",
            path=normalized,
            actual=expected_hash,
        )
        result[normalized] = (resolved, expected_hash.casefold())
    return result


def _verify_manifest_hashes(
    entries: Mapping[str, tuple[Path, str]],
    stage: str,
) -> None:
    for normalized in sorted(entries):
        path, expected = entries[normalized]
        _require(path.is_file(), stage, "file_exists", "Manifest file is missing", path=normalized)
        actual = sha256_file(path)
        _require(
            actual.casefold() == expected,
            stage,
            "sha256_match",
            f"{stage} SHA-256 mismatch",
            path=normalized,
            expected=expected,
            actual=actual,
        )


def _verify_strict_interpretation_contract(summary: Mapping[str, Any]) -> None:
    manifest_record = _mapping(
        summary.get("input_manifest"),
        "input_manifest",
        "input_manifest_record",
    )
    _, manifest_path = _repo_path(manifest_record.get("path"), "input_manifest")
    manifest = _read_json(manifest_path, "input_manifest")
    analysis_contract = _mapping(
        manifest.get("analysis_contract"),
        "input_manifest",
        "analysis_contract",
    )
    empirical_test_case = str(analysis_contract.get("empirical_test_case", ""))
    _require(
        re.search(r"\bpharmac(?:y|ies)\b", empirical_test_case.casefold()) is not None,
        "input_manifest",
        "empirical_test_case",
        "The authenticated analysis contract must identify pharmacy establishments "
        "as the empirical test case",
        actual=empirical_test_case,
    )

    criterion = str(summary.get("criterion_definition", ""))
    criterion_folded = criterion.casefold()
    _require(
        "operational reference" in criterion_folded
        and re.search(r"\bground(?:[\s_-]+)?truth\b", criterion_folded) is None,
        "invariants",
        "criterion_definition",
        "criterion_definition must identify the Board screen as an operational "
        "reference and must not describe it as ground truth",
        actual=criterion,
    )

    spatial = _mapping(summary.get("spatial"), "invariants", "spatial")
    interpretation = str(spatial.get("distance_interpretation", ""))
    interpretation_folded = interpretation.casefold()
    metric_is_explicit = (
        "euclidean" in interpretation_folded
        or re.search(r"\bstraight[\s-]+line\b", interpretation_folded) is not None
    )
    transport_boundary_is_explicit = (
        re.search(
            r"\b(?:not|no)\b[^.;]{0,160}"
            r"\b(?:road|network|travel(?:[\s-]+time)?)\b",
            interpretation_folded,
        )
        is not None
        or re.search(
            r"\bnon[\s-]+(?:road|network|travel(?:[\s-]+time)?)\b",
            interpretation_folded,
        )
        is not None
    )
    realized_access_present = (
        re.search(r"\brealized[\s-]+access\b", interpretation_folded) is not None
    )
    realized_access_is_negated = (
        re.search(
            r"\b(?:not|no)\b(?:(?!\bbut\b)[^.;]){0,160}"
            r"\brealized[\s-]+access\b",
            interpretation_folded,
        )
        is not None
        or re.search(
            r"\bnon[\s-]+realized[\s-]+access\b",
            interpretation_folded,
        )
        is not None
    )
    _require(
        "potential" in interpretation_folded
        and metric_is_explicit
        and transport_boundary_is_explicit
        and (not realized_access_present or realized_access_is_negated),
        "invariants",
        "distance_interpretation",
        "distance_interpretation must label Euclidean or straight-line potential "
        "proximity, reject road/network travel interpretation, and make no positive "
        "realized-access claim",
        actual=interpretation,
    )


def _strict_manifest_contract(
    code_entries: Mapping[str, tuple[Path, str]],
    input_entries: Mapping[str, tuple[Path, str]],
    output_entries: Mapping[str, tuple[Path, str]],
    output_root: Path,
) -> None:
    _require(
        output_root.resolve() != ANALYSIS_ROOT.resolve()
        and ANALYSIS_ROOT.resolve() in output_root.resolve().parents,
        "output_sha256",
        "output_root_scope",
        "Strict output root must be a dedicated directory under repository/analysis",
        path=output_root,
    )
    code_keys = set(code_entries)
    input_keys = set(input_entries)
    output_keys = set(output_entries)
    prohibited_input_paths = sorted(
        path
        for path in input_keys
        if "nets" in path.casefold() or "nppes" in path.casefold()
    )
    _require(
        not prohibited_input_paths,
        "input_sha256",
        "prohibited_input_dependency",
        "Strict delivery must not declare NETS or NPPES input dependencies",
        actual=prohibited_input_paths,
    )
    output_prefix = output_root.relative_to(ROOT)
    requirements = (
        ("code_sha256", {path.as_posix() for path in STRICT_REQUIRED_CODE}, code_keys),
        ("input_sha256", {path.as_posix() for path in STRICT_REQUIRED_INPUTS}, input_keys),
        (
            "output_sha256",
            {
                (output_prefix / path).as_posix()
                for path in CANONICAL_RELATIVE_OUTPUTS
            },
            output_keys,
        ),
    )
    for label, required, actual in requirements:
        missing = sorted(required - actual)
        _require(
            not missing,
            label,
            "manifest_coverage",
            f"{label} omits required pipeline dependencies",
            expected=missing,
        )
    required_outputs = {
        (output_prefix / path).as_posix() for path in CANONICAL_RELATIVE_OUTPUTS
    }
    extra_outputs = sorted(output_keys - required_outputs)
    _require(
        not extra_outputs,
        "output_sha256",
        "manifest_exact_set",
        "output_sha256 must contain exactly the 13 canonical delivery outputs",
        expected=sorted(required_outputs),
        actual=sorted(output_keys),
    )
    overlap = (code_keys & input_keys) | (code_keys & output_keys) | (input_keys & output_keys)
    _require(
        not overlap,
        "manifests",
        "manifest_role_overlap",
        "A path cannot be declared as more than one of code, input, or output",
        actual=sorted(overlap),
    )
    for normalized, (path, _) in output_entries.items():
        _require(
            path != output_root and output_root in path.parents,
            "output_sha256",
            "output_scope",
            "Every declared output must be beneath --output-root",
            path=normalized,
        )
    summary_key = (output_prefix / "results/analysis_summary.json").as_posix()
    _require(
        summary_key not in output_keys,
        "output_sha256",
        "summary_self_hash",
        "analysis_summary.json cannot contain its own SHA-256",
        path=summary_key,
    )
    physical_files: set[str] = set()
    for path in output_root.rglob("*"):
        if path.is_symlink():
            _fail(
                "output_sha256",
                "output_symlink",
                "Strict output package must not contain symbolic links",
                path=path,
            )
        if path.is_file():
            physical_files.add(path.relative_to(ROOT).as_posix())
    expected_physical_files = output_keys | {summary_key}
    _require(
        physical_files == expected_physical_files,
        "output_sha256",
        "physical_output_exact_set",
        "Output root must contain exactly the 13 hashed outputs plus analysis_summary.json",
        expected=sorted(expected_physical_files),
        actual=sorted(physical_files),
    )
    _verify_strict_interpretation_contract(
        _read_json(output_root / "results" / "analysis_summary.json", "summary")
    )


def _verify_input_manifest_record(
    summary: Mapping[str, Any],
    input_entries: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    record = _mapping(
        _nested(summary, ["input_manifest"], "input_manifest"),
        "input_manifest",
        "input_manifest_record",
    )
    normalized_path, manifest_path = _repo_path(
        record.get("path"),
        "input_manifest",
    )
    _require(
        normalized_path == "analysis/input_manifest.json",
        "input_manifest",
        "canonical_path",
        "Strict delivery must use the canonical analysis/input_manifest.json",
        expected="analysis/input_manifest.json",
        actual=normalized_path,
    )
    _require(
        manifest_path.is_file(),
        "input_manifest",
        "file_exists",
        "Declared input manifest is missing",
        path=normalized_path,
    )
    expected_hash = record.get("sha256")
    _require(
        isinstance(expected_hash, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is not None,
        "input_manifest",
        "sha256_format",
        "Summary input-manifest SHA-256 is malformed",
        actual=expected_hash,
    )
    actual_hash = sha256_file(manifest_path)
    _require(
        actual_hash.casefold() == expected_hash.casefold(),
        "input_manifest",
        "sha256_match",
        "Input-manifest SHA-256 differs from the summary record",
        path=normalized_path,
        expected=expected_hash,
        actual=actual_hash,
    )
    manifest = _read_json(manifest_path, "input_manifest")
    _check_number(manifest.get("schema_version"), record.get("schema_version"), "input_manifest.schema_version")
    _require(
        manifest.get("schema_version") == 1,
        "input_manifest",
        "schema_version",
        "Unsupported input-manifest schema version",
        expected=1,
        actual=manifest.get("schema_version"),
    )
    _require(
        str(manifest.get("hash_algorithm", "")).casefold() == "sha256",
        "input_manifest",
        "hash_algorithm",
        "Input manifest must declare SHA-256",
        expected="sha256",
        actual=manifest.get("hash_algorithm"),
    )
    raw_inputs = _sequence(
        manifest.get("inputs"),
        "input_manifest",
        "inputs",
    )
    _check_number(len(raw_inputs), manifest.get("entry_count"), "input_manifest.entry_count")
    _check_number(len(raw_inputs), record.get("entry_count"), "summary.input_manifest.entry_count")
    declared: dict[str, str] = {}
    for raw_entry in raw_inputs:
        entry = _mapping(raw_entry, "input_manifest", "input_entry")
        normalized, input_path = _repo_path(entry.get("path"), "input_manifest")
        _require(
            normalized not in declared,
            "input_manifest",
            "duplicate_input_path",
            "Input manifest contains a duplicate normalized path",
            path=normalized,
        )
        entry_hash = entry.get("sha256")
        _require(
            isinstance(entry_hash, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", entry_hash) is not None,
            "input_manifest",
            "entry_sha256_format",
            "Input-manifest entry SHA-256 is malformed",
            path=normalized,
            actual=entry_hash,
        )
        declared[normalized] = entry_hash.casefold()
        if entry.get("bytes") is not None:
            _check_number(input_path.stat().st_size, entry["bytes"], f"input_manifest.bytes.{normalized}")
        for metadata_key in ("role", "vintage", "status"):
            _require(
                metadata_key in entry,
                "input_manifest",
                "provenance_metadata",
                f"Input-manifest entry lacks {metadata_key}",
                path=normalized,
            )

    expected = {
        normalized: expected_hash
        for normalized, (_, expected_hash) in input_entries.items()
    }
    _require(
        declared == expected,
        "input_manifest",
        "input_sha256_equivalence",
        "Input-manifest paths and hashes must exactly equal summary.input_sha256",
        expected=expected,
        actual=declared,
    )
    return {
        "path": normalized_path,
        "sha256": actual_hash,
        "schema_version": manifest["schema_version"],
        "entry_count": len(raw_inputs),
    }


def _canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def _parse_requirements_contract(text: str) -> tuple[str, str, dict[str, str]]:
    implementation_versions: list[tuple[str, str]] = []
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.search(
                r"Canonical analysis interpreter:\s*([A-Za-z0-9_-]+)\s+([0-9]+(?:\.[0-9]+)+)\s*$",
                line[1:].strip(),
                flags=re.IGNORECASE,
            )
            if match:
                implementation_versions.append((match.group(1), match.group(2)))
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        _require(
            match is not None,
            "environment",
            "exact_requirement_pin",
            "Every analysis requirement must be an unconditional exact name==version pin",
            path=f"requirements-analysis.txt:{line_number}",
            actual=line,
        )
        name = _canonical_package_name(match.group(1))
        _require(
            name not in pins,
            "environment",
            "duplicate_requirement",
            "requirements-analysis.txt contains a duplicate normalized package name",
            path=f"requirements-analysis.txt:{line_number}",
            actual=name,
        )
        pins[name] = match.group(2)
    _require(
        len(implementation_versions) == 1,
        "environment",
        "interpreter_pin",
        "requirements-analysis.txt must declare exactly one canonical interpreter version",
        actual=implementation_versions,
    )
    required_packages = set(STRICT_REQUIRED_PACKAGES)
    _require(
        set(pins) == required_packages,
        "environment",
        "package_pin_exact_set",
        "requirements-analysis.txt must contain exactly the nine canonical direct packages",
        expected=sorted(required_packages),
        actual=sorted(pins),
    )
    implementation, python_version = implementation_versions[0]
    return implementation, python_version, pins


def _runtime_software() -> dict[str, str]:
    runtime: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in STRICT_REQUIRED_PACKAGES:
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            _fail(
                "environment",
                "runtime_package_missing",
                "A pinned analysis package is not installed in the verifier runtime",
                path=package,
            )
    try:
        geopandas = importlib.import_module("geopandas")
        pyogrio = importlib.import_module("pyogrio")
        shapely = importlib.import_module("shapely")
        pyproj = importlib.import_module("pyproj")
        linkage = importlib.import_module("geo_match")
    except ImportError as exc:
        _fail(
            "environment",
            "runtime_import",
            f"Could not import a native-version provider: {exc}",
        )
    runtime.update(
        {
            "gdal": str(pyogrio.__gdal_version_string__),
            "geos": str(shapely.geos_version_string),
            "proj": str(pyproj.proj_version_str),
            "geopandas_io_engine": (
                "auto"
                if geopandas.options.io_engine is None
                else str(geopandas.options.io_engine)
            ),
            "address_parser": (
                "regex"
                if getattr(linkage, "_PARSER", None)
                is getattr(linkage, "parse_address_regex", object())
                else str(getattr(getattr(linkage, "_PARSER", None), "__name__", "unknown"))
            ),
        }
    )
    return runtime


def _verify_environment_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        REQUIREMENTS_ANALYSIS_PATH.is_file(),
        "environment",
        "requirements_file_exists",
        "requirements-analysis.txt is required for strict verification",
        path=REQUIREMENTS_ANALYSIS_PATH,
    )
    try:
        requirements_text = REQUIREMENTS_ANALYSIS_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(
            "environment",
            "requirements_read",
            f"Could not read requirements-analysis.txt: {exc}",
            path=REQUIREMENTS_ANALYSIS_PATH,
        )
    implementation, python_pin, pins = _parse_requirements_contract(requirements_text)
    _require(
        platform.python_implementation().casefold() == implementation.casefold(),
        "environment",
        "python_implementation",
        "Verifier runtime implementation differs from the canonical interpreter pin",
        expected=implementation,
        actual=platform.python_implementation(),
    )
    runtime = _runtime_software()
    software = _mapping(
        _nested(summary, ["software"], "environment"),
        "environment",
        "software",
    )
    _require(
        set(software) == set(runtime),
        "environment",
        "software_exact_set",
        "summary.software must contain exactly the pinned, native, and runtime-control fields",
        expected=sorted(runtime),
        actual=sorted(software),
    )
    _require(
        python_pin == runtime["python"] == str(software["python"]),
        "environment",
        "python_version",
        "Pinned, summary, and verifier-runtime Python versions differ",
        expected={"requirements": python_pin, "runtime": runtime["python"]},
        actual=software["python"],
    )
    for package, pinned_version in pins.items():
        _require(
            pinned_version == runtime[package] == str(software[package]),
            "environment",
            "package_version",
            f"Pinned, summary, and verifier-runtime versions differ for {package}",
            path=package,
            expected={"requirements": pinned_version, "runtime": runtime[package]},
            actual=software[package],
        )
    runtime_only_fields = (
        "platform",
        "machine",
        "gdal",
        "geos",
        "proj",
        "geopandas_io_engine",
        "address_parser",
    )
    for field in runtime_only_fields:
        _require(
            str(software[field]) == runtime[field],
            "environment",
            "runtime_software",
            f"summary.software.{field} differs from the verifier runtime",
            path=field,
            expected=runtime[field],
            actual=software[field],
        )
    return {
        "requirements_path": REQUIREMENTS_ANALYSIS_PATH.relative_to(ROOT).as_posix(),
        "requirements_sha256": sha256_file(REQUIREMENTS_ANALYSIS_PATH),
        "python": {"implementation": implementation, "version": python_pin},
        "exact_pins_verified_at_runtime": pins,
        "native_libraries_verified_at_runtime": {
            field: runtime[field] for field in ("gdal", "geos", "proj")
        },
        "runtime_controls": {
            field: runtime[field]
            for field in ("platform", "machine", "geopandas_io_engine", "address_parser")
        },
    }


def _required_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    _require(
        not missing,
        "outputs",
        "required_columns",
        f"{label} is missing required columns",
        expected=missing,
    )


def _boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    text = series.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(text) - set(mapping))
    _require(
        not invalid,
        "outputs",
        "boolean_parse",
        f"{label} contains non-boolean values",
        actual=invalid[:10],
    )
    return text.map(mapping).astype(bool)


def _numeric(series: pd.Series, label: str, *, finite: bool = True) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        _fail("outputs", "numeric_parse", f"{label} is not numeric: {exc}")
    if finite:
        _require(
            bool(np.isfinite(values.to_numpy(dtype=float)).all()),
            "outputs",
            "finite_values",
            f"{label} contains non-finite values",
        )
    return values


def _same_number(actual: Any, expected: Any, *, tolerance: float = 1e-8) -> bool:
    actual_missing = actual is None or (
        np.isscalar(actual) and bool(pd.isna(actual))
    )
    expected_missing = expected is None or (
        np.isscalar(expected) and bool(pd.isna(expected))
    )
    if actual_missing or expected_missing:
        return actual_missing and expected_missing
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _check_number(
    actual: Any,
    expected: Any,
    check: str,
    *,
    tolerance: float = 1e-8,
) -> None:
    _require(
        _same_number(actual, expected, tolerance=tolerance),
        "invariants",
        check,
        f"Numeric invariant failed for {check}",
        expected=expected,
        actual=actual,
    )


def _rate_percent(values: pd.Series) -> float | None:
    return round(100.0 * float(values.astype(bool).mean()), 2) if len(values) else None


def _weighted_rate_percent(values: pd.Series, weights: pd.Series) -> float | None:
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & numeric_weights.notna() & numeric_weights.gt(0)
    if not valid.any() or float(numeric_weights.loc[valid].sum()) <= 0:
        return None
    result = np.average(
        values.loc[valid].astype(float),
        weights=numeric_weights.loc[valid].astype(float),
    )
    return round(100.0 * float(result), 2)


def _confusion(board_low: pd.Series, google_low: pd.Series) -> dict[str, Any]:
    board = board_low.astype(bool)
    google = google_low.astype(bool)
    both_low = int((board & google).sum())
    both_covered = int((~board & ~google).sum())
    google_only = int((~board & google).sum())
    board_only = int((board & ~google).sum())
    total = len(board)
    board_positive = int(board.sum())
    board_negative = total - board_positive
    return {
        "n": total,
        "both_beyond_threshold": both_low,
        "both_within_threshold": both_covered,
        "google_only_beyond_threshold": google_only,
        "board_only_beyond_threshold": board_only,
        "agreement_pct": round(100.0 * (both_low + both_covered) / total, 2)
        if total
        else None,
        "google_beyond_given_board_beyond_pct": round(100.0 * both_low / board_positive, 2)
        if board_positive
        else None,
        "google_within_given_board_within_pct": round(
            100.0 * both_covered / board_negative, 2
        )
        if board_negative
        else None,
    }


def _check_confusion(actual: Mapping[str, Any], expected: Mapping[str, Any], check: str) -> None:
    for key, value in expected.items():
        _require(key in actual, "invariants", check, f"Summary confusion table is missing {key}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _check_number(actual[key], value, f"{check}.{key}")
        else:
            _require(
                actual[key] == value,
                "invariants",
                f"{check}.{key}",
                f"Confusion invariant failed for {check}.{key}",
                expected=value,
                actual=actual[key],
            )


def _nonempty_unique(series: pd.Series, label: str) -> None:
    text = series.astype(str).str.strip()
    _require(not text.eq("").any(), "invariants", "nonempty_id", f"{label} contains empty IDs")
    _require(not text.duplicated().any(), "invariants", "unique_id", f"{label} contains duplicate IDs")


def _record_id(value: Any) -> str:
    """Canonicalize CSV's harmless integral-float rendering of numeric IDs."""
    text = str(value).strip()
    match = re.fullmatch(r"([+-]?\d+)\.0+", text)
    return match.group(1) if match else text


def _check_signature_recurrence(frame: pd.DataFrame, label: str) -> None:
    _required_columns(
        frame,
        ("name_component_id", "location_key", "signature_location_count"),
        label,
    )
    _require(
        not frame["name_component_id"].astype(str).str.strip().eq("").any(),
        "invariants",
        "name_component_id",
        f"{label} contains empty name-component IDs",
    )
    _require(
        not frame["location_key"].astype(str).str.strip().eq("").any(),
        "invariants",
        "location_key",
        f"{label} contains empty location keys",
    )
    recurrence = frame.groupby("name_component_id")["location_key"].nunique()
    observed = _numeric(frame["signature_location_count"], f"{label}.signature_location_count")
    expected = frame["name_component_id"].map(recurrence).astype(int)
    _require(
        observed.astype(int).equals(expected),
        "invariants",
        "signature_recurrence",
        f"{label} recurrence counts do not equal distinct source-local locations",
    )


def _primary_linkage(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    frame = _mapping(
        _nested(summary, ["linkage", "primary_fixed_target_frame"], "invariants"),
        "invariants",
        "primary_fixed_target_frame",
    )
    primary_name = _nested(frame, ["primary_strategy"], "invariants")
    strategies = _sequence(_nested(frame, ["strategies"], "invariants"), "invariants", "strategies")
    matches = [
        row
        for row in strategies
        if isinstance(row, Mapping) and row.get("strategy") == primary_name
    ]
    _require(
        len(matches) == 1,
        "invariants",
        "primary_strategy",
        "Exactly one strategy row must match primary_strategy",
        expected=primary_name,
        actual=len(matches),
    )
    return frame, matches[0]


def _check_linkage_frame(frame: Mapping[str, Any], label: str) -> None:
    ai_rows = int(_nested(frame, ["ai_rows"], "invariants"))
    criterion_rows = int(_nested(frame, ["criterion_rows"], "invariants"))
    _require(
        ai_rows > 0 and criterion_rows > 0,
        "invariants",
        "linkage_frame_rows",
        f"{label} must contain positive source-frame counts",
        actual={"ai_rows": ai_rows, "criterion_rows": criterion_rows},
    )
    primary_name = str(_nested(frame, ["primary_strategy"], "invariants"))
    primary_objective = str(
        _nested(frame, ["primary_assignment_objective"], "invariants")
    )
    strategies = [
        _mapping(value, "invariants", f"{label}.strategy")
        for value in _sequence(frame["strategies"], "invariants", f"{label}.strategies")
    ]
    strategy_names = [str(value["strategy"]) for value in strategies]
    _require(
        len(strategy_names) == len(set(strategy_names)),
        "invariants",
        "linkage_strategy_unique",
        f"{label} contains duplicate strategy rows",
    )
    _require(
        strategy_names.count(primary_name) == 1,
        "invariants",
        "primary_strategy",
        f"{label} must contain its declared primary strategy exactly once",
    )
    linked_by_strategy: dict[str, int] = {}
    for strategy in strategies:
        name = str(strategy["strategy"])
        linked = int(strategy["linked_pairs"])
        linked_by_strategy[name] = linked
        _require(
            0 <= linked <= min(ai_rows, criterion_rows),
            "invariants",
            "linked_pair_bounds",
            f"{label}.{name} has an impossible one-to-one pair count",
            actual=linked,
        )
        _check_number(strategy["unlinked_ai"], ai_rows - linked, f"{label}.{name}.unlinked_ai")
        _check_number(strategy["unlinked_board"], criterion_rows - linked, f"{label}.{name}.unlinked_board")
        _check_number(strategy["candidate_linked_pct"], round(100.0 * linked / ai_rows, 2), f"{label}.{name}.candidate_linked_pct")
        _check_number(strategy["criterion_covered_pct"], round(100.0 * linked / criterion_rows, 2), f"{label}.{name}.criterion_covered_pct")
        _require(0.0 <= float(strategy["threshold"]) <= 100.0, "invariants", "linkage_threshold", f"{label}.{name} threshold lies outside the 0--100 score scale", actual=strategy["threshold"])

    primary_linked = linked_by_strategy[primary_name]
    primary_threshold = float(
        next(value for value in strategies if value["strategy"] == primary_name)["threshold"]
    )
    threshold_rows = [
        _mapping(value, "invariants", f"{label}.threshold_sensitivity")
        for value in _sequence(
            frame["primary_threshold_sensitivity"],
            "invariants",
            f"{label}.primary_threshold_sensitivity",
        )
    ]
    _require(
        len({float(value["threshold"]) for value in threshold_rows}) == len(threshold_rows),
        "invariants",
        "threshold_sensitivity_unique",
        f"{label} contains duplicate threshold-sensitivity rows",
    )
    for value in threshold_rows:
        linked = int(value["linked_pairs"])
        _require(0 <= linked <= min(ai_rows, criterion_rows), "invariants", "threshold_link_bounds", f"{label} threshold sensitivity has an impossible pair count", actual=linked)
        _check_number(value["candidate_linked_pct"], round(100.0 * linked / ai_rows, 2), f"{label}.threshold.candidate_linked_pct")
        _check_number(value["criterion_covered_pct"], round(100.0 * linked / criterion_rows, 2), f"{label}.threshold.criterion_covered_pct")
    matching_threshold = [
        value for value in threshold_rows if float(value["threshold"]) == primary_threshold
    ]
    _require(len(matching_threshold) == 1, "invariants", "primary_threshold_sensitivity", f"{label} lacks exactly one sensitivity row at its primary threshold")
    _check_number(matching_threshold[0]["linked_pairs"], primary_linked, f"{label}.primary_threshold_links")

    objective_rows = [
        _mapping(value, "invariants", f"{label}.objective_sensitivity")
        for value in _sequence(
            frame["primary_objective_sensitivity"],
            "invariants",
            f"{label}.primary_objective_sensitivity",
        )
    ]
    _require(len({str(value["objective"]) for value in objective_rows}) == len(objective_rows), "invariants", "objective_sensitivity_unique", f"{label} contains duplicate assignment-objective rows")
    for value in objective_rows:
        linked = int(value["linked_pairs"])
        total_similarity = float(value["total_similarity"])
        _require(0 <= linked <= min(ai_rows, criterion_rows), "invariants", "objective_link_bounds", f"{label} objective sensitivity has an impossible pair count", actual=linked)
        _require(0.0 <= total_similarity <= 100.0 * linked, "invariants", "objective_similarity_bounds", f"{label} objective total similarity lies outside the 0--100 score contract", actual=total_similarity)
    matching_objective = [
        value for value in objective_rows if str(value["objective"]) == primary_objective
    ]
    _require(len(matching_objective) == 1, "invariants", "primary_objective_sensitivity", f"{label} lacks exactly one row for its primary assignment objective")
    _check_number(matching_objective[0]["linked_pairs"], primary_linked, f"{label}.primary_objective_links")

    overlap_rows = [
        _mapping(value, "invariants", f"{label}.overlap")
        for value in _sequence(
            frame["matched_license_overlap"],
            "invariants",
            f"{label}.matched_license_overlap",
        )
    ]
    _require({str(value["strategy"]) for value in overlap_rows} == set(strategy_names), "invariants", "overlap_strategy_coverage", f"{label} overlap rows do not cover every strategy exactly")
    for value in overlap_rows:
        name = str(value["strategy"])
        matched = int(value["matched_board_licenses"])
        overlap = int(value["overlap_with_primary"])
        _check_number(matched, linked_by_strategy[name], f"{label}.{name}.matched_board_licenses")
        _check_number(value["only_this_strategy"], matched - overlap, f"{label}.{name}.only_this_strategy")
        _check_number(value["only_primary"], primary_linked - overlap, f"{label}.{name}.only_primary")


def _verify_inventory_and_linkage(
    summary: Mapping[str, Any],
    criterion: pd.DataFrame,
    extended: pd.DataFrame,
    links: pd.DataFrame,
    status: pd.DataFrame,
) -> dict[str, int]:
    _require(summary.get("manual_labels_used") is False, "invariants", "manual_labels", "manual_labels_used must be false")
    _require(summary.get("ownership_inference_made") is False, "invariants", "ownership", "ownership_inference_made must be false")
    _require(summary.get("naics_accuracy_claim_made") is False, "invariants", "naics_claim", "naics_accuracy_claim_made must be false")
    _require(normalize_name_signature("Wal-Mart Pharmacy #123") == "walmart", "invariants", "normalization", "Wal-Mart normalization changed")
    _require(normalize_name_signature("Sam's Club Pharmacy") == "sams club", "invariants", "normalization", "Sam's Club normalization changed")
    _require(normalize_name_signature("Hy-Vee Pharmacy") == "hyvee", "invariants", "normalization", "Hy-Vee normalization changed")
    _require(normalize_legal_signature("Example L.L.C. d/b/a Lake Pharmacy") == "example", "invariants", "normalization", "Legal-name normalization changed")

    inventory = _mapping(_nested(summary, ["inventory"], "invariants"), "invariants", "inventory")
    linkage_frame, primary = _primary_linkage(summary)
    _check_linkage_frame(linkage_frame, "primary_fixed_target_frame")
    sensitivities = _mapping(
        _nested(summary, ["linkage", "sensitivities"], "invariants"),
        "invariants",
        "linkage_sensitivities",
    )
    for label, sensitivity in sensitivities.items():
        _check_linkage_frame(
            _mapping(sensitivity, "invariants", f"linkage.sensitivities.{label}"),
            f"sensitivity.{label}",
        )
    required_inventory = (
        "board_uniform_criterion_fixed_target_rows",
        "board_uniform_criterion_fixed_target_geocoded_rows",
        "board_uniform_criterion_extended_rows",
        "board_uniform_criterion_extended_geocoded_rows",
    )
    for key in required_inventory:
        _nested(inventory, [key], "invariants")

    _required_columns(
        criterion,
        ("license_nbr", "zip5", "name_signature", "name_component_id", "location_key", "signature_location_count", "has_geocode"),
        "fixed Board criterion",
    )
    _required_columns(extended, criterion.columns, "extended Board criterion")
    _nonempty_unique(criterion["license_nbr"], "fixed Board criterion license_nbr")
    _nonempty_unique(extended["license_nbr"], "extended Board criterion license_nbr")
    criterion_ids = set(criterion["license_nbr"].astype(str))
    extended_ids = set(extended["license_nbr"].astype(str))
    _require(criterion_ids <= extended_ids, "invariants", "criterion_subset", "Fixed Board criterion is not a subset of the extended criterion")
    fixed_zips = criterion["zip5"].astype(str).str.zfill(5)
    _require(set(fixed_zips) <= set(TARGET_ZIPS), "invariants", "target_zip_frame", "Fixed Board criterion contains ZIPs outside the declared target frame", actual=sorted(set(fixed_zips) - set(TARGET_ZIPS)))
    _check_signature_recurrence(criterion, "fixed Board criterion")
    _check_signature_recurrence(extended, "extended Board criterion")

    criterion_geocoded = int(_boolean(criterion["has_geocode"], "criterion.has_geocode").sum())
    extended_geocoded = int(_boolean(extended["has_geocode"], "extended.has_geocode").sum())
    _check_number(len(criterion), inventory["board_uniform_criterion_fixed_target_rows"], "criterion_rows.inventory")
    _check_number(len(criterion), linkage_frame["criterion_rows"], "criterion_rows.linkage")
    _check_number(criterion_geocoded, inventory["board_uniform_criterion_fixed_target_geocoded_rows"], "criterion_geocoded")
    _check_number(len(extended), inventory["board_uniform_criterion_extended_rows"], "extended_rows")
    _check_number(extended_geocoded, inventory["board_uniform_criterion_extended_geocoded_rows"], "extended_geocoded")

    _required_columns(
        links,
        ("ai_row_id", "board_license_nbr", "composite_score", "name_similarity", "address_similarity"),
        "record links",
    )
    _nonempty_unique(links["ai_row_id"], "record-link ai_row_id")
    _nonempty_unique(links["board_license_nbr"], "record-link board_license_nbr")
    _require(set(links["board_license_nbr"].astype(str)) <= criterion_ids, "invariants", "linked_board_subset", "Record links reference Board licenses outside the fixed criterion")
    _check_number(len(links), primary["linked_pairs"], "primary_linked_pairs")
    ai_rows = int(linkage_frame["ai_rows"])
    criterion_rows = len(criterion)
    _check_number(primary["unlinked_ai"], ai_rows - len(links), "primary_unlinked_ai")
    _check_number(primary["unlinked_board"], criterion_rows - len(links), "primary_unlinked_board")
    _check_number(primary["candidate_linked_pct"], round(100.0 * len(links) / ai_rows, 2), "candidate_linked_pct")
    _check_number(primary["criterion_covered_pct"], round(100.0 * len(links) / criterion_rows, 2), "criterion_covered_pct")
    score = _numeric(links["composite_score"], "links.composite_score")
    threshold = float(primary["threshold"])
    _require(bool(score.between(threshold, 100.0, inclusive="both").all()), "invariants", "link_score_range", "Primary link scores must lie between the declared threshold and 100", expected=[threshold, 100.0], actual=[float(score.min()), float(score.max())])
    for column in ("name_similarity", "address_similarity"):
        values = _numeric(links[column], f"links.{column}")
        _require(bool(values.between(0.0, 100.0, inclusive="both").all()), "invariants", "similarity_range", f"{column} must lie on a 0--100 scale")

    _required_columns(status, ("source", "source_record_id", "linked", "linked_source_record_id"), "source status")
    _require(not status.duplicated(["source", "source_record_id"]).any(), "invariants", "status_unique", "Source status contains duplicate source-record rows")
    expected_source_counts = {
        BOARD_SOURCE: criterion_rows,
        GOOGLE_SOURCE: ai_rows,
    }
    actual_source_counts = {
        str(key): int(value) for key, value in status["source"].value_counts().items()
    }
    _require(actual_source_counts == expected_source_counts, "invariants", "status_source_counts", "Source-status row counts do not match the primary linkage frame", expected=expected_source_counts, actual=actual_source_counts)
    linked_status = _boolean(status["linked"], "status.linked")
    _check_number(int(linked_status.sum()), 2 * len(links), "status_linked_rows")

    status_work = status.copy()
    status_work["_linked"] = linked_status
    pair_ai_to_board = dict(
        zip(
            links["ai_row_id"].map(_record_id),
            links["board_license_nbr"].map(_record_id),
        )
    )
    pair_board_to_ai = {value: key for key, value in pair_ai_to_board.items()}
    for source, expected_links in (
        (GOOGLE_SOURCE, pair_ai_to_board),
        (BOARD_SOURCE, pair_board_to_ai),
    ):
        rows = status_work.loc[status_work["source"].eq(source)].copy()
        observed_ids = set(rows.loc[rows["_linked"], "source_record_id"].map(_record_id))
        _require(observed_ids == set(expected_links), "invariants", "status_link_membership", f"Linked {source} status rows do not match the pair table", expected=sorted(expected_links), actual=sorted(observed_ids))
        for _, row in rows.iterrows():
            source_id = _record_id(row["source_record_id"])
            linked_id = _record_id(row["linked_source_record_id"])
            if bool(row["_linked"]):
                _require(linked_id == expected_links[source_id], "invariants", "status_reciprocity", "Source-status counterpart does not match the pair table", expected=expected_links[source_id], actual=linked_id)
            else:
                _require(linked_id == "", "invariants", "unlinked_counterpart", "Unlinked source-status row has a counterpart ID", actual=linked_id)

    return {
        "fixed_board_rows": criterion_rows,
        "extended_board_rows": len(extended),
        "google_location_rows": ai_rows,
        "primary_linked_pairs": len(links),
    }


def _verify_signatures(
    summary: Mapping[str, Any],
    signature_summary: Mapping[str, Any],
    groups: pd.DataFrame,
    counts: Mapping[str, int],
) -> None:
    embedded = _nested(summary, ["name_signatures"], "invariants")
    _require(embedded == signature_summary, "invariants", "signature_summary_copy", "Standalone name-signature summary differs from analysis_summary.json")
    rows_payload = _sequence(_nested(signature_summary, ["rows"], "invariants"), "invariants", "signature_rows")
    rows = pd.DataFrame(rows_payload)
    _required_columns(rows, ("source", "linkage_strategy", "recurrence_threshold", "group_code", "group", "records", "linked_records", "linked_pct"), "signature summary rows")
    thresholds = [
        int(value)
        for value in _sequence(
            _nested(signature_summary, ["thresholds"], "invariants"),
            "invariants",
            "signature_thresholds",
        )
    ]
    _require(len(thresholds) == len(set(thresholds)) and all(value >= 2 for value in thresholds), "invariants", "signature_thresholds", "Signature thresholds must be unique integers of at least two", actual=thresholds)
    strategies = list(dict.fromkeys(rows["linkage_strategy"].astype(str)))
    _require(bool(strategies), "invariants", "signature_strategies", "Signature summary contains no linkage strategies")
    summary_key = [
        "source",
        "linkage_strategy",
        "recurrence_threshold",
        "group_code",
    ]
    _require(
        not rows.duplicated(summary_key).any(),
        "invariants",
        "signature_summary_unique",
        "Name-signature summary contains duplicate source/strategy/threshold/group rows",
    )
    source_counts = {
        BOARD_SOURCE: counts["fixed_board_rows"],
        GOOGLE_SOURCE: counts["google_location_rows"],
    }

    _required_columns(groups, ("source", "source_record_id", "name_component_id", "location_key", "signature_location_count", "linkage_strategy", "recurrence_threshold", "name_component_group", "linked"), "name-signature detail")
    groups = groups.copy()
    groups["recurrence_threshold"] = _numeric(groups["recurrence_threshold"], "groups.recurrence_threshold").astype(int)
    groups["signature_location_count"] = _numeric(groups["signature_location_count"], "groups.signature_location_count").astype(int)
    groups["_linked"] = _boolean(groups["linked"], "groups.linked")
    duplicate_key = ["source", "source_record_id", "linkage_strategy", "recurrence_threshold"]
    _require(not groups.duplicated(duplicate_key).any(), "invariants", "signature_detail_unique", "Name-signature detail contains duplicate source/record/strategy/threshold rows")
    _require(set(groups["source"].astype(str)) == set(source_counts), "invariants", "signature_sources", "Name-signature detail has unexpected sources", expected=sorted(source_counts), actual=sorted(set(groups["source"].astype(str))))
    _require(set(groups["linkage_strategy"].astype(str)) == set(strategies), "invariants", "signature_strategies", "Name-signature detail and summary strategies differ")
    _require(set(groups["recurrence_threshold"]) == set(thresholds), "invariants", "signature_thresholds", "Name-signature detail and summary thresholds differ")
    expected_detail_rows = sum(source_counts.values()) * len(strategies) * len(thresholds)
    _check_number(len(groups), expected_detail_rows, "signature_detail_rows")

    base_columns = ["source", "source_record_id", "name_component_id", "location_key", "signature_location_count"]
    base = groups[base_columns].drop_duplicates()
    _check_number(len(base), sum(source_counts.values()), "signature_base_records")
    for source, source_count in source_counts.items():
        source_base = base.loc[base["source"].eq(source)].copy()
        _check_number(len(source_base), source_count, f"signature_base_records.{source}")
        _check_signature_recurrence(source_base, f"signature detail {source}")
    linked_consistency = groups.groupby(["source", "source_record_id", "linkage_strategy"])["_linked"].nunique()
    _require(bool(linked_consistency.eq(1).all()), "invariants", "signature_linked_threshold_consistency", "A record's linked status changes across recurrence thresholds")

    strategy_link_counts: dict[str, int] = {}
    linkage_frame, _ = _primary_linkage(summary)
    for strategy_row in _sequence(linkage_frame["strategies"], "invariants", "strategies"):
        strategy = _mapping(strategy_row, "invariants", "strategy_row")
        strategy_link_counts[str(strategy["strategy"])] = int(strategy["linked_pairs"])

    for row in rows.to_dict(orient="records"):
        source = str(row["source"])
        strategy = str(row["linkage_strategy"])
        threshold = int(row["recurrence_threshold"])
        group_name = str(row["group"])
        subset = groups.loc[
            groups["source"].eq(source)
            & groups["linkage_strategy"].eq(strategy)
            & groups["recurrence_threshold"].eq(threshold)
            & groups["name_component_group"].eq(group_name)
        ]
        records = len(subset)
        linked = int(subset["_linked"].sum())
        linked_pct = round(100.0 * linked / records, 2) if records else None
        _check_number(records, row["records"], "signature_summary.records")
        _check_number(linked, row["linked_records"], "signature_summary.linked_records")
        _check_number(linked_pct, row["linked_pct"], "signature_summary.linked_pct")
        for bound in ("cluster_bootstrap_95_low_pct", "cluster_bootstrap_95_high_pct"):
            value = row.get(bound)
            if value is not None and not bool(pd.isna(value)):
                _require(0.0 <= float(value) <= 100.0, "invariants", "signature_interval_range", f"{bound} lies outside 0--100", actual=value)

    for source, source_count in source_counts.items():
        for strategy in strategies:
            for threshold in thresholds:
                subset = rows.loc[
                    rows["source"].eq(source)
                    & rows["linkage_strategy"].eq(strategy)
                    & pd.to_numeric(rows["recurrence_threshold"]).eq(threshold)
                ]
                _check_number(pd.to_numeric(subset["records"]).sum(), source_count, "signature_partition_records")
                _check_number(pd.to_numeric(subset["linked_records"]).sum(), strategy_link_counts[strategy], "signature_partition_links")

    gaps = _sequence(_nested(signature_summary, ["repeated_minus_below_threshold_gaps"], "invariants"), "invariants", "signature_gaps")
    for gap_payload in gaps:
        gap = _mapping(gap_payload, "invariants", "signature_gap")
        subset = rows.loc[
            rows["source"].eq(gap["source"])
            & rows["linkage_strategy"].eq(gap["linkage_strategy"])
            & pd.to_numeric(rows["recurrence_threshold"]).eq(int(gap["recurrence_threshold"]))
        ].set_index("group_code")
        _require({"repeated", "below_threshold"} <= set(subset.index), "invariants", "signature_gap_groups", "Signature gap lacks repeated or below-threshold groups")
        expected_gap = round(
            float(
                subset.loc["repeated", "linked_pct"]
                - subset.loc["below_threshold", "linked_pct"]
            ),
            2,
        )
        _check_number(gap["repeated_minus_below_threshold_percentage_points"], expected_gap, "signature_gap")


def _recompute_spatial_source_audit() -> dict[str, Any]:
    """Rebuild the origin frame from authenticated TIGER/ACS inputs."""
    county_fips = [code for code, _ in SPATIAL_COUNTY_FRAME]
    tracts = gpd.read_file(f"zip://{TRACT_SOURCE_PATH.as_posix()}")
    _require(tracts.crs is not None, "invariants", "tract_source_crs", "Tract source has no CRS metadata")
    tracts = tracts.loc[tracts["COUNTYFP"].astype(str).isin(county_fips)].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    _require(not tracts["GEOID"].duplicated().any(), "invariants", "tract_source_geoid", "Authenticated tract source contains duplicate GEOIDs")

    acs = pd.read_csv(ACS_SOURCE_PATH, dtype={"GEOID": str})
    vehicles = pd.read_csv(VEHICLES_SOURCE_PATH, dtype={"GEOID": str})
    _require(not acs["GEOID"].duplicated().any(), "invariants", "acs_geoid", "Authenticated ACS source contains duplicate GEOIDs")
    _require(not vehicles["GEOID"].duplicated().any(), "invariants", "vehicles_geoid", "Authenticated vehicle source contains duplicate GEOIDs")
    joined = tracts.merge(acs, on="GEOID", how="left", validate="one_to_one")
    joined = joined.merge(vehicles, on="GEOID", how="left", validate="one_to_one")
    valid_income = pd.to_numeric(joined["med_hh_income"], errors="coerce").gt(0)
    household_count = pd.to_numeric(joined["total_hh"], errors="coerce")
    valid_households = household_count.notna() & household_count.gt(0)
    eligible = joined.loc[valid_income & valid_households].copy()

    zctas = gpd.read_file(
        f"zip://{ZCTA_SOURCE_PATH.as_posix()}",
        bbox=(-94.6, 44.1, -91.9, 46.1),
    )
    _require(zctas.crs is not None, "invariants", "zcta_source_crs", "ZCTA source has no CRS metadata")
    zctas["ZCTA5CE20"] = zctas["ZCTA5CE20"].astype(str).str.zfill(5)
    target_zctas = zctas.loc[
        zctas["ZCTA5CE20"].isin(TARGET_ZIPS), ["ZCTA5CE20", "geometry"]
    ].copy()

    internal_points = gpd.GeoDataFrame(
        eligible[["GEOID", "COUNTYFP"]].copy(),
        geometry=gpd.points_from_xy(
            pd.to_numeric(eligible["INTPTLON"], errors="coerce"),
            pd.to_numeric(eligible["INTPTLAT"], errors="coerce"),
        ),
        crs=tracts.crs,
    )
    if internal_points.crs != target_zctas.crs:
        internal_points = internal_points.to_crs(target_zctas.crs)
    origin_join = gpd.sjoin(
        internal_points,
        target_zctas,
        how="inner",
        predicate="intersects",
    )
    _require(not origin_join["GEOID"].duplicated().any(), "invariants", "origin_source_membership", "An authenticated tract internal point intersects multiple target ZCTAs")
    origins = origin_join[["GEOID", "COUNTYFP", "ZCTA5CE20"]].rename(
        columns={"ZCTA5CE20": "target_zcta"}
    )
    origins["GEOID"] = origins["GEOID"].astype(str)
    origins["COUNTYFP"] = origins["COUNTYFP"].astype(str).str.zfill(3)
    origins["target_zcta"] = origins["target_zcta"].astype(str).str.zfill(5)
    return {
        "county_fips": county_fips,
        "tract_source_rows_hennepin_ramsey": len(tracts),
        "tract_rows_invalid_or_missing_median_household_income": int((~valid_income).sum()),
        "tract_rows_missing_household_count": int(household_count.isna().sum()),
        "tract_rows_nonpositive_or_missing_household_count": int((~valid_households).sum()),
        "eligible_tracts_after_acs": len(eligible),
        "target_zctas_loaded": int(target_zctas["ZCTA5CE20"].nunique()),
        "origins": pd.DataFrame(origins).reset_index(drop=True),
    }


def _verify_spatial_origin_contract(
    summary: Mapping[str, Any],
    tracts: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any],
) -> None:
    """Bind the reported two-county origin estimand to authenticated source rows."""
    spatial = _mapping(_nested(summary, ["spatial"], "invariants"), "invariants", "spatial")
    frame = _mapping(spatial.get("spatial_origin_frame"), "invariants", "spatial_origin_frame")
    expected_fips = [code for code, _ in SPATIAL_COUNTY_FRAME]
    expected_names = [name for _, name in SPATIAL_COUNTY_FRAME]
    _require(frame.get("county_fips") == expected_fips, "invariants", "spatial_county_fips", "Spatial origin frame must be restricted to Hennepin and Ramsey County FIPS", expected=expected_fips, actual=frame.get("county_fips"))
    _require(frame.get("county_names") == expected_names, "invariants", "spatial_county_names", "Spatial origin frame county names are stale", expected=expected_names, actual=frame.get("county_names"))
    eligibility = str(frame.get("acs_eligibility", "")).casefold()
    _require("greater than zero" in eligibility and "sentinel" in eligibility, "invariants", "spatial_acs_eligibility", "ACS eligibility must require positive income/households and reject negative sentinel values", actual=frame.get("acs_eligibility"))
    membership_text = str(frame.get("target_membership", "")).casefold()
    _require("internal point" in membership_text and "60" in membership_text and "zcta" in membership_text, "invariants", "spatial_target_membership", "Spatial origin membership rule is incomplete", actual=frame.get("target_membership"))

    crs_assumption = _mapping(spatial.get("supply_coordinate_crs_assumption"), "invariants", "supply_coordinate_crs_assumption")
    crs_basis = str(crs_assumption.get("basis", "")).casefold()
    crs_scope = str(crs_assumption.get("scope", "")).casefold()
    _require(
        crs_assumption.get("assumed_crs") == SUPPLY_INPUT_CRS_ASSUMPTION
        and "google" in crs_scope
        and "board" in crs_scope
        and "machine-readable crs metadata" in crs_basis,
        "invariants",
        "supply_coordinate_crs_assumption",
        "Supply inputs must disclose the shared EPSG:4326 assumption and absent machine-readable CRS metadata",
        actual=crs_assumption,
    )

    audit_keys = (
        "tract_source_rows_hennepin_ramsey",
        "tract_rows_invalid_or_missing_median_household_income",
        "tract_rows_missing_household_count",
        "tract_rows_nonpositive_or_missing_household_count",
        "eligible_tracts_after_acs",
        "target_zctas_loaded",
    )
    for key in audit_keys:
        _check_number(spatial[key], source_audit[key], key)

    expected_origins = pd.DataFrame(source_audit["origins"]).copy()
    observed_origins = tracts[["GEOID", "COUNTYFP", "target_zcta"]].copy()
    for frame_data in (expected_origins, observed_origins):
        frame_data["GEOID"] = frame_data["GEOID"].astype(str)
        frame_data["COUNTYFP"] = frame_data["COUNTYFP"].astype(str).str.zfill(3)
        frame_data["target_zcta"] = frame_data["target_zcta"].astype(str).str.zfill(5)
    expected_membership = set(map(tuple, expected_origins[["GEOID", "COUNTYFP", "target_zcta"]].to_numpy()))
    observed_membership = set(map(tuple, observed_origins[["GEOID", "COUNTYFP", "target_zcta"]].to_numpy()))
    _require(observed_membership == expected_membership, "invariants", "spatial_origin_membership", "Canonical tract origins differ from the authenticated TIGER/ACS/ZCTA reconstruction", expected=len(expected_membership), actual=len(observed_membership))

    origin_count = len(expected_origins)
    zcta_count = int(expected_origins["target_zcta"].nunique())
    _check_number(spatial["fixed_target_zcta_origins"], origin_count, "fixed_target_zcta_origins")
    _check_number(spatial["target_zctas_with_at_least_one_origin"], zcta_count, "target_zctas_with_origins")
    missing_zctas = sorted(set(TARGET_ZIPS) - set(expected_origins["target_zcta"]))
    _require(sorted(spatial["target_zctas_without_eligible_internal_point_origin"]) == missing_zctas, "invariants", "target_zctas_without_origin", "Declared target ZCTAs without origins differ from authenticated sources", expected=missing_zctas, actual=spatial["target_zctas_without_eligible_internal_point_origin"])
    definition = str(summary.get("spatial_origin_definition", ""))
    definition_folded = definition.casefold()
    _require(
        "hennepin" in definition_folded
        and "ramsey" in definition_folded
        and "acs-eligible" in definition_folded
        and f"{origin_count} origins" in definition_folded
        and f"{zcta_count} target zctas" in definition_folded,
        "invariants",
        "spatial_origin_definition",
        "Top-level spatial origin definition is incomplete or contains stale counts",
        actual=definition,
    )


def _verify_spatial(
    summary: Mapping[str, Any],
    tracts: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any] | None = None,
) -> int:
    spatial = _mapping(_nested(summary, ["spatial"], "invariants"), "invariants", "spatial")
    if source_audit is not None:
        _verify_spatial_origin_contract(summary, tracts, source_audit=source_audit)
    _require(
        spatial.get("distance_crs") == "EPSG:26915",
        "invariants",
        "distance_crs",
        "Canonical proximity distances must use the declared projected CRS EPSG:26915",
        expected="EPSG:26915",
        actual=spatial.get("distance_crs"),
    )
    axis_units = [str(value).casefold() for value in spatial.get("distance_crs_axis_units", [])]
    _require(
        bool(axis_units) and all(value in {"metre", "meter"} for value in axis_units),
        "invariants",
        "distance_units",
        "Projected-distance axis units must be metres",
        actual=axis_units,
    )
    _required_columns(tracts, ("GEOID", "target_zcta", "frame_edge_m", "missing_board_mailing_zip_zcta_min_m", "google_nearest_m", "google_nearest_tie_count", "google_raw_candidate_nearest_m", "google_raw_candidate_nearest_tie_count", "board_nearest_m", "board_nearest_tie_count", "board_exact_geocode_only_nearest_m", "board_exact_geocode_only_nearest_tie_count", "board_legacy_405_nearest_m", "board_all_licensed_nearest_m", "google_minus_board_m", "centroid_google_nearest_m", "centroid_board_nearest_m", "internal_point_within_own_tract", "centroid_within_own_tract", "total_pop", "total_hh", "income_quartile", "nonwhite_quartile"), "tract proximity")
    _require(
        "missing_board_zcta_min_m" not in tracts.columns,
        "invariants",
        "legacy_spatial_schema",
        "Legacy unqualified missing_board_zcta_min_m must not appear in canonical output",
    )
    _nonempty_unique(tracts["GEOID"], "tract GEOID")
    target_zctas = set(tracts["target_zcta"].astype(str).str.zfill(5))
    _require(target_zctas <= set(TARGET_ZIPS), "invariants", "tract_target_zcta", "Tract results contain origins outside the target ZCTAs", actual=sorted(target_zctas - set(TARGET_ZIPS)))
    _check_number(len(tracts), spatial["fixed_target_zcta_origins"], "fixed_target_zcta_origins")
    _check_number(tracts["target_zcta"].astype(str).nunique(), spatial["target_zctas_with_at_least_one_origin"], "target_zctas_with_origins")

    distance_columns = (
        "google_nearest_m",
        "board_nearest_m",
        "board_exact_geocode_only_nearest_m",
        "board_legacy_405_nearest_m",
        "board_all_licensed_nearest_m",
        "google_minus_board_m",
        "google_raw_candidate_nearest_m",
        "centroid_google_nearest_m",
        "centroid_board_nearest_m",
        "frame_edge_m",
    )
    numeric: dict[str, pd.Series] = {
        column: _numeric(tracts[column], f"tracts.{column}")
        for column in distance_columns
    }
    missing_board_distance = _numeric(
        tracts["missing_board_mailing_zip_zcta_min_m"],
        "tracts.missing_board_mailing_zip_zcta_min_m",
        finite=False,
    )
    for column in set(distance_columns) - {"google_minus_board_m"}:
        _require(bool(numeric[column].ge(0).all()), "invariants", "nonnegative_distance", f"{column} contains negative distances")
    difference = numeric["google_nearest_m"] - numeric["board_nearest_m"]
    _require(bool(np.allclose(difference, numeric["google_minus_board_m"], rtol=0.0, atol=1e-7)), "invariants", "distance_difference", "google_minus_board_m is inconsistent with the source distances")
    _require(
        bool(
            numeric["board_exact_geocode_only_nearest_m"]
            .add(1e-7)
            .ge(numeric["board_nearest_m"])
            .all()
        ),
        "invariants",
        "exact_geocode_subset_distance",
        "Exact-only Board nearest distances cannot be shorter than the primary Board surface",
    )
    for column in (
        "google_nearest_tie_count",
        "google_raw_candidate_nearest_tie_count",
        "board_nearest_tie_count",
        "board_exact_geocode_only_nearest_tie_count",
    ):
        tie_count = _numeric(tracts[column], f"tracts.{column}").astype(int)
        _require(
            bool(tie_count.ge(1).all()),
            "invariants",
            "nearest_tie_count",
            f"{column} must be at least one for every finite nearest distance",
        )
    origin_inside = _boolean(
        tracts["internal_point_within_own_tract"],
        "tracts.internal_point_within_own_tract",
    )
    centroid_inside = _boolean(
        tracts["centroid_within_own_tract"],
        "tracts.centroid_within_own_tract",
    )
    _check_number(
        int((~origin_inside).sum()),
        spatial["internal_point_origins_outside_own_tract"],
        "internal_point_origins_outside_own_tract",
    )
    _check_number(
        int((~centroid_inside).sum()),
        spatial["geometric_centroids_outside_own_tract"],
        "geometric_centroids_outside_own_tract",
    )
    inventory = _mapping(summary["inventory"], "invariants", "inventory")
    for spatial_key, inventory_key in (
        ("board_keyword_screen_supply_rows", "board_uniform_criterion_extended_rows"),
        ("board_keyword_screen_supply_geocoded", "board_uniform_criterion_extended_geocoded_rows"),
        ("board_all_licensed_supply_rows", "board_raw_active_mn_observed_frame_rows"),
        ("board_all_licensed_supply_geocoded", "board_all_active_geocoded_rows"),
        ("google_model_target_exact_location_supply_rows", "google_model_target_exact_address_locations"),
        ("google_raw_candidate_exact_location_supply_rows", "google_raw_exact_address_locations"),
    ):
        _check_number(spatial[spatial_key], inventory[inventory_key], f"spatial_inventory.{spatial_key}")
    _check_number(
        spatial["board_keyword_screen_supply_missing_geocodes"],
        int(spatial["board_keyword_screen_supply_rows"])
        - int(spatial["board_keyword_screen_supply_geocoded"]),
        "board_keyword_screen_supply_missing_geocodes",
    )
    geocode_type_counts = _mapping(
        inventory["board_uniform_geocode_type_counts"],
        "invariants",
        "board_uniform_geocode_type_counts",
    )
    _check_number(
        spatial["board_keyword_screen_supply_exact_geocodes"],
        geocode_type_counts.get("Exact", 0),
        "board_keyword_screen_supply_exact_geocodes",
    )
    _check_number(
        spatial["board_keyword_screen_supply_non_exact_geocodes"],
        geocode_type_counts.get("Non_Exact", 0),
        "board_keyword_screen_supply_non_exact_geocodes",
    )
    _check_number(
        int(spatial["board_keyword_screen_supply_exact_geocodes"])
        + int(spatial["board_keyword_screen_supply_non_exact_geocodes"]),
        spatial["board_keyword_screen_supply_geocoded"],
        "board_geocode_quality_partition",
    )
    _check_number(
        int(spatial["board_keyword_screen_supply_geocoded"])
        + int(spatial["board_keyword_screen_supply_missing_geocodes"]),
        spatial["board_keyword_screen_supply_rows"],
        "board_geocode_completeness_partition",
    )
    coordinate_policy = str(spatial.get("board_primary_coordinate_policy", "")).casefold()
    _require(
        "exact" in coordinate_policy
        and "non_exact" in coordinate_policy
        and "exact-only" in coordinate_policy,
        "invariants",
        "board_coordinate_policy",
        "Board coordinate policy must distinguish primary Exact/Non_Exact use from the Exact-only sensitivity",
        actual=spatial.get("board_primary_coordinate_policy"),
    )
    _check_number(spatial["target_zctas_requested"], len(TARGET_ZIPS), "target_zctas_requested")
    _check_number(spatial["target_zctas_loaded"], len(TARGET_ZIPS), "target_zctas_loaded")
    expected_zctas_without_origin = sorted(set(TARGET_ZIPS) - target_zctas)
    _require(
        sorted(spatial["target_zctas_without_eligible_internal_point_origin"])
        == expected_zctas_without_origin,
        "invariants",
        "target_zctas_without_origin",
        "Declared target ZCTAs without an eligible origin differ from the tract table",
        expected=expected_zctas_without_origin,
        actual=spatial["target_zctas_without_eligible_internal_point_origin"],
    )
    interpretation = str(spatial.get("distance_interpretation", "")).casefold()
    _require(
        "euclidean" in interpretation and "not" in interpretation,
        "invariants",
        "distance_interpretation",
        "Distance interpretation must explicitly identify the Euclidean proxy and reject travel-network interpretation",
        actual=spatial.get("distance_interpretation"),
    )
    conditional_assumptions = _mapping(
        spatial.get("conditional_range_assumptions"),
        "invariants",
        "conditional_range_assumptions",
    )
    assumption_status = str(conditional_assumptions.get("status", "")).casefold()
    _require(
        "conditional" in assumption_status
        and "not" in assumption_status
        and "unconditional logical bound" in assumption_status,
        "invariants",
        "conditional_range_status",
        "Conditional ranges must be explicitly distinguished from unconditional logical bounds",
        actual=conditional_assumptions.get("status"),
    )
    google_assumption = str(conditional_assumptions.get("google", "")).casefold()
    board_assumption = str(conditional_assumptions.get("board", "")).casefold()
    _require(
        "collection frame" in google_assumption and "exhaust" in google_assumption,
        "invariants",
        "google_conditional_assumption",
        "Google conditional-range assumption must state observed-frame exhaustion",
        actual=conditional_assumptions.get("google"),
    )
    _require(
        "mailing zip" in board_assumption
        and "physical location" in board_assumption
        and "not guaranteed" in board_assumption,
        "invariants",
        "board_conditional_assumption",
        "Board conditional-range assumption must state the mailing-ZIP/physical-location limitation",
        actual=conditional_assumptions.get("board"),
    )
    zcta_audit = _mapping(spatial["zcta_geometry_audit"], "invariants", "zcta_geometry_audit")
    for key in ("duplicate_zcta_ids", "null_geometries", "empty_geometries", "invalid_geometries"):
        _check_number(zcta_audit[key], 0, f"zcta_geometry_audit.{key}")
    _require(
        str(zcta_audit["source_crs"]) == "EPSG:4269",
        "invariants",
        "zcta_source_crs",
        "ZCTA geometry audit must identify the source CRS",
        expected="EPSG:4269",
        actual=zcta_audit["source_crs"],
    )
    transformations = _mapping(spatial["coordinate_transformations"], "invariants", "coordinate_transformations")
    for name, source_crs in (
        ("tiger_nad83_to_distance_crs", "EPSG:4269"),
        ("supply_wgs84_to_distance_crs", "EPSG:4326"),
    ):
        transformation = _mapping(transformations[name], "invariants", name)
        _require(str(transformation["source_crs"]) == source_crs, "invariants", "transform_source_crs", f"{name} has the wrong source CRS", expected=source_crs, actual=transformation["source_crs"])
        _require(str(transformation["target_crs"]) == "EPSG:26915", "invariants", "transform_target_crs", f"{name} has the wrong target CRS", expected="EPSG:26915", actual=transformation["target_crs"])
        _require(bool(str(transformation["selected_operation"]).strip()), "invariants", "transform_operation", f"{name} does not record a selected coordinate operation")

    tie_summary = _mapping(spatial["nearest_ties"], "invariants", "nearest_ties")
    google_ties = _numeric(tracts["google_nearest_tie_count"], "google_nearest_tie_count").astype(int)
    board_ties = _numeric(tracts["board_nearest_tie_count"], "board_nearest_tie_count").astype(int)
    board_exact_ties = _numeric(
        tracts["board_exact_geocode_only_nearest_tie_count"],
        "board_exact_geocode_only_nearest_tie_count",
    ).astype(int)
    _check_number(tie_summary["google_origins_with_ties"], int(google_ties.gt(1).sum()), "google_origins_with_ties")
    _check_number(tie_summary["google_max_tie_count"], int(google_ties.max()), "google_max_tie_count")
    _check_number(tie_summary["board_origins_with_ties"], int(board_ties.gt(1).sum()), "board_origins_with_ties")
    _check_number(tie_summary["board_max_tie_count"], int(board_ties.max()), "board_max_tie_count")
    _check_number(tie_summary["board_exact_geocode_only_origins_with_ties"], int(board_exact_ties.gt(1).sum()), "board_exact_geocode_only_origins_with_ties")
    _check_number(tie_summary["board_exact_geocode_only_max_tie_count"], int(board_exact_ties.max()), "board_exact_geocode_only_max_tie_count")

    thresholds = _sequence(
        _nested(spatial, ["thresholds"], "invariants"),
        "invariants",
        "spatial_thresholds",
    )
    for row_payload in thresholds:
        row = _mapping(row_payload, "invariants", "spatial_threshold")
        miles = float(row["threshold_miles"])
        meters = miles * M_PER_MILE
        suffix = str(row["threshold_miles"]).replace(".", "_") + "mi"
        columns = {
            "google": f"google_beyond_{suffix}",
            "google_raw": f"google_raw_candidate_beyond_{suffix}",
            "board": f"board_beyond_{suffix}",
            "board_exact": f"board_exact_geocode_only_beyond_{suffix}",
            "legacy": f"board_legacy_405_beyond_{suffix}",
            "all": f"board_all_licensed_beyond_{suffix}",
            "discordance": f"discordance_type_{suffix}",
            "google_conditionally_determinate": f"google_conditionally_determinate_{suffix}",
            "board_conditionally_determinate": f"board_conditionally_determinate_{suffix}",
        }
        _required_columns(tracts, columns.values(), f"tract threshold {miles}")
        for legacy_column in (
            f"google_determinate_{suffix}",
            f"board_determinate_{suffix}",
        ):
            _require(
                legacy_column not in tracts.columns,
                "invariants",
                "legacy_determinate_schema",
                "Legacy unqualified determinacy columns must not appear in canonical output",
                actual=legacy_column,
            )
        beyond = {
            key: _boolean(tracts[column], column)
            for key, column in columns.items()
            if key != "discordance"
        }
        recomputed = {
            "google": numeric["google_nearest_m"].gt(meters),
            "google_raw": numeric["google_raw_candidate_nearest_m"].gt(meters),
            "board": numeric["board_nearest_m"].gt(meters),
            "board_exact": numeric["board_exact_geocode_only_nearest_m"].gt(meters),
            "legacy": numeric["board_legacy_405_nearest_m"].gt(meters),
            "all": numeric["board_all_licensed_nearest_m"].gt(meters),
        }
        for key, expected in recomputed.items():
            _require(
                beyond[key].equals(expected),
                "invariants",
                "beyond_threshold_classification",
                f"Stored {key} flags differ from Euclidean distance > threshold at {miles} miles",
            )
        _require(
            bool((~beyond["board"] | beyond["board_exact"]).all()),
            "invariants",
            "exact_geocode_beyond_superset",
            "Every tract beyond threshold under the primary Board surface must also be beyond under its Exact-only subset",
        )

        expected_google_determinate = (
            numeric["google_nearest_m"].le(meters)
            | numeric["frame_edge_m"].gt(meters)
        )
        expected_board_determinate = (
            numeric["board_nearest_m"].le(meters)
            | (
                numeric["frame_edge_m"].gt(meters)
                & missing_board_distance.gt(meters)
            )
        )
        _require(beyond["google_conditionally_determinate"].equals(expected_google_determinate), "invariants", "google_conditionally_determinate", f"Google conditional-determinacy flags are inconsistent at {miles} miles")
        _require(beyond["board_conditionally_determinate"].equals(expected_board_determinate), "invariants", "board_conditionally_determinate", f"Board conditional-determinacy flags are inconsistent at {miles} miles")

        _check_number(row["threshold_meters"], round(meters, 3), "threshold_meters", tolerance=0.001)
        _check_number(row["all_fixed_origins"], len(tracts), "threshold_origin_count")
        for key, count_name, pct_name in (
            ("google", "google_beyond_threshold_count", "google_beyond_threshold_pct"),
            ("google_raw", "google_raw_candidate_beyond_threshold_count", "google_raw_candidate_beyond_threshold_pct"),
            ("board", "board_beyond_threshold_count", "board_beyond_threshold_pct"),
            ("board_exact", "board_exact_geocode_only_beyond_threshold_count", "board_exact_geocode_only_beyond_threshold_pct"),
            ("legacy", "legacy_405_beyond_threshold_count", "legacy_405_beyond_threshold_pct"),
            ("all", "board_all_licensed_beyond_threshold_count", "board_all_licensed_beyond_threshold_pct"),
        ):
            _check_number(row[count_name], int(beyond[key].sum()), count_name)
            _check_number(row[pct_name], _rate_percent(beyond[key]), pct_name)
        _check_number(
            row["board_exact_geocode_only_minus_primary_count"],
            int(beyond["board_exact"].sum() - beyond["board"].sum()),
            "board_exact_geocode_only_minus_primary_count",
        )
        _check_number(
            row["google_minus_board_percentage_points"],
            round(_rate_percent(beyond["google"]) - _rate_percent(beyond["board"]), 2),
            "google_minus_board_percentage_points",
        )

        expected_discordance = np.select(
            [
                beyond["board"] & beyond["google"],
                ~beyond["board"] & ~beyond["google"],
                ~beyond["board"] & beyond["google"],
                beyond["board"] & ~beyond["google"],
            ],
            [
                "both_beyond_threshold",
                "both_within_threshold",
                "google_only_beyond_threshold",
                "board_only_beyond_threshold",
            ],
            default="unresolved",
        )
        observed_discordance = tracts[columns["discordance"]].astype(str).to_numpy()
        _require(bool(np.array_equal(expected_discordance, observed_discordance)), "invariants", "discordance_type", f"Discordance labels are inconsistent at {miles} miles")
        _check_confusion(_mapping(row["observed_supply_comparison"], "invariants", "observed_supply_comparison"), _confusion(beyond["board"], beyond["google"]), "observed_supply_comparison")
        _check_confusion(_mapping(row["google_vs_board_exact_geocode_only_comparison"], "invariants", "google_vs_board_exact_geocode_only_comparison"), _confusion(beyond["board_exact"], beyond["google"]), "google_vs_board_exact_geocode_only_comparison")

        both_determinate = beyond["google_conditionally_determinate"] & beyond["board_conditionally_determinate"]
        _check_number(row["both_sources_conditionally_determinate_count"], int(both_determinate.sum()), "both_sources_conditionally_determinate_count")
        _check_confusion(_mapping(row["conditionally_determinate_comparison"], "invariants", "conditionally_determinate_comparison"), _confusion(beyond["board"].loc[both_determinate], beyond["google"].loc[both_determinate]), "conditionally_determinate_comparison")

        interior = numeric["frame_edge_m"].gt(meters)
        interior_summary = _mapping(row["collection_frame_interior_sensitivity"], "invariants", "collection_frame_interior_sensitivity")
        _check_number(interior_summary["tracts"], int(interior.sum()), "interior_tracts")
        _check_number(interior_summary["google_beyond_count"], int(beyond["google"].loc[interior].sum()), "interior_google_count")
        _check_number(interior_summary["board_beyond_count"], int(beyond["board"].loc[interior].sum()), "interior_board_count")
        _check_number(interior_summary["google_beyond_pct"], _rate_percent(beyond["google"].loc[interior]), "interior_google_pct")
        _check_number(interior_summary["board_beyond_pct"], _rate_percent(beyond["board"].loc[interior]), "interior_board_pct")
        _check_confusion(_mapping(interior_summary["comparison"], "invariants", "interior_comparison"), _confusion(beyond["board"].loc[interior], beyond["google"].loc[interior]), "interior_comparison")

        centroid_google = numeric["centroid_google_nearest_m"].gt(meters)
        centroid_board = numeric["centroid_board_nearest_m"].gt(meters)
        centroid_summary = _mapping(row["geometric_centroid_origin_sensitivity"], "invariants", "centroid_sensitivity")
        _check_number(centroid_summary["tracts"], len(tracts), "centroid_tracts")
        _check_number(centroid_summary["google_beyond_count"], int(centroid_google.sum()), "centroid_google_count")
        _check_number(centroid_summary["board_beyond_count"], int(centroid_board.sum()), "centroid_board_count")
        _check_number(centroid_summary["google_beyond_pct"], _rate_percent(centroid_google), "centroid_google_pct")
        _check_number(centroid_summary["board_beyond_pct"], _rate_percent(centroid_board), "centroid_board_pct")
        _check_confusion(_mapping(centroid_summary["comparison"], "invariants", "centroid_comparison"), _confusion(centroid_board, centroid_google), "centroid_comparison")

        for weight_name, weight_column in (("population_weighted", "total_pop"), ("household_weighted", "total_hh")):
            weighted = _mapping(row[weight_name], "invariants", weight_name)
            _check_number(weighted["google_beyond_pct"], _weighted_rate_percent(beyond["google"], tracts[weight_column]), f"{weight_name}.google_beyond_pct")
            _check_number(weighted["board_beyond_pct"], _weighted_rate_percent(beyond["board"], tracts[weight_column]), f"{weight_name}.board_beyond_pct")

        board_definite_within = numeric["board_nearest_m"].le(meters)
        board_definite_beyond = (
            numeric["board_nearest_m"].gt(meters)
            & numeric["frame_edge_m"].gt(meters)
            & missing_board_distance.gt(meters)
        )
        board_uncertain = ~(board_definite_within | board_definite_beyond)
        google_definite_within = numeric["google_nearest_m"].le(meters)
        google_definite_beyond = (
            numeric["google_nearest_m"].gt(meters)
            & numeric["frame_edge_m"].gt(meters)
        )
        google_uncertain = ~(google_definite_within | google_definite_beyond)
        expected_bounds = {
            "board_beyond_rate_conditional_lower_pct": round(100.0 * board_definite_beyond.mean(), 2),
            "board_beyond_rate_conditional_upper_pct": round(100.0 * (board_definite_beyond | board_uncertain).mean(), 2),
            "google_beyond_rate_conditional_lower_pct": round(100.0 * google_definite_beyond.mean(), 2),
            "google_beyond_rate_conditional_upper_pct": round(100.0 * (google_definite_beyond | google_uncertain).mean(), 2),
            "board_conditionally_unresolved_count": int(board_uncertain.sum()),
            "google_conditionally_unresolved_count": int(google_uncertain.sum()),
        }
        for key, value in expected_bounds.items():
            _check_number(row[key], value, key)
        for lower_key, upper_key in (
            ("board_beyond_rate_conditional_lower_pct", "board_beyond_rate_conditional_upper_pct"),
            ("google_beyond_rate_conditional_lower_pct", "google_beyond_rate_conditional_upper_pct"),
        ):
            lower = float(row[lower_key])
            upper = float(row[upper_key])
            _require(0.0 <= lower <= upper <= 100.0, "invariants", "conditional_sensitivity_range", f"Invalid conditional sensitivity range at {miles} miles", actual=[lower, upper])

        for legacy_key in (
            "board_beyond_rate_logical_lower_pct",
            "board_beyond_rate_logical_upper_pct",
            "google_beyond_rate_logical_lower_pct",
            "google_beyond_rate_logical_upper_pct",
            "board_uncertain_count",
            "google_uncertain_count",
            "both_sources_determinate_count",
            "determinate_comparison",
        ):
            _require(
                legacy_key not in row,
                "invariants",
                "legacy_conditional_schema",
                "Legacy unqualified logical/determinate summary fields must not appear",
                actual=legacy_key,
            )

        transform_tolerance = _mapping(row["four_meter_transform_tolerance"], "invariants", "four_meter_transform_tolerance")
        _check_number(transform_tolerance["google_origins_within_4m_of_threshold"], int(numeric["google_nearest_m"].sub(meters).abs().le(4.0).sum()), "google_four_meter_tolerance")
        _check_number(transform_tolerance["board_origins_within_4m_of_threshold"], int(numeric["board_nearest_m"].sub(meters).abs().le(4.0).sum()), "board_four_meter_tolerance")

    distance_summary = _mapping(_nested(spatial, ["distance_difference"], "invariants"), "invariants", "distance_difference")
    _check_number(distance_summary["n"], len(tracts), "distance_summary.n")
    _check_number(distance_summary["google_nearest_median_m"], round(float(numeric["google_nearest_m"].median()), 2), "google_nearest_median_m")
    _check_number(distance_summary["board_nearest_median_m"], round(float(numeric["board_nearest_m"].median()), 2), "board_nearest_median_m")
    _check_number(distance_summary["google_minus_board_mean_m"], round(float(difference.mean()), 2), "google_minus_board_mean_m")
    _check_number(distance_summary["google_minus_board_median_m"], round(float(difference.median()), 2), "google_minus_board_median_m")
    _check_number(distance_summary["google_minus_board_q25_m"], round(float(difference.quantile(0.25)), 2), "google_minus_board_q25_m")
    _check_number(distance_summary["google_minus_board_q75_m"], round(float(difference.quantile(0.75)), 2), "google_minus_board_q75_m")
    _check_number(distance_summary["google_farther_count"], int(difference.gt(0).sum()), "google_farther_count")
    _check_number(distance_summary["equal_distance_count"], int(np.isclose(difference, 0.0).sum()), "equal_distance_count")
    _check_number(distance_summary["google_closer_count"], int(difference.lt(0).sum()), "google_closer_count")

    strata = _sequence(_nested(spatial, ["neighborhood_strata"], "invariants"), "invariants", "neighborhood_strata")
    for row_payload in strata:
        row = _mapping(row_payload, "invariants", "stratum")
        stratifier = str(row["stratifier"])
        quartile = int(row["quartile"])
        _require(stratifier in {"income_quartile", "nonwhite_quartile"}, "invariants", "stratifier", "Unexpected neighborhood stratifier", actual=stratifier)
        subset = tracts.loc[_numeric(tracts[stratifier], stratifier).astype(int).eq(quartile)]
        subset_difference = _numeric(subset["google_minus_board_m"], "stratum.google_minus_board_m")
        _check_number(row["tracts"], len(subset), "stratum.tracts")
        _check_number(row["google_minus_board_median_m"], round(float(subset_difference.median()), 2), "stratum.median")
        _check_number(row["google_minus_board_mean_m"], round(float(subset_difference.mean()), 2), "stratum.mean")
        for threshold_payload in thresholds:
            miles_value = threshold_payload["threshold_miles"]
            suffix = str(miles_value).replace(".", "_") + "mi"
            google_beyond = _boolean(subset[f"google_beyond_{suffix}"], f"stratum.google_beyond_{suffix}")
            board_beyond = _boolean(subset[f"board_beyond_{suffix}"], f"stratum.board_beyond_{suffix}")
            google_pct = _rate_percent(google_beyond)
            board_pct = _rate_percent(board_beyond)
            _check_number(row[f"google_beyond_pct_{suffix}"], google_pct, f"stratum.google_beyond_pct_{suffix}")
            _check_number(row[f"board_beyond_pct_{suffix}"], board_pct, f"stratum.board_beyond_pct_{suffix}")
            _check_number(row[f"google_minus_board_pp_{suffix}"], round(float(google_pct - board_pct), 2), f"stratum.google_minus_board_pp_{suffix}")
    return len(tracts)


def verify_delivery(output_root: Path, *, strict: bool = False) -> dict[str, Any]:
    """Verify hashes first, then cross-check every canonical analytical table."""
    output_root = _resolve_output_root(output_root)
    summary_path = output_root / "results" / "analysis_summary.json"
    summary = _read_json(summary_path, "summary")

    # Trust ordering is part of the contract: code, then inputs, then outputs.
    code_entries = _manifest_entries(summary, "code_sha256")
    _verify_manifest_hashes(code_entries, "code_sha256")
    input_entries = _manifest_entries(summary, "input_sha256")
    _verify_manifest_hashes(input_entries, "input_sha256")
    input_manifest_report = (
        _verify_input_manifest_record(summary, input_entries) if strict else None
    )
    environment_report = _verify_environment_contract(summary) if strict else None
    output_entries = _manifest_entries(summary, "output_sha256")
    if strict:
        _strict_manifest_contract(code_entries, input_entries, output_entries, output_root)
    _verify_manifest_hashes(output_entries, "output_sha256")

    core = {path: output_root / path for path in CORE_RELATIVE_OUTPUTS}
    criterion = _read_csv(core[Path("private/board_keyword_screen_fixed.csv")], dtype={"license_nbr": str, "zip5": str})
    extended = _read_csv(core[Path("private/board_keyword_screen_extended.csv")], dtype={"license_nbr": str, "zip5": str})
    links = _read_csv(core[Path("private/board_google_record_links.csv")], dtype={"ai_row_id": str, "board_license_nbr": str})
    status = _read_csv(core[Path("private/board_google_source_status.csv")], dtype=str)
    groups = _read_csv(core[Path("private/board_google_name_signature_groups.csv")], dtype=str)
    tracts = _read_csv(core[Path("results/board_google_tract_proximity.csv")], dtype={"GEOID": str, "target_zcta": str})
    signature_summary = _read_json(core[Path("results/name_signature_summary.json")], "outputs")

    counts = _verify_inventory_and_linkage(summary, criterion, extended, links, status)
    _verify_signatures(summary, signature_summary, groups, counts)
    spatial_source_audit = _recompute_spatial_source_audit() if strict else None
    counts["tract_rows"] = _verify_spatial(
        summary,
        tracts,
        source_audit=spatial_source_audit,
    )
    return {
        "status": "PASS",
        "strict": strict,
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "verified_sha256": {
            "code_files": len(code_entries),
            "input_files": len(input_entries),
            "output_files": len(output_entries),
        },
        "input_manifest": input_manifest_report,
        "environment": environment_report,
        "counts": counts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the canonical Board--Google analysis delivery without modifying it."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Analysis output directory (default: analysis/delivery).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require complete canonical code/input/output manifest coverage and isolation.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_delivery(args.output_root, strict=args.strict)
    except VerificationError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2, default=str))
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "stage": "verifier",
                    "check": "unexpected_exception",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
