"""Authenticated old-versus-corrected evidence-run comparison (phase 5 only).

This module performs no acquisition, classification, linkage or spatial analysis.
It compares completed saved outputs and preserves both empirical runs unchanged.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from corrected_run_store import canonical, file_sha, read_sealed, sha, unit_key
from forced_choice_eligibility import reduce_forced_choice_eligibility
from forced_choice_models import CORE_FIELDS, LABELS, ForcedChoiceDecision
from run_forced_choice_workflow import _location_from_record
from v2_delivery import _write_csv


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN_ID = "20260902T012636Z"
OLD_ROOT = ROOT / "data/v2" / OLD_RUN_ID
OLD_AUDIT = ROOT / f"analysis/v2/results_audit_{OLD_RUN_ID}_forced_choice_final/results_audit.json"
OLD_STAGE = ROOT / f"analysis/v2/board_linkage_stage_diagnostic_{OLD_RUN_ID}_v5"
EXPECTED_CANDIDATES = 1226
EXPECTED_ORIGINS = 327
SCHEMA = "nets-ai-corrected-impact-v1"
BASELINE_PINNED_HASHES = {
    "data/v2/20260902T012636Z/run_manifest.json": "5f0463e7fe99c942f9a0854bf20a6bd0a741b62d75542df1a41ed347350da9f6",
    "data/v2/20260902T012636Z/delivery/run_manifest.json": "e7138170e740647993afcdae42a02c5171c0f8292db04c0ddae73debeafa1d78",
    "analysis/v2/results_audit_20260902T012636Z_forced_choice_final/results_audit.json": "f73d6cb573a8ce8c3baa2eaaa4a6b24f18b27d4b832eab9b36d4f8c001133a24",
    "analysis/v2/board_linkage_stage_diagnostic_20260902T012636Z_v5/artifact_manifest.json": "45ac8dd52be81130d5b2c5fc945ac5ceb6176cc69257a8ad99130f7274b407a3",
}
_BOUNDARY = (
    "This is a corrected evidence-collection run audit, not a model comparison or "
    "a Wayback causal effect. Place Details and reviews were recollected at a later "
    "timestamp, so differences cannot be attributed solely to Wayback evidence."
)
_STATES = {"both_beyond", "neither_beyond", "google_only_beyond", "board_only_beyond"}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _verify_files(root, hashes, *, exact_directory=None):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("artifact hashes are missing")
    observed = {}
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("artifact path escaped package")
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
        observed[relative] = expected
    if exact_directory is not None:
        files = {path.relative_to(root).as_posix() for path in exact_directory.rglob("*") if path.is_file()}
        if files != set(hashes):
            raise ValueError("completed file set differs from manifest")
    return observed


def _authenticate_package(root, *, status, phase=None):
    manifest = _json(root / "run_manifest.json")
    if (manifest.get("status") != status
            or manifest.get("candidate_count") != EXPECTED_CANDIDATES
            or manifest.get("origin_count") != EXPECTED_ORIGINS
            or manifest.get("runtime_failure_count") != 0
            or manifest.get("technical_unresolved_count", 0) != 0
            or (phase is not None and manifest.get("phase") != phase)):
        raise ValueError("downstream phase manifest is not complete")
    artifacts = manifest.get("artifacts")
    _verify_files(root, artifacts)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(artifacts) | {"run_manifest.json"}:
        raise ValueError("downstream artifact file set differs from manifest")
    semantic = {key: value for key, value in manifest.items() if key not in {"run_metadata", "semantic_manifest_sha256"}}
    if sha(semantic) != manifest.get("semantic_manifest_sha256"):
        raise ValueError("downstream semantic manifest hash mismatch")
    return manifest


def _validated_unit(unit):
    place_id = unit["place_id"]
    location = _location_from_record(unit["location"])
    decision = ForcedChoiceDecision.from_public_dict(unit["decision"])
    if location.place_id != place_id or location.candidate_id != unit["candidate_id"] or decision.candidate_id != unit["candidate_id"]:
        raise ValueError("classification candidate binding mismatch")
    expected = unit.get("decision_sha256", unit.get("classification_decision_sha256"))
    if expected != decision.decision_sha256:
        raise ValueError("classification decision hash mismatch")
    reduced = reduce_forced_choice_eligibility(decision, place_id=place_id, candidate_id=unit["candidate_id"])
    if "eligibility" in unit and unit["eligibility"] != reduced.to_public_dict():
        raise ValueError("saved eligibility differs from unchanged reducer")
    return {
        "place_id": place_id, "candidate_id": unit["candidate_id"],
        "decision": decision, "eligibility": reduced,
        "lineage": {"place_id": place_id, "candidate_id": unit["candidate_id"], "candidate_snapshot_sha256": location.candidate_snapshot_sha256, "classification_decision_sha256": decision.decision_sha256},
        "retrieved_at": unit["location"]["retrieved_at"],
    }


def _load_baseline():
    for relative, digest in BASELINE_PINNED_HASHES.items():
        if file_sha(ROOT / relative) != digest:
            raise ValueError(f"preserved baseline changed: {relative}")
    manifest = _json(OLD_ROOT / "run_manifest.json")
    if manifest.get("status") != "full_complete" or manifest.get("run_id") != OLD_RUN_ID:
        raise ValueError("September 2 run is not the completed baseline")
    completed = manifest["completed_unit_hashes"]
    directory = OLD_ROOT / "private/completed_candidates"
    _verify_files(directory, completed, exact_directory=directory)
    units = [_validated_unit(_json(directory / name)) for name in sorted(completed)]
    delivered = _authenticate_package(OLD_ROOT / "delivery", status="full_delivery_complete")
    if file_sha(OLD_ROOT / "delivery/run_manifest.json") != manifest["delivery_manifest_sha256"]:
        raise ValueError("old delivery manifest differs from run commitment")
    baseline = _json(OLD_AUDIT)
    if baseline["run_id"] != OLD_RUN_ID or baseline["source_artifacts"] != delivered["artifacts"] or baseline["delivery_semantic_manifest_sha256"] != delivered["semantic_manifest_sha256"]:
        raise ValueError("old Results audit is not bound to preserved delivery")
    stage_manifest = _json(OLD_STAGE / "artifact_manifest.json")
    _verify_files(OLD_STAGE, stage_manifest["artifacts"])
    stage = _json(OLD_STAGE / "board_linkage_stage_diagnostic.json")
    if stage["run_id"] != OLD_RUN_ID or stage["provenance"]["input_hashes"]["delivery_manifest"] != file_sha(OLD_ROOT / "delivery/run_manifest.json") or not all(stage["verification_checks"].values()):
        raise ValueError("old stage diagnostic does not match preserved delivery")
    return units, delivered, baseline, stage


def _load_new(root, phase3, phase4):
    classification = read_sealed(root / "classification_manifest.json")
    if (classification.get("passed") is not True or classification.get("candidate_count") != EXPECTED_CANDIDATES
            or classification.get("statuses") != {"completed": EXPECTED_CANDIDATES}
            or classification.get("technical_unresolved_place_ids") != []
            or len(classification.get("returned_models", [])) != 1):
        raise ValueError("complete corrected classification gate has not passed")
    scope = read_sealed(root / "run_scope.json")
    if scope["run_id"] == OLD_RUN_ID or scope["source_run_id"] != OLD_RUN_ID or classification.get("run_id") != scope["run_id"]:
        raise ValueError("corrected scope is not a new September 2 successor")
    evidence = read_sealed(root / "evidence_manifest.json")
    preflight_path = root / "acquisition_preflight.json"
    preflight = _json(preflight_path)
    if evidence.get("run_id") != scope["run_id"] or evidence["candidate_count"] != EXPECTED_CANDIDATES or file_sha(preflight_path) != evidence["preflight_sha256"] or any(preflight.get(key) is not True for key in ("passed", "board_absent_from_inputs", "full_packet_reconstruction_passed", "frame_equality_passed")):
        raise ValueError("corrected acquisition gate is not authenticated")
    for key in ("packet_files", "details_files"):
        if len(evidence[key]) != EXPECTED_CANDIDATES:
            raise ValueError("corrected evidence frame is incomplete")
        _verify_files(root, evidence[key])
    packets = {}
    for relative in evidence["packet_files"]:
        packet = read_sealed(root / relative)
        if packet["place_id"] in packets:
            raise ValueError("duplicate candidate in corrected evidence packets")
        packets[packet["place_id"]] = packet
    _verify_files(root, classification["completed_files"], exact_directory=root / "private/classifications")
    units = []
    for relative in sorted(classification["completed_files"]):
        unit = read_sealed(root / relative)
        if relative != f"private/classifications/{unit_key(unit['place_id'])}.json" or unit.get("status") != "completed" or unit["returned_model"] != classification["returned_models"][0]:
            raise ValueError("corrected completed unit binding/status/model mismatch")
        packet = packets.get(unit["place_id"], {})
        if any(unit.get(key) != packet.get(key) or key not in packet for key in ("candidate_id", "packet_sha256", "messages_sha256")):
            raise ValueError("corrected classification differs from its frozen evidence packet")
        units.append(_validated_unit(unit))
    three = _authenticate_package(phase3, status="linkage_phase_complete", phase=3)
    four = _authenticate_package(phase4, status="spatial_phase_complete", phase=4)
    lineage = sha([unit["lineage"] for unit in sorted(units, key=lambda item: item["place_id"])])
    for manifest in (three, four):
        metadata = manifest["run_metadata"]
        if (metadata["run_id"] != scope["run_id"] or metadata["classification_manifest_sha256"] != file_sha(root / "classification_manifest.json")
                or metadata["evidence_manifest_sha256"] != file_sha(root / "evidence_manifest.json")
                or manifest["candidate_lineage_sha256"] != lineage):
            raise ValueError("downstream phase belongs to different corrected evidence/classifications")
    if three["contracts"] != four["contracts"] or three["source_hashes"] != {**four["source_hashes"], **{key: value for key, value in three["source_hashes"].items() if key == "data/Minneapolis-StPaul_pharmacy_20260413_201606.csv"}}:
        raise ValueError("corrected downstream phase contracts/source hashes disagree")
    return units, scope, classification, evidence, preflight, three, four


def _count_distribution(values):
    counts = Counter(values)
    total = sum(counts.values())
    return {key: {"count": count, "pct": round(100 * count / total, 2) if total else None} for key, count in sorted(counts.items())}


def _rq1(units):
    result = {
        "candidate_count": len(units),
        "eligibility": _count_distribution(unit["eligibility"].eligibility_state for unit in units),
        "core_judgments": {field: _count_distribution(getattr(unit["decision"], field) for unit in units) for field in CORE_FIELDS},
        "confidence": {field: _count_distribution(unit["decision"].claim_confidence[field] for unit in units) for field in CORE_FIELDS},
        "exclusion_combinations": _count_distribution(" + ".join(unit["eligibility"].reason_codes) for unit in units if unit["eligibility"].eligibility_state == "exclude"),
    }
    for value in ("include", "exclude", "unresolved"):
        result["eligibility"].setdefault(value, {"count": 0, "pct": 0.0})
    for field in CORE_FIELDS:
        for label in LABELS[field]:
            result["core_judgments"][field].setdefault(label, {"count": 0, "pct": 0.0})
        for label in ("High", "Moderate", "Low"):
            result["confidence"][field].setdefault(label, {"count": 0, "pct": 0.0})
    return result


def compare_candidates(old, new):
    """Exact-ID comparison without merging or substituting classifications."""
    left = {row["place_id"]: row for row in old}
    right = {row["place_id"]: row for row in new}
    if len(old) != EXPECTED_CANDIDATES or len(new) != EXPECTED_CANDIDATES or len(left) != len(old) or len(right) != len(new) or set(left) != set(right):
        raise ValueError("old/corrected comparison requires the exact 1,226 unique Place IDs")
    rows = []
    for place_id in sorted(left):
        before, after = left[place_id], right[place_id]
        if before["candidate_id"] != after["candidate_id"]:
            raise ValueError("candidate identity changed between evidence runs")
        row = {"place_id": place_id, "candidate_id": before["candidate_id"], "old_details_retrieved_at": before["retrieved_at"], "corrected_details_retrieved_at": after["retrieved_at"]}
        for field in CORE_FIELDS:
            row[f"old_{field}"] = getattr(before["decision"], field)
            row[f"corrected_{field}"] = getattr(after["decision"], field)
            row[f"{field}_changed"] = row[f"old_{field}"] != row[f"corrected_{field}"]
            row[f"old_{field}_confidence"] = before["decision"].claim_confidence[field]
            row[f"corrected_{field}_confidence"] = after["decision"].claim_confidence[field]
        row["old_eligibility"] = before["eligibility"].eligibility_state
        row["corrected_eligibility"] = after["eligibility"].eligibility_state
        row["eligibility_changed"] = row["old_eligibility"] != row["corrected_eligibility"]
        row["any_core_judgment_changed"] = any(row[f"{field}_changed"] for field in CORE_FIELDS)
        rows.append(row)
    frame = pd.DataFrame(rows)
    changes = {field: int(frame[f"{field}_changed"].sum()) for field in (*CORE_FIELDS, "eligibility")}
    changes["any_core_judgment"] = int(frame["any_core_judgment_changed"].sum())
    return {"old": _rq1(old), "corrected": _rq1(new), "changed_candidates": changes}, frame


def compare_tracts(old, new):
    old = old.sort_values("GEOID").reset_index(drop=True)
    new = new.sort_values("GEOID").reset_index(drop=True)
    if len(old) != EXPECTED_ORIGINS or len(new) != EXPECTED_ORIGINS or old["GEOID"].duplicated().any() or new["GEOID"].duplicated().any() or old["GEOID"].tolist() != new["GEOID"].tolist():
        raise ValueError("paired impact requires the same 327 unique tract origins")
    for column in ("income_quartile", "nonwhite_quartile", "total_pop", "total_hh"):
        if not np.array_equal(old[column].to_numpy(), new[column].to_numpy()):
            raise ValueError("fixed neighborhood fields changed between runs")
    result = pd.DataFrame({"GEOID": old["GEOID"]})
    for label, column in (("workflow", "google_include_nearest_m"), ("board", "board_screened_nearest_m")):
        before, after = old[column].to_numpy(float) / 1609.344, new[column].to_numpy(float) / 1609.344
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("paired distance impact requires finite distances")
        result[f"old_{label}_nearest_miles"] = before
        result[f"corrected_{label}_nearest_miles"] = after
        result[f"{label}_nearest_miles_change"] = after - before
    result["old_signed_workflow_minus_board_miles"] = result["old_workflow_nearest_miles"] - result["old_board_nearest_miles"]
    result["corrected_signed_workflow_minus_board_miles"] = result["corrected_workflow_nearest_miles"] - result["corrected_board_nearest_miles"]
    result["signed_workflow_minus_board_miles_change"] = result["corrected_signed_workflow_minus_board_miles"] - result["old_signed_workflow_minus_board_miles"]
    transitions = {}
    for threshold, suffix in (("0.5", "0_5"), ("1.0", "1_0"), ("2.0", "2_0")):
        column = f"discordance_type_{suffix}mi"
        if not set(old[column]) <= _STATES or not set(new[column]) <= _STATES:
            raise ValueError("invalid paired tract state")
        result[f"old_state_{suffix}mi"] = old[column]
        result[f"corrected_state_{suffix}mi"] = new[column]
        changed = old[column].ne(new[column])
        result[f"state_changed_{suffix}mi"] = changed
        transitions[threshold] = {"state_changed_tracts_between_runs": int(changed.sum()), "transition_counts": dict(sorted(Counter(f"{a} -> {b}" for a, b in zip(old[column], new[column])).items()))}
    return transitions, result


def _numeric_comparison(old, new):
    return {"old": old, "corrected": new, "change": round(new - old, 8) if isinstance(old, (float, int)) and isinstance(new, (float, int)) else None, "changed": old != new}


def _flatten_numbers(value, prefix=""):
    rows = {}
    if isinstance(value, dict):
        for key, item in value.items():
            rows.update(_flatten_numbers(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows[prefix] = value
    return rows


def _paired_numeric_tree(old, new):
    left, right = _flatten_numbers(old), _flatten_numbers(new)
    return {key: _numeric_comparison(left.get(key, 0 if key.startswith("rq1.exclusion_combinations.") else None), right.get(key, 0 if key.startswith("rq1.exclusion_combinations.") else None)) for key in sorted(set(left) | set(right))}


def _same_distribution(left, right):
    return all(left.get(key, {"count": 0, "pct": 0.0}) == right.get(key, {"count": 0, "pct": 0.0}) for key in set(left) | set(right))


def _reconcile_spatial_rows(summary, frame):
    """Check saved report values against its saved tract rows without new GIS."""
    for key, column in (("google_include_nearest_miles", "google_include_nearest_m"), ("board_screened_nearest_miles", "board_screened_nearest_m")):
        values = frame[column].to_numpy(float) / 1609.344
        if not np.isfinite(values).all() or summary[key]["n"] != len(values):
            raise ValueError("spatial summary has a different finite origin count")
        if summary[key]["mean"] != round(float(np.mean(values)), 3) or summary[key]["median"] != round(float(np.median(values)), 3):
            raise ValueError("spatial summary mean/median does not match saved tract distances")
    for threshold, suffix in (("0.5", "0_5"), ("1.0", "1_0"), ("2.0", "2_0")):
        counts = Counter(frame[f"discordance_type_{suffix}mi"])
        reported = summary["thresholds"][threshold]["scenarios"]["primary_google_include_vs_board_screened"]
        if any(reported["states"][key]["count"] != counts[key] for key in _STATES) or reported["classification_changed"]["count"] != counts["google_only_beyond"] + counts["board_only_beyond"]:
            raise ValueError("spatial summary paired states do not match saved tract rows")


def _primary(metrics):
    rows = [row for row in metrics["strata"] if row["stratum"] == "include_screened_primary"]
    if len(rows) != 1:
        raise ValueError("primary linkage stratum missing or repeated")
    return {key: rows[0][key] for key in ("google_rows", "board_rows", "linked_pairs", "google_link_agreement_pct", "board_recovery_pct", "unmatched_google_rows", "unmatched_board_rows")}


def _recurrence(summary):
    return {f"{row['source']}|{row['recurrence_threshold']}|{row['group_code']}": {key: row[key] for key in ("records", "distinct_name_components", "linked_records", "linked_pct", "cluster_bootstrap_95_low_pct", "cluster_bootstrap_95_high_pct")} for row in summary["rows"] if row["linkage_strategy"] == summary["primary_strategy"]}


def _old_stage_metrics(stage):
    section = stage["place_id_analysis"]
    core = dict(section["core_stage_specific_measures"])
    identities = section["board_assignment_identity_comparison"]
    denominator = identities["linked in both assignments"] + identities["linked only before LLM/reducer filtering"]
    for name, numerator in (("board_referenced_assignment_retention", identities["linked in both assignments"]), ("board_referenced_assignment_attrition", identities["linked only before LLM/reducer filtering"])):
        core[name] = {"numerator": numerator, "denominator": denominator, "pct": round(100 * numerator / denominator, 2) if denominator else None}
    return {"core_metrics": core, "raw_assignment": section["raw_vs_included_hungarian"][0], "included_assignment": section["raw_vs_included_hungarian"][1], "board_stages": {row["stage"]: row["count"] for row in section["board_four_stage_decomposition"]}}


def _new_stage_metrics(stage):
    section = stage["place_id_diagnostic"]
    return {"core_metrics": section["core_metrics"], "raw_assignment": section["assignments"][0], "included_assignment": section["assignments"][1], "board_stages": {row["stage"]: row["count"] for row in section["board_stages"]}}


def _spatial_metrics(summary):
    return {
        "workflow_nearest_miles": summary["google_include_nearest_miles"],
        "board_nearest_miles": summary["board_screened_nearest_miles"],
        "signed_workflow_minus_board_miles": summary["signed_google_minus_board_miles"],
        "thresholds": {threshold: {"classification_changed": row["scenarios"]["primary_google_include_vs_board_screened"]["classification_changed"], "states": row["scenarios"]["primary_google_include_vs_board_screened"]["states"]} for threshold, row in summary["thresholds"].items()},
        "income_quartiles": summary["income_quartiles"], "nonwhite_quartiles": summary["nonwhite_quartiles"],
    }


def _manuscript_targets(comparisons):
    paths = [ROOT / "docs/thesis/chapters/03_methodology.tex", ROOT / "docs/thesis/chapters/04_results.tex", *sorted((ROOT / "docs/thesis/tables").glob("table*.tex"))]
    documents, hashes = [], {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        hashes[relative] = file_sha(path)
        section = "Methods" if "03_methodology" in path.name else "Results"
        label_match = re.search(r"\\label\{([^}]+)\}", text)
        label = label_match[1] if label_match else None
        for number, line in enumerate(text.splitlines(), 1):
            match = re.search(r"\\section\{([^}]+)\}", line)
            if match:
                section = match[1]
            documents.append({"path": relative, "line": number, "section": section, "label": label, "text": line})
    revisions = []
    for metric, comparison in comparisons.items():
        if not comparison["changed"] or comparison["old"] is None:
            continue
        old = comparison["old"]
        tokens = {str(old), f"{old:,.0f}" if float(old).is_integer() else f"{old:.2f}", f"{old:.3f}"}
        occurrence = []
        for row in documents:
            group = metric.split(".", 1)[0]
            relevant = (group == "rq1" and ("RQ1" in row["section"] or "table1" in row["path"])) or (group == "rq2" and ("RQ2" in row["section"] or "table2" in row["path"] or "table3" in row["path"] or "03_methodology" in row["path"])) or (group == "rq3" and ("RQ3" in row["section"] or "table4" in row["path"] or "table5" in row["path"]))
            if relevant and any(re.search(r"(?<![\d.,])" + re.escape(token) + r"(?![\d.,])", row["text"]) for token in tokens):
                occurrence.append(row)
        revisions.append({"metric": metric, **comparison, "matching_current_manuscript_occurrences": occurrence, "replacement_rule": "Review the metric and sentence/table row together; repeated numeric tokens are not automatic replacement instructions."})
    execution = [row for row in documents if ("03_methodology" in row["path"] or "04_results" in row["path"]) and re.search(r"pilot|smoke test|Wayback|20260902|September 2|gpt-5-mini|hashes|transient", row["text"], re.I)]
    return revisions, execution, hashes


def _markdown(summary):
    lines = [f"# Corrected evidence-collection run impact audit: {summary['corrected_run_id']}", "", _BOUNDARY, "", f"Compared {summary['candidate_count']} exact Place IDs and {summary['origin_count']} unchanged tract origins with preserved September 2 run `{OLD_RUN_ID}`.", "", "| Quantity | September 2 | Corrected | Change |", "|---|---:|---:|---:|"]
    wanted = ["rq1.eligibility.include.count", "rq1.eligibility.exclude.count", "rq2.primary.google_rows", "rq2.primary.linked_pairs", "rq2.primary.google_link_agreement_pct", "rq2.primary.board_recovery_pct", "rq2.stage.core_metrics.board_referenced_assignment_retention.pct", "rq3.workflow_nearest_miles.mean", "rq3.workflow_nearest_miles.median", "rq3.board_nearest_miles.mean", "rq3.board_nearest_miles.median", "rq3.signed_workflow_minus_board_miles.mean", "rq3.signed_workflow_minus_board_miles.median", *(f"rq3.thresholds.{threshold}.classification_changed.count" for threshold in ("0.5", "1.0", "2.0"))]
    wanted[2:2] = [f"rq1.core_judgments.{field}.{label}.count" for field, label in (("target_construct_status", "target_pharmacy"), ("business_status", "active"), ("candidate_plausibility", "plausible"), ("identity_consistency", "consistent"))]
    for key in wanted:
        row = summary["metric_comparisons"][key]
        lines.append(f"| {key} | {row['old']} | {row['corrected']} | {row['change']} |")
    lines += ["", "Candidate-level changed decisions: " + "; ".join(f"{key}: {value}" for key, value in summary["rq1"]["changed_candidates"].items()) + ".", "", "Between-run tract state changes: " + "; ".join(f"{key} miles: {row['state_changed_tracts_between_runs']}" for key, row in summary["tract_state_changes"].items()) + ".", "", "Exact old/new values for core judgments, confidence, exclusion combinations, primary name-recurrence groups and intervals, Board-stage retention/attrition, neighborhood quartiles and signed-distance distribution are in `summary.json`. Candidate and tract transition rows are in the two CSVs.", "", "## Thesis revision ledger", "", "The following changed quantities identify old numbers requiring revision. The JSON ledger includes current chapter/table paths, labels, line numbers and matching source text; matching numeric tokens require contextual review.", ""]
    targets = {}
    for item in summary["thesis_revision_ledger"]:
        for row in item["matching_current_manuscript_occurrences"]:
            targets.setdefault((row["path"], row["label"]), set()).add(row["line"])
    for (path, label), numbers in sorted(targets.items()):
        lines.append(f"- `{path}`" + (f" (`{label}`)" if label else "") + ": affected numeric occurrences at lines " + ", ".join(map(str, sorted(numbers))) + ".")
    lines += ["", f"`summary.json` records {len(summary['thesis_revision_ledger'])} changed numerical quantities with exact old/new values, including name-recurrence intervals; the ledger distinguishes these from unchanged design constants."]
    lines += ["", "The previous empirical run and all baseline artifacts remain preserved. This package does not edit thesis prose or designate a canonical run.", ""]
    return "\n".join(lines)


def compare_runs(new_run_root, phase3_root, phase4_root, *, output_root):
    """Authenticate completed phases, compare saved outputs, publish phase 5 once."""
    root, phase3, phase4, target = map(lambda path: Path(path).resolve(), (new_run_root, phase3_root, phase4_root, output_root))
    if not target.is_relative_to(ROOT) or OLD_RUN_ID in str(target):
        raise ValueError("impact output must be a new project directory outside old artifacts")
    if target.exists():
        raise FileExistsError("impact output already exists")
    new, scope, classification, evidence, preflight, three, four = _load_new(root, phase3, phase4)
    old, old_delivery, old_audit, old_stage = _load_baseline()
    if scope["source_manifest_sha256"] != file_sha(OLD_ROOT / "run_manifest.json"):
        raise ValueError("corrected run does not authenticate the September 2 baseline")
    if scope["frame"]["fixed_frame_sha256"] != old_audit["fixed_frame_sha256"] or evidence["fixed_frame_sha256"] != old_audit["fixed_frame_sha256"]:
        raise ValueError("fixed candidate-frame commitment changed")
    old_spatial = _json(OLD_ROOT / "delivery/spatial_summary.json")
    if four["origin_sha256"] != old_spatial["origin_sha256"]:
        raise ValueError("fixed tract origins differ from September 2")
    for key in ("classifier_contract_sha256", "eligibility_contract_sha256", "spatial_contract_sha256"):
        if three["contracts"][key] != old_delivery["contracts"][key]:
            raise ValueError("a declared scientific contract changed between runs")
    rq1, candidate_frame = compare_candidates(old, new)
    delivered_rq1 = _json(phase3 / "rq1_summary.json")
    if (not _same_distribution(rq1["corrected"]["eligibility"], delivered_rq1["eligibility"])
            or any(not _same_distribution(rq1["corrected"]["core_judgments"][field], delivered_rq1["core_judgment_distributions"][field]) for field in CORE_FIELDS)
            or any(not _same_distribution(rq1["corrected"]["confidence"][field], delivered_rq1["core_confidence_distributions"][field]) for field in CORE_FIELDS)):
        raise ValueError("corrected RQ1 summary differs from saved decisions")
    baseline_rq1 = old_audit["rq1_candidate_flow"]
    if not _same_distribution(rq1["old"]["eligibility"], baseline_rq1["eligibility"]) or any(not _same_distribution(rq1["old"]["core_judgments"][field], baseline_rq1["core_judgment_distributions"][field]) for field in CORE_FIELDS):
        raise ValueError("old RQ1 audit differs from preserved decisions")
    old_primary = _primary(_json(OLD_ROOT / "delivery/performance_metrics.json"))
    new_primary = _primary(_json(phase3 / "performance_metrics.json"))
    old_name = _recurrence(_json(OLD_ROOT / "delivery/name_recurrence_summary.json"))
    new_name = _recurrence(_json(phase3 / "name_recurrence_summary.json"))
    old_stage_values, new_stage_values = _old_stage_metrics(old_stage), _new_stage_metrics(_json(phase3 / "stage_specific_diagnostic.json"))
    new_spatial_report = _json(phase4 / "rq3_summary.json")
    old_rq3, new_rq3 = _spatial_metrics(old_audit["rq3_spatial"]), _spatial_metrics(new_spatial_report)
    old_tracts = pd.read_csv(OLD_ROOT / "delivery/tract_spatial.csv", dtype={"GEOID": str})
    new_tracts = pd.read_csv(phase4 / "tract_spatial.csv", dtype={"GEOID": str})
    _reconcile_spatial_rows(old_audit["rq3_spatial"], old_tracts)
    _reconcile_spatial_rows(new_spatial_report, new_tracts)
    transitions, tract_frame = compare_tracts(old_tracts, new_tracts)
    before = {"rq1": rq1["old"], "rq2": {"primary": old_primary, "stage": old_stage_values, "name_recurrence": old_name}, "rq3": old_rq3}
    after = {"rq1": rq1["corrected"], "rq2": {"primary": new_primary, "stage": new_stage_values, "name_recurrence": new_name}, "rq3": new_rq3}
    comparisons = _paired_numeric_tree(before, after)
    revisions, execution, manuscript_hashes = _manuscript_targets(comparisons)
    source_paths = [root / name for name in ("run_scope.json", "classification_manifest.json", "evidence_manifest.json", "acquisition_preflight.json")] + [phase3 / "run_manifest.json", phase4 / "run_manifest.json", OLD_AUDIT, OLD_STAGE / "artifact_manifest.json", OLD_STAGE / "board_linkage_stage_diagnostic.json", Path(__file__).resolve()]
    sources = {path.relative_to(ROOT).as_posix(): file_sha(path) for path in source_paths}
    for directory, manifest in ((OLD_ROOT / "delivery", old_delivery), (phase3, three), (phase4, four)):
        sources.update({(directory / relative).relative_to(ROOT).as_posix(): digest for relative, digest in manifest["artifacts"].items()})
    sources.update({(root / relative).relative_to(ROOT).as_posix(): digest for relative, digest in classification["completed_files"].items()})
    for name in ("packet_files", "details_files"):
        sources.update({(root / relative).relative_to(ROOT).as_posix(): digest for relative, digest in evidence[name].items()})
    old_completed = _json(OLD_ROOT / "run_manifest.json")["completed_unit_hashes"]
    sources.update({(OLD_ROOT / "private/completed_candidates" / relative).relative_to(ROOT).as_posix(): digest for relative, digest in old_completed.items()})
    sources.update(BASELINE_PINNED_HASHES)
    sources.update(manuscript_hashes)
    summary = {"schema_version": SCHEMA, "status": "impact_audit_complete", "old_run_id": OLD_RUN_ID, "corrected_run_id": scope["run_id"], "candidate_count": EXPECTED_CANDIDATES, "origin_count": EXPECTED_ORIGINS, "interpretation": _BOUNDARY, "rq1": rq1, "rq2": {"old": before["rq2"], "corrected": after["rq2"]}, "rq3": {"old": old_rq3, "corrected": new_rq3}, "tract_state_changes": transitions, "metric_comparisons": comparisons, "thesis_revision_ledger": revisions, "execution_description_review_targets": execution, "new_acquisition_preflight": preflight, "returned_models": classification["returned_models"], "source_hashes": dict(sorted(sources.items())), "old_run_preserved": True, "checks": {"exact_candidate_ids": True, "same_origin_hash": True, "phase3_artifacts_authenticated": True, "phase4_artifacts_authenticated": True, "classification_gate_passed": True, "old_baseline_authenticated": True, "rq1_reconciles": True}}
    markdown = _markdown(summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corrected-impact-", dir=target.parent) as directory:
        staged = Path(directory) / "package"
        staged.mkdir()
        (staged / "summary.json").write_bytes(canonical(summary) + b"\n")
        _write_csv(staged / "candidate_changes.csv", candidate_frame)
        _write_csv(staged / "tract_changes.csv", tract_frame)
        (staged / "impact_audit.md").write_text(markdown, encoding="utf-8", newline="\n")
        # Recheck every source used in the summary before publishing it atomically.
        for relative, digest in sources.items():
            if file_sha(ROOT / relative) != digest:
                raise ValueError("impact source changed during comparison")
        manifest = {"schema_version": SCHEMA, "phase": 5, "status": "impact_audit_complete", "old_run_id": OLD_RUN_ID, "run_id": scope["run_id"], "candidate_count": EXPECTED_CANDIDATES, "origin_count": EXPECTED_ORIGINS, "source_hashes": summary["source_hashes"], "artifacts": {path.name: file_sha(path) for path in sorted(staged.iterdir())}}
        manifest["semantic_manifest_sha256"] = sha(manifest)
        (staged / "run_manifest.json").write_bytes(canonical(manifest) + b"\n")
        if target.exists():
            raise FileExistsError("impact output appeared during comparison")
        os.replace(staged, target)
    return manifest


__all__ = ["SCHEMA", "BASELINE_PINNED_HASHES", "compare_runs", "compare_candidates", "compare_tracts"]
