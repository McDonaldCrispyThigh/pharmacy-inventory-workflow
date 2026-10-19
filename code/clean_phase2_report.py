"""Offline RQ1 and old-versus-corrected classification audit; no RQ2/RQ3."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import io
import json

from wayback_snapshot_fetcher import digest, encoded, read_json, write, write_json
from clean_phase2_evidence import MODEL, CORE_FIELDS, CLAIM_FIELDS, validate_decision
from forced_choice_models import ForcedChoiceDecision, LABELS
from forced_choice_eligibility import reduce_forced_choice_eligibility
from run_clean_phase2 import verify_inputs, verify_completed, inventory
from clean_archive_preflight_postfetch import verify_files, verify_preservation


def save(root, name, value):
    write_json(root / name, value, immutable=True)


def csv_table(root, name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    write(root / name, stream.getvalue().encode(), immutable=True)


def distribution(values, levels=()):
    counts = Counter(values)
    denominator = sum(counts.values())
    return {key: {'count': counts[key], 'percent': 100 * counts[key] / denominator if denominator else None}
            for key in sorted(set(counts) | set(levels))}


def rq1(results, inputs):
    valid = [r for r in results if r['status'] == 'completed']
    counts = Counter(r['reducer']['eligibility_state'] for r in results)
    table = {label: {state: sum(r['decision']['target_construct_status'] == label and
                                r['reducer']['eligibility_state'] == state for r in valid)
                     for state in ('include', 'exclude')} for label in ('target_pharmacy', 'non_target')}
    perfect = (len(valid) == len(results) and table['target_pharmacy']['exclude'] == 0
               and table['non_target']['include'] == 0)
    coverage = {}
    for status in ('include', 'exclude', 'unresolved'):
        records = [inputs[r['place_id']] for r in results if r['reducer']['eligibility_state'] == status]
        websites = sum(r['archive']['usable_website'] for r in records)
        text = sum(bool(r['archive']['verified_archived_sources']) for r in records)
        business = sum(any(any(s['identity_matches'].values()) or bool(s['service_evidence'] or s['contrary_evidence'] or s['opening_claims'])
                           for s in r['archive']['verified_archived_sources']) for r in records)
        coverage[status] = {'candidates': len(records), 'with_usable_website': websites,
            'with_archive_text': text, 'archive_text_percent_of_group': 100*text/len(records) if records else None,
            'archive_text_percent_of_website_group': 100*text/websites if websites else None,
            'with_business_evidence': business, 'business_evidence_percent_of_group': 100*business/len(records) if records else None,
            'archive_states': dict(Counter(r['archive']['state'] for r in records))}
    return {'candidate_count': len(results), 'valid_classifications': len(valid),
        'technical_unresolved_count': len(results)-len(valid),
        'eligibility': distribution((r['reducer']['eligibility_state'] for r in results), ('include', 'exclude', 'unresolved')),
        'four_judgments': {field: distribution((r['decision'][field] for r in valid), LABELS[field]) for field in CORE_FIELDS},
        'confidence': {field: distribution((r['decision']['claim_confidence'][field] for r in valid),
                            ('High','Moderate','Low','NotAssessable') if field=='historical_year' else ('High','Moderate','Low'))
                       for field in CLAIM_FIELDS},
        'exclusion_reason_combinations': distribution(' + '.join(r['reducer']['reason_codes']) for r in valid
                                                    if r['reducer']['eligibility_state']=='exclude'),
        'target_status_by_eligibility': table, 'target_status_perfectly_partitions_eligibility': perfect,
        'target_partition_assessment': 'All four original judgments still determine eligibility; this table tests observed agreement rather than assuming target status alone suffices.',
        'archive_coverage_by_eligibility': coverage,
        'historical_year_types': distribution(r['decision']['historical_year_type'] for r in valid)}


def compare(old, results):
    new = {r['place_id']: r for r in results}
    if len(new) != len(results) or set(old) != set(new):
        raise ValueError('old and corrected candidate identities differ')
    changes = []
    for cid in sorted(new):
        after = new[cid]
        before = old[cid]
        row = {'place_id': cid, 'candidate_id': after['candidate_id'],
               'corrected_status': after['status'], 'old_eligibility': before['reducer']['eligibility_state'],
               'corrected_eligibility': after['reducer']['eligibility_state']}
        for field in CORE_FIELDS:
            row['old_' + field] = before['decision'][field]
            row['corrected_' + field] = after['decision'][field] if after['decision'] else None
            row[field + '_changed'] = row['old_' + field] != row['corrected_' + field] if after['decision'] else None
        row['final_eligibility_changed'] = row['old_eligibility'] != row['corrected_eligibility']
        changes.append(row)
    summary = {'wording': 'Difference between the September 2 run and the corrected evidence-collection run.',
        'not_a_causal_wayback_effect': True,
        'comparison_scope': 'Classification provenance only. Corrected inputs also contain reacquired September 5 Google metadata/reviews and complete returned text; changed labels cannot isolate a Wayback effect.',
        'old_run_id': '20260902T012636Z', 'candidate_count': len(changes),
        'comparable_valid_classifications': sum(r['decision'] is not None for r in results),
        'change_counts': {field: sum(row[field + '_changed'] is True for row in changes)
                          for field in (*CORE_FIELDS, 'final_eligibility')},
        'eligibility_transition_counts': dict(Counter(row['old_eligibility']+' -> '+row['corrected_eligibility'] for row in changes))}
    return summary, changes


def load_old(root):
    """Called only after the new complete classification layer is frozen."""
    manifest = read_json(root / 'run_manifest.json')
    if manifest['status'] != 'full_complete' or manifest['run_id'] != '20260902T012636Z':
        raise ValueError('old reference run is not the approved September 2 run')
    hashes = manifest['completed_unit_hashes']
    directory = root / 'private/completed_candidates'
    verify_files(directory, hashes)
    units = {}
    for name in sorted(hashes):
        unit = read_json(directory / name)
        decision = ForcedChoiceDecision.from_public_dict(unit['decision'])
        if decision.decision_sha256 != unit.get('decision_sha256', unit.get('classification_decision_sha256')):
            raise ValueError('old decision hash mismatch')
        if unit['place_id'] in units:
            raise ValueError('old completed candidates contain duplicate IDs')
        reducer = reduce_forced_choice_eligibility(decision, place_id=unit['place_id'], candidate_id=unit['candidate_id'])
        units[unit['place_id']] = {'decision': decision.to_public_dict(), 'reducer': reducer.to_public_dict()}
    if len(units) != 1226:
        raise ValueError('old frame differs from 1226')
    return units, {'run_manifest_sha256': digest((root / 'run_manifest.json').read_bytes()), 'completed_unit_hashes': hashes}


def finish(root, old_root):
    freeze = read_json(root / 'classification_freeze.json')
    if freeze['candidate_count'] != 1226:
        raise ValueError('full classification accounting gate not reached')
    verify_files(root, freeze['files'])
    records, protocol = verify_inputs(root)
    inputs = {r['place_id']: r for r in records}
    results = [verify_completed(root, r, read_json(root / f'private/classifications/{r["place_id"]}.json')) for r in records]
    returned_models, attempts, invalid, errors, usage = Counter(), 0, Counter(), Counter(), Counter()
    for record in records:
        directory = root / 'private/transactions' / record['place_id']
        for path in directory.glob('*.request.json'):
            request = read_json(path)
            if request['parameters'] != record['request'] or request['binding']['packet_sha256'] != record['packet_sha256']:
                raise ValueError('actual request differs from frozen complete packet')
            attempts += 1
            response_path = path.with_name(path.name.replace('.request.', '.response.'))
            if response_path.exists():
                envelope = read_json(response_path)
                if envelope['binding'] != request['binding'] or envelope['requested_at'] != request['requested_at']:
                    raise ValueError('request-response transaction association mismatch')
                if datetime.fromisoformat(envelope['received_at'].replace('Z','+00:00')) < datetime.fromisoformat(envelope['requested_at'].replace('Z','+00:00')):
                    raise ValueError('response predates request')
                if digest(envelope['raw_response_utf8'].encode()) != envelope['raw_response_sha256']:
                    raise ValueError('raw received body hash mismatch')
                body = json.loads(envelope['raw_response_utf8'])
                returned_models[body['model']] += 1
                usage.update({k: body.get('usage', {}).get(k,0) or 0 for k in ('input_tokens','output_tokens','total_tokens')})
            validation_path = path.with_name(path.name.replace('.request.', '.validation.'))
            if validation_path.exists():
                validation = read_json(validation_path)
                if not validation['valid']:
                    invalid[validation['reason']] += 1
            error_path = path.with_name(path.name.replace('.request.', '.error.'))
            if error_path.exists():
                errors[read_json(error_path)['reason']] += 1
    if set(returned_models) != {MODEL}:
        raise ValueError('returned model set differs from exact authorized model')
    # Old classifications enter only this separate, post-run comparison.
    old, old_commitment = load_old(old_root)
    comparison, changed_rows = compare(old, results)
    stats = rq1(results, inputs)
    source, closure = Path(protocol['source']), Path(protocol['closure'])
    if inventory(closure) != read_json(root / 'closure_preservation_before.json'):
        raise ValueError('accepted clean closure artifacts or timestamps changed')
    prior = verify_preservation(source)
    if not prior['unchanged']:
        raise ValueError('prior protected artifact preservation failed')
    implementation = read_json(root / 'classification_implementation.json')
    verify_files(Path(__file__).resolve().parents[1], implementation['files'])
    checks = {'all_1226_ids_accounted': len(results)==1226 and len(inputs)==1226,
        'no_duplicate_or_missing_ids': set(inputs)==set(old), 'all_valid_outputs_schema_conformant': True,
        'raw_response_and_packet_hashes_verified': True, 'full_classifier_packets_reconstructable': True,
        'archive_provenance_references_verified': True, 'board_absent_from_input_blocks': True,
        'original_reducer_unchanged_and_deterministic': True, 'exact_dated_model_only': True,
        'accepted_closure_and_prior_artifacts_preserved': True}
    ready = all(checks.values()) and stats['technical_unresolved_count']==0
    report = {'schema': 'clean-phase2-review-v1', 'run_id': root.name, 'requested_model': MODEL,
        'returned_models': dict(returned_models), 'completion_count': len(results), 'rq1': stats,
        'old_versus_corrected': comparison, 'verification': checks,
        'technical_unresolved': [{'place_id': r['place_id'], 'reasons': r['technical_reasons']} for r in results if r['status']!='completed'],
        'transactions': {'request_count': attempts, 'invalid_response_reasons': dict(invalid), 'api_error_reasons': dict(errors), 'token_usage_all_attempts': dict(usage)},
        'phase3_scientifically_ready_for_review': ready, 'phase3_executed': False,
        'rq2_rq3_recomputed': False, 'thesis_modified': False, 'stop_for_review': True}
    save(root, 'rq1_summary.json', stats)
    save(root, 'old_classification_commitments.json', old_commitment)
    save(root, 'old_versus_corrected.json', comparison)
    csv_table(root, 'old_versus_corrected_candidates.csv', changed_rows)
    csv_table(root, 'rq1_candidate_classifications.csv', [{
        'place_id': r['place_id'], 'candidate_id': r['candidate_id'], 'runtime_status': r['status'],
        'eligibility': r['reducer']['eligibility_state'], 'archive_state': inputs[r['place_id']]['archive']['state'],
        **{k:r['decision'][k] if r['decision'] else None for k in CORE_FIELDS},
        'confidence': r['decision']['claim_confidence'] if r['decision'] else None,
        'exclusion_reasons': r['reducer']['reason_codes']} for r in results])
    save(root, 'preservation_verification.json', {'accepted_closure_unchanged': True, 'prior_artifacts': prior})
    save(root, 'phase2_review.json', report)
    write(root / 'REVIEW_REPORT.md', markdown(report).encode(), immutable=True)
    reports = ['rq1_summary.json','old_classification_commitments.json','old_versus_corrected.json',
        'old_versus_corrected_candidates.csv','rq1_candidate_classifications.csv','preservation_verification.json',
        'phase2_review.json','REVIEW_REPORT.md']
    save(root, 'phase2_review_freeze.json', {'files': {p:digest((root/p).read_bytes()) for p in reports},
        'classification_freeze_sha256': digest((root/'classification_freeze.json').read_bytes()),
        'report_code_sha256': digest(Path(__file__).read_bytes()), 'phase3_execution_authorized': False})
    print(json.dumps(report, indent=2), flush=True)


def markdown(report):
    r = report['rq1']
    lines = ['# Corrected Phase-2 classification review', '',
        f'Run: `{report["run_id"]}`. Requested model: `{report["requested_model"]}`. '
        f'All returned models: {list(report["returned_models"])}. Candidates accounted: {report["completion_count"]}.', '',
        '| Eligibility | Count | Percent |', '|---|---:|---:|']
    for state, value in r['eligibility'].items():
        lines.append(f'| {state} | {value["count"]} | {value["percent"]:.2f}% |')
    lines += ['', '| Judgment | Distribution |', '|---|---|']
    for field, values in r['four_judgments'].items():
        lines.append('| '+field+' | '+', '.join(f'{label}: {v["count"]} ({v["percent"]:.2f}%)' for label,v in values.items())+' |')
    lines += ['', 'Confidence distributions:', '', '| Judgment | Distribution |', '|---|---|']
    for field, values in r['confidence'].items():
        lines.append('| '+field+' | '+', '.join(f'{label}: {v["count"]}' for label,v in values.items())+' |')
    lines += ['', 'Exclusion reason combinations:', '']
    lines += [f'- {key}: {value["count"]} ({value["percent"]:.2f}% of exclusions)' for key,value in r['exclusion_reason_combinations'].items()]
    lines += ['', f'Target status perfectly partitions eligibility: **{r["target_status_perfectly_partitions_eligibility"]}**. '
        f'Observed target/eligibility table: {r["target_status_by_eligibility"]}.', '',
        '| Eligibility group | Candidates | With websites | With archive text | Text coverage in group | With business evidence |',
        '|---|---:|---:|---:|---:|---:|']
    for state,v in r['archive_coverage_by_eligibility'].items():
        fraction = f'{v["archive_text_percent_of_group"]:.2f}%' if v['candidates'] else 'not applicable'
        lines.append(f'| {state} | {v["candidates"]} | {v["with_usable_website"]} | {v["with_archive_text"]} | {fraction} | {v["with_business_evidence"]} |')
    lines += ['', 'Difference between the September 2 run and the corrected evidence-collection run:', '',
              '| Changed field | Candidates |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in report['old_versus_corrected']['change_counts'].items()]
    lines += ['', report['old_versus_corrected']['comparison_scope'], '',
        f'Verification: {report["verification"]}. Technical-unresolved: {r["technical_unresolved_count"]}.', '',
        f'**Phase 3 scientifically ready for review: {report["phase3_scientifically_ready_for_review"]}.** '
        'RQ2/RQ3 and thesis changes were not executed. Stop for user review.', '',
        'Complete exact inputs, raw provider responses, parsed responses, validator/reducer outcomes, timestamps and hashes are preserved privately. '
        'The archived text represents partial historical coverage, not a complete history of each candidate.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True)
    a=p.parse_args();finish(a.root.resolve(),a.old_root.resolve())
