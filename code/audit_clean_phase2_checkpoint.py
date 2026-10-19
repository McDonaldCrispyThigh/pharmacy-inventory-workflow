"""Offline interrupted-run audit. Never advances the full Phase-2 gate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from clean_phase2_evidence import MODEL
from clean_archive_preflight_postfetch import verify_files, verify_preservation
from run_clean_phase2 import verify_inputs, verify_completed, inventory
from wayback_snapshot_fetcher import digest, read_json, write, write_json


def audit(root):
    # Refuse to audit a live writer as an interrupted checkpoint.
    lock = sqlite3.connect(root / 'runtime_lock.sqlite3', timeout=0)
    lock.execute('BEGIN IMMEDIATE')
    try:
        records, protocol = verify_inputs(root)
        inputs = {r['place_id']: r for r in records}
        if len(inputs) != len(records) or len(inputs) != 1226:
            raise ValueError('candidate universe mismatch')
        implementation = read_json(root / 'classification_implementation.json')
        verify_files(Path(__file__).resolve().parents[1], implementation['files'])
        commitments = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                       for name in ('private/classifications', 'private/transactions')
                       for p in (root / name).rglob('*.json')}
        results = {}
        for path in (root / 'private/classifications').glob('*.json'):
            r = read_json(path)
            cid = r['place_id']
            if cid not in inputs or cid in results:
                raise ValueError('unexpected/duplicate classifier output')
            results[cid] = verify_completed(root, inputs[cid], r)
        requests, responses, errors, invalid, models, usage = 0, 0, Counter(), Counter(), Counter(), Counter()
        started, credit_affected, interrupted = set(), set(), []
        for path in (root / 'private/transactions').glob('*/*.request.json'):
            req = read_json(path)
            cid = req['binding']['place_id']
            record = inputs[cid]
            if req['parameters'] != record['request'] or req['binding']['packet_sha256'] != record['packet_sha256']:
                raise ValueError('request differs from frozen evidence')
            started.add(cid)
            requests += 1
            rsp_path = path.with_name(path.name.replace('.request.', '.response.'))
            err_path = path.with_name(path.name.replace('.request.', '.error.'))
            val_path = path.with_name(path.name.replace('.request.', '.validation.'))
            if rsp_path.exists():
                rsp = read_json(rsp_path)
                if rsp['binding'] != req['binding'] or rsp['requested_at'] != req['requested_at']:
                    raise ValueError('response association mismatch')
                if datetime.fromisoformat(rsp['received_at']) < datetime.fromisoformat(req['requested_at']):
                    raise ValueError('response predates request')
                if digest(rsp['raw_response_utf8'].encode()) != rsp['raw_response_sha256']:
                    raise ValueError('raw response hash mismatch')
                body = json.loads(rsp['raw_response_utf8'])
                models[body['model']] += 1
                usage.update({k: body.get('usage', {}).get(k, 0) or 0 for k in ('input_tokens', 'output_tokens', 'total_tokens')})
                responses += 1
            if err_path.exists():
                err = read_json(err_path)
                if err['binding'] != req['binding']:
                    raise ValueError('error association mismatch')
                errors[str(err.get('code') or err['reason'])] += 1
                if err.get('code') == 'credit_balance_exhausted':
                    credit_affected.add(cid)
            if val_path.exists():
                val = read_json(val_path)
                if not val['valid']:
                    invalid[val['reason']] += 1
            if not rsp_path.exists() and not err_path.exists():
                interrupted.append(path.relative_to(root).as_posix())
        if set(models) != {MODEL}:
            raise ValueError('model mismatch')
        closure = Path(protocol['closure'])
        if inventory(closure) != read_json(root / 'closure_preservation_before.json'):
            raise ValueError('accepted closure changed')
        prior = verify_preservation(Path(protocol['source']))
        if not prior['unchanged']:
            raise ValueError('protected historical artifacts changed')
        verify_files(root, commitments)
        counts = Counter(r['status'] for r in results.values())
        reasons = Counter(reason for r in results.values() if r['status'] != 'completed'
                          for reason in set(r['technical_reasons']))
        report = {
            'schema': 'clean-phase2-interrupted-checkpoint-v1',
            'run_id': root.name, 'audited_at': datetime.now(timezone.utc).isoformat(),
            'status': 'blocked_credit_balance_exhausted', 'candidate_universe': 1226,
            'requested_model': MODEL, 'returned_models': dict(models),
            'finished_candidate_records': len(results), 'status_counts': dict(counts),
            'pending_candidate_count': len(inputs) - len(results),
            'pending_place_ids': sorted(set(inputs) - set(results)),
            'never_requested_count': len(set(inputs) - started),
            'started_without_final_record_count': len(started - set(results)),
            'technical_unresolved_reasons_nonexclusive': dict(reasons),
            'credit_exhaustion_affected_candidates': len(credit_affected),
            'credit_affected_place_ids': sorted(credit_affected),
            'request_receipts': requests, 'raw_response_receipts': responses,
            'api_error_codes': dict(errors), 'invalid_response_reasons': dict(invalid),
            'request_receipts_without_response_or_error': interrupted,
            'token_usage_received_responses': dict(usage),
            'verification': {'all_1226_frozen_inputs_reconstructed': True,
                'existing_completed_outputs_revalidated': True,
                'exact_dated_model_only': True, 'actual_requests_match_frozen_packets': True,
                'raw_response_hashes_verified': True, 'transaction_files_stable': True,
                'classifier_implementation_unchanged': True, 'accepted_closure_unchanged': True,
                'prior_artifacts_unchanged': True},
            'prior_preservation_verification': prior,
            'classification_layer_complete': False, 'full_rq1_report_generated': False,
            'old_classifications_read': False, 'phase3_ready': False,
            'continuation_note': 'Preserve all validated candidates and failed attempts. Restore API credits before further calls. Credit exhaustion must stop execution; it is not a transient rate limit. Do not treat this partial checkpoint as the final RQ1 or classification review.',
        }
        out = root / 'interrupted_checkpoint_v1'
        write_json(out / 'checkpoint_report.json', report, immutable=True)
        write_json(out / 'transaction_commitments.json', commitments, immutable=True)
        text = ('# Phase-2 interrupted checkpoint\n\n'
                f'Run `{root.name}` stopped after the API returned `credit_balance_exhausted`.\n\n'
                f'Frozen candidates: 1,226. Validated: {counts["completed"]}. Technical unresolved: {counts["technical_unresolved"]}. '
                f'Pending: {report["pending_candidate_count"]}.\n\n'
                'This is not the complete Phase-2 review. Full RQ1 and the September 2 comparison remain pending. '
                'Phase 3 is not ready. No RQ2/RQ3 or thesis work was performed.\n\n'
                'All frozen input packets reconstruct; existing valid outputs, raw responses, unchanged classifier implementation, '
                'accepted archive closure and prior protected files passed the offline checks. '
                'See checkpoint_report.json for counts, failure codes and the resumable candidate lists.\n')
        write(out / 'CHECKPOINT_REPORT.md', text.encode(), immutable=True)
        write(out / 'audit_source.py', Path(__file__).read_bytes(), immutable=True)
        write_json(out / 'checkpoint_freeze.json', {
            'files': {p.name: digest(p.read_bytes()) for p in out.iterdir() if p.is_file()},
            'input_freeze_sha256': digest((root / 'input_freeze.json').read_bytes()),
            'classification_implementation_sha256': digest((root / 'classification_implementation.json').read_bytes()),
            'phase2_complete': False, 'phase3_execution_allowed': False}, immutable=True)
        print(json.dumps({k: v for k, v in report.items() if k not in {
            'pending_place_ids', 'credit_affected_place_ids', 'prior_preservation_verification'}}, indent=2))
    finally:
        lock.rollback()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root.resolve())
