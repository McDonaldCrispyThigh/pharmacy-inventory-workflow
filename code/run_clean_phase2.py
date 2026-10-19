"""Authorized Phase 2 only: immutable complete inputs and resumable transactions."""
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import sqlite3
import threading
import time
from jsonschema.exceptions import ValidationError

from wayback_snapshot_fetcher import encoded, digest, read_json, write_json, write, now, retry_after
from clean_phase2_evidence import MODEL, VERSION, STATES, build_packet, validate_record, validate_decision, output_text, save
from forced_choice_eligibility import reduce_forced_choice_eligibility
from clean_archive_preflight_postfetch import verify_files

REPO = Path(__file__).resolve().parents[1]


class ClassificationStop(RuntimeError):
    pass


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': digest(p.read_bytes()),
        'mtime_ns': p.stat().st_mtime_ns, 'birthtime_ns': getattr(p.stat(), 'st_birthtime_ns', None)}
        for p in sorted(root.rglob('*')) if p.is_file()}


def prepare(root, source, closure):
    from verify_clean_archive_closure import verify
    if (root / 'input_freeze.json').exists():
        verify_files(root, read_json(root / 'input_freeze.json')['files'])
        print('Existing full input freeze verified; no candidate calls.', flush=True)
        return
    availability = read_json(root / 'model_availability.json')
    if availability['status'] != 'available' or availability['returned_model'] != MODEL:
        raise ClassificationStop('exact_dated_model_not_verified_available')
    print('Independently verifying accepted Phase-1 provenance before preparing classifier inputs.', flush=True)
    verify(source, closure)
    closure_report = read_json(closure / 'closure_preflight.json')
    if closure_report['counts']['candidate_states'] != STATES:
        raise ValueError('accepted archive state distribution changed')
    plan = read_json(source / 'plan.json')
    files, rows, sizes = {}, [], []
    for row in plan['candidates']:
        cid = row['candidate_id']
        path = closure / f'private/packets/{cid}.json'
        packet = build_packet(read_json(path), digest(path.read_bytes()), str(closure))
        validate_record(packet)
        # UTF-8 byte count is a conservative byte-level tokenizer bound. Reserve
        # the documented maximum 128k output plus request framing under 400k.
        upper = len(encoded(packet['request'])) + 128000 + 4096
        if upper > 400000:
            raise ValueError('complete request exceeds conservative context envelope; no truncation allowed')
        relative = f'private/inputs/{cid}.json'
        files[relative] = save(root, relative, packet)
        rows.append({'place_id': cid, 'candidate_id': packet['candidate_id'], 'file': relative,
                     'file_sha256': files[relative], 'packet_sha256': packet['packet_sha256'],
                     'request_sha256': packet['request_sha256'], 'archive_state': packet['archive']['state']})
        sizes.append(len(encoded(packet['request'])))
    if len(rows) != 1226 or len({r['place_id'] for r in rows}) != 1226:
        raise ValueError('classifier frame does not contain 1226 unique Place IDs')
    baseline = inventory(closure)
    files['closure_preservation_before.json'] = save(root, 'closure_preservation_before.json', baseline)
    protocol = {'schema': VERSION, 'run_id': root.name, 'model': MODEL, 'source': str(source), 'closure': str(closure),
        'candidate_count': len(rows), 'archive_state_counts': STATES, 'candidate_rows': rows,
        'max_attempts': 3, 'workers': 6, 'timeout_seconds': 180, 'backoff_seconds': [5, 15, 30],
        'model_call_parameters': 'Original structured Responses request, store=false, dated model; no temperature/reasoning/output-limit override.',
        'full_reviews': True, 'full_verified_archive_text': True, 'raw_html_in_prompts': False,
        'candidate_selection': 'All 1226 frozen Place IDs, in original frame order; no outcome-based selection.',
        'unchanged_contract_files': {p: digest((REPO / p).read_bytes()) for p in (
            'code/forced_choice_classifier.py', 'code/forced_choice_models.py', 'code/forced_choice_eligibility.py')},
        'new_adapter_reason': 'Legacy typed inputs cap review/archive text and encode CDX states. The new adapter retains full accepted clean evidence; it leaves the four labels, JSON schema, decision object and reducer unchanged.',
        'historical_validation': 'Only closure-approved active-by service text and explicit opening claims qualify. Capture time or outlet scope alone cannot create a historical claim.',
        'context_check': {'method': 'conservative UTF-8 bytes + 128000 output + 4096 framing <=400000',
                          'maximum_request_bytes': max(sizes), 'total_request_bytes': sum(sizes),
                          'official_source': 'https://developers.openai.com/api/docs/models/gpt-5-mini'},
        'stop_after_classification_rq1_and_provenance_comparison': True, 'phase3_execution_allowed': False}
    files['classification_protocol.json'] = save(root, 'classification_protocol.json', protocol)
    for name in ('user_authorization.txt', 'stage_scope.json', 'model_availability.json',
                 'model_metadata_response.json', 'model_probe_request.json', 'model_probe_response.json'):
        files[name] = digest((root / name).read_bytes())
    save(root, 'input_freeze.json', {'schema': VERSION, 'files': files, 'candidate_count': 1226,
        'all_packets_frozen_before_substantive_calls': True,
        'closure_freeze_sha256': digest((closure / 'closure_freeze.json').read_bytes()),
        'closure_verification_sha256': digest((closure / 'verification_receipt.json').read_bytes())})
    print(json.dumps({'frozen_candidates': 1226, 'maximum_request_bytes': max(sizes),
                      'archive_states': STATES, 'substantive_calls': 0}), flush=True)


class Runtime:
    def __init__(self, root, client, *, sleep=time.sleep, clock=time.time):
        self.root, self.client, self.sleep, self.clock = root, client, sleep, clock
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.next_allowed = 0.0
        cooldown = root / 'runtime_cooldown.json'
        if cooldown.exists():
            self.next_allowed = read_json(cooldown)['next_allowed_epoch']

    def pause(self, seconds):
        with self.lock:
            self.next_allowed = max(self.next_allowed, self.clock() + seconds)
            write_json(self.root / 'runtime_cooldown.json', {'next_allowed_epoch': self.next_allowed})

    def wait_ready(self):
        while self.next_allowed > self.clock():
            if self.stop.is_set():
                raise ClassificationStop('run_stopped')
            self.sleep(min(2, self.next_allowed - self.clock()))

    def candidate(self, record, max_attempts=3):
        validate_record(record)
        cid = record['place_id']
        destination = self.root / f'private/classifications/{cid}.json'
        if destination.exists():
            return verify_completed(self.root, record, read_json(destination))
        directory = self.root / 'private/transactions' / cid
        binding = {'place_id': cid, 'candidate_id': record['candidate_id'], 'packet_sha256': record['packet_sha256'],
                   'request_sha256': record['request_sha256'], 'requested_model': MODEL}
        reasons = []
        for attempt in range(1, max_attempts + 1):
            prefix = directory / f'{attempt:02d}'
            request_path = prefix.with_suffix('.request.json')
            response_path = prefix.with_suffix('.response.json')
            error_path = prefix.with_suffix('.error.json')
            if error_path.exists():
                failure = read_json(error_path)
                if failure['binding'] != binding:
                    raise ValueError('failed transaction packet binding changed')
                if failure['fatal']:
                    self.stop.set()
                    raise ClassificationStop(failure['reason'])
                reasons.append(failure['reason'])
                continue
            if not response_path.exists():
                if request_path.exists():
                    prior = read_json(request_path)
                    if prior['binding'] != binding:
                        raise ValueError('interrupted request binding changed')
                    reason = 'interrupted_request_without_received_response'
                    save(self.root, error_path.relative_to(self.root), {'binding': binding, 'attempt': attempt,
                        'fatal': False, 'reason': reason, 'recorded_at': now()})
                    reasons.append(reason)
                    continue
                if self.stop.is_set():
                    raise ClassificationStop('run_stopped')
                self.wait_ready()
                request = {'binding': binding, 'attempt': attempt, 'requested_at': now(),
                           'parameters': record['request'], 'prompt_version': record['prompt_version'],
                           'schema_version': record['schema_version']}
                save(self.root, request_path.relative_to(self.root), request)
                try:
                    received = self.client.responses.with_raw_response.create(**record['request'])
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    code = getattr(exc, 'code', None)
                    kind = type(exc).__name__
                    transient = status in {408, 409, 429, 500, 502, 503, 504} or kind in {'APITimeoutError', 'APIConnectionError'}
                    fatal = not transient or code in {'model_not_found', 'insufficient_quota', 'invalid_api_key'}
                    reason = 'requested_model_unavailable' if status == 404 or code == 'model_not_found' else kind
                    save(self.root, error_path.relative_to(self.root), {'binding': binding, 'attempt': attempt,
                        'recorded_at': now(), 'error_type': kind, 'http_status': status, 'code': code,
                        'fatal': fatal, 'reason': reason})
                    if fatal:
                        self.stop.set()
                        raise ClassificationStop(reason) from None
                    retry_header = getattr(getattr(exc, 'response', None), 'headers', {}).get('retry-after')
                    delay = retry_after(retry_header, self.clock()) or [5, 15, 30][attempt-1]
                    self.pause(delay)
                    reasons.append(reason)
                    continue
                envelope = {'binding': binding, 'attempt': attempt, 'requested_at': request['requested_at'],
                    'received_at': now(), 'http_status': received.status_code, 'request_id': received.request_id,
                    'raw_response_utf8': received.content.decode('utf-8'), 'raw_response_sha256': digest(received.content)}
                # Atomic envelope preserves exact raw bytes AND local timestamps
                # before parsing, validation or reducer execution can fail.
                save(self.root, response_path.relative_to(self.root), envelope)
            envelope = read_json(response_path)
            if envelope['binding'] != binding or digest(envelope['raw_response_utf8'].encode()) != envelope['raw_response_sha256']:
                raise ValueError('received model transaction commitment changed')
            response = json.loads(envelope['raw_response_utf8'])
            if response.get('model') != MODEL:
                self.stop.set()
                raise ClassificationStop('returned_model_mismatch')
            try:
                raw = json.loads(output_text(response))
                decision, reducer = validate_decision(record, raw)
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                reasons.append(reason)
                save(self.root, prefix.with_suffix('.validation.json').relative_to(self.root), {
                    'binding': binding, 'attempt': attempt, 'valid': False, 'reason': reason,
                    'raw_response_sha256': envelope['raw_response_sha256']})
                if attempt < max_attempts:
                    self.sleep([5, 15, 30][attempt-1])
                continue
            save(self.root, prefix.with_suffix('.parsed.json').relative_to(self.root), raw)
            save(self.root, prefix.with_suffix('.validation.json').relative_to(self.root), {
                'binding': binding, 'attempt': attempt, 'valid': True, 'decision_sha256': decision.decision_sha256,
                'raw_response_sha256': envelope['raw_response_sha256'], 'reducer': reducer.to_public_dict()})
            result = {**binding, 'status': 'completed', 'attempt': attempt, 'returned_model': response['model'],
                'response_file': response_path.relative_to(self.root).as_posix(),
                'response_file_sha256': digest(response_path.read_bytes()),
                'decision': decision.to_public_dict(), 'parsed_response': raw, 'reducer': reducer.to_public_dict(),
                'usage': response.get('usage'), 'requested_at': envelope['requested_at'], 'received_at': envelope['received_at'],
                'prompt_version': record['prompt_version'], 'schema_version': record['schema_version']}
            save(self.root, destination.relative_to(self.root), result)
            return result
        result = {**binding, 'status': 'technical_unresolved', 'attempt': max_attempts,
            'decision': None, 'returned_model': None, 'technical_reasons': reasons,
            'reducer': reduce_forced_choice_eligibility(None, place_id=cid, candidate_id=record['candidate_id']).to_public_dict()}
        save(self.root, destination.relative_to(self.root), result)
        return result


def verify_completed(root, record, result):
    if result['place_id'] != record['place_id'] or result['packet_sha256'] != record['packet_sha256'] or result['request_sha256'] != record['request_sha256']:
        raise ValueError('completed candidate refers to a different frozen packet')
    if result['status'] == 'completed':
        path = root / result['response_file']
        if digest(path.read_bytes()) != result['response_file_sha256']:
            raise ValueError('completed raw response hash changed')
        envelope = read_json(path)
        if digest(envelope['raw_response_utf8'].encode()) != envelope['raw_response_sha256']:
            raise ValueError('raw model response bytes changed')
        response = json.loads(envelope['raw_response_utf8'])
        parsed = json.loads(output_text(response))
        decision, reducer = validate_decision(record, parsed)
        if result['parsed_response'] != parsed or result['decision'] != decision.to_public_dict() or result['reducer'] != reducer.to_public_dict():
            raise ValueError('completed decision/reducer does not reproduce')
    elif result['status'] != 'technical_unresolved' or result['decision'] is not None:
        raise ValueError('invalid completion status')
    return result


def verify_inputs(root):
    freeze = read_json(root / 'input_freeze.json')
    verify_files(root, freeze['files'])
    protocol = read_json(root / 'classification_protocol.json')
    closure = Path(protocol['closure'])
    if digest((closure / 'closure_freeze.json').read_bytes()) != freeze['closure_freeze_sha256']:
        raise ValueError('source closure seal changed')
    records = []
    for row in protocol['candidate_rows']:
        record = validate_record(read_json(root / row['file']))
        source_path = closure / record['source_packet_file']
        source_sha = digest(source_path.read_bytes())
        rebuilt = build_packet(read_json(source_path), source_sha, str(closure))
        if encoded(rebuilt) != encoded(record):
            raise ValueError('complete packet cannot be reconstructed from frozen closure')
        records.append(record)
    return records, protocol


def run(root):
    from openai import OpenAI
    from dotenv import dotenv_values
    records, protocol = verify_inputs(root)
    verify_files(REPO, read_json(root / 'classification_implementation.json')['files'])
    key = os.environ.get('OPENAI_API_KEY') or dotenv_values(REPO / '.env', interpolate=False).get('OPENAI_API_KEY')
    if not key:
        raise ClassificationStop('missing_openai_credential')
    client = OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=protocol['timeout_seconds'])
    runtime = Runtime(root, client)
    connection = sqlite3.connect(root / 'runtime_lock.sqlite3', timeout=0)
    connection.execute('BEGIN IMMEDIATE')
    results = []
    try:
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            iterator = iter(records)
            pending = {}
            for _ in range(protocol['workers']):
                record = next(iterator, None)
                if record:
                    pending[pool.submit(runtime.candidate, record, protocol['max_attempts'])] = record
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result = future.result()
                    results.append(result)
                    progress = {'accounted': len(results), 'total': 1226,
                        'status_counts': dict(Counter(r['status'] for r in results)), 'updated_at': now()}
                    write_json(root / 'classification_progress.json', progress)
                    if len(results) % 10 == 0 or result['status'] != 'completed' or len(results) == 1226:
                        print(json.dumps(progress), flush=True)
                    record = next(iterator, None)
                    if record:
                        pending[pool.submit(runtime.candidate, record, protocol['max_attempts'])] = record
    except BaseException:
        runtime.stop.set()
        raise
    finally:
        connection.rollback()
        connection.close()
        client.close()
    if {r['place_id'] for r in results} != {r['place_id'] for r in records} or len(results) != 1226:
        raise ValueError('classification accounting differs from frozen frame')
    files = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for directory in
             ('private/classifications', 'private/transactions') for p in (root / directory).rglob('*.json')}
    save(root, 'classification_freeze.json', {'schema': VERSION, 'candidate_count': 1226, 'files': files,
        'input_freeze_sha256': digest((root / 'input_freeze.json').read_bytes()),
        'implementation_sha256': digest((root / 'classification_implementation.json').read_bytes()),
        'status_counts': dict(Counter(r['status'] for r in results)), 'phase3_execution_allowed': False})
    print('Classification layer complete and frozen; RQ1/provenance audit still required.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--closure', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root.resolve(), args.source.resolve(), args.closure.resolve())
    else:
        run(args.root.resolve())
