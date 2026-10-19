"""Same bounded runtime with versioned exact-citation validation."""
from run_clean_phase2 import *
from clean_phase2_closure_contract import validate_record, validate_decision

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
                if self.stop.is_set():
                    raise ClassificationStop('run_stopped')
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
                    fatal = not transient or code in {'model_not_found', 'insufficient_quota', 'invalid_api_key', 'credit_balance_exhausted'}
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

