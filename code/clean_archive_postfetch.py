"""Offline, explicit custody audit after a raw-crawler sealing failure.

Never restores a quarantined file, reads known blocked body hashes, fetches a
replacement, or rewrites the frozen crawler's states and attempt receipts.
"""
from pathlib import Path
from collections import Counter
import argparse
import re

from wayback_snapshot_fetcher import encoded, digest, read_json, write_json, bound

VERSION = 'clean-archive-postfetch-v1'
REPOSITORY = Path(__file__).resolve().parents[1]


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact escapes raw root')
    return path


def verify_implementation(root):
    manifest = read_json(root / 'postfetch_implementation_manifest.json')
    for relative, expected in manifest['files'].items():
        if digest((REPOSITORY / relative).read_bytes()) != expected:
            raise ValueError('postfetch implementation changed: ' + relative)


def audit_raw(root, snapshots, plan):
    root = Path(root).resolve()
    raw_root = root / 'raw_retrieval'
    security = read_json(root / 'security/quarantine_receipt.json')
    for relative, expected in security.get('source_files_sha256', {}).items():
        if digest(safe(root, relative).read_bytes()) != expected:
            raise ValueError('security source receipt changed')
    blocked = set(security['blocked_body_sha256'])
    if any(not re.fullmatch(r'[a-f0-9]{64}', value) for value in blocked):
        raise ValueError('invalid blocked content digest')
    snapshot_sha = digest((root / 'snapshots.json').read_bytes())
    if (raw_root / 'input_manifest.json').read_bytes() != (root / 'snapshots.json').read_bytes():
        raise ValueError('raw input manifest changed')
    files, excluded, cache, verified, outcomes = {}, {}, {}, {}, {}
    attempts, executions = {}, set()

    def metadata(relative, expected=None):
        content = safe(raw_root, relative).read_bytes()
        actual = digest(content)
        if expected is not None and expected != actual:
            raise ValueError('metadata commitment changed: ' + relative)
        files[relative] = actual
        return read_json(safe(raw_root, relative))

    def body(relative, expected):
        if relative in excluded:
            return None
        if relative in cache:
            return cache[relative]
        reason = None
        actual = None
        if expected in blocked:
            reason = 'raw_body_quarantined'
        else:
            try:
                value = safe(raw_root, relative).read_bytes()
                actual = digest(value)
                if actual != expected:
                    reason = 'raw_body_hash_mismatch'
                else:
                    files[relative] = actual
                    cache[relative] = value
                    return value
            except OSError as exc:
                reason = 'raw_body_unreadable_' + type(exc).__name__
        excluded[relative] = {'expected_sha256': expected, 'observed_sha256': actual,
                              'reason': reason, 'read_attempted': expected not in blocked}
        return None

    metadata('input_manifest.json', snapshot_sha)
    original_counts = Counter()
    original_html = 0
    effective_html = 0
    integrity_ids = []
    quarantined_ids = []
    for record in snapshots:
        ident, stamp = record['candidate_id'], record['confirmed_snapshot_timestamp']
        state_path = f'state/{ident}/{stamp}.json'
        state = metadata(state_path)
        if state['input_record_sha256'] != digest(encoded(record)) or not state['history']:
            raise ValueError('raw state record binding or history is incomplete')
        before = set(excluded)
        for entry in state['history']:
            attempt = metadata(entry)
            if (attempt['candidate_id'] != ident or attempt['confirmed_snapshot_timestamp'] != stamp):
                raise ValueError('attempt identity differs from record')
            attempts[entry] = attempt
            execution = attempt['execution']
            if not re.fullmatch(r'[a-f0-9]{32}', execution):
                raise ValueError('unsafe original execution identity')
            executions.add(execution)
            directory = Path(entry).parent.as_posix()
            started = metadata(directory + '/started.json')
            if started['record'] != record or started['execution'] != execution:
                raise ValueError('attempt start record changed')
            for index, hop in enumerate(attempt['redirect_chain']):
                folder = directory + '/' + str(index)
                request = metadata(folder + '/request.json')
                response = metadata(folder + '/response.json')
                if request['url'] != hop['requested_url']:
                    raise ValueError('attempt request changed')
                if any(response.get(k) != hop.get(k) for k in
                       ('sha256', 'body_length', 'http_status', 'final_url', 'headers', 'error')):
                    raise ValueError('attempt response changed')
                value = body(folder + '/body.bin', hop['sha256'])
                if value is not None and len(value) != hop['body_length']:
                    raise ValueError('attempt body length changed')
        terminal = attempts[state['history'][-1]]
        if any(terminal[k] != state[k] for k in ('state', 'reason', 'sha256', 'http_status')):
            raise ValueError('original state differs from terminal receipt')
        if state['state'] not in {'success', 'permanent_failure', 'retryable_failure'}:
            raise ValueError('unfinished raw request')
        original_counts[state['state']] += 1
        page = None
        is_html = False
        if state['state'] == 'success':
            required = {'body.bin', 'headers.json', 'request.json', 'response.json', 'provenance.json'}
            if not required <= set(state['files']):
                raise ValueError('successful state omits required raw files')
            relative = f'raw/{ident}/{stamp}'
            sidecars = {}
            for name, expected in state['files'].items():
                if Path(name).name != name:
                    raise ValueError('unsafe raw filename')
                if name in {'body.bin', 'page.html'}:
                    if expected != state['sha256']:
                        raise ValueError('successful body commitment inconsistent')
                    body(relative + '/' + name, expected)
                else:
                    sidecars[name] = metadata(relative + '/' + name, expected)
            provenance = sidecars['provenance.json']
            if (provenance['input_record_sha256'] != digest(encoded(record))
                    or provenance['input_manifest_sha256'] != snapshot_sha
                    or provenance['sha256'] != state['sha256']
                    or provenance['http_status'] != 200 or not bound(record, provenance['final_url'])):
                raise ValueError('successful snapshot provenance changed')
            is_html = provenance['content_type'].split(';')[0].lower() in {'text/html', 'application/xhtml+xml'}
            original_html += is_html
            page = {'state': 'success', 'raw_folder': raw_root / relative,
                    'provenance': provenance, 'body': cache.get(relative + '/body.bin'),
                    'headers': sidecars['headers.json'], 'reason': None}
        losses = sorted(set(excluded) - before)
        if losses:
            integrity_ids.append(ident)
            quarantined = any(excluded[p]['reason'] == 'raw_body_quarantined' for p in losses)
            if quarantined:
                quarantined_ids.append(ident)
            effective = {'state': 'permanent_failure', 'http_status': state['http_status'],
                         'reason': 'raw_body_quarantined' if quarantined else 'raw_artifact_integrity_failure'}
        elif page is not None:
            effective = page
            effective_html += is_html
        else:
            effective = {k: state[k] for k in ('state', 'reason', 'http_status')}
        verified[ident] = effective
        outcomes[ident] = {'state': effective['state'], 'reason': effective['reason'],
                          'http_status': state['http_status'], 'original_state': state['state'],
                          'original_reason': state['reason'], 'state_file': state_path,
                          'excluded_files': losses}
    for execution in executions:
        settings = metadata(f'executions/{execution}/settings.json')
        if settings['input_sha256'] != snapshot_sha or any(settings.get(k) != v for k, v in plan['fetch'].items()):
            raise ValueError('original execution settings differ from frozen plan')
    hops = [hop for attempt in attempts.values() for hop in attempt['redirect_chain']]
    statuses = Counter(str(h['http_status']) for h in hops)
    manifest = {'schema': VERSION, 'input_manifest_sha256': snapshot_sha, 'input_count': len(snapshots),
        'security_receipt_sha256': digest((root / 'security/quarantine_receipt.json').read_bytes()),
        'execution': 'offline-postfetch-audit', 'source_executions': sorted(executions),
        'settings': plan['fetch'], 'original_state_counts': dict(original_counts),
        'state_counts': dict(Counter(v['state'] for v in verified.values())),
        'raw_content_successes': sum(v['state'] == 'success' for v in verified.values()),
        'successful_html_bodies': effective_html, 'raw_http_successful_snapshots': original_counts['success'],
        'raw_http_html_successes': original_html, 'responses_429': statuses['429'],
        'verified_429_responses_all_sealed_attempts': statuses['429'],
        'http_status_counts': dict(statuses), 'attempt_results': list(attempts.values()),
        'integrity_failure_snapshot_ids': integrity_ids, 'quarantined_snapshot_ids': quarantined_ids,
        'files': files, 'exclusions': excluded, 'snapshot_outcomes': outcomes,
        'network_calls_in_audit': 0, 'original_states_modified': False,
        'scope': 'Verified admissible files plus explicit custody gaps; not a claim that every received body survived.'}
    return verified, manifest


def verify_raw(root, snapshots, plan):
    verify_implementation(root)
    seal = read_json(root / 'raw_postfetch_seal.json')
    data = (root / 'raw_postfetch_manifest.json').read_bytes()
    if digest(data) != seal['manifest_sha256']:
        raise ValueError('postfetch manifest changed')
    verified, actual = audit_raw(root, snapshots, plan)
    if encoded(actual) != data:
        raise ValueError('postfetch audit did not reproduce exactly')
    return verified, actual


def seal_raw(root):
    from clean_archive_preflight import verify_discovery
    root = Path(root).resolve()
    verify_implementation(root)
    plan, snapshots, _, _ = verify_discovery(root)
    _, manifest = audit_raw(root, snapshots, plan)
    write_json(root / 'raw_postfetch_manifest.json', manifest, immutable=True)
    write_json(root / 'raw_postfetch_seal.json', {'schema': VERSION,
        'manifest_sha256': digest(encoded(manifest)),
        'snapshot_manifest_sha256': manifest['input_manifest_sha256'],
        'all_original_body_commitments_verified': not manifest['exclusions'],
        'network_calls': 0, 'quarantined_content_restored': False}, immutable=True)
    print(encoded({k: v for k, v in manifest.items() if k not in
                  {'files', 'attempt_results', 'snapshot_outcomes'}}).decode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    seal_raw(parser.parse_args().run_dir)
