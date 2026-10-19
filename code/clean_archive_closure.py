"""Versioned final closure audit. No discovery/provider/model calls.

The prepare command only reads stored official responses and seals a supplemental
exact-snapshot manifest. The separately frozen standalone crawler may fill only
the missing bodies. Finalization verifies and freezes a new set of 1,226 packets.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import copy
import csv
import io
import json
from urllib.parse import urlencode

from wayback_snapshot_fetcher import encoded, digest, read_json, write, write_json, website_key, parse_replay, bound
from clean_archive_closure_rules import VERSION, CATEGORIES, ROUTES, TRACKING, compare_urls, revalidate
from clean_archive_preflight_postfetch import (verify_inputs, verify_files, verify_preservation,
    parse_twice, validate_claim_spans, candidate_state, opening_summary, technical_raw_failure, STATES)
from clean_archive_postfetch import verify_raw

REPOSITORY = Path(__file__).resolve().parents[1]


def save(root, relative, value):
    write_json(root / relative, value, immutable=True)
    return digest(encoded(value))


def table(root, relative, rows):
    if not rows:
        return
    out = io.StringIO(newline='')
    columns = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(out, columns, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()})
    write(root / relative, out.getvalue().encode('utf-8'), immutable=True)


def source_inventory(source):
    """Inventory metadata and safe hashes; never touch excluded body aliases."""
    excluded = {'raw_retrieval/' + p for p in read_json(source / 'raw_postfetch_manifest.json')['exclusions']}
    files = {}
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if relative in excluded:
            continue
        stat = path.stat()
        files[relative] = {'sha256': digest(path.read_bytes()), 'mtime_ns': stat.st_mtime_ns,
                           'birthtime_ns': getattr(stat, 'st_birthtime_ns', None)}
    return {'files': files, 'excluded_body_aliases_never_read': sorted(excluded)}


def audit(source):
    plan = read_json(source / 'plan.json')
    old_snapshots = {s['candidate_id']: s for s in read_json(source / 'snapshots.json')['records']}
    failures, candidates, discoveries, additions = [], [], {}, {}
    for row in plan['candidates']:
        cid = row['candidate_id']
        if not row['usable_website']:
            continue
        discovery = read_json(source / f'discovery/candidates/{cid}.json')
        if not discovery['discovery_failure']:
            continue
        packet = read_json(source / f'private/packets/{cid}.json')
        fixed = copy.deepcopy(discovery)
        candidate_rows = []
        for index, old in enumerate(discovery['queries']):
            if old['state'] != 'failure':
                continue
            body_path = source / 'discovery' / Path(old['final_receipt']).with_suffix('.bin')
            body = body_path.read_bytes()
            if digest(body) != old['raw_response_sha256']:
                raise ValueError('affected response hash changed')
            receipt = read_json(source / 'discovery' / old['final_receipt'])
            if receipt['http_status'] != 200 or receipt['error']:
                correction = {'state': 'failure', 'category': 'G', 'reason': 'official_http_failure'}
            else:
                correction = revalidate(body, row['website_url'], old['requested_timestamp'], plan['snapshot_cutoff'])
            correction = json.loads(encoded(correction))
            item = {'candidate_id': cid, 'frozen_website_url': row['website_url'],
                'requested_timestamp': old['requested_timestamp'], 'request_key': old['request_key'],
                'returned_archived_url': correction.get('returned_archived_url'),
                'returned_original_url': correction.get('returned_original_url'),
                'returned_timestamp': correction.get('returned_timestamp'), 'returned_status': correction.get('returned_status'),
                'failure_reason': old['reason'], **correction.get('comparison', {}),
                'also_has_prior_valid_snapshot': bool(discovery['snapshot_ids']),
                'category': correction['category'], 'category_label': CATEGORIES[correction['category']],
                'resolution': correction['state'], 'resolution_reason': correction['reason'],
                'source_receipt': 'discovery/' + old['final_receipt'], 'raw_response_sha256': digest(body)}
            failures.append(item)
            candidate_rows.append(item)
            fixed['queries'][index] = {**old, **correction, 'original_failure_reason': old['reason'],
                                      'closure_rule': VERSION}
            if correction['state'] == 'confirmed':
                identity = 'snapshot_' + digest(encoded([correction['confirmed_snapshot_timestamp'],
                                                         website_key(correction['original_target_url'])]))
                if identity.removeprefix('snapshot_') not in fixed['snapshot_ids']:
                    fixed['snapshot_ids'].append(identity.removeprefix('snapshot_'))
                if identity not in additions:
                    additions[identity] = {
                        'candidate_id': identity,
                        # Transport binds to the exact returned original. The separate
                        # candidate certificate retains the actual frozen website.
                        'original_website_url': correction['original_target_url'],
                        'requested_archive_timestamp': old['requested_timestamp'],
                        'confirmed_snapshot_timestamp': correction['confirmed_snapshot_timestamp'],
                        'exact_wayback_snapshot_url': correction['exact_wayback_snapshot_url'],
                        'candidate_bindings': [], 'discovery_receipts': [],
                        'transport_binding_role': 'official_returned_target_after_closure_binding_validation'}
                record = additions[identity]
                if cid not in record['candidate_bindings']:
                    record['candidate_bindings'].append(cid)
                record['discovery_receipts'].append({'candidate_id': cid, 'request_key': old['request_key'],
                    'requested_timestamp': old['requested_timestamp'], 'raw_response_sha256': digest(body),
                    'returned_exact_url': correction['exact_wayback_snapshot_url'], 'frozen_website_url': row['website_url'],
                    'closure_rule': VERSION, 'binding_comparison': correction['comparison']})
        remaining = any(q['state'] == 'failure' for q in fixed['queries'])
        fixed.update(discovery_failure=remaining, bounded_lookup_complete=not remaining,
                     state='discovery_failure' if remaining else 'confirmed' if fixed['snapshot_ids'] else 'no_snapshot_in_bounded_lookup',
                     closure_rule=VERSION, original_discovery_sha256=digest(encoded(discovery)))
        discoveries[cid] = fixed
        candidates.append({'candidate_id': cid, 'frozen_website_url': row['website_url'], 'prior_state': packet['state'],
            'prior_confirmed_snapshots': len(discovery['snapshot_ids']), 'corrected_confirmed_snapshots': len(fixed['snapshot_ids']),
            'failed_responses': len(candidate_rows), 'resolved_responses': sum(x['resolution'] == 'confirmed' for x in candidate_rows),
            'remaining_failed_responses': sum(x['resolution'] == 'failure' for x in candidate_rows),
            'categories': dict(Counter(x['category'] for x in candidate_rows)),
            'all_discovery_failures_resolved': not remaining,
            'exclusive_discovery_review': ('stored_official_response_establishes_snapshot' if fixed['snapshot_ids']
                else 'technically_unresolved' if remaining else 'bounded_no_snapshot')})
    if len(candidates) != 48 or sum(c['prior_state'] == 'discovery_failure' for c in candidates) != 13:
        raise ValueError('closure frame differs from approved 48/13')
    return plan, failures, candidates, discoveries, additions, old_snapshots


def prepare(source, output, instruction):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'audit_seal.json').exists():
        verify_files(output, read_json(output / 'audit_seal.json')['files'])
        print(json.dumps(read_json(output / 'closure_plan.json'), indent=2), flush=True)
        return
    print('Verifying frozen discovery inputs; no network requests.', flush=True)
    verify_inputs(source)
    verify_files(source, read_json(source / 'acquisition_freeze.json')['files'])
    verify_files(source, read_json(source / 'review_delivery_manifest.json')['files'])
    plan, failures, candidates, discoveries, additions, old = audit(source)
    quarantine = read_json(source / 'security/quarantine_receipt.json')
    if quarantine['snapshot_id'] in additions:
        raise ValueError('quarantined snapshot cannot be reacquired')
    save(output, 'rule.json', {'version': VERSION, 'route_patterns': ROUTES, 'tracking_keys': sorted(TRACKING),
        'rule': 'Same exact normalized host/port; complete ASCII path may differ only by case inside listed routes. All slug characters and numeric IDs remain. Only percent encoding of known tracking VALUES may normalize; never drop parameters or change values.',
        'registered_domain_method': 'Reporting only: observed .com/.org suffixes; all other suffixes unknown. Host equality, not registered-domain equality, is required.',
        'cross_domain_allowed': False, 'path_or_query_blanket_ignoring_allowed': False})
    write(output / 'user_instruction.txt', instruction.read_bytes(), immutable=True)
    for cid, discovery in discoveries.items():
        save(output, f'discovery/candidates/{cid}.json', discovery)
    save(output, 'failure_responses.json', failures)
    table(output, 'failure_responses.csv', failures)
    save(output, 'affected_candidates.json', candidates)
    table(output, 'affected_candidates.csv', candidates)
    table(output, 'exclusive_13_review.csv', [c for c in candidates if c['prior_state'] == 'discovery_failure'])
    save(output, 'corrected_snapshot_additions.json', {'records': sorted(additions.values(), key=lambda x: x['candidate_id'])})
    missing = [record for sid, record in additions.items() if sid not in old]
    save(output, 'supplemental_snapshots.json', {'schema': VERSION,
        'scope': 'Only previously rejected, now independently bound exact identities lacking any stored response in the completed clean acquisition.',
        'rule_sha256': digest((output / 'rule.json').read_bytes()),
        'records': sorted(missing, key=lambda x: x['candidate_id'])})
    save(output, 'source_preservation_before.json', source_inventory(source))
    summary = {'source_run': source.relative_to(REPOSITORY).as_posix() if source.is_relative_to(REPOSITORY) else str(source),
        'candidate_count': 1226, 'affected_candidates': 48, 'failed_candidate_query_responses': len(failures),
        'unique_failed_queries': len({r['request_key'] for r in failures}),
        'category_response_counts': dict(Counter(r['category'] for r in failures)),
        'fully_resolved_candidates': sum(c['all_discovery_failures_resolved'] for c in candidates),
        'remaining_discovery_failure_candidates': sum(not c['all_discovery_failures_resolved'] for c in candidates),
        'resolved_candidate_query_responses': sum(r['resolution'] == 'confirmed' for r in failures),
        'newly_confirmed_unique_identities': len(additions), 'already_stored_unique_identities': len(additions) - len(missing),
        'missing_raw_responses_requiring_bounded_supplement': len(missing),
        'discovery_network_requests': 0, 'fetch_settings': plan['fetch'],
        'supplement_authorization': 'User permits new requests only when stored responses are genuinely insufficient; these exact bodies were absent because their discovery binding was previously rejected. No new discovery or alternate captures.',
        'openai_allowed': False, 'google_allowed': False, 'outscraper_allowed': False}
    save(output, 'closure_plan.json', summary)
    files = {p.relative_to(output).as_posix(): digest(p.read_bytes()) for p in output.rglob('*') if p.is_file()}
    save(output, 'audit_seal.json', {'files': files, 'discovery_complete_before_supplemental_fetch': True})
    print(json.dumps(summary, indent=2), flush=True)


def supplemental_raw(output, blocked):
    root = output / 'raw_retrieval'
    manifest_body = (output / 'supplemental_snapshots.json').read_bytes()
    records = json.loads(manifest_body)['records']
    if not records:
        return {}, {'files': {}, 'responses_429': 0, 'raw_content_successes': 0}
    if (root / 'input_manifest.json').read_bytes() != manifest_body:
        raise ValueError('supplemental raw input differs from sealed exact identities')
    pointer = read_json(root / 'latest_manifest.json')
    manifest_path = root / 'executions' / pointer['execution'] / 'manifest.json'
    if digest(manifest_path.read_bytes()) != pointer['manifest_sha256']:
        raise ValueError('supplemental raw execution manifest changed')
    manifest = read_json(manifest_path)
    if manifest['input_manifest_sha256'] != digest(manifest_body) or manifest['input_count'] != len(records):
        raise ValueError('supplemental count or input changed')
    for relative, expected in manifest['files'].items():
        if expected in blocked:
            raise ValueError('supplement unexpectedly contains a blocked body; refuse to read it')
    verify_files(root, manifest['files'])
    expected_settings = read_json(output / 'closure_plan.json')['fetch_settings']
    if any(manifest['settings'].get(k) != v for k, v in expected_settings.items()):
        raise ValueError('supplemental fetch settings changed')
    verified = {}
    for record in records:
        sid, stamp = record['candidate_id'], record['confirmed_snapshot_timestamp']
        state = read_json(root / f'state/{sid}/{stamp}.json')
        if state['input_record_sha256'] != digest(encoded(record)) or not state['history']:
            raise ValueError('supplemental state binding changed')
        terminal = read_json(root / state['history'][-1])
        if any(state[k] != terminal[k] for k in ('state', 'reason', 'sha256', 'http_status')):
            raise ValueError('supplemental terminal state changed')
        for relative in state['history']:
            attempt = read_json(root / relative)
            if relative not in manifest['files'] or attempt['candidate_id'] != sid or attempt['confirmed_snapshot_timestamp'] != stamp:
                raise ValueError('supplemental attempt binding changed')
            start = read_json(root / Path(relative).parent / 'started.json')
            if start['record'] != record:
                raise ValueError('supplemental started record changed')
        if state['state'] == 'success':
            folder = root / 'raw' / sid / stamp
            if state['sha256'] in blocked or any(sha in blocked for sha in state['files'].values()):
                raise ValueError('blocked body cannot be opened for verification')
            verify_files(folder, state['files'])
            provenance = read_json(folder / 'provenance.json')
            if (state['sha256'] in blocked or provenance['sha256'] in blocked):
                raise ValueError('blocked body cannot enter parser')
            body = (folder / 'body.bin').read_bytes()
            if (digest(body) != state['sha256'] or provenance['sha256'] != state['sha256']
                    or not bound(record, provenance['final_url']) or provenance['http_status'] != 200
                    or provenance['input_record_sha256'] != digest(encoded(record))
                    or provenance['input_manifest_sha256'] != digest(manifest_body)):
                raise ValueError('supplemental preserved raw provenance failed')
            verified[sid] = {'state': 'success', 'reason': None, 'http_status': 200, 'raw_folder': folder,
                             'body': body, 'headers': read_json(folder / 'headers.json'), 'provenance': provenance}
        elif state['state'] in {'permanent_failure', 'retryable_failure'}:
            verified[sid] = {k: state[k] for k in ('state', 'reason', 'http_status')}
        else:
            raise ValueError('unfinished supplemental fetch')
    return verified, manifest


def artifact(root, path, label):
    return {'root': label, 'path': path.relative_to(root).as_posix(), 'sha256': digest(path.read_bytes())}


def identity(query):
    return 'snapshot_' + digest(encoded([query['confirmed_snapshot_timestamp'], website_key(query['original_target_url'])]))


def certify_query(source, row, query, cutoff):
    """Independently reconstruct candidate -> receipt -> exact identity link."""
    if query['website_url'] != row['website_url']:
        raise ValueError('candidate website differs from official query')
    if query['request_key'] != digest(encoded([row['website_url'], query['requested_timestamp']])):
        raise ValueError('official cache association differs from candidate/date')
    receipt_path = source / 'discovery' / query['final_receipt']
    body_path = receipt_path.with_suffix('.bin')
    receipt, body = read_json(receipt_path), body_path.read_bytes()
    expected_url = 'https://archive.org/wayback/available?' + urlencode({
        'url': row['website_url'], 'timestamp': query['requested_timestamp']})
    if (receipt['request_url'] != expected_url or receipt['final_url'] != expected_url
            or receipt['website_url'] != row['website_url']
            or receipt['requested_timestamp'] != query['requested_timestamp']):
        raise ValueError('official request receipt association changed')
    if receipt['body_sha256'] != digest(body) or query['raw_response_sha256'] != digest(body):
        raise ValueError('official body hash mismatch')
    if receipt['http_status'] != 200 or receipt['error']:
        raise ValueError('failed official receipt cannot certify evidence')
    checked = revalidate(body, row['website_url'], query['requested_timestamp'], cutoff)
    if (checked['state'] != 'confirmed' or identity(checked) != identity(query)
            or checked['exact_wayback_snapshot_url'] != query['exact_wayback_snapshot_url']):
        raise ValueError('official identity cannot be independently reproduced')
    return {'candidate_id': row['candidate_id'], 'frozen_website_url': row['website_url'],
        'frozen_details': artifact(source, source / row['details_file'], 'source'),
        'official_response': artifact(source, body_path, 'source'),
        'official_response_metadata': artifact(source, receipt_path, 'source'),
        'requested_timestamp': query['requested_timestamp'], 'returned_timestamp': checked['confirmed_snapshot_timestamp'],
        'returned_archived_url': checked['returned_archived_url'], 'returned_original_url': checked['original_target_url'],
        'exact_snapshot_url': checked['exact_wayback_snapshot_url'], 'binding_rule': VERSION,
        'binding_comparison': checked['comparison']}


def provenance_page(source, output, row, sid, raw, queries, details, cutoff):
    certificates = [certify_query(source, row, q, cutoff) for q in queries if q['state'] == 'confirmed' and identity(q) == sid]
    if not certificates:
        raise ValueError('no official candidate-bound certificate for retained raw page')
    actual = parse_replay(raw['provenance']['final_url'])
    for certificate in certificates:
        if (actual['timestamp'] != certificate['returned_timestamp']
                or website_key(actual['original_url']) != website_key(certificate['returned_original_url'])):
            raise ValueError('raw replay differs from officially confirmed exact identity')
    folder = raw['raw_folder']
    raw_root, label = (source, 'source') if folder.is_relative_to(source) else (output, 'closure')
    chain = {'candidate_id': row['candidate_id'], 'frozen_website_url': row['website_url'],
        'official_discovery': certificates,
        'preserved_raw_body': artifact(raw_root, folder / 'body.bin', label),
        'raw_crawler_provenance': artifact(raw_root, folder / 'provenance.json', label),
        'raw_headers': artifact(raw_root, folder / 'headers.json', label),
        'exact_final_replay_url': raw['provenance']['final_url'], 'snapshot_id': sid,
        'text_origin': 'local deterministic extraction from a preserved raw archived HTTP response'}
    provenance = {**raw['provenance'], 'snapshot_id': sid, 'candidate_binding': row['candidate_id'],
        'candidate_details_sha256': row['details_sha256'], 'frozen_candidate_website_url': row['website_url'],
        'closure_provenance_chain': chain}
    parsed = parse_twice(raw, details, provenance)
    if parsed['parse_status'] == 'parser_exception':
        raise ValueError('closure parser exception requires explicit repair')
    relative = f'private/parsed/{sid}/{row["candidate_id"]}.json'
    parsed_sha = save(output, relative, parsed)
    return {'parsed_file': relative, 'parsed_sha256': parsed_sha, 'snapshot_id': sid,
            'provenance_chain': chain, **parsed}


def classifier_item(page):
    if not page['substantive_parse'] or page['parse_status'] != 'success':
        raise ValueError('unusable text must not be exposed as archive evidence')
    chain = {**page['provenance_chain'], 'verified_parser_output': {
        'root': 'closure', 'path': page['parsed_file'], 'sha256': page['parsed_sha256']}}
    return {'snapshot_id': page['snapshot_id'], 'scope': page['scope'], 'provenance_chain': chain,
        'text': page['visible_text'], 'text_sha256': page['visible_text_sha256'],
        'fragment': {'start': 0, 'end': len(page['visible_text']), 'text_sha256': page['visible_text_sha256'],
                     'raw_sha256': page['raw_sha256'], 'parser_output_sha256': page['parsed_sha256']},
        **{key: page[key] for key in ('matches', 'service_evidence', 'contrary_evidence', 'opening_claims',
                                     'identity_ambiguities', 'active_by_eligible', 'active_by_year', 'explicit_opening_year')}}


def gate(counts, verified):
    # Conservative operational meaning of "small", declared before final page
    # inspection. Partial history gaps are counted separately, never erased.
    denominator = counts['usable_websites']
    unresolved_fraction = counts['unresolved_technical_candidates_without_text'] / denominator
    transport_fraction = counts['terminal_raw_technical_failure_snapshots'] / counts['unique_confirmed_snapshots']
    return {'scientifically_ready_for_phase2_review': (
        all(verified.values()) and unresolved_fraction <= 0.05 and
        transport_fraction <= 0.05 and
        counts['source_and_supplement_terminal_429'] == 0 and
        counts['candidates_with_archive_text'] > counts['unresolved_technical_candidates_without_text']),
        'small_residual_rule': 'At most 5% of website-bearing candidates lack text due unresolved technical errors. Partial-history failures remain separately enumerated.',
        'unresolved_technical_fraction_without_text': unresolved_fraction,
        'terminal_raw_technical_failure_fraction': transport_fraction,
        'systemic_transport_check': 'At most 5% of exact snapshots end in technical retrieval/custody failure; no remaining terminal 429.',
        'phase2_execution_authorized': False, 'classification_performed': False,
        'coverage_100_percent_required': False,
        'supersedes': 'Prior any-query-failure and any-quarantine gate blockers are superseded by the latest closure instruction when exclusions and complete retained provenance are verified.'}


def finalize(source, output):
    verify_files(output, read_json(output / 'audit_seal.json')['files'])
    implementation = read_json(output / 'closure_implementation.json')
    verify_files(REPOSITORY, implementation['files'])
    print('Verifying discovery, all admissible raw hashes, and original packets.', flush=True)
    plan, snapshots, details, original_discoveries = verify_inputs(source)
    verify_files(source, read_json(source / 'acquisition_freeze.json')['files'])
    raw, raw_manifest = verify_raw(source, snapshots, plan)
    _, failures, affected, corrected, additions, originals = audit(source)
    if encoded(failures) != (output / 'failure_responses.json').read_bytes():
        raise ValueError('failure audit not reproducible')
    for cid, proof in corrected.items():
        if encoded(proof) != (output / f'discovery/candidates/{cid}.json').read_bytes():
            raise ValueError('corrected discovery not reproducible')
    quarantine = read_json(source / 'security/quarantine_receipt.json')
    blocked = set(quarantine['blocked_body_sha256'])
    supplement, supplement_manifest = supplemental_raw(output, blocked)
    raw.update(supplement)
    records = {**originals, **{sid: record for sid, record in additions.items() if sid not in originals}}
    by_candidate = defaultdict(set)
    for record in list(originals.values()) + list(additions.values()):
        for cid in record['candidate_bindings']:
            by_candidate[cid].add(record['candidate_id'])
    packets, transitions, quarantine_rows = [], [], []
    spans, old_parses_verified = 0, 0
    for index, row in enumerate(plan['candidates'], 1):
        cid = row['candidate_id']
        old_packet = read_json(source / f'private/packets/{cid}.json')
        # Verify every original saved parser output against original raw bytes,
        # including diagnostic rejects; never read the quarantined object.
        for old_page in old_packet['archive_pages']:
            old_raw = raw[old_page['snapshot_id']]
            if old_raw['state'] != 'success' or old_page['raw_sha256'] in blocked:
                raise ValueError('original packet contains excluded raw text')
            old_parsed = parse_twice(old_raw, details[cid], old_page['provenance'])
            old_bytes = (source / old_page['parsed_file']).read_bytes()
            if encoded(old_parsed) != old_bytes or digest(old_bytes) != old_page['parsed_sha256']:
                raise ValueError('original parser output not reproducible')
            old_parses_verified += 1
        proof = corrected.get(cid, original_discoveries.get(cid))
        pages, outcomes = [], []
        for sid in sorted(by_candidate[cid]):
            page = raw[sid]
            outcomes.append({'snapshot_id': sid, 'timestamp': records[sid]['confirmed_snapshot_timestamp'],
                'state': page['state'], 'reason': page['reason'],
                'http_status': page.get('http_status', page.get('provenance', {}).get('http_status'))})
            if page['state'] == 'success':
                if page['provenance']['sha256'] in blocked or sid == quarantine['snapshot_id']:
                    raise ValueError('quarantined object cannot enter closure packet')
                parsed = provenance_page(source, output, row, sid, page, proof['queries'], details[cid], plan['snapshot_cutoff'])
                spans += validate_claim_spans(parsed)
                pages.append(parsed)
        state = candidate_state(row, proof, pages, outcomes)
        active = [p for p in pages if p['active_by_eligible']]
        packet = {'schema': VERSION, 'candidate_id': cid, 'details': details[cid],
            'frozen_details': artifact(source, source / row['details_file'], 'source'),
            'website_url': row['website_url'], 'usable_website': row['usable_website'],
            'excluded_website_reason': row.get('no_website_reason'), 'state': state,
            'original_packet': artifact(source, source / f'private/packets/{cid}.json', 'source'),
            'discovery': (artifact(output if cid in corrected else source,
                (output if cid in corrected else source) / f'discovery/candidates/{cid}.json',
                'closure' if cid in corrected else 'source') if proof else None),
            'bounded_discovery_failure': bool(proof and proof['discovery_failure']),
            'bounded_lookup_complete': bool(proof and proof['bounded_lookup_complete']),
            'snapshot_ids': sorted(by_candidate[cid]), 'raw_snapshot_outcomes': outcomes,
            'archive_pages': pages, 'classifier_archive_evidence': [classifier_item(p) for p in pages if p['substantive_parse']],
            'substantive_page_count': sum(p['substantive_parse'] for p in pages),
            'scope_categories': sorted({p['scope'] for p in pages if p['substantive_parse']}),
            'active_by_year': min((p['active_by_year'] for p in active), default=None),
            'active_by_sources': [p['parsed_file'] for p in active], **opening_summary(pages),
            'historical_rule': old_packet['historical_rule'], 'classification_performed': False}
        save(output, f'private/packets/{cid}.json', packet)
        packets.append(packet)
        if cid in corrected:
            transitions.append({'candidate_id': cid, 'old_state': old_packet['state'], 'new_state': state,
                'also_has_confirmed_snapshot': bool(by_candidate[cid]),
                'remaining_failed_queries': sum(q['state'] == 'failure' for q in proof['queries']),
                'has_archive_text': bool(packet['substantive_page_count'])})
        if quarantine['snapshot_id'] in by_candidate[cid]:
            quarantine_rows.append({'candidate_id': cid, 'quarantined_snapshot_id': quarantine['snapshot_id'],
                'timestamp': quarantine['capture_timestamp'], 'permanently_unavailable': True,
                'alternative_confirmed_snapshots': sorted(by_candidate[cid] - {quarantine['snapshot_id']}),
                'alternative_valid_archived_evidence': bool(packet['classifier_archive_evidence']),
                'alternative_text_snapshot_ids': [p['snapshot_id'] for p in pages if p['substantive_parse']],
                'old_state': old_packet['state'], 'new_state': state,
                'evidence_consequence': 'Excluded quarantined identity entirely from parser/classifier text; other exact snapshots remain independently admissible.'})
        if index % 100 == 0:
            print(f'Closure packets verified and written: {index}/1226', flush=True)
    def business(p):
        return any(p.get('matches', {}).values()) or bool(p.get('service_evidence') or p.get('opening_claims') or p.get('contrary_evidence'))
    # Distinguish loss of all candidate text from partial-history gaps. This
    # enumerates encoding failures too; no candidate or response failure is hidden.
    technical_rows = []
    for p in packets:
        reasons = []
        if p['bounded_discovery_failure']:
            reasons.append('remaining_discovery_validation_failure')
        reasons.extend('raw:' + str(s['reason']) for s in p['raw_snapshot_outcomes'] if technical_raw_failure(s))
        reasons.extend('parser:' + str(page['reason']) for page in p['archive_pages']
                       if page['reason'] in {'invalid_text_encoding', 'parser_exception'})
        if reasons:
            technical_rows.append({'candidate_id': p['candidate_id'], 'state': p['state'],
                'has_archive_text': bool(p['substantive_page_count']), 'technical_reasons': sorted(set(reasons)),
                'consequence': 'partial_history_gap' if p['substantive_page_count'] else 'no_admissible_text_technical_gap'})
    save(output, 'candidate_state_transitions.json', transitions)
    table(output, 'candidate_state_transitions.csv', transitions)
    save(output, 'quarantine_impact.json', quarantine_rows)
    table(output, 'quarantine_impact.csv', quarantine_rows)
    save(output, 'unresolved_technical_candidates.json', technical_rows)
    table(output, 'unresolved_technical_candidates.csv', technical_rows)
    print('Verifying preservation of completed acquisition and earlier runs.', flush=True)
    preservation = verify_preservation(source)
    if not preservation['unchanged']:
        raise ValueError('prior protected artifacts changed')
    current_inventory = source_inventory(source)
    if encoded(current_inventory) != (output / 'source_preservation_before.json').read_bytes():
        raise ValueError('completed clean acquisition files or recorded times changed')
    save(output, 'preservation_verification.json', {'prior_runs': preservation,
        'completed_clean_acquisition_unchanged': True, 'completed_clean_files_checked': len(current_inventory['files']),
        'excluded_body_aliases_never_read': current_inventory['excluded_body_aliases_never_read']})
    all_pages = [page for p in packets for page in p['archive_pages']]
    counts = {'candidates': len(packets), 'usable_websites': sum(p['usable_website'] for p in packets),
        'candidate_states': {state: sum(p['state'] == state for p in packets) for state in STATES},
        'candidates_with_confirmed_snapshots': sum(bool(p['snapshot_ids']) for p in packets),
        'candidates_with_archive_text': sum(bool(p['substantive_page_count']) for p in packets),
        'candidates_with_business_evidence': sum(any(business(x) for x in p['archive_pages']) for p in packets),
        'scope_candidates_overlapping': {scope: sum(scope in p['scope_categories'] for p in packets)
                                        for scope in ('outlet_specific', 'broader_brand', 'unclear')},
        'active_by_candidates': sum(p['active_by_year'] is not None for p in packets),
        'explicit_opening_year_candidates': sum(p['explicit_verified_opening_year'] is not None for p in packets),
        'fully_resolved_of_48_candidates': sum(c['all_discovery_failures_resolved'] for c in affected),
        'resolved_candidate_query_failures': sum(r['resolution'] == 'confirmed' for r in failures),
        'remaining_discovery_failure_candidates_overlapping': sum(p['bounded_discovery_failure'] for p in packets),
        'remaining_discovery_failed_candidate_queries': sum(r['resolution'] == 'failure' for r in failures),
        'remaining_ambiguous_response_failures': sum(r['resolution'] == 'failure' and r['category'] in {'B', 'D'} for r in failures),
        'remaining_mismatch_or_status_response_failures': sum(r['resolution'] == 'failure' and r['category'] in {'E', 'F'} for r in failures),
        'unresolved_technical_candidates_without_text': sum(not r['has_archive_text'] for r in technical_rows),
        'candidates_with_partial_technical_history_gaps': sum(r['has_archive_text'] for r in technical_rows),
        'candidates_with_any_unresolved_technical_component': len(technical_rows),
        'technical_reason_candidate_counts_overlapping': dict(Counter(reason for r in technical_rows for reason in r['technical_reasons'])),
        'all_parser_rejection_reasons_candidate_pages': dict(Counter(p['reason'] for p in all_pages if not p['substantive_parse'])),
        'unique_confirmed_snapshots': len(records), 'supplemental_exact_snapshots': len(supplement),
        'supplemental_raw_http_successes': supplement_manifest['raw_content_successes'],
        'supplemental_raw_failure_reasons': dict(Counter(r['reason'] for r in supplement.values() if r['state'] != 'success')),
        'verified_saved_responses': sum(r['state'] == 'success' for r in raw.values()),
        'terminal_raw_technical_failure_snapshots': sum(technical_raw_failure(r) for r in raw.values()),
        'source_and_supplement_429_responses': raw_manifest['responses_429'] + supplement_manifest['responses_429'],
        'source_and_supplement_terminal_429': sum(r.get('http_status') == 429 for r in raw.values()),
        'quarantined_snapshots_excluded': 1, 'quarantine_affected_candidates': len(quarantine_rows),
        'prior_parser_outputs_reproduced': old_parses_verified, 'final_candidate_page_parses': len(all_pages),
        'traceable_claim_spans_verified': spans,
        'classifier_visible_full_text_fragments_verified': sum(len(p['classifier_archive_evidence']) for p in packets)}
    if counts['candidates'] != 1226 or sum(counts['candidate_states'].values()) != 1226:
        raise ValueError('final candidate states do not partition fixed frame')
    for field in ('candidates_with_archive_text', 'candidates_with_business_evidence'):
        counts[field + '_fraction_of_websites'] = counts[field] / counts['usable_websites']
        counts[field + '_fraction_of_all_candidates'] = counts[field] / counts['candidates']
    verified = {'complete_candidate_to_fragment_provenance': True, 'all_retained_hashes_verified': True,
        'original_and_final_parser_outputs_reproducible': True, 'no_timestamp_only_active_by_or_opening_claim': True,
        'quarantined_body_and_derived_text_excluded': True, 'prior_artifacts_preserved': True,
        'all_1226_packets_frozen': True}
    report = {'schema': VERSION, 'counts': counts, 'verification': verified, 'promotion': gate(counts, verified),
        'source_root': str(source), 'closure_root': str(output),
        'discovery_network_calls': 0, 'classification_performed': False, 'stop_for_final_review': True,
        'limitations': [
            'Bounded no-snapshot is not exhaustive historical archive absence.',
            'Text coverage and identifiable-business-evidence coverage are separate; neither is a model classification.',
            'Outlet, brand and unclear candidate counts overlap across pages.',
            'All partial technical history gaps, including frozen-parser encoding rejects, remain enumerated separately from candidates with no admissible text.',
            'The quarantined received object remains permanently unavailable, so the original all-received-body custody claim remains false; every RETAINED evidence body is verified.',
            'Existing source acquisition and its previous stricter preflight report remain unchanged. This version applies the latest user-authorized promotion criteria.']}
    save(output, 'closure_preflight.json', report)
    write(output / 'REVIEW_REPORT.md', render(report).encode('utf-8'), immutable=True)
    files = {p.relative_to(output).as_posix(): digest(p.read_bytes()) for p in output.rglob('*')
             if p.is_file() and p.name not in {'closure_freeze.json', 'verification_receipt.json'}
             and 'raw_retrieval' not in p.relative_to(output).parts}
    verify_files(output, files)
    save(output, 'closure_freeze.json', {'schema': VERSION, 'files': files, 'packet_count': 1226,
        'source_acquisition_freeze_sha256': digest((source / 'acquisition_freeze.json').read_bytes()),
        'source_raw_manifest_sha256': digest((source / 'raw_postfetch_manifest.json').read_bytes()),
        'supplemental_raw_manifest_sha256': digest(encoded(supplement_manifest)),
        'classification_performed': False, 'stop_for_final_review': True})
    print(json.dumps({'counts': counts, 'promotion': report['promotion']}, indent=2), flush=True)


def render(report):
    c = report['counts']
    lines = ['# Final clean-archive closure review', '',
        'All 1,226 evidence packets are frozen. Phase 2 has not been executed.', '',
        '| Candidate state | Count |', '|---|---:|']
    lines += [f'| {state} | {count} |' for state, count in c['candidate_states'].items()]
    lines += ['', f'Discovery validation resolved for **{c["fully_resolved_of_48_candidates"]}/48 candidates**, '
        f'covering {c["resolved_candidate_query_failures"]} failed candidate/date responses. '
        f'{c["remaining_discovery_failure_candidates_overlapping"]} candidate retains an anomalous response and is explicitly preserved.', '',
        f'Archive text: **{c["candidates_with_archive_text"]}/1,088 ({c["candidates_with_archive_text_fraction_of_websites"]:.2%})**; '
        f'{c["candidates_with_archive_text_fraction_of_all_candidates"]:.2%} of all 1,226 candidates.', '',
        f'Identifiable business evidence: **{c["candidates_with_business_evidence"]}/1,088 ({c["candidates_with_business_evidence_fraction_of_websites"]:.2%})**; '
        f'{c["candidates_with_business_evidence_fraction_of_all_candidates"]:.2%} of all candidates.', '',
        f'Candidate evidence scopes (overlapping): {c["scope_candidates_overlapping"]}. '
        f'Eligible active-by: {c["active_by_candidates"]}; explicit opening year: {c["explicit_opening_year_candidates"]}.', '',
        f'Unresolved technical candidates without text: **{c["unresolved_technical_candidates_without_text"]}**. '
        f'Another **{c["candidates_with_partial_technical_history_gaps"]}** candidates have text but retain partial technical history gaps '
        f'({c["candidates_with_any_unresolved_technical_component"]} with any technical component in total). '
        'See `unresolved_technical_candidates.csv`; partial gaps are not recoded as no archive.', '',
        f'Technical reasons, overlapping candidate counts: {c["technical_reason_candidate_counts_overlapping"]}.', '',
        f'New exact raw requests: {c["supplemental_exact_snapshots"]} missing identities; '
        f'{c["supplemental_raw_http_successes"]} successes. No new discovery, Google, Outscraper or model calls. '
        f'All retained responses: {c["verified_saved_responses"]}; all 429 responses: {c["source_and_supplement_429_responses"]}.', '',
        f'One quarantined snapshot stays permanently excluded; {c["quarantine_affected_candidates"]} candidate bindings are detailed in `quarantine_impact.csv`. '
        'No blocked object was restored, opened, recovered or supplied to the parser/classifier.', '',
        f'Provenance verified for {c["classifier_visible_full_text_fragments_verified"]} visible-text fragments and '
        f'{c["traceable_claim_spans_verified"]} claim spans. Original parser outputs reproduced: {c["prior_parser_outputs_reproduced"]}. '
        'Every retained fragment links the candidate, frozen website, official bounded response, exact snapshot, saved raw body and deterministic parser output.', '',
        f'**Scientifically ready for Phase 2 approval: {report["promotion"]["scientifically_ready_for_phase2_review"]}.** '
        'This is a recommendation only. OpenAI remains disabled pending user review.', '',
        'Audit tables: `affected_candidates.csv` (48 candidates), `failure_responses.csv` (every failed response), '
        '`exclusive_13_review.csv`, `candidate_state_transitions.csv`, `quarantine_impact.csv`.', '']
    lines += [note + '\n' for note in report['limitations']]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'finalize'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.instruction is None:
            parser.error('--instruction is required for prepare')
        prepare(args.source.resolve(), args.output.resolve(), args.instruction)
    else:
        finalize(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
