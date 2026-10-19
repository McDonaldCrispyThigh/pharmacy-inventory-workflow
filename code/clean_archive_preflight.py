"""Offline final stage for the clean archive run; no provider or model calls.

Verifies sealed discovery and completed retrieval before deterministic parsing.
Every evidence claim retains character spans, a text digest and raw byte digest.
All outputs are immutable and may be independently reproduced by rerunning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from wayback_snapshot_fetcher import encoded, digest, read_json, write, write_json, bound
from clean_archive_discovery import Discovery
from run_clean_archive import unseal, verify_implementation, verify_snapshots
from clean_archive_parser import parse_archive_page

VERSION = 'clean-archive-preflight-v1'
REPOSITORY = Path(__file__).resolve().parents[1]
STATES = ('archive_content_available', 'snapshot_confirmed_but_fetch_failed',
          'archive_content_unusable', 'no_snapshot_in_bounded_lookup',
          'no_website', 'discovery_failure')


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path escapes its commitment root')
    return path


def verify_files(root, files):
    for relative, expected in files.items():
        if digest(checked_path(root, relative).read_bytes()) != expected:
            raise ValueError('committed artifact changed: ' + relative)


def verify_preservation(root, repository=REPOSITORY):
    inventory = read_json(root / 'preservation_before.json')
    changed = []
    for relative, expected in inventory.items():
        path = checked_path(repository, relative)
        if not path.is_file():
            changed.append({'path': relative, 'reason': 'missing'})
            continue
        stat = path.stat()
        actual = {'sha256': digest(path.read_bytes()), 'mtime_ns': stat.st_mtime_ns,
                  'birthtime_ns': getattr(stat, 'st_birthtime_ns', None)}
        differences = [key for key, value in expected.items() if actual.get(key) != value]
        if differences:
            changed.append({'path': relative, 'changed_fields': differences})
    return {'inventory_sha256': digest((root / 'preservation_before.json').read_bytes()),
            'checked_files': len(inventory), 'unchanged': not changed, 'changes': changed}


def verify_parser_implementation(root):
    manifest = read_json(root / 'parser_implementation_manifest.json')
    required = {'code/clean_archive_parser.py', 'code/clean_archive_preflight.py'}
    if not required <= set(manifest['files']):
        raise ValueError('parser implementation commitment is incomplete')
    verify_files(REPOSITORY, manifest['files'])


def verify_query_association(row, query):
    if query['website_url'] != row['website_url']:
        raise ValueError('discovery website binding changed')
    key = digest(encoded([row['website_url'], query['requested_timestamp']]))
    receipt = Path(query['final_receipt'])
    if query['request_key'] != key or receipt.parent.as_posix() != 'requests/' + key:
        raise ValueError('discovery query cache key or receipt association changed')


def verify_discovery(root):
    """Audit every positive and negative candidate/query association before fetch."""
    root = Path(root).resolve()
    verify_implementation(root)
    plan, snapshots = verify_snapshots(root)
    rows = plan['candidates']
    if len(rows) != 1226 or len({r['candidate_id'] for r in rows}) != 1226:
        raise ValueError('candidate frame must contain exactly 1226 unique IDs')
    if sum(bool(r['usable_website']) for r in rows) != 1088:
        raise ValueError('usable website denominator changed')
    details = {}
    discoveries = {}
    verifier = Discovery(root / 'discovery', plan)
    discovery_manifest = read_json(root / 'discovery_manifest.json')
    snapshots_by_id = {r['candidate_id'].removeprefix('snapshot_'): r for r in snapshots}
    if len(snapshots_by_id) != len(snapshots):
        raise ValueError('duplicate unique snapshot identity')
    observed_bindings = defaultdict(set)
    for row in rows:
        ident = row['candidate_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', ident):
            raise ValueError('unsafe candidate ID')
        path = checked_path(root, row['details_file'])
        if digest(path.read_bytes()) != row['details_sha256']:
            raise ValueError('copied Details changed')
        payload = unseal(path)
        if payload['place_id'] != ident or payload['details'].get('websiteUri') != row['website_url']:
            raise ValueError('Details candidate or website binding changed')
        details[ident] = payload['details']
        if not row['usable_website']:
            continue
        relative = 'discovery/candidates/' + ident + '.json'
        if relative not in discovery_manifest['files']:
            raise ValueError('candidate discovery not committed')
        proof = read_json(root / relative)
        if proof['candidate_id'] != ident:
            raise ValueError('discovery candidate binding changed')
        if [q['requested_timestamp'] for q in proof['queries']] != plan['schedule']:
            raise ValueError('bounded lookup schedule incomplete or changed')
        expected_ids = []
        for query in proof['queries']:
            verify_query_association(row, query)
            verifier.verify(query)  # This method only reads and re-parses local receipts.
            if query['state'] == 'confirmed':
                from wayback_snapshot_fetcher import website_key
                identity = digest(encoded([query['confirmed_snapshot_timestamp'],
                                           website_key(query['original_target_url'])]))
                if identity not in expected_ids:
                    expected_ids.append(identity)
                if identity not in snapshots_by_id:
                    raise ValueError('confirmed query omitted from snapshot manifest')
                observed_bindings[identity].add(ident)
            elif query['state'] not in {'no_snapshot', 'failure'}:
                raise ValueError('unknown discovery outcome')
        failure = any(q['state'] == 'failure' for q in proof['queries'])
        expected_state = ('discovery_failure' if failure else 'confirmed' if expected_ids
                          else 'no_snapshot_in_bounded_lookup')
        if (proof['snapshot_ids'] != expected_ids or proof['discovery_failure'] != failure
                or proof['bounded_lookup_complete'] != (not failure) or proof['state'] != expected_state):
            raise ValueError('discovery outcome differs from committed raw responses')
        discoveries[ident] = proof
    for identity, record in snapshots_by_id.items():
        if set(record['candidate_bindings']) != observed_bindings[identity]:
            raise ValueError('snapshot candidate bindings are incomplete')
        if len(record['candidate_bindings']) != len(set(record['candidate_bindings'])):
            raise ValueError('duplicate snapshot candidate binding')
    return plan, snapshots, details, discoveries


def verify_inputs(root):
    verify_parser_implementation(root)
    return verify_discovery(root)


def verify_raw(root, snapshots, plan=None):
    """Use sealed attempt results, not an unverified mutable state summary."""
    snapshot_sha = digest((root / 'snapshots.json').read_bytes())
    if not snapshots:
        marker = read_json(root / 'no_confirmed_snapshots.json')
        if marker != {'raw_fetch_count': 0}:
            raise ValueError('invalid explicit zero-fetch completion')
        return {}, {'files': {}, 'attempt_results': [], 'raw_content_successes': 0,
                    'successful_html_bodies': 0, 'responses_429': 0, 'execution': None}
    seal = read_json(root / 'raw_fetch_seal.json')
    if seal['snapshot_manifest_sha256'] != snapshot_sha:
        raise ValueError('raw fetch seal refers to another snapshot input')
    if not re.fullmatch(r'[a-f0-9]{32}', seal['execution']):
        raise ValueError('invalid raw execution ID')
    raw_root = root / 'raw_retrieval'
    manifest_path = raw_root / 'executions' / seal['execution'] / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != seal['manifest_sha256']:
        raise ValueError('raw retrieval manifest changed')
    manifest = read_json(manifest_path)
    if plan is not None and any(manifest['settings'].get(k) != v for k, v in plan['fetch'].items()):
        raise ValueError('raw fetch settings differ from the frozen plan')
    if (manifest['input_manifest_sha256'] != snapshot_sha or manifest['input_count'] != len(snapshots)
            or manifest['execution'] != seal['execution']):
        raise ValueError('raw manifest input binding changed')
    if (raw_root / 'input_manifest.json').read_bytes() != (root / 'snapshots.json').read_bytes():
        raise ValueError('raw retrieval copied input changed')
    verify_files(raw_root, manifest['files'])
    verified = {}
    all_attempts = {}
    for record in snapshots:
        ident, stamp = record['candidate_id'], record['confirmed_snapshot_timestamp']
        relative = f'raw/{ident}/{stamp}'
        state = read_json(raw_root / 'state' / ident / (stamp + '.json'))
        if state['input_record_sha256'] != digest(encoded(record)) or not state['history']:
            raise ValueError('snapshot state input or history is incomplete')
        for entry in state['history']:
            if entry not in manifest['files']:
                raise ValueError('snapshot state refers to uncommitted history')
            all_attempts[entry] = read_json(checked_path(raw_root, entry))
        terminal = read_json(checked_path(raw_root, state['history'][-1]))
        if terminal['candidate_id'] != ident or terminal['confirmed_snapshot_timestamp'] != stamp:
            raise ValueError('terminal attempt snapshot binding changed')
        for key in ('state', 'reason', 'sha256', 'http_status'):
            if terminal[key] != state[key]:
                raise ValueError('mutable snapshot state differs from sealed terminal attempt')
        if state['state'] not in {'success', 'permanent_failure', 'retryable_failure'}:
            raise ValueError('raw snapshot is unresolved')
        if state['state'] == 'success':
            required = {'body.bin', 'headers.json', 'request.json', 'response.json', 'provenance.json'}
            if not required <= set(state['files']):
                raise ValueError('successful raw commitment is incomplete')
            for name, expected in state['files'].items():
                if Path(name).name != name or manifest['files'].get(relative + '/' + name) != expected:
                    raise ValueError('successful raw state commitment changed')
            folder = raw_root / relative
            provenance = read_json(folder / 'provenance.json')
            body = (folder / 'body.bin').read_bytes()
            if (provenance['input_record_sha256'] != digest(encoded(record))
                    or provenance['input_manifest_sha256'] != snapshot_sha
                    or provenance['sha256'] != digest(body) or state['sha256'] != digest(body)
                    or not bound(record, provenance['final_url']) or provenance['http_status'] != 200):
                raise ValueError('successful raw identity, timestamp or body binding failed')
            verified[ident] = {'state': 'success', 'raw_folder': folder,
                               'provenance': provenance, 'body': body,
                               'headers': read_json(folder / 'headers.json'), 'reason': None}
        else:
            verified[ident] = {'state': state['state'], 'reason': state['reason'],
                               'http_status': state['http_status']}
    state_counts = dict(Counter(v['state'] for v in verified.values()))
    if state_counts != manifest['state_counts']:
        raise ValueError('raw manifest state counts differ from sealed attempts')
    if sum(v['state'] == 'success' for v in verified.values()) != manifest['raw_content_successes']:
        raise ValueError('raw manifest successful count changed')
    manifest = {**manifest, 'verified_429_responses_all_sealed_attempts': sum(
        hop['http_status'] == 429 for attempt in all_attempts.values() for hop in attempt['redirect_chain'])}
    return verified, manifest


def validate_claim_spans(parsed):
    text = parsed['visible_text']
    text_sha = digest(text.encode('utf-8'))
    if text and parsed['visible_text_sha256'] != text_sha:
        raise ValueError('parsed visible-text digest mismatch')
    count = 0
    def visit(value):
        nonlocal count
        if isinstance(value, dict):
            if {'start', 'end', 'text'} <= set(value):
                start, end = value['start'], value['end']
                if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                        or text[start:end] != value['text'] or value.get('raw_sha256') != parsed['raw_sha256']
                        or value.get('visible_text_sha256') != text_sha
                        or value.get('provenance_sha256') != parsed['provenance_sha256']):
                    raise ValueError('claim span is not traceable to the saved raw body and text')
                count += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for key in ('matches', 'service_evidence', 'contrary_evidence', 'opening_claims', 'identity_ambiguities'):
        visit(parsed[key])
    for key in ('service_evidence', 'contrary_evidence', 'opening_claims'):
        for claim in parsed[key]:
            for span_key in ('span', 'context_span'):
                if not isinstance(claim.get(span_key), dict) or not {'start', 'end', 'text'} <= set(claim[span_key]):
                    raise ValueError('evidence claim omits its required text span')
    if parsed['active_by_eligible'] and not any(x['eligible'] for x in parsed['service_evidence']):
        raise ValueError('active-by claim has no eligible service text')
    if parsed['explicit_opening_year'] is not None:
        if not any(x['eligible'] and x['year'] == parsed['explicit_opening_year'] for x in parsed['opening_claims']):
            raise ValueError('opening year has no eligible explicit text')
    return count


def parse_twice(raw, details, provenance):
    def attempt():
        try:
            return parse_archive_page(raw['body'], raw['headers'], details, provenance)
        except Exception as exc:
            return {'parse_status': 'parser_exception', 'reason': type(exc).__name__,
                    'raw_sha256': digest(raw['body']), 'substantive_parse': False,
                    'scope': 'unclear', 'active_by_eligible': False, 'explicit_opening_year': None,
                    'provenance': provenance}
    first, second = attempt(), attempt()
    if encoded(first) != encoded(second):
        raise ValueError('local parser output is not reproducible')
    if first['parse_status'] != 'parser_exception':
        if first['raw_sha256'] != digest(raw['body']):
            raise ValueError('parser raw digest differs from verified body')
        if first['provenance_sha256'] != digest(encoded(provenance)):
            raise ValueError('parser provenance digest differs from supplied provenance')
        validate_claim_spans(first)
    return first


def candidate_state(row, discovery, pages, raw_states):
    if not row['usable_website']:
        return 'no_website'
    if any(p['substantive_parse'] for p in pages):
        return 'archive_content_available'
    if any(r['state'] == 'success' for r in raw_states):
        return 'archive_content_unusable'
    if raw_states:
        return 'snapshot_confirmed_but_fetch_failed'
    return 'discovery_failure' if discovery['discovery_failure'] else 'no_snapshot_in_bounded_lookup'


def opening_summary(pages):
    eligible = [p for p in pages if p['explicit_opening_year'] is not None]
    years = sorted({p['explicit_opening_year'] for p in eligible})
    rejected_local = any(claim['scope'] == 'outlet_specific' and not claim['eligible']
                         for p in pages for claim in p.get('opening_claims', []))
    conflict = (len(years) > 1 or any(p.get('opening_year_state') == 'conflict' for p in pages)
                or bool(years and rejected_local))
    return {'opening_year_state': 'conflict' if conflict else 'single' if years else 'none',
            'explicit_verified_opening_year': years[0] if len(years) == 1 and not conflict else None,
            'opening_year_sources': [p['parsed_file'] for p in eligible],
            'all_opening_claim_sources': [p['parsed_file'] for p in pages if p.get('opening_claims')]}


def technical_raw_failure(state):
    # A received permanent 404/410 is reported separately from transport, access,
    # redirect-binding, rate-limit and other unresolved retrieval failures.
    return state['state'] != 'success' and state.get('reason') not in {'http_404', 'http_410'}


def promotion_assessment(counts):
    confirmed = counts['candidates_with_confirmed_snapshot']
    transport_fraction = counts['confirmed_candidates_without_content_with_technical_fetch_failure'] / confirmed if confirmed else 0
    parser_fraction = counts['confirmed_candidates_with_only_unusable_content'] / confirmed if confirmed else 0
    user_reasons = []
    if counts['bounded_discovery_failure_candidates']:
        user_reasons.append('one_or_more_bounded_lookups_failed')
    if transport_fraction >= 0.5:
        user_reasons.append('technical_transport_failures_dominate_confirmed_candidates')
    if parser_fraction >= 0.5:
        user_reasons.append('unusable_parsed_content_dominates_confirmed_candidates')
    if confirmed and not counts['candidates_with_substantive_content']:
        user_reasons.append('confirmed_snapshots_produced_no_substantive_content')
    conservative = []
    if counts['raw_snapshots_failed']:
        conservative.append('frozen_plan_holds_any_residual_raw_fetch_failure_for_review')
    if counts['candidate_page_parser_exceptions']:
        conservative.append('parser_exceptions_require_review')
    if not counts['preservation_unchanged']:
        user_reasons.append('prior_artifact_preservation_failed')
    return {'user_conditions_met': not user_reasons, 'user_condition_blockers': user_reasons,
            'technical_failure_dominance_threshold': 0.5,
            'technical_failure_definition': 'Any raw failure except a received permanent HTTP 404 or 410; those still remain failed fetches and are never no-snapshot outcomes.',
            'technical_no_content_fraction_of_confirmed_candidates': transport_fraction,
            'unusable_content_fraction_of_confirmed_candidates': parser_fraction,
            'conservative_review_hold': bool(conservative), 'conservative_review_hold_reasons': conservative,
            'recommend_all_1226_gpt_5_mini': not user_reasons,
            'recommendation_scope': 'Recommendation only; stop for user review. No model call is authorized by this report.',
            'dominance_rule_note': 'Dominance is operationalized as at least half of confirmed candidates; the stricter frozen plan hold is a review annotation, not a new approval prerequisite.'}


def build_preflight(root):
    root = Path(root).resolve()
    plan, snapshots, details, discoveries = verify_inputs(root)
    raw, raw_manifest = verify_raw(root, snapshots, plan)
    preservation = verify_preservation(root)
    parsed_by_candidate = defaultdict(list)
    records_by_candidate = defaultdict(list)
    files = {}
    def save(relative, value):
        write_json(root / relative, value, immutable=True)
        files[relative] = digest((root / relative).read_bytes())
    save('preservation_verification.json', preservation)
    span_count = 0
    for record in snapshots:
        snapshot_id = record['candidate_id']
        for ident in record['candidate_bindings']:
            records_by_candidate[ident].append(record)
            if raw[snapshot_id]['state'] != 'success':
                continue
            page = raw[snapshot_id]
            provenance = {**page['provenance'], 'snapshot_id': snapshot_id,
                          'candidate_binding': ident,
                          'candidate_details_sha256': next(r['details_sha256'] for r in plan['candidates'] if r['candidate_id'] == ident)}
            parsed = parse_twice(page, details[ident], provenance)
            if parsed['parse_status'] != 'parser_exception':
                span_count += validate_claim_spans(parsed)
            relative = f'private/parsed/{snapshot_id}/{ident}.json'
            save(relative, parsed)
            parsed_by_candidate[ident].append({'parsed_file': relative, 'parsed_sha256': files[relative],
                'raw_provenance_file': page['raw_folder'].relative_to(root).as_posix() + '/provenance.json',
                'raw_body_file': page['raw_folder'].relative_to(root).as_posix() + '/body.bin',
                'snapshot_id': snapshot_id, **parsed})
    packets = []
    for row in plan['candidates']:
        ident = row['candidate_id']
        pages = parsed_by_candidate[ident]
        records = records_by_candidate[ident]
        discovery = discoveries.get(ident)
        states = [raw[r['candidate_id']] for r in records]
        active = [p for p in pages if p['active_by_eligible']]
        state = candidate_state(row, discovery, pages, states)
        packet = {'schema': VERSION, 'candidate_id': ident, 'details_file': row['details_file'],
            'details_sha256': row['details_sha256'], 'details': details[ident],
            'state': state, 'website_url': row['website_url'], 'usable_website': row['usable_website'],
            'excluded_website_reason': row.get('no_website_reason'),
            'discovery_file': f'discovery/candidates/{ident}.json' if discovery else None,
            'bounded_discovery_failure': bool(discovery and discovery['discovery_failure']),
            'bounded_lookup_complete': bool(discovery and discovery['bounded_lookup_complete']),
            'snapshot_ids': [r['candidate_id'] for r in records],
            'raw_snapshot_outcomes': [{'snapshot_id': r['candidate_id'],
                **{k: raw[r['candidate_id']].get(k) for k in ('state', 'reason', 'http_status')}} for r in records],
            'archive_pages': pages, 'substantive_page_count': sum(p['substantive_parse'] for p in pages),
            'scope_categories': sorted({p['scope'] for p in pages if p['substantive_parse']}),
            'active_by_year': min((p['active_by_year'] for p in active), default=None),
            'active_by_sources': [p['parsed_file'] for p in active],
            **opening_summary(pages),
            'historical_rule': 'A capture timestamp alone never establishes an opening year; active-by requires eligible outlet service text.',
            'classification_performed': False}
        save(f'private/packets/{ident}.json', packet)
        packets.append(packet)
    pages = [p for group in parsed_by_candidate.values() for p in group]
    successful_pages = [p for p in pages if p['parse_status'] == 'success']
    substantive = [p for p in pages if p['substantive_parse']]
    def business_evidence(page):
        return (any(page.get('matches', {}).values()) or bool(page.get('service_evidence'))
                or bool(page.get('opening_claims')) or bool(page.get('contrary_evidence')))
    confirmed_packets = [p for p in packets if p['snapshot_ids']]
    unique_status = {p['snapshot_id'] for p in successful_pages}
    unique_substantive = {p['snapshot_id'] for p in substantive}
    counts = {'candidates': len(packets), 'usable_website_candidates': 1088,
        'returned_website_candidates': sum(bool(p['website_url']) for p in packets),
        'missing_website_candidates': sum(not p['website_url'] for p in packets),
        'rejected_website_candidates': sum(bool(p['website_url']) and not p['usable_website'] for p in packets),
        'no_usable_website_candidates': sum(not p['usable_website'] for p in packets),
        'candidates_with_confirmed_snapshot': len(confirmed_packets),
        'no_snapshot_in_bounded_lookup_candidates': sum(p['state'] == 'no_snapshot_in_bounded_lookup' for p in packets),
        'bounded_discovery_failure_candidates': sum(p['bounded_discovery_failure'] for p in packets),
        'discovery_failure_and_confirmed_snapshot_candidates': sum(p['bounded_discovery_failure'] for p in confirmed_packets),
        'discovery_failure_and_substantive_content_candidates': sum(p['bounded_discovery_failure'] and bool(p['substantive_page_count']) for p in packets),
        'unique_snapshots': len(snapshots), 'candidate_snapshot_pairs': sum(len(p['snapshot_ids']) for p in packets),
        'raw_successful_snapshots': sum(v['state'] == 'success' for v in raw.values()),
        'raw_snapshots_failed': sum(v['state'] != 'success' for v in raw.values()),
        'raw_technical_failure_snapshots': sum(technical_raw_failure(v) for v in raw.values()),
        'raw_permanent_404_or_410_snapshots': sum(v['state'] != 'success' and not technical_raw_failure(v) for v in raw.values()),
        'raw_permanent_failure_snapshots': sum(v['state'] == 'permanent_failure' for v in raw.values()),
        'raw_retryable_failure_snapshots': sum(v['state'] == 'retryable_failure' for v in raw.values()),
        'raw_terminal_429_snapshots': sum(v.get('http_status') == 429 for v in raw.values()),
        'raw_429_responses_this_execution': raw_manifest['responses_429'],
        'raw_429_responses_all_sealed_attempts': raw_manifest.get('verified_429_responses_all_sealed_attempts', 0),
        'raw_html_successful_snapshots': raw_manifest['successful_html_bodies'],
        'parsed_unique_pages': len(unique_status), 'substantive_unique_pages': len(unique_substantive),
        'candidate_page_parse_attempts': len(pages), 'candidate_page_parse_successes': len(successful_pages),
        'candidate_page_substantive_parses': len(substantive),
        'candidate_page_parser_exceptions': sum(p['parse_status'] == 'parser_exception' for p in pages),
        'candidates_with_substantive_content': sum(bool(p['substantive_page_count']) for p in packets),
        'candidates_with_identifiable_business_evidence': sum(any(business_evidence(page) for page in p['archive_pages']) for p in packets),
        'unique_pages_with_identifiable_business_evidence': len({p['snapshot_id'] for p in pages if business_evidence(p)}),
        'candidate_pages_with_identifiable_business_evidence': sum(business_evidence(p) for p in pages),
        'candidate_page_scope_counts': dict(Counter(p['scope'] for p in substantive)),
        'unique_page_scope_counts_overlapping': {scope: len({p['snapshot_id'] for p in substantive if p['scope'] == scope})
            for scope in ('outlet_specific', 'broader_brand', 'unclear')},
        'candidate_scope_counts_overlapping': {scope: sum(scope in p['scope_categories'] for p in packets)
            for scope in ('outlet_specific', 'broader_brand', 'unclear')},
        'active_by_eligible_candidates': sum(p['active_by_year'] is not None for p in packets),
        'active_by_eligible_candidate_pages': sum(p['active_by_eligible'] for p in pages),
        'explicit_verified_opening_year_candidates': sum(p['explicit_verified_opening_year'] is not None for p in packets),
        'explicit_verified_opening_year_candidate_pages': sum(p['explicit_opening_year'] is not None for p in pages),
        'confirmed_candidates_without_content_with_technical_fetch_failure': sum(not p['substantive_page_count'] and any(technical_raw_failure(s) for s in p['raw_snapshot_outcomes']) for p in confirmed_packets),
        'confirmed_candidates_with_only_unusable_content': sum(p['state'] == 'archive_content_unusable' for p in confirmed_packets),
        'candidate_state_counts': {state: sum(p['state'] == state for p in packets) for state in STATES},
        'raw_failure_reasons_by_unique_snapshot': dict(Counter(v['reason'] for v in raw.values() if v['state'] != 'success')),
        'raw_technical_failure_reasons_by_unique_snapshot': dict(Counter(v['reason'] for v in raw.values() if technical_raw_failure(v))),
        'parser_failure_reasons_by_candidate_page': dict(Counter(p['reason'] for p in pages if p['parse_status'] != 'success' or not p['substantive_parse'])),
        'bounded_discovery_failure_reasons_by_candidate_query': dict(Counter(q['reason'] for d in discoveries.values() for q in d['queries'] if q['state'] == 'failure')),
        'traceable_text_spans_verified': span_count, 'preservation_unchanged': preservation['unchanged']}
    report = {'schema': VERSION, 'counts': counts, 'promotion': promotion_assessment(counts),
        'verification': {'implementation_commitments_verified': True, 'frozen_details_verified': True,
            'discovery_receipts_reparsed': True, 'raw_manifest_and_files_verified': True,
            'parser_reproduced_exactly': True, 'claim_spans_verified': True,
            'all_1226_packets_frozen': len(packets) == 1226},
        'denominators': {'candidate_frame': 1226, 'usable_websites': 1088,
            'confirmed_candidates': len(confirmed_packets), 'unique_snapshots': len(snapshots),
            'candidate_snapshot_pairs': counts['candidate_snapshot_pairs'],
            'successful_raw_unique_pages': counts['raw_successful_snapshots'],
            'parsed_candidate_pages': len(successful_pages)},
        'interpretation': [plan['bounded_scope'],
            'A bounded no-snapshot outcome is not proof that no historical archive exists.',
            'Transport HTTP 200 is not substantive archive content; rejected, empty, error and interstitial responses remain failures or unusable content.',
            'Shared pages are parsed separately for each candidate. Page and candidate denominators must not be interchanged.',
            'Candidate state is mutually exclusive, but a candidate may have substantive content and failed discovery queries; overlapping counts are explicit.',
            'Outlet scope and historical eligibility are deterministic evidence screens, not final classifier judgments.',
            'The frozen parser substantive_parse field means nonempty non-error visible text; the separate identifiable-business-evidence count requires a candidate identity match or extracted service, contrary, or historical statement.',
            'No usable website includes missing websites and the predeclared rejected-URL exclusion; excluded URLs did not enter the 1,088-candidate discovery frame and are not live lookup failures.',
            'Verified opening means an eligible explicit textual claim tied to the outlet and verified raw snapshot, not independent external confirmation of the historical event.'],
        'classification_performed': False, 'stop_for_user_review': True}
    save('acquisition_preflight.json', report)
    markdown = render_markdown(report)
    write(root / 'acquisition_preflight.md', markdown.encode('utf-8'), immutable=True)
    files['acquisition_preflight.md'] = digest(markdown.encode('utf-8'))
    for name in ('plan.json', 'implementation_manifest.json', 'parser_implementation_manifest.json',
                 'discovery_manifest.json', 'snapshots.json', 'snapshot_seal.json', 'preservation_before.json'):
        files[name] = digest((root / name).read_bytes())
    completion = 'raw_fetch_seal.json' if snapshots else 'no_confirmed_snapshots.json'
    files[completion] = digest((root / completion).read_bytes())
    files.update({r['details_file']: r['details_sha256'] for r in plan['candidates']})
    verify_files(root, files)
    write_json(root / 'acquisition_freeze.json', {'schema': VERSION, 'files': files,
        'packet_count': len(packets), 'parser_reproducible': True,
        'classification_performed': False, 'stop_for_user_review': True}, immutable=True)
    return report


def render_markdown(report):
    c, gate = report['counts'], report['promotion']
    lines = ['# Clean archive acquisition preflight', '',
        'Acquisition is frozen for user review. No model classification was performed.', '',
        '| Measure | Count |', '|---|---:|']
    for key, label in (
        ('candidates', 'Frozen candidates'), ('usable_website_candidates', 'Usable website candidates'),
        ('missing_website_candidates', 'Missing website candidates'), ('rejected_website_candidates', 'Rejected website candidates'),
        ('candidates_with_confirmed_snapshot', 'Candidates with confirmed snapshots'),
        ('no_snapshot_in_bounded_lookup_candidates', 'Candidates with no snapshot under the bounded schedule'),
        ('bounded_discovery_failure_candidates', 'Candidates with failed bounded queries (overlapping)'),
        ('discovery_failure_and_substantive_content_candidates', 'Candidates with both failed discovery and substantive content'),
        ('unique_snapshots', 'Unique confirmed snapshots'), ('candidate_snapshot_pairs', 'Candidate/snapshot pairs'),
        ('raw_successful_snapshots', 'Raw HTTP success: unique pages'), ('raw_terminal_429_snapshots', 'Terminal 429: unique pages'),
        ('raw_429_responses_this_execution', 'All 429 responses in the sealed execution'),
        ('raw_429_responses_all_sealed_attempts', 'All 429 responses across sealed attempt history'),
        ('raw_permanent_failure_snapshots', 'Permanent raw failures: unique pages'),
        ('raw_technical_failure_snapshots', 'Technical raw failures: unique pages'),
        ('raw_permanent_404_or_410_snapshots', 'Permanent HTTP 404/410 failures: unique pages'),
        ('raw_retryable_failure_snapshots', 'Remaining retryable raw failures: unique pages'),
        ('raw_html_successful_snapshots', 'HTML MIME raw successes: unique pages'),
        ('parsed_unique_pages', 'Successfully parsed unique pages'), ('substantive_unique_pages', 'Substantive unique pages'),
        ('candidate_page_parse_attempts', 'Candidate/page parse attempts'), ('candidate_page_parse_successes', 'Candidate/page parse successes'),
        ('candidates_with_substantive_content', 'Candidates with substantive archived content'),
        ('candidates_with_identifiable_business_evidence', 'Candidates with identifiable business evidence'),
        ('active_by_eligible_candidates', 'Candidates with eligible active-by evidence'),
        ('explicit_verified_opening_year_candidates', 'Candidates with eligible explicit opening year')):
        lines.append(f'| {label} | {c[key]} |')
    lines += ['', 'Candidate states (mutually exclusive):', '', '| State | Candidates |', '|---|---:|']
    lines += [f'| {state} | {count} |' for state, count in c['candidate_state_counts'].items()]
    lines += ['', '| Substantive content scope | Candidate/page pairs | Candidates (overlapping) |', '|---|---:|---:|']
    lines += [f'| {scope} | {c["candidate_page_scope_counts"].get(scope, 0)} | {c["candidate_scope_counts_overlapping"][scope]} |'
              for scope in ('outlet_specific', 'broader_brand', 'unclear')]
    lines += ['', f'User acquisition conditions met: **{gate["user_conditions_met"]}**. '
        f'Stricter frozen-plan review hold: **{gate["conservative_review_hold"]}**. '
        f'Recommend all 1,226 to gpt-5-mini: **{gate["recommend_all_1226_gpt_5_mini"]}**.', '',
        f'Technical no-content failures affect {gate["technical_no_content_fraction_of_confirmed_candidates"]:.2%} '
        f'of confirmed candidates; unusable-only content affects {gate["unusable_content_fraction_of_confirmed_candidates"]:.2%}. '
        + gate['dominance_rule_note'], '']
    lines += [gate['technical_failure_definition'], '']
    lines += ['- ' + reason for reason in gate['user_condition_blockers'] + gate['conservative_review_hold_reasons']]
    lines += ['', 'Failure reasons:', '']
    for field in ('raw_failure_reasons_by_unique_snapshot', 'parser_failure_reasons_by_candidate_page',
                  'raw_technical_failure_reasons_by_unique_snapshot',
                  'bounded_discovery_failure_reasons_by_candidate_query'):
        lines.append(f'- {field}: ' + (', '.join(f'{k}: {v}' for k, v in sorted(c[field].items(), key=lambda x: (-x[1], str(x[0])))) or 'none'))
    lines += ['', f'Parser equality and {c["traceable_text_spans_verified"]} evidence spans verified. '
              f'Prior file hashes and recorded times unchanged: **{c["preservation_unchanged"]}**.', '']
    for note in report['interpretation']:
        lines.extend([note, ''])
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    result = build_preflight(args.run_dir)
    print(encoded({'counts': result['counts'], 'promotion': result['promotion']}).decode('utf-8'))
