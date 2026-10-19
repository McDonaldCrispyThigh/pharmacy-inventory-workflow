"""Explicit, exact, candidate-local identifiers; no guessed reference mapping."""
import copy
import json
import re
from jsonschema import Draft202012Validator
from clean_phase2_evidence import (
    validate_record as validate_original, DECISION_SCHEMA, CORE_FIELDS, CLAIM_FIELDS,
    MODEL, escaped, output_text, no_forbidden_fields)
from clean_phase2_closure_decision import ClosureDecision
from forced_choice_eligibility import reduce_forced_choice_eligibility
from wayback_snapshot_fetcher import encoded, digest

VERSION = '2026-09-07-phase2-exact-citation-closure-v1'


def walk(value, path='$'):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + '.' + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, f'{path}[{i}]')
    elif isinstance(value, str):
        yield path, value


def visible_blocks(base):
    text = base['messages'][1]['content']
    return {name: json.loads(re.search(r'<'+name+r'>\n(.*?)\n</'+name+r'>', text, re.S).group(1))
            for name in ('UNTRUSTED_CANDIDATE_JSON','UNTRUSTED_REVIEWS_JSON','UNTRUSTED_WAYBACK_JSON')}


def identifier_registry(base):
    validate_original(base)
    blocks = visible_blocks(base)
    leaves = list(walk(blocks))
    registry = {}
    for eid, provenance in base['citation_map'].items():
        locations = [path for path, value in leaves if value == eid]
        if not locations:
            raise ValueError('expected evidence ID missing from exact model-visible blocks')
        registry[eid] = {'kind': provenance['kind'], 'exact_visible_locations': locations,
                         'original_citation_map_member': True}
    # These are already exact identifiers in typed fields of the original prompt.
    # Retain them as themselves: never rewrite one identifier into another.
    place_id = base['place_id']
    candidate = blocks['UNTRUSTED_CANDIDATE_JSON']
    if candidate['source_place_id'] != place_id or candidate['google_metadata']['id'] != place_id:
        raise ValueError('Google identifier binding mismatch')
    registry[place_id] = {'kind': 'google_metadata_source_id',
        'exact_visible_locations': [path for path,value in leaves if value == place_id],
        'original_citation_map_member': False}
    for path, value in leaves:
        if path.startswith('$.UNTRUSTED_WAYBACK_JSON.') and path.endswith('.snapshot_id'):
            if re.fullmatch(r'snapshot_[0-9a-f]{64}', value) is None:
                raise ValueError('malformed supplied snapshot identifier')
            registry.setdefault(value, {'kind': 'archive_snapshot_context',
                'exact_visible_locations': [], 'original_citation_map_member': False})['exact_visible_locations'].append(path)
    return registry


def validate_decision(record, raw):
    base = record.get('original_packet', record)
    validate_record(record)
    Draft202012Validator(DECISION_SCHEMA).validate(raw)
    if record.get('closure_version') == VERSION:
        Draft202012Validator(record['request']['text']['format']['schema']).validate(raw)
    if raw['candidate_id'] != base['candidate_id']:
        raise ValueError('model output candidate binding mismatch')
    if any(not text.strip() for text in raw['reasoning'].values()):
        raise ValueError('empty substantive reasoning')
    registry = identifier_registry(base)
    for ids in raw['claim_evidence_ids'].values():
        if not set(ids) <= set(registry):
            raise ValueError('decision cites identifier absent from candidate-bound supplied identifiers')
    decision = ClosureDecision(raw['candidate_id'], *(raw[k] for k in CORE_FIELDS),
        raw['historical_year_value'], raw['historical_year_type'], raw['claim_confidence'], raw['claim_evidence_ids'],
        {k:digest(v.encode()) for k,v in raw['reasoning'].items()}, frozenset(registry))
    eligible = [s for s in base['archive']['verified_archived_sources'] if s['active_by_eligible']]
    if decision.historical_year_type == 'active_by_upper_bound':
        if not eligible:
            raise ValueError('timestamp or scope alone cannot support active-by')
        earliest = min(s['active_by_year'] for s in eligible)
        ids = {s['evidence_id'] for s in eligible if s['active_by_year']==earliest}
        if decision.historical_year_value != earliest or not set(decision.claim_evidence_ids['historical_year']) & ids:
            raise ValueError('active-by year lacks eligible supplied service-text source')
        if not set(decision.claim_evidence_ids['historical_year']) <= set(base['citation_map']):
            raise ValueError('structural or Google identifiers cannot substantiate a historical year')
    elif decision.historical_year_type == 'explicit_archived_opening_claim':
        raise ValueError('frozen closure contains no eligible explicit opening claim')
    return decision, reduce_forced_choice_eligibility(decision, place_id=base['place_id'], candidate_id=base['candidate_id'])


def build_retry_record(base):
    validate_original(base)
    registry = identifier_registry(base)
    schema = copy.deepcopy(DECISION_SCHEMA)
    schema['properties']['candidate_id']['enum'] = [base['candidate_id']]
    for field in CLAIM_FIELDS:
        schema['properties']['claim_evidence_ids']['properties'][field]['items']['enum'] = sorted(registry)
    if not any(s['active_by_eligible'] for s in base['archive']['verified_archived_sources']):
        schema['properties']['claim_evidence_ids']['properties']['historical_year']['maxItems'] = 0
    declaration = {'version': VERSION, 'candidate_evidence_id': base['candidate_id'],
        'allowed_evidence_identifiers': {eid: value['kind'] for eid,value in sorted(registry.items())},
        'copy_exactly': True, 'do_not_create_or_normalize_identifiers': True,
        'each_core_judgment_requires_the_candidate_evidence_id': True,
        'structural_snapshot_ids_identify_only_supplied_snapshot_context': True,
        'structural_ids_do_not_imply_archived_text_or_support_a_historical_year': True}
    messages = copy.deepcopy(base['messages'])
    messages[1]['content'] += '\n<AUTHENTICATED_EVIDENCE_ID_CONTRACT_JSON>\n'+escaped(declaration)+'\n</AUTHENTICATED_EVIDENCE_ID_CONTRACT_JSON>'
    request = copy.deepcopy(base['request'])
    request['input'] = messages
    request['text']['format']['schema'] = schema
    result = {**copy.deepcopy(base), 'original_packet': base, 'closure_version': VERSION,
        'original_packet_sha256': base['packet_sha256'], 'prompt_version': VERSION,
        'schema_version': base['schema_version']+'+exact-citation-enum-v1',
        'messages': messages, 'messages_sha256': digest(encoded(messages)),
        'schema_sha256': digest(encoded(schema)), 'request': request,
        'request_sha256': digest(encoded(request))}
    result.pop('packet_sha256')
    result['packet_sha256'] = digest(encoded(result))
    return result


def validate_record(record):
    if record.get('closure_version') != VERSION:
        return validate_original(record)
    if encoded(build_retry_record(record['original_packet'])) != encoded(record):
        raise ValueError('versioned retry packet does not reconstruct exactly')
    no_forbidden_fields([record['google_metadata'],record['reviews'],record['archive']])
    return record
