"""Complete, Board-blind Phase-2 packets from the accepted clean closure.

The four-label contract, JSON schema, decision class, and eligibility reducer are
unchanged. A new adapter retains full reviews/text instead of legacy text limits.
Archive citation and year checks bind directly to the accepted closure evidence.
"""
from pathlib import Path
import json
import re
from collections import Counter
from jsonschema import Draft202012Validator

from wayback_snapshot_fetcher import encoded, digest, read_json, write_json
from forced_choice_classifier import DECISION_SCHEMA, SYSTEM_INSTRUCTIONS, PROMPT_VERSION, PROMPT_TEMPLATE_SHA256
from forced_choice_models import (CORE_FIELDS, CLAIM_FIELDS, FORCED_CHOICE_CONTRACT,
    FORCED_CHOICE_CONTRACT_SHA256, FORCED_CHOICE_SCHEMA_VERSION, ForcedChoiceDecision, candidate_id_for_place_id)
from forced_choice_eligibility import reduce_forced_choice_eligibility

MODEL = 'gpt-5-mini-2025-08-07'
VERSION = '2026-09-07-clean-archive-phase2-v1'
STATES = {'archive_content_available': 653, 'archive_content_unusable': 33,
          'snapshot_confirmed_but_fetch_failed': 2, 'no_snapshot_in_bounded_lookup': 400,
          'no_website': 138, 'discovery_failure': 0}
DETAIL_KEYS = {'addressComponents', 'businessStatus', 'displayName', 'formattedAddress', 'id',
              'internationalPhoneNumber', 'location', 'nationalPhoneNumber', 'rating',
              'regularOpeningHours', 'reviews', 'types', 'userRatingCount', 'websiteUri'}
FORBIDDEN_KEYS = {'board', 'board_record', 'board_records', 'board_match', 'license_nbr',
    'board_correspondence', 'linkage', 'linkage_results', 'rq2', 'rq3', 'spatial_results',
    'old_classification', 'prior_classification', 'eligibility_state', 'classification_result'}
ARCHIVE_INTERPRETATION = {
    'version': VERSION,
    'archive_is_not_a_fifth_eligibility_criterion': True,
    'missing_archive_does_not_permit_abstention': True,
    'text_scope_does_not_alone_support_a_historical_year': True,
    'active_by_rule': 'Only source IDs explicitly flagged active_by_eligible=true may support active_by_upper_bound; use the earliest eligible supplied year. Outlet scope or capture time alone is insufficient.',
    'opening_rule': 'Only eligible explicit opening claims and their supplied IDs may support explicit_archived_opening_claim. Otherwise use unknown/null/NotAssessable and no historical citations.',
    'bounded_lookup_rule': 'No snapshot returned under the fixed schedule is not exhaustive evidence of archive absence.',
    'technical_states_rule': 'Unusable, failed, and quarantined objects provide no archived page text. Partial history gaps do not negate independently verified text from other captures.',
    'full_text_policy': 'All supplied Google review text and accepted locally parsed archive text are retained in full, in source order; no raw HTML is supplied.',
}


def save(root, relative, value):
    write_json(root / relative, value, immutable=True)
    return digest(encoded(value))


def no_forbidden_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError('forbidden classifier evidence field: ' + str(key))
            no_forbidden_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            no_forbidden_fields(child)


def escaped(value):
    return encoded(value).decode('utf-8').replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')


def build_packet(source_packet, source_packet_sha, source_root_label):
    place_id = source_packet['candidate_id']
    cid = candidate_id_for_place_id(place_id)
    details = source_packet['details']
    if set(details) - DETAIL_KEYS or details['id'] != place_id:
        raise ValueError('Google input fields or Place ID differ from frozen frame')
    no_forbidden_fields(details)
    state = source_packet['state']
    if state not in STATES:
        raise ValueError('unknown authoritative archive state')
    candidate = {'candidate_id': cid, 'source_place_id': place_id,
                 'google_metadata': {k: v for k, v in details.items() if k != 'reviews'}}
    reviews = [{'evidence_id': 'review_' + digest(encoded({'place_id': place_id, 'provider_order_index': i, 'review': review})),
                'provider_order_index': i, 'review': review}
               for i, review in enumerate(details.get('reviews', []))]
    if len(reviews) > 5:
        raise ValueError('returned review subset exceeds provider contract')
    sources, citation_map = [], {cid: {'kind': 'candidate', 'google_details_sha256': source_packet['frozen_details']['sha256']}}
    for review in reviews:
        citation_map[review['evidence_id']] = {'kind': 'review', 'provider_order_index': review['provider_order_index'],
                                             'review_sha256': digest(encoded(review['review']))}
    parsed_by_id = {p['snapshot_id']: p for p in source_packet['archive_pages']}
    for item in source_packet['classifier_archive_evidence']:
        page = parsed_by_id[item['snapshot_id']]
        if not page['substantive_parse'] or page['parse_status'] != 'success':
            raise ValueError('unusable archive text cannot enter model packet')
        if digest(item['text'].encode()) != item['text_sha256'] or item['text'] != page['visible_text']:
            raise ValueError('classifier archive text differs from verified parser output')
        eid = 'archive_source_v2_' + digest(encoded({'place_id': place_id, 'snapshot_id': item['snapshot_id'],
                                                   'text_sha256': item['text_sha256'], 'parsed_sha256': page['parsed_sha256']}))
        chain = item['provenance_chain']
        if chain['candidate_id'] != place_id or chain['frozen_website_url'] != source_packet['website_url']:
            raise ValueError('archive candidate website binding changed')
        value = {'evidence_id': eid, 'snapshot_id': item['snapshot_id'], 'scope': item['scope'],
            'exact_snapshot_url': chain['exact_final_replay_url'],
            'capture_timestamp': page['provenance']['confirmed_snapshot_timestamp'],
            'archived_page_text': item['text'], 'text_sha256': item['text_sha256'],
            'raw_sha256': page['raw_sha256'], 'parsed_sha256': page['parsed_sha256'],
            'text_origin': chain['text_origin'], 'active_by_eligible': item['active_by_eligible'],
            'active_by_year': item['active_by_year'], 'explicit_opening_year': item['explicit_opening_year'],
            'identity_matches': item['matches'], 'service_evidence': item['service_evidence'],
            'contrary_evidence': item['contrary_evidence'], 'opening_claims': item['opening_claims'],
            'identity_ambiguities': item['identity_ambiguities']}
        if value['active_by_eligible'] and not any(s['eligible'] for s in value['service_evidence']):
            raise ValueError('active-by flag lacks substantive service evidence')
        # Accepted closure contains no eligible explicit opening claim. Stop if
        # that authoritative contract changes, rather than inventing claim IDs.
        if value['explicit_opening_year'] is not None:
            raise ValueError('unexpected explicit opening claim outside frozen closure')
        sources.append(value)
        citation_map[eid] = {'kind': 'archive_source', 'snapshot_id': item['snapshot_id'], 'provenance_chain': chain,
                             'text_sha256': item['text_sha256'], 'parsed_sha256': page['parsed_sha256']}
    if bool(sources) != (state == 'archive_content_available'):
        raise ValueError('archive state and model-visible text disagree')
    archive = {'state': state, 'frozen_website_url': source_packet['website_url'],
        'usable_website': source_packet['usable_website'],
        'no_website_reason': source_packet['excluded_website_reason'],
        'bounded_lookup_complete': source_packet['bounded_lookup_complete'],
        'has_partial_discovery_validation_failure': source_packet['bounded_discovery_failure'],
        'confirmed_snapshot_count': len(source_packet['snapshot_ids']),
        'raw_snapshot_outcomes': source_packet['raw_snapshot_outcomes'],
        'unusable_parse_outcomes': [{'snapshot_id': p['snapshot_id'], 'parse_status': p['parse_status'],
                                    'reason': p['reason'], 'parsed_sha256': p['parsed_sha256']}
                                   for p in source_packet['archive_pages'] if not p['substantive_parse']],
        'verified_archived_sources': sources,
        'accepted_active_by_year': source_packet['active_by_year'],
        'accepted_explicit_opening_year': source_packet['explicit_verified_opening_year']}
    no_forbidden_fields([candidate, reviews, archive])
    blocks = [
        ('AUTHENTICATED_CONTRACT_JSON', {**FORCED_CHOICE_CONTRACT, 'contract_sha256': FORCED_CHOICE_CONTRACT_SHA256}),
        ('AUTHENTICATED_ARCHIVE_INTERPRETATION_JSON', ARCHIVE_INTERPRETATION),
        ('UNTRUSTED_CANDIDATE_JSON', candidate),
        ('UNTRUSTED_REVIEWS_JSON', {'selection': 'complete provider-returned subset in provider order', 'returned_count': len(reviews), 'reviews': reviews}),
        ('UNTRUSTED_WAYBACK_JSON', archive)]
    content = ['ABLATION_CONDITION=C_metadata_reviews_wayback']
    for name, value in blocks:
        content.extend((f'<{name}>', escaped(value), f'</{name}>'))
    messages = [{'role': 'system', 'content': SYSTEM_INSTRUCTIONS}, {'role': 'user', 'content': '\n'.join(content)}]
    request = {'model': MODEL, 'input': messages,
        'text': {'format': {'type': 'json_schema', 'name': 'nets_ai_forced_choice_decision', 'strict': True, 'schema': DECISION_SCHEMA}},
        'store': False}
    record = {'schema': VERSION, 'place_id': place_id, 'candidate_id': cid, 'model': MODEL,
        'source_closure_root': source_root_label, 'source_packet_file': f'private/packets/{place_id}.json',
        'source_packet_sha256': source_packet_sha, 'frozen_details': source_packet['frozen_details'],
        'google_metadata': candidate['google_metadata'], 'reviews': reviews, 'archive': archive, 'citation_map': citation_map,
        'prompt_version': VERSION, 'base_prompt_version': PROMPT_VERSION, 'base_prompt_sha256': PROMPT_TEMPLATE_SHA256,
        'base_system_sha256': digest(SYSTEM_INSTRUCTIONS.encode()), 'schema_version': FORCED_CHOICE_SCHEMA_VERSION,
        'schema_sha256': digest(encoded(DECISION_SCHEMA)), 'contract_sha256': FORCED_CHOICE_CONTRACT_SHA256,
        'messages': messages, 'messages_sha256': digest(encoded(messages)), 'request': request,
        'request_sha256': digest(encoded(request))}
    record['packet_sha256'] = digest(encoded(record))
    return record


def validate_record(record):
    if digest(encoded({k: v for k, v in record.items() if k != 'packet_sha256'})) != record['packet_sha256']:
        raise ValueError('classifier packet hash mismatch')
    if record['model'] != MODEL or record['request']['model'] != MODEL:
        raise ValueError('requested model differs from authorized dated snapshot')
    if record['messages_sha256'] != digest(encoded(record['messages'])) or record['request_sha256'] != digest(encoded(record['request'])):
        raise ValueError('request/messages hash mismatch')
    if record['request']['input'] != record['messages'] or record['request']['text']['format']['schema'] != DECISION_SCHEMA:
        raise ValueError('schema or message request changed')
    if record['messages'][0]['content'] != SYSTEM_INSTRUCTIONS:
        raise ValueError('four-judgment system instructions changed')
    no_forbidden_fields([record['google_metadata'], record['reviews'], record['archive']])
    return record


def validate_decision(record, raw):
    validate_record(record)
    Draft202012Validator(DECISION_SCHEMA).validate(raw)
    if any(not text.strip() for text in raw['reasoning'].values()):
        raise ValueError('empty substantive reasoning')
    if raw['candidate_id'] != record['candidate_id']:
        raise ValueError('model output candidate binding mismatch')
    decision = ForcedChoiceDecision(raw['candidate_id'], *(raw[k] for k in CORE_FIELDS),
        raw['historical_year_value'], raw['historical_year_type'], raw['claim_confidence'], raw['claim_evidence_ids'],
        {k: digest(v.encode()) for k, v in raw['reasoning'].items()})
    if any(not set(ids) <= set(record['citation_map']) for ids in decision.claim_evidence_ids.values()):
        raise ValueError('decision cites unsupplied evidence')
    sources = record['archive']['verified_archived_sources']
    eligible = [s for s in sources if s['active_by_eligible']]
    if decision.historical_year_type == 'active_by_upper_bound':
        if not eligible:
            raise ValueError('timestamp or scope alone cannot support active-by')
        earliest = min(s['active_by_year'] for s in eligible)
        ids = {s['evidence_id'] for s in eligible if s['active_by_year'] == earliest}
        if decision.historical_year_value != earliest or not set(decision.claim_evidence_ids['historical_year']) & ids:
            raise ValueError('active-by year lacks eligible supplied service-text source')
    elif decision.historical_year_type == 'explicit_archived_opening_claim':
        raise ValueError('frozen closure contains no eligible explicit opening claim')
    return decision, reduce_forced_choice_eligibility(decision, place_id=record['place_id'], candidate_id=record['candidate_id'])


def output_text(response):
    if response.get('model') != MODEL:
        raise RuntimeError('returned_model_mismatch')
    if response.get('status') != 'completed':
        raise ValueError('model response not completed')
    texts = [part['text'] for item in response.get('output', []) if item.get('type') == 'message'
             for part in item.get('content', []) if part.get('type') == 'output_text']
    if len(texts) != 1:
        raise ValueError('model response lacks exactly one structured output')
    return texts[0]
