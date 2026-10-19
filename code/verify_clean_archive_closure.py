"""Read-only verification of the final closure; writes one immutable receipt.

No providers or model calls. Known quarantined digests are rejected BEFORE reads.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
from urllib.parse import urlencode

from wayback_snapshot_fetcher import read_json, digest, encoded, write_json, parse_replay, website_key
from clean_archive_parser import parse_archive_page
from clean_archive_closure_rules import revalidate
from clean_archive_preflight_postfetch import validate_claim_spans, opening_summary


def verify(source, closure):
    roots = {'source': source.resolve(), 'closure': closure.resolve()}
    quarantine = read_json(source / 'security/quarantine_receipt.json')
    blocked = set(quarantine['blocked_body_sha256'])
    cache = {}
    def reference(ref):
        if ref['sha256'] in blocked:
            raise ValueError('blocked reference rejected before reading')
        root = roots[ref['root']]
        path = (root / ref['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('reference escapes declared root')
        key = str(path)
        if key not in cache:
            cache[key] = digest(path.read_bytes())
        if cache[key] != ref['sha256']:
            raise ValueError('retained artifact hash mismatch: ' + str(path))
        return path
    freeze = read_json(closure / 'closure_freeze.json')
    for relative, sha in freeze['files'].items():
        reference({'root': 'closure', 'path': relative, 'sha256': sha})
    manifest = read_json(closure / 'closure_implementation.json')
    repo = Path(__file__).resolve().parents[1]
    for relative, sha in manifest['files'].items():
        if digest((repo / relative).read_bytes()) != sha:
            raise ValueError('frozen implementation mismatch')
    if digest((source / 'acquisition_freeze.json').read_bytes()) != freeze['source_acquisition_freeze_sha256']:
        raise ValueError('source freeze changed')
    if digest((source / 'raw_postfetch_manifest.json').read_bytes()) != freeze['source_raw_manifest_sha256']:
        raise ValueError('source raw seal changed')
    plan = read_json(source / 'plan.json')
    counts, spans, fragments, pages_verified, claims = Counter(), 0, 0, 0, Counter()
    for row in plan['candidates']:
        cid = row['candidate_id']
        packet_path = closure / f'private/packets/{cid}.json'
        packet = read_json(packet_path)
        if packet['candidate_id'] != cid or packet['website_url'] != row['website_url'] or packet['classification_performed']:
            raise ValueError('packet frame, website, or execution boundary changed')
        frozen_details = read_json(reference(packet['frozen_details']))
        if frozen_details['payload']['details'] != packet['details']:
            raise ValueError('packet Details differ from frozen source')
        if packet['frozen_details']['sha256'] != row['details_sha256']:
            raise ValueError('Details hash differs from plan')
        counts[packet['state']] += 1
        parsed_by_id = {}
        for page in packet['archive_pages']:
            sid = page['snapshot_id']
            if sid == quarantine['snapshot_id'] or page['raw_sha256'] in blocked:
                raise ValueError('quarantine entered parsed evidence')
            parsed_ref = {'root': 'closure', 'path': page['parsed_file'], 'sha256': page['parsed_sha256']}
            saved = read_json(reference(parsed_ref))
            if any(page.get(k) != v for k, v in saved.items()):
                raise ValueError('embedded parser data differ from frozen parser file')
            chain = page['provenance_chain']
            if chain != saved['provenance']['closure_provenance_chain']:
                raise ValueError('page chain differs from parser commitment')
            if chain['candidate_id'] != cid or chain['frozen_website_url'] != row['website_url']:
                raise ValueError('page candidate binding mismatch')
            body = reference(chain['preserved_raw_body']).read_bytes()
            headers = read_json(reference(chain['raw_headers']))
            raw_provenance = read_json(reference(chain['raw_crawler_provenance']))
            if digest(body) != saved['raw_sha256'] or raw_provenance['sha256'] != saved['raw_sha256']:
                raise ValueError('raw byte identity mismatch')
            actual = parse_replay(raw_provenance['final_url'])
            if not chain['official_discovery']:
                raise ValueError('missing official discovery certificate')
            for cert in chain['official_discovery']:
                if cert['candidate_id'] != cid or cert['frozen_website_url'] != row['website_url']:
                    raise ValueError('certificate candidate mismatch')
                reference(cert['frozen_details'])
                official = reference(cert['official_response']).read_bytes()
                metadata = read_json(reference(cert['official_response_metadata']))
                request_url = 'https://archive.org/wayback/available?' + urlencode({
                    'url': row['website_url'], 'timestamp': cert['requested_timestamp']})
                if (metadata['request_url'] != request_url or metadata['final_url'] != request_url
                        or metadata['website_url'] != row['website_url'] or metadata['http_status'] != 200
                        or metadata['error'] or metadata['body_sha256'] != digest(official)
                        or cert['requested_timestamp'] not in plan['schedule']):
                    raise ValueError('official source receipt mismatch')
                checked = revalidate(official, row['website_url'], cert['requested_timestamp'], plan['snapshot_cutoff'])
                if (checked['state'] != 'confirmed' or encoded(checked['comparison']) != encoded(cert['binding_comparison'])
                        or checked['exact_wayback_snapshot_url'] != cert['exact_snapshot_url']
                        or checked['confirmed_snapshot_timestamp'] != actual['timestamp']
                        or website_key(checked['original_target_url']) != website_key(actual['original_url'])):
                    raise ValueError('official exact identity not established')
            reproduced = parse_archive_page(body, headers, packet['details'], saved['provenance'])
            if encoded(reproduced) != encoded(saved):
                raise ValueError('local parser output does not reproduce')
            spans += validate_claim_spans(saved)
            parsed_by_id[sid] = page
            pages_verified += 1
        eligible_ids = {sid for sid, page in parsed_by_id.items() if page['substantive_parse']}
        items = packet['classifier_archive_evidence']
        if {i['snapshot_id'] for i in items} != eligible_ids or len(items) != len(eligible_ids):
            raise ValueError('classifier evidence selection changed or duplicated')
        for item in items:
            page = parsed_by_id[item['snapshot_id']]
            if item['text'] != page['visible_text'] or digest(item['text'].encode()) != item['text_sha256']:
                raise ValueError('classifier text differs from verified parser output')
            if (item['fragment']['start'] != 0 or item['fragment']['end'] != len(item['text'])
                    or item['fragment']['raw_sha256'] != page['raw_sha256']
                    or item['fragment']['parser_output_sha256'] != page['parsed_sha256']):
                raise ValueError('classifier text fragment provenance mismatch')
            for key in ('scope', 'matches', 'service_evidence', 'contrary_evidence', 'opening_claims',
                        'identity_ambiguities', 'active_by_eligible', 'active_by_year', 'explicit_opening_year'):
                if item[key] != page[key]:
                    raise ValueError('classifier claim differs from verified parser claim')
            expected_chain = {**page['provenance_chain'], 'verified_parser_output': {
                'root': 'closure', 'path': page['parsed_file'], 'sha256': page['parsed_sha256']}}
            if item['provenance_chain'] != expected_chain:
                raise ValueError('classifier chain differs from verified page')
            fragments += 1
        expected_opening = opening_summary(packet['archive_pages'])
        if any(packet[k] != v for k, v in expected_opening.items()):
            raise ValueError('candidate opening aggregate changed')
        active = [p for p in packet['archive_pages'] if p['active_by_eligible']]
        if packet['active_by_year'] != min((p['active_by_year'] for p in active), default=None):
            raise ValueError('candidate active-by aggregate changed')
        claims['active_by'] += packet['active_by_year'] is not None
        claims['opening_year'] += packet['explicit_verified_opening_year'] is not None
    report = read_json(closure / 'closure_preflight.json')
    if (dict(counts) != {k: v for k, v in report['counts']['candidate_states'].items() if v}
            or spans != report['counts']['traceable_claim_spans_verified']
            or fragments != report['counts']['classifier_visible_full_text_fragments_verified']
            or sum(counts.values()) != 1226):
        raise ValueError('reported totals differ from independently verified packets')
    receipt = {'schema': 'independent-clean-closure-verification-v1',
        'closure_freeze_sha256': digest((closure / 'closure_freeze.json').read_bytes()),
        'verifier_sha256': digest(Path(__file__).read_bytes()), 'candidate_count': 1226,
        'candidate_states': dict(counts), 'retained_file_references_hash_verified': len(cache),
        'parser_outputs_reproduced': pages_verified, 'classifier_fragments_verified': fragments,
        'claim_spans_verified': spans, 'historical_claims': dict(claims),
        'quarantined_object_opened': False, 'quarantined_text_in_classifier_inputs': False,
        'network_calls': 0, 'classification_performed': False, 'passed': True}
    write_json(closure / 'verification_receipt.json', receipt, immutable=True)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--closure', type=Path, required=True)
    args = parser.parse_args()
    verify(args.source, args.closure)
