"""Audit every original failed transaction before any new classification call."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path

from clean_phase2_closure_contract import identifier_registry, visible_blocks, walk, validate_decision
from clean_phase2_evidence import output_text, CORE_FIELDS
from wayback_snapshot_fetcher import read_json, write_json, write, encoded, digest
from clean_archive_preflight_postfetch import verify_files

CATEGORIES = {'A':'validator namespace/mapping bug', 'B':'valid evidence ID rejected incorrectly',
    'C':'model cited nonexistent evidence ID', 'D':'classifier packet missing expected evidence ID',
    'E':'response/schema failure', 'F':'API/credit/transport failure', 'G':'other'}


def audit(base, out):
    manifest = read_json(out/'valid_1156_manifest.json')
    for r in manifest['records']:
        verify_files(base, r['files'])
    frame = read_json(out/'unresolved_frame.json')['place_ids']
    all_global = {}
    for path in (base/'private/inputs').glob('*.json'):
        packet = read_json(path)
        for eid in packet['citation_map']:
            all_global.setdefault(eid, []).append(packet['place_id'])
    freeze = read_json(base/'classification_freeze.json')
    rows, salvage, retry = [], [], []
    for cid in frame:
        original = read_json(base/f'private/inputs/{cid}.json')
        final_old = read_json(base/freeze['result_roots'][cid]/f'private/classifications/{cid}.json')
        registry = identifier_registry(original)
        leaves = list(walk(visible_blocks(original)))
        exact_strings = {v for _,v in leaves}
        attempts, recovered = [], None
        for folder in freeze['transaction_directories']:
            for path in sorted((base/folder/cid).glob('*.request.json')):
                request = read_json(path)
                row = {'request_file':path.relative_to(base).as_posix(), 'requested_at':request['requested_at'],
                    'raw_response_available':False,'stage':'during_model_call', 'validator_error':None,
                    'cited_ids':{},'id_checks':{},'parsed_substantive_judgments':None,
                    'deterministic_revalidation_valid':False}
                response_path = path.with_name(path.name.replace('.request.','.response.'))
                error_path = path.with_name(path.name.replace('.request.','.error.'))
                validation_path = path.with_name(path.name.replace('.request.','.validation.'))
                if error_path.exists():
                    row['api_error'] = read_json(error_path)
                if validation_path.exists():
                    row['validator_error'] = read_json(validation_path).get('reason')
                if response_path.exists():
                    envelope = read_json(response_path)
                    if digest(envelope['raw_response_utf8'].encode()) != envelope['raw_response_sha256']:
                        raise ValueError('original raw response changed')
                    row.update(raw_response_available=True, raw_response_file=response_path.relative_to(base).as_posix(),
                        raw_response_sha256=envelope['raw_response_sha256'], stage='during_parsing')
                    try:
                        raw = json.loads(output_text(json.loads(envelope['raw_response_utf8'])))
                        row['parsed_substantive_judgments'] = {k:raw.get(k) for k in CORE_FIELDS}
                        row['cited_ids'] = raw.get('claim_evidence_ids',{})
                        row['stage'] = 'post_response_validation'
                        for eid in set(i for ids in row['cited_ids'].values() for i in ids):
                            missing = eid not in registry
                            row['id_checks'][eid] = {
                                'exists_as_exact_scalar_in_original_prompt':eid in exact_strings,
                                'exists_in_original_citation_map':eid in original['citation_map'],
                                'exists_in_corrected_typed_registry':not missing,
                                'exact_visible_locations':[p for p,v in leaves if v==eid],
                                'registry_entry':registry.get(eid),
                                'same_id_in_other_candidates':[other for other in all_global.get(eid,[]) if other!=cid],
                                'case_only_matches':[x for x in registry if missing and x.casefold()==eid.casefold()],
                                'truncated_prefix_matches':[x for x in registry if missing and len(eid)>16 and x.startswith(eid)],
                                'namespace_same_suffix_matches':[x for x in registry if missing and len(eid)>=64 and x[-64:]==eid[-64:]],
                                'normalization_or_alias_mapping_applied':False}
                        row['duplicate_id_arrays'] = [k for k,ids in row['cited_ids'].items() if len(ids)!=len(set(ids))]
                        d,reducer = validate_decision(original,raw)
                        row['deterministic_revalidation_valid'] = True
                        if recovered is None:
                            recovered = {'place_id':cid,'candidate_id':original['candidate_id'],
                                'packet_sha256':original['packet_sha256'],'request_sha256':original['request_sha256'],
                                'requested_model':original['model'],'returned_model':json.loads(envelope['raw_response_utf8'])['model'],
                                'status':'completed','recovery_method':'deterministic_revalidation',
                                'response_file':str(response_path),'response_file_sha256':digest(response_path.read_bytes()),
                                'raw_response_sha256':envelope['raw_response_sha256'],
                                'parsed_response':raw,'parsed_output_sha256':digest(encoded(raw)),
                                'decision':d.to_public_dict(),'reducer':reducer.to_public_dict(),
                                'requested_at':envelope['requested_at'],'received_at':envelope['received_at'],
                                'validator_version':'2026-09-07-phase2-exact-citation-closure-v1',
                                'evidence_references_rewritten':False}
                    except (ValueError,TypeError,KeyError,RuntimeError) as exc:
                        row['revalidation_error'] = str(exc)
                else:
                    row['revalidation_error'] = row.get('api_error',{}).get('reason','interrupted_request_without_received_response')
                attempts.append(row)
        attempts.sort(key=lambda r:r['requested_at'])
        if recovered:
            category = 'A' if any(not registry[eid]['original_citation_map_member'] for ids in recovered['parsed_response']['claim_evidence_ids'].values() for eid in ids) else 'B'
            salvage.append(cid)
            write_json(out/f'private/salvaged/{cid}.json',recovered,immutable=True)
        else:
            retry.append(cid)
            returned = [a for a in attempts if a['raw_response_available']]
            selected = returned[-1] if returned else attempts[-1]
            category = ('C' if any(not v['exists_in_corrected_typed_registry'] for v in selected['id_checks'].values())
                        else 'E' if returned else 'F')
        rows.append({'place_id':cid,'candidate_id':original['candidate_id'], 'failure_category':category,
            'failure_category_label':CATEGORIES[category], 'original_validator_errors':final_old['technical_reasons'],
            'classifier_visible_evidence_ids':registry,'raw_model_response_available':any(a['raw_response_available'] for a in attempts),
            'recovered_without_model_call':bool(recovered), 'attempts':attempts,
            'mapping_audit': {'input_packet_sha256':original['packet_sha256'],
                'exact_json_scalar_equality_required':True,'literal_substring_is_not_presence':True,
                'candidate_local_registry_used':True,'current_global_namespace_checked_for_diagnostics_only':True,
                'no_sanitization_case_fold_truncation_or_alias_rewrite_permitted':True,
                'preclosure_or_old_run_ids_not_accepted':True}})
    write_json(out/'technical_audit_70.json',{'candidate_count':70,'categories':CATEGORIES,'rows':rows},immutable=True)
    stream=io.StringIO(newline='')
    w=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    write(out/'technical_audit_70.csv',stream.getvalue().encode(),immutable=True)
    summary={'candidate_count':70,'category_counts':{k:sum(r['failure_category']==k for r in rows) for k in CATEGORIES},
        'salvaged_count':len(salvage),'salvaged_place_ids':salvage,'retry_count':len(retry),'retry_place_ids':retry,
        'new_model_calls':0,'raw_responses_rewritten':0,'evidence_ids_rewritten':0,
        'salvage_selection':'First original response in chronological segment/attempt order passing every corrected check; no outcome-based selection.',
        'namespace_error_65_group':[r['place_id'] for r in rows if 'invalid forced-choice evidence IDs' in r['original_validator_errors']],
        'other_5_group':[r['place_id'] for r in rows if 'invalid forced-choice evidence IDs' not in r['original_validator_errors']]}
    write_json(out/'technical_audit_summary.json',summary,immutable=True)
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('_ids') and not k.endswith('_group')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.base.resolve(),a.out.resolve())
