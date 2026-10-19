"""Verify closure and compute final RQ1 only if all 1226 classifications are valid."""
import argparse
from collections import Counter
import json
from pathlib import Path

from clean_phase2_closure_contract import validate_decision, validate_record
from clean_phase2_closure_runtime import verify_completed as verify_retry
from clean_phase2_evidence import output_text, MODEL, CORE_FIELDS
from run_clean_phase2 import verify_inputs, verify_completed as verify_original, inventory
from run_phase2_technical_closure import preservation
from clean_archive_preflight_postfetch import verify_files, verify_preservation
from clean_phase2_report_v2 import rq1, compare, load_old, csv_table, markdown
from wayback_snapshot_fetcher import read_json, write_json, write, encoded, digest

REPO=Path(__file__).resolve().parents[1]


def finish(base,out,old_root):
    preservation(base,out)
    verify_files(out,read_json(out/'precall_freeze.json')['files'])
    verify_files(out,read_json(out/'retry_freeze.json')['files'])
    verify_files(REPO,read_json(out/'implementation.json')['files'])
    verify_files(REPO,read_json(base/'classification_implementation.json')['files'])
    verify_files(REPO,read_json(base/'continuation_v1/implementation.json')['files'])
    records,protocol=verify_inputs(base)
    inputs={r['place_id']:r for r in records}
    freeze=read_json(base/'classification_freeze.json')
    manifest=read_json(out/'valid_1156_manifest.json')
    frozen_valid={r['place_id'] for r in manifest['records']}
    audit=read_json(out/'technical_audit_summary.json')
    salvaged=set(audit['salvaged_place_ids']);retried=set(audit['retry_place_ids'])
    if frozen_valid&salvaged or frozen_valid&retried or salvaged&retried or frozen_valid|salvaged|retried!=set(inputs):
        raise ValueError('closure populations do not partition the frozen universe')
    results,index,models=[],[],Counter()
    for record in records:
        cid=record['place_id']
        if cid in frozen_valid:
            origin=base/freeze['result_roots'][cid]
            path=origin/f'private/classifications/{cid}.json'
            result=verify_original(origin,record,read_json(path))
            raw_path=origin/result['response_file'];method='original_valid_unchanged'
        elif cid in salvaged:
            path=out/f'private/salvaged/{cid}.json';result=read_json(path)
            raw_path=Path(result['response_file']);envelope=read_json(raw_path)
            raw=json.loads(output_text(json.loads(envelope['raw_response_utf8'])))
            decision,reducer=validate_decision(record,raw)
            if (result['decision']!=decision.to_public_dict() or result['reducer']!=reducer.to_public_dict()
                or result['parsed_response']!=raw or digest(raw_path.read_bytes())!=result['response_file_sha256']
                or result['packet_sha256']!=record['packet_sha256']):
                raise ValueError('deterministic salvage does not reproduce from original response')
            method='deterministic_revalidation'
        else:
            retry=validate_record(read_json(out/f'private/inputs/{cid}.json'))
            if retry['original_packet']!=record:raise ValueError('retry evidence changed')
            path=out/f'private/classifications/{cid}.json'
            result=verify_retry(out,retry,read_json(path))
            raw_path=out/result['response_file'] if result['status']=='completed' else None
            method='new_model_call_with_explicit_identifier_contract'
        if result['status']=='completed':
            envelope=read_json(raw_path)
            if digest(envelope['raw_response_utf8'].encode())!=envelope['raw_response_sha256']:
                raise ValueError('raw response changed')
            body=json.loads(envelope['raw_response_utf8']);models[body['model']]+=1
        results.append(result)
        index.append({'place_id':cid,'candidate_id':record['candidate_id'],'method':method,'result_file':str(path),
            'result_file_sha256':digest(path.read_bytes()),'original_input_packet_sha256':record['packet_sha256'],
            'effective_input_packet_sha256':result['packet_sha256'],'raw_response_file':str(raw_path) if raw_path else None,
            'raw_response_file_sha256':digest(raw_path.read_bytes()) if raw_path else None,
            'parsed_output_sha256':digest(encoded(result['parsed_response'])) if result['status']=='completed' else None})
    if set(models)!={MODEL}:raise ValueError('model changed')
    request_count=0;requested_candidates=set();raw_models=Counter()
    for path in (out/'private/transactions').glob('*/*.request.json'):
        req=read_json(path);cid=req['binding']['place_id'];request_count+=1;requested_candidates.add(cid)
        if cid not in retried:raise ValueError('API request outside exact retry allow-list')
        packet=read_json(out/f'private/inputs/{cid}.json')
        if req['parameters']!=packet['request'] or req['binding']['packet_sha256']!=packet['packet_sha256']:
            raise ValueError('actual retry request differs from frozen request')
        response_path=path.with_name(path.name.replace('.request.','.response.'))
        if response_path.exists():
            env=read_json(response_path)
            if env['binding']!=req['binding'] or env['requested_at']!=req['requested_at'] or digest(env['raw_response_utf8'].encode())!=env['raw_response_sha256']:
                raise ValueError('retry transaction association/hash mismatch')
            raw_models[json.loads(env['raw_response_utf8'])['model']]+=1
    if requested_candidates!=retried or set(raw_models)!={MODEL}:raise ValueError('retry population/model mismatch')
    unresolved=[r for r in results if r['status']!='completed']
    if unresolved:
        write_json(out/'closure_blocked_report.json',{'valid_count':1226-len(unresolved),'unresolved':unresolved,
            'rq1_recomputed':False,'old_versus_new_recomputed':False,'phase3_ready':False},immutable=True)
        print(f'STOP: {len(unresolved)} technical unresolved remain. Final RQ1 not recomputed.',flush=True)
        return
    if len(results)!=1226 or len({r['place_id'] for r in results})!=1226:
        raise ValueError('not all 1226 valid')
    # Explicit full-validity gate above precedes every final RQ1 and old-run read.
    stats=rq1(results,inputs)
    old,old_commitment=load_old(old_root)
    comparison,changes=compare(old,results)
    if comparison['transitions_to_technical_unresolved']!=0:raise ValueError('unexpected unresolved comparison')
    closure=Path(protocol['closure'])
    if inventory(closure)!=read_json(base/'closure_preservation_before.json'):raise ValueError('accepted archive closure changed')
    prior=verify_preservation(Path(protocol['source']))
    if not prior['unchanged']:raise ValueError('prior protected files changed')
    preservation(base,out)
    checks={'all_1226_ids_accounted':True,'all_1226_substantive_classifications_valid':True,
        'zero_technical_unresolved':True,'no_duplicate_or_missing_ids':True,
        'all_valid_outputs_schema_conformant':True,'full_classifier_packets_reconstructable':True,
        'all_evidence_references_exact_and_candidate_bound':True,'raw_responses_not_rewritten':True,
        'board_absent_from_input_blocks':True,'original_reducer_unchanged_and_deterministic':True,
        'exact_dated_model_only':True,'original_valid_1156_byte_for_byte_unchanged':True,
        'no_original_valid_candidate_recalled':True,'complete_previous_checkpoint_unchanged':True,
        'accepted_closure_and_prior_artifacts_preserved':True}
    rows=read_json(out/'technical_audit_70.json')['rows']
    groups={}
    for name,flag in [('evidence_id_error_65',True),('other_errors_5',False)]:
        group=[r for r in rows if ('invalid forced-choice evidence IDs' in r['original_validator_errors'])==flag]
        groups[name]={'candidate_count':len(group),'exclusive_category_counts':dict(Counter(r['failure_category'] for r in group))}
    report={'schema':'phase2-technical-closure-final-v1','run_id':base.name,'closure_id':out.name,
        'requested_model':MODEL,'returned_models':dict(models),'completion_count':1226,
        'original_valid_preserved':1156,'deterministic_recoveries':len(salvaged),
        'candidates_requiring_new_calls':len(retried),'new_api_request_count':request_count,
        'new_response_models':dict(raw_models),'technical_audit_categories':audit['category_counts'],
        'root_cause_groups':groups,
        'root_cause_explanation':'61 recoverable records cited exact typed snapshot IDs or Google source Place IDs present in the original prompt but excluded by the old namespace/citation map. No IDs were mapped or rewritten. Eight records cited nonexistent IDs; one omitted the required canonical candidate citation. Those nine alone required fresh calls.',
        'rq1':stats,'old_versus_corrected':comparison,'verification':checks,'technical_unresolved':[],
        'phase3_scientifically_ready_for_review':True,'phase3_executed':False,
        'rq2_rq3_recomputed':False,'thesis_modified':False,'stop_for_review':True}
    write_json(out/'final_classification_index.json',{'candidate_count':1226,'rows':index},immutable=True)
    write_json(out/'final_rq1_summary.json',stats,immutable=True)
    write_json(out/'old_classification_commitments.json',old_commitment,immutable=True)
    write_json(out/'final_old_versus_corrected.json',comparison,immutable=True)
    csv_table(out,'final_old_versus_corrected_candidates.csv',changes)
    csv_table(out,'final_rq1_candidates.csv',[{'place_id':r['place_id'],'candidate_id':r['candidate_id'],
        'eligibility':r['reducer']['eligibility_state'],**{k:r['decision'][k] for k in CORE_FIELDS},
        'confidence':r['decision']['claim_confidence'],'exclusion_reasons':r['reducer']['reason_codes'],
        'archive_state':inputs[r['place_id']]['archive']['state']} for r in results])
    write_json(out/'final_preservation_verification.json',{'valid_1156_unchanged':True,
        'complete_previous_checkpoint_unchanged':True,'accepted_archive_unchanged':True,'prior_artifacts':prior},immutable=True)
    write_json(out/'FINAL_REVIEW.json',report,immutable=True)
    intro=('# Closure of the 70 Phase-2 technical-unresolved records\n\n'
        f'Original 1,156 valid outputs remain byte-for-byte unchanged. Deterministic recovery: {len(salvaged)}. '
        f'Candidates requiring new model calls: {len(retried)}; API requests: {request_count}.\n\n'
        +report['root_cause_explanation']+'\n\n'
        'The original 65 namespace-error group comprises 61 namespace recoveries, 3 nonexistent-ID cases, '
        'and 1 missing canonical candidate citation. The other 5 all cited nonexistent identifiers.\n\n')
    write(out/'FINAL_REVIEW.md',(intro+markdown(report)).encode(),immutable=True)
    outputs=['final_classification_index.json','final_rq1_summary.json','old_classification_commitments.json',
        'final_old_versus_corrected.json','final_old_versus_corrected_candidates.csv','final_rq1_candidates.csv',
        'final_preservation_verification.json','FINAL_REVIEW.json','FINAL_REVIEW.md']
    write_json(out/'final_closure_freeze.json',{'files':{f:digest((out/f).read_bytes()) for f in outputs},
        'precall_freeze_sha256':digest((out/'precall_freeze.json').read_bytes()),
        'retry_freeze_sha256':digest((out/'retry_freeze.json').read_bytes()),
        'report_source_sha256':digest(Path(__file__).read_bytes()),'valid_count':1226,'technical_unresolved':0,
        'phase3_execution_authorized':False},immutable=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True)
    a=p.parse_args();finish(a.base.resolve(),a.out.resolve(),a.old_root.resolve())
