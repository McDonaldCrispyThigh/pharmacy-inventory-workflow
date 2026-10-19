"""Provider-free downstream execution from the accepted, closed Phase-2 layer."""
import argparse
from collections import Counter
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import socket

os.environ['PROJ_NETWORK']='OFF'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

from wayback_snapshot_fetcher import read_json, write_json, write, digest, encoded
from clean_archive_preflight_postfetch import verify_files, verify_preservation
from clean_phase2_closure_contract import validate_decision, validate_record
from clean_phase2_evidence import output_text, MODEL
from run_clean_phase2 import inventory
from v2_evidence_models import QueryProvenanceV2
from v2_downstream_models import candidate_location_from_details
from forced_choice_delivery import ForcedChoiceDeliveryCandidate, load_forced_choice_local_context
from corrected_downstream import implementation_hashes, run_linkage_phase, run_spatial_phase
from study_design import TARGET_ZIPS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/clean_classification/20260907T192546Z'
CLOSURE=ROOT/'data/clean_classification_closure/20260907_closure_v1'
ARCHIVE=ROOT/'data/clean_archive_closure/20260907_closure_v1'
SOURCE=ROOT/'data/clean_archive/20260906T040605Z'
OUTPUT=ROOT/'analysis/closed_phase3/20260907T192546Z_v1'


def deny_network(*args,**kwargs):
    raise RuntimeError('Phase 3 is deterministic/local-only; network use is blocked')


def load_candidates():
    seal=read_json(CLOSURE/'final_closure_freeze.json')
    verify_files(CLOSURE,seal['files'])
    review=read_json(CLOSURE/'FINAL_REVIEW.json')
    if (seal['valid_count']!=1226 or seal['technical_unresolved']!=0
        or review['rq1']['eligibility']['include']['count']!=539
        or review['rq1']['eligibility']['exclude']['count']!=687):
        raise ValueError('accepted Phase-2 closure gate changed')
    index=read_json(CLOSURE/'final_classification_index.json')['rows']
    candidates=[];lineage=[]
    for row in index:
        cid=row['place_id'];path=Path(row['result_file'])
        if digest(path.read_bytes())!=row['result_file_sha256']:raise ValueError('closed classification changed')
        result=read_json(path)
        raw_path=Path(row['raw_response_file'])
        if digest(raw_path.read_bytes())!=row['raw_response_file_sha256']:raise ValueError('raw response changed')
        envelope=read_json(raw_path)
        if digest(envelope['raw_response_utf8'].encode())!=envelope['raw_response_sha256']:raise ValueError('raw response content changed')
        raw=json.loads(output_text(json.loads(envelope['raw_response_utf8'])))
        original=read_json(BASE/f'private/inputs/{cid}.json')
        effective=(validate_record(read_json(CLOSURE/f'private/inputs/{cid}.json'))
                   if row['method']=='new_model_call_with_explicit_identifier_contract' else original)
        decision,reducer=validate_decision(effective,raw)
        if result['status']!='completed' or result['decision']!=decision.to_public_dict() or result['reducer']!=reducer.to_public_dict():
            raise ValueError('final classifier/reducer does not reproduce')
        if original['packet_sha256']!=row['original_input_packet_sha256'] or effective['packet_sha256']!=row['effective_input_packet_sha256']:
            raise ValueError('closed input packet binding differs')
        receipt_path=SOURCE/f'private/details/{cid}.json'
        if digest(receipt_path.read_bytes())!=original['frozen_details']['sha256']:raise ValueError('frozen September 5 Details changed')
        receipt=read_json(receipt_path)
        if digest(encoded(receipt['payload']))!=receipt['sha256']:raise ValueError('Details envelope commitment failed')
        payload=receipt['payload'];details=payload['details'];frame=payload['frame_entry']
        if {k:v for k,v in details.items() if k!='reviews'}!=original['google_metadata']:
            raise ValueError('downstream location is not the final frozen Google observation')
        provenance=QueryProvenanceV2(frame['query_definition_id'],frame['source_zip'],'places_search')
        location=candidate_location_from_details(details,provenance,datetime.fromisoformat(payload['retrieved_at'].replace('Z','+00:00')))
        item=ForcedChoiceDeliveryCandidate(location,decision)
        candidates.append(item)
        lineage.append({**item.lineage_record(),'closed_result_sha256':row['result_file_sha256'],
            'original_input_packet_sha256':original['packet_sha256'],'effective_input_packet_sha256':effective['packet_sha256'],
            'frozen_details_sha256':original['frozen_details']['sha256'],'eligibility_state':item.eligibility.eligibility_state})
    counts=Counter(c.eligibility.eligibility_state for c in candidates)
    if len(candidates)!=1226 or len({c.location.place_id for c in candidates})!=1226 or counts!={'include':539,'exclude':687}:
        raise ValueError('closed candidate inventory differs from accepted 1226/539/687')
    return tuple(candidates),lineage


def protected():
    return {'phase2':inventory(BASE),'phase2_closure':inventory(CLOSURE),'accepted_archive':inventory(ARCHIVE)}


def prepare():
    candidates,lineage=load_candidates()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(OUTPUT/'user_authorization.txt',Path(r'C:\Users\16067\.codex\attachments\ebd8eaa5-7a82-4939-896d-9a663f1d6d25\pasted-text.txt').read_bytes(),immutable=True)
    write_json(OUTPUT/'protected_before.json',protected(),immutable=True)
    write_json(OUTPUT/'candidate_lineage.json',{'candidate_count':1226,'include':539,'exclude':687,'rows':lineage},immutable=True)
    pins=implementation_hashes()
    for name in ('code/run_closed_phase3.py','code/clean_phase2_closure_contract.py','code/clean_phase2_closure_decision.py',
                 'code/build_final_thesis_figures_v1.py'):
        pins[name]=digest((ROOT/name).read_bytes())
    for name in pins:write(OUTPUT/'implementation'/name,(ROOT/name).read_bytes(),immutable=True)
    write_json(OUTPUT/'implementation.json',{'files':pins},immutable=True)
    versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','geopandas','shapely','pyproj','matplotlib','rapidfuzz')}
    write_json(OUTPUT/'run_scope.json',{'user_phase':3,'scope':'deterministic RQ2/RQ3 and only approved Figures 4.1–4.3; stop before thesis editing',
        'logical_classifier_run_id':BASE.name,'phase2_closure':CLOSURE.name,'phase2_closure_sha256':digest((CLOSURE/'final_closure_freeze.json').read_bytes()),
        'candidate_count':1226,'included_inventory_rows':539,'excluded_rows':687,'technical_unresolved':0,
        'classifier_calls_allowed':False,'provider_calls_allowed':False,'thesis_edits_allowed':False,
        'geographic_zip_counts':[60,101,106],'origins':327,'distance_crs':'EPSG:26915',
        'distance_estimand':'paired tract-level straight-line nearest distance','thresholds_miles':[0.5,1,2],
        'recurrence_bootstrap_draws':5000,'recurrence_cutoffs':[2,3,5],
        'internal_legacy_phase_labels':{'rq2':3,'rq3':4},'environment_versions':versions},immutable=True)
    print(json.dumps({'frozen_candidates':1226,'included_inventory':539,'returned_zip_in_primary':sum(c.eligibility.eligibility_state=='include' and c.location.returned_postal_code in TARGET_ZIPS for c in candidates),'environment':versions}),flush=True)


def run(kind):
    socket.socket.connect=deny_network
    socket.create_connection=deny_network
    verify_files(ROOT,read_json(OUTPUT/'implementation.json')['files'])
    candidates,lineage=load_candidates()
    if lineage!=read_json(OUTPUT/'candidate_lineage.json')['rows']:raise ValueError('candidate lineage changed')
    print('Validated final 1226 classifications and frozen September 5 locations; loading local Board/TIGER/ACS.',flush=True)
    context=load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',TARGET_ZIPS)
    meta={'run_id':BASE.name,'user_stage':'Phase 3 deterministic downstream','phase2_closure_sha256':digest((CLOSURE/'final_closure_freeze.json').read_bytes())}
    hashes={'phase2_final_closure':meta['phase2_closure_sha256'],'closed_candidate_lineage':digest((OUTPUT/'candidate_lineage.json').read_bytes())}
    function=run_linkage_phase if kind=='rq2' else run_spatial_phase
    print(f'Running unchanged {kind} implementation.',flush=True)
    manifest=function(candidates,context=context,target_zips=TARGET_ZIPS,run_metadata=meta,source_hashes=hashes,output_root=OUTPUT/kind)
    verify_files(OUTPUT/kind,manifest['artifacts'])
    print(json.dumps({'phase':kind,'status':manifest['status'],'artifacts':len(manifest['artifacts']),'checks':manifest['checks']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','rq2','rq3'])
    args=parser.parse_args();prepare() if args.command=='prepare' else run(args.command)
