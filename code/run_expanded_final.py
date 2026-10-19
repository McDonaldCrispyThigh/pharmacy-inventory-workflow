"""Versioned offline expanded-frame analysis; upstream scientific code stays frozen."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
import argparse, inspect, json, os, shutil, socket, sys, types
os.environ['PROJ_NETWORK']='OFF'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_closed_phase3 as old
import run_supplemental_836 as supplemental
import corrected_downstream as downstream
import forced_choice_delivery as delivery
import audit_forced_choice_results as audit
from clean_phase2_closure_runtime import verify_completed
from clean_phase2_closure_contract import validate_decision
from clean_phase2_evidence import output_text
from wayback_snapshot_fetcher import read_json,write_json,write,digest,encoded,now
from clean_archive_preflight import verify_files

ROOT=old.ROOT
OUT=ROOT/'analysis/expanded_final/20260908_v1'
SUP=supplemental.ROOT
PRIOR=old.OUTPUT
GRID=ROOT/'analysis/maintenance/saturated_zip_grid_20260907_v1'
REFINE=ROOT/'analysis/maintenance/saturated_zip_grid_55369_refinement_20260907_v2'
AUTH=Path(r'C:\Users\16067\.codex\attachments\866f8f88-9def-42cf-9609-07ab5d92f43f\pasted-text.txt')

def save(path,value):write_json(path,value,immutable=True)
def sha(path):return digest(path.read_bytes())
def sealed(root,name='final_freeze.json'):
    value=read_json(root/name);verify_files(root,value['files']);return value

def discovery_provenance():
    found=defaultdict(list)
    for root in (GRID,REFINE):
        for path in sorted((root/'queries').rglob('result.json')):
            r=read_json(path);job=r['job']
            proof={'result_file':path.relative_to(ROOT).as_posix(),'result_sha256':sha(path),
                   'query_job':job['id'],'query_definition_id':job['query_definition_id'],
                   'source_zip':job['source_zip'],'level':job['level'],'bounds':job['bounds'],
                   'finished_at':r['finished_at'],'method':'bounded_spatial_restriction',
                   'state':r['state'],'saturated':r['saturated']}
            for pid in set(r['ids']):found[pid].append(proof)
    return dict(found)

def load_candidates():
    candidates,old_lineage=old.load_candidates()
    candidates=list(candidates);lineage=[];found=discovery_provenance()
    for item,row in zip(candidates,old_lineage):
        lineage.append({**row,'cohort':'original_1226','retrieval_pathways':['base_zip_text']+(['supplemental_spatial'] if item.location.place_id in found else []),
                        'supplemental_search_provenance':found.get(item.location.place_id,[])})
    for row in read_json(SUP/'supplemental_classification_index.json'):
        pid=row['place_id'];origin=SUP/row['origin'];record=read_json(origin/f'private/inputs/{pid}.json')
        result=verify_completed(origin,record,read_json(SUP/row['result_file']))
        if result['status']=='completed':
            response=read_json(origin/result['response_file'])
            decision,reducer=validate_decision(record,json.loads(output_text(json.loads(response['raw_response_utf8']))))
        else:decision=None
        path=SUP/f'private/details/{pid}.json';receipt=read_json(path)
        assert digest(encoded(receipt['payload']))==receipt['sha256']
        details=receipt['payload']['details'];sources=receipt['payload']['source_responses']
        base=read_json(SUP/f'classification/private/inputs/{pid}.json')
        assert {k:v for k,v in details.items() if k!='reviews'}==base['google_metadata']
        assert sha(path)==base['frozen_details']['sha256']
        proofs=found[pid];query=proofs[0]
        provenance=old.QueryProvenanceV2(query['query_definition_id'],query['source_zip'],'places_search')
        location=old.candidate_location_from_details(details,provenance,datetime.fromisoformat(sources[0]['retrieved_at'].replace('Z','+00:00')))
        item=delivery.ForcedChoiceDeliveryCandidate(location,decision,None if decision else result['status'])
        assert item.eligibility.to_public_dict()==result['reducer']
        candidates.append(item)
        lineage.append({**item.lineage_record(),'cohort':'supplemental_836','retrieval_pathways':['supplemental_spatial'],
            'supplemental_search_provenance':proofs,'closed_result_file':(SUP/row['result_file']).relative_to(ROOT).as_posix(),
            'closed_result_sha256':sha(SUP/row['result_file']),'original_input_packet_sha256':base['packet_sha256'],
            'effective_input_packet_sha256':record['packet_sha256'],'frozen_details_sha256':sha(path),
            'details_source_responses':sources,'eligibility_state':item.eligibility.eligibility_state})
    assert len(candidates)==len({c.location.place_id for c in candidates})==2062
    assert Counter(c.eligibility.eligibility_state for c in candidates)=={'include':592,'exclude':1470}
    return tuple(candidates),sorted(lineage,key=lambda r:r['place_id'])

def prepare():
    if (OUT/'phase0_verification.json').exists():
        verify_files(ROOT,read_json(OUT/'implementation.json')['files']);return
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'user_authorization.txt',AUTH.read_bytes(),immutable=True)
    checks={}
    for label,root,name in [('supplemental',SUP,'final_freeze.json'),('prior_analysis',PRIOR,'final_freeze.json'),
                            ('grid',GRID,'final_manifest.json'),('refinement',REFINE,'final_manifest.json')]:
        value=sealed(root,name);checks[label]={'files':len(value['files']),'manifest_sha256':sha(root/name)}
        print('Verified frozen package',label,len(value['files']),flush=True)
    supplemental.check_pins()
    protected=read_json(SUP/'preservation_before.json')
    volatile=set(read_json(SUP/'preservation_policy.json')['volatile_office_owner_files'])
    for relative,expected in protected.items():
        if relative in volatile:continue
        path=ROOT/relative
        assert sha(path)==expected['sha256'] and path.stat().st_mtime_ns==expected['mtime_ns'],relative
    candidates,lineage=load_candidates()
    old_ids={r['place_id'] for r in read_json(SUP/'old_final_index.json')};new_ids=set(read_json(SUP/'input_place_ids.json'))
    assert (len(old_ids),len(new_ids),len(old_ids&new_ids))==(1226,836,0)
    assert old_ids|new_ids=={c.location.place_id for c in candidates}
    assert all(p['place_id'] in new_ids for p in read_json(SUP/'model_call_ledger.json'))
    frames=read_json(SUP/'frames.json')
    original_zip={c.location.returned_postal_code for c in candidates if c.location.place_id in old_ids}
    assert set(frames['supply_99'])==set(old.TARGET_ZIPS)|original_zip
    observed={c.location.returned_postal_code for c in candidates}
    save(OUT/'frames.json',frames)
    save(OUT/'frame_verification.json',{'fixed_primary_zips':frames['primary_60'],'fixed_supply_zips':frames['supply_99'],
         'original_supply_reconstructed':True,'expanded_observed_zip_count':len(observed),
         'observed_zips_outside_fixed_supply':sorted(observed-set(frames['supply_99'])),
         'board_and_collection_frame_policy':'Preserve the original fixed 99 ZIPs. All included workflow candidates remain eligible supply under the original workflow-supply rule.'})
    save(OUT/'candidate_lineage.json',{'candidate_count':2062,'rows':lineage})
    save(OUT/'retrieval_summary.json',{'base_candidates':1226,'supplemental_candidates':836,'combined_candidates':2062,
         'base_and_supplemental_pathway_overlap':len(old_ids&set(discovery_provenance())),
         'grid':read_json(GRID/'summary.json'),'grid_protocol':read_json(GRID/'protocol.json'),
         'refinement':read_json(REFINE/'summary.json'),'base_terms':3,'base_target_zips':60,
         'exhaustive_enumeration_claim':False,'causal_attribution_to_subdivision':False})
    pins=downstream.implementation_hashes()
    pins.update({p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__).resolve(),ROOT/'code/run_closed_phase3.py')})
    for name in pins:write(OUT/'implementation'/name,(ROOT/name).read_bytes(),immutable=True)
    save(OUT/'implementation.json',{'files':pins})
    # Preserve the entire existing thesis before the newly authorized synchronization.
    thesis={}
    for p in (ROOT/'docs/thesis').rglob('*'):
        if not p.is_file() or p.name.startswith('~$'):continue
        rel=p.relative_to(ROOT);dest=OUT/'thesis_before'/p.relative_to(ROOT/'docs/thesis')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        thesis[rel.as_posix()]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns}
    save(OUT/'thesis_before.json',thesis)
    save(OUT/'phase0_verification.json',{'passed':True,'packages':checks,'original_candidates':1226,
        'supplemental_candidates':836,'overlap':0,'union':2062,'terminal_supplemental':836,
        'original_classifications_unchanged':True,'thesis_unchanged_during_supplemental':True,
        'classifier_contract_unchanged':True,'no_signal_eligibility_filter':True,'new_model_calls':0,
        'preserved_persistent_files':len(protected)-len(volatile),'verified_at':now()})
    print('Phase 0 passed; expanded frame frozen.',flush=True)

def rq1():
    candidates,_=load_candidates();summaries={}
    old_ids={r['place_id'] for r in read_json(SUP/'old_final_index.json')}
    for cohort,items in [('original',tuple(c for c in candidates if c.location.place_id in old_ids)),
                         ('supplemental',tuple(c for c in candidates if c.location.place_id not in old_ids)),('combined',candidates)]:
        dest=OUT/'rq1'/cohort;dest.mkdir(parents=True,exist_ok=True)
        labels=delivery._eligibility_frame(items)
        for name,frame in [('eligibility.csv',labels),('candidate_inventory.csv',delivery._candidate_inventory(items)),
                           ('unresolved_cases.csv',labels.loc[labels.eligibility_state.eq('unresolved')])]:
            downstream._write_csv(dest/name,frame)
        summary=audit.audit_candidate_flow(dest)
        summary['target_designated_excluded_by_other_dimension']=int((labels.is_target.eq('yes')&labels.eligibility_state.eq('exclude')).sum())
        save(dest/'summary.json',summary);summaries[cohort]=summary
    save(OUT/'rq1/summary.json',summaries)
    save(OUT/'rq1/freeze.json',{'files':{p.relative_to(OUT/'rq1').as_posix():sha(p) for p in (OUT/'rq1').rglob('*') if p.is_file() and p.name!='freeze.json'}})
    print('RQ1 frozen',summaries['combined']['eligibility'],flush=True)

def configured_downstream(count=2062):
    """Reuse frozen function bytecode with explicit new count and fixed frame globals."""
    frame=set(read_json(OUT/'frames.json')['supply_99'])
    globals_=dict(vars(downstream));globals_['FROZEN_CANDIDATE_COUNT']=count
    namespace={}
    for name,value in vars(downstream).items():
        if inspect.isfunction(value) and value.__module__==downstream.__name__:
            namespace[name]=types.FunctionType(value.__code__,globals_,name,value.__defaults__,value.__closure__)
            namespace[name].__kwdefaults__=value.__kwdefaults__
    globals_.update(namespace)
    original_prepare=downstream.prepare_v2_board_frames
    def fixed_prepare(board,target_zips,observed_google_zips=()):
        return original_prepare(board,target_zips,observed_google_zips=sorted(frame))
    globals_['prepare_v2_board_frames']=fixed_prepare
    finalizer=downstream.finalizer
    source=inspect.getsource(finalizer.spatial_sensitivity_extension)
    start=source.index('    returned_zips = {');end=source.index('    boards = ',start)
    adapted=source[:start]+'    returned_zips = set(FIXED_SUPPLY_ZIPS)\n'+source[end:]
    adapted=adapted.replace('60 target ZIPs union ZIPs observed among all frozen forced-choice candidates',
                            'Original fixed 99-ZIP collection frame: 60 target ZIPs union original-candidate returned ZIPs')
    env=dict(vars(finalizer));env['FIXED_SUPPLY_ZIPS']=frame
    exec(compile(adapted,'<expanded-fixed-supply-adapter>','exec'),env)
    globals_['finalizer']=types.SimpleNamespace(**{**vars(finalizer),'spatial_sensitivity_extension':env['spatial_sensitivity_extension']})
    return types.SimpleNamespace(**namespace)

def run(kind):
    verify_files(ROOT,read_json(OUT/'implementation.json')['files'])
    verify_files(OUT/'rq1',read_json(OUT/'rq1/freeze.json')['files'])
    candidates,lineage=load_candidates();assert lineage==read_json(OUT/'candidate_lineage.json')['rows']
    context=delivery.load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',old.TARGET_ZIPS)
    engine=configured_downstream();fn=engine.run_linkage_phase if kind=='rq2' else engine.run_spatial_phase
    meta={'run_id':'expanded_2062_20260908_v1','candidate_frame':'original plus saturation-resolution supplemental',
          'fixed_supply_frame':99,'new_classifier_calls':0,'scientific_algorithms':'original frozen implementation; candidate count and fixed supply-frame binding adapted explicitly'}
    hashes={'expanded_candidate_lineage':sha(OUT/'candidate_lineage.json'),'expanded_runner':sha(Path(__file__)),
            'supplemental_final_freeze':sha(SUP/'final_freeze.json'),'fixed_frames':sha(OUT/'frames.json')}
    result=fn(candidates,context=context,target_zips=old.TARGET_ZIPS,run_metadata=meta,source_hashes=hashes,output_root=OUT/kind)
    verify_files(OUT/kind,result['artifacts'])
    if kind=='rq2':assert read_json(OUT/'rq2/rq1_summary.json')=={k:v for k,v in read_json(OUT/'rq1/combined/summary.json').items() if k!='target_designated_excluded_by_other_dimension'}
    print(kind,'complete',result['checks'],flush=True)

if __name__=='__main__':
    socket.socket.connect=old.deny_network;socket.create_connection=old.deny_network
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','rq1','rq2','rq3']);stage=p.parse_args().stage
    prepare() if stage=='prepare' else rq1() if stage=='rq1' else run(stage)
