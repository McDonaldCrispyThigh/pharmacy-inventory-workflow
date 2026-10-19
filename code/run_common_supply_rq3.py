"""Recompute RQ3 with a shared final-candidate ZIP frame; no RQ1/RQ2 execution."""
from pathlib import Path
import inspect, json, os, re, shutil, socket, sys, types
os.environ['PROJ_NETWORK']='OFF'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_expanded_final as frozen
import pandas as pd
import numpy as np

OUT=ROOT/'analysis/rq3_common_supply/20260908_v2'
BASE=frozen.OUT
THESIS=ROOT/'docs/thesis'
def save(path,value):frozen.write_json(path,value,immutable=True)

def common_engine(frame,count):
    original=frozen.downstream
    env=dict(vars(original));env['FROZEN_CANDIDATE_COUNT']=count
    copied={}
    for name,value in vars(original).items():
        if inspect.isfunction(value) and value.__module__==original.__name__:
            copied[name]=types.FunctionType(value.__code__,env,name,value.__defaults__,value.__closure__)
            copied[name].__kwdefaults__=value.__kwdefaults__
    env.update(copied)
    def same_frame(candidates,context,target_zips):
        google,_=original._google_and_boards(candidates,context,target_zips)
        google=google.loc[google.Zip_Code.isin(frame)].copy()
        boards=original.prepare_v2_board_frames(context.board,target_zips,observed_google_zips=sorted(frame))
        assert set(google.Zip_Code)<=frame
        assert set(boards.screened_extended.zip5)<=frame
        assert set(boards.full_active_extended.zip5)<=frame
        return google,boards
    env['_google_and_boards']=same_frame
    finalizer=original.finalizer
    def extension(google,board,delivery):
        result=finalizer.spatial_sensitivity_extension(google,board,delivery)
        cf=result['collection_frame']
        assert cf['zip_count']==len(frame)
        cf.update({'rule':'Union of 60 target ZIPs and all valid returned five-digit ZIPs in the final 2,062 candidates; identical admission frame for workflow and Board.',
                   'represented_zcta_count':len(frame)-len(cf['missing_zcta_geometry']),
                   'geometry_rule':'Union of all cached corresponding ZCTAs; nonrepresented postal ZIPs retain supply membership and receive no invented polygon.'})
        result['conditional_assumptions']['missing_geometry']='If any missing-coordinate Board mailing ZIP has no cached ZCTA geometry, minimum possible distance is zero for every origin; observed within-threshold records remain definite within.'
        return result
    env['finalizer']=types.SimpleNamespace(**{**vars(finalizer),'spatial_sensitivity_extension':extension})
    return types.SimpleNamespace(**{**copied,'_google_and_boards':same_frame})

def main():
    socket.socket.connect=frozen.old.deny_network;socket.create_connection=frozen.old.deny_network
    assert not OUT.exists(),'Use a new version for another run.'
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json',{
        'status':'user_authorized_primary_RQ3_common_final_supply_frame',
        'rule':'60 target ZIPs union all valid returned five-digit ZIPs among final candidates, regardless of classification; derive count from records.',
        'workflow':'Included records with usable coordinates whose returned ZIP belongs to the common frame.',
        'board':'Screened active Minnesota Board records with mailing ZIP in the same common frame; use usable cached coordinates.',
        'rq2':'Unchanged fixed 60 target ZIPs; linkage matches never filter spatial supply.',
        'fixed':'327 TIGER internal points; EPSG:26915; projected Euclidean nearest distance; 0.5, 1, 2 miles; ACS variables, ordering, quartiles, all existing sensitivity algorithms.',
        'boundary':'Corresponding cached ZCTA union for the final common supply ZIP set. Report every missing geometry explicitly; no polygon substitution.',
        'missing_geometry':'Retain the existing zero-minimum-distance rule for any missing-coordinate Board mailing ZIP without ZCTA geometry.',
        'delivery':'New versioned RQ3 outputs and specified thesis text/tables only; figures and existing PDF/DOCX preserved for the later figure update.',
        'new_api_calls':0})
    # Record current manuscript and figure state before scientific or editorial changes.
    protected=[p for p in THESIS.rglob('*') if p.is_file() and p.suffix in {'.tex','.bib','.pdf','.docx','.png','.svg'}]
    save(OUT/'manuscript_before_hashes.json',{p.relative_to(ROOT).as_posix():frozen.sha(p) for p in protected})
    for p in [*THESIS.glob('chapters/*.tex'),*THESIS.glob('tables/*.tex')]:
        dest=OUT/'before'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Revalidate every saved classification through the accepted local loader.
    candidates,lineage=frozen.load_candidates()
    assert lineage==frozen.read_json(BASE/'candidate_lineage.json')['rows']
    assert len(candidates)==2062 and len({c.location.place_id for c in candidates})==len(candidates)
    rows=[]
    for c in candidates:
        raw=c.location.returned_postal_code
        zipcode=str(raw).strip() if raw is not None else ''
        valid=bool(re.fullmatch(r'[0-9]{5}',zipcode))
        rows.append({'place_id':c.location.place_id,'returned_zip':zipcode,'valid_five_digit_zip':valid,'eligibility':c.eligibility.eligibility_state})
    ledger=pd.DataFrame(rows);ledger.to_csv(OUT/'candidate_zip_ledger.csv',index=False)
    returned=set(ledger.loc[ledger.valid_five_digit_zip,'returned_zip'])
    target=set(frozen.old.TARGET_ZIPS)
    assert all(re.fullmatch(r'[0-9]{5}',z) for z in target)
    frame=target|returned
    oldframe=set(frozen.read_json(BASE/'frames.json')['supply_99'])
    save(OUT/'common_supply_frame.json',{'rule':'target ZIP union final candidate valid returned ZIPs','candidate_count':len(candidates),
        'target_zip_count':len(target),'valid_returned_zip_count':len(returned),'common_zip_count':len(frame),
        'common_zips':sorted(frame),'target_zips':sorted(target),'valid_returned_zips':sorted(returned),
        'invalid_or_missing_candidate_zip_count':int((~ledger.valid_five_digit_zip).sum()),
        'added_vs_previous':sorted(frame-oldframe),'removed_vs_previous':sorted(oldframe-frame),
        'candidate_zip_ledger_sha256':frozen.sha(OUT/'candidate_zip_ledger.csv')})
    save(OUT/'implementation.json',{'files':{p:frozen.sha(ROOT/p) for p in ['code/run_common_supply_rq3.py','code/run_expanded_final.py',*frozen.downstream.IMPLEMENTATION_FILES]}})
    context=frozen.delivery.load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',frozen.old.TARGET_ZIPS)
    engine=common_engine(frame,len(candidates))
    print(f'Common frame derived: {len(frame)} ZIPs from {len(candidates)} candidates; computing primary and predefined sensitivities.',flush=True)
    result=engine.run_spatial_phase(candidates,context=context,target_zips=frozen.old.TARGET_ZIPS,
        run_metadata={'run_id':'rq3_common_final_supply_20260908_v2','common_supply_zip_count':len(frame),'primary_linkage_zip_count':len(target),'new_model_calls':0,'new_provider_calls':0},
        source_hashes={'runner':frozen.sha(Path(__file__)),'protocol':frozen.sha(OUT/'protocol.json'),'common_frame':frozen.sha(OUT/'common_supply_frame.json'),'expanded_candidate_freeze':frozen.sha(BASE/'final_freeze.json')},output_root=OUT/'rq3')
    frozen.verify_files(OUT/'rq3',result['artifacts'])
    google,boards=engine._google_and_boards(candidates,context,frozen.old.TARGET_ZIPS)
    counts={'workflow_included_records':int(google.eligibility_state.eq('include').sum()),'screened_board_records':len(boards.screened_extended),'all_active_board_records':len(boards.full_active_extended)}
    save(OUT/'supply_counts.json',counts)
    current=pd.read_csv(OUT/'rq3/tract_spatial.csv',dtype={'GEOID':str})
    prior=pd.read_csv(BASE/'rq3/tract_spatial.csv',dtype={'GEOID':str})
    assert current.GEOID.tolist()==prior.GEOID.tolist() and len(current)==327
    primary_columns=['GEOID','google_include_nearest_m','board_screened_nearest_m','income_quartile','nonwhite_quartile']
    assert current[primary_columns].equals(prior[primary_columns])
    current_manifest=frozen.read_json(OUT/'rq3/run_manifest.json')
    assert current_manifest['origin_sha256']==frozen.read_json(BASE/'rq3/run_manifest.json')['origin_sha256']
    save(OUT/'initial_verification.json',{'all_artifact_hashes_passed':True,'same_origin_hash':True,'primary_columns_identical_to_previous':True,'supply_counts':counts})
    print(json.dumps({'complete':True,'common_zips':len(frame),**counts,'origin_count':len(current)}),flush=True)

if __name__=='__main__':main()
