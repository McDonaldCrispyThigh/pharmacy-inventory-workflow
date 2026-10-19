"""Offline verification and publication data for the expanded frozen candidate frame."""
import argparse, json, socket, sys, tempfile, subprocess, types
from pathlib import Path
import numpy as np
import pandas as pd
import run_expanded_final as run
import close_closed_phase3 as previous
import corrected_impact as impact
from wayback_snapshot_fetcher import read_json,write_json
ROOT,OUT,PRIOR=run.ROOT,run.OUT,run.PRIOR
R2,R3=OUT/'rq2',OUT/'rq3'
csv=previous.csv

def authenticate():
    run.verify_files(ROOT,read_json(OUT/'implementation.json')['files'])
    fn=types.FunctionType(impact._authenticate_package.__code__,{**vars(impact),'EXPECTED_CANDIDATES':2062})
    fn.__kwdefaults__=impact._authenticate_package.__kwdefaults__
    return (fn(R2,status='linkage_phase_complete',phase=3),fn(R3,status='spatial_phase_complete',phase=4))

def replay():
    original=authenticate();candidates,_=run.load_candidates()
    context=run.delivery.load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',run.old.TARGET_ZIPS)
    engine=run.configured_downstream();verified={}
    (OUT/'verification').mkdir(exist_ok=True)
    for kind,manifest,fn in zip(('rq2','rq3'),original,(engine.run_linkage_phase,engine.run_spatial_phase)):
        print('Independent deterministic replay:',kind,flush=True)
        with tempfile.TemporaryDirectory(prefix='replay-',dir=OUT/'verification') as temp:
            target=Path(temp)/'package'
            keys=('expanded_candidate_lineage','expanded_runner','supplemental_final_freeze','fixed_frames')
            result=fn(candidates,context=context,target_zips=run.old.TARGET_ZIPS,
                run_metadata=manifest['run_metadata'],source_hashes={k:manifest['source_hashes'][k] for k in keys},output_root=target)
            assert result==manifest,kind+' manifest differs'
            run.verify_files(target,manifest['artifacts'])
            verified[kind]={'manifest_identical':True,'byte_identical_artifacts':len(manifest['artifacts'])}
    run.save(OUT/'verification/deterministic_replay.json',{'passed':True,'phases':verified})

def figures():
    previous.OUT=OUT;previous.R2=R2;previous.R3=R3;previous.authenticate=authenticate
    previous.build_figures()

def audit():
    two,three=authenticate()
    old2=impact._authenticate_package(PRIOR/'rq2',status='linkage_phase_complete',phase=3)
    old3=impact._authenticate_package(PRIOR/'rq3',status='spatial_phase_complete',phase=4)
    assert two['contracts']==old2['contracts'] and three['contracts']==old3['contracts']
    assert three['origin_sha256']==old3['origin_sha256']
    tracts=csv(R3/'tract_spatial.csv');prior=csv(PRIOR/'rq3/tract_spatial.csv')
    assert len(tracts)==327 and tracts.GEOID.is_unique and tracts.GEOID.tolist()==prior.GEOID.tolist()
    fixed=[c for c in tracts.columns if not any(s in c for s in ('google','discordance','delta','changed'))]
    # Compare original demographics, origins and Board surfaces; combined scenario flags vary legitimately.
    fixed=[c for c in fixed if c in prior and not c.startswith(('both_','conditional_'))]
    same=[]
    for c in fixed:
        if tracts[c].equals(prior[c]):same.append(c)
    assert np.array_equal(tracts.board_screened_nearest_m,prior.board_screened_nearest_m)
    for c in ('GEOID','median_household_income','pct_non_white','income_quartile','nonwhite_quartile'):
        if c in tracts:assert c in same,c
    metrics=read_json(R2/'performance_metrics.json');stage=read_json(R2/'stage_specific_diagnostic.json')
    rec=read_json(R2/'name_recurrence_summary.json');rq3=read_json(R3/'rq3_summary.json')
    enrich=read_json(R3/'spatial_enrichment.json');ext=read_json(R3/'spatial_sensitivity_extension.json')
    impact._reconcile_spatial_rows(rq3,tracts)
    assert ext['collection_frame']['zip_count']==99 and metrics['board_frames']['extended_zip_count']==99
    diff=(tracts.google_include_nearest_m-tracts.board_screened_nearest_m)/1609.344
    assert np.allclose(csv(R3/'signed_distances.csv').workflow_minus_board_miles,diff,atol=1e-8,rtol=0)
    for source,denom in [('Workflow inventory locations',413),('Board keyword-screen reference',178)]:
        for cutoff in (2,3,5):
            rows=[r for r in rec['rows'] if r['source']==source and r['recurrence_threshold']==cutoff and r['linkage_strategy']==rec['primary_strategy']]
            assert sum(r['records'] for r in rows)==denom and sum(r['linked_records'] for r in rows)==150
    for groups in enrich['quartiles'].values():assert sum(r['tract_count'] for r in groups.values())==327
    geo=csv(R2/'geography_sensitivities.csv').to_dict('records')
    assert [r['zip_count'] for r in geo]==[60,101,106]
    before={};after={}
    for dest,base in ((before,PRIOR),(after,OUT)):
        dest.update({'rq1':read_json(base/'rq2/rq1_summary.json'),
            'rq2':{'primary':impact._primary(read_json(base/'rq2/performance_metrics.json')),
                   'stage':impact._new_stage_metrics(read_json(base/'rq2/stage_specific_diagnostic.json')),
                   'name_recurrence':impact._recurrence(read_json(base/'rq2/name_recurrence_summary.json'))},
            'rq3':impact._spatial_metrics(read_json(base/'rq3/rq3_summary.json')),
            'neighborhoods':read_json(base/'rq3/spatial_enrichment.json')['quartiles']})
    differences=impact._paired_numeric_tree(before,after)
    transitions,rows=impact.compare_tracts(prior,tracts)
    rows.to_csv(OUT/'version_comparison_tracts.csv',index=False,lineterminator='\n')
    pd.DataFrame([{'metric':k,**v} for k,v in differences.items()]).to_csv(OUT/'version_comparison_metrics.csv',index=False,lineterminator='\n')
    run.save(OUT/'version_comparison.json',{'label':'original candidate-frame analysis versus saturation-corrected expanded candidate-frame analysis',
        'interpretation':'Descriptive version comparison; no causal attribution to retrieval date, subdivision, archive evidence or model.',
        'original':before,'expanded':after,'numeric_changes':differences,'tract_state_transitions':transitions})
    thresholds=[]
    for t in ('0.5','1.0','2.0'):
        v=rq3['thresholds'][t]['scenarios']['primary_google_include_vs_board_screened']
        assert sum(x['count'] for x in v['states'].values())==327
        thresholds.append({'miles':t,**{k:x['count'] for k,x in v['states'].items()},'changed':v['classification_changed']['count'],'changed_pct':v['classification_changed']['pct']})
    run.save(OUT/'analysis_report.json',{'candidate_count':2062,'rq1':read_json(OUT/'rq1/summary.json'),
        'rq2_primary':impact._primary(metrics),'stage_specific':stage,'name_recurrence':rec,
        'linkage_sensitivities':csv(R2/'linkage_sensitivities.csv').to_dict('records'),'geographic_sensitivities':geo,
        'all_active_board':next(r for r in metrics['strata'] if r['stratum']=='include_all_active_sensitivity'),
        'rq3':rq3,'enrichment':enrich,'extension':ext,'threshold_states':thresholds,
        'nearer_counts':{'workflow_nearer':int((diff < -1e-7).sum()),'board_nearer':int((diff > 1e-7).sum()),'ties':int((abs(diff)<=1e-7).sum())}})
    run.save(OUT/'verification/analysis_verification.json',{'passed':True,'same_scientific_contracts':True,
        'same_327_origin_hash':True,'same_board_distances':True,'identical_tract_columns':same,
        'six_way_and_quartile_rows_reconciled':True,'same_fixed_60_99_101_106_frames':True,
        'recurrence_denominators':True,'signed_distances_reconciled':True,'new_model_calls':0,'new_provider_calls':0})
    print(json.dumps({'rq2':impact._primary(metrics),'rq3':thresholds,'nearest':{k:enrich[k] for k in ('workflow_nearest_miles','board_nearest_miles','signed_workflow_minus_board_miles')}}),flush=True)

if __name__=='__main__':
    socket.socket.connect=run.old.deny_network;socket.create_connection=run.old.deny_network
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['replay','figures','audit'])
    globals()[parser.parse_args().stage]()
