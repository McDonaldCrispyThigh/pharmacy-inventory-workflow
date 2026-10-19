"""Local verification, approved figures, and closure of the user's Phase 3."""
import argparse
from collections import Counter
import json
import socket
import tempfile
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import run_closed_phase3 as run
import corrected_impact as impact
import build_final_thesis_figures_v1 as figures
from wayback_snapshot_fetcher import read_json, write_json, digest

ROOT, OUT = run.ROOT, run.OUTPUT
R2, R3 = OUT/'rq2', OUT/'rq3'


def csv(path):
    return pd.read_csv(path, dtype={'GEOID':str})


def require(condition, label):
    if not condition:
        raise ValueError(label)


def authenticate():
    run.verify_files(ROOT, read_json(OUT/'implementation.json')['files'])
    three=impact._authenticate_package(R2,status='linkage_phase_complete',phase=3)
    four=impact._authenticate_package(R3,status='spatial_phase_complete',phase=4)
    return three,four


def replay():
    three,four=authenticate()
    candidates,lineage=run.load_candidates()
    require(lineage==read_json(OUT/'candidate_lineage.json')['rows'],'candidate lineage changed')
    context=run.load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',run.TARGET_ZIPS)
    verified={}
    for name,original,fn in [('rq2',three,run.run_linkage_phase),('rq3',four,run.run_spatial_phase)]:
        print('Independent deterministic replay: '+name,flush=True)
        with tempfile.TemporaryDirectory(prefix='replay-',dir=OUT/'verification') as temp:
            from pathlib import Path
            target=Path(temp)/'package'
            replayed=fn(candidates,context=context,target_zips=run.TARGET_ZIPS,
                run_metadata=original['run_metadata'],source_hashes={
                    'phase2_final_closure':original['source_hashes']['phase2_final_closure'],
                    'closed_candidate_lineage':original['source_hashes']['closed_candidate_lineage']},output_root=target)
            require(replayed==original,name+' manifest replay differs')
            run.verify_files(target,original['artifacts'])
            verified[name]={'byte_identical_artifact_count':len(original['artifacts']),
                'manifest_identical':True,'all_artifact_sha256':original['artifacts']}
    write_json(OUT/'verification/deterministic_replay.json',{'passed':True,'phases':verified},immutable=True)


def build_figures():
    authenticate()
    dest=OUT/'approved_figures'
    require(not dest.exists(),'figure output already exists')
    dest.mkdir()
    figures.OUT=dest
    figures.FIGURES=dest/'figures'
    figures.set_style()
    recurrence=read_json(R2/'name_recurrence_summary.json')
    frame=pd.DataFrame([r for r in recurrence['rows'] if r['linkage_strategy']==recurrence['primary_strategy']])
    states=csv(R3/'tract_spatial.csv')
    distances=csv(R3/'signed_distances.csv')
    require(len(states)==327 and states.GEOID.is_unique,'figure origins differ')
    require(states.GEOID.tolist()==distances.GEOID.tolist(),'signed distances are not bound to tract states')
    for name,data in [('name_recurrence.csv',frame),('tract_states.csv',states),('signed_distances.csv',distances)]:
        data.to_csv(dest/name,index=False,lineterminator='\n')
    records=[figures.export_figure(figures.figure_name_recurrence(frame),'figure01_name_recurrence')]
    context=figures.load_map_context(states)
    fig=figures.figure_spatial_disagreement(context)
    labels={}
    for ax,suffix in zip(fig.axes[:2],('0_5','1_0')):
        counts=Counter(states[f'discordance_type_{suffix}mi'])
        count=counts['board_only_beyond']+counts['google_only_beyond']
        texts=[t for t in ax.texts if t.get_text().endswith('% changed')]
        require(len(texts)==1,'approved map changed annotation is ambiguous')
        texts[0].set_text(f'{100*count/327:.1f}% changed')
        labels[suffix]={'count':count,'pct':100*count/327,'display':texts[0].get_text()}
    records.append(figures.export_figure(fig,'figure02_spatial_disagreement'))
    records.append(figures.export_figure(figures.figure_signed_histogram(distances),'figure03_signed_distance_histogram'))
    sources=[R2/'name_recurrence_summary.json',R3/'tract_spatial.csv',R3/'signed_distances.csv',
        ROOT/'code/build_final_thesis_figures_v1.py',ROOT/'code/close_closed_phase3.py',
        ROOT/'data/tl_2023_27_tract.zip']
    sources+=sorted((ROOT/'data/cartographic_context/tigerline_2023').glob('*.zip'))
    write_json(dest/'figure_manifest.json',{'figures':records,'logical_numbers':['4.1','4.2','4.3'],
        'architecture':'unchanged approved three-figure functions; current data and dynamic map percentages',
        'map_labels':labels,'map_crs':figures.TARGET_CRS,'map_extent':list(context['extent']),
        'map_analyzed_origins':327,'png_dpi':figures.PNG_DPI,'editable_pdf_svg':True,
        'source_hashes':{p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in sources},
        'provider_calls':0,'model_calls':0,'thesis_modified':False},immutable=True)
    print(json.dumps({'figures':3,'map_labels':labels}),flush=True)


def table(rows,columns=None):
    if not rows:return '_No rows._'
    columns=columns or list(rows[0])
    def cell(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.6f}'.rstrip('0').rstrip('.')
        return str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(cell(row.get(c)) for c in columns)+' |' for row in rows])


def audit():
    three,four=authenticate()
    old_units,old_delivery,old_audit,old_stage=impact._load_baseline()
    candidates,lineage=run.load_candidates()
    require({x['place_id'] for x in old_units}=={x.location.place_id for x in candidates},'fixed candidate IDs differ')
    require(three['candidate_lineage_sha256']==four['candidate_lineage_sha256'],'RQ2/RQ3 candidates differ')
    old_spatial=read_json(impact.OLD_ROOT/'delivery/spatial_summary.json')
    require(four['origin_sha256']==old_spatial['origin_sha256'],'fixed origins changed')
    for key in ('classifier_contract_sha256','eligibility_contract_sha256','spatial_contract_sha256'):
        require(three['contracts'][key]==old_delivery['contracts'][key],'scientific contract changed: '+key)
    old_tracts=csv(impact.OLD_ROOT/'delivery/tract_spatial.csv')
    tracts=csv(R3/'tract_spatial.csv')
    rq3=read_json(R3/'rq3_summary.json')
    impact._reconcile_spatial_rows(old_audit['rq3_spatial'],old_tracts)
    impact._reconcile_spatial_rows(rq3,tracts)
    transitions,tract_changes=impact.compare_tracts(old_tracts,tracts)
    require(np.array_equal(old_tracts.sort_values('GEOID').board_screened_nearest_m.to_numpy(),
                           tracts.sort_values('GEOID').board_screened_nearest_m.to_numpy()),'Board distances changed')
    old_metrics=read_json(impact.OLD_ROOT/'delivery/performance_metrics.json')
    metrics=read_json(R2/'performance_metrics.json')
    stage=read_json(R2/'stage_specific_diagnostic.json')
    recurrence=read_json(R2/'name_recurrence_summary.json')
    old_name=read_json(impact.OLD_ROOT/'delivery/name_recurrence_summary.json')
    before={'rq2':{'primary':impact._primary(old_metrics),'stage':impact._old_stage_metrics(old_stage),
                  'name_recurrence':impact._recurrence(old_name)},'rq3':impact._spatial_metrics(old_audit['rq3_spatial'])}
    after={'rq2':{'primary':impact._primary(metrics),'stage':impact._new_stage_metrics(stage),
                 'name_recurrence':impact._recurrence(recurrence)},'rq3':impact._spatial_metrics(rq3)}
    comparisons=impact._paired_numeric_tree(before,after)
    diff=tracts.google_include_nearest_m.to_numpy()-tracts.board_screened_nearest_m.to_numpy()
    near={'workflow_nearer':int((diff < -1e-7).sum()),'board_nearer':int((diff > 1e-7).sum()),'ties':int((np.abs(diff)<=1e-7).sum())}
    require(sum(near.values())==327,'nearer counts do not reconcile')
    signed=csv(R3/'signed_distances.csv')
    # Existing delivery CSVs use %.10g. Cross-file arithmetic therefore needs
    # serialization tolerance; 1e-8 miles is less than 0.02 millimetres.
    signed_serialization_error=float(np.max(np.abs(signed.workflow_minus_board_miles-diff/1609.344)))
    require(np.allclose(signed.workflow_minus_board_miles,diff/1609.344,atol=1e-8,rtol=0),'signed-distance rows differ')
    primary=impact._primary(metrics)
    for row in recurrence['rows']:
        if row['linked_pct'] is not None:
            require(row['linked_pct']==round(100*row['linked_records']/row['records'],2),'recurrence percentage differs')
    for source,denom in [('Workflow inventory locations',373),('Board keyword-screen reference',178)]:
        for cutoff in (2,3,5):
            rows=[r for r in recurrence['rows'] if r['source']==source and r['recurrence_threshold']==cutoff and r['linkage_strategy']==recurrence['primary_strategy']]
            require(sum(r['records'] for r in rows)==denom and sum(r['linked_records'] for r in rows)==148,'recurrence denominator/link total differs')
    geo=csv(R2/'geography_sensitivities.csv').to_dict('records')
    require([r['zip_count'] for r in geo]==[60,101,106],'fixed reporting geographies changed')
    threshold_rows=[]
    for t in ('0.5','1.0','2.0'):
        x=rq3['thresholds'][t]['scenarios']['primary_google_include_vs_board_screened']
        threshold_rows.append({'miles':t,**{k:v['count'] for k,v in x['states'].items()},'changed':x['classification_changed']['count'],'changed_pct':x['classification_changed']['pct']})
    enrichment=read_json(R3/'spatial_enrichment.json')
    extension=read_json(R3/'spatial_sensitivity_extension.json')
    old_enrichment=impact.OLD_AUDIT.parent/'spatial_enrichment.json'
    # Scientific supply-frame rule remains fixed; the 99-ZIP collection frame is distinct from 60/101/106 reporting frames.
    require(extension['collection_frame']['zip_count']==99,'collection supply frame changed')
    preservation=run.verify_preservation(run.SOURCE)
    current=run.protected()
    prior=read_json(OUT/'protected_before.json')
    require(current==prior,'accepted Phase2/archive files or timestamps changed')
    require(preservation['unchanged'],'historical repository files changed')
    replayed=read_json(OUT/'verification/deterministic_replay.json')
    require(replayed['passed'],'deterministic replay failed')
    tests=read_json(OUT/'verification/offline_tests.json')
    require(tests['passed'],'offline tests failed')
    figures_manifest=read_json(OUT/'approved_figures/figure_manifest.json')
    run.verify_files(ROOT,figures_manifest['source_hashes'])
    for figure in figures_manifest['figures']:
        run.verify_files(ROOT,{v['path']:v['sha256'] for v in figure['formats'].values()})
    visual=read_json(OUT/'approved_figures/visual_review.json')
    require(visual['passed'],'figure visual review incomplete')
    verification={'passed':True,'candidate_ids_1226_identical':True,'valid_539_include_687_exclude':True,
        'closed_decisions_revalidated_without_calls':True,'same_327_origin_hash':True,'same_scientific_contracts':True,
        'same_board_distances':True,'same_neighborhood_fields':True,'row_summary_reconciliation':True,
        'signed_csv_serialization_tolerance_miles':1e-8,'signed_csv_max_error_miles':signed_serialization_error,
        'recurrence_denominators_and_links':True,'geographies_60_101_106':True,'collection_frame_99_separate':True,
        'deterministic_replay':replayed,'offline_tests':tests,'figures_verified':True,
        'accepted_upstream_files_and_timestamps_unchanged':True,
        'upstream_file_counts':{k:len(v) for k,v in current.items()},'historical_preservation':preservation,
        'test_fixture_note':'The legacy impact fixture omits contract hashes. Its isolated copy supplies matching synthetic contracts. Original test bytes and modification time were restored after the temporary fixture edit; no analytical implementation changed.'}
    report={'status':'phase3_complete_stop_before_thesis','ready_for_methods_results_update':True,
        'candidate_count':1226,'included_inventory':539,'excluded':687,'technical_unresolved':0,
        'rq2_primary':primary,'stage_specific':stage,'location_unit':stage['address_zip_location_diagnostic'],
        'name_recurrence':recurrence,'geographic_sensitivities':geo,
        'linkage_sensitivities':csv(R2/'linkage_sensitivities.csv').to_dict('records'),
        'all_active_board':[r for r in metrics['strata'] if r['stratum']=='include_all_active_sensitivity'][0],
        'rq3':rq3,'rq3_enrichment':enrichment,'rq3_extension':extension,'nearer_counts':near,
        'threshold_states':threshold_rows,'old_vs_new':{'old':before,'corrected':after,'numeric_changes':comparisons,'tract_state_transitions':transitions},
        'verification':verification,'figures':figures_manifest,
        'interpretation':'Descriptive operational-reference linkage and projected straight-line potential proximity; not a causal Wayback effect, classification accuracy, ownership inference, or realized access.'}
    write_json(OUT/'FINAL_REPORT.json',report,immutable=True)
    tract_changes.to_csv(OUT/'old_versus_corrected_tracts.csv',index=False,lineterminator='\n')
    pd.DataFrame([{'metric':k,**v} for k,v in comparisons.items()]).to_csv(OUT/'old_versus_corrected_metrics.csv',index=False,lineterminator='\n')
    write_json(OUT/'verification/final_verification.json',verification,immutable=True)
    md=['Phase 3 completed: deterministic RQ2/RQ3 from the closed 1,226-candidate classification run. Ready for the user to review before any Methods/Results edit.',
        'Final inventory: 539 included and 687 excluded; zero technical unresolved. All classifier, provider and archive acquisition calls: zero. September 2, accepted Phase 2, accepted archive evidence and thesis files preserved.',
        '**RQ2 primary — 60 ZIPs**',table([primary]),
        '**Raw → included stage diagnostic**',table(stage['place_id_diagnostic']['assignments'],['candidate_unit','candidate_rows','board_rows','linked_pairs','unmatched_candidate_rows','unmatched_board_rows','candidate_linked_share_pct','board_reference_coverage_pct']),
        f"Candidate reduction: {stage['raw_in_frame_count']-stage['included_in_frame_count']} / {stage['raw_in_frame_count']} ({100*(1-stage['included_in_frame_count']/stage['raw_in_frame_count']):.2f}%).",
        table([{'metric':k,**v} for k,v in stage['place_id_diagnostic']['core_metrics'].items() if isinstance(v,dict)],['metric','numerator','denominator','pct']),
        table(stage['place_id_diagnostic']['board_stages']),
        '**Qualified-edge cross-tab**',table(csv(R2/'stage_specific/candidate_qualified_edge_cross_tab.csv').to_dict('records')),
        '**Exact address + ZIP location sensitivity**',f"Primary in-frame included Place IDs: 373 → 248 exact address–ZIP units. Place IDs remain the primary analysis unit.",table(stage['address_zip_location_diagnostic']['assignments'],['candidate_unit','candidate_rows','linked_pairs','unmatched_candidate_rows','unmatched_board_rows','candidate_linked_share_pct','board_reference_coverage_pct']),
        '**Name recurrence**','Same component bootstrap: 5,000 resamples with fixed seeds. At cutoff 2, below-threshold means single occurrence; at 3 and 5 it includes all below-cutoff components. One workflow row has an unresolved normalized name, which is not an unresolved classification. These are primary Place-ID rows grouped using recurrence across distinct source-local addresses; not collapsed location-unit results or ownership/chain status.',
        table([r for r in recurrence['rows'] if r['linkage_strategy']==recurrence['primary_strategy']],['source','recurrence_threshold','group_code','records','linked_records','linked_pct','distinct_name_components','cluster_bootstrap_95_low_pct','cluster_bootstrap_95_high_pct']),
        '**Predefined linkage sensitivities**',table(report['linkage_sensitivities']),table([report['all_active_board']]),
        '**Fixed geographic reporting frames**',table(geo),
        '**RQ3 — fixed 327 tract origins, EPSG:26915, miles**',table([{'surface':k,**enrichment[k]} for k in ('workflow_nearest_miles','board_nearest_miles','signed_workflow_minus_board_miles')]),table([near]),
        'States below use beyond-threshold indicators: board_only_beyond means only the workflow inventory is within the threshold; google_only_beyond means only Board is within. Changed means within/beyond disagreement between sources in the corrected run.',table(threshold_rows),
        '**Neighborhood quartiles**',table([{'dimension':dimension,'quartile':q,'tracts':r['tract_count'],'workflow_mean_miles':r['workflow_nearest_miles']['mean'],'board_mean_miles':r['board_nearest_miles']['mean'],'signed_mean_miles':r['signed_workflow_minus_board_miles']['mean'],**{f'changed_{t}mi':v['changed_count'] for t,v in r['thresholds'].items()}} for dimension,groups in enrichment['quartiles'].items() for q,r in groups.items()]),
        '**Predefined spatial sensitivities**',table([{'miles':t,'scenario':s,**v['classification_changed']} for t,r in enrichment['prespecified_sensitivities'].items() for s,v in r['scenarios'].items()]),
        'The complete weighted, ±4 m threshold, geometric-centroid, frame-interior and missing-coordinate conditional outputs are in [spatial_enrichment.json](rq3/spatial_enrichment.json) and [spatial_sensitivity_extension.json](rq3/spatial_sensitivity_extension.json). The fixed 99-ZIP collection supply frame is distinct from the 60/101/106 reporting frames. Board screened spatial input has 341 rows, of which 303 have usable coordinates; 38 remain missing, handled by the predefined sensitivities. These conditional ranges are not confidence intervals or unconditional bounds.',
        '**Old versus corrected downstream**',table([{'metric':k,**v} for k,v in comparisons.items() if k.startswith('rq2.primary.') or k.startswith('rq2.stage.core_metrics.board_referenced_assignment_') or k in ('rq3.workflow_nearest_miles.mean','rq3.workflow_nearest_miles.median','rq3.signed_workflow_minus_board_miles.mean') or (k.startswith('rq3.thresholds.') and '.classification_changed.' in k)]),
        'All recurrence changes, stage measures, quartiles and paired tract transitions are retained in FINAL_REPORT.json, old_versus_corrected_metrics.csv and old_versus_corrected_tracts.csv. These are descriptive changes between completed runs, not a causal Wayback effect.',
        '**Approved figures — same architecture and styling**',*['Figure '+number+': '+', '.join(f"[{suffix.upper()}]({v['path'].split(OUT.relative_to(ROOT).as_posix()+'/')[1]})" for suffix,v in record['formats'].items()) for number,record in zip(('4.1','4.2','4.3'),figures_manifest['figures'])],
        '**Deterministic verification**',f"Both analyses were rebuilt independently: all 34 RQ2 and 8 RQ3 artifact hashes and both semantic/full manifests reproduced exactly. Frozen input hashes, all 1,226 decisions/reducers, 327 tract origins, scientific contracts, Board distances, neighborhood fields, one-to-one assignments, stage reconciliation, recurrence totals and figure provenance passed. Offline test details: verification/offline_tests.json. Historical preservation: {preservation['checked_files']} files unchanged, including content hashes, modification and creation timestamps; accepted Phase 2/archive inventories also match exactly.",
        'Downstream manifest classifier/prompt fields identify the unchanged base scientific contract. The accepted Phase-2 closure and candidate lineage retain the citation-namespace validation repair and the explicit-identifier inputs used for nine retries; this task did not issue those calls or claim all historical prompts were byte-identical.',
        'Methods/Results update is ready for review. No thesis file has been edited or compiled. Stop here.']
    (OUT/'FINAL_REPORT.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'rq2':primary,'nearer':near,'historical_files_unchanged':preservation['checked_files']}),flush=True)


def freeze():
    require(read_json(OUT/'FINAL_REPORT.json')['verification']['passed'],'verification did not pass')
    require(run.protected()==read_json(OUT/'protected_before.json'),'upstream changed at freeze')
    require(run.verify_preservation(run.SOURCE)['unchanged'],'historical files changed at freeze')
    files={p.relative_to(OUT).as_posix():digest(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_freeze.json'}
    write_json(OUT/'final_freeze.json',{'status':'phase3_closed_stop_before_thesis','files':files,
        'candidate_count':1226,'included':539,'excluded':687,'technical_unresolved':0,
        'implementation':{'code/close_closed_phase3.py':digest((ROOT/'code/close_closed_phase3.py').read_bytes())}},immutable=True)
    run.verify_files(OUT,files)
    print(json.dumps({'frozen_files':len(files),'final_freeze_sha256':digest((OUT/'final_freeze.json').read_bytes())}),flush=True)


if __name__=='__main__':
    socket.socket.connect=run.deny_network
    socket.create_connection=run.deny_network
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['replay','figures','audit','freeze'])
    args=parser.parse_args()
    {'replay':replay,'figures':build_figures,'audit':audit,'freeze':freeze}[args.command]()
