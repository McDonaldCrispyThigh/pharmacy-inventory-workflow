"""Close the authorized expanded analyses and synchronized Chapters 1--4 only."""
import argparse,json,re,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
import fitz
from docx import Document
import run_expanded_final as run
import close_expanded_final as close
from wayback_snapshot_fetcher import read_json,now
ROOT,OUT=run.ROOT,run.OUT
THESIS=ROOT/'docs/thesis'
DOCX='From_Digital_Evidence_to_Pharmacy_Access_Introduction_through_Results.docx'
PDF='introduction_literature_methods_results_final_figures.pdf'

def preserve():
    records={}
    for label,root,name in [('supplemental',run.SUP,'final_freeze.json'),('prior_analysis',run.PRIOR,'final_freeze.json'),
        ('grid',run.GRID,'final_manifest.json'),('refinement',run.REFINE,'final_manifest.json')]:
        value=run.sealed(root,name);records[label]={'files_verified':len(value['files']),'manifest_sha256':run.sha(root/name)}
        print('Preserved',label,records[label]['files_verified'],flush=True)
    before=read_json(run.SUP/'preservation_before.json');volatile=set(read_json(run.SUP/'preservation_policy.json')['volatile_office_owner_files'])
    checked=0
    for name,expected in before.items():
        if name in volatile or name.startswith('docs/thesis/'):continue
        path=ROOT/name
        assert run.sha(path)==expected['sha256'] and path.stat().st_mtime_ns==expected['mtime_ns'],name
        checked+=1
    run.supplemental.check_pins()
    original=read_json(OUT/'thesis_before.json')
    untouched=[]
    for name,expected in original.items():
        if any(x in name.lower() for x in ('discussion','conclusion','abstract')):
            assert run.sha(ROOT/name)==expected['sha256'],name
            untouched.append(name)
    run.save(OUT/'verification/upstream_preservation.json',{'passed':True,'packages':records,
        'historical_nonmanuscript_persistent_files_verified':checked,'classifier_pins_unchanged':True,
        'unactivated_later_chapters_unchanged':untouched,'thesis_changes_authorized_by_current_user_instruction':True})

def numeric(text):
    text=text.replace('--','–').replace('−','-').replace(',','').replace('$','')
    return re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',text)

def consistency():
    close.authenticate();run.verify_files(OUT/'rq1',read_json(OUT/'rq1/freeze.json')['files'])
    assert read_json(OUT/'verification/deterministic_replay.json')['passed']
    assert read_json(OUT/'verification/offline_tests.json')['passed']
    assert read_json(OUT/'verification/upstream_preservation.json')['passed']
    assert read_json(OUT/'visual_qa/visual_review.json')['passed']
    structural=read_json(OUT/'visual_qa/document_structural_qa.json')
    assert structural['docx_structure']['passed']
    for name,path in [('pdf',THESIS/PDF),('word',OUT/'docx/plain_docx_review.pdf')]:
        assert structural[name]['file_sha256']==run.sha(path)
    d=Document(OUT/'docx'/DOCX)
    names=['table1_candidate_flow.tex','table2_board_correspondence.tex','table3_stage_specific_filtering.tex','table3_name_recurrence.tex','table4_paired_spatial.tex','table5_neighborhood.tex']
    table_checks={}
    for table,name in zip(d.tables[2:],names):
        text=(THESIS/'tables'/name).read_text(encoding='utf-8');body=text.split('\\midrule',1)[1].split('\\bottomrule')[0]
        lines=[z for z in body.splitlines() if z.rstrip().endswith('\\\\')]
        expected=[numeric(z) for z in lines]
        actual=[numeric(' '.join(c.text for c in row.cells)) for row in table.rows][-len(lines):]
        assert expected==actual,(name,expected,actual)
        table_checks[name]={'all_body_numeric_tokens_match_docx':True,'body_rows':len(lines)}
    tracts=pd.read_csv(OUT/'rq3/tract_spatial.csv',dtype={'GEOID':str})
    prior=pd.read_csv(run.PRIOR/'rq3/tract_spatial.csv',dtype={'GEOID':str})
    frozen=['GEOID','target_zcta','COUNTYFP','med_hh_income','total_pop','pct_nonwhite','total_hh','carless_rate','income_quartile','nonwhite_quartile']
    assert tracts[frozen].equals(prior[frozen])
    assert all(len(tracts.groupby(c))==4 for c in ('income_quartile','nonwhite_quartile'))
    sources={};obsolete={}
    for name in ['01_introduction_rewrite.tex','02_literature_rewrite.tex','03_methodology.tex','04_results.tex']:
        path=THESIS/'chapters'/name;text=path.read_text(encoding='utf-8')
        sources[path.relative_to(ROOT).as_posix()]=run.sha(path)
        obsolete[name]=[{'line':i+1,'text':line} for i,line in enumerate(text.splitlines()) if re.search(r'1,226|\b539\b|\b687\b|\b373\b|\b148\b|0\.660|0\.576|pilot passed|did not issue new Text',line)]
    # Historical numbers may occur only with explicit original-cohort provenance.
    results=(THESIS/'chapters/04_results.tex').read_text(encoding='utf-8')
    assert not re.search(r'\b373\b|\b148\b|0\.660|0\.576|pilot passed',results)
    wrapper=(THESIS/'introduction_literature_methods_results_final_figures.tex').read_text(encoding='utf-8')
    inputs=re.findall(r'\\input\{([^}]+)\}',wrapper)
    assert inputs==['chapters/01_introduction_rewrite','chapters/02_literature_rewrite','chapters/03_methodology','chapters/04_results']
    assert len(re.findall(r'\\includegraphics',results))==3
    method=(THESIS/'chapters/03_methodology.tex').read_text(encoding='utf-8')
    assert method.count('\\begin{equation}')==1
    log=(THESIS/'introduction_literature_methods_results_final_figures.log').read_text(encoding='utf-8')
    assert not re.search(r'Overfull|Undefined control|undefined references|Missing character',log)
    for fig in read_json(OUT/'approved_figures/figure_manifest.json')['figures']:
        for record in fig['formats'].values():
            assert run.sha(ROOT/record['path'])==record['sha256']
            assert run.sha(THESIS/'figures/results'/Path(record['path']).name)==record['sha256']
    # All requested sensitivity dimensions remain present in frozen outputs.
    r=read_json(OUT/'analysis_report.json')
    assert len(r['linkage_sensitivities'])==8
    for t in ('0.5','1.0','2.0'):
        threshold=r['rq3']['thresholds'][t]
        assert len(threshold['scenarios'])==4 and len(threshold['threshold_tolerance'])==3 and len(threshold['weighting'])==3
        assert {'collection_frame_interior','geometric_centroid_origin','conditional_ranges'}<=set(r['extension']['thresholds'][t])
    verification={'passed':True,'candidate_count':2062,'include':592,'exclude':1470,'technical_unresolved':0,
        'original_and_supplemental_frozen':True,'same_frozen_classifier_and_reducer':True,'no_provider_signal_filter':True,
        'same_327_origins_demographics_and_quartiles':True,'six_results_tables_match_word':table_checks,
        'all_predefined_linkage_and_spatial_sensitivities_recomputed':True,'exactly_three_results_figures':True,
        'one_formal_equation_as_image_in_word':True,'no_overfull_or_missing_character_tex_diagnostics':True,
        'manuscript_source_hashes':sources,'historical_value_scan':obsolete,
        'only_chapters_1_to_4_compiled':inputs,'new_model_calls':0,'new_provider_calls':0,
        'deterministic_replay':read_json(OUT/'verification/deterministic_replay.json'),
        'visual_review':read_json(OUT/'visual_qa/visual_review.json'),
        'skills_runtime_note':structural['docx_structure']['runtime_note']}
    run.save(OUT/'verification/final_verification.json',verification)
    shutil.copy2(OUT/'docx'/DOCX,THESIS/DOCX)
    assert run.sha(THESIS/DOCX)==run.sha(OUT/'docx'/DOCX)
    print('Final consistency passed; original working DOCX synchronized.',flush=True)

def freeze():
    verification=read_json(OUT/'verification/final_verification.json');assert verification['passed']
    assert run.sha(THESIS/DOCX)==run.sha(OUT/'docx'/DOCX)
    r=read_json(OUT/'analysis_report.json');r['qualified_edges']=read_json(OUT/'verification/qualified_edges.json')
    r['status']='expanded_final_complete_stop';r['verification']=verification
    r['new_model_calls']=r['new_provider_calls']=0
    r['archive_states_combined']={}
    old=read_json(ROOT/'data/clean_archive_closure/20260907_closure_v1/closure_preflight.json')['counts']['candidate_states']
    sup=read_json(run.SUP/'FINAL_REPORT.json')['archive_preflight']['archive_states']
    for key in old:r['archive_states_combined'][key]=old[key]+sup[key]
    assert sum(r['archive_states_combined'].values())==2062
    r['archive_text_candidates']=1128;r['usable_websites']=1850
    run.save(OUT/'FINAL_REPORT.json',r)
    rows=[{'cohort':name,'candidates':v['candidate_denominator'],**{k:z['count'] for k,z in v['eligibility'].items()}} for name,v in r['rq1'].items()]
    table=close.previous.table
    rec=r['name_recurrence'];rr=[v for v in rec['rows'] if v['linkage_strategy']==rec['primary_strategy']]
    md=['# Corrected expanded final analysis — complete',
      'Frozen union: 2,062 unique Place IDs (1,226 original + 836 supplemental; no overlap). Every supplemental candidate was classified; provider-signal categories were diagnostic only. The original classifications and evidence remain unchanged. This downstream and manuscript task made **zero model and zero provider calls**.',
      '## RQ1',table(rows),
      'All core judgments, field-specific confidence, exclusion combinations and target-designated exclusions are retained for each cohort in `rq1/summary.json`. Three target-designated records were excluded for identity inconsistency.',
      '## Primary RQ2 — fixed 60 ZIPs',table([r['rq2_primary']]),table([r['qualified_edges']]),
      'There are 1,748 raw in-frame candidates and 413 included rows. Raw assignment covers 154 Board identities; 150 remain covered (97.40% retention), with four lost (2.60%). The included 413 Place-ID rows collapse to 272 exact address–ZIP units: 150 links, 122 unmatched locations, 28 unmatched Board records; 55.15% location linked share. Of 263 unmatched Place-ID rows, 123 share a location with an already linked Place ID.',
      '## Name recurrence',table(rr,['source','recurrence_threshold','group_code','records','linked_records','linked_pct','cluster_bootstrap_95_low_pct','cluster_bootstrap_95_high_pct']),
      'Intervals use the same 5,000-component bootstrap. One workflow recurrence signature is unresolved; its classification is valid. Recurrence does not identify ownership.',
      '## RQ3 — fixed 327 origins; miles',table([{'surface':key,**r['enrichment'][key]} for key in ('workflow_nearest_miles','board_nearest_miles','signed_workflow_minus_board_miles')]),table([r['nearer_counts']]),table(r['threshold_states']),
      '`board_only_beyond` means workflow only within; `google_only_beyond` means Board only within. Each four-state row sums to 327.',
      'The Board/collection frame remains the original fixed 99 ZIPs. Expanded candidate records observe 102 returned ZIPs, with 55075, 55122 and 55144 outside the fixed 99. All 592 included workflow coordinates remain eligible under the original workflow supply rule; one included supplemental record lies outside the fixed Board frame. Board supply remains 341 screened records, 303 with usable coordinates and 38 without them.',
      '## All predefined sensitivities',table(r['linkage_sensitivities']),table([r['all_active_board']]),table(r['geographic_sensitivities']),
      'Completed: exact address–ZIP collapse; recurrence cutoffs 2/3/5; fixed 101/106 ZIP frames; include-plus-unresolved; all-active Board; exact cached Board coordinates; tract centroids; collection-frame interior; ±4 m tolerance; population and household weighting; missing-coordinate mailing-ZCTA and frame-edge conditional scenarios. Full results: `rq3/rq3_summary.json`, `rq3/spatial_enrichment.json`, `rq3/spatial_sensitivity_extension.json`. All eight neighborhood groups retain the same original quartiles. No new analytical specification was added.',
      '## Original candidate-frame analysis versus saturation-corrected expanded candidate-frame analysis',
      'The expanded version has 53 more included records, 40 more primary-frame workflow rows and two more primary links (148 → 150). Workflow linked share is 39.68% → 36.32%, Board coverage 83.15% → 84.27%, and raw-to-included Board retention 98.01% → 97.40%. Location units increase 248 → 272. Mean workflow nearest distance is 0.660 → 0.631 miles, median 0.576 → 0.558; signed mean −0.089 → −0.117. Threshold disagreement changes 43 → 48, 40 → 46, and 4 → 4. These are descriptive version differences, not causal effects of subdivision, collection date, archive evidence or classification. Detailed neighborhood and recurrence changes: `version_comparison.json`, `version_comparison_metrics.csv`, `version_comparison_tracts.csv`.',
      '## Retrieval and archive provenance',
      '112 saturated ZIP–term definitions across 59 ZIPs; 444 initial intersecting rectangles, four first-stage refinement children, and four further 55369 children. The first supplemental pass returned 1,672 unique IDs: 837 original IDs and 835 new. Further refinement added one new ID. Later spatially restricted retrieval is dated separately; no exhaustive-enumeration claim is made.',table([r['archive_states_combined']]),
      'Usable archive text: 1,128 / 1,850 website-bearing candidates (60.97%). Archive outcomes preserve collection failures and unusable evidence separately from completed bounded no-snapshot lookups. These accepted historical acquisition limitations remain in the evidence packets. Earlier supplemental classification used 897 model requests, recorded in its frozen ledger; none was repeated here.',
      '## Delivery and verification',
      'Six Results tables and exactly three approved Results figures were regenerated. Figure 3.1 shows bounded saturation resolution and a development-only pilot; fixed Board supply wording is explicit. Methods and Results are synchronized; Introduction has only retrieval-consistency edits; Literature Review is unchanged. No Discussion, Conclusion or Abstract was written or compiled.',
      'Independent replay reproduced all 34 RQ2 and eight RQ3 artifact hashes and both manifests. Seventeen offline tests passed. All table body numeric tokens match the editable Word tables. The canonical PDF and native Word rendering were visually reviewed, including diagrams, formulas and table notes. Word uses plain black 11-point text, eight editable tables, five figure panels and one formal equation image; it contains no Word equation objects.',
      '- [Canonical PDF](../../../docs/thesis/'+PDF+')\n- [Original plain DOCX](../../../docs/thesis/'+DOCX+')\n- [Figure 3.1 panels](methods_diagram/)\n- [Figures 4.1–4.3](approved_figures/figures/)\n- [Version comparison](version_comparison.json)\n- [Verification](verification/final_verification.json)\n- [Expanded freeze](final_freeze.json)',
      'Reproduction: `code/run_expanded_rq1.py` is the canonical expanded RQ1 entry point. `code/run_expanded_final.py rq2` and `rq3` bind the frozen algorithm to 2,062 candidates and the original 99-ZIP frame; `code/close_expanded_final.py replay` independently reproduces existing outputs. The unused historical `rq1` dispatch in the initial runner is superseded by `run_expanded_rq1.py`. Source copies and hashes are retained in `implementation/`. Network access is disabled in the analysis runner.',
      '**Stop:** the authorized expanded analysis and Chapters 1–4 synchronization are complete.']
    (OUT/'FINAL_REPORT.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    lineage=read_json(OUT/'candidate_lineage.json')
    run.save(OUT/'classification_manifest.json',{'candidate_count':2062,'candidate_lineage_sha256':run.sha(OUT/'candidate_lineage.json'),
        'original_cohort':1226,'supplemental_cohort':836,'source_freezes':read_json(OUT/'verification/upstream_preservation.json')['packages'],
        'all_candidates_revalidated_at_phase0':True,'original_classifications_recalled':0,'all_836_included_in_candidate_selection':True})
    deliver=OUT/'deliverables';deliver.mkdir(exist_ok=True)
    files=[THESIS/PDF,THESIS/DOCX,THESIS/'introduction_literature_methods_results_final_figures.tex',THESIS/'references.bib',THESIS/'thesis.cls']
    files+=list((THESIS/'chapters').glob('0[1234]*rewrite.tex'))+[THESIS/'chapters/03_methodology.tex',THESIS/'chapters/04_results.tex']
    files+=list((THESIS/'tables').glob('*.tex'))
    files+=list(THESIS.glob('*.cls'))+list(THESIS.glob('*.sty'))
    for name in ('figure01_name_recurrence','figure02_spatial_disagreement','figure03_signed_distance_histogram'):
        files += [THESIS/'figures/results'/f'{name}.{suffix}' for suffix in ('pdf','png','svg')]
    files += [THESIS/'figures/results'/name for name in ('Figure_3.1A_inventory_construction_corrected.pdf','Figure_3.1B_correspondence_proximity_matched_scale.pdf')]
    for p in set(files):
        dest=deliver/p.relative_to(THESIS);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['run_expanded_final.py','run_expanded_rq1.py','close_expanded_final.py','sync_expanded_thesis.py','update_expanded_workflow_diagram.py','build_expanded_plain_docx.py','qa_expanded_documents.py','freeze_expanded_final.py']:
        source=ROOT/'code'/name;dest=OUT/'implementation/code'/name
        if dest.exists():assert run.sha(source)==run.sha(dest)
        else:shutil.copy2(source,dest)
    for name in ['test_expanded_final.py','test_corrected_downstream.py']:
        dest=OUT/'implementation/tests'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'tests'/name,dest)
    external={p.relative_to(ROOT).as_posix():run.sha(p) for p in set(files)}
    frozen={p.relative_to(OUT).as_posix():run.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_freeze.json'}
    run.save(OUT/'final_freeze.json',{'status':'expanded_final_complete_stop','created_at':now(),'candidate_count':2062,
        'included':592,'excluded':1470,'technical_unresolved':0,'new_model_calls':0,'new_provider_calls':0,
        'canonical_deliverable_hashes':external,'files':frozen})
    run.verify_files(OUT,frozen);run.verify_files(ROOT,external)
    print('Final expanded freeze verified:',len(frozen),'files.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preserve','consistency','freeze'])
    globals()[parser.parse_args().stage]()
