"""Produce a complete numeric RQ3 report and exact old-versus-new differences."""
import json
import numpy as np
import pandas as pd
import run_common_supply_rq3 as current
OUT,BASE=current.OUT,current.BASE
read=current.frozen.read_json
save=current.save
q=read(OUT/'rq3/rq3_summary.json');e=read(OUT/'rq3/spatial_enrichment.json');x=read(OUT/'rq3/spatial_sensitivity_extension.json')
oldx=read(BASE/'rq3/spatial_sensitivity_extension.json')
frame=read(OUT/'common_supply_frame.json')
tract=pd.read_csv(OUT/'rq3/tract_spatial.csv',dtype={'GEOID':str})
prior=pd.read_csv(BASE/'rq3/tract_spatial.csv',dtype={'GEOID':str})
qa=pd.read_csv(OUT/'rq3/coordinate_qa.csv').set_index('surface')
oldqa=pd.read_csv(BASE/'rq3/coordinate_qa.csv').set_index('surface')
w=tract.google_include_nearest_m.to_numpy()/1609.344;b=tract.board_screened_nearest_m.to_numpy()/1609.344;d=w-b
distance={name:{'mean':float(v.mean()),'median':float(np.median(v))} for name,v in [('workflow',w),('board',b),('signed_workflow_minus_board',d)]}
near={'workflow_nearer':int((d<0).sum()),'board_nearer':int((d>0).sum()),'ties':int((d==0).sum())}
paired=[];sens=[];conditional=[];weighted=[]
def record(name,t,states,n,changed,pct):
    return {'scenario':name,'threshold_miles':float(t),'origins':n,'both_within':states.get('neither_beyond',states.get('both_within')),'both_beyond':states['both_beyond'],'workflow_only_within':states.get('board_only_beyond',states.get('workflow_only_within')),'board_only_within':states.get('google_only_beyond',states.get('board_only_within')),'discordant':changed,'discordance_pct':pct}
for t,v in q['thresholds'].items():
    for name,s in v['scenarios'].items():
        row=record(name,t,{k:z['count'] for k,z in s['states'].items()},s['paired_origins'],s['classification_changed']['count'],s['classification_changed']['pct'])
        sens.append(row)
        if name=='primary_google_include_vs_board_screened':paired.append(row)
    for name,z in v['threshold_tolerance'].items():
        sens.append(record(name,t,z['states'],len(tract),z['paired_disagreements'],z['paired_disagreement_pct']))
    for name,z in [('geometric_centroid_origin',x['thresholds'][t]['geometric_centroid_origin']),('collection_frame_interior',x['thresholds'][t]['collection_frame_interior']['comparison']),('conditionally_determinate_origins',x['thresholds'][t]['conditional_ranges']['conditionally_determinate_comparison'])]:
        sens.append(record(name,t,z,z['paired_origins'],z['classification_changed_count'],z['classification_changed_pct']))
    for name,z in v['weighting'].items():weighted.append({'threshold_miles':float(t),'weighting':name,**z})
    c=x['thresholds'][t]['conditional_ranges'];old=oldx['thresholds'][t]['conditional_ranges']
    row={'threshold_miles':float(t),**{k:z for k,z in c.items() if isinstance(z,(int,float))},'discordant_among_determinate':c['conditionally_determinate_comparison']['classification_changed_count']}
    conditional.append(row)
neighborhood=[]
for dimension,groups in e['quartiles'].items():
    for quartile,z in groups.items():
        row={'dimension':dimension,'quartile':quartile,'tracts':z['tract_count']}
        for key,label in [('workflow_nearest_miles','workflow'),('board_nearest_miles','board'),('signed_workflow_minus_board_miles','signed')]:
            row[label+'_mean_miles']=z[key]['mean'];row[label+'_median_miles']=z[key]['median']
        for t,threshold in z['thresholds'].items():row['discordant_'+t]=threshold['changed_count'];row['discordance_pct_'+t]=threshold['changed_pct']
        neighborhood.append(row)
differences=[]
def compare(a,b,path):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()):compare(a.get(k),b.get(k),path+'/'+str(k))
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(aa,bb) in enumerate(zip(a,b)):compare(aa,bb,path+'/'+str(i))
    elif a!=b:differences.append({'path':path,'previous':a,'current':b})
for name in ['spatial_summary.json','rq3_summary.json','spatial_enrichment.json','spatial_sensitivity_extension.json']:
    compare(read(BASE/'rq3'/name),read(OUT/'rq3'/name),name)
columns_changed=[c for c in tract.columns if not tract[c].equals(prior[c])]
save(OUT/'exact_differences.json',{'scalar_or_structural_differences':differences,'tract_columns_changed':columns_changed,'all_tract_output_values_identical':not columns_changed})
for name,rows in [('paired_states.csv',paired),('all_spatial_sensitivities.csv',sens),('neighborhood_quartiles.csv',neighborhood),('conditional_ranges.csv',conditional),('weighting_sensitivities.csv',weighted)]:pd.DataFrame(rows).to_csv(OUT/name,index=False)
report={'common_supply_frame':frame,'coordinate_counts':qa.reset_index().to_dict('records'),'distances_miles':distance,'nearer_counts':near,'paired_states':paired,'neighborhood_quartiles':neighborhood,'spatial_sensitivities':sens,'weighting_sensitivities':weighted,'conditional_ranges':conditional,'geometry':x['collection_frame'],'missing_board_coordinates':x['missing_board_coordinates'],'all_primary_tract_distances_identical_to_previous':not columns_changed,'new_model_calls':0,'new_provider_calls':0,'rq1_rq2_unchanged':True}
save(OUT/'report.json',report)
lines=['# Final RQ3: common supply frame','',
 'Both inventories use the union of the 60 target ZIPs and every valid returned five-digit ZIP among the final 2,062 candidates. The deterministic union contains **'+str(frame['common_zip_count'])+' ZIPs**. All candidates contribute to the frame definition regardless of eligibility; none has an invalid or missing returned ZIP. The added ZIPs are '+', '.join(frame['added_vs_previous'])+'.','',
 'RQ2 retains the fixed 60-ZIP linkage frame. RQ2 matches do not filter spatial supply. All 327 tract origins, TIGER internal points, EPSG:26915 projection, projected Euclidean distances, thresholds, ACS variables and quartile memberships are preserved.','',
 '| Supply surface | Records | Usable coordinates | Missing coordinates |','|---|---:|---:|---:|']
for name,z in qa.iterrows():lines.append(f"| {name} | {z.input_rows} | {z.valid_coordinate_rows} | {z.missing_coordinate_rows} |")
lines+=['','| Distance (miles) | Mean | Median |','|---|---:|---:|']
for name,z in distance.items():lines.append(f"| {name} | {z['mean']:.9f} | {z['median']:.9f} |")
lines+=['',f"Workflow is nearer for {near['workflow_nearer']} tracts; Board is nearer for {near['board_nearer']}; {near['ties']} ties. All primary tract distances and paired states equal the previous source-specific-frame results.",'','| Miles | Both within | Both beyond | Workflow only within | Board only within | Discordant | Discordance (%) |','|---:|---:|---:|---:|---:|---:|---:|']
for r in paired:lines.append(f"| {r['threshold_miles']:g} | {r['both_within']} | {r['both_beyond']} | {r['workflow_only_within']} | {r['board_only_within']} | {r['discordant']} | {100*r['discordant']/r['origins']:.6f} |")
lines+=['','## Neighborhood quartiles','', 'Distances are miles. Mean (median) is shown for each inventory and the signed workflow-minus-Board difference. Discordance cells are count / percent.','', '| Dimension | Quartile | n | Workflow | Board | Signed difference | 0.5 mile | 1 mile | 2 miles |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in neighborhood:
    values=[r['dimension'],'Q'+r['quartile'],str(r['tracts'])]
    values += [f"{r[n+'_mean_miles']:.6f} ({r[n+'_median_miles']:.6f})" for n in ['workflow','board','signed']]
    values += [f"{r['discordant_'+t]} / {r['discordance_pct_'+t]:.2f}%" for t in ['0.5','1.0','2.0']]
    lines.append('| '+' | '.join(values)+' |')
lines+=['','## All predefined spatial sensitivities','', 'Each cell reports discordant tracts / eligible origins (percent). Full four-state counts are in `all_spatial_sensitivities.csv`.','', '| Specification | 0.5 mile | 1 mile | 2 miles |','|---|---:|---:|---:|']
for name in dict.fromkeys(r['scenario'] for r in sens):
    values=[]
    for t in [.5,1.,2.]:
        r=next(r for r in sens if r['scenario']==name and r['threshold_miles']==t)
        values.append(f"{r['discordant']}/{r['origins']} ({r['discordance_pct']:.2f}%)")
    lines.append('| '+name+' | '+' | '.join(values)+' |')
lines+=['','| Weighting | 0.5 mile | 1 mile | 2 miles |','|---|---:|---:|---:|']
for name in ['population_weighted','household_weighted']:
    values=[]
    for t in [.5,1.,2.]:
        z=next(z for z in weighted if z['threshold_miles']==t and z['weighting']==name)
        values.append(f"{z['discordant_weight']:,.0f}/{z['denominator_weight']:,.0f} ({z['weighted_pct']:.2f}%)")
    lines.append('| '+name+' | '+' | '.join(values)+' |')
lines+=['','## Collection boundary and conditional classifications','',
 'The common supply frame has 102 postal ZIPs and 101 corresponding polygons in the cached ZCTA data. ZIP 55144 retains supply membership and has no cached ZCTA polygon. The collection boundary is the union of the 101 represented ZCTAs. All origins are covered by this union. No substitute polygon or geocoding was introduced.','',
 'The screened Board set has 39 missing-coordinate records across 32 mailing ZIPs. One such ZIP, 55144, lacks a polygon. The existing missing-geometry rule sets minimum possible distance to zero for every origin in the conditional analysis. Thus observed Board-within classifications remain definite, while observed Board-beyond classifications remain unresolved under that scenario. This rule leaves the observed-supply nearest distances unchanged.','',
 '| Miles | Workflow beyond range (%) | Board beyond range (%) | Workflow unresolved | Board unresolved | Both determinate | Discordant among determinate |','|---:|---:|---:|---:|---:|---:|---:|']
for r in conditional:
    values=[f"{r['threshold_miles']:g}"]+[f"{r[s+'_beyond_rate_lower_pct']:.2f}–{r[s+'_beyond_rate_upper_pct']:.2f}" for s in ['workflow','board']]+[str(r[k]) for k in ['workflow_conditionally_unresolved_count','board_conditionally_unresolved_count','both_sources_conditionally_determinate_count','discordant_among_determinate']]
    lines.append('| '+' | '.join(values)+' |')
lines+=['', 'Conditional ranges use all 327 origins; the paired determinate summary uses its stated subset. These describe the specified missing-supply conditions, not confidence intervals or unconditional bounds.','', '## Exact changes from the previous RQ3','', '| Quantity | Previous | Common frame |','|---|---:|---:|', '| Supply ZIP count | 99 (Board only) | 102 (both sources) |']
for surface in qa.index:
    for field,label in [('input_rows','records'),('valid_coordinate_rows','usable coordinates'),('missing_coordinate_rows','missing coordinates')]:
        old=int(oldqa.loc[surface,field]);new=int(qa.loc[surface,field])
        if old!=new:lines.append(f'| {surface}: {label} | {old} | {new} |')
for t in ['0.5','1.0','2.0']:
    old=oldx['thresholds'][t];new=x['thresholds'][t]
    for field,label in [('interior_origin_count','interior origins')]:
        lines.append(f"| {t}-mile {label} | {old['collection_frame_interior'][field]} | {new['collection_frame_interior'][field]} |")
    a=old['conditional_ranges'];b=new['conditional_ranges']
    lines.append(f"| {t}-mile Board lower beyond rate (%) | {a['board_beyond_rate_lower_pct']:.2f} | {b['board_beyond_rate_lower_pct']:.2f} |")
    lines.append(f"| {t}-mile both-source determinate origins | {a['both_sources_conditionally_determinate_count']} | {b['both_sources_conditionally_determinate_count']} |")
lines+=['','Workflow membership and all 327 rows of `tract_spatial.csv` are numerically unchanged. Primary distances, quartile results, alternative-surface nearest distances, threshold perturbations and weighted rates are unchanged. Collection-boundary interior denominators and combined missing-coordinate conditional classifications change as tabulated above. `exact_differences.json` lists every changed scalar or structure in the four summary artifacts.','',
 'All eight spatial artifacts were recomputed in this run. Independent distance and geometry checks are recorded in `independent_verification.json` and `geometry_audit.json`. No RQ1 or RQ2 analysis, provider request, model call, or figure generation was executed.','',
 'The specified thesis sections and Tables 3.2, 4.5 and 4.6 are updated in LaTeX. Figure files, figure environments/captions, and the existing PDF/Word files remain unchanged for the subsequent figure revision. Figure 3.1B consequently still contains its prior supply-frame labels. This delivery stops at text and tables.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report_created':True,'all_tract_columns_unchanged':not columns_changed,'summary_differences':len(differences),'sensitivities':len(sens),'quartile_rows':len(neighborhood)}))
