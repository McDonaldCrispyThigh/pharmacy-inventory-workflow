"""Generate six existing Results tables and descriptive Results from frozen outputs."""
from pathlib import Path
import re,shutil,json
import pandas as pd
import run_expanded_final as run
from wayback_snapshot_fetcher import read_json
ROOT,OUT=run.ROOT,run.OUT
THESIS=ROOT/'docs/thesis'
r=read_json(OUT/'analysis_report.json');q=r['rq1']['combined'];e=r['enrichment'];x=r['extension']
s=r['stage_specific'];p=r['rq2_primary'];ts=r['threshold_states'];a=r['all_active_board']
rec=[v for v in r['name_recurrence']['rows'] if v['linkage_strategy']==r['name_recurrence']['primary_strategy']]
def pct(v,n=2):return f'{v:.{n}f}'+r'\%'
def count(v):return f'{v:,}'
def row(cells):return ' & '.join(map(str,cells))+r' \\'
def recrow(source,group,cutoff=2):return next(v for v in rec if v['source'].startswith(source) and v['group_code']==group and v['recurrence_threshold']==cutoff)
def fmtrec(v):return f"{v['linked_records']} of {v['records']} ({pct(v['linked_pct'])}; 95\\% interval {v['cluster_bootstrap_95_low_pct']:.2f}--{v['cluster_bootstrap_95_high_pct']:.2f}\\%)"
def table(name,rows,note=None):
    path=THESIS/'tables'/name;text=path.read_text(encoding='utf-8')
    first=text.index('\\midrule')+len('\\midrule');last=text.index('\\bottomrule')
    text=text[:first]+'\n'+'\n'.join(rows)+'\n'+text[last:]
    if note is not None:
        text=re.sub(r'\\footnotesize Note\..*?(?=\\end\{minipage\})',lambda m:'\\footnotesize Note. '+note+'\n',text,flags=re.S)
    path.write_text(text,encoding='utf-8',newline='\n')

def tables():
    rows=[row(['A. Eligibility','All candidates','2,062','100.00'])]
    for label,key in [('Include','include'),('Exclude','exclude'),('Technical unresolved','unresolved')]:
        v=q['eligibility'][key];rows.append(row(['',label,count(v['count']),f"{v['pct']:.2f}"]))
    for panel,field,values in [('B. Target construct','target_construct_status',[('Target pharmacy','target_pharmacy'),('Non-target','non_target')]),
        ('C. Operating status','business_status',[('Active','active'),('Inactive','inactive')]),
        ('D. Candidate plausibility','candidate_plausibility',[('Plausible','plausible'),('Implausible','implausible')]),
        ('E. Identity consistency','identity_consistency',[('Consistent','consistent'),('Inconsistent','inconsistent')])]:
        rows.append(r'\addlinespace')
        for i,(label,key) in enumerate(values):
            v=q['core_judgment_distributions'][field][key];rows.append(row([panel if i==0 else '',label,count(v['count']),f"{v['pct']:.2f}"]))
    table('table1_candidate_flow.tex',rows,'The denominator for every percentage is 2,062. The four substantive judgments are reported separately and are not mutually exclusive across panels. Eligibility follows the predefined deterministic reducer.')
    labels=['Name only, greedy','Name only, Hungarian','Address only, Hungarian','Name--address, Hungarian (primary)','Name--address--geography, Hungarian','Maximum cardinality then weight','Name--address, threshold 70','Name--address, threshold 80']
    rows=[row([label,v['linked_pairs'],f"{v['unmatched_workflow']} / {v['unmatched_board']}",f"{v['workflow_linked_share_pct']:.2f}",f"{v['board_reference_coverage_pct']:.2f}"]) for label,v in zip(labels,r['linkage_sensitivities'])]
    v=s['address_zip_location_diagnostic']['assignments'][1]
    rows.extend([r'\addlinespace',row(['Address--ZIP location collapse (post hoc)',v['linked_pairs'],f"{v['unmatched_candidate_rows']} / {v['unmatched_board_rows']}",f"{v['candidate_linked_share_pct']:.2f}",f"{v['board_reference_coverage_pct']:.2f}"])])
    table('table2_board_correspondence.tex',rows,'W/B denotes unmatched workflow/Board rows. The predefined rows use 413 Place-ID-defined workflow rows and 178 screened Board records in the primary 60-ZIP frame. The post hoc location-unit row uses 272 exact normalized-address--ZIP groups and the same 178 Board records, score, 75-point threshold, and Hungarian assignment. Linked share and Board-reference coverage describe correspondence under each rule.')
    cross=pd.read_csv(OUT/'rq2/stage_specific/candidate_qualified_edge_cross_tab.csv')['count'].tolist()
    inc,exc,noinc,noexc=cross
    rows=[row(['Include',inc,noinc,inc+noinc]),row(['Exclude',exc,noexc,exc+noexc]),row(['All in-frame candidates',inc+exc,noinc+noexc,sum(cross)]),r'\addlinespace',row([r'\textit{Panel B. Board-record stage decomposition}','','','']),row(['Board-stage outcome','Count','Percent of 178','']),r'\midrule']
    labels=['No qualified raw-candidate edge','Qualified raw edge, but all endpoints excluded','Qualified included edge, but no final assignment','Final assigned link']
    for label,v in zip(labels,s['place_id_diagnostic']['board_stages']):rows.append(row([label,v['count'],f"{v['pct_of_board_records']:.2f}",'']))
    rows.append(row(['Total',178,'100.00','']))
    table('table3_stage_specific_filtering.tex',rows,r'A qualified edge has a primary name--address score of at least 75. Board-level percentages use 178 as the denominator. The four Board records associated with attrition at the LLM/reducer eligibility stage equal 2.60\% of the 154 Board records covered by the raw-candidate assignment. Absence of a qualified edge does not establish non-pharmacy status. The Board supplies the screened operational reference and is not treated as a complete census of real-world pharmacy outlets.')
    rows=[]
    for source in ('Workflow','Board'):
        if source=='Board':rows.append(r'\addlinespace')
        for group,label in [('repeated','Repeated-name component'),('below_threshold','Single-occurrence component'),('unresolved','Recurrence unresolved')]:
            v=recrow(source,group)
            if not v['records']:continue
            interval='---' if v['cluster_bootstrap_95_low_pct'] is None else f"{v['cluster_bootstrap_95_low_pct']:.2f}--{v['cluster_bootstrap_95_high_pct']:.2f}"
            rows.append(row([source,label,v['records'],v['linked_records'],f"{v['linked_pct']:.2f}",interval]))
    table('table3_name_recurrence.tex',rows)
    table('table4_paired_spatial.tex',[row([v['miles'],v['neither_beyond'],v['both_beyond'],v['board_only_beyond'],v['google_only_beyond'],v['changed'],f"{v['changed_pct']:.2f}"]) for v in ts])
    rows=[]
    for dimension,label in [('income','Income'),('nonwhite','Non-White')]:
        if dimension=='nonwhite':rows.append(r'\addlinespace')
        for quart,v in e['quartiles'][dimension].items():
            w,b=v['workflow_nearest_miles'],v['board_nearest_miles']
            rows.append(row([label,'Q'+quart,v['tract_count'],f"{w['mean']:.3f} ({w['median']:.3f})",f"{b['mean']:.3f} ({b['median']:.3f})",f"${v['signed_workflow_minus_board_miles']['mean']:.3f}$",f"{v['thresholds']['0.5']['changed_pct']:.2f}",f"{v['thresholds']['1.0']['changed_pct']:.2f}"]))
    note='Signed mean is workflow minus Board nearest distance; negative values indicate a nearer workflow location. Q1 is the lowest quartile and Q4 the highest. At 2 miles, changed-tract rates were '
    note+=', '.join(f"{v['thresholds']['2.0']['changed_pct']:.2f}" for v in e['quartiles']['income'].values())+r'\% across income Q1--Q4 and '
    note+=', '.join(f"{v['thresholds']['2.0']['changed_pct']:.2f}" for v in e['quartiles']['nonwhite'].values())+r'\% across non-White Q1--Q4.'
    table('table5_neighborhood.tex',rows,note)

def results():
    inc,exc,noinc,noexc=pd.read_csv(OUT/'rq2/stage_specific/candidate_qualified_edge_cross_tab.csv')['count'].tolist()
    groups=pd.read_csv(OUT/'rq2/stage_specific/included_location_groups.csv')
    links=pd.read_csv(OUT/'rq2/board_links.csv');linked=set(links.loc[links.stratum.eq('include_screened_primary'),'place_id'])
    unmatched=pd.read_csv(OUT/'rq2/unmatched_google.csv');unmatched=set(unmatched.loc[unmatched.stratum.eq('include_screened_primary'),'place_id'])
    group_sets=[set(z.split('|')) for z in groups.constituent_place_ids]
    shared=sum(len(z&unmatched) for z in group_sets if len(z)>1)
    withlinked=sum(len(z&unmatched) for z in group_sets if z&linked)
    sizes=groups.place_id_count.value_counts().sort_index().to_dict()
    w,b,d=e['workflow_nearest_miles'],e['board_nearest_miles'],e['signed_workflow_minus_board_miles']
    near=r['nearer_counts'];income=e['quartiles']['income'];nonwhite=e['quartiles']['nonwhite']
    text=rf'''% Generated from analysis/expanded_final/20260908_v1; descriptive final expanded analysis.
\chapter{{Results}}
\label{{sec:results}}

The corrected expanded frame contains 2,062 unique candidate Place IDs: 1,226 from
the original retrieval and 836 additional candidates from later saturation-resolution
retrieval. The sections below report the frozen classifications, correspondence with
the screened Board reference, and paired tract-level potential proximity.

\section{{Workflow Output: RQ1}}
\label{{sec:results_rq1}}

The deterministic reducer included 592 of 2,062 candidates (28.71\%) and excluded
1,470 (71.29\%); none remained technical-unresolved after bounded closure
(Table~\ref{{tab:candidate_flow}}). The original cohort contributed 539 included and
687 excluded records; the supplemental cohort contributed 53 included and 783 excluded
records. All 836 supplemental candidates were classified, including candidates without
provider-side pharmacy signals. Across the combined frame, 595 records (28.86\%) were
designated target pharmacies, 2,049 (99.37\%) active, 1,302 (63.14\%) plausible, and
1,986 (96.31\%) identity-consistent.

Of the 595 target-designated records, 592 satisfied all four inclusion conditions.
The other three were excluded because their identity was inconsistent, despite
affirmative target, active, and plausible judgments. No non-target record was included.

\input{{tables/table1_candidate_flow}}

Among the 1,470 exclusions, 688 (46.80\%) were non-target only; 699 (47.55\%) combined
non-target and implausible judgments; 50 (3.40\%) additionally had an inconsistent
identity; and 17 (1.16\%) combined non-target and identity inconsistency. Two (0.14\%)
combined non-target and inactive judgments, five (0.34\%) combined non-target, inactive,
and implausible judgments, and six (0.41\%) had all four negative judgments. The
remaining three (0.20\%) were excluded for identity inconsistency alone. In total,
691 exclusions had one negative judgment and 779 had multiple negative judgments.

High confidence occurred in 1,792 target decisions (86.91\%), 1,898 operating-status
decisions (92.05\%), 1,905 plausibility decisions (92.39\%), and 1,838 identity decisions
(89.14\%). Moderate confidence occurred in 270, 164, 157, and 219 decisions,
respectively. Five identity decisions had low confidence (0.24\%); none of the other
three fields had low-confidence decisions.

The preceding eight-candidate development pilot produced 24 structurally valid
outputs across evidence conditions A, B, and C, with no change in a core judgment or
eligibility. It tested execution and evidence order and did not estimate population
performance or validate the corrected archive packets. The preceding one-candidate
smoke test also passed. These checks were not repeated on the corrected packets.
Both final cohorts requested and returned \texttt{{gpt-5-mini-2025-08-07}}.

\clearpage
\section{{Board Correspondence: RQ2}}
\label{{sec:results_rq2}}

The active Minnesota Board source contained 1,110 records, of which 245 had mailing
ZIPs in the primary 60-ZIP frame. The automatic screen excluded 64 institutional-or-hospital
records and three noncommunity-service records; none was excluded under the
compounding-only rule. The screened primary reference therefore contained 178 records.

The primary frame contained 413 of the 592 included workflow rows; the other 179
were outside it. The primary Hungarian assignment linked 150 workflow rows to 150
Board records, giving a workflow linked share of 36.32\% and Board-reference coverage
of 84.27\% (Table~\ref{{tab:board_correspondence}}). It left 263 workflow rows and 28
Board records unmatched. The 150 scores had a mean of 98.30, median of 100, and range
of 75.31--100.

The unfiltered primary-frame pool contained 1,748 candidates. Under the same linkage
rule, its assignment covered 154 of the 178 screened Board records (86.52\%). The
eligibility stage removed 1,335 candidate rows ({pct(1335/1748*100)}) and retained 150
of those 154 Board identities (97.40\%). Four previously covered Board records were
no longer covered (2.60\% of the raw-assignment denominator).

At least one qualified Board edge existed for {inc+exc} raw candidates: {inc} included
and {exc} excluded. The remaining {noinc+noexc:,} candidates had no qualified edge:
{noinc} included and {noexc:,} excluded (Table~\ref{{tab:stage_specific_filtering}}).
The included shares were {pct(inc/(inc+exc)*100)} among candidates with a qualified
edge and {pct(noinc/(noinc+noexc)*100)} among those without one. Nine of the 13 excluded
candidates with an edge reached a Board record covered by another included candidate.
At the Board-record level, 24 had no qualified raw-candidate edge, four had only
excluded endpoints, zero had an included edge without an assignment, and 150 were
assigned. These are descriptive correspondence measures under the declared rule,
not classification-performance estimates.

Exact normalized address--ZIP collapse preserved the raw and included assignment
counts of 154 and 150, and the 97.40\% Board-referenced retention. The 413 included
Place-ID rows formed 272 location units. Their Place-ID multiplicities were
{'; '.join(f'{n} units with {k} Place ID'+('s' if k>1 else '') for k,n in sizes.items())}.
The location-unit assignment left 122 workflow locations and 28 Board records
unmatched, with 55.15\% workflow linked share and 84.27\% Board coverage. Of the 263
unmatched Place-ID rows, {withlinked} collapsed into a location containing a linked
Place ID. The Place-ID analysis remains primary.

\input{{tables/table2_board_correspondence}}
\input{{tables/table3_stage_specific_filtering}}

At threshold 75, the predefined component specifications produced 150--162 links.
Name-only greedy and Hungarian assignments each produced 154, address-only produced
162, and name--address--geography produced 150. Maximum-cardinality-then-weight
reproduced 150. Threshold 70 produced 157 links and threshold 80 produced 149.
Against all 245 active target-ZIP Board records, the independent assignment linked
179 pairs, giving 43.34\% workflow linked share and 73.06\% Board coverage.

Among the 263 unmatched workflow rows, {shared} ({pct(shared/263*100)}) shared an exact
normalized address and ZIP with another included row, and {withlinked}
({pct(withlinked/263*100)}) shared that location with a linked row. These counts overlap.
Of the {inc} included candidates with a qualified edge, 150 linked and {inc-150} did
not; the latter accounted for {pct((inc-150)/263*100)} of unmatched rows. The other
{noinc} unmatched rows ({pct(noinc/263*100)}) had no qualified edge. Neither category
independently determines whether a record represents a pharmacy outlet.

Correspondence was higher for repeated-name components than for single-occurrence
components (Table~\ref{{tab:name_recurrence}}; Figure~\ref{{fig:name_recurrence_final}}).
Workflow correspondence was {fmtrec(recrow('Workflow','repeated'))} for repeated-name
rows and {fmtrec(recrow('Workflow','below_threshold'))} for single-occurrence rows.
Board coverage was {fmtrec(recrow('Board','repeated'))} and
{fmtrec(recrow('Board','below_threshold'))}, respectively. One workflow row had an
unresolved normalized-name component; its classification was valid.

\input{{tables/table3_name_recurrence}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=6.20in]{{figures/results/figure01_name_recurrence.pdf}}
  \caption[Primary correspondence by normalized-name recurrence]{{Primary correspondence
  by normalized-name recurrence. Panel (a) reports workflow linked share and panel (b)
  Board-reference coverage. Points are percentages; intervals are 95\% component-level
  percentile bootstrap intervals from 5,000 resamples. The display omits one
  recurrence-unresolved workflow row. Recurrence does not establish ownership.}}
  \label{{fig:name_recurrence_final}}
\end{{figure}}

At cutoff three, repeated and below-threshold workflow components had correspondence
of {pct(recrow('Workflow','repeated',3)['linked_pct'])} and {pct(recrow('Workflow','below_threshold',3)['linked_pct'])};
Board values were {pct(recrow('Board','repeated',3)['linked_pct'])} and {pct(recrow('Board','below_threshold',3)['linked_pct'])}.
At cutoff five, the workflow values were {pct(recrow('Workflow','repeated',5)['linked_pct'])}
and {pct(recrow('Workflow','below_threshold',5)['linked_pct'])}, with Board values of
{pct(recrow('Board','repeated',5)['linked_pct'])} and {pct(recrow('Board','below_threshold',5)['linked_pct'])}.

The fixed 101-ZIP diagnostic frame contained 584 workflow rows and 375 screened
Board records, with 244 links (41.78\% workflow linked share; 65.07\% Board coverage).
The fixed 106-ZIP historical-v1-union frame contained 587 and 376 records, with 244
links (41.57\% and 64.89\%). These broader frames retain their historical definitions
and do not represent alternative estimates using the primary denominator.

\clearpage
\section{{Paired Potential Proximity: RQ3}}
\label{{sec:results_rq3}}

The spatial comparison used the same 327 tract origins, all 592 included workflow
records with usable coordinates, and the screened Board supply from the fixed
99-ZIP frame. Inventory choice changed {ts[0]['changed']} classifications ({pct(ts[0]['changed_pct'])})
at 0.5 miles, {ts[1]['changed']} ({pct(ts[1]['changed_pct'])}) at 1 mile, and four (1.22\%)
at 2 miles (Table~\ref{{tab:paired_spatial}}). At 0.5 miles, 38 tracts were workflow
only within and ten were Board only within; at 1 mile, these counts were 41 and five.
All four 2-mile changes were workflow only within.

\input{{tables/table4_paired_spatial}}
\clearpage
\begin{{landscape}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=8.90in]{{figures/results/figure02_spatial_disagreement.pdf}}
  \caption[Paired spatial disagreement at 0.5- and 1-mile thresholds]{{Paired spatial
  disagreement among 327 tract origins: 48 classifications differ at 0.5 mile and 46
  at 1 mile. Workflow only within means within the threshold only under the workflow
  inventory; Board only within is the reverse. Both panels use the same buffered
  rectangular extent, equal aspect, and EPSG:26915 (NAD83 / UTM Zone 15N; meters).
  Tract, county, and area-water context comes from 2023 U.S. Census TIGER/Line files.}}
  \label{{fig:spatial_disagreement_final}}
\end{{figure}}
\end{{landscape}}
\clearpage

Mean nearest distances were {w['mean']:.3f} miles for workflow and {b['mean']:.3f} miles
for Board, with medians of {w['median']:.3f} and {b['median']:.3f} miles. The signed
workflow-minus-Board difference had mean ${d['mean']:.3f}$ and median ${d['median']:.3f}$
miles. Its middle 50\% lay between ${d['q1']:.3f}$ and {d['q3']:.3f} miles, with an
observed range of ${d['min']:.3f}$ to {d['max']:.3f} miles. Workflow supply was nearer
for {near['workflow_nearer']} tracts ({pct(near['workflow_nearer']/327*100)}) and Board
supply for {near['board_nearer']} ({pct(near['board_nearer']/327*100)}); there were no ties
(Figure~\ref{{fig:signed_distance_final}}).

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=6.20in]{{figures/results/figure03_signed_distance_histogram.pdf}}
  \caption[Workflow-minus-Board nearest-distance differences]{{Workflow-minus-Board
  nearest-distance differences across 327 origins. Negative values indicate nearer
  workflow supply. Fixed 0.1-mile bins cover the observed range of ${d['min']:.3f}$ to
  {d['max']:.3f} miles. The median is ${d['median']:.3f}$ miles; the first and third
  quartiles are ${d['q1']:.3f}$ and {d['q3']:.3f} miles.}}
  \label{{fig:signed_distance_final}}
\end{{figure}}

Mean nearest distance increased across income Q1--Q4 from {income['1']['workflow_nearest_miles']['mean']:.3f}
to {income['4']['workflow_nearest_miles']['mean']:.3f} miles for workflow and from
{income['1']['board_nearest_miles']['mean']:.3f} to {income['4']['board_nearest_miles']['mean']:.3f}
miles for Board (Table~\ref{{tab:neighborhood_patterns}}). Change rates were not monotonic:
the 0.5-mile rate was highest in Q2 (19.51\%) and the 1-mile rate in Q4 (20.73\%).
All four 2-mile changes occurred in income Q4.

Across percentage-non-White quartiles, mean distances and disagreement rates were
also nonmonotonic. Q3 had the largest signed mean difference in magnitude
($-0.185$ miles) and highest 1-mile change rate (20.99\%); Q2 had the highest
0.5-mile rate (18.29\%). The 2-mile rate was at most 2.47\%. These groupings describe
tracts and do not support individual-level or causal interpretations.

\input{{tables/table5_neighborhood}}
'''
    for key,label in [('google_include_vs_board_all_active','Using all active Board records'),('google_include_vs_board_screened_exact','Using only exact cached Board coordinates')]:
        values=[e['prespecified_sensitivities'][t]['scenarios'][key]['classification_changed']['count'] for t in ('0.5','1.0','2.0')]
        text+=f"\n{label} produced {values[0]}, {values[1]}, and {values[2]} discordant tracts at 0.5, 1, and 2 miles, respectively.\n"
    cent=[x['thresholds'][t]['geometric_centroid_origin']['classification_changed_count'] for t in ('0.5','1.0','2.0')]
    interior=[x['thresholds'][t]['collection_frame_interior']['comparison'] for t in ('0.5','1.0','2.0')]
    text+=f"\nGeometric centroids produced {cent[0]}, {cent[1]}, and {cent[2]} discordant tracts. The collection-frame interior retained 318, 304, and 266 origins, with "+', '.join(str(v['classification_changed_count']) for v in interior)+' discordant tracts ('+', '.join(pct(v['classification_changed_pct']) for v in interior)+'), respectively.\n'
    for t in ('0.5','1.0','2.0'):
        tol=r['rq3']['thresholds'][t]['threshold_tolerance']
        text+=f"At {t} miles, moving the threshold four meters below or above its declared value produced {tol['minus_4m_threshold']['paired_disagreements']} and {tol['plus_4m_threshold']['paired_disagreements']} discordant tracts, respectively.\n"
    text+='The include-plus-unresolved alternative coincided with the primary inventory because no candidate remained technically unresolved.\n'
    for key,label in [('population_weighted','Population-weighted'),('household_weighted','Household-weighted')]:
        values=[r['rq3']['thresholds'][t]['weighting'][key]['weighted_pct'] for t in ('0.5','1.0','2.0')]
        text+='\n'+label+' disagreement was '+', '.join(pct(v) for v in values)+' at 0.5, 1, and 2 miles, respectively.\n'
    text+='\nThe fixed screened Board spatial frame contained 341 records, including 303 with usable coordinates and 38 without them across 31 mailing ZIPs. Under the combined mailing-ZCTA and frame-edge conditions in Methods, the workflow beyond-threshold rate ranged from '
    for source in ('workflow','board'):
        if source=='board':text+='The corresponding Board ranges were '
        text+=', '.join(f"{x['thresholds'][t]['conditional_ranges'][source+'_beyond_rate_lower_pct']:.2f}--{x['thresholds'][t]['conditional_ranges'][source+'_beyond_rate_upper_pct']:.2f}\\%" for t in ('0.5','1.0','2.0'))+' at 0.5, 1, and 2 miles. '
    text+='Both sources were conditionally determinate for 225, 267, and 322 origins; '+', '.join(str(x['thresholds'][t]['conditional_ranges']['conditionally_determinate_comparison']['classification_changed_count']) for t in ('0.5','1.0','2.0'))+' of these were discordant. These scenario-dependent denominators and ranges do not replace the primary observed-supply comparison and are not confidence intervals or unconditional bounds.\n\n\\clearpage\n'
    (THESIS/'chapters/04_results.tex').write_text(text,encoding='utf-8',newline='\n')
    run.save(OUT/'manuscript_derived_audit.json',{'location_multiplicities':{str(k):int(v) for k,v in sizes.items()},'unmatched_shared_location':shared,'unmatched_in_linked_location':withlinked,'qualified_included_unassigned':inc-150,'results_source_sha256':run.sha(OUT/'analysis_report.json')})

if __name__=='__main__':
    assert read_json(OUT/'verification/analysis_verification.json')['passed']
    tables();results()
    for name in ('figure01_name_recurrence','figure02_spatial_disagreement','figure03_signed_distance_histogram'):
        for suffix in ('pdf','png','svg'):shutil.copy2(OUT/'approved_figures/figures'/f'{name}.{suffix}',THESIS/'figures/results'/f'{name}.{suffix}')
    print('Six Results tables, Results chapter, and three approved figures synchronized.')
