"""Update authorized RQ3 manuscript text and tables; preserve all figure blocks."""
import difflib,json,re,shutil
import run_common_supply_rq3 as current
OUT,ROOT,THESIS=current.OUT,current.ROOT,current.THESIS
read=current.frozen.read_json
r=read(OUT/'report.json');geo=read(OUT/'geometry_audit.json')
assert read(OUT/'independent_verification.json')['passed']
n=r['common_supply_frame']['common_zip_count'];z=geo['represented_zcta_count']
paths=[THESIS/'chapters'/name for name in ['01_introduction_rewrite.tex','02_literature_rewrite.tex','03_methodology.tex','04_results.tex']]
texts={p:p.read_text(encoding='utf-8') for p in paths};original=dict(texts)
changes=[]
def replace(p,old,new):
    assert texts[p].count(old)==1,(p.name,old[:100])
    texts[p]=texts[p].replace(old,new,1)
    changes.append({'file':p.relative_to(ROOT).as_posix(),'before':old,'after':new})
p=paths[0]
replace(p,'''then use the same tract origins, projection, and distance thresholds. The comparison
reflects the inventories and their declared geographic supply coverage.''',f'''then enter the same {n}-ZIP supply frame, defined by the target ZIPs and valid
returned ZIPs in the final candidate set. With common tract origins, projection,
and distance thresholds, the spatial comparison measures differences associated
with inventory choice under a shared geographic specification.''')
p=paths[1]
replace(p,'''proximity calculation with common origins, projection, and distance thresholds.
The geographic supply coverage of each inventory also enters this comparison.''',f'''proximity calculation within the same final {n}-ZIP supply frame, with common
origins, projection, and distance thresholds. This shared spatial specification
connects inventory differences to tract-level potential proximity.''')
p=paths[2]
start=texts[p].index('Four frames organize the analysis.')
end=texts[p].index('\nThe 60 target ZIPs also define',start)
replace(p,texts[p][start:end],f'''Four frames organize the analysis. The query frame contains the 60 ZIP-coded search
definitions. The primary linkage frame restricts workflow records by returned ZIP
and Board records by mailing ZIP to those same 60 ZIPs. The spatial origin frame
consists of eligible Hennepin and Ramsey County census tracts whose Census-provided
internal point falls within a target ZCTA.

The common spatial supply frame is the union of the 60 target ZIPs and every valid
returned five-digit ZIP observed among the final 2,062 candidates. This rule yields
{n} ZIPs, including search spillover, and uses all candidates regardless of their
classification. Workflow supply consists of final included candidates with usable
coordinates whose returned ZIP is in this frame. Board supply consists of screened
active Minnesota records whose mailing ZIP is in the same frame, using usable
cached coordinates for the primary observed-supply analysis. RQ2 linkage matches
do not filter either inventory. Both sources are restricted to this common frame
before nearest-distance calculation; the frame is defined by postal membership
rather than a distance buffer or county boundary.

The collection boundary for frame-edge sensitivities is the union of the cached
ZCTA polygons corresponding to this common supply frame. The {n} postal ZIPs have
{z} corresponding polygons in the cached 2023 TIGER/Line ZCTA data: ZIP 55144 has
no represented polygon. It retains supply-frame membership, while the boundary uses
the {z} available ZCTAs. This geographic admission rule is independent of
classification and linkage outcomes. The primary linkage and tract-origin frames
remain restricted as described above.
''')
replace(p,'Figure~\\ref{fig:workflow_overview} summarizes how the four fixed analysis frames',
          'Figure~\\ref{fig:workflow_overview} summarizes how the four analysis frames')
replace(p,'''Both inventories use the same origins, projection, and thresholds of 0.5, 1, and 2
miles. An origin lies beyond threshold $r$ when $d_{is}>r$.''',f'''Both inventories are restricted to the same final {n}-ZIP supply frame before
distance calculation. They use the same origins, projection, and thresholds of
0.5, 1, and 2 miles. An origin lies beyond threshold $r$ when $d_{{is}}>r$.''')
replace(p,'''its projected distance to the boundary of the union of the 99 collection-frame ZCTAs
is strictly greater than that threshold. This is an origin restriction; it does not
remove observed supply. For the coordinate scenario, the minimum distance from each
origin to the union of ZCTAs associated with missing-coordinate Board records is
computed as point-to-polygon distance (zero inside a polygon). No random imputation
or farthest-point placement is used. Missing ZCTA geometry would set this minimum to
zero, leaving the origin potentially affected; all required ZCTA geometries were
available in this run.''',f'''its projected distance to the boundary of the common final-supply ZCTA union
is strictly greater than that threshold. This union contains the {z} represented
ZCTAs associated with the {n}-ZIP supply frame. The restriction selects origins
while retaining the same observed supply. For the coordinate scenario, minimum
distance from each origin to the union of ZCTAs associated with missing-coordinate
Board records is computed as point-to-polygon distance, with zero inside a polygon.
The method uses no random imputation or farthest-point placement. If a
missing-coordinate Board record has no corresponding ZCTA geometry, the minimum
possible distance is set to zero for every origin. This rule applies to mailing
ZIP 55144 in the final frame: observed Board-within classifications remain fixed,
while Board-beyond classifications remain potentially affected under the scenario.''')
replace(p,'''lies in its mailing-ZIP ZCTA. An observed outlet within the threshold fixes that''',
          '''lies in its mailing-ZIP ZCTA when that geometry is represented; the missing-geometry
rule applies otherwise. An observed outlet within the threshold fixes that''')
replace(p,'''The analytic delivery separates
derived classifications, identifiers, linkage and spatial outputs from private''',
'''The RQ3 reproducibility record also preserves the candidate-derived common supply
ZIP list, postal-to-ZCTA geometry coverage, source-specific coordinate ledgers, and
the independent distance and boundary checks. This common-frame spatial computation
uses the frozen classifications and cached inputs; the RQ1 and RQ2 outputs remain
unchanged. The analytic delivery separates derived classifications, identifiers,
linkage and spatial outputs from private''')
p=paths[3]
replace(p,'''The spatial comparison used the same 327 tract origins, all 592 included workflow
records with usable coordinates, and the screened Board supply from the fixed
99-ZIP frame. These source-specific supply definitions differ geographically, as
described in Methods. The observed inventories differed on 48 classifications''',f'''The spatial comparison restricted both inventories to the same final {n}-ZIP
supply frame and used the same 327 tract origins. Workflow supply comprised 592
included records with usable coordinates. The screened Board frame contained 346
records, of which 307 had usable cached coordinates and entered the primary
observed-supply calculation. The inventories differed on 48 classifications''')
replace(p,'''Using all active Board records produced 42, 32, and 3 discordant tracts at 0.5, 1, and 2
miles, respectively. Restricting supply to exact cached Board coordinates produced
70, 64, and 8 discordant tracts at 0.5, 1, and 2 miles, respectively.''',f'''Using all active Board records in the common {n}-ZIP frame admitted 383 located
records and produced 42, 32, and 3 discordant tracts at 0.5, 1, and 2 miles,
respectively. Restricting screened Board supply to 237 exact cached coordinates
produced 70, 64, and 8 discordant tracts at those thresholds.''')
replace(p,'''Geometric centroids produced 48, 44, and 5 discordant tracts. The collection-frame interior retained 318, 304, and 266 origins, with 47, 45, 3 discordant tracts (14.78\\%, 14.80\\%, 1.13\\%), respectively.''',
'''Geometric centroids produced 48, 44, and 5 discordant tracts. The common-supply
collection-frame interior retained 318, 305, and 269 origins at 0.5, 1, and 2 miles,
with 47, 45, and 3 discordant tracts (14.78\\%, 14.75\\%, and 1.12\\%), respectively.''')
start=texts[p].index('The fixed screened Board spatial frame contained')
end=texts[p].index('\n\n\\clearpage',start)
replace(p,texts[p][start:end],'''The screened Board supply frame contained 39 records without usable coordinates
across 32 mailing ZIPs. ZIP 55144 had no represented ZCTA geometry, activating the
zero-minimum-distance rule in Methods. Under the combined mailing-ZCTA and
frame-edge conditions, the workflow beyond-threshold rate ranged from
56.57--57.49\\%, 14.07--14.68\\%, and 0.00--0.31\\% at 0.5, 1, and 2 miles.
The corresponding Board ranges were 0.00--66.06\\%, 0.00--25.69\\%, and
0.00--1.53\\%. Board classifications remained conditionally unresolved for 216,
84, and 5 origins; workflow classifications were unresolved for 3, 2, and 1.
Both sources were conditionally determinate for 111, 243, and 322 origins, with
10, 5, and 0 discordant classifications (9.01\\%, 2.06\\%, and 0.00\\%).
These ranges describe the specified missing-supply conditions using all 327 origins;
the paired determinate summaries use their stated subsets. The ranges are not
confidence intervals or unconditional bounds.''')
# Assert the complete RQ1/RQ2 prose and every figure environment are byte-preserved.
assert texts[p].split('\\section{Paired Potential Proximity: RQ3}')[0]==original[p].split('\\section{Paired Potential Proximity: RQ3}')[0]
for p in paths:
    pattern=r'\\begin\{figure\}.*?\\end\{figure\}'
    assert re.findall(pattern,texts[p],re.S)==re.findall(pattern,original[p],re.S)
    p.write_text(texts[p],encoding='utf-8',newline='\n')
    (OUT/(p.name+'.diff')).write_text(''.join(difflib.unified_diff(original[p].splitlines(True),texts[p].splitlines(True))),encoding='utf-8')
# Rebuild only the two RQ3 Results tables from the new numeric report.
def row(values):return ' & '.join(str(v) for v in values)+r' \\'
def table(name,rows,note):
    p=THESIS/'tables'/name;old=p.read_text(encoding='utf-8')
    a=old.index('\\midrule')+len('\\midrule');b=old.index('\\bottomrule')
    text=old[:a]+'\n'+'\n'.join(rows)+'\n'+old[b:]
    text=re.sub(r'\\footnotesize Note\..*?(?=\\end\{minipage\})',lambda m:'\\footnotesize Note. '+note+'\n',text,flags=re.S)
    p.write_text(text,encoding='utf-8',newline='\n')
    (OUT/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True))),encoding='utf-8')
table('table4_paired_spatial.tex',[row([f"{v['threshold_miles']:.1f}",v['both_within'],v['both_beyond'],v['workflow_only_within'],v['board_only_within'],v['discordant'],f"{v['discordance_pct']:.2f}"]) for v in r['paired_states']],f'''Both inventories use the same final {n}-ZIP supply frame: 592 workflow records and
307 screened Board records with usable coordinates. Each row contains the same
327 tract origins. ``Workflow only within'' means the workflow distance is at or
below the threshold while the Board distance is beyond it; ``Board only within''
is the reverse.''')
rows=[]
for dim in ['income','nonwhite']:
    if dim=='nonwhite':rows.append(r'\addlinespace')
    for v in r['neighborhood_quartiles']:
        if v['dimension']!=dim:continue
        rows.append(row(['Income' if dim=='income' else 'Non-White','Q'+v['quartile'],v['tracts'],f"{v['workflow_mean_miles']:.3f} ({v['workflow_median_miles']:.3f})",f"{v['board_mean_miles']:.3f} ({v['board_median_miles']:.3f})",f"${v['signed_mean_miles']:.3f}$",f"{v['discordance_pct_0.5']:.2f}",f"{v['discordance_pct_1.0']:.2f}"]))
note=f'Both inventories use the same final {n}-ZIP supply frame. Signed mean is workflow minus Board nearest distance; negative values indicate nearer workflow supply. Q1 is the lowest quartile and Q4 the highest. At 2 miles, changed-tract rates were '
for dim in ['income','nonwhite']:
    values=[f"{v['discordance_pct_2.0']:.2f}" for v in r['neighborhood_quartiles'] if v['dimension']==dim]
    note+=', '.join(values)+(r'\% across income Q1--Q4 and ' if dim=='income' else r'\% across non-White Q1--Q4.')
table('table5_neighborhood.tex',rows,note)
p=THESIS/'tables/table_methods_spatial_sensitivities.tex';old=p.read_text(encoding='utf-8')
new=old.replace('Workflow eligibility & Final included candidates','Workflow eligibility & Final included candidates in the common supply frame')
new=new.replace('99-ZIP supply frame',f'common {n}-ZIP supply frame')
new=new.replace('the 99-ZCTA union boundary',f'the common final-supply ZCTA union boundary ({z} represented ZCTAs)')
new=new.replace('point-to-polygon distance to identify potentially changed classifications.','point-to-polygon distance to identify potentially changed classifications. A missing mailing-ZIP polygon sets the minimum possible distance to zero.')
new=new.replace('the applicable threshold. Frame-conditional',f'the applicable threshold. The {n}-ZIP supply frame has {z} cached ZCTA polygons; ZIP 55144 retains postal membership without a polygon. Frame-conditional')
p.write_text(new,encoding='utf-8',newline='\n')
(OUT/(p.name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),encoding='utf-8')
current.save(OUT/'text_changes.json',changes)
print('Updated authorized RQ3 text and Tables 3.2, 4.5 and 4.6; RQ1/RQ2 prose and all figure blocks preserved.')
