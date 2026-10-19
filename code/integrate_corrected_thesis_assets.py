"""Bind manuscript tables and existing figure assets to the frozen corrected run."""
from pathlib import Path
import hashlib
import json
import shutil
import re
import pandas as pd
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
THESIS=ROOT/'docs/thesis'
RUN=ROOT/'analysis/closed_phase3/20260907T192546Z_v1'
OUT=ROOT/'analysis/maintenance/corrected_thesis_20260907_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
bindings=[]

def replace_rows(name,rows,notes=()):
    path=THESIS/'tables'/name
    text=(OUT/'before'/path.relative_to(ROOT)).read_text(encoding='utf-8')
    for prefix,cells,source in rows:
        lines=text.splitlines()
        matches=[i for i,line in enumerate(lines) if line.startswith(prefix+' &')]
        if len(matches)!=1:raise ValueError((name,prefix,matches))
        replacement=prefix+' & '+' & '.join(map(str,cells))+r' \\'
        lines[matches[0]]=replacement
        text='\n'.join(lines)+'\n'
        bindings.append({'table':name,'row':replacement,'source':source})
    for old,new in notes:
        if old not in text:raise ValueError((name,old))
        text=text.replace(old,new)
    path.write_text(text,encoding='utf-8',newline='\n')

def tables():
    d=read(RUN/'rq2/rq1_summary.json')
    rows=[]
    for label,state in [(' & Include','include'),(' & Exclude','exclude')]:
        r=d['eligibility'][state];rows.append((label,[f"{r['count']:,}",f"{r['pct']:.2f}"],'rq2/rq1_summary.json#/eligibility/'+state))
    # Leading ampersands are part of empty-panel rows, so use their exact prefixes.
    rows=[]
    for prefix,count in [(' & Include',539),(' & Exclude',687),('B. Target construct & Target pharmacy',542),
                         (' & Non-target',684),('C. Operating status & Active',1220),(' & Inactive',6),
                         ('D. Candidate plausibility & Plausible',887),(' & Implausible',339),
                         ('E. Identity consistency & Consistent',1173),(' & Inconsistent',53)]:
        rows.append((prefix,[f'{count:,}',f'{count/1226*100:.2f}'],'rq2/rq1_summary.json'))
    replace_rows('table1_candidate_flow.tex',rows)
    sens=pd.read_csv(RUN/'rq2/linkage_sensitivities.csv').to_dict('records')
    labels=['Name only, greedy','Name only, Hungarian','Address only, Hungarian','Name--address, Hungarian (primary)',
            'Name--address--geography, Hungarian','Maximum cardinality then weight','Name--address, threshold 70','Name--address, threshold 80']
    rows=[(label,[r['linked_pairs'],f"{r['unmatched_workflow']} / {r['unmatched_board']}",f"{r['workflow_linked_share_pct']:.2f}",f"{r['board_reference_coverage_pct']:.2f}"],f"rq2/linkage_sensitivities.csv#/{r['specification']}") for label,r in zip(labels,sens)]
    rows.append(('Address--ZIP location collapse (post hoc)',[148,'100 / 30','59.68','83.15'],'rq2/stage_specific_diagnostic.json#/address_zip_location_diagnostic'))
    replace_rows('table2_board_correspondence.tex',rows,[('339 Place-ID-defined','373 Place-ID-defined'),('239 exact','248 exact')])
    replace_rows('table3_stage_specific_filtering.tex',[
        ('Include',[219,154,373],'rq2/stage_specific/candidate_qualified_edge_cross_tab.csv'),
        ('Exclude',[11,600,611],'rq2/stage_specific/candidate_qualified_edge_cross_tab.csv'),
        ('Qualified raw edge, but all endpoints excluded',[3,'1.69',''],'rq2/stage_specific/place_id_board_stages.csv'),
        ('Final assigned link',[148,'83.15',''],'rq2/stage_specific/place_id_board_stages.csv')],
        [('The five Board records','The three Board records'),('equal 3.31\\%','equal 1.99\\%')])
    recurrence=read(RUN/'rq2/name_recurrence_summary.json')
    rows=[]
    for r in recurrence['rows']:
        if r['recurrence_threshold']!=2 or r['linkage_strategy']!=recurrence['primary_strategy']:continue
        source='Workflow' if r['source'].startswith('Workflow') else 'Board'
        if source=='Board' and r['group_code']=='unresolved':continue
        group={'repeated':'Repeated-name component','below_threshold':'Single-occurrence component','unresolved':'Recurrence unresolved'}[r['group_code']]
        ci='---' if r['cluster_bootstrap_95_low_pct'] is None else f"{r['cluster_bootstrap_95_low_pct']:.2f}--{r['cluster_bootstrap_95_high_pct']:.2f}"
        rows.append((source+' & '+group,[r['records'],r['linked_records'],f"{r['linked_pct']:.2f}",ci],'rq2/name_recurrence_summary.json'))
    replace_rows('table3_name_recurrence.tex',rows)
    rq3=read(RUN/'rq3/rq3_summary.json')
    rows=[]
    for t in ('0.5','1.0','2.0'):
        r=rq3['thresholds'][t]['scenarios']['primary_google_include_vs_board_screened'];s=r['states']
        rows.append((t,[s[k]['count'] for k in ['neither_beyond','both_beyond','board_only_beyond','google_only_beyond']]+[r['classification_changed']['count'],f"{r['classification_changed']['pct']:.2f}"],'rq3/rq3_summary.json#/thresholds/'+t))
    replace_rows('table4_paired_spatial.tex',rows)
    enrich=read(RUN/'rq3/spatial_enrichment.json');rows=[]
    for dim,groups in enrich['quartiles'].items():
        for q,r in groups.items():
            w,b,s=(r[k] for k in ('workflow_nearest_miles','board_nearest_miles','signed_workflow_minus_board_miles'))
            rows.append((('Income' if dim=='income' else 'Non-White')+' & Q'+q,[r['tract_count'],f"{w['mean']:.3f} ({w['median']:.3f})",f"{b['mean']:.3f} ({b['median']:.3f})",f"${s['mean']:.3f}$",f"{r['thresholds']['0.5']['changed_pct']:.2f}",f"{r['thresholds']['1.0']['changed_pct']:.2f}"],'rq3/spatial_enrichment.json#/quartiles/'+dim+'/'+q))
    replace_rows('table5_neighborhood.tex',rows, [('1.22, 0, 2.47, and 1.22','1.22, 0, 2.47, and 1.22')])
    (OUT/'table_claim_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n',encoding='utf-8')

def figures():
    manifest=read(RUN/'approved_figures/figure_manifest.json')
    copied={}
    for figure in manifest['figures']:
        for record in figure['formats'].values():
            source=ROOT/record['path'];assert sha(source)==record['sha256']
            dest=THESIS/'figures/results'/source.name
            shutil.copy2(source,dest);assert sha(dest)==record['sha256']
            copied[dest.relative_to(ROOT).as_posix()]=sha(dest)
    # Text-only updates in the existing vector Methods diagram, retaining its layout.
    source=OUT/'before/docs/thesis/figures/results/Figure_3.1A_inventory_construction_thesis.pdf'
    target=THESIS/'figures/results/Figure_3.1A_inventory_construction_corrected.pdf'
    document=fitz.open(source);page=document[0]
    replacements=[('• Website → CDX','• Seven-date official'),('records + selected','Availability lookup;'),
                  ('archived pages.','direct snapshot fetch.'),('• Requested for every','• Frozen Sept. 5'),
                  ('unique Place ID.','response per Place ID.')]
    targets=[]
    for old,new in replacements:
        rects=page.search_for(old)
        if len(rects)!=1:raise ValueError(('diagram text',old,rects))
        rect=rects[0];targets.append((rect,new))
        page.add_redact_annot(rect,fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    for rect,new in targets:
        page.insert_text((rect.x0,rect.y1-1.9),new,fontsize=9.04,fontname='ArialCorrected',
                         fontfile=r'C:\Windows\Fonts\arial.ttf',color=(36/255,55/255,70/255))
    document.save(target,garbage=4,deflate=True);document.close()
    doc=fitz.open(target);assert 'CDX' not in doc[0].get_text()
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/'methods_diagram_review.png');doc.close()
    method=THESIS/'chapters/03_methodology.tex'
    text=method.read_text(encoding='utf-8').replace('Figure_3.1A_inventory_construction_thesis.pdf','Figure_3.1A_inventory_construction_corrected.pdf')
    method.write_text(text,encoding='utf-8',newline='\n')
    (OUT/'figure_binding.json').write_text(json.dumps({'results_assets_exact_copies':copied,'methods_diagram_original_sha256':sha(source),
        'methods_diagram_corrected_sha256':sha(target),'text_replacements':replacements},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    tables();figures()
    print('Six tables bound to frozen outputs; nine Results assets copied exactly; existing Methods diagram wording corrected.')
