"""Read-only scientific/manuscript checks and a versioned integration receipt."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from run_closed_phase3 import protected

THESIS=ROOT/'docs/thesis'
PHASE=ROOT/'analysis/closed_phase3/20260907T192546Z_v1'
OUT=ROOT/'analysis/maintenance/corrected_thesis_20260907_v1'
STEM='introduction_literature_methods_results_final_figures'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks={}

def require(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)

def flat(text):
    return re.sub(r'\s+',' ',text).strip()

def main():
    seal=read(PHASE/'final_freeze.json')
    require('phase3_93_frozen_files_unchanged',all(sha(PHASE/p)==h for p,h in seal['files'].items()))
    require('accepted_classification_and_archive_inventories_unchanged',protected()==read(PHASE/'protected_before.json'))
    before=read(OUT/'thesis_before.json')
    sources=['chapters/01_introduction_rewrite.tex','chapters/03_methodology.tex','chapters/04_results.tex','references.bib']
    sources+=['tables/'+p.name for p in (THESIS/'tables').glob('*.tex')]
    fig=read(OUT/'figure_binding.json')
    allowed={('docs/thesis/'+s) for s in sources}|set(fig['results_assets_exact_copies'])
    allowed|={f'docs/thesis/{STEM}.{ext}' for ext in ['pdf','log','aux','out','toc','lot','lof','bbl','blg']}
    changed=[p for p,v in before.items() if sha(ROOT/p)!=v['sha256']]
    require('only_authorized_existing_thesis_files_changed',set(changed)<=allowed)
    preserved=0
    historical=read(ROOT/'data/clean_archive/20260906T040605Z/preservation_before.json')
    for relative,expected in historical.items():
        if relative in allowed:continue
        p=ROOT/relative;s=p.stat()
        actual={'sha256':sha(p),'mtime_ns':s.st_mtime_ns,'birthtime_ns':getattr(s,'st_birthtime_ns',None)}
        require('historical_file:'+relative,actual==expected)
        preserved+=1
    for path,h in fig['results_assets_exact_copies'].items():require('approved_figure:'+path,sha(ROOT/path)==h)
    require('corrected_methods_diagram_bound',sha(THESIS/'figures/results/Figure_3.1A_inventory_construction_corrected.pdf')==fig['methods_diagram_corrected_sha256'])
    table_bindings=read(OUT/'table_claim_bindings.json')
    for row in table_bindings:
        require('table_row:'+row['table']+':'+row['row'],row['row'] in (THESIS/'tables'/row['table']).read_text(encoding='utf-8'))
    method=(THESIS/'chapters/03_methodology.tex').read_text(encoding='utf-8')
    results=(THESIS/'chapters/04_results.tex').read_text(encoding='utf-8')
    require('methods_actual_archive_design',all(x in method for x in ['1,088','seven-date','1996, 2001, 2006, 2011, 2016, and 2021','Availability API','standalone direct snapshot crawler','Deterministic local parsing','not an exhaustive search']))
    require('methods_inputs_preserved',all(x in method for x in ['September 5, 2026 Place Details','reconstructable classifier-visible evidence packets','Preserved model request and response','store=false']))
    require('methods_no_obsolete_cdx_or_nonretention','CDX' not in method and 'were not retained' not in method)
    require('fixed_model_and_completed_closure','gpt-5-mini-2025-08-07' in method and 'none remained technical-unresolved' in method)
    require('rq1_target_not_eligibility_partition','542' in results and 'other three were excluded because their identity was inconsistent' in flat(results))
    require('recurrence_unresolved_distinguished','its substantive classification was valid' in results)
    require('old_pilot_explicitly_historical','not repeated on the corrected evidence packets' in flat(results))
    for obsolete in ['498 of 1,226','728 (59.38','43.07','82.02','96.69','65.55','54.50','21.55','0.677','1.044','193 unmatched','perfectly partitioned']:
        require('obsolete_result_absent:'+obsolete,obsolete not in results)
    log=(THESIS/(STEM+'.log')).read_text(encoding='utf-8')
    require('no_overfull_underfull_or_unresolved_references',not re.search(r'Overfull|Underfull|undefined|multiply defined',log,re.I))
    wrapper=(THESIS/(STEM+'.tex')).read_text(encoding='utf-8')
    included=re.findall(r'\\input\{([^}]+)\}',wrapper)
    require('only_chapters_1_to_4_compiled',included==['chapters/01_introduction_rewrite','chapters/02_literature_rewrite','chapters/03_methodology','chapters/04_results'])
    lot=(THESIS/(STEM+'.lot')).read_text(encoding='utf-8')
    lof=(THESIS/(STEM+'.lof')).read_text(encoding='utf-8')
    require('table_numbers_4_1_to_4_6',re.findall(r'\\numberline \{(4\.\d+)\}',lot)==[f'4.{i}' for i in range(1,7)])
    require('only_three_results_figures',re.findall(r'\\numberline \{(4\.\d+)\}',lof)==['4.1','4.2','4.3'])
    doc=fitz.open(THESIS/(STEM+'.pdf'))
    texts=[flat(p.get_text()) for p in doc]
    require('compiled_pdf_has_44_pages',len(doc)==44)
    require('all_pdf_fonts_embedded',all(doc.extract_font(f[0])[3] for p in doc for f in p.get_fonts(full=True)))
    required={'RQ1':['539 of 1,226','687 (56.04%)','542 records','1,220 (99.51%)'],
              'RQ2':['148 of 373','148 of 178','39.68%','83.15%','98.01%','248 outlet-location units'],
              'RQ3':['43 tracts (13.15%)','40 (12.23%)','197 tracts (60.24%)','130 (39.76%)','0.660 miles','0.576 and 0.644 miles']}
    full=' '.join(texts)
    for group,phrases in required.items():
        for phrase in phrases:require('compiled_claim:'+group+':'+phrase,phrase in full)
    require('compiled_availability_reference','Wayback Machine APIs' in full and 'Internet Archive (2013)' in full)
    page_map={}
    for prefix in [f'Table 4.{i}:' for i in range(1,7)]+[f'Figure 4.{i}:' for i in range(1,4)]:
        hits=[i+1 for i,t in enumerate(texts) if prefix in t]
        require('single_display:'+prefix,len(hits)==1);page_map[prefix]=hits[0]
    bibliography=read(OUT/'bibverify_stdout.json')
    require('bibverify_zero_write_mode',bibliography['dry_run'] and not bibliography['applied'] and all(x is None for x in bibliography['files'].values()))
    hashes=read(OUT/'citation_hash_verification.json')
    require('bibliography_unchanged_during_qa',sha(THESIS/'references.bib')==hashes['bib_before']==hashes['bib_after'])
    require('existing_bibliographic_entries_unchanged',(THESIS/'references.bib').read_text(encoding='utf-8').split('@misc{internetarchive_availability,')[0].rstrip()==(OUT/'before/docs/thesis/references.bib').read_text(encoding='utf-8').rstrip())
    summary={'status':'methods_results_updated_and_compiled','pdf':(THESIS/(STEM+'.pdf')).relative_to(ROOT).as_posix(),
        'pdf_sha256':sha(THESIS/(STEM+'.pdf')),'physical_pages':len(doc),'front_matter_pages':5,'arabic_numbered_pages':39,
        'scope':'Chapters 1-4 and bibliography; no Abstract, Discussion, Conclusion or appendices',
        'methods_sections_changed':['3.1 (existing Figure 3.1a text only)','3.2','3.3','3.4','3.6','3.7','3.8','3.9'],
        'results_sections_changed':['4.1','4.2','4.3'],'results_tables_changed':[f'4.{i}' for i in range(1,7)],
        'minimal_other_change':'Section 1.2: replace final-workflow CDX description with bounded Availability wording; framing unchanged.',
        'display_physical_pages':page_map,'changed_existing_files':changed,'historical_files_preserved_outside_authorized_surface':preserved,
        'scientific_checks_passed':all(checks.values()),'checks':checks,
        'remaining_internal_inconsistencies':[],
        'reference_qa':{'new_reference':'internetarchive_availability','official_source_verified_by_agent':'https://archive.org/help/wayback_api.php',
                        'metadata_automation_complete':bibliography['complete'],'counts':bibliography['counts'],
                        'limitation':'Bibverify Crossref/OpenAlex connections were blocked by the execution environment. This is source_unavailable, not not_found; official documentation title, corporate author and update date were checked directly.'},
        'rendering_checks':{'visual_review_pages':[21,30,33,34,35,36,37,38,39,41],
                            'methods_diagram_reviewed':True,'numeric_columns_separate_at_closeup':True,
                            'no_clipping_observed':True,'all_fonts_embedded':True,
                            'engine_warnings':'Tectonic reports cached font-path and PDF-version notices; exported fonts are embedded, text extracts, and inspected pages render correctly.'},
        'provider_or_model_reacquisition':False,'new_analyses':False,'git_used':False,
        'frozen_phase3_sha256':sha(PHASE/'final_freeze.json')}
    (OUT/'integration_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Retain a versioned delivery alongside the user-facing compiled PDF.
    delivery=OUT/'compiled';delivery.mkdir(exist_ok=True)
    for ext in ['pdf','log','aux','lot','lof','toc','bbl','blg']:
        shutil.copy2(THESIS/(STEM+'.'+ext),delivery/(STEM+'.'+ext))
    source_after=OUT/'after'
    for relative in sources+['figures/results/Figure_3.1A_inventory_construction_corrected.pdf']:
        p=THESIS/relative;q=source_after/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    md=f'''Methods and Results were updated from the accepted closed corrected run and compiled successfully.

Methods changed: 3.1 (wording in the existing Figure 3.1a), 3.2 (frozen September 5 Details, bounded seven-date official Availability discovery and direct raw retrieval), 3.3 (exact model and final validation closure), 3.4 (historical pilot clearly distinguished), 3.6 (373-row stage diagnostic and non-causal wording), 3.7 (Place-ID recurrence denominators), 3.8 (descriptive inventory comparison wording), 3.9 (reconstructable inputs and retained requests/responses, provenance, resume and verification).

Results changed: 4.1, 4.2 and 4.3; Tables 4.1-4.6; approved Figures 4.1-4.3 copied byte-for-byte from the frozen Phase-3 exports. Existing organization and visual design retained. Unsupported superseded unmatched-name subcategories were replaced with direct summaries of the frozen qualified-edge and location-group outputs; no new analysis was introduced.

Section 1.2 received one factual archive-method correction, and one official Internet Archive API reference was added. Chapter 2 and the thesis framing remain unchanged.

PDF: [current compiled thesis](../../../docs/thesis/{STEM}.pdf). Total: {len(doc)} physical pages, comprising 5 front-matter pages and 39 Arabic-numbered pages (including bibliography).

All scientific consistency, table binding, figure hash, frozen evidence preservation, font embedding, reference resolution and display-number checks passed. No overfull/underfull boxes or unresolved references. No remaining internal numeric or methodological inconsistency was found in the current build.

Bibliographic QA limitation: Bibverify ran in dry-run mode with no changes but could not connect to Crossref/OpenAlex. The new web-document reference was checked directly against its official page; no automated metadata-verification success is claimed.

The September 2 run and all accepted classification/archive/Phase-3 artifacts remain unchanged. Original manuscript files and the prior compiled PDF are retained under before/. Discussion, Conclusion and Abstract were neither written nor compiled.
'''
    (OUT/'REVIEW_REPORT.md').write_text(md,encoding='utf-8',newline='\n')
    print(json.dumps({k:summary[k] for k in ['physical_pages','scientific_checks_passed','remaining_internal_inconsistencies','historical_files_preserved_outside_authorized_surface']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
