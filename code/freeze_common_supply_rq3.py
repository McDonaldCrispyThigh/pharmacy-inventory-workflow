"""Freeze the common-frame RQ3 text/table delivery and verify protected artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,re,difflib
import pandas as pd
import run_common_supply_rq3 as current
ROOT,OUT,THESIS=current.ROOT,current.OUT,current.THESIS
read=current.frozen.read_json;sha=current.frozen.sha;save=current.save
assert not (OUT/'final_freeze.json').exists()
assert read(OUT/'independent_verification.json')['passed'] and read(OUT/'text_table_consistency.json')['passed']
protected={}
for folder in [current.BASE,ROOT/'analysis/maintenance/anti_defensive_20260908_v1',ROOT/'analysis/maintenance/common_supply_audit_20260908_v1',ROOT/'analysis/maintenance/figure31a_user_replacement_20260908_v1']:
    manifest=read(folder/'final_freeze.json')
    bad=[name for name,h in manifest['files'].items() if sha(folder/name)!=h]
    assert not bad,(folder,bad)
    protected[folder.relative_to(ROOT).as_posix()]=len(manifest['files'])
manifest=read(OUT/'rq3/run_manifest.json')
assert all(sha(OUT/'rq3'/p)==h for p,h in manifest['artifacts'].items())
assert all(sha(ROOT/p)==h for p,h in read(OUT/'implementation.json')['files'].items())
before=read(OUT/'manuscript_before_hashes.json')
modified=[p for p,h in before.items() if sha(ROOT/p)!=h]
allowed={f'docs/thesis/chapters/{n}' for n in ['01_introduction_rewrite.tex','02_literature_rewrite.tex','03_methodology.tex','04_results.tex']}|{f'docs/thesis/tables/{n}' for n in ['table_methods_spatial_sensitivities.tex','table4_paired_spatial.tex','table5_neighborhood.tex']}
assert set(modified)==allowed,modified
# Explicitly verify every figure and the existing Word/PDF, plus all RQ1/RQ2 tables.
assert all(sha(ROOT/p)==h for p,h in before.items() if p not in allowed)
rq3_start='\\section{Paired Potential Proximity: RQ3}'
old=(OUT/'before/docs/thesis/chapters/04_results.tex').read_text(encoding='utf-8')
new=(THESIS/'chapters/04_results.tex').read_text(encoding='utf-8')
assert old.split(rq3_start)[0]==new.split(rq3_start)[0]
# Preserve sources, current text, and compact readable diffs inside the new freeze.
for rel in allowed:
    dest=OUT/'deliverables'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
for name in ['run_common_supply_rq3.py','verify_common_supply_rq3.py','report_common_supply_rq3.py','sync_common_supply_rq3_text.py','qa_common_supply_text.py','freeze_common_supply_rq3.py']:
    dest=OUT/'implementation'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'code'/name,dest)
shutil.copy2(ROOT/'AGENTS.md',OUT/'deliverables/AGENTS.md')
old=(OUT/'before/AGENTS.md').read_text(encoding='utf-8');new=(ROOT/'AGENTS.md').read_text(encoding='utf-8')
(OUT/'AGENTS.md.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),encoding='utf-8')
save(OUT/'visual_review.json',{'reviewed':'The three standalone rendered table pages were inspected; all table bodies and notes are legible without clipping. Figures and the existing thesis PDF/Word were not rendered or modified.','tables':['3.2','4.5','4.6'],'passed':True})
save(OUT/'final_verification.json',{'passed':True,'protected_frozen_files':protected,'unchanged_rq1_rq2':True,'unchanged_figure_files_and_figure_blocks':True,'unchanged_existing_pdf_word':True,'modified_thesis_files':modified,'new_model_calls':0,'new_provider_calls':0,'stage':'RQ3 common-frame primary adopted; text and tables complete; figure revision pending'})
files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_freeze.json'}
save(OUT/'final_freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'status':'common_supply_rq3_text_tables_complete_stop_before_figures','candidate_count':2062,'origin_count':327,'common_supply_zip_count':read(OUT/'common_supply_frame.json')['common_zip_count'],'files':files,'new_model_calls':0,'new_provider_calls':0})
assert all(sha(OUT/p)==h for p,h in files.items())
save(OUT.parent/'CURRENT.json',{'primary_rq3':OUT.relative_to(ROOT).as_posix(),'final_freeze_sha256':sha(OUT/'final_freeze.json'),'rq1_rq2_source':current.BASE.relative_to(ROOT).as_posix(),'stage':'text_tables_complete_figures_pending'})
setup=OUT.parent/'20260908_v1'
save(setup/'setup_status.json',{'status':'setup_only_no_spatial_outputs','reason':'Initial adapter used an incorrect BoardFramesV2 attribute before spatial calculation. Corrected implementation and full results are in 20260908_v2.','preserved_as_provenance':True})
print(json.dumps({'passed':True,'frozen_files':len(files),'protected_freezes':protected,'changed_thesis_files':len(modified),'new_api_calls':0,'figures_unchanged':True}))
