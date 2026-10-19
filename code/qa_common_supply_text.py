"""Local text/table consistency checks without changing thesis PDFs or figures."""
from pathlib import Path
import json,re,shutil,subprocess
import run_common_supply_rq3 as current
OUT,ROOT,THESIS=current.OUT,current.ROOT,current.THESIS
read=current.frozen.read_json
report=read(OUT/'report.json')
methods=(THESIS/'chapters/03_methodology.tex').read_text(encoding='utf-8')
results=(THESIS/'chapters/04_results.tex').read_text(encoding='utf-8')
assert report['common_supply_frame']['common_zip_count']==102
assert [v['discordant'] for v in report['paired_states']]==[48,46,4]
assert report['nearer_counts']=={'workflow_nearer':206,'board_nearer':121,'ties':0}
counts={r['surface']:r for r in report['coordinate_counts']}
assert counts['google_include']['valid_coordinate_rows']==592
assert counts['board_screened']['input_rows']==346 and counts['board_screened']['valid_coordinate_rows']==307
assert report['missing_board_coordinates']['missing_coordinate_rows']==39
assert report['missing_board_coordinates']['mailing_zip_count']==32
interior=[v for v in report['spatial_sensitivities'] if v['scenario']=='collection_frame_interior']
assert [v['origins'] for v in interior]==[318,305,269]
assert [v['discordant'] for v in interior]==[47,45,3]
assert [v['discordance_pct'] for v in interior]==[14.78,14.75,1.12]
conditional=report['conditional_ranges']
assert [v['both_sources_conditionally_determinate_count'] for v in conditional]==[111,243,322]
assert [v['discordant_among_determinate'] for v in conditional]==[10,5,0]
assert [v['board_conditionally_unresolved_count'] for v in conditional]==[216,84,5]
assert [v['workflow_conditionally_unresolved_count'] for v in conditional]==[3,2,1]
assert all(v['board_beyond_rate_lower_pct']==0 for v in conditional)
assert all(v['board_beyond_rate_upper_pct']==target for v,target in zip(conditional,[66.06,25.69,1.53]))
assert all(v['workflow_beyond_rate_lower_pct']==target for v,target in zip(conditional,[56.57,14.07,0]))
assert all(v['workflow_beyond_rate_upper_pct']==target for v,target in zip(conditional,[57.49,14.68,.31]))
text='\n'.join((THESIS/'chapters'/n).read_text(encoding='utf-8') for n in ['01_introduction_rewrite.tex','02_literature_rewrite.tex','03_methodology.tex','04_results.tex'])
for phrase in ['supply admission differs','not common to both inventories','99-ZIP','99 collection-frame','source-specific supply']:
    assert phrase not in text,phrase
# Verify that editing was limited to the explicitly authorized sections.
allowed={'01_introduction_rewrite.tex':{'Study Design and Research Questions','Contributions and Scope'},'02_literature_rewrite.tex':{'Spatial Measurement and the Position of This Study'},'03_methodology.tex':{'Study Area and Analysis Frames','Paired Potential-Proximity Comparison','Reproducibility and Data Handling'},'04_results.tex':{'Paired Potential Proximity: RQ3'}}
def sections(t):
    parts=re.split(r'(\\section\{[^}]+\})',t)
    return {'preamble':parts[0],**{parts[i]:parts[i+1] for i in range(1,len(parts),2)}}
changed_sections={}
for name,permitted in allowed.items():
    a=sections((OUT/'before/docs/thesis/chapters'/name).read_text(encoding='utf-8'))
    b=sections((THESIS/'chapters'/name).read_text(encoding='utf-8'))
    assert a.keys()==b.keys()
    changed=[key for key in a if a[key]!=b[key]]
    assert all(key in {'\\section{'+v+'}' for v in permitted} for key in changed),(name,changed)
    changed_sections[name]=changed
    assert re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',a.get('preamble',''),re.S)==re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',b.get('preamble',''),re.S)
    old=(OUT/'before/docs/thesis/chapters'/name).read_text(encoding='utf-8')
    new=(THESIS/'chapters'/name).read_text(encoding='utf-8')
    assert re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',old,re.S)==re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',new,re.S)
# Tables are compiled alone using the thesis class and the existing type size.
# This does not load any figure or manuscript chapter, or touch the thesis builds.
qa=OUT/'table_qa';qa.mkdir(exist_ok=True)
assert not (OUT/'text_table_consistency.json').exists()
shutil.copy2(THESIS/'thesis.cls',qa/'thesis.cls')
names=['table_methods_spatial_sensitivities','table4_paired_spatial','table5_neighborhood']
for n in names:
    source=(THESIS/'tables'/f'{n}.tex').read_text(encoding='utf-8')
    source=re.sub(r'\\begin\{table\}\[[^]]+\]',lambda m:r'\begin{table}[H]',source)
    (qa/f'{n}.tex').write_text(source,encoding='utf-8')
wrapper=r'''\documentclass[defaultstyle,11pt]{thesis}
\usepackage{booktabs,array,graphicx,amsmath,float}
\setlength{\emergencystretch}{2em}
\SuspendPrologue
\begin{document}
\setstretch{1}\normalsize
\setcounter{chapter}{3}\setcounter{table}{1}
\input{table_methods_spatial_sensitivities}
\clearpage
\setcounter{chapter}{4}\setcounter{table}{4}
\input{table4_paired_spatial}
\clearpage
\input{table5_neighborhood}
\clearpage
\end{document}
'''
(qa/'tables.tex').write_text(wrapper,encoding='utf-8')
build=subprocess.run([str(THESIS/'tectonic.exe'),'--only-cached','--keep-logs','tables.tex'],cwd=qa,capture_output=True,text=True)
(qa/'build_output.txt').write_text(build.stdout+build.stderr,encoding='utf-8')
assert build.returncode==0,build.stderr
log=(qa/'tables.log').read_text(encoding='utf-8',errors='replace')
assert not re.search('Overfull|Float too large|Missing character:',log)
import fitz
with fitz.open(qa/'tables.pdf') as doc:
    content_pages=[page for page in doc if len(page.get_text().strip())>20]
    assert len(content_pages)==3
    for i,page in enumerate(content_pages):page.get_pixmap(dpi=120).save(qa/f'page_{i+1}.png')
current.save(OUT/'text_table_consistency.json',{'passed':True,'numeric_claims_checked':True,'changed_sections':changed_sections,'figure_blocks_unchanged':True,'rq1_rq2_sections_unchanged':True,'table_content_pages':3,'qa_wrapper_leading_blank_page':True,'table_overflow':False,'canonical_pdf_and_word_not_built':True})
print(json.dumps({'passed':True,'table_pages':3,'changed_sections':changed_sections}))
