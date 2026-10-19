"""Minimal offline vector text edits in the two approved Methods panels."""
import fitz,shutil
import run_expanded_final as run
OUT,ROOT=run.OUT,run.ROOT
DEST=OUT/'methods_diagram';DEST.mkdir(exist_ok=True)
FONT=r'C:\Windows\Fonts\arial.ttf';BOLD=r'C:\Windows\Fonts\arialbd.ttf'
body=(36/255,55/255,70/255);blue=(28/255,78/255,121/255)

def edit(name,changes):
    source=OUT/'thesis_before/figures/results'/name
    doc=fitz.open(source);page=doc[0];blocks=page.get_text('blocks');targets=[]
    for prefix,text,rect,size,bold,fill in changes:
        found=[b for b in blocks if b[4].startswith(prefix)]
        assert len(found)==1,(prefix,found)
        erase=fitz.Rect(found[0][:4]);erase.x0-=.3;erase.x1+=.3
        page.add_redact_annot(erase,fill=fill)
        targets.append((text,rect,size,bold))
    page.apply_redactions(images=0,graphics=0)
    for text,rect,size,bold in targets:
        remaining=page.insert_textbox(fitz.Rect(rect),text,fontsize=size,lineheight=1.16,
            fontname='ExpandedBold' if bold else 'ExpandedBody',fontfile=BOLD if bold else FONT,
            color=blue if bold else body)
        assert remaining>=0,(text,remaining)
    path=DEST/name;doc.save(path,garbage=4,deflate=True);doc.close()
    check=fitz.open(path);assert 'pilot passed' not in check[0].get_text()
    check[0].get_pixmap(dpi=150).save(DEST/(name+'.png'));check.close()
    shutil.copy2(path,ROOT/'docs/thesis/figures/results'/name)
    return {'source_sha256':run.sha(source),'result_sha256':run.sha(path),'changes':[{'old_prefix':c[0],'new_text':c[1]} for c in changes]}

records={}
name='Figure_3.1A_inventory_construction_corrected.pdf'
records[name]=edit(name,[
 ('Fixed query\nplan','Fixed base\nquery plan',(65.7,221,149,248),10.6,True,(1,1,1)),
 ('• Retrieve candidate','Base retrieval\n> 60-result check\n> Bounded grid search\n  (saturated queries;\n  refine if saturated)\n> Place-ID union /\n  deduplication\nDated query ledger',(65,433,150,537),8.4,False,(1,1,1)),
 ('•\xa0Frozen\xa0Sept.','Frozen responses;\ncohort/source dates.',(210,421,303,444),9.04,False,(1,1,1)),
 ('Pre-run evidence-order pilot','Development-only evidence pilot',(365.6,143,570,159),9.4,True,(1,1,1)),
 ('Checks execution, evidence separation','Checks execution and evidence order.\nDoes not validate corrected archive packets.',(365.6,216.9,570,239),8.2,False,(1,1,1)),
 ('pilot passed','development only',(397.5,279.4,480,291),8,False,(1,1,1)),
 ('60 ZIP-coded search definitions','Base: 60 ZIPs x 3 terms; saturated queries receive bounded subdivision. Spillover remains observable.',(55.9,46.9,764,63),8.8,False,(247/255,250/255,252/255))])
name='Figure_3.1B_correspondence_proximity_matched_scale.pdf'
records[name]=edit(name,[
 ('Observed workflow and Board outlets','Workflow: all included, usable coordinates. Board: fixed\n99 ZIPs (60 targets + original-candidate returned ZIPs).',(461.2,111.9,729,137),8.5,False,(247/255,250/255,252/255)),
 ('Screened Board outlet\ncoordinates','Usable screened Board\ncoordinates in the fixed\n99-ZIP supply frame.',(221.6,722,316,766),9,False,(1,1,1))])
run.save(DEST/'provenance.json',{'method':'Local vector text replacement; original diagrams preserved; existing iconography and geometry retained.','panels':records})
print('Both Methods panels updated and rendered.')
