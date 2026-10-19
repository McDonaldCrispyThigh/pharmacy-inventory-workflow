"""Predeclared transport diagnostic; selection uses public names and acquisition only."""
from collections import Counter
from pathlib import Path
import re
import unicodedata

from corrected_run_store import read_sealed, seal_json, file_sha, sha, unit_key, utcnow

SEED = 'nets-ai-official-archive-transport-40-v1'
STRATA = ('outscraper_extracted', 'structural_only', 'empty_extraction', 'cdx_parser', 'url_binding')
SCHEDULE = tuple(f'{year}0101000000' for year in (1996,2001,2006,2011,2016,2021)) + ('20260905000000',)
OUTCOMES = ('direct_raw','outscraper_extracted','structural_only','no_archive','no_website','collection_failure')


def public_name(value):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold()))


def select_sample(rows):
    selected = []
    for stratum in STRATA:
        for name_group in ('repeated','single'):
            pool = [r for r in rows if r['stratum']==stratum and r['name_group']==name_group]
            pool.sort(key=lambda r:sha([SEED,r['place_id']]))
            if len(pool)<4:
                raise ValueError('insufficient candidates for fixed diagnostic stratum')
            selected.extend(pool[:4])
    selected.sort(key=lambda r:sha([SEED,'execution',r['place_id']]))
    if len(selected)!=40 or len({r['place_id'] for r in selected})!=40:
        raise ValueError('diagnostic must contain exactly 40 unique candidates')
    return selected


def diagnostic_gate(records):
    counts = Counter(r['outcome'] for r in records)
    technical = sum(r['outcome']=='collection_failure' or
                    (r['outcome']=='structural_only' and any(p['status']!='success' for p in r.get('pages',[])))
                    for r in records)
    valid = counts['direct_raw']+counts['outscraper_extracted']+counts['no_archive']
    binding = sum(r.get('accepted_source_binding_failures',0) for r in records)
    passed = (len(records)==40 and len({r['place_id'] for r in records})==40 and
              technical<20 and valid>technical and binding==0)
    return {'passed':passed,'candidate_count':len(records),'technical_candidates_without_text':technical,
            'valid_text_or_complete_negative_candidates':valid,'accepted_source_binding_failures':binding,
            'rule':'Exactly 40 unique candidates; technical failure including textless replay failures must be below 50% and outnumbered by substantive evidence or complete valid negative lookup; no accepted binding violation.',
            'inference_boundary':'Stratified transport diagnostic, not a population archive-coverage estimate.'}


def initialize_plan(old_root, new_root):
    old_root,new_root=Path(old_root),Path(new_root)
    destination=new_root/'plan.json'
    if destination.exists():
        return read_sealed(destination)
    manifest=read_sealed(old_root/'evidence_manifest.json')
    audit=read_sealed(old_root/'phase1_failure_audit.json')
    report=__import__('json').loads((old_root/'acquisition_preflight.json').read_text(encoding='utf-8'))
    expected=dict(zip(OUTCOMES,(0,33,40,0,138,1015)))
    if report['wayback_outcomes']!=expected or report['usable_website_candidates']!=1088:
        raise ValueError('authoritative failed acquisition distribution differs')
    for relative,digest in manifest['details_files'].items():
        if file_sha(old_root/relative)!=digest:
            raise ValueError('frozen September 5 Details hash mismatch')
    details={}
    for relative in manifest['details_files']:
        received=read_sealed(old_root/relative)
        details[received['place_id']]=received
    name_counts=Counter(public_name(r['details']['displayName']['text']) for r in details.values())
    failures={r['place_id']:r['primary_reason'] for r in audit['failure_rows']}
    by_outcome={r['place_id']:r['frozen_outcome'] for r in audit['candidate_rows']}
    rows=[]
    for ident,received in details.items():
        outcome=by_outcome[ident]
        reason=failures.get(ident)
        stratum=(outcome if outcome in STRATA else 'empty_extraction' if reason=='outscraper_empty_extraction'
                 else 'cdx_parser' if reason=='cdx_parser_validation_failed'
                 else 'url_binding' if reason in {'outscraper_cdx_original_target_mismatch','outscraper_target_mismatch'} else None)
        name=received['details']['displayName']['text']
        rows.append({'place_id':ident,'website':received['details'].get('websiteUri'),
                     'public_name':name,'name_group':'repeated' if name_counts[public_name(name)]>1 else 'single',
                     'stratum':stratum,'prior_outcome':outcome,'prior_failure_reason':reason,
                     'details_relative_path':f'private/details/{unit_key(ident)}.json',
                     'details_sha256':manifest['details_files'][f'private/details/{unit_key(ident)}.json']})
    sample=select_sample(rows)
    usable=[r for r in rows if r['prior_outcome']!='no_website']
    if len(rows)!=1226 or len(usable)!=1088:
        raise ValueError('fixed source frame differs')
    plan={'schema':'nets-ai-official-transport-plan-v1','run_id':new_root.name,'created_at':utcnow(),
          'old_run_directory':old_root.resolve().as_posix(),'old_manifest_sha256':file_sha(old_root/'evidence_manifest.json'),
          'authoritative_failed_outcomes':expected,'candidate_count':1226,'usable_websites':1088,
          'rows':sorted(rows,key=lambda r:r['place_id']),'diagnostic_sample':sample,'sample_seed':SEED,
          'availability_schedule':list(SCHEDULE),'availability_scope':'Seven fixed nearest-capture queries; never an exhaustive archive census or standalone proof of archive absence.',
          'diagnostic_page_selection':'Latest officially identified suitable snapshot at or before plan creation; identical snapshot tested independently via direct and Outscraper.',
          'full_page_selection':'Oldest successful snapshot per returned calendar year; at most three chronological candidate captures per year.',
          'transport':{'timeout_seconds':15,'max_body_bytes':2000000,'min_interval_seconds':1,
                       'implicit_429_cooldown_seconds':5,'retry_after_honored':True,'attempts_per_request':1,
                       'max_cdx_pages':10,'max_cdx_rows':500,'max_timemap_pages':10,
                       'max_outscraper_jobs':20000,'max_outscraper_age_seconds':3600},
          'full_run_gate':diagnostic_gate([])['rule'],
          'constraints':{'openai_calls_allowed':False,'classification_allowed':False,'google_calls_allowed':False,
                         'outscraper_index_extraction_allowed':False,'preserve_prior_artifacts':True,
                         'full_archive_requires_diagnostic_pass':True,'stop_after_full_preflight':True},
          'official_contracts':['https://archive.org/help/wayback_api.php',
              'https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server',
              'https://datatracker.ietf.org/doc/html/rfc7089']}
    seal_json(destination,plan)
    return plan
