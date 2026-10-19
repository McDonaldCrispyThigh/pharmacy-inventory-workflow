"""Separate clean archive stages. No model, Google, Board, or old collector API."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json

from wayback_snapshot_fetcher import encoded,digest,write,write_json,read_json,website_key,Crawler,input_records,now
from clean_archive_discovery import SCHEDULE,run_discovery,Discovery


def unseal(path):
    value=read_json(path)
    if value.get('sha256')!=digest(encoded(value['payload'])):raise ValueError('frozen source record changed')
    return value['payload']


def prepare(root,source):
    root,source=Path(root),Path(source)
    if (root/'plan.json').exists():return read_json(root/'plan.json')
    manifest=unseal(source/'evidence_manifest.json');rows=[]
    for relative,expected in sorted(manifest['details_files'].items()):
        path=source/relative;body=path.read_bytes()
        if digest(body)!=expected:raise ValueError('frozen September 5 Details changed')
        received=unseal(path);details=received['details'];website=details.get('websiteUri')
        usable=False;reason='missing_website'
        if website:
            try:website_key(website);usable=True;reason=None
            except ValueError:reason='invalid_website_under_frozen_contract'
        destination=Path('private/details')/(received['place_id']+'.json')
        write(root/destination,body,immutable=True)
        rows.append({'candidate_id':received['place_id'],'website_url':website,'usable_website':usable,
                     'no_website_reason':reason,'details_file':destination.as_posix(),'details_sha256':expected,
                     'returned_review_count':len(details.get('reviews',[]))})
    counts={'candidates':len(rows),'successful_details':len(rows),'returned_websites':sum(bool(r['website_url']) for r in rows),
            'usable_websites':sum(r['usable_website'] for r in rows),'returned_reviews':sum(r['returned_review_count'] for r in rows)}
    if counts!={'candidates':1226,'successful_details':1226,'returned_websites':1089,'usable_websites':1088,'returned_reviews':5401}:
        raise ValueError('frozen input frame counts differ')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    plan={'schema':'clean-full-archive-plan-v1','created_at':now(),'run_id':root.name,'candidates':rows,
          'input_counts':counts,'source_manifest_sha256':digest((source/'evidence_manifest.json').read_bytes()),
          'schedule':list(SCHEDULE),'snapshot_cutoff':stamp,
          'bounded_scope':'Seven fixed nearest-currently-accessible snapshot requests, not exhaustive CDX history. No candidate-adaptive dates.',
          'discovery':{'timeout_seconds':20,'max_attempts':3,'minimum_interval_seconds':0.1,
                       'cache':'identical raw website URL plus requested date only; every candidate retains all seven query bindings'},
          'fetch':{'concurrency':1,'timeout':20,'max_attempts':3,'backoff_base':30,'backoff_cap':300},
          'identity_deduplication':'capture timestamp plus normalized archived original URL; shared raw snapshot bound independently to each candidate',
          'phase_order':['all_discovery','seal_all_snapshots','all_raw_fetch','all_local_parse','preflight_stop'],
          'promotion_rule':'Every bounded lookup successful, all raw commitments valid, parser reproducible; any residual raw-fetch failure conservatively blocks recommendation until reviewed. No automatic classification.',
          'openai_allowed':False,'google_allowed':False,'outscraper_allowed':False,'thesis_changes_allowed':False,
          'historical_opening_year_from_capture_alone':False}
    write_json(root/'plan.json',plan,immutable=True)
    print(json.dumps(counts,indent=2))
    return plan


def verify_implementation(root):
    commitment=read_json(root/'implementation_manifest.json')
    for path,expected in commitment['files'].items():
        if digest(Path(path).read_bytes())!=expected:raise ValueError('frozen clean implementation changed')


def verify_snapshots(root):
    plan=read_json(root/'plan.json');seal=read_json(root/'snapshot_seal.json')
    snapshot_bytes=(root/'snapshots.json').read_bytes()
    if digest(snapshot_bytes)!=seal['sha256']:raise ValueError('sealed snapshot manifest changed')
    discovery=read_json(root/'discovery_manifest.json')
    if digest((root/'discovery_manifest.json').read_bytes())!=seal['discovery_manifest_sha256']:
        raise ValueError('sealed discovery manifest changed')
    if discovery['plan_sha256']!=digest((root/'plan.json').read_bytes()):raise ValueError('discovery plan changed')
    for relative,expected in discovery['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes())!=expected:
            raise ValueError('sealed discovery receipt changed')
    verifier=Discovery(root/'discovery',plan)
    candidates={r['candidate_id']:r for r in plan['candidates']}
    records=json.loads(snapshot_bytes)['records']
    if records:input_records(snapshot_bytes)
    if len(records)!=seal['snapshot_count']:raise ValueError('snapshot count changed')
    for record in records:
        for ident in record['candidate_bindings']:
            row=candidates[ident]
            proof=read_json(root/'discovery/candidates'/(ident+'.json'))
            if record['candidate_id'].removeprefix('snapshot_') not in proof['snapshot_ids']:
                raise ValueError('candidate snapshot association changed')
            for q in proof['queries']:verifier.verify(q)
            if website_key(row['website_url'])!=website_key(record['original_website_url']):
                raise ValueError('shared snapshot candidate website differs')
    return plan,records


def fetch(root):
    verify_implementation(root)
    plan,records=verify_snapshots(root)
    if not records:
        write_json(root/'no_confirmed_snapshots.json',{'raw_fetch_count':0},immutable=True)
        return
    result=Crawler(root/'raw_retrieval',**plan['fetch']).run(root/'snapshots.json')
    write_json(root/'raw_fetch_seal.json',{'execution':result['execution'],
               'manifest_sha256':digest(encoded(result)),'snapshot_manifest_sha256':digest((root/'snapshots.json').read_bytes())},immutable=True)
    print(json.dumps({k:v for k,v in result.items() if k not in {'files','attempt_results','successful_responses'}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','discover','fetch'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source',type=Path)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.run_dir,args.source)
    elif args.stage=='discover':
        verify_implementation(args.run_dir)
        run_discovery(args.run_dir,read_json(args.run_dir/'plan.json'))
    else:fetch(args.run_dir)
