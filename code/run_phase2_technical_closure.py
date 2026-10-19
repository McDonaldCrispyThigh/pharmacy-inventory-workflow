"""Allow-listed retries for the nine failures remaining after offline salvage."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sqlite3

from clean_phase2_closure_runtime import Runtime, verify_completed
from clean_phase2_closure_contract import build_retry_record, validate_record
from clean_phase2_evidence import MODEL
from run_clean_phase2 import inventory
from clean_archive_preflight_postfetch import verify_files
from wayback_snapshot_fetcher import read_json, write_json, write, digest, now

REPO=Path(__file__).resolve().parents[1]


def preservation(base,out):
    if inventory(base)!=read_json(out/'checkpoint_preservation_before.json'):
        raise ValueError('frozen original Phase-2 checkpoint changed')
    for row in read_json(out/'valid_1156_manifest.json')['records']:
        verify_files(base,row['files'])


def prepare(base,out):
    preservation(base,out)
    summary=read_json(out/'technical_audit_summary.json')
    valid={r['place_id'] for r in read_json(out/'valid_1156_manifest.json')['records']}
    ids=summary['retry_place_ids']
    if len(ids)!=len(set(ids)) or set(ids)&valid or len(ids)+summary['salvaged_count']!=70:
        raise ValueError('retry allow-list violates frozen frame')
    rows=[]
    for cid in ids:
        record=build_retry_record(read_json(base/f'private/inputs/{cid}.json'))
        validate_record(record)
        if len(__import__('json').dumps(record['request']).encode())+128000+4096>400000:
            raise ValueError('retry request exceeds conservative context bound')
        name=f'private/inputs/{cid}.json'
        write_json(out/name,record,immutable=True)
        rows.append({'place_id':cid,'path':name,'sha256':digest((out/name).read_bytes()),
                     'original_packet_sha256':record['original_packet_sha256'],'packet_sha256':record['packet_sha256']})
    plan={'model':MODEL,'candidate_rows':rows,'candidate_count':len(rows),'workers':3,'max_attempts':3,
        'timeout_seconds':180,'created_at':now(),'only_unresolved_after_offline_revalidation':True,
        'original_valid_1156_excluded':True,'substantive_system_instructions_unchanged':True,
        'new_model_calls_before_audit_completion':0,'no_implicit_further_retry_rounds':True}
    write_json(out/'retry_plan.json',plan,immutable=True)
    print(f'Frozen retry packets for {len(rows)} candidates.',flush=True)


def run(base,out):
    from openai import OpenAI
    from dotenv import dotenv_values
    preservation(base,out)
    freeze=read_json(out/'precall_freeze.json')
    verify_files(out,freeze['files'])
    verify_files(REPO,read_json(out/'implementation.json')['files'])
    plan=read_json(out/'retry_plan.json')
    if plan['model']!=MODEL:raise ValueError('model changed')
    records=[validate_record(read_json(out/r['path'])) for r in plan['candidate_rows']]
    allowed=set(read_json(out/'technical_audit_summary.json')['retry_place_ids'])
    if {r['place_id'] for r in records}!=allowed:raise ValueError('retry frame changed')
    key=os.environ.get('OPENAI_API_KEY') or dotenv_values(REPO/'.env',interpolate=False).get('OPENAI_API_KEY')
    if not key:raise ValueError('missing OpenAI credential')
    client=OpenAI(api_key=key,base_url='https://api.openai.com/v1',max_retries=0,timeout=180)
    runtime=Runtime(out,client)
    lock=sqlite3.connect(out/'runtime_lock.sqlite3',timeout=0);lock.execute('BEGIN IMMEDIATE')
    results=[]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(runtime.candidate,r,3) for r in records]
            try:
                for f in as_completed(futures):
                    results.append(f.result())
                    progress={'retry_accounted':len(results),'retry_total':len(records),
                        'status_counts':dict(Counter(r['status'] for r in results)), 'updated_at':now()}
                    write_json(out/'retry_progress.json',progress)
                    print(__import__('json').dumps(progress),flush=True)
            except BaseException:
                runtime.stop.set()
                for f in futures:f.cancel()
                raise
    finally:
        runtime.stop.set();lock.rollback();lock.close();client.close()
    preservation(base,out)
    files={p.relative_to(out).as_posix():digest(p.read_bytes())
        for d in ('private/classifications','private/transactions') for p in (out/d).rglob('*.json')}
    write_json(out/'retry_freeze.json',{'candidate_count':len(results),'files':files,
        'status_counts':dict(Counter(r['status'] for r in results)),
        'precall_freeze_sha256':digest((out/'precall_freeze.json').read_bytes()),'phase3_allowed':False},immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();(prepare if a.command=='prepare' else run)(a.base.resolve(),a.out.resolve())
