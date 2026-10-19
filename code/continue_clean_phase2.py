"""Resume the same frozen Phase-2 run after an explicitly restored credit balance."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import sqlite3

from run_clean_phase2 import REPO, verify_inputs, verify_completed, ClassificationStop
from clean_phase2_transport_resume import Runtime
from clean_phase2_evidence import MODEL, VERSION, save
from forced_choice_eligibility import reduce_forced_choice_eligibility
from clean_archive_preflight_postfetch import verify_files
from wayback_snapshot_fetcher import digest, read_json, write_json, now

SEGMENT = 'continuation_v1'


def continuation_choice(root, record):
    cid = record['place_id']
    path = root / f'private/classifications/{cid}.json'
    prior = verify_completed(root, record, read_json(path)) if path.exists() else None
    requests = sorted((root / 'private/transactions' / cid).glob('*.request.json'))
    consumed, credit, prior_reasons = 0, False, []
    for req in requests:
        error = req.with_name(req.name.replace('.request.', '.error.'))
        failure = read_json(error) if error.exists() else {}
        if failure.get('code') == 'credit_balance_exhausted':
            credit = True
            continue
        consumed += 1
        validation = req.with_name(req.name.replace('.request.', '.validation.'))
        if validation.exists() and not read_json(validation)['valid']:
            prior_reasons.append(read_json(validation)['reason'])
        elif failure:
            prior_reasons.append(failure['reason'])
        elif not req.with_name(req.name.replace('.request.', '.response.')).exists():
            prior_reasons.append('interrupted_request_without_received_response')
    reuse = bool(prior and (prior['status'] == 'completed' or not credit))
    return {'place_id': cid, 'reuse_original_result': reuse,
            'credit_exhaustion_observed': credit,
            'prior_noncredit_attempts': consumed,
            'remaining_attempts': max(0, 3-consumed), 'prior_reasons': prior_reasons}


def prepare(root):
    checkpoint = root / 'interrupted_checkpoint_v1'
    verify_files(checkpoint, read_json(checkpoint / 'checkpoint_freeze.json')['files'])
    verify_files(root, read_json(checkpoint / 'transaction_commitments.json'))
    records, _ = verify_inputs(root)
    choices = [continuation_choice(root, r) for r in records]
    plan = {'schema': 'clean-phase2-credit-continuation-v1', 'logical_run_id': root.name,
        'segment': SEGMENT, 'model': MODEL, 'candidate_choices': choices,
        'transport_only_repair': ['credit_balance_exhausted stops calls immediately',
            'check shared stop after waiting for cooldown'],
        'retry_policy': 'Original maximum three noncredit attempts across both segments. Explicit credit-exhaustion rejections do not consume classification attempts. Interrupted requests still consume an attempt. Completed candidates and noncredit exhausted failures are reused unchanged.',
        'unchanged_frozen_inputs_prompts_schema_model_and_reducer': True,
        'workers': 6, 'timeout_seconds': 180,
        'checkpoint_freeze_sha256': digest((checkpoint / 'checkpoint_freeze.json').read_bytes())}
    save(root, f'{SEGMENT}/continuation_plan.json', plan)
    print(json.dumps({'reused': sum(c['reuse_original_result'] for c in choices),
        'continuation_candidates': sum(not c['reuse_original_result'] for c in choices),
        'with_remaining_attempts': sum(not c['reuse_original_result'] and c['remaining_attempts']>0 for c in choices)}), flush=True)


def run(root):
    from dotenv import dotenv_values
    from openai import OpenAI
    records, protocol = verify_inputs(root)
    verify_files(REPO, read_json(root / 'classification_implementation.json')['files'])
    pin = read_json(root / SEGMENT / 'implementation.json')
    verify_files(REPO, pin['files'])
    plan_path = root / SEGMENT / 'continuation_plan.json'
    if digest(plan_path.read_bytes()) != pin['continuation_plan_sha256']:
        raise ValueError('continuation plan changed')
    plan = read_json(plan_path)
    checkpoint = root / 'interrupted_checkpoint_v1'
    verify_files(root, read_json(checkpoint / 'transaction_commitments.json'))
    choices = {c['place_id']: c for c in plan['candidate_choices']}
    key = os.environ.get('OPENAI_API_KEY') or dotenv_values(REPO / '.env', interpolate=False).get('OPENAI_API_KEY')
    if not key:
        raise ClassificationStop('missing_openai_credential')
    client = OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=180)
    segment = root / SEGMENT
    runtime = Runtime(segment, client)
    connection = sqlite3.connect(root / 'runtime_lock.sqlite3', timeout=0)
    connection.execute('BEGIN IMMEDIATE')
    results, origins, work = [], {}, []
    try:
        for r in records:
            c = choices[r['place_id']]
            if c['reuse_original_result']:
                results.append(verify_completed(root, r, read_json(root / f'private/classifications/{r["place_id"]}.json')))
                origins[r['place_id']] = '.'
            else:
                work.append(r)

        def classify(record):
            choice = choices[record['place_id']]
            if choice['remaining_attempts'] > 0:
                return runtime.candidate(record, choice['remaining_attempts'])
            result = {'place_id': record['place_id'], 'candidate_id': record['candidate_id'],
                'packet_sha256': record['packet_sha256'], 'request_sha256': record['request_sha256'],
                'requested_model': MODEL, 'returned_model': None, 'decision': None, 'attempt': 0,
                'status': 'technical_unresolved', 'technical_reasons': choice['prior_reasons'],
                'reducer': reduce_forced_choice_eligibility(None, place_id=record['place_id'], candidate_id=record['candidate_id']).to_public_dict()}
            save(segment, f'private/classifications/{record["place_id"]}.json', result)
            return result

        def progress():
            p = {'accounted': len(results), 'total': 1226, 'status_counts': dict(Counter(r['status'] for r in results)),
                 'updated_at': now(), 'segment': SEGMENT}
            write_json(segment / 'classification_progress.json', p)
            print(json.dumps(p), flush=True)

        progress()
        with ThreadPoolExecutor(max_workers=6) as pool:
            iterator = iter(work)
            pending = {}
            for _ in range(6):
                record = next(iterator, None)
                if record:
                    pending[pool.submit(classify, record)] = record
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        record = pending.pop(future)
                        result = future.result()
                        results.append(result)
                        origins[record['place_id']] = SEGMENT
                        progress()
                        following = next(iterator, None)
                        if following:
                            pending[pool.submit(classify, following)] = following
            except BaseException:
                runtime.stop.set()
                for future in pending:
                    future.cancel()
                raise
    finally:
        runtime.stop.set()
        connection.rollback()
        connection.close()
        client.close()
    if len(results) != 1226 or set(origins) != {r['place_id'] for r in records}:
        raise ValueError('full 1226 accounting not reached')
    verify_files(root, read_json(checkpoint / 'transaction_commitments.json'))
    directories = ['private/transactions', f'{SEGMENT}/private/transactions']
    files = {p.relative_to(root).as_posix(): digest(p.read_bytes())
        for d in directories + ['private/classifications', f'{SEGMENT}/private/classifications']
        for p in (root / d).rglob('*.json')}
    for name in (f'{SEGMENT}/continuation_plan.json', f'{SEGMENT}/implementation.json',
                 'interrupted_checkpoint_v1/checkpoint_freeze.json'):
        files[name] = digest((root / name).read_bytes())
    save(root, 'classification_freeze.json', {'schema': 'clean-phase2-segmented-freeze-v1',
        'candidate_count': 1226, 'files': files, 'result_roots': origins,
        'transaction_directories': directories,
        'input_freeze_sha256': digest((root/'input_freeze.json').read_bytes()),
        'implementation_sha256': digest((root/'classification_implementation.json').read_bytes()),
        'status_counts': dict(Counter(r['status'] for r in results)), 'phase3_execution_allowed': False})
    print('All 1226 candidates accounted; classification freeze complete. Offline Phase-2 review remains.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    (prepare if a.command == 'prepare' else run)(a.root.resolve())
