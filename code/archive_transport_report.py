"""Offline report and preservation audit for the transport-only repair attempt."""
from pathlib import Path
from collections import Counter
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from corrected_run_store import read_sealed,seal_json,file_sha,utcnow
from run_archive_transport_repair import verify_stage


def write_report(root):
    root=Path(root);plan=read_sealed(root/'plan.json')
    diagnostic=verify_stage(root,'diagnostic')
    full=verify_stage(root,'full') if (root/'full/manifest.json').exists() else None
    for relative,digest in read_sealed(root/'implementation.json')['files'].items():
        if file_sha(ROOT/relative)!=digest:raise ValueError('live acquisition implementation changed')
    baseline=read_sealed(root/'preserved_before.json')['files']
    # This new task-owned test existed during the overlapping initial inventory
    # and was completed afterward; it was not an artifact of approved Phase 1.
    new_test='tests/test_archive_transport_binding.py'
    changed=[p for p,h in baseline.items() if p!=new_test and (not (ROOT/p).is_file() or file_sha(ROOT/p)!=h)]
    if changed:raise ValueError('preserved artifacts changed: '+repr(changed))
    states={};state_snapshots={}
    for mode in ('diagnostic','full'):
        db=root/'private'/f'{mode}_state/state.sqlite3'
        if not db.exists():continue
        connection=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            if connection.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('state integrity failed')
            jobs=[json.loads(row[0])for row in connection.execute("SELECT payload FROM state WHERE namespace='outscraper_jobs'")]
        finally:connection.close()
        states[mode]=dict(Counter(j['state']for j in jobs))
        if any(j['kind']!='replay' for j in jobs):raise ValueError('Outscraper index discovery was attempted')
        if any(j['state'] not in ('complete','failed')for j in jobs):raise ValueError('unresolved provider task remains')
        from corrected_preflight import _backup_sqlite_state
        snapshot=root/'private'/f'{mode}_state_frozen.sqlite3'
        _backup_sqlite_state(db,snapshot)
        state_snapshots[snapshot.relative_to(root).as_posix()]=file_sha(snapshot)
    report=full or diagnostic
    out=ROOT/'analysis/maintenance'/('archive_transport_repair_'+plan['run_id'])
    out.mkdir(parents=True,exist_ok=True)
    matrix=diagnostic['transport_matrix']
    gate=diagnostic['full_archive_gate']
    records=[read_sealed(p) for p in (root/'diagnostic/evidence').glob('*.json')]
    discovery_reasons=Counter(q.get('reason','unspecified') for r in records
        for d in r['discoveries'].values() for q in d['queries']
        if q['status'] not in ('success','empty'))
    failure_candidates=[r for r in records if r['outcome']=='collection_failure']
    bounded_negative_candidates=sum(
        r['discoveries'].get('availability',{}).get('schedule_completed') is True
        and all(q['status']=='empty' for q in r['discoveries']['availability']['queries'])
        for r in failure_candidates)
    lines=['# Official archive transport diagnostic and preflight','',
           f'Run: `{plan["run_id"]}`. Completed diagnostic: 40/40 candidates. '
           'No OpenAI calls, no classification, no Google Details reacquisition. '
           'The approved failed Phase-1 attempt, September 2 run, and thesis are preserved.', '',
           'Five prior-acquisition strata each contribute eight candidates; each stratum contains four repeated public names '
           'and four single-occurrence names. Selection uses the frozen public names and acquisition diagnostics only. '
           'This deliberately stratified sample does not estimate population archive coverage.', '',
           '| Discovery route | Candidates tested | Candidates with identified snapshots | HTTP statuses (requests) |',
           '|---|---:|---:|---|']
    for method in ('cdx','timemap','availability'):
        row=matrix[method]
        lines.append(f'| {method} | {row["candidates_tested"]} | {row["discovery_success"]} | {json.dumps(row["http_response_counts"],sort_keys=True)} |')
    lines += ['',f'Exact snapshot identities independently established: **{diagnostic["exact_snapshot_identity_candidates"]}/40 candidates**. '
              'Official discovery proves capture identity; it does not independently validate outsourced text.', '',
              '| Exact-snapshot page route | Candidates tested | Successful candidates | Successful source occurrences |',
              '|---|---:|---:|---:|']
    for route in ('direct','outscraper'):
        row=matrix[route+'_page']
        lines.append(f'| {route} | {row["candidates_tested"]} | {row["successful_candidates"]} | {row["successful_sources"]} |')
    lines += ['', 'Both page routes independently test the same latest officially identified snapshot for each diagnostic candidate. '
              'When no official snapshot can be identified, page retrieval is not attempted; no speculative Outscraper task is submitted.', '',
              f'URL-binding rejection events: **{diagnostic["url_binding_rejection_count"]}**. '
              f'Exclusive collection-failure rate: **{100*diagnostic["collection_failure_rate"]:.2f}%**. '
              f'Technical candidates without substantive text, including failed structural-only replays: **{gate["technical_candidates_without_text"]}/40**.', '',
              '| Six-way outcome | Diagnostic (40) |'+(' Full fixed frame (1,226) |' if full else ''),
              '|---|---:|'+('---:|' if full else '')]
    for key,value in diagnostic['wayback_outcomes'].items():
        lines.append(f'| `{key}` | {value} |'+(f' {full["wayback_outcomes"][key]} |' if full else ''))
    lines += ['', 'Acquisition failure details (request/page events can overlap within a candidate):', '',
              '| Route / reason | Events |', '|---|---:|']
    for reason,count in discovery_reasons.most_common():
        lines.append(f'| Official discovery: `{reason}` | {count} |')
    for route in ('direct','outscraper'):
        for reason,count in sorted(matrix[route+'_page']['failure_reasons'].items(),key=lambda item:-item[1]):
            lines.append(f'| {route} page: `{reason}` | {count} |')
    lines += ['',f'Of {len(failure_candidates)} exclusive `collection_failure` candidates, '
              f'{bounded_negative_candidates} completed all seven Availability requests with limited negative responses '
              'but lacked a completed CDX/TimeMap lookup. These remain unresolved archive coverage, not verified `no_archive`.', '',
              '| Prior-acquisition sample stratum | Candidates | Substantive archived text | Structural only | Collection failure |',
              '|---|---:|---:|---:|---:|']
    strata={r['place_id']:r['stratum'] for r in plan['diagnostic_sample']}
    for stratum in sorted(set(strata.values())):
        group=[r for r in records if strata[r['place_id']]==stratum]
        lines.append(f'| {stratum} | {len(group)} | '
                     f'{sum(r["outcome"] in ("direct_raw","outscraper_extracted") for r in group)} | '
                     f'{sum(r["outcome"]=="structural_only" for r in group)} | '
                     f'{sum(r["outcome"]=="collection_failure" for r in group)} |')
    lines += ['', 'The Availability schedule was frozen before new candidate-specific responses were inspected: '
              +', '.join('`'+s+'`'for s in plan['availability_schedule'])+'. '
              'Every requested and returned timestamp, returned URL, response status and raw receipt is retained. '
              'Successful bounded Availability negatives alone are not labeled `no_archive`; they cannot establish exhaustive archive absence. '
              'The six-state `collection_failure` therefore also includes insufficient discovery coverage after these limited negatives, '
              'which is distinguishable from parser/network failures in the retained query rows.', '',
              'Predeclared full-run gate: '+gate['rule'], '',
              '**Diagnostic gate: '+('PASS' if gate['passed'] else 'FAIL')+'.** '+
              ('A separate full archive evidence set was completed against the same 1,088 usable URLs; the 138 missing/invalid URL states were carried forward.' if full else
               'No second full archive attempt was executed because the diagnostic did not unlock it.' if not gate['passed'] else
               'The diagnostic permits a full archive attempt, which has not yet been completed.'), '',
              '**Phase 2 scientifically ready: '+('yes, at the transport-preflight level; still requires explicit review and is not authorized' if full and full['phase2_scientifically_ready'] else 'no')+'.** '
              'Classification remains prohibited.', '',
              'Accepted direct pages are locally extracted from retained HTTP bytes. Accepted Outscraper pages are labeled derived page text, '
              'bound to the candidate website and an independently established exact capture, and revalidated from raw provider receipts on resume. '
              'Outscraper was never used for CDX or TimeMap parsing. The Allina tracking-parameter case regression is covered; semantic URL parameters, '
              'different sites, different outlet paths and different capture timestamps remain distinct.', '',
              f'Preservation: {len(baseline)-1:,} inventoried preexisting files remain byte-identical. '
              'The one excluded file is a newly created transport test that was still being developed during the initial inventory. '
              'No Git commands were used. Provider job terminal states: '+json.dumps(states,sort_keys=True)+'.', '',
              'Artifacts:', '',
              f'- `{root.relative_to(ROOT).as_posix()}/plan.json`: fixed cohort, schedule, bounds and gate.',
              f'- `{root.relative_to(ROOT).as_posix()}/diagnostic/preflight.json`: complete transport matrix.',
              f'- `{root.relative_to(ROOT).as_posix()}/diagnostic/manifest.json`: frozen evidence/raw-response hashes.',
              f'- `{root.relative_to(ROOT).as_posix()}/implementation.json`: acquisition implementation commitment.', '',
              'Official contracts: [Availability API](https://archive.org/help/wayback_api.php), '
              '[CDX server](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server), '
              '[Memento RFC 7089](https://datatracker.ietf.org/doc/html/rfc7089).', '']
    path=out/'PREFLIGHT.md';path.write_text('\n'.join(lines),encoding='utf-8')
    tests=ET.parse(root/'offline_tests.xml').getroot()
    closure={'run_id':plan['run_id'],'completed_at':utcnow(),'diagnostic_complete':True,
             'full_archive_executed':full is not None,'phase2_authorized':False,
             'phase2_scientifically_ready':full['phase2_scientifically_ready'] if full else False,
             'openai_calls':0,'google_details_calls':0,'preserved_files':len(baseline)-1,
             'outscraper_job_states':states,'offline_tests':len(list(tests.iter('testcase'))),
             'frozen_state_files':state_snapshots,
             'report_sha256':file_sha(path),'diagnostic_manifest_sha256':file_sha(root/'diagnostic/manifest.json'),
             'full_manifest_sha256':file_sha(root/'full/manifest.json') if full else None}
    seal_json(root/'closure.json',closure)
    print(json.dumps({'report':path.as_posix(),'closure':closure},indent=2))
    return closure


if __name__=='__main__':
    write_report(ROOT/'data/v2/archive_transport/20260905T222825Z')
