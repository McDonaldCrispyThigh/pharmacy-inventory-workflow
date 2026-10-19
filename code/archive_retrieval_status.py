"""Freeze completed available routes without claiming an untested proxy result."""
from pathlib import Path
from collections import Counter
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from corrected_run_store import read_sealed,seal_json,file_sha,utcnow
from run_archive_retrieval_experiment import report,DEFAULT
from corrected_preflight import _backup_sqlite_state


def freeze_available(root):
    summary=report(root)
    plan=read_sealed(root/'plan.json')
    routes=[name for name,row in summary['routes'].items() if row['tested']==31]
    if set(routes)!={'direct','outscraper'}:
        raise ValueError('this status freeze is only for the completed direct/Outscraper stage')
    if summary['routes']['proxy']['tested']:
        raise ValueError('proxy results require the full comparison report')
    baseline=read_sealed(root/'preserved_before.json')['files']
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or file_sha(ROOT/p)!=h]
    if changed:raise ValueError('approved artifact preservation failed: '+repr(changed))
    commitment=read_sealed(root/'implementation.json')
    if any(file_sha(ROOT/p)!=h for p,h in commitment['files'].items()):
        raise ValueError('live implementation changed')
    database=root/'private/provider_state/state.sqlite3'
    connection=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        if connection.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('state integrity failed')
        jobs=[json.loads(r[0]) for r in connection.execute('SELECT payload FROM state WHERE namespace=?',('outscraper_jobs',))]
    finally:connection.close()
    if any(j['kind']!='replay' or j['state'] not in {'complete','failed'} for j in jobs):
        raise ValueError('unexpected or unresolved provider job')
    frozen_state=root/'private/direct_outscraper_state_frozen.sqlite3'
    _backup_sqlite_state(database,frozen_state)
    direct=summary['routes']['direct']['success'];outscraper=summary['routes']['outscraper']['success']
    substantive=summary['combined_substantive_observed'];pending=31-substantive
    test_tree=ET.parse(root/'offline_tests.xml').getroot()
    failures=len(list(test_tree.iter('failure')))+len(list(test_tree.iter('error')))
    if failures:raise ValueError('offline checks failed')
    tests=len(list(test_tree.iter('testcase')))
    maintenance=ROOT/'analysis/maintenance'/('archive_retrieval_'+plan['run_id'])
    maintenance.mkdir(parents=True,exist_ok=True)
    path=maintenance/'AVAILABLE_ROUTES_PREFLIGHT.md'
    lines=['# Fixed-snapshot retrieval experiment: available routes completed','',
           '**Status: incomplete three-route diagnostic; raw proxy is not configured or tested. No promotion.**','',
           'The approved 40-candidate baseline is unchanged. This experiment reuses its exact 40 candidate IDs and all 31 selected '
           'Availability-confirmed snapshot identities (31 distinct replay URLs). The frozen seven-timestamp Availability responses '
           'are reused with requested/returned times, URLs, statuses and hashes. No new snapshot was selected. '
           'No CDX, TimeMap, Google Details or OpenAI request was made.','',
           '| Page route | Newly tested snapshots | Usable page content | Success rate |',
           '|---|---:|---:|---:|',
           f'| Local direct raw | 31 | {direct} | {direct/31:.1%} |',
           '| Raw proxy | 0 | Not measured | Not measured |',
           f'| Outscraper derived text | 31 | {outscraper} | {outscraper/31:.1%} |','',
           f'The completed two routes supplied substantive page text for {substantive}/31 ({substantive/31:.1%}). '
           f'The remaining {pending}/31 have confirmed snapshot identities but no usable text from these two routes. '
           'They remain provisional structural-only observations while the proxy route is untested. '
           'Among these 31, unresolved snapshot-identity collection failures are 0; this does not mean page transport is working. '
           'The other nine members of the original 40 remain the unchanged baseline collection_failure observations; '
           'they were not sent to speculative page retrieval.','',
           'Proxy-and-Outscraper overlap, proxy-only, Outscraper-only and neither are **not measurable yet**. '
           'The final three-route outcome distribution and combined retrieval rate are withheld until all 31 proxy attempts complete.','',
           '| Route | Failure reason | Count |','|---|---|---:|']
    for route in ('direct','outscraper'):
        for reason,count in summary['routes'][route]['failures'].items():
            lines.append(f'| {route} | `{reason}` | {count} |')
    lines += ['', 'Accepted sources were revalidated from retained receipts and independently confirmed official snapshot identities. '
              'Outscraper text remains derived extraction; capture verification does not independently verify its wording. '
              'Snapshot existence does not establish opening year or outlet-level active-by status. Existing historical rules remain unchanged.','',
              f'Validation: {tests} offline tests passed; {len(baseline):,} inventoried approved files remain byte-identical. '
              'All Outscraper jobs are terminal: '+json.dumps(dict(Counter(j['state'] for j in jobs)))+'.','',
              '## Configuration needed to finish the authorized experiment','',
              'Supply an existing HTTP(S) CONNECT proxy in `WAYBACK_RAW_PROXY_URL`; optionally set '
              '`WAYBACK_RAW_PROXY_PROVIDER` to a short provider label. Values belong in the local environment or `.env`; '
              'do not place credentials in reports or messages. The adapter requires verified TLS and disables ambient proxy settings, '
              'automatic redirects and automatic content decoding. Each response body and redirect hop is preserved before deterministic local parsing. '
              'It does not accept an AI extraction API as a raw proxy.','',
              'After configuration, resume only the missing route:','',
              '```powershell',
              r'& .\AIAGENTNETS\Scripts\python.exe -B -X utf8 code/run_archive_retrieval_experiment.py run --routes proxy',
              '```','',
              'This command has no full-acquisition or classifier mode. A completed three-route result still requires review; '
              'the 1,088-website full acquisition and Phase 2 remain unauthorized.','',
              'Transport contracts: [Requests raw response bytes](https://requests.readthedocs.io/en/latest/user/quickstart/#raw-response-content); '
              '[Outscraper AI extraction](https://docs.outscraper.com/endpoints/universal-scraper/).','']
    if path.exists():raise ValueError('available-routes report already frozen')
    path.write_text('\n'.join(lines),encoding='utf-8')
    files={p.relative_to(root).as_posix():file_sha(p) for parent in (root/'results',root/'raw',root/'private/provider_state/responses')
           for p in parent.rglob('*.json')}
    files[frozen_state.relative_to(root).as_posix()]=file_sha(frozen_state)
    snapshot={'created_at':utcnow(),'status':'awaiting_raw_proxy_configuration','completed_routes':routes,
              'summary':summary,'plan_sha256':file_sha(root/'plan.json'),'files':files,
              'report_path':path.relative_to(ROOT).as_posix(),'report_sha256':file_sha(path),
              'preserved_files':len(baseline),'offline_tests':tests,
              'full_acquisition_authorized':False,'phase2_authorized':False}
    seal_json(root/'direct_outscraper_preflight.json',snapshot)
    print(json.dumps({'report':path.as_posix(),'status':snapshot['status'],'preserved_files':len(baseline)},indent=2))


if __name__=='__main__':freeze_available(DEFAULT)
