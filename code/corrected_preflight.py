"""Fail-closed whole-frame evidence freeze before any corrected classification."""
from collections import Counter
from pathlib import Path
import re
import json
import shutil
import sqlite3
import os
import tempfile

from corrected_run_store import read_sealed, seal_json, atomic_json, unit_key, file_sha, utcnow, validate_frame_ids

OUTCOMES = ('direct_raw','outscraper_extracted','structural_only','no_archive','no_website','collection_failure')


def assess_coverage(substantive, websites, failures, *, failed_replays_without_text=0, provenance_failures=0):
    if not (0 <= substantive <= websites and 0 <= failures <= websites):
        raise ValueError('invalid website evidence denominator')
    # No scientific minimum coverage is imposed. Pending user review, any
    # unresolved acquisition failure prevents automatic promotion; this also
    # prevents a small number of successful sources from masking mass failure.
    if not 0 <= failed_replays_without_text <= websites:
        raise ValueError('invalid failed-replay denominator')
    if type(provenance_failures) is not int or provenance_failures < 0:
        raise ValueError('invalid source provenance failure count')
    return {'passed': failures == 0 and failed_replays_without_text == 0 and provenance_failures == 0,
            'reason': ('source_binding_or_derivation_failed' if provenance_failures else
                       'unresolved_collection_failures_require_repair_or_review' if failures else
                       'archive_replay_failures_without_substantive_text' if failed_replays_without_text else
                       'completed_queries_without_collection_failure'),
            'collection_failure_fraction': failures / websites if websites else None,
            'failed_replay_candidates_without_substantive_text': failed_replays_without_text,
            'substantive_coverage_fraction': substantive / websites if websites else None,
            'operational_alarm_only': 'Conservative automatic-promotion rule: unresolved collection failures require repair or explicit review. This is not a substantive archive-coverage minimum; completed negative queries may yield zero coverage.'}


def check_promotion_hold(root: Path):
    path = Path(root) / 'promotion_hold.json'
    if path.exists():
        hold = read_sealed(path)
        if hold.get('blocking') is not False or hold.get('phase2_authorized') is not True:
            raise ValueError('Phase 1 review hold: no OpenAI classification is authorized')


def require_classification_promotion(root: Path):
    """Rechecked immediately before each API attempt; missing approval fails closed.

    The CLI separately validates every packet and raw provenance commitment.
    A future explicit user instruction must be bound to this exact freeze before
    a promotion record is created. This function never creates that record.
    """
    root = Path(root)
    check_promotion_hold(root)
    path = root / 'phase2_promotion.json'
    if not path.exists():
        raise ValueError('explicit Phase 2 promotion is absent')
    promotion = read_sealed(path)
    manifest = read_sealed(root / 'evidence_manifest.json')
    report = json.loads((root / 'acquisition_preflight.json').read_text(encoding='utf-8'))
    scope = read_sealed(root / 'run_scope.json')
    if (promotion.get('phase2_authorized') is not True
            or promotion.get('run_id') != scope['run_id']
            or promotion.get('evidence_manifest_sha256') != file_sha(root / 'evidence_manifest.json')
            or promotion.get('preflight_sha256') != file_sha(root / 'acquisition_preflight.json')
            or manifest.get('preflight_sha256') != file_sha(root / 'acquisition_preflight.json')
            or report.get('freeze_verified') is not True
            or report.get('coverage_gate', {}).get('passed') is not True):
        raise ValueError('Phase 2 promotion or acquisition quality gate has not passed')
    return promotion


def _provenance_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'provenance_sha256' and item is not None:
                if not isinstance(item, str) or not re.fullmatch('[0-9a-f]{64}', item):
                    raise ValueError('invalid archive provenance reference')
                yield item
            else:
                yield from _provenance_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _provenance_refs(item)


def _sqlite_rows(path):
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if connection.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('archive state snapshot integrity failed')
        return connection.execute('SELECT namespace,key,payload,sha256 FROM state ORDER BY namespace,key').fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError('archive state snapshot integrity failed') from exc
    finally:
        connection.close()


def _backup_sqlite_state(database, target):
    expected = _sqlite_rows(database)
    if target.exists():
        if _sqlite_rows(target) != expected:
            raise ValueError('existing archive state snapshot is stale or incomplete')
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.state-backup-', suffix='.sqlite3', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        source_db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        target_db = sqlite3.connect(temporary)
        try:
            source_db.backup(target_db)
        finally:
            target_db.close()
            source_db.close()
        if _sqlite_rows(temporary) != expected:
            raise ValueError('archive state changed during snapshot')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_raw_archive_state(root, store, refs):
    """Commit run-local copies; later shared-cache changes cannot erase evidence."""
    destination = root / 'private/frozen_wayback_state'
    destination.mkdir(parents=True, exist_ok=True)
    # The corrected acquisition uses a separate new state directory. Keep all
    # currently retained responses, including unsuccessful/pending HTTP receipts
    # that older collection objects did not directly reference.
    refs.update(path.stem for path in (store.root / 'responses').glob('*/*.json'))
    for digest in sorted(refs):
        store.read_record(digest)
        source = store.root / 'responses' / digest[:2] / (digest + '.json')
        target = destination / 'responses' / digest[:2] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha(target) != digest:
                raise ValueError('frozen raw archive response differs')
        else:
            shutil.copy2(source, target)
    database = store.root / 'state.sqlite3'
    if database.exists():
        _backup_sqlite_state(database, destination / 'state.sqlite3')
    return {path.relative_to(root).as_posix(): file_sha(path)
            for path in sorted(destination.rglob('*')) if path.is_file()}


def freeze_acquisition(root: Path):
    from corrected_evidence import restore_packet
    from wayback_state import WaybackStateStore
    if (root / 'evidence_manifest.json').exists():
        verify_frozen_acquisition(root)
        return json.loads((root / 'acquisition_preflight.json').read_text(encoding='utf-8'))
    scope = read_sealed(root / 'run_scope.json')
    frame = scope['frame']['places']
    audit_path = root / 'phase1_failure_audit.json'
    if (root / 'promotion_hold.json').exists() and not audit_path.exists():
        raise ValueError('complete failure and provenance audit required before the review freeze')
    failure_audit = read_sealed(audit_path) if audit_path.exists() else None
    paths = list((root / 'private/evidence_packets').glob('*.json'))
    if len(paths) != 1226 or len(frame) != 1226:
        raise ValueError(f'all 1226 evidence packets required; found {len(paths)}')
    classification_requests=list((root/'private/classification_attempts').rglob('*.request.json'))
    if classification_requests:
        raise ValueError('classification was attempted before the complete evidence freeze')
    project = Path(__file__).resolve().parents[1]
    config = read_sealed(root / 'archive_configuration.json')
    store = WaybackStateStore(project / config['shared_state_directory'])
    by_id = {entry['place_id']: entry for entry in frame}
    ids = []
    counts = Counter({name: 0 for name in OUTCOMES})
    candidate_counts = Counter()
    source_counts = Counter()
    refs = set()
    packet_files = {}
    details_files = {}
    source_rows = []
    website_rows = Counter()
    failed_replays_without_text = 0
    for path in paths:
        value = read_sealed(path)
        packet = restore_packet(value)
        ident = packet.candidate.source_place_id
        ids.append(ident)
        if value['frame_entry'] != by_id.get(ident) or path.name != unit_key(ident) + '.json' or value['fixed_frame_sha256'] != scope['frame']['fixed_frame_sha256']:
            raise ValueError('frozen packet frame binding mismatch')
        detail_path = root / 'private/details' / path.name
        detail = read_sealed(detail_path)
        from corrected_evidence import sanitize_details
        if value['google_details'] != sanitize_details(detail['details']) or value['retrieved_at'] != detail['retrieved_at']:
            raise ValueError('frozen Details differs from actual received record')
        packet_files[path.relative_to(root).as_posix()] = file_sha(path)
        details_files[detail_path.relative_to(root).as_posix()] = file_sha(detail_path)
        summary = value['archive_summary']
        archive = packet.wayback_collection.archive_evidence
        failed_replays_without_text += int(not archive.source_snapshots and
                                           any(item.status != 'success' for item in archive.fetch_records))
        if summary['outcome'] not in OUTCOMES:
            raise ValueError('unknown Wayback outcome')
        website_rows['returned_nonempty_url'] += int(bool(str(detail['details'].get('websiteUri') or '').strip()))
        website_rows['missing_url'] += int(summary['query_status'] == 'missing_url')
        website_rows['invalid_url'] += int(summary['query_status'] == 'invalid_url')
        counts[summary['outcome']] += 1
        candidate_counts['substantive_archived_evidence'] += int(summary['substantive_source_count'] > 0)
        for field in ('outlet_specific_archive_evidence_count','broader_brand_archive_evidence_count','unclear_scope_archive_evidence_count','active_by_evidence_count','explicit_opening_year_evidence_count'):
            candidate_counts[field] += int(summary[field] > 0)
            source_counts[field] += summary[field]
        for source in summary['sources']:
            source_counts['substantive_sources'] += 1
            source_counts['official_snapshot_confirmed'] += int(source['independent_snapshot_check'].startswith('verified_official'))
        refs.update(_provenance_refs(value['wayback_observed']))
        refs.update(_provenance_refs(value['archive_verifications']))
        source_rows.append({'place_id':ident,'candidate_id':value['candidate_id'],'packet_sha256':value['packet_sha256'],**summary})
    validate_frame_ids(ids, list(by_id))
    if failure_audit is not None:
        if (failure_audit.get('run_id') != scope['run_id']
                or failure_audit.get('complete_frame') is not True
                or failure_audit.get('packet_count') != 1226):
            raise ValueError('failure audit does not cover the complete current frame')
        audit_inputs = failure_audit.get('input_commitments', {})
        if (audit_inputs.get('packet_files') != packet_files
                or audit_inputs.get('run_scope_sha256') != file_sha(root/'run_scope.json')
                or audit_inputs.get('archive_configuration_sha256') != file_sha(root/'archive_configuration.json')
                or failure_audit['summary'].get('wayback_outcomes') != dict(counts)):
            raise ValueError('failure audit input commitments or outcomes differ from this freeze')
        if failure_audit['summary'].get('unresolved_scoped_job_count') != 0:
            raise ValueError('unresolved fallback jobs remain; acquisition is not complete')
        refs.update(failure_audit.get('archive_provenance_sha256', []))
    frozen_state_files = _freeze_raw_archive_state(root, store, refs)
    if sum(counts.values()) != 1226:
        raise ValueError('Wayback outcomes are not an exhaustive mutually exclusive partition')
    coverage = assess_coverage(candidate_counts['substantive_archived_evidence'], website_rows['returned_nonempty_url'],
                               counts['collection_failure'], failed_replays_without_text=failed_replays_without_text,
                               provenance_failures=(failure_audit['summary']['source_url_binding_failures'] +
                                                    failure_audit['summary']['source_derivation_failures']) if failure_audit else 0)
    report = {'schema':'nets-ai-corrected-acquisition-preflight-v2','run_id':scope['run_id'],'candidate_count':1226,'details_success':1226,'details_failure':0,'website_candidates':website_rows['returned_nonempty_url'],'usable_website_candidates':1226-counts['no_website'],'website_status_counts':dict(website_rows),'wayback_outcomes':dict(counts),'candidate_evidence_counts':dict(candidate_counts),'source_evidence_counts':dict(source_counts),'archive_provenance_records_verified':len(refs),'board_absent_from_inputs':True,'full_packet_reconstruction_passed':True,'frame_equality_passed':True,'coverage_gate':coverage,'freeze_verified':True,'promotion_allowed':False,'promotion_reason':'explicit_phase1_review_required','passed':False,'google_text_search_calls':0,'classification_calls_before_freeze':0,'checked_at':utcnow()}
    if failure_audit is not None:
        report['failure_and_provenance_audit'] = failure_audit['summary']
    atomic_json(root / 'acquisition_preflight.json', report)
    # The whole acquisition must be frozen even when its quality blocks the
    # next phase. An integrity commitment is not permission to classify.
    existing = root / 'evidence_manifest.json'
    seal_json(root/'archive_candidate_summary.json', sorted(source_rows,key=lambda row:row['place_id']))
    payload = {'schema':'nets-ai-corrected-evidence-manifest-v2','run_id':scope['run_id'],'candidate_count':1226,'fixed_frame_sha256':scope['frame']['fixed_frame_sha256'],'requested_model':scope['requested_model'],'packet_files':dict(sorted(packet_files.items())),'details_files':dict(sorted(details_files.items())),'archive_provenance_sha256':sorted(refs),'shared_state_directory':config['shared_state_directory'],'frozen_state_directory':'private/frozen_wayback_state','frozen_state_files':frozen_state_files,'preflight_sha256':file_sha(root/'acquisition_preflight.json'),'run_scope_sha256':file_sha(root/'run_scope.json'),'archive_candidate_summary_sha256':file_sha(root/'archive_candidate_summary.json')}
    payload['supporting_files'] = {path.relative_to(root).as_posix(): file_sha(path)
        for path in (audit_path, root/'promotion_hold.json', root/'archive_configuration.json',
                     root/'details_acquisition_summary.json') if path.exists()}
    seal_json(existing, payload)
    return report


def verify_frozen_acquisition(root: Path):
    from wayback_state import WaybackStateStore
    from corrected_evidence import MODEL, restore_packet, sanitize_details
    manifest = read_sealed(root/'evidence_manifest.json')
    scope = read_sealed(root/'run_scope.json')
    preflight = json.loads((root/'acquisition_preflight.json').read_text(encoding='utf-8'))
    if manifest['candidate_count'] != 1226 or len(manifest['packet_files']) != 1226 or len(manifest['details_files']) != 1226:
        raise ValueError('frozen acquisition is incomplete')
    if (manifest['run_id'] != scope['run_id'] or manifest['fixed_frame_sha256'] != scope['frame']['fixed_frame_sha256']
            or manifest['requested_model'] != scope['requested_model'] or scope['requested_model'] != MODEL
            or preflight.get('freeze_verified') is not True or preflight.get('candidate_count') != 1226
            or preflight.get('run_id') != scope['run_id']):
        raise ValueError('acquisition scope, model, or verified freeze binding differs')
    frame={entry['place_id']:entry for entry in scope['frame']['places']}
    expected={f'private/evidence_packets/{unit_key(ident)}.json' for ident in frame}
    expected_details={f'private/details/{unit_key(ident)}.json' for ident in frame}
    if set(manifest['packet_files'])!=expected or set(manifest['details_files'])!=expected_details:
        raise ValueError('frozen acquisition paths differ from the exact frame')
    for section in ('packet_files','details_files','frozen_state_files','supporting_files'):
        for relative,digest in manifest[section].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha(path)!=digest:
                raise ValueError('frozen acquisition file hash mismatch')
    ids=[]
    for relative in manifest['packet_files']:
        record=read_sealed(root/relative)
        packet=restore_packet(record)
        ident=packet.candidate.source_place_id
        ids.append(ident)
        if record['model']!=MODEL or record['frame_entry']!=frame.get(ident) or record['fixed_frame_sha256']!=manifest['fixed_frame_sha256']:
            raise ValueError('current evidence packet is not bound to the frozen model and frame')
        detail=read_sealed(root/f'private/details/{unit_key(ident)}.json')
        if record['google_details']!=sanitize_details(detail['details']) or record['retrieved_at']!=detail['retrieved_at']:
            raise ValueError('packet no longer matches its received Details')
    validate_frame_ids(ids,list(frame))
    if file_sha(root/'acquisition_preflight.json')!=manifest['preflight_sha256']:
        raise ValueError('acquisition preflight commitment changed')
    if file_sha(root/'run_scope.json')!=manifest['run_scope_sha256'] or file_sha(root/'archive_candidate_summary.json')!=manifest['archive_candidate_summary_sha256']:
        raise ValueError('acquisition summary or scope commitment changed')
    store=WaybackStateStore(root/manifest['frozen_state_directory'])
    for digest in manifest['archive_provenance_sha256']:
        store.read_record(digest)
    return manifest
