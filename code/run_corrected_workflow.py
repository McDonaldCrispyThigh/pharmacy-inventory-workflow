"""Phase-separated corrected run; the frozen Sep2 frame has no search route."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from corrected_run_store import acquire_details, atomic_json, file_sha, read_sealed, seal_json, sha, unit_key, utcnow, validate_frame_ids, run_resumable_queue


def bounded_parallel_map(items, worker, *, workers=8, on_result=None):
    """Admit at most one bounded worker set; stop admission on any fatal result."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be 1..8')
    source=iter(items)
    pool=ThreadPoolExecutor(max_workers=workers)
    pending=set()
    results=[]
    def admit():
        try:
            item=next(source)
        except StopIteration:
            return
        pending.add(pool.submit(worker,item))
    try:
        for _ in range(workers):
            admit()
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            # Inspect every completed member before admitting replacements.
            values=[future.result() for future in done]
            for value in values:
                if on_result:
                    on_result(value)
                results.append(value)
            for _ in values:
                admit()
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


def verify_classification_manifest(root: Path):
    """Verify decisions against saved raw receipts and the current frozen packet."""
    from corrected_preflight import verify_frozen_acquisition
    from corrected_evidence import restore_packet, MODEL
    from corrected_classification import _validate_received, _completed
    from run_forced_choice_workflow import _location_record
    from v2_downstream_models import candidate_location_from_details
    from collections import Counter
    evidence=verify_frozen_acquisition(root)
    scope=read_sealed(root/'run_scope.json')
    manifest=read_sealed(root/'classification_manifest.json')
    for field,relative in (('evidence_manifest_sha256','evidence_manifest.json'),('run_scope_sha256','run_scope.json'),('implementation_sha256','classification_implementation.json'),('model_availability_sha256','model_availability.json')):
        if manifest.get(field)!=file_sha(root/relative):
            raise ValueError('classification manifest binding changed: '+field)
    if manifest.get('run_id')!=scope['run_id'] or manifest.get('candidate_count')!=1226 or manifest.get('requested_model')!=MODEL:
        raise ValueError('classification manifest scope differs')
    implementation=read_sealed(root/'classification_implementation.json')
    for relative,digest in implementation.items():
        if file_sha(ROOT/relative)!=digest:
            raise ValueError('frozen classifier implementation differs')
    expected={f'private/classifications/{unit_key(entry["place_id"])}.json' for entry in scope['frame']['places']}
    if set(manifest['completed_files'])!=expected:
        raise ValueError('classification manifest must contain exactly 1226 expected files')
    for relative,digest in manifest['attempt_files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to((root/'private/classification_attempts').resolve()) or file_sha(path)!=digest:
            raise ValueError('retained classification attempt changed')
    outcomes=[]
    for relative,digest in manifest['completed_files'].items():
        path=root/relative
        if file_sha(path)!=digest:
            raise ValueError('completed classification file changed')
        value=read_sealed(path)
        record=read_sealed(root/'private/evidence_packets'/path.name)
        packet=restore_packet(record)
        if any(value.get(key)!=record[key] for key in ('candidate_id','place_id','packet_sha256','messages_sha256','model')):
            raise ValueError('classification is not bound to its current frozen evidence')
        if value['status']=='completed':
            response_path=(root/'private'/value['received_response_path']).resolve()
            if not response_path.is_relative_to((root/'private/classification_attempts').resolve()) or file_sha(response_path)!=value['received_response_file_sha256']:
                raise ValueError('successful model raw response changed')
            event=read_sealed(response_path)
            decision=_validate_received(event,packet,record,value['attempt_count'])
            location=_location_record(candidate_location_from_details(record['google_details'],packet.candidate.query_provenance,packet.candidate.retrieved_at))
            rebuilt=_completed(record,location,event,decision,response_path,(root/'private').resolve(),value['completed_at'])
            if rebuilt!=value:
                raise ValueError('completed decision differs from raw response, validator, or reducer')
        outcomes.append(value)
    statuses=dict(Counter(v['status'] for v in outcomes))
    models=sorted({v['returned_model'] for v in outcomes if v.get('returned_model')})
    unresolved=sorted(v['place_id'] for v in outcomes if v['status']!='completed')
    passed=statuses=={'completed':1226} and models==[MODEL]
    if manifest['statuses']!=statuses or manifest['returned_models']!=models or manifest['technical_unresolved_place_ids']!=unresolved or manifest['passed']!=passed:
        raise ValueError('classification run-level reconciliation differs')
    return manifest


def initialize(run_id: str, maintenance: Path) -> Path:
    from run_forced_choice_workflow import authenticate_fixed_frame
    import re
    if not re.fullmatch(r"\d{8}T\d{6}Z", run_id):
        raise ValueError("invalid corrected run ID")
    root = ROOT / "data" / "v2" / run_id
    if root.exists():
        read_sealed(root / "run_scope.json")
        return root
    frame = authenticate_fixed_frame()
    old = ROOT / "data" / "v2" / "20260902T012636Z"
    manifest = json.loads((old / "run_manifest.json").read_text(encoding="utf-8"))
    old_ids = []
    old_hashes = {}
    for path in sorted((old / "private" / "completed_candidates").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        old_ids.append(value["place_id"])
        old_hashes[path.name] = file_sha(path)
    if old_hashes != manifest.get("completed_unit_hashes"):
        raise ValueError("Sep2 completed unit hashes differ from its manifest")
    validate_frame_ids([p["place_id"] for p in frame["places"]], old_ids)
    # Current frame is derived from Aug31, and independently reconciled to all
    # Sep2 completed IDs and its recorded frame commitment.
    if manifest.get("fixed_candidate_frame_sha256", manifest.get("fixed_frame_sha256")) != frame["fixed_frame_sha256"]:
        raise ValueError("Sep2 frame commitment differs")
    scope = {"schema": "nets-ai-corrected-run-scope-v1", "run_id": run_id, "created_at": utcnow(), "candidate_count": 1226, "source_run_id": "20260902T012636Z", "source_manifest_sha256": file_sha(old / "run_manifest.json"), "source_completed_hashes": old_hashes, "frame": frame, "google_text_search_calls": 0, "requested_model": "gpt-5-mini-2025-08-07", "authorization_sha256": file_sha(maintenance / "user_authorization.txt"), "maintenance_directory": maintenance.relative_to(ROOT).as_posix(), "execution_order": ["acquisition_and_freeze", "classification", "linkage", "spatial", "impact", "thesis"], "old_run_preservation": "preserved; superseded designation conditional on corrected delivery gates"}
    seal_json(root / "run_scope.json", scope)
    atomic_json(root / "run_status.json", {"status": "initialized", "run_id": run_id, "updated_at": utcnow()})
    return root


def details_phase(root: Path, *, workers: int = 4, limit: int | None = None) -> dict:
    from dotenv import dotenv_values
    from run_v2_workflow import _GoogleHTTPTransport
    import os
    scope = read_sealed(root / "run_scope.json")
    frame = scope["frame"]["places"]
    config = dotenv_values(ROOT / ".env", interpolate=False)
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or config.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("Google Places credential unavailable")
    selected = frame[:limit] if limit is not None else frame
    private = root / "private"
    count = 0
    def collect(entry):
        return acquire_details(private, entry, key, _GoogleHTTPTransport())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect, entry) for entry in selected]
        for future in as_completed(futures):
            value = future.result()
            count += 1
            if count % 25 == 0 or count == len(selected) or value["status"] != "success":
                print(json.dumps({"phase": "details", "processed_this_invocation": count, "selected": len(selected), "latest_status": value["status"]}), flush=True)
    successes = list((private / "details").glob("*.json"))
    for p in successes:
        read_sealed(p)
    report = {"phase": "details", "candidate_count": len(frame), "details_success": len(successes), "details_remaining_or_failed": len(frame) - len(successes), "google_text_search_calls": 0, "updated_at": utcnow()}
    atomic_json(root / "details_progress.json", report)
    return report


def archive_phase(root: Path, *, max_active: int = 24, limit: int | None = None) -> dict:
    from datetime import datetime
    from dataclasses import asdict
    from collections import Counter
    import time
    from corrected_evidence import make_packet, restore_packet
    from corrected_archive_verification import verify_collection
    from outscraper_wayback import OutscraperFallback, load_api_key
    from skills.wayback_agent import WaybackCollectorConfig, WaybackEvidenceCollector
    from run_v2_workflow import _WaybackArchiveTransport
    from wayback_state import WaybackStateStore
    from v2_classifier import build_candidate_facts
    from v2_evidence_models import QueryProvenanceV2
    scope = read_sealed(root / "run_scope.json")
    frame = scope["frame"]["places"]
    private = root / "private"
    details = {}
    for entry in frame:
        record = read_sealed(private / "details" / (unit_key(entry["place_id"]) + ".json"))
        if record["place_id"] != entry["place_id"] or record["frame_entry"] != entry:
            raise ValueError("Details frame authentication failed")
        details[entry["place_id"]] = record
    state_root = ROOT / "data" / "v2" / "private" / "corrected_wayback_state"
    store = WaybackStateStore(state_root)
    config = WaybackCollectorConfig()
    acquisition_config = {"collector": asdict(config), "outscraper_max_jobs": 50000, "outscraper_raw_body_budget": 10000000000, "scope": scope["run_id"], "shared_state_directory": state_root.relative_to(ROOT).as_posix(), "reason_for_operational_budget": "1226 fixed candidates, up to 500 CDX records and three candidate captures per year; original scientific selection bounds retained"}
    seal_json(root / "archive_configuration.json", acquisition_config)
    key = load_api_key(ROOT / ".env")
    fallback = OutscraperFallback(store, key, scope=scope["run_id"], max_jobs=50000, max_total_response_bytes=10000000000) if key else None
    collector = WaybackEvidenceCollector(_WaybackArchiveTransport(), private / "wayback_cache", state_store=store, fallback=fallback, config=config)
    selected = frame[:limit] if limit is not None else frame
    last_report = [0.0]
    counts = Counter()
    def collect(entry):
        ident = entry["place_id"]
        packet_path = private / "evidence_packets" / (unit_key(ident) + ".json")
        if packet_path.exists():
            record = read_sealed(packet_path)
            restore_packet(record)
            if record["frame_entry"] != entry or record["fixed_frame_sha256"] != scope["frame"]["fixed_frame_sha256"]:
                raise ValueError("packet frame commitment differs")
            return record["archive_summary"]
        source = details[ident]
        retrieved = datetime.fromisoformat(source["retrieved_at"].replace("Z", "+00:00"))
        provenance = QueryProvenanceV2(entry["query_definition_id"], entry["source_zip"], "places_search")
        candidate = build_candidate_facts(source["details"], provenance, retrieved)
        collection = collector.collect(candidate)
        verifications = verify_collection(collection, store, scope=scope["run_id"])
        for verification in verifications:
            if verification.get("provenance_sha256"):
                verification["response"] = store.read_record(verification["provenance_sha256"])
        packet = make_packet(source["details"], provenance, retrieved, collection,
                             frame_entry=entry, fixed_frame_sha256=scope["frame"]["fixed_frame_sha256"],
                             model=scope["requested_model"], archive_verifications=verifications)
        restore_packet(packet)
        seal_json(packet_path, packet)
        counts[packet["archive_summary"].get("outcome", packet["archive_summary"].get("state", "unknown"))] += 1
        return packet["archive_summary"]
    def event(entry, value):
        atomic_json(private / "archive_queue" / (unit_key(entry["place_id"]) + ".json"), {"place_id": entry["place_id"], "updated_at": utcnow(), **value})
        if time.time() - last_report[0] >= 25 or value["status"] == "complete":
            last_report[0] = time.time()
            progress = {"phase": "archive", "latest": value, "packets_frozen": len(list((private / "evidence_packets").glob("*.json"))), "candidate_count":1226,"updated_at":utcnow()}
            atomic_json(root / "archive_progress.json", progress)
            print(json.dumps(progress), flush=True)
    results = run_resumable_queue(selected, collect, max_active=max_active, on_event=event)
    report = {"phase":"archive", "processed":len(results), "packets_frozen":len(list((private / "evidence_packets").glob("*.json"))), "new_outcomes":dict(counts),"updated_at":utcnow()}
    atomic_json(root / "archive_progress.json", report)
    return report


def classification_phase(root: Path, *, workers: int = 8):
    from corrected_preflight import verify_frozen_acquisition, check_promotion_hold, require_classification_promotion
    check_promotion_hold(root)
    from corrected_classification import classify_frozen_candidate
    from corrected_evidence import restore_packet
    from dotenv import dotenv_values
    from openai import OpenAI
    from collections import Counter
    import os
    if (root/'classification_manifest.json').exists():
        return verify_classification_manifest(root)
    manifest = verify_frozen_acquisition(root)
    require_classification_promotion(root)
    scope = read_sealed(root / 'run_scope.json')
    availability = read_sealed(root / 'model_availability.json')
    if availability.get('status') != 'available' or availability.get('returned_id') != scope['requested_model']:
        raise ValueError('requested model availability not verified')
    config = dotenv_values(ROOT / '.env', interpolate=False)
    key = os.environ.get('OPENAI_API_KEY') or config.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OpenAI credential unavailable')
    # Exact frozen messages, schema and typed packet reconstruction are verified
    # for the complete frame before the first request can enter the thread pool.
    records = [read_sealed(root / relative) for relative in manifest['packet_files']]
    for record in records:
        restore_packet(record)
    implementation = {relative: file_sha(ROOT / relative) for relative in (
        'code/corrected_classification.py','code/corrected_evidence.py',
        'code/corrected_preflight.py','code/corrected_run_store.py',
        'code/run_corrected_workflow.py','code/run_forced_choice_workflow.py',
        'code/forced_choice_models.py','code/forced_choice_classifier.py',
        'code/forced_choice_eligibility.py','code/v2_evidence_models.py',
        'code/v2_classifier.py','code/v2_downstream_models.py')}
    seal_json(root/'classification_implementation.json', implementation)
    client = OpenAI(api_key=key, max_retries=0, timeout=180)
    def classify(record):
        return classify_frozen_candidate(record, root/'private', client)
    accounted=[]
    def progress(value):
        accounted.append(value)
        if len(accounted)%10==0 or value['status']!='completed' or len(accounted)==1226:
            status={'phase':'classification','accounted':len(accounted),'total':1226,'statuses':dict(Counter(item['status'] for item in accounted)),'updated_at':utcnow()}
            atomic_json(root/'classification_progress.json',status)
            print(json.dumps(status),flush=True)
    outcomes=bounded_parallel_map(records,classify,workers=workers,on_result=progress)
    validate_frame_ids([v['place_id'] for v in outcomes], [entry['place_id'] for entry in scope['frame']['places']])
    verify_frozen_acquisition(root)
    final = {'run_id':scope['run_id'],'candidate_count':1226,'requested_model':scope['requested_model'],'statuses':dict(Counter(v['status'] for v in outcomes)), 'returned_models':sorted({v['returned_model'] for v in outcomes if v.get('returned_model')}),'technical_unresolved_place_ids':sorted(v['place_id'] for v in outcomes if v['status']!='completed'),'completed_files':{p.relative_to(root).as_posix():file_sha(p) for p in (root/'private/classifications').glob('*.json')},'attempt_files':{p.relative_to(root).as_posix():file_sha(p) for p in (root/'private/classification_attempts').rglob('*.json')},'finished_at':utcnow(),'evidence_manifest_sha256':file_sha(root/'evidence_manifest.json'),'run_scope_sha256':file_sha(root/'run_scope.json'),'implementation_sha256':file_sha(root/'classification_implementation.json'),'model_availability_sha256':file_sha(root/'model_availability.json')}
    final['passed'] = all(v['status']=='completed' for v in outcomes) and final['returned_models']==[scope['requested_model']]
    seal_json(root/'classification_manifest.json', final)
    return verify_classification_manifest(root)


def downstream_phase(root: Path, phase: int):
    from corrected_preflight import verify_frozen_acquisition
    from corrected_downstream import run_linkage_phase, run_spatial_phase, implementation_hashes
    from forced_choice_delivery import ForcedChoiceDeliveryCandidate, load_forced_choice_local_context
    from forced_choice_models import ForcedChoiceDecision
    from run_forced_choice_workflow import _location_from_record
    from study_design import TARGET_ZIPS
    evidence = verify_frozen_acquisition(root)
    classification = verify_classification_manifest(root)
    if classification.get('passed') is not True or classification['candidate_count']!=1226:
        raise ValueError('complete classification gate not passed; report technical unresolved before promotion')
    candidates=[]
    for relative,digest in classification['completed_files'].items():
        path=root/relative
        if file_sha(path)!=digest:
            raise ValueError('completed classification hash mismatch')
        value=read_sealed(path)
        candidates.append(ForcedChoiceDeliveryCandidate(_location_from_record(value['location']),ForcedChoiceDecision.from_public_dict(value['decision'])))
    scope=read_sealed(root/'run_scope.json')
    validate_frame_ids([item.location.place_id for item in candidates],[entry['place_id'] for entry in scope['frame']['places']])
    context=load_forced_choice_local_context(ROOT/'analysis/input_manifest.json', TARGET_ZIPS)
    from v2_spatial_adapter import SpatialContractV2, _origin_sha256
    spatial_contract=SpatialContractV2()
    expected_origin=json.loads((ROOT/'data/v2/20260902T012636Z/delivery/spatial_summary.json').read_text(encoding='utf-8'))['origin_sha256']
    if _origin_sha256(context.origins.to_crs(spatial_contract.distance_crs).copy(),spatial_contract.origin_id_column)!=expected_origin:
        raise ValueError('fixed tract origins differ from September 2 before downstream calculation')
    hashes=implementation_hashes()
    seal_json(root/'downstream_implementation.json',hashes)
    output=ROOT/'analysis/v2'/('corrected_'+scope['run_id'])/('phase3_linkage' if phase==3 else 'phase4_spatial')
    metadata={'run_id':scope['run_id'],'model':scope['requested_model'],'returned_model':scope['requested_model'],'candidate_frame_sha256':evidence['fixed_frame_sha256'],'evidence_manifest_sha256':file_sha(root/'evidence_manifest.json'),'classification_manifest_sha256':file_sha(root/'classification_manifest.json')}
    from forced_choice_delivery import _canonical_sha256
    lineage=_canonical_sha256([item.lineage_record() for item in sorted(candidates,key=lambda item:item.location.place_id)])
    def check_package(directory,which):
        saved=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
        expected_status='linkage_phase_complete' if which==3 else 'spatial_phase_complete'
        if saved.get('phase')!=which or saved.get('status')!=expected_status or saved.get('candidate_count')!=1226 or saved.get('run_metadata')!=metadata or saved.get('candidate_lineage_sha256')!=lineage:
            raise ValueError('downstream package phase or run binding differs')
        if not saved.get('artifacts'):
            raise ValueError('downstream package contains no artifacts')
        for relative,digest in saved['artifacts'].items():
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory.resolve()) or file_sha(path)!=digest:
                raise ValueError('downstream output changed')
        semantic={k:v for k,v in saved.items() if k not in ('run_metadata','semantic_manifest_sha256')}
        if _canonical_sha256(semantic)!=saved['semantic_manifest_sha256']:
            raise ValueError('downstream semantic manifest mismatch')
        for relative,digest in hashes.items():
            if saved['source_hashes'].get(relative)!=digest:
                raise ValueError('downstream implementation binding differs')
        for relative,digest in context.source_hashes.items():
            if saved['source_hashes'].get(relative)!=digest:
                raise ValueError('downstream local input binding differs')
        for relative,digest in saved['source_hashes'].items():
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT) or file_sha(path)!=digest:
                raise ValueError('downstream saved source file changed')
        if which==4 and saved.get('origin_sha256')!=expected_origin:
            raise ValueError('fixed tract origins differ from September 2')
        return saved
    if phase==4:
        check_package(output.parent/'phase3_linkage',3)
    if output.exists():
        return check_package(output,phase)
    function=run_linkage_phase if phase==3 else run_spatial_phase
    result=function(candidates,context=context,target_zips=TARGET_ZIPS,run_metadata=metadata,source_hashes=hashes,output_root=output)
    return check_package(output,phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["initialize", "details", "archive", "freeze", "classify", "linkage", "spatial", "impact", "thesis"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--maintenance", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-active", type=int, default=24)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'\d{8}T\d{6}Z',args.run_id):
        parser.error('run-id must be a UTC timestamp')
    root = ROOT / "data" / "v2" / args.run_id
    if args.phase == "initialize":
        if args.maintenance is None:
            parser.error("--maintenance is required for initialization")
        root = initialize(args.run_id, args.maintenance.resolve())
        print(json.dumps({"status": "initialized", "run_id": args.run_id, "candidate_count": 1226, "directory": str(root)}))
    elif args.phase == "details":
        if not 1 <= args.workers <= 8:
            parser.error("workers must be 1..8")
        print(json.dumps(details_phase(root, workers=args.workers, limit=args.limit)))
    elif args.phase == "archive":
        print(json.dumps(archive_phase(root, max_active=args.max_active, limit=args.limit)))
    elif args.phase == "freeze":
        from corrected_preflight import freeze_acquisition
        print(json.dumps(freeze_acquisition(root)))
    elif args.phase == "classify":
        print(json.dumps(classification_phase(root, workers=args.workers)))
    elif args.phase in ('linkage','spatial'):
        print(json.dumps(downstream_phase(root,3 if args.phase=='linkage' else 4)))
    elif args.phase=='impact':
        from corrected_impact import compare_runs
        verify_classification_manifest(root)
        downstream_phase(root,4)
        output=ROOT/'analysis/v2'/('corrected_'+args.run_id)
        print(json.dumps(compare_runs(root,output/'phase3_linkage',output/'phase4_spatial',output_root=output/'phase5_impact')))
    elif args.phase=='thesis':
        from corrected_thesis import deliver_thesis
        print(json.dumps(deliver_thesis(root)))


if __name__ == "__main__":
    main()
