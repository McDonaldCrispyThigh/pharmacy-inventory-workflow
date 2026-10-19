"""Freeze original, supplemental and expanded RQ1 using the existing flow auditor."""
import run_expanded_final as run

def main():
    candidates,_=run.load_candidates();summaries={}
    old_ids={r['place_id'] for r in run.read_json(run.SUP/'old_final_index.json')}
    for cohort,items in [('original',tuple(c for c in candidates if c.location.place_id in old_ids)),
                         ('supplemental',tuple(c for c in candidates if c.location.place_id not in old_ids)),('combined',candidates)]:
        dest=run.OUT/'rq1'/cohort;dest.mkdir(parents=True,exist_ok=True)
        labels=run.delivery._eligibility_frame(items)
        for name,frame in [('eligibility.csv',labels),('candidate_inventory.csv',run.delivery._candidate_inventory(items)),
                           ('unresolved_cases.csv',labels.loc[labels.eligibility_state.eq('unresolved')])]:
            run.downstream._write_csv(dest/name,frame)
        summary=run.audit.audit_candidate_flow(dest)
        summary['target_designated_excluded_by_other_dimension']=int((labels.target_construct_status.eq('target_pharmacy')&labels.eligibility_state.eq('exclude')).sum())
        run.save(dest/'summary.json',summary);summaries[cohort]=summary
    run.save(run.OUT/'rq1/summary.json',summaries)
    run.save(run.OUT/'rq1/freeze.json',{'runner_sha256':run.sha(run.Path(__file__)),
        'files':{p.relative_to(run.OUT/'rq1').as_posix():run.sha(p) for p in (run.OUT/'rq1').rglob('*') if p.is_file() and p.name!='freeze.json'}})
    run.write(run.OUT/'implementation/code/run_expanded_rq1.py',run.Path(__file__).read_bytes(),immutable=True)
    print('RQ1 frozen',summaries['combined']['eligibility'],flush=True)

if __name__=='__main__':
    run.socket.socket.connect=run.old.deny_network;run.socket.create_connection=run.old.deny_network
    main()
