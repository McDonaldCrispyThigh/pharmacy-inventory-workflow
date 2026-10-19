# Code execution boundary

This directory contains the versioned establishment-data workflow alongside older
project scripts retained for provenance. Pharmacy is the empirical test case. File
presence does not make a script canonical.

NETS cost, timing, discontinuation, and location-quality limits motivate the project;
the active workflow uses no NETS empirical input or comparator.

## Frozen v1 canonical downstream analysis

The authorized analysis sequence is:

```text
frozen Google Places Legacy metadata and limited returned reviews
  -> stored constrained LLM labels
  -> post-hoc Wayback summary fields (not model evidence; ignored downstream)
  -> exact-address-plus-ZIP Google location units
  -> uniform pre-linkage Board eligibility
  -> deterministic one-to-one linkage
  -> automatic exact-name recurrence
  -> paired projected-Euclidean potential proximity
  -> strict verification
```

The canonical code surface is:

| File | Role |
|---|---|
| `study_design.py` | Side-effect-free target frame, Board eligibility, normalization, location keys, and hashing rules |
| `geo_match.py` | Deterministic linkage scoring and threshold-aware assignment helpers |
| `board_google_comparison.py` | Authenticated input loading, declared analyses, isolated outputs, and provenance |
| `verify_board_google_outputs.py` | Read-only, fail-closed output authentication and invariant checks |

Run from the repository root using the exact commands in `analysis/README.md`.
`analysis/delivery/` must be new or empty. Generated files must remain below the
selected `analysis/` output root; current code must not write into thesis chapters or
historical figure directories.

## Canonical v2 workflow

The v2 implementation surface is:

| File | Role |
|---|---|
| `run_v2_workflow.py` | Fail-closed command-line entry point and bounded provider transports |
| `v2_workflow.py` | Authorization gate, dry run, checkpoint authentication, orchestration, and safe manifests |
| `v2_evidence_models.py` | Transient Places/review packets and authenticated Wayback/decision contracts |
| `v2_classifier.py` | Wayback-before-LLM prompt construction, strict structured response, and deterministic validation |
| `v2_eligibility.py` | Board-blind `include`/`exclude`/`unresolved` reducer and authenticated reason-code contract |
| `v2_downstream_models.py` | Non-serializable transient location envelopes bound to the classified Details snapshot |
| `v2_board_adapter.py` | Screened/all-active Board frames and independent deterministic linkage strata |
| `v2_spatial_adapter.py` | Fixed-origin EPSG:26915 straight-line proximity surfaces and coordinate QA |
| `v2_pilot.py` | Representative 8--10 record selection, A/B/C ablation, and closed safety summary |
| `v2_delivery.py` | Lazy local-input authentication and atomic full/offline or pilot delivery packages |
| `verify_v2_delivery.py` | Read-only delivery hash, schema, privacy, and reconciliation verifier |
| `../skills/google_maps.py` | Places API (New) Text Search and Details adapters alongside the v1 compatibility helper |
| `../skills/wayback_agent.py` | Independent v2 CDX/archived-page collector alongside the v1 summary compatibility helper |

The v2 order is Places API (New) candidate search, transient Place Details plus the
complete provider-returned review subset of at most five in relevance order,
independent Wayback evidence collection, constrained classification, deterministic
validation, then the later Board/linkage/proximity stages. Board records and later
outcomes never enter the model packet.

The v2.1 end-to-end implementation includes the three-state reducer, Board and spatial
adapters, bounded pilot, A/B/C ablation, atomic delivery, and strict verification.
After a clean 8-candidate, 24-condition pilot and one-candidate smoke test, full run
`20260831T230500Z` completed under `gpt-4.1`. Its manifest records 1,226 completed
candidates and zero runtime failures; its delivery passes the read-only verifier for
13 artifacts and 327 origins. The canonical pointer is
`data/v2/canonical_thesis_empirical_run.json`. Verify the saved package without any
provider call:

```powershell
& .\AIAGENTNETS\Scripts\python.exe .\code\verify_v2_delivery.py `
  .\data\v2\20260831T230500Z\delivery --expect-empirical
```

This completed run does not recreate or overwrite the frozen analytical snapshot
because v1 lacks stable Place IDs, the complete query log, raw provider responses,
and raw model responses. Earlier v2 attempts and their manifests remain preserved.

The v2 model stage is constrained by a target category, NAICS definition, explicit
decision rules, authenticated evidence IDs, and a strict output schema. Its fields are
candidate labels. Conventional code validates citations and historical-year meaning;
Board eligibility, cross-source record identity, name recurrence, and spatial proximity
remain later conventional-code stages. See `docs/PROMPT_GUIDE.md`.

`main.py` and `agent_workflow.py` preserve the v1 collector. They are not v2 entry
points. In v1, returned Google metadata and at most three review-text snippets entered
the prompt, while Wayback year/count summaries were attached afterward.

## Other scripts

Other Python files in this directory are historical, exploratory, support, or
superseded workflows. Do not infer current status from their names, comments, output
paths, or previously generated artifacts. Do not run them as current thesis analysis
or use them to populate later manuscript chapters unless a new explicit scope grant
and a fresh scientific audit authorize that work.

## Claim boundary

- V1 and v2 Google candidates reflect platform discoverability under their respective
  retrieval protocols; neither candidate frame is a census.
- Stored model labels are not independently adjudicated labels.
- A first Wayback capture is not an opening year. V2 permits an outlet-specific
  active-by bound or a separately validated explicit outlet opening claim.
- The screened Board inventory is a declared operational reference.
- Linkage outputs describe deterministic agreement under specified rules.
- Repeated names do not prove ownership or chain identity.
- Euclidean nearest-facility distance describes straight-line potential proximity, not
  route distance, travel time, realized access, service capacity, or health effects.

## Security and manuscript isolation

- Never print, copy, or commit `.env` values.
- Do not distribute `NETS-AI.zip`, private inputs, or
  `analysis/delivery/private/`.
- New acquisition must not overwrite the frozen input without a new versioned manifest
  and explicit authorization.
- Thesis-stage live execution is user-authorized, but its warning and provenance must
  not be represented as provider-issued permission or publication/release approval.
- The current manuscript gate includes Introduction, Literature Review, and Methods.
  The parallel analysis implementation does not authorize Results, Discussion,
  Conclusion, an analytical appendix, or a full-thesis build.
