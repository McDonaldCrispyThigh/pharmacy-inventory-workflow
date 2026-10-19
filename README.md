# Pharmacy inventory workflow

Code, prompts, and derived results for the honors thesis *From Digital Listings to Neighborhood Measurement: An Evidence-Guided Workflow to Build and Validate Pharmacy Inventories* (Congyuan (Othello) Zheng, Department of Geography, University of Colorado Boulder, 2026).

The workflow turns Google Places listings into a pharmacy inventory for 60 ZIP codes in Minneapolis–St. Paul. A language model (gpt-5-mini, snapshot 2025-08-07) reads an evidence packet for each candidate (the listing, its reviews, and archived business websites) and makes four forced-choice judgments; a fixed rule turns them into inclusion decisions. The inventory is then compared with the Minnesota Board of Pharmacy license list (RQ2) and used to classify census tracts by straight-line distance to the nearest pharmacy (RQ3).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Pipeline code: retrieval, archive evidence, classification, Board linkage, spatial analysis, and verification scripts. Older project scripts are kept for provenance; `code/README.md` describes the code surface. |
| `skills/` | Early retrieval helpers kept for provenance; the final results do not use them. |
| `analysis/rq2_linkage_calibration/` | Calibration of the four linkage parameters across 3,087 combinations (thesis Appendix A.5.6), with the selection rule, its amendment, and result grids in `results/`. |
| `analysis/rq3_common_supply/results/` | Tract-level RQ3 results: paired classifications, sensitivity scenarios, conditional ranges, weighting, and neighborhood quartiles. |
| `analysis/review_diagnostics/` | Read-only checks used in the committee-review response (causes of unmatched Board records; historical-year outputs). |
| `analysis/input_manifest.json` | Paths, hashes, roles, and vintages of the frozen inputs. |
| `results/thesis_tables/` | Every table in the thesis and its appendices as CSV (one file per table or panel). |
| `results/figures/` | Thesis figures (PNG, with PDF/SVG sources where they exist). |

## Main entry points

| Stage | Thesis section | Code |
|---|---|---|
| Candidate retrieval and Place Details | 3.2, A.2 | `code/v2_workflow.py`, `code/run_v2_workflow.py`, `code/run_supplemental_836.py`, `code/screen_supplemental_details.py`, `code/diagnose_saturated_zip_search.py` |
| Website archive evidence | 3.2, A.3 | `code/clean_archive_discovery.py`, `code/clean_archive_parser.py`, `code/clean_archive_closure.py`, `code/wayback_*.py` |
| Evidence packets | 3.3, A.4 | `code/v2_evidence_models.py` |
| LLM judgments (system prompt and fixed rules) | 3.3, A.4 | `code/forced_choice_classifier.py`, `code/forced_choice_models.py`, `code/run_forced_choice_workflow.py` |
| Inclusion rule | 3.3, A.4 | `code/forced_choice_eligibility.py` |
| Board screening and linkage | 3.4, A.5 | `code/v2_board_adapter.py`, `code/geo_match.py`, `code/study_design.py` |
| Downstream RQ1 / RQ2 | 4.1–4.2 | `code/run_expanded_rq1.py`, `code/run_expanded_final.py` |
| Linkage calibration | A.5.6 | `analysis/rq2_linkage_calibration/calibrate.py`, `factorial.py`, `core.py` |
| RQ3 spatial comparison | 3.5, 4.3, A.7 | `code/run_common_supply_rq3.py`, `code/v2_spatial_adapter.py` |

Developed with Python 3.14; install dependencies with `pip install -r requirements.txt`. Downstream scripts start from the frozen inputs listed in `analysis/input_manifest.json` and make no calls to Google, the archive services, or the model. Rerunning retrieval or classification needs API keys (see `.env.example`; keep `.env` local) and produces new observations rather than a reproduction.

## Data availability

Included: code, the prompt text (in `code/forced_choice_classifier.py`; reproduced verbatim in thesis Appendix A.4), aggregate and tract-level results, and the thesis tables and figures.

Not included:

- Google Places responses, reviews, and coordinates, which the provider's terms do not allow us to redistribute.
- Archived page text and model requests and responses (the study's private records).
- Per-record files keyed by Google Place ID or Board license number.
- The Minnesota Board of Pharmacy export (13 April 2026). It is a public record available from the Board, but it includes contact e-mail addresses.
- 2019–2023 ACS estimates and 2023 TIGER/Line geometries, which are public from the U.S. Census Bureau.

The provenance record does not document how or when the Board coordinate cache was produced.

## License

Code is released under the MIT License (see `LICENSE`).
