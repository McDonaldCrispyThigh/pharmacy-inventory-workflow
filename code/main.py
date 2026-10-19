"""Entry point for the upstream establishment-candidate workflow.

The repository name records the project's NETS-related history.  This command
does not read, reproduce, or compare NETS data.  It retrieves Google Places
candidates and sends selected place metadata to a constrained LLM labeling
step.  The authenticated downstream analysis starts from a frozen local
snapshot; see ``analysis/README.md``.

Usage
-----
    python code/main.py --task coffee
    python code/main.py --task pharmacy --city Minneapolis
    python code/main.py --task library --city Minneapolis
    python code/main.py --task gym --city Minneapolis --zips 55401 55402 55403

Available tasks: library, park, coffee, gym, grocery, civic, religion, pharmacy
"""

import argparse
import os
import sys

# Force UTF-8 stdout so non-ASCII place names don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from skills.google_maps import GoogleMapsAgent
from agent_workflow import NETSAgentWorkflow

# ──────────────────────────────────────────────────────────────────────────────
# Business-category configuration
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_CONFIG = {
    "library": {
        "search_term":   "Public Library",
        "target_naics":  "519120",
        "sic_code":      "8231",
        "definition":    (
            "Facility that lends books and provides quiet study areas. "
            "Key signs: 'Librarian', 'Checkout', 'Computers'. Non-commercial."
        ),
    },
    "park": {
        "search_term":   "Park",
        "target_naics":  "712190",
        "sic_code":      "7999",
        "definition":    (
            "Designated outdoor area for nature and recreation. "
            "Key signs: 'Trail', 'Playground', 'Grass'. "
            "Distinct from 'Mobile Home Park' (Residential)."
        ),
    },
    "coffee": {
        "search_term":   "Coffee Shop",
        "target_naics":  "722515",
        "sic_code":      "5812",
        "definition":    (
            "Focuses on coffee/tea and light food. "
            "Key signs: Opens early (6-8 AM), serves breakfast. "
            "If it opens early, it is a Coffee Shop even if it serves beer."
        ),
    },
    "gym": {
        "search_term":   "Gym",
        "target_naics":  "713940",
        "sic_code":      "7991",
        "definition":    (
            "Facility for physical exercise. "
            "Key signs: 'Weights', 'Treadmills', 'Membership', 'Classes'. "
            "Distinct from 'Playground equipment store'."
        ),
    },
    "grocery": {
        "search_term":   "Grocery Store",
        "target_naics":  "445110",
        "sic_code":      "5411",
        "definition":    (
            "Retail store primarily selling fresh food, produce, and meats. "
            "Distinct from 'Convenience Store' (gas stations) or 'Liquor Store'."
        ),
    },
    "civic": {
        "search_term":   "Community Center",
        "target_naics":  "813410",
        "sic_code":      "8641",
        "definition":    (
            "Non-profit facility for social interaction and community support. "
            "Key signs: 'Volunteers', 'Community Events', 'Hall Rental'."
        ),
    },
    "religion": {
        "search_term":   "Place of Worship",
        "target_naics":  "813110",
        "sic_code":      "8661",
        "definition":    (
            "Facility for religious services. "
            "Key signs: 'Service', 'Prayer', 'Worship', 'Church/Mosque/Synagogue'."
        ),
    },
    "pharmacy": {
        "search_term":   "Pharmacy",
        # Multiple variants beat the 60-result/query cap via place_id dedup
        "search_variants": ["Pharmacy", "Drug Store", "Drugstore"],
        "target_naics":  "446110",
        # NAICS vintage matters and must travel with the code.
        # 446110 "Pharmacies and Drug Stores" is the 2017 NAICS code. The 2022
        # revision moved retail trade around and the equivalent code is 456110
        # (same title). Both are recorded so downstream users can tell which
        # taxonomy a record was classified under, and so a 2022-vintage
        # comparison source can be joined without silent mismatches.
        "naics_version": "2017",
        "target_naics_2022": "456110",
        "sic_code":      "5912",
        "definition":    (
            "Retail establishment dispensing prescription medications under a licensed pharmacist. "
            "Key signs: 'Prescription', 'Rx', 'Pharmacist on duty', 'Drive-thru Pharmacy'. "
            "Distinct from 'Health food store' (no Rx dispensing) and 'Medical clinic' (provides care, not drugs). "
            "Include hospital outpatient pharmacies; exclude mail-order-only facilities."
        ),
        # This upstream candidate definition is broader than the automatic
        # eligibility screen applied to Board records in the current pharmacy
        # test case.  The resulting source-universe asymmetry is documented as
        # a limitation; it must not be presented as a symmetric census.
        "classification_logic": (
            "1. Any name containing 'Pharmacy', 'Drug', or 'Rx' is almost certainly NAICS 446110.\n"
            "2. Hospital-based and clinic-adjacent pharmacies are also 446110.\n"
            "3. Health food stores or vitamin shops without prescription dispensing are NOT 446110.\n"
            "4. Grocery stores with pharmacy counters: assign 446110 only if the pharmacy is the primary focus of the Google Places listing.\n"
            "5. Reviews mentioning 'prescription', 'pharmacist', 'medication', 'insurance' confirm 446110."
        ),
    },
}

# Default ZIP codes for Minneapolis
DEFAULT_ZIP_CODES = [
    "55401", "55402", "55403", "55404", "55405",
    "55406", "55407", "55408", "55409", "55410",
    "55411", "55412", "55413", "55414", "55415",
    "55454", "55455",
]

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Google Places establishment candidates and assign "
            "task-specific labels with a constrained LLM step."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(CATEGORY_CONFIG.keys()),
        help="Business category to collect (e.g. pharmacy, coffee, library, gym).",
    )
    parser.add_argument(
        "--city",
        default="Minneapolis",
        help="Target city name (default: Minneapolis).",
    )
    parser.add_argument(
        "--zips",
        nargs="+",
        default=None,
        help="Space-separated ZIP codes to scan. Defaults to all Minneapolis ZIPs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "../data"),
        help="Directory to write the output CSV (default: ./data).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    args = parse_args()

    zip_codes = args.zips or DEFAULT_ZIP_CODES
    config    = CATEGORY_CONFIG[args.task]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    maps_agent    = GoogleMapsAgent()

    print(f">>> Task: {args.task} | City: {args.city} | ZIPs: {len(zip_codes)}")

    workflow = NETSAgentWorkflow(
        config        = config,
        maps_agent    = maps_agent,
        openai_client = openai_client,
        city_name     = args.city,
        zip_codes     = zip_codes,
    )
    output_path = workflow.run(output_dir=args.output_dir, task_name=args.task)

    if output_path:
        print(f"\n>>> Done. Output: {output_path}")
    else:
        print("\n>>> No records collected.")


if __name__ == "__main__":
    main()
