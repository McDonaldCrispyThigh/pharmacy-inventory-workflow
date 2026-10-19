"""
validate_nppes.py  --  Compare AI-collected pharmacy data vs NPPES NPI Registry.

The NPPES (National Plan and Provider Enumeration System) registry is a free,
publicly accessible CMS database of all US healthcare providers, including
every licensed pharmacy. No API key or data request is required.

Usage
-----
    # Run after collecting pharmacy data with main.py:
    python code/validate_nppes.py --ai-csv data/Minneapolis_pharmacy_20250412_123456.csv

    # Query by ZIP code instead of city name (more precise):
    python code/validate_nppes.py --ai-csv data/Minneapolis_pharmacy_20250412_123456.csv --use-zips

    # Save the matched table to a CSV for further analysis:
    python code/validate_nppes.py --ai-csv data/Minneapolis_pharmacy_20250412_123456.csv --output data/validation_result.csv

    # Adjust fuzzy-match sensitivity (default 0.75, lower = more lenient):
    python code/validate_nppes.py --ai-csv data/Minneapolis_pharmacy_20250412_123456.csv --threshold 0.70
"""

import argparse
import json
import time
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

NPPES_API = "https://npiregistry.cms.hhs.gov/api/?version=2.1"

# NUCC taxonomy codes covering all storefront pharmacy types
PHARMACY_TAXONOMY_CODES = {
    "3336C0003X",  # Community/Retail Pharmacy
    "3336I0012X",  # Institutional Pharmacy (hospital outpatient)
    "3336L0003X",  # Long-Term Care Pharmacy
    "3336H0001X",  # Home Infusion Therapy Pharmacy
    "3336S0011X",  # Specialty Pharmacy
}


# ──────────────────────────────────────────────────────────────────────────────
# NPPES fetch
# ──────────────────────────────────────────────────────────────────────────────

def fetch_nppes_pharmacies(state: str, city: str = None,
                           zip_codes: list[str] = None) -> list[dict]:
    """
    Pull all active pharmacies from NPPES for the given area.

    Queries by ZIP code list when --use-zips is set (more precise), otherwise
    queries by city name and paginates through all results.
    """
    records = []

    if zip_codes:
        for zc in zip_codes:
            params = {
                "taxonomy_description": "Pharmacy",
                "state":       state,
                "postal_code": zc,
                "limit":       200,
                "skip":        0,
                "version":     "2.1",
            }
            print(f"    Querying NPPES ZIP {zc} ...", end=" ", flush=True)
            chunk = _fetch_all_pages(params)
            print(f"{len(chunk)} results")
            records.extend(chunk)
            time.sleep(0.3)
    else:
        params = {
            "taxonomy_description": "Pharmacy",
            "state":   state,
            "city":    city or "",
            "limit":   200,
            "skip":    0,
            "version": "2.1",
        }
        records.extend(_fetch_all_pages(params))

    # De-duplicate by NPI number
    seen: set[str] = set()
    unique = []
    for r in records:
        npi = r.get("number")
        if npi and npi not in seen:
            seen.add(npi)
            unique.append(r)

    return unique


def _fetch_all_pages(base_params: dict, max_retries: int = 3) -> list[dict]:
    """Paginate through NPPES results using the skip parameter, with retry."""
    all_results: list[dict] = []
    skip = 0

    while True:
        params = {**base_params, "skip": skip}
        url = NPPES_API + "?" + urlencode(params)

        data = None
        for attempt in range(max_retries):
            try:
                with urlopen(url, timeout=15) as resp:
                    data = json.loads(resp.read())
                break
            except HTTPError as e:
                if e.code == 429:
                    wait = 2 ** attempt
                    print(f"\n    [NPPES rate limit] retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"\n    [NPPES HTTP {e.code}] {e.reason}")
                    return all_results
            except URLError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"\n    [NPPES network error] {e}")
                    return all_results
            except json.JSONDecodeError as e:
                print(f"\n    [NPPES parse error] {e}")
                return all_results

        if data is None:
            break

        results = data.get("results", [])
        all_results.extend(results)

        total = data.get("result_count", 0)
        skip += len(results)
        if skip >= total or not results:
            break

    return all_results


def nppes_to_df(records: list[dict]) -> pd.DataFrame:
    """Flatten NPPES JSON records into a tidy DataFrame; keep only active pharmacies."""
    rows = []
    for r in records:
        basic = r.get("basic", {})

        # Skip inactive records
        if basic.get("status") != "A":
            continue

        # Prefer LOCATION address over mailing address
        addresses = r.get("addresses", [])
        addr = next(
            (a for a in addresses if a.get("address_purpose") == "LOCATION"),
            addresses[0] if addresses else {},
        )

        taxonomies = r.get("taxonomies", [])
        tax_codes  = [t.get("code", "") for t in taxonomies]
        tax_descs  = [t.get("desc", "") for t in taxonomies]

        # Only keep pharmacy-type providers
        if not any(c in PHARMACY_TAXONOMY_CODES for c in tax_codes):
            continue

        rows.append({
            "NPI":            r.get("number"),
            "Name":           basic.get("organization_name", ""),
            "Address":        addr.get("address_1", ""),
            "City":           addr.get("city", ""),
            "State":          addr.get("state", ""),
            "ZIP":            str(addr.get("postal_code", ""))[:5],
            "Taxonomy_Codes": "|".join(tax_codes),
            "Taxonomy_Descs": "|".join(tax_descs),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Name normalization and fuzzy matching
# ──────────────────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", ".,'-#&/")
_DROP_WORDS   = frozenset({
    "pharmacy", "drug", "rx", "drugs", "store", "stores",
    "inc", "llc", "corp", "co", "the", "and",
})


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, remove generic pharmacy words."""
    s = name.lower().translate(_PUNCT_TABLE)
    tokens = [w for w in s.split() if w not in _DROP_WORDS]
    result = " ".join(tokens)
    return result if result else name.lower()  # fallback if all tokens dropped


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def match_records(ai_df: pd.DataFrame, nppes_df: pd.DataFrame,
                  threshold: float = 0.75) -> pd.DataFrame:
    """
    Greedy 1-to-1 matching: each NPPES record is claimed by at most one AI record.

    Algorithm:
    1. Score every (AI, NPPES) pair; apply 15% cross-ZIP penalty.
    2. Sort all candidate pairs by score descending.
    3. Greedily assign: take the best pair, mark both sides used, repeat.

    This prevents one pharmacy chain (e.g. CVS) from absorbing all NPPES matches
    and inflating True Positives beyond the total NPPES count.
    """
    ai_df = ai_df.copy()
    ai_df["NPPES_Match_Name"]  = None
    ai_df["NPPES_Match_NPI"]   = None
    ai_df["NPPES_Match_Score"] = 0.0
    ai_df["NPPES_Match_ZIP"]   = None
    ai_df["Is_NPPES_Match"]    = False

    # Build all candidate pairs above threshold
    candidates: list[tuple[float, int, int]] = []   # (score, ai_idx, nppes_idx)
    for i, row in ai_df.iterrows():
        ai_zip = str(row.get("Zip_Code", ""))[:5]
        for j, nrec in nppes_df.iterrows():
            score = _fuzzy(str(row["Company"]), str(nrec["Name"]))
            if ai_zip and nrec["ZIP"] and ai_zip != nrec["ZIP"]:
                score *= 0.85
            if score >= threshold:
                candidates.append((score, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_ai:    set[int] = set()
    used_nppes: set[int] = set()

    for score, ai_idx, nppes_idx in candidates:
        if ai_idx in used_ai or nppes_idx in used_nppes:
            continue
        nrec = nppes_df.loc[nppes_idx]
        ai_df.at[ai_idx, "NPPES_Match_Name"]  = nrec["Name"]
        ai_df.at[ai_idx, "NPPES_Match_NPI"]   = nrec["NPI"]
        ai_df.at[ai_idx, "NPPES_Match_Score"] = round(score, 3)
        ai_df.at[ai_idx, "NPPES_Match_ZIP"]   = nrec["ZIP"]
        ai_df.at[ai_idx, "Is_NPPES_Match"]    = True
        used_ai.add(ai_idx)
        used_nppes.add(nppes_idx)

    return ai_df


# ──────────────────────────────────────────────────────────────────────────────
# Metrics and reporting
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(ai_matched: pd.DataFrame, nppes_df: pd.DataFrame) -> dict:
    total_ai   = len(ai_matched)
    tp         = int(ai_matched["Is_NPPES_Match"].sum())
    fp         = total_ai - tp

    matched_npis = set(
        ai_matched.loc[ai_matched["Is_NPPES_Match"], "NPPES_Match_NPI"].dropna()
    )
    total_nppes = len(nppes_df)
    fn          = total_nppes - len(matched_npis)

    precision = tp / total_ai       if total_ai    > 0 else 0.0
    recall    = len(matched_npis) / total_nppes if total_nppes > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "ai_total":        total_ai,
        "nppes_total":     total_nppes,
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
    }


def print_report(metrics: dict) -> None:
    sep = "=" * 57
    print(f"\n{sep}")
    print("  VALIDATION REPORT   (AI output  vs  NPPES NPI Registry)")
    print(sep)
    print(f"  AI records collected   : {metrics['ai_total']}")
    print(f"  NPPES pharmacies found : {metrics['nppes_total']}")
    print(f"  True  positives  (TP)  : {metrics['true_positives']}")
    print(f"  False positives  (FP)  : {metrics['false_positives']}")
    print(f"  False negatives  (FN)  : {metrics['false_negatives']}")
    print("-" * 57)
    print(f"  Precision              : {metrics['precision']:.1%}")
    print(f"  Recall                 : {metrics['recall']:.1%}")
    print(f"  F1 Score               : {metrics['f1']:.1%}")
    print(sep)


def _print_false_positives(ai_matched: pd.DataFrame, n: int = 10) -> None:
    fp_df = ai_matched[~ai_matched["Is_NPPES_Match"]]
    if fp_df.empty:
        return
    print(f"\n  False Positives (AI found, not in NPPES) -- first {min(n, len(fp_df))}:")
    for _, r in fp_df.head(n).iterrows():
        print(f"    AI: {r['Company']} | {r.get('Street_Address', '')} ZIP {r.get('Zip_Code', '')}")


def _print_false_negatives(ai_matched: pd.DataFrame, nppes_df: pd.DataFrame,
                            n: int = 10) -> None:
    matched_npis = set(
        ai_matched.loc[ai_matched["Is_NPPES_Match"], "NPPES_Match_NPI"].dropna()
    )
    fn_df = nppes_df[~nppes_df["NPI"].isin(matched_npis)]
    if fn_df.empty:
        return
    print(f"\n  False Negatives (in NPPES, not found by AI) -- first {min(n, len(fn_df))}:")
    for _, r in fn_df.head(n).iterrows():
        print(f"    NPPES: {r['Name']} | {r['Address']}, {r['City']} {r['ZIP']}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate AI pharmacy CSV against NPPES NPI Registry."
    )
    p.add_argument(
        "--ai-csv", required=True,
        help="Path to the AI-generated CSV produced by main.py --task pharmacy.",
    )
    p.add_argument(
        "--state", default="MN",
        help="US state abbreviation for NPPES query (default: MN).",
    )
    p.add_argument(
        "--city", default=None,
        help="City name for NPPES query. Inferred from the CSV when omitted.",
    )
    p.add_argument(
        "--use-zips", action="store_true",
        help="Query NPPES by each ZIP code in the AI CSV rather than by city name. "
             "Slower but more precise for MSA-level studies.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.75,
        help="Fuzzy name-similarity threshold 0-1 (default 0.75). "
             "Lower values increase recall at the cost of precision.",
    )
    p.add_argument(
        "--output", default=None,
        help="Save the matched AI table (with NPPES columns appended) to this CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load AI output
    ai_df = pd.read_csv(args.ai_csv)
    print(f">>> Loaded {len(ai_df)} AI records from: {args.ai_csv}")

    # Infer city from data if not specified
    city = args.city
    if not city and "City" in ai_df.columns:
        city = str(ai_df["City"].mode()[0])
    city = city or "Minneapolis"

    # Build ZIP list when --use-zips
    zip_codes = None
    if args.use_zips and "Zip_Code" in ai_df.columns:
        zip_codes = sorted(
            ai_df["Zip_Code"].dropna().astype(str).str[:5].unique().tolist()
        )
        print(f">>> ZIP-level query mode: {len(zip_codes)} ZIPs")

    # Fetch NPPES ground truth
    print(f">>> Fetching NPPES pharmacies [state={args.state}, city={city}] ...")
    nppes_records = fetch_nppes_pharmacies(
        state=args.state, city=city, zip_codes=zip_codes
    )
    nppes_df = nppes_to_df(nppes_records)
    print(f">>> {len(nppes_df)} active NPPES pharmacy records retrieved.")

    if nppes_df.empty:
        print("No NPPES records found. Try --city or check --state parameter.")
        return

    # Match AI output against NPPES
    print(f">>> Matching records (threshold={args.threshold}) ...")
    ai_matched = match_records(ai_df, nppes_df, threshold=args.threshold)

    # Print report
    metrics = compute_metrics(ai_matched, nppes_df)
    print_report(metrics)
    _print_false_positives(ai_matched)
    _print_false_negatives(ai_matched, nppes_df)

    # Save matched table
    if args.output:
        ai_matched.to_csv(args.output, index=False)
        print(f"\n>>> Matched results saved to: {args.output}")


if __name__ == "__main__":
    main()
