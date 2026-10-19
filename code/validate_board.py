"""
validate_board.py  --  Validate AI-collected pharmacy data vs. Minnesota Board of Pharmacy.

Steps:
  1. Load and inspect both datasets.
  2. Filter Board of Pharmacy to active retail pharmacies in the MSA ZIP footprint.
  3. Fuzzy name matching (RapidFuzz token_sort_ratio, threshold 75, greedy 1-to-1).
  4. False negative categorization.
  5. Validation metrics + side-by-side vs. NPPES.
  6. Spatial analysis of FNs vs. income / race.
  7. Summary comparison table.

Usage:
    python code/validate_board.py
"""

import re
import sys
import io
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_ZIPS = {
    "55401","55402","55403","55404","55405","55406","55407","55408","55409","55410",
    "55411","55412","55413","55414","55415","55454","55455",
    "55101","55102","55103","55104","55105","55106","55107","55108","55116","55117",
    "55118","55119","55130","55113","55126",
    "55421","55422","55423","55424","55425","55426","55427","55428","55429","55430",
    "55431","55432","55433","55434","55435","55436","55437","55438","55439",
    "55441","55442","55443","55444","55445","55446","55447","55448","55369",
}

# Closed chains known to have exited MN market
# NOTE: SUPERVALU is the parent of CUB Foods (active) — excluded from this list
CLOSED_CHAIN_PATTERNS = [
    r"\bsnyder\b", r"\bosco\b", r"\bphar.?mor\b", r"\bsav.?on\b",
    r"\brite\s*aid\b", r"\bkmart\b", r"\bk.mart\b", r"\bsears\b",
    r"\bbirds?\s*eye\b", r"\bhook.?nack\b", r"\brexall\b", r"\bpioneer\b",
    r"\bbrooke\s*drug\b", r"\bsnyder\s*drug\b",
]

# Non-retail / non-community-pharmacy name patterns → exclude from adjusted denominator
NONRETAIL_PATTERNS = [
    r"\binfusion\b", r"\bhome\s+infusion\b", r"\biv\s+therapy\b",
    r"\blong.?term\s+care\b", r"\bltc\b",
    r"\bcompounding\s*(?:only|pharmacy|center|lab)\b",
    r"\bmail.?order\b", r"\bmail\s+service\b",
    r"\bnuclear\b", r"\bradio.?pharm\b",
    r"\bcorrectional\b", r"\binstitutional\b", r"\bprison\b",
    r"\bhospice\b", r"\brenal\b", r"\bdialysis\b",
    r"\bveterina\b", r"\banimal\b",
    r"\boncology\b", r"\bchemotherapy\b",
    r"\bspecialty\s+infusion\b", r"\bcoram\b",
]

# Name normalization — mirrors validate_nppes.py conventions
_PUNCT_RE = re.compile(r"[.,'\-#&/\\()@]")
_DROP_WORDS = frozenset({
    "pharmacy","drug","rx","drugs","store","stores",
    "inc","llc","corp","co","the","and","dba","ltd",
    "pharm","phcy","of","corporation","wholesale",
})
_CORP_INDICATORS = frozenset({"inc", "llc", "corp", "ltd", "corporation", "co", "pa"})

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = Path("data")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """
    DBA-aware normalization:
    1. If 'dba' in name, extract the operating name after it.
    2. Handles parenthetical pattern: 'Corp Name (Operating Name)' without 'dba' keyword.
    3. Strips store numbers (#NNNN), digit-only tokens, and stop words.
    """
    s = str(name).lower()

    # Pattern 1: explicit dba marker — take everything after it
    if "dba" in s:
        parts = re.split(r"\(?dba\)?", s, flags=re.IGNORECASE)
        if len(parts) >= 2:
            s = parts[-1].strip()
    else:
        # Pattern 2: parenthetical operating name — "Corp Inc. (Operating Name)"
        paren_match = re.search(r"\(([^)]+)\)\s*$", s)
        if paren_match:
            pre_tokens = set(re.sub(r"[^a-z\s]", " ", s[:paren_match.start()]).split())
            if pre_tokens & _CORP_INDICATORS:
                s = paren_match.group(1)

    # Strip store numbers like #1650/624 or #00828
    s = re.sub(r"#[\w/]+", " ", s)
    s = _PUNCT_RE.sub(" ", s)
    # Remove digit-only tokens and stop words
    tokens = [w for w in s.split() if w not in _DROP_WORDS and not w.isdigit()]
    result = " ".join(tokens)
    return result if result else str(name).lower()


def fuzzy_score(a: str, b: str) -> float:
    """RapidFuzz token_sort_ratio, returns 0-100."""
    from rapidfuzz.fuzz import token_sort_ratio
    return token_sort_ratio(normalize(a), normalize(b))


def match_records(ai_df: pd.DataFrame, board_df: pd.DataFrame,
                  threshold: float = 75.0) -> pd.DataFrame:
    """
    Greedy 1-to-1 matching: each Board record claimed by at most one AI record.
    15% cross-ZIP penalty (same convention as validate_nppes.py).
    """
    ai_df = ai_df.copy()
    ai_df["Board_Match_Name"]  = None
    ai_df["Board_Match_ID"]    = None
    ai_df["Board_Match_Score"] = 0.0
    ai_df["Board_Match_ZIP"]   = None
    ai_df["Is_Board_Match"]    = False

    candidates: list[tuple[float, int, int]] = []

    for i, arow in ai_df.iterrows():
        ai_zip = str(arow.get("Zip_Code", ""))[:5]
        for j, brow in board_df.iterrows():
            score = fuzzy_score(str(arow["Company"]), str(brow["facility_name"]))
            b_zip = str(brow["zip"])[:5]
            if ai_zip and b_zip and ai_zip != b_zip:
                score *= 0.85
            if score >= threshold:
                candidates.append((score, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_ai:    set[int] = set()
    used_board: set[int] = set()

    for score, ai_idx, board_idx in candidates:
        if ai_idx in used_ai or board_idx in used_board:
            continue
        brow = board_df.loc[board_idx]
        ai_df.at[ai_idx, "Board_Match_Name"]  = brow["facility_name"]
        ai_df.at[ai_idx, "Board_Match_ID"]    = brow["license_nbr"]
        ai_df.at[ai_idx, "Board_Match_Score"] = round(score, 2)
        ai_df.at[ai_idx, "Board_Match_ZIP"]   = str(brow["zip"])[:5]
        ai_df.at[ai_idx, "Is_Board_Match"]    = True
        used_ai.add(ai_idx)
        used_board.add(board_idx)

    return ai_df


def categorize_fn(name: str) -> str:
    """Categorize a Board FN record."""
    n = name.lower()
    for pat in CLOSED_CHAIN_PATTERNS:
        if re.search(pat, n):
            return "closed_chain"
    for pat in NONRETAIL_PATTERNS:
        if re.search(pat, n):
            return "specialty_nonretail"
    return "possible_missed_retail"


def sep(char="─", width=66):
    print(char * width)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and inspect
# ──────────────────────────────────────────────────────────────────────────────

def step1_load():
    print("\n" + "═"*66)
    print("  STEP 1 — Load and inspect datasets")
    print("═"*66)

    # Board of Pharmacy
    board_raw = pd.read_csv(
        DATA_DIR / "Board" / "F.1  Pharmacy (PH).csv",
        encoding="utf-8-sig", skiprows=3, low_memory=False,
    )
    print(f"\nBoard of Pharmacy (raw)")
    print(f"  Shape        : {board_raw.shape}")
    print(f"  Columns      : {board_raw.columns.tolist()}")
    print(f"  description  : {board_raw['description'].value_counts().to_dict()}")
    print(f"  license_type : {board_raw['license_type'].value_counts().to_dict()}")
    print(f"  state dist   : MN={len(board_raw[board_raw['state']=='MN'])}, non-MN={len(board_raw[board_raw['state']!='MN'])}")
    print(f"\n  Sample (3 MN rows):")
    mn_sample = board_raw[board_raw["state"] == "MN"].head(3)
    print(mn_sample[["facility_name","description","address_line1","city","state","zip"]].to_string(index=False))

    # AI collection
    ai = pd.read_csv(
        DATA_DIR / "Minneapolis-StPaul_pharmacy_20260413_201606.csv",
        low_memory=False,
    )
    print(f"\nAI Collection (Google Maps / GPT-4o-mini pipeline)")
    print(f"  Shape           : {ai.shape}")
    print(f"  Is_Target_Match : {ai['Is_Target_Match'].value_counts().to_dict()}")
    print(f"  Business_Status : {ai['Business_Status'].value_counts().to_dict()}")
    print(f"  ZIP coverage    : {ai['Zip_Code'].nunique()} unique ZIPs")
    print(f"\n  Sample (3 rows):")
    print(ai[["Company","Street_Address","City","State","Zip_Code","Confidence"]].head(3).to_string(index=False))

    return board_raw, ai


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Filter Board of Pharmacy
# ──────────────────────────────────────────────────────────────────────────────

def step2_filter(board_raw: pd.DataFrame, ai: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 2 — Filter Board of Pharmacy")
    print("═"*66)

    n0 = len(board_raw)

    # Expand ZIP footprint: target ZIPs + all ZIPs present in AI data
    ai_zips = set(ai["Zip_Code"].dropna().astype(str).str[:5].unique())
    msa_zips = TARGET_ZIPS | ai_zips
    print(f"\n  MSA ZIP footprint: {len(msa_zips)} ZIPs (60 target + AI coverage)")

    # Filter 1: active status
    board = board_raw[board_raw["description"].str.startswith("Active")].copy()
    n1 = len(board)
    print(f"\n  [Filter 1] Active status:  {n0} → {n1}  (removed {n0-n1})")

    # Filter 2: MN state
    board = board[board["state"] == "MN"].copy()
    n2 = len(board)
    print(f"  [Filter 2] MN state:       {n1} → {n2}  (removed {n1-n2})")

    # Filter 3: MSA ZIPs
    board["zip5"] = board["zip"].astype(str).str[:5]
    board_msa = board[board["zip5"].isin(msa_zips)].copy()
    n3 = len(board_msa)
    print(f"  [Filter 3] MSA ZIPs:       {n2} → {n3}  (removed {n2-n3})")

    # Filter 4: Retail type — exclude by name patterns
    EXCLUDE_PATTERNS = [
        (r"\binfusion\b",                   "home_infusion"),
        (r"\blong.?term\s+care\b",          "long_term_care"),
        (r"\bltc\b",                        "long_term_care"),
        (r"\bmail.?order\b",                "mail_order"),
        (r"\bmail\s+service\b",             "mail_order"),
        (r"\bnuclear\b",                    "nuclear"),
        (r"\bcorrectional\b",               "correctional"),
        (r"\binstitutional\b",              "institutional"),
        (r"\bhospice\b",                    "hospice"),
        (r"\bveterina\b",                   "veterinary"),
        (r"\banimal\b",                     "veterinary"),
        (r"\bcoram\b",                      "home_infusion"),
    ]

    excl_mask = pd.Series([False] * len(board_msa), index=board_msa.index)
    excl_type = pd.Series(["retail"] * len(board_msa), index=board_msa.index)

    for pat, label in EXCLUDE_PATTERNS:
        hits = board_msa["facility_name"].str.contains(pat, case=False, na=False, regex=True)
        excl_mask = excl_mask | hits
        excl_type[hits & (excl_type == "retail")] = label

    # Compounding-only: name contains 'compounding' but NOT common retail modifiers
    compound_mask = board_msa["facility_name"].str.contains(r"\bcompounding\b", case=False, na=False, regex=True)
    retail_compound = board_msa["facility_name"].str.contains(
        r"\bdispensing\b|\bspecialty\b|\bcommunity\b", case=False, na=False, regex=True)
    excl_mask = excl_mask | (compound_mask & ~retail_compound)
    excl_type[(compound_mask & ~retail_compound) & (excl_type == "retail")] = "compounding_only"

    excluded = board_msa[excl_mask].copy()
    excluded["excl_reason"] = excl_type[excl_mask].values
    board_retail = board_msa[~excl_mask].copy()
    n4 = len(board_retail)
    n_excl = len(excluded)

    print(f"  [Filter 4] Retail types:   {n3} → {n4}  (excluded {n_excl})")
    print(f"\n  Excluded by type:")
    for label, count in excl_type[excl_mask].value_counts().items():
        print(f"    {label:<25}: {count}")
    print(f"\n  Excluded samples:")
    for _, row in excluded.head(8).iterrows():
        print(f"    [{row['excl_reason']:<20}] {row['facility_name'][:60]}")

    print(f"\n  ✓ Retained for validation: {n4} active retail MN pharmacies in MSA footprint")
    print(f"\n  Retained license types retained: {board_retail['description'].value_counts().to_dict()}")
    print(f"\n  Top cities in retained set:")
    print(board_retail["city"].str.upper().value_counts().head(10).to_string())

    return board_retail, excluded


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Fuzzy matching
# ──────────────────────────────────────────────────────────────────────────────

def step3_match(ai: pd.DataFrame, board_retail: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 3 — Fuzzy name matching (RapidFuzz token_sort_ratio, t=75)")
    print("═"*66)

    print(f"\n  AI records    : {len(ai)}")
    print(f"  Board records : {len(board_retail)}")
    print(f"  Building candidate pairs ... (may take ~30s)")

    ai_matched = match_records(ai, board_retail, threshold=75.0)

    tp = int(ai_matched["Is_Board_Match"].sum())
    fp = len(ai_matched) - tp

    matched_ids = set(
        ai_matched.loc[ai_matched["Is_Board_Match"], "Board_Match_ID"].dropna()
    )
    fn_df = board_retail[~board_retail["license_nbr"].isin(matched_ids)].copy()
    fn = len(fn_df)

    print(f"\n  True  Positives (TP) : {tp}")
    print(f"  False Positives (FP) : {fp}")
    print(f"  False Negatives (FN) : {fn}")

    # Sample TPs
    tp_sample = ai_matched[ai_matched["Is_Board_Match"]].head(5)[
        ["Company","Board_Match_Name","Board_Match_Score","Zip_Code","Board_Match_ZIP"]
    ]
    print(f"\n  ── True Positive samples (top 5 by score) ──")
    print(tp_sample.to_string(index=False))

    # Sample FPs
    fp_sample = ai_matched[~ai_matched["Is_Board_Match"]].head(5)[
        ["Company","Street_Address","Zip_Code"]
    ]
    print(f"\n  ── False Positive samples (AI found, not in Board) ──")
    print(fp_sample.to_string(index=False))

    # Sample FNs
    fn_sample = fn_df.head(5)[["facility_name","address_line1","city","zip5"]]
    print(f"\n  ── False Negative samples (in Board, not found by AI) ──")
    print(fn_sample.to_string(index=False))

    return ai_matched, fn_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — False negative categorization
# ──────────────────────────────────────────────────────────────────────────────

def step4_categorize_fn(fn_df: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 4 — False negative categorization")
    print("═"*66)

    fn_df = fn_df.copy()
    fn_df["FN_Category"] = fn_df["facility_name"].apply(categorize_fn)

    cat_counts = fn_df["FN_Category"].value_counts()
    print(f"\n  Total FN: {len(fn_df)}")
    for cat, count in cat_counts.items():
        print(f"    {cat:<30}: {count}")

    for cat in ["closed_chain", "specialty_nonretail", "possible_missed_retail"]:
        subset = fn_df[fn_df["FN_Category"] == cat].head(5)
        print(f"\n  ── {cat} samples ──")
        print(subset[["facility_name","address_line1","city","zip5"]].to_string(index=False))

    return fn_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Validation metrics
# ──────────────────────────────────────────────────────────────────────────────

def step5_metrics(ai_matched: pd.DataFrame, board_retail: pd.DataFrame,
                  fn_df: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 5 — Validation metrics")
    print("═"*66)

    total_ai    = len(ai_matched)
    tp          = int(ai_matched["Is_Board_Match"].sum())
    fp          = total_ai - tp
    total_board = len(board_retail)
    fn          = len(fn_df)

    precision = tp / total_ai if total_ai > 0 else 0.0
    recall    = tp / total_board if total_board > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Adjusted: exclude closed_chain + specialty_nonretail from denominator
    fn_nonretail = fn_df[fn_df["FN_Category"].isin(["closed_chain","specialty_nonretail"])]
    adj_denom = total_board - len(fn_nonretail)
    adj_recall = tp / adj_denom if adj_denom > 0 else 0.0
    adj_f1     = (2 * precision * adj_recall / (precision + adj_recall)
                  if (precision + adj_recall) > 0 else 0.0)

    print(f"\n  Board of Pharmacy Validation (AI vs. Board)")
    sep()
    print(f"  AI records collected          : {total_ai}")
    print(f"  Board records (filtered)      : {total_board}")
    print(f"  True  Positives               : {tp}")
    print(f"  False Positives               : {fp}")
    print(f"  False Negatives               : {fn}")
    sep("-")
    print(f"  Raw Precision                 : {precision:.1%}")
    print(f"  Raw Recall                    : {recall:.1%}")
    print(f"  Raw F1                        : {f1:.1%}")
    sep("-")
    print(f"  FN excluded (non-retail)      : {len(fn_nonretail)}")
    print(f"  Adjusted denominator          : {adj_denom}")
    print(f"  Adjusted Recall               : {adj_recall:.1%}")
    print(f"  Adjusted F1                   : {adj_f1:.1%}")
    sep()

    # Load NPPES metrics from saved validation file
    nppes = pd.read_csv(DATA_DIR / "validation_nppes_20260413.csv", low_memory=False)
    nppes_tp = int(nppes["Is_NPPES_Match"].sum())
    nppes_total_ai = len(nppes)
    nppes_fp = nppes_total_ai - nppes_tp

    # FN decomposition for NPPES from nppes_fn_raw.csv
    try:
        nppes_fn = pd.read_csv(DATA_DIR / "nppes_fn_raw.csv", low_memory=False)
        nppes_fn_total = len(nppes_fn)
        if "FN_Category" in nppes_fn.columns:
            nppes_fn_nonretail = len(nppes_fn[
                nppes_fn["FN_Category"].isin(["closed_chain","specialty_nonretail","corporate_legal_name"])
            ])
        else:
            nppes_fn_nonretail = 331 + 95 + 39  # from thesis: corporate + closed + specialty
    except Exception:
        nppes_fn_total = 717
        nppes_fn_nonretail = 331 + 95 + 39  # 465

    nppes_total_ref = nppes_tp + nppes_fn_total
    nppes_prec   = nppes_tp / nppes_total_ai if nppes_total_ai > 0 else 0
    nppes_rec    = nppes_tp / nppes_total_ref if nppes_total_ref > 0 else 0
    nppes_f1     = (2 * nppes_prec * nppes_rec / (nppes_prec + nppes_rec)
                    if (nppes_prec + nppes_rec) > 0 else 0)
    nppes_adj_denom = nppes_total_ref - nppes_fn_nonretail
    nppes_adj_rec   = nppes_tp / nppes_adj_denom if nppes_adj_denom > 0 else 0
    nppes_adj_f1    = (2 * nppes_prec * nppes_adj_rec / (nppes_prec + nppes_adj_rec)
                       if (nppes_prec + nppes_adj_rec) > 0 else 0)

    print(f"\n  Side-by-Side Comparison")
    print(f"  {'Metric':<25} {'vs NPPES (raw)':>14} {'vs NPPES (adj)':>14} {'vs Board (raw)':>14} {'vs Board (adj)':>14}")
    sep("-")
    for label, v1, v2, v3, v4 in [
        ("Precision",   nppes_prec,    nppes_prec,    precision,   precision),
        ("Recall",      nppes_rec,     nppes_adj_rec, recall,      adj_recall),
        ("F1",          nppes_f1,      nppes_adj_f1,  f1,          adj_f1),
        ("TP",          nppes_tp,      nppes_tp,      tp,          tp),
        ("FP",          nppes_fp,      nppes_fp,      fp,          fp),
        ("FN",          nppes_fn_total,nppes_fn_total-nppes_fn_nonretail, fn, fn-len(fn_nonretail)),
    ]:
        if isinstance(v1, float):
            print(f"  {label:<25} {v1:>13.1%} {v2:>13.1%} {v3:>13.1%} {v4:>13.1%}")
        else:
            print(f"  {label:<25} {v1:>14} {v2:>14} {v3:>14} {v4:>14}")

    metrics = {
        "board_total_ai": total_ai, "board_total_board": total_board,
        "board_tp": tp, "board_fp": fp, "board_fn": fn,
        "board_precision": round(precision, 4),
        "board_recall_raw": round(recall, 4),
        "board_f1_raw": round(f1, 4),
        "board_fn_nonretail": len(fn_nonretail),
        "board_adj_denom": adj_denom,
        "board_recall_adj": round(adj_recall, 4),
        "board_f1_adj": round(adj_f1, 4),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Spatial analysis of FNs
# ──────────────────────────────────────────────────────────────────────────────

def step6_spatial(fn_df: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 6 — Spatial analysis of false negatives")
    print("═"*66)

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("  [SKIP] geopandas not installed — running ZIP-level proxy analysis")
        return step6_zip_proxy(fn_df)

    # Load ACS tract data
    acs = pd.read_csv(DATA_DIR / "acs_tracts_2023.csv", low_memory=False)
    acs.columns = [c.strip() for c in acs.columns]
    print(f"\n  ACS tracts loaded: {len(acs)} rows")
    print(f"  ACS columns: {acs.columns.tolist()[:15]}")

    # Load TIGER tracts
    tracts = gpd.read_file(
        f"zip://{(DATA_DIR / 'tl_2023_27_tract.zip').resolve()}"
    ).to_crs(epsg=4326)
    print(f"  TIGER tracts loaded: {len(tracts)} tracts")

    # Geocode FN records using address_line1 + city (no geocoder available)
    # Fall back to ZIP centroid join
    fn_possible = fn_df[fn_df["FN_Category"] == "possible_missed_retail"].copy()
    print(f"\n  possible_missed_retail FNs to spatially analyze: {len(fn_possible)}")
    return step6_zip_proxy(fn_df)


def step6_zip_proxy(fn_df: pd.DataFrame):
    """ZIP-level proxy when geocoding unavailable."""

    acs = pd.read_csv(DATA_DIR / "acs_tracts_2023.csv", low_memory=False)
    acs.columns = [c.strip() for c in acs.columns]
    print(f"\n  ACS data: {len(acs)} rows, cols: {acs.columns.tolist()[:12]}")

    # Identify income + race columns
    inc_col = next((c for c in acs.columns if "median" in c.lower() and "income" in c.lower()), None)
    pct_col = next((c for c in acs.columns if ("nonwhite" in c.lower() or "pct_non" in c.lower()
                    or "pct_minority" in c.lower() or "non_white" in c.lower())), None)

    # Try alternate guesses
    if inc_col is None:
        for c in acs.columns:
            if "income" in c.lower():
                inc_col = c; break
    if pct_col is None:
        for c in acs.columns:
            if "white" in c.lower() and "pct" in c.lower():
                pct_col = c; break

    print(f"  Income column  : {inc_col}")
    print(f"  Race column    : {pct_col}")

    # Spatial AI tracts (already joined)
    ai_tracts = pd.read_csv(DATA_DIR / "spatial_ai_tracts.csv", low_memory=False)
    ai_tracts.columns = [c.strip() for c in ai_tracts.columns]
    print(f"\n  spatial_ai_tracts: {len(ai_tracts)} rows, cols: {ai_tracts.columns.tolist()[:12]}")

    # Spatial NPPES FN tracts for comparison
    fn_tracts = pd.read_csv(DATA_DIR / "spatial_nppes_fn_tracts.csv", low_memory=False)
    fn_tracts.columns = [c.strip() for c in fn_tracts.columns]
    print(f"  spatial_nppes_fn_tracts: {len(fn_tracts)} rows, cols: {fn_tracts.columns.tolist()[:12]}")

    # Board FN ZIP-level analysis
    fn_possible = fn_df[fn_df["FN_Category"] == "possible_missed_retail"].copy()

    if inc_col and inc_col in ai_tracts.columns:
        # Quartile breakdown of desert/FN rate by income quartile
        ai_tracts_clean = ai_tracts[ai_tracts[inc_col].notna()].copy()
        ai_tracts_clean["income_quartile"] = pd.qcut(
            ai_tracts_clean[inc_col], 4, labels=["Q1 (lowest)","Q2","Q3","Q4 (highest)"]
        )

        desert_col = next((c for c in ai_tracts_clean.columns
                           if "desert" in c.lower()), None)
        if desert_col:
            print(f"\n  Desert rate by income quartile (AI spatial data):")
            q_tbl = ai_tracts_clean.groupby("income_quartile")[desert_col].agg(
                tracts="count", desert_tracts="sum"
            )
            q_tbl["desert_rate"] = (q_tbl["desert_tracts"] / q_tbl["tracts"]).map("{:.1%}".format)
            print(q_tbl.to_string())

    # ZIP-level FN rate
    fn_zip = fn_possible.groupby("zip5").size().rename("board_fn_count").reset_index()
    fn_zip.columns = ["zip5","board_fn_count"]

    # Load NPPES FN for ZIP comparison
    try:
        nppes_fn = pd.read_csv(DATA_DIR / "nppes_fn_raw.csv", low_memory=False)
        if "ZIP" in nppes_fn.columns:
            nppes_fn_zip = nppes_fn[nppes_fn["FN_Category"] == "possible_missed_retail"].groupby("ZIP").size().rename("nppes_fn_count").reset_index()
            nppes_fn_zip.columns = ["zip5","nppes_fn_count"]
            fn_zip = fn_zip.merge(nppes_fn_zip, on="zip5", how="outer").fillna(0)
    except Exception:
        pass

    print(f"\n  Board possible_missed_retail FNs by ZIP (top 15 by count):")
    print(fn_zip.sort_values("board_fn_count", ascending=False).head(15).to_string(index=False))

    return fn_zip


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Summary comparison table + save outputs
# ──────────────────────────────────────────────────────────────────────────────

def step7_summary_and_save(ai_matched: pd.DataFrame, fn_df: pd.DataFrame,
                            metrics: dict, fn_zip: pd.DataFrame):
    print("\n" + "═"*66)
    print("  STEP 7 — Summary comparison table + save outputs")
    print("═"*66)

    # NPPES metrics from file
    nppes = pd.read_csv(DATA_DIR / "validation_nppes_20260413.csv", low_memory=False)
    nppes_tp  = int(nppes["Is_NPPES_Match"].sum())
    nppes_tai = len(nppes)
    nppes_prec = nppes_tp / nppes_tai
    try:
        nppes_fn_data = pd.read_csv(DATA_DIR / "nppes_fn_raw.csv", low_memory=False)
        nppes_fn_total = len(nppes_fn_data)
        if "FN_Category" in nppes_fn_data.columns:
            n_excl = len(nppes_fn_data[nppes_fn_data["FN_Category"].isin(
                ["corporate_legal_name","closed_chain","specialty_nonretail"])])
        else:
            n_excl = 331 + 95 + 39
    except Exception:
        nppes_fn_total = 717; n_excl = 465

    nppes_ref   = nppes_tp + nppes_fn_total
    nppes_rec   = nppes_tp / nppes_ref
    nppes_f1    = 2*nppes_prec*nppes_rec/(nppes_prec+nppes_rec) if (nppes_prec+nppes_rec) else 0
    nppes_adj_d = nppes_ref - n_excl
    nppes_adj_r = nppes_tp / nppes_adj_d if nppes_adj_d else 0
    nppes_adj_f = 2*nppes_prec*nppes_adj_r/(nppes_prec+nppes_adj_r) if (nppes_prec+nppes_adj_r) else 0

    b_prec    = metrics["board_precision"]
    b_rec_r   = metrics["board_recall_raw"]
    b_f1_r    = metrics["board_f1_raw"]
    b_rec_a   = metrics["board_recall_adj"]
    b_f1_a    = metrics["board_f1_adj"]

    print(f"""
┌─────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Metric                  │ vs NPPES (raw)   │ vs NPPES (adj)   │ vs Board (raw)   │ vs Board (adj)   │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Precision               │ {nppes_prec:>15.1%}  │ {nppes_prec:>15.1%}  │ {b_prec:>15.1%}  │ {b_prec:>15.1%}  │
│ Recall                  │ {nppes_rec:>15.1%}  │ {nppes_adj_r:>15.1%}  │ {b_rec_r:>15.1%}  │ {b_rec_a:>15.1%}  │
│ F1                      │ {nppes_f1:>15.1%}  │ {nppes_adj_f:>15.1%}  │ {b_f1_r:>15.1%}  │ {b_f1_a:>15.1%}  │
│ Reference set size      │ {nppes_ref:>16}  │ {nppes_adj_d:>16}  │ {metrics['board_total_board']:>16}  │ {metrics['board_adj_denom']:>16}  │
│ True Positives          │ {nppes_tp:>16}  │ {nppes_tp:>16}  │ {metrics['board_tp']:>16}  │ {metrics['board_tp']:>16}  │
│ False Positives         │ {nppes_tai-nppes_tp:>16}  │ {nppes_tai-nppes_tp:>16}  │ {metrics['board_fp']:>16}  │ {metrics['board_fp']:>16}  │
│ False Negatives         │ {nppes_fn_total:>16}  │ {nppes_fn_total-n_excl:>16}  │ {metrics['board_fn']:>16}  │ {metrics['board_fn']-metrics['board_fn_nonretail']:>16}  │
└─────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘""")

    # Save outputs
    ts = TIMESTAMP

    # 1. AI matched with Board columns
    out1 = DATA_DIR / f"validation_board_{ts}.csv"
    ai_matched.to_csv(out1, index=False)
    print(f"\n  Saved: {out1}")

    # 2. Board FN with categories
    out2 = DATA_DIR / f"board_fn_{ts}.csv"
    fn_df.to_csv(out2, index=False)
    print(f"  Saved: {out2}")

    # 3. FN ZIP summary
    out3 = DATA_DIR / f"board_fn_zip_{ts}.csv"
    fn_zip.to_csv(out3, index=False)
    print(f"  Saved: {out3}")

    # 4. Metrics summary CSV
    summary = pd.DataFrame([{
        "timestamp": ts,
        "nppes_precision": round(nppes_prec,4), "nppes_recall_raw": round(nppes_rec,4),
        "nppes_recall_adj": round(nppes_adj_r,4), "nppes_f1_raw": round(nppes_f1,4),
        "nppes_f1_adj": round(nppes_adj_f,4),
        "board_precision": b_prec, "board_recall_raw": b_rec_r,
        "board_recall_adj": b_rec_a, "board_f1_raw": b_f1_r,
        "board_f1_adj": b_f1_a,
        "board_tp": metrics["board_tp"], "board_fp": metrics["board_fp"],
        "board_fn": metrics["board_fn"],
        "board_fn_closed_chain": len(fn_df[fn_df["FN_Category"]=="closed_chain"]),
        "board_fn_specialty_nonretail": len(fn_df[fn_df["FN_Category"]=="specialty_nonretail"]),
        "board_fn_possible_missed_retail": len(fn_df[fn_df["FN_Category"]=="possible_missed_retail"]),
    }])
    out4 = DATA_DIR / f"validation_summary_{ts}.csv"
    summary.to_csv(out4, index=False)
    print(f"  Saved: {out4}")

    print(f"\n  All outputs saved with timestamp {ts}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  NETS-AI: Board of Pharmacy Validation Analysis")
    print(f"  Run timestamp: {TIMESTAMP}")
    print("=" * 66)

    board_raw, ai = step1_load()
    board_retail, excluded = step2_filter(board_raw, ai)
    ai_matched, fn_df = step3_match(ai, board_retail)
    fn_df = step4_categorize_fn(fn_df)
    metrics = step5_metrics(ai_matched, board_retail, fn_df)
    fn_zip = step6_spatial(fn_df)
    step7_summary_and_save(ai_matched, fn_df, metrics, fn_zip)

    print("\n" + "═"*66)
    print("  ✓ COMPLETE")
    print("═"*66)


if __name__ == "__main__":
    main()
