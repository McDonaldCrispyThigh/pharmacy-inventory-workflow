"""Reusable deterministic matchers for Board--Google record linkage.

The current analysis entry point, ``board_google_comparison.py``, supplies the fixed
evaluation frame and Board criterion.  This module provides the scoring and
assignment strategies:

    A. name_only_greedy   -- RapidFuzz token-sort + DBA normalization, greedy
    B. name_hungarian     -- same name score, optional 1-to-1 assignment
    C. name_addr          -- name + address-string signal, optional assignment
    D. name_geo           -- name + address + geocoded distance, optional assignment

Address parsing is deterministic (leading street number + normalized street name);
no LLM is used anywhere in the matching layer.

Evaluation metrics are intentionally not defined here: without independently
adjudicated pair labels, generated links support coverage/agreement summaries but
not precision, recall, accuracy, or F1 claims.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import token_sort_ratio
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_design import normalize_match_name as normalize

# ── Address parsing (deterministic) ──────────────────────────────────────────
_DIR = {"n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}
_SUFFIX = {"ave", "avenue", "st", "street", "rd", "road", "blvd", "boulevard",
           "dr", "drive", "ln", "lane", "way", "ct", "court", "pkwy", "parkway",
           "cir", "circle", "ter", "terrace", "hwy", "highway", "pl", "place",
           "trl", "trail", "sq", "square", "loop"}
_UNIT_RE = re.compile(
    r"(?:\b(?:suite|ste|unit|apt|bldg|building|floor|fl)\b|#)\s*.*$",
    re.I,
)


def parse_address_regex(addr: str) -> tuple[str, str]:
    """Lightweight regex parser: (street_number, normalized_street_name)."""
    if not isinstance(addr, str) or not addr.strip():
        return "", ""
    s = addr.split(",")[0].strip().lower()      # street portion before first comma
    s = _UNIT_RE.sub("", s)                      # drop suite/unit tails
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = s.split()
    if not toks:
        return "", ""
    num = toks[0] if toks[0].isdigit() else ""
    rest = toks[1:] if num else toks
    rest = [_DIR.get(t, t) for t in rest if t not in _SUFFIX]  # expand dirs, drop suffix
    return num, " ".join(rest)


try:
    import usaddress
    _HAVE_USADDR = True
except ImportError:
    _HAVE_USADDR = False

try:
    from postal.parser import parse_address as _libpostal_parse
    _HAVE_LIBPOSTAL = True
except Exception:
    _HAVE_LIBPOSTAL = False


def parse_address_usaddress(addr: str) -> tuple[str, str]:
    """usaddress CRF parser: keeps house number + street name + directionals, drops unit."""
    if not isinstance(addr, str) or not addr.strip():
        return "", ""
    try:
        tags, _ = usaddress.tag(addr)
    except Exception:
        return parse_address_regex(addr)
    num = tags.get("AddressNumber", "")
    parts = []
    for k in ("StreetNamePreDirectional", "StreetName", "StreetNamePostType",
              "StreetNamePostDirectional"):
        if k in tags:
            parts.append(tags[k].lower())
    street = " ".join(_DIR.get(t, t) for t in " ".join(parts).split() if t not in _SUFFIX)
    return num, street


def parse_address_libpostal(addr: str) -> tuple[str, str]:
    """libpostal parser: house_number + road."""
    if not isinstance(addr, str) or not addr.strip():
        return "", ""
    try:
        d = dict((v, k) for k, v in _libpostal_parse(addr))
    except Exception:
        return parse_address_regex(addr)
    num = d.get("house_number", "")
    road = d.get("road", "").lower()
    street = " ".join(_DIR.get(t, t) for t in road.split() if t not in _SUFFIX)
    return num, street


# The canonical workflow fixes the regex parser so that installing an optional
# package cannot silently change linkage results.  Alternative parsers remain
# available only for explicitly requested sensitivity runs.
_PARSER = parse_address_regex


def set_parser(which: str) -> None:
    global _PARSER
    _PARSER = {"regex": parse_address_regex,
               "usaddress": parse_address_usaddress,
               "libpostal": parse_address_libpostal}[which]


def parse_address(addr: str) -> tuple[str, str]:
    return _PARSER(addr)


def address_sim(ai_addr: str, board_addr: str) -> float:
    """0-100 address similarity. Street-number agreement is the dominant signal."""
    an, astreet = parse_address(ai_addr)
    bn, bstreet = parse_address(board_addr)
    if not astreet or not bstreet:
        return 0.0
    street = token_sort_ratio(astreet, bstreet)
    if an and bn:
        num = 100.0 if an == bn else 0.0
        return 0.6 * num + 0.4 * street          # number match dominates, street confirms
    return 0.4 * street                          # no number on one side: weak signal only


def name_sim(a: str, b: str) -> float:
    return token_sort_ratio(normalize(a), normalize(b))


def candidate_name_sim(candidate_names: object, board_name: str) -> float:
    """Return the best name score across aliases in one exact-location cluster."""
    if isinstance(candidate_names, (tuple, list, set, np.ndarray, pd.Series)):
        names = [str(value) for value in candidate_names if str(value).strip()]
    else:
        names = [str(candidate_names)]
    return max((name_sim(name, board_name) for name in names), default=0.0)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Data loading (reproduces validate_board filtering) ───────────────────────
# ── Scoring + assignment ─────────────────────────────────────────────────────
def score_matrix(ai: pd.DataFrame, board: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Return composite similarities on one common 0--100 scale.

    Each enabled component is itself bounded to 0--100.  Dividing the weighted
    sum by the sum of enabled weights makes a threshold comparable across the
    name-only, name-plus-address, and name-plus-address-plus-geography
    strategies.  Missing geographic coordinates contribute zero to the
    geographic component rather than changing the denominator.
    """
    component_weights = {
        "name": float(cfg.get("w_name", 0.0)),
        "addr": float(cfg.get("w_addr", 0.0)),
        "geo": float(cfg.get("w_geo", 0.0)),
    }
    if any(not np.isfinite(weight) or weight < 0.0 for weight in component_weights.values()):
        raise ValueError("Similarity weights must be finite and nonnegative")
    weight_sum = sum(component_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("At least one positive similarity weight is required")
    if component_weights["geo"] > 0.0:
        geo_scale = float(cfg.get("geo_scale", 300.0))
        if not np.isfinite(geo_scale) or geo_scale <= 0.0:
            raise ValueError("geo_scale must be finite and positive")

    n, m = len(ai), len(board)
    S = np.zeros((n, m))
    b_names = board["facility_name"].astype(str).tolist()
    b_addrs = board["address_line1"].astype(str).tolist()
    b_zips = board["zip5"].astype(str).tolist()
    b_lat = board["_lat"].tolist() if "_lat" in board else [np.nan] * m
    b_lon = board["_lon"].tolist() if "_lon" in board else [np.nan] * m
    for i, ar in ai.iterrows():
        az = str(ar.get("Zip_Code", ""))[:5]
        aname = ar.get("candidate_names", ar["Company"])
        aaddr = str(ar["Street_Address"])
        alat, alon = ar.get("Latitude"), ar.get("Longitude")
        for j in range(m):
            ns = candidate_name_sim(aname, b_names[j])
            sc = component_weights["name"] * ns
            if component_weights["addr"]:
                sc += component_weights["addr"] * address_sim(aaddr, b_addrs[j])
            if component_weights["geo"]:
                d = haversine_m(alat, alon, b_lat[j], b_lon[j])
                geo = 100.0 * np.exp(-d / geo_scale) if np.isfinite(d) else 0.0
                sc += component_weights["geo"] * geo
            sc /= weight_sum
            if az and b_zips[j] and az != b_zips[j]:
                sc *= 0.85
            S[i, j] = float(np.clip(sc, 0.0, 100.0))
    return S


def assign_greedy(S: np.ndarray, thr: float):
    cands = [(S[i, j], i, j) for i in range(S.shape[0]) for j in range(S.shape[1]) if S[i, j] >= thr]
    cands.sort(reverse=True)
    ua, ub, pairs = set(), set(), []
    for s, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); pairs.append((i, j, s))
    return pairs


def assign_hungarian(
    S: np.ndarray,
    thr: float,
    *,
    objective: str = "max_weight",
):
    """Assign qualified pairs while allowing either source record to remain unmatched.

    The threshold is applied *before* optimization.  Dummy rows and columns let
    the optimizer leave records unmatched, preventing an ineligible edge from
    occupying a real assignment and suppressing a qualified alternative.

    ``max_weight`` is the primary objective.  The
    ``max_cardinality_then_weight`` option is a declared sensitivity that first
    maximizes the number of qualified links and then their total similarity.
    """
    scores = np.asarray(S, dtype=float)
    if scores.ndim != 2:
        raise ValueError("S must be a two-dimensional score matrix")
    if not np.isfinite(scores).all():
        raise ValueError("S must contain only finite scores")
    threshold = float(thr)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("thr must be finite and nonnegative")
    if objective not in {"max_weight", "max_cardinality_then_weight"}:
        raise ValueError(f"Unknown assignment objective: {objective}")

    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    eligible = scores >= threshold
    if not eligible.any():
        return []

    real_utility = scores.copy()
    if objective == "max_cardinality_then_weight":
        max_score = max(float(np.max(np.abs(scores[eligible]))), 1.0)
        cardinality_bonus = (min(n, m) + 1) * max_score
        real_utility[eligible] += cardinality_bonus

    # A square (n + m) assignment gives every real row/column enough dummy
    # alternatives.  Forbidden real-real edges are worse than every dummy edge.
    size = n + m
    max_utility = max(float(np.max(np.abs(real_utility[eligible]))), 1.0)
    forbidden = -(size + 1) * max_utility
    utility = np.zeros((size, size), dtype=float)
    utility[:n, :m] = np.where(eligible, real_utility, forbidden)

    rows, columns = linear_sum_assignment(-utility)
    pairs = [
        (int(i), int(j), float(scores[i, j]))
        for i, j in zip(rows, columns)
        if i < n and j < m and eligible[i, j]
    ]
    return sorted(pairs, key=lambda pair: (pair[0], pair[1]))


# ── Metrics ──────────────────────────────────────────────────────────────────
