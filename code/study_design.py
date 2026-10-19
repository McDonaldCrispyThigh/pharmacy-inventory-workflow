"""Side-effect-free constants and deterministic rules for the pharmacy study.

This module contains definitions that must be fixed before record linkage.  It has
no network calls, file writes, or import-time stdout changes, so analysis modules
can reuse the same study frame, Board eligibility screen, and name normalization.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


TARGET_ZIPS = frozenset(
    {
        "55401", "55402", "55403", "55404", "55405", "55406", "55407",
        "55408", "55409", "55410", "55411", "55412", "55413", "55414",
        "55415", "55454", "55455", "55101", "55102", "55103", "55104",
        "55105", "55106", "55107", "55108", "55116", "55117", "55118",
        "55119", "55130", "55113", "55126", "55421", "55422", "55423",
        "55424", "55425", "55426", "55427", "55428", "55429", "55430",
        "55431", "55432", "55433", "55434", "55435", "55436", "55437",
        "55438", "55439", "55441", "55442", "55443", "55444", "55445",
        "55446", "55447", "55448", "55369",
    }
)


# These rules are applied uniformly to every Board record before linkage.  They
# operationalize the access analysis as public-facing/community pharmacy access.
# The all-licensed Board inventory is retained as a sensitivity definition.
BOARD_ACCESS_EXCLUSION_RULES: tuple[tuple[str, str], ...] = (
    (
        "institutional_or_hospital",
        r"\bhospital\b|\bclinic\b|\bmedical\s+center\b|\bhealth\s+system\b|"
        r"\boutpatient\b|\ballina\b|\bfairview\b|\bhealthpartners\b|"
        r"\bhennepin\s+healthcare\b|\bm\s+health\b|\bva\b|\bveterans\b|"
        r"\buniversity\s+of\s+minnesota\b|\bchildren.?s\b|\babbott\b|"
        r"\bregions\b|\bnorth\s+memorial\b|\bpark\s+nicollet\b|"
        r"\bunited\s+hospital\b|\bst\.?\s+paul\s+regions\b|\baccredo\b|"
        r"\bacadia\b|\bmental\s+health\b|\bdrug\s+monitoring\b",
    ),
    (
        "noncommunity_service",
        r"\binfusion\b|\bhome\s+infusion\b|\biv\s+therapy\b|"
        r"\blong.?term\s+care\b|\bltc\b|\bmail.?order\b|"
        r"\bmail\s+service\b|\bnuclear\b|\bradio.?pharm\b|"
        r"\bcorrectional\b|\binstitutional\b|\bprison\b|\bhospice\b|"
        r"\brenal\b|\bdialysis\b|\bveterina\b|\banimal\b|\boncology\b|"
        r"\bchemotherapy\b|\bspecialty\s+infusion\b|\bcoram\b",
    ),
    (
        "compounding_only",
        r"\bcompounding\s*(?:only|pharmacy|center|lab)\b",
    ),
)


_PUNCT_RE = re.compile(r"[.,'\-#&/\\()@]")
_MATCH_DROP_WORDS = frozenset(
    {
        "pharmacy", "drug", "rx", "drugs", "store", "stores", "inc", "llc",
        "corp", "co", "the", "and", "dba", "ltd", "pharm", "phcy", "of",
        "corporation", "wholesale",
    }
)
_CORP_INDICATORS = frozenset(
    {"inc", "llc", "corp", "ltd", "corporation", "co", "pa"}
)

_SIGNATURE_DROP_WORDS = _MATCH_DROP_WORDS | frozenset(
    {
        "pharmacies", "company", "limited", "pllc", "pc", "number", "no",
        "branch", "location", "department", "dept", "minnesota", "mn",
    }
)
_DBA_RE = re.compile(r"\(?\bd\s*[./]?\s*b\s*[./]?\s*a\b\)?", re.I)


def _ascii_fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii")


def _operating_name(value: object) -> str:
    """Extract the public operating-name portion when a DBA form is present."""
    text = _ascii_fold(value).casefold().strip()
    pieces = _DBA_RE.split(text)
    if len(pieces) > 1:
        # Board strings occasionally append the legal entity after a semicolon.
        return pieces[-1].split(";", 1)[0].strip()

    parenthetical = re.search(r"\(([^)]+)\)\s*$", text)
    if parenthetical:
        prefix_tokens = set(re.sub(r"[^a-z\s]", " ", text[: parenthetical.start()]).split())
        if prefix_tokens & _CORP_INDICATORS:
            return parenthetical.group(1).strip()
    return text


def normalize_match_name(value: object) -> str:
    """Replicate the legacy DBA-aware linkage normalization without side effects."""
    text = _operating_name(value)
    text = re.sub(r"#[\w/]+", " ", text)
    text = _PUNCT_RE.sub(" ", text)
    tokens = [
        token
        for token in text.split()
        if token not in _MATCH_DROP_WORDS and not token.isdigit()
    ]
    result = " ".join(tokens)
    return result if result else _ascii_fold(value).casefold()


def normalize_name_signature(value: object, city: object = "") -> str:
    """Return a deterministic public-name signature with no chain dictionary.

    The transformation removes legal/generic pharmacy terms, branch numbers, and
    tokens naming the record's city.  Exact recurrence of the resulting signature
    across distinct addresses defines the repeated-name group.  It is a name-based
    recognizability construct and does not identify corporate ownership.
    """
    text = _operating_name(value)
    # Delete punctuation that occurs inside names before general tokenization:
    # Wal-Mart -> walmart, Sam's -> sams, Hy-Vee -> hyvee, L.L.C. -> llc.
    text = re.sub(r"[.'-]", "", text)
    text = re.sub(
        r"\b(?:store|location|branch|pharmacy)?\s*(?:no\.?|number|#)\s*\d+[a-z0-9/-]*\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"#[a-z0-9/-]+", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    city_tokens = {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", _ascii_fold(city).casefold()).split()
        if len(token) > 1
    }
    raw_tokens = [
        token
        for token in text.split()
        if token not in _SIGNATURE_DROP_WORDS
        and not token.isdigit()
    ]
    tokens = [token for token in raw_tokens if token not in city_tokens]
    # A pharmacy named only for its city remains a distinctive local singleton.
    # Falling back before city removal prevents unrelated city-named pharmacies
    # from collapsing into one artificial `unresolved` cluster.
    if not tokens:
        tokens = raw_tokens
    return " ".join(tokens) if tokens else "unresolved"


def normalize_legal_signature(value: object) -> str:
    """Return the legal-name side of a Board legal/DBA string."""
    text = _ascii_fold(value).casefold().strip()
    pieces = _DBA_RE.split(text)
    if len(pieces) > 1:
        text = pieces[0]
    else:
        parenthetical = re.search(r"\(([^)]+)\)\s*$", text)
        if parenthetical:
            prefix_tokens = set(
                re.sub(r"[^a-z\s]", " ", text[: parenthetical.start()]).split()
            )
            if prefix_tokens & _CORP_INDICATORS:
                text = text[: parenthetical.start()]
    text = re.sub(r"[.'-]", "", text)
    text = re.sub(r"#[a-z0-9/-]+", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [
        token
        for token in text.split()
        if token not in _SIGNATURE_DROP_WORDS and not token.isdigit()
    ]
    return " ".join(tokens) if tokens else "unresolved"


def board_exclusion_reason(name: object) -> str:
    """Return the first pre-linkage public-access exclusion rule, or an empty string."""
    text = _ascii_fold(name).casefold()
    for label, pattern in BOARD_ACCESS_EXCLUSION_RULES:
        if re.search(pattern, text, flags=re.I):
            return label
    return ""


def normalized_location_key(address: object, zip_code: object) -> str:
    """Normalize a source-local location key for recurrence counts."""
    address_text = re.sub(r"[^a-z0-9\s]", " ", _ascii_fold(address).casefold())
    address_text = " ".join(address_text.split())
    zip5 = re.sub(r"\D", "", str(zip_code))[:5]
    return f"{address_text}|{zip5}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
