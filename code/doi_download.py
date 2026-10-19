"""Resolve DOIs to open-access PDFs, verify each one, and write a manifest.

Usage
-----
    python code/doi_download.py 10.1377/hlthaff.2013.1397 -o References/
    python code/doi_download.py dois.txt                  -o References/
    python code/doi_download.py refs.bib                  -o References/
    python code/doi_download.py refs.csv                  -o References/ -m manifest.csv

Input may be a bare DOI, a .txt of one DOI per line, a .bib, or a .csv with a
`doi` column (optionally also `title`, `authors_crossref`, `venue`, `year`).

Why the verification step exists
--------------------------------
Checking that a download starts with `%PDF` only proves it is a PDF, not that it
is the *right* PDF. Keyword-search fallbacks in general-purpose paper tooling were
observed on this corpus returning a valid PDF of an entirely different article 3
times out of 4. Every candidate here is therefore opened and matched against the
DOI/title it claims to be; anything that fails is discarded rather than saved
under a misleading filename.

Only lawful open-access routes are used: Unpaywall, Europe PMC, PMC, arXiv.
No Sci-Hub. Subscription-only articles are reported as failures, not bypassed.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF is required for verification: pip install PyMuPDF")

UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")  # Unpaywall asks for a contact e-mail; set it in the environment

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

ILLEGAL = r'[<>:"/\\|?*\x00-\x1f]'
STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
        "a", "an", "of", "in", "on", "to", "by", "as", "at", "is", "its", "it",
        "using", "use", "based"}
DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9<>\[\]]+')


# ---------------------------------------------------------------- input parsing

def normalize_doi(raw: str) -> str:
    """Strip doi:, https://doi.org/, trailing punctuation."""
    s = (raw or "").strip()
    s = re.sub(r'^\s*(doi:\s*|https?://(dx\.)?doi\.org/)', '', s, flags=re.I)
    m = DOI_RE.search(s)
    return m.group(0).rstrip('.,;)') if m else ''


def load_records(src: str) -> list[dict]:
    """Return [{doi, title, authors, venue, year}, ...] from a DOI or a file."""
    p = Path(src)
    if not p.exists():
        doi = normalize_doi(src)
        if not doi:
            sys.exit(f"Not a DOI and not a file: {src}")
        return [{"doi": doi, "title": "", "authors": "", "venue": "", "year": ""}]

    text = p.read_text(encoding="utf-8-sig", errors="replace")

    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        out = []
        for r in rows:
            key = next((k for k in r if k and k.lower().strip() == "doi"), None)
            doi = normalize_doi(r.get(key, "")) if key else ""
            if doi:
                out.append({
                    "doi": doi,
                    "title": r.get("title", ""),
                    "authors": r.get("authors_crossref") or r.get("authors", ""),
                    "venue": r.get("venue", ""),
                    "year": r.get("year", ""),
                })
        return out

    if p.suffix.lower() == ".bib":
        out = []
        for entry in re.split(r'(?=@\w+\s*\{)', text):
            doi = normalize_doi(
                (re.search(r'\bdoi\s*=\s*[{"]([^}"]+)', entry, re.I) or [None, ""])[1])
            if not doi:
                continue
            title = (re.search(r'\btitle\s*=\s*[{"](.+?)[}"]\s*,', entry, re.I | re.S)
                     or [None, ""])[1]
            out.append({"doi": doi, "title": re.sub(r'[{}]', '', title).strip(),
                        "authors": "", "venue": "", "year": ""})
        return out

    # plain text: one DOI per line
    out = []
    for line in text.splitlines():
        doi = normalize_doi(line)
        if doi:
            out.append({"doi": doi, "title": "", "authors": "", "venue": "", "year": ""})
    return out


# ------------------------------------------------------------------- utilities

def safe_name(title: str, doi: str, maxlen: int = 130) -> str:
    """Filename from the title, falling back to a DOI slug."""
    t = unicodedata.normalize("NFC", (title or "").strip())
    t = re.sub(ILLEGAL, " ", t)
    t = re.sub(r"\s+", " ", t).strip().rstrip(". ")
    if not t:
        t = re.sub(ILLEGAL, "_", doi).replace("/", "_")
    if len(t) > maxlen:
        t = t[:maxlen].rsplit(" ", 1)[0].rstrip(". ")
    return t or "untitled"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s)


def _tokens(s: str) -> list[str]:
    return [w for w in _norm(s).split() if len(w) > 2 and w not in STOP]


def fetch(url: str, timeout: int = 50) -> bytes | None:
    """GET a URL and return the body only if it is actually a PDF."""
    try:
        r = SESSION.get(url, timeout=timeout, allow_redirects=True, stream=True)
        if r.status_code != 200:
            return None
        buf, total = [], 0
        for chunk in r.iter_content(65536):
            buf.append(chunk)
            total += len(chunk)
            if total > 90 * 1024 * 1024:
                return None
        body = b"".join(buf)
        return body if body[:4] == b"%PDF" else None
    except Exception:
        return None


def verify(pdf: bytes, doi: str, title: str) -> tuple[bool, str]:
    """Open the PDF and confirm it is the article we asked for.

    A correct `%PDF` header is not evidence of identity. Accept only if the DOI
    appears in the first three pages, or >=70% of the title's content words do.
    """
    try:
        doc = fitz.open(stream=pdf, filetype="pdf")
    except Exception:
        return False, "unreadable"
    try:
        if doc.page_count == 0:
            return False, "empty"
        text = " ".join(doc[i].get_text() for i in range(min(3, doc.page_count)))
    finally:
        doc.close()

    if doi and doi.lower().replace(" ", "") in text.lower().replace(" ", ""):
        return True, "doi"
    if not title:
        return False, "no-title-to-check"
    want = set(_tokens(title))
    if not want:
        return False, "no-signal"
    ratio = sum(1 for w in want if w in _norm(text)) / len(want)
    return (ratio >= 0.70), (f"title{ratio:.2f}" if ratio >= 0.70 else f"mismatch{ratio:.2f}")


# ------------------------------------------------------------ OA resolution

def crossref_meta(doi: str) -> dict:
    """Backfill title/authors/venue/year when the input had only a DOI."""
    try:
        r = SESSION.get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
                        timeout=30)
        if r.status_code != 200:
            return {}
        m = r.json()["message"]
        auth = "; ".join(a.get("family", "") for a in (m.get("author") or [])[:6])
        return {
            "title": (m.get("title") or [""])[0],
            "authors": auth,
            "venue": (m.get("container-title") or [""])[0],
            "year": str((m.get("issued", {}).get("date-parts") or [[""]])[0][0] or ""),
        }
    except Exception:
        return {}


def unpaywall(doi: str) -> tuple[list[str], str | None, str]:
    """Return (pdf_urls, pmcid, oa_status)."""
    try:
        r = SESSION.get(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}",
                        params={"email": UNPAYWALL_EMAIL}, timeout=30)
        if r.status_code != 200:
            return [], None, f"unpaywall_http_{r.status_code}"
        data = r.json()
    except Exception:
        return [], None, "unpaywall_error"

    urls, seen, pmcid = [], set(), None
    locations = ([data["best_oa_location"]] if data.get("best_oa_location") else [])
    locations += data.get("oa_locations") or []
    for loc in locations:
        m = re.search(r"pubmedcentral\.nih\.gov:(\d+)", loc.get("pmh_id") or "")
        if m and not pmcid:
            pmcid = "PMC" + m.group(1)
        url = loc.get("url_for_pdf")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls, pmcid, ("oa" if data.get("is_oa") else "closed")


def europepmc(doi: str) -> list[str]:
    """Exact-DOI lookup only — never a keyword search, which returns wrong papers."""
    try:
        r = SESSION.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                        params={"query": f'DOI:"{doi}"', "format": "json",
                                "resultType": "core", "pageSize": 1}, timeout=30)
        hits = r.json().get("resultList", {}).get("result", [])
        if not hits or (hits[0].get("doi") or "").lower() != doi.lower():
            return []
        rec, urls = hits[0], []
        if rec.get("pmcid"):
            urls.append(f"https://europepmc.org/api/getPdf?pmcid={rec['pmcid']}")
            urls.append("https://www.ebi.ac.uk/europepmc/webservices/rest/"
                        f"{rec['pmcid']}/fullTextPDF")
        for ft in (rec.get("fullTextUrlList", {}) or {}).get("fullTextUrl", []):
            if ft.get("documentStyle") == "pdf" and ft.get("url"):
                urls.append(ft["url"])
        return urls
    except Exception:
        return []


def pmc(pmcid: str | None) -> list[str]:
    if not pmcid:
        return []
    return [f"https://europepmc.org/api/getPdf?pmcid={pmcid}",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"]


def arxiv(title: str) -> list[str]:
    """Title-exact match only, to avoid pulling an unrelated preprint."""
    if not title:
        return []
    try:
        r = SESSION.get("http://export.arxiv.org/api/query",
                        params={"search_query": f'ti:"{title[:110]}"',
                                "max_results": 1}, timeout=30)
        ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", r.text)
        titles = re.findall(r"<title>(.*?)</title>", r.text, re.S)
        if ids and len(titles) > 1:
            got, want = set(_tokens(titles[1])), set(_tokens(title))
            if want and len(got & want) / len(want) >= 0.85:
                return [f"https://arxiv.org/pdf/{ids[0]}"]
    except Exception:
        pass
    return []


# ------------------------------------------------------------------------ main

def resolve_one(rec: dict, outdir: Path, overwrite: bool) -> dict:
    doi, title = rec["doi"], rec.get("title", "")
    if not title:
        rec.update({k: v for k, v in crossref_meta(doi).items() if v})
        title = rec.get("title", "")

    base = safe_name(title, doi)
    pdf_path, txt_path = outdir / f"{base}.pdf", outdir / f"{base}.txt"

    if pdf_path.exists() and not overwrite:
        return {**rec, "result": "skip_exists", "file": pdf_path.name,
                "why": "", "url": ""}

    urls, pmcid, oa = unpaywall(doi)
    candidates = ([("unpaywall", u) for u in urls]
                  + [("europepmc", u) for u in europepmc(doi)]
                  + [("pmc", u) for u in pmc(pmcid)]
                  + [("arxiv", u) for u in arxiv(title)])

    rejected = 0
    for source, url in candidates:
        body = fetch(url)
        if not body:
            continue
        ok, reason = verify(body, doi, title)
        if ok:
            pdf_path.write_bytes(body)
            if txt_path.exists():
                txt_path.unlink()
            return {**rec, "result": "pdf", "file": pdf_path.name,
                    "why": f"{source}:{reason}", "url": url}
        rejected += 1

    note = (f"{rejected} candidate PDF(s) fetched but failed identity verification"
            if rejected else f"no open-access location ({oa})")
    txt_path.write_text(
        f"{rec.get('authors','')}. {title}. {rec.get('venue','')}. "
        f"{rec.get('year','')}. doi: {doi}.\n"
        f"[No verified open-access PDF. {note}. "
        f"Resolve at https://doi.org/{doi} or via your library.]\n",
        encoding="utf-8")
    return {**rec, "result": "stub", "file": txt_path.name, "why": note, "url": ""}


def main() -> None:
    # Titles routinely contain U+2010 hyphens, en dashes and curly quotes, which
    # a cp1252 Windows console cannot encode -- printing one kills the whole run
    # mid-corpus. Force UTF-8 and degrade gracefully rather than raise.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="A DOI, or a .txt / .bib / .csv containing DOIs")
    ap.add_argument("-o", "--out", default="References",
                    help="Directory for PDFs and stubs (default: References)")
    ap.add_argument("-m", "--manifest", default="",
                    help="Manifest CSV path (default: <out>/download_manifest.csv)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if the PDF already exists")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Seconds between records (default: 0.3)")
    args = ap.parse_args()

    records = load_records(args.source)
    if not records:
        sys.exit("No DOIs found in input.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest) if args.manifest else outdir / "download_manifest.csv"

    results, counts = [], {"pdf": 0, "stub": 0, "skip_exists": 0}
    for i, rec in enumerate(records, 1):
        row = resolve_one(rec, outdir, args.overwrite)
        results.append(row)
        counts[row["result"]] = counts.get(row["result"], 0) + 1
        tag = {"pdf": "PDF ", "stub": "stub", "skip_exists": "SKIP"}[row["result"]]
        print(f"[{i}/{len(records)}] {tag} {row['why'][:46]:<46} {row['file'][:52]}",
              flush=True)
        time.sleep(args.delay)

    cols = ["doi", "title", "authors", "venue", "year", "result", "file", "why", "url"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\nverified PDFs: {counts['pdf']}   stubs: {counts['stub']}   "
          f"skipped: {counts['skip_exists']}   total: {len(records)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
