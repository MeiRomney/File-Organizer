"""Download World Bank reports into data/raw/report/ (more variety than GAO alone).

Uses the public Documents & Reports API (https://search.worldbank.org/api/v2/wds).
Several search terms are used so the reports differ in topic and style.
Non-PDF formats are made from the PDF text, so the class gets a format mix.

    python scripts/fetch_worldbank.py --contact you@uni.hu -n 40
    python scripts/fetch_worldbank.py --contact you@uni.hu --dry-run      # list, do not download

Licence: World Bank documents carry their own terms (many are CC BY 3.0 IGO,
not all). The script records "check terms" in sources.csv: open the document's
page and confirm before you publish or share any file. Do not commit the files.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse

from common import DEFAULT_FORMATS, Store, http_get, pdf_to_text, pick_format

API = "https://search.worldbank.org/api/v2/wds"
TERMS = ["annual report", "economic update", "evaluation report", "status report",
         "country report", "progress report", "sector assessment", "review"]
DEFAULT_TYPES = ("Report,Annual Report,Economic & Sector Work,Implementation Completion Report,"
                 "Public Expenditure Review,IEG Evaluation")
LICENCE = "World Bank Documents & Reports (terms per document: check before sharing)"
MAX_BYTES = 25 * 2 ** 20


def candidates(payload: dict, types: set[str]) -> list[dict]:
    """Pick English documents of the wanted types that have a PDF link."""
    out = []
    for doc in (payload.get("documents") or {}).values():
        if not isinstance(doc, dict) or not doc.get("pdfurl"):
            continue
        if doc.get("lang") not in (None, "English") or (types and doc.get("docty") not in types):
            continue
        out.append(doc)
    return out


def limit_series(docs: list[dict], cap: int) -> list[dict]:
    """Keep at most ``cap`` documents per series (same first three title words), so no series dominates."""
    counts: Counter = Counter()
    out = []
    for d in docs:
        words = re.sub(r"[^a-z]+", " ", (d.get("display_title") or "").lower()).split()
        key = " ".join(words[:3])
        if counts[key] < cap:
            counts[key] += 1
            out.append(d)
    return out


def search(term: str, rows: int, offset: int, ua: str) -> dict:
    params = {"format": "json", "qterm": term, "rows": rows, "os": offset,
              "fl": "docty,pdfurl,display_title,lang,docdt"}
    return json.loads(http_get(f"{API}?{urlencode(params)}", ua, pause=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--types", default=DEFAULT_TYPES, help="comma-separated document types ('' = any type)")
    ap.add_argument("--formats", default="pdf:0.4,docx:0.3,txt:0.3")
    ap.add_argument("--per-series", type=int, default=2, help="max reports per series (keeps the class varied)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ua = f"fileorg-thesis-dataset/0.1 (mailto:{args.contact})"
    rng = random.Random(args.seed)
    types = {t.strip() for t in args.types.split(",") if t.strip()}

    pool, seen_types = [], Counter()
    for term in TERMS:
        try:
            payload = search(term, 100, 0, ua)
        except (HTTPError, RuntimeError, ValueError) as exc:
            print(f"search failed for {term!r}: {exc}")
            continue
        for doc in (payload.get("documents") or {}).values():
            if isinstance(doc, dict) and doc.get("docty"):
                seen_types[doc["docty"]] += 1
        found = candidates(payload, types)
        print(f"{term!r}: {len(found)} usable documents")
        pool += found
    print("document types seen:", dict(seen_types.most_common(12)))

    unique = {d["pdfurl"]: d for d in pool}
    pool = list(unique.values())
    rng.shuffle(pool)
    pool = limit_series(pool, args.per_series)
    if not pool:
        print("Nothing matched. Adjust --types to one of the document types printed above.")
        return
    if args.dry_run:
        for d in pool[:args.n]:
            print(f"{d.get('docty')}: {d.get('display_title', '')[:70]}  {d['pdfurl']}")
        return

    store, saved, failures = Store(), 0, 0
    for doc in pool:
        if saved >= args.n:
            break
        if failures >= 5:
            print("5 downloads in a row failed: check your connection (try curl.exe -L -I <url>) and rerun.")
            break
        url = doc["pdfurl"].replace("http://", "https://", 1)
        print(f"downloading {url[-70:]}", flush=True)
        try:
            data = http_get(url, ua, pause=2, timeout=120, retries=2)
        except (HTTPError, RuntimeError) as exc:
            failures += 1
            Store.exclude(url, f"download failed: {exc}")
            continue
        failures = 0
        if not data.startswith(b"%PDF") or len(data) > MAX_BYTES:
            Store.exclude(url, "not a PDF or larger than 25 MB")
            continue
        stem, fmt = Path(urlparse(url).path).stem[:60], pick_format(rng, args.formats)
        if fmt == "pdf":
            ok = store.save("report", f"{stem}.pdf", data, url, LICENCE)
        else:
            try:
                text = pdf_to_text(data)
                if len(text) < 3000:
                    Store.exclude(url, "too little extractable text (scanned?)")
                    continue
                ok = store.save_text("report", stem, text, fmt, url + " (text converted)", LICENCE)
            except Exception as exc:  # noqa: BLE001
                Store.exclude(url, f"conversion failed: {type(exc).__name__}: {exc}")
                continue
        if ok:
            saved += 1
            print(f"[{saved}/{args.n}] {ok.name}")
    print(f"saved {saved}; report now has {store.count('report')} files")


if __name__ == "__main__":
    main()