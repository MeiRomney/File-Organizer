"""Download real contracts (exhibits to SEC filings) into data/raw/contract/.

Uses the EDGAR full-text search endpoint (the one behind efts.sec.gov search
UI, undocumented, so the JSON fields may change) and the normal Archives URLs.
SEC requires a descriptive User-Agent with contact details and at most 10
requests per second; this script does about 5.

    python scripts/fetch_edgar_contracts.py --contact "Mei Romney you@uni.hu" -n 100
"""
from __future__ import annotations

import argparse
import json
import random
import re
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlencode

from common import DEFAULT_FORMATS, Store, http_get, pick_format

SEARCH = "https://efts.sec.gov/LATEST/search-index"
QUERIES = ["non-disclosure agreement", "services agreement", "lease agreement",
           "employment agreement", "license agreement", "consulting agreement",
           "supply agreement", "distribution agreement"]
BLOCK = {"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(raw: bytes) -> str:
    parser = _Text()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def hits_for(query, start, end, offset, ua):
    params = {"q": f'"{query}"', "forms": "8-K,10-K,10-Q", "dateRange": "custom",
              "startdt": start, "enddt": end, "from": offset}
    data = json.loads(http_get(f"{SEARCH}?{urlencode(params)}", ua, pause=0.2))
    return data.get("hits", {}).get("hits", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--contact", required=True, help='"Your Name you@example.com" (SEC User-Agent rule)')
    ap.add_argument("--from", dest="start", default="2023-01-01")
    ap.add_argument("--until", default="2024-12-31")
    ap.add_argument("--formats", default=DEFAULT_FORMATS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ua = args.contact
    rng = random.Random(args.seed)
    per_query = -(-args.n // len(QUERIES)) + 3
    store, saved, seen_ids = Store(), 0, set()

    for query in QUERIES:
        got = 0
        for page in range(6):
            if got >= per_query or saved >= args.n:
                break
            try:
                hits = hits_for(query, args.start, args.until, page * 100, ua)
            except (HTTPError, RuntimeError, ValueError) as exc:
                print(f"search failed for {query!r}: {exc}")
                break
            if not hits:
                break
            rng.shuffle(hits)
            for hit in hits:
                if got >= per_query or saved >= args.n:
                    break
                src = hit.get("_source", {})
                if not str(src.get("file_type", "")).startswith("EX-10") or hit["_id"] in seen_ids:
                    continue
                seen_ids.add(hit["_id"])
                adsh, fname = hit["_id"].split(":", 1)
                cik = int(src["ciks"][0])
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fname}"
                try:
                    raw = http_get(url, ua, pause=0.2)
                except (HTTPError, RuntimeError) as exc:
                    Store.exclude(url, f"download failed: {exc}")
                    continue
                text = html_to_text(raw) if fname.lower().endswith((".htm", ".html")) else raw.decode("utf-8", "replace")
                head = text[:1500].lower()
                if len(text) < 4000:
                    Store.exclude(url, "too short")
                elif not re.search(r"agreement|contract|lease", head):
                    Store.exclude(url, "no agreement wording in the first 1500 characters")
                elif re.search(r"\bamendment\b", head[:400]):
                    Store.exclude(url, "amendment, not a full contract")
                else:
                    stem = f"{cik}_{adsh.replace('-', '')}_{fname.rsplit('.', 1)[0]}"
                    ok = store.save_text("contract", stem, text[:150_000],
                                         pick_format(rng, args.formats), url,
                                         "SEC EDGAR public filing")
                    if ok:
                        saved += 1
                        got += 1
                        print(f"[{saved}/{args.n}] {ok.name}")
    print(f"saved {saved} contracts; contract now has {store.count('contract')} files")


if __name__ == "__main__":
    main()