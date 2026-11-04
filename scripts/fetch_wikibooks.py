"""Download Wikibooks pages (CC BY-SA) as text for the ``other`` class.

Good for recipes. The ``other`` class must be varied, so also use
fetch_manifest.py for manuals, forms, flyers, meeting notes, policies, etc.

    python scripts/fetch_wikibooks.py --contact you@uni.hu -n 40 --category Category:Recipes
    python scripts/fetch_wikibooks.py --contact you@uni.hu -n 20 --category "Category:Cookbook" --label other
"""
from __future__ import annotations

import argparse
import json
import random
from urllib.parse import quote, urlencode

from common import DEFAULT_FORMATS, Store, http_get, pick_format

API = "https://en.wikibooks.org/w/api.php"


def api(params: dict, ua: str) -> dict:
    return json.loads(http_get(f"{API}?{urlencode({**params, 'format': 'json'})}", ua, pause=0.5))


def members(category: str, ua: str, cap: int) -> list[str]:
    titles, cont = [], {}
    while len(titles) < cap:
        data = api({"action": "query", "list": "categorymembers", "cmtitle": category,
                    "cmlimit": 50, "cmtype": "page", **cont}, ua)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    return titles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--category", action="append", required=True, help="repeatable, e.g. Category:Recipes")
    ap.add_argument("--label", default="other", choices=["other", "report", "letter"])
    ap.add_argument("--formats", default=DEFAULT_FORMATS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ua = f"fileorg-thesis-dataset/0.1 (mailto:{args.contact})"
    rng = random.Random(args.seed)
    titles = []
    for cat in args.category:
        titles += members(cat, ua, 400)
    rng.shuffle(titles)

    store, saved = Store(), 0
    for title in titles:
        if saved >= args.n:
            break
        data = api({"action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "redirects": 1}, ua)
        page = next(iter(data["query"]["pages"].values()))
        text = (page.get("extract") or "").strip()
        url = f"https://en.wikibooks.org/wiki/{quote(title.replace(' ', '_'))}"
        if len(text) < 600:
            Store.exclude(url, "too short")
            continue
        stem = "".join(ch if ch.isalnum() else "_" for ch in title)[:60]
        ok = store.save_text(args.label, stem, text, pick_format(rng, args.formats),
                             url, "CC BY-SA 4.0 (Wikibooks contributors)")
        if ok:
            saved += 1
            print(f"[{saved}/{args.n}] {ok.name}")
    print(f"saved {saved}; {args.label} now has {store.count(args.label)} files")


if __name__ == "__main__":
    main()