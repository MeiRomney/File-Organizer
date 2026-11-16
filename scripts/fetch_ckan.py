"""Download varied open-licensed documents from a CKAN open-data portal.

Built for the ``other`` class (manuals, guides, forms, meeting minutes, agendas,
flyers, policies, schedules, checklists), which must be as varied as possible.
Each search phrase is limited to a few files, so no single document type or
publisher dominates.

Only resources whose dataset licence is on the allow-list are kept, and only
PDF/DOCX/TXT files. The licence of each dataset is written to sources.csv.

    python scripts/fetch_ckan.py --contact you@uni.hu -n 60
    python scripts/fetch_ckan.py --contact you@uni.hu --dry-run
    python scripts/fetch_ckan.py --contact you@uni.hu --catalog https://data.gov.uk

Default portal: open.canada.ca (CKAN, Open Government Licence - Canada; only English
resources are kept). catalog.data.gov retired its CKAN API in 2026, so it no longer works.

Always spot-check the result. Labelling rule: if a file is ambiguous (for
example a policy report that looks like a ``report``), delete it and log it in
data/excluded.csv instead of keeping it as ``other``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, urlparse

from common import Store, http_get
from fetch_manifest import detect_ext

PHRASES = ["user manual", "user guide", "installation guide", "meeting minutes", "meeting agenda",
           "application form", "registration form", "flyer", "brochure", "policy and procedures",
           "handbook", "checklist", "schedule", "timetable", "newsletter", "fact sheet",
           "instructions", "recipe", "training manual", "event program"]
OPEN_LICENCES = {"cc-by", "cc-by-sa", "cc-zero", "odc-by", "odc-pddl", "us-pd", "other-pd", "other-open",
                 "uk-ogl", "ca-ogl-lgo", "ca-ogl-bc", "usgov-pd", "public-domain"}
WANTED = {"PDF", "DOCX", "TXT"}
MAX_BYTES = 25 * 2 ** 20


def resources(package: dict, allowed: set[str]) -> list[tuple[str, str, str]]:
    """(url, format, licence title) for every usable document resource of a dataset."""
    if (package.get("license_id") or "").lower() not in allowed:
        return []
    licence = package.get("license_title") or package.get("license_id") or ""
    out = []
    for res in package.get("resources") or []:
        url = (res.get("url") or "").strip()
        fmt = (res.get("format") or "").strip().upper()
        langs = res.get("language")
        if langs:
            langs = [str(x).lower() for x in (langs if isinstance(langs, list) else [langs])]
            if "en" not in langs:  # bilingual portals: keep English documents only
                continue
        if not url.lower().startswith("http"):
            continue
        if fmt in WANTED or Path(urlparse(url).path).suffix.upper().lstrip(".") in WANTED:
            out.append((url, fmt, licence))
    return out


def search(catalog: str, phrase: str, rows: int, ua: str) -> list[dict]:
    params = {"q": phrase, "fq": "res_format:PDF", "rows": rows}
    last: Exception | None = None
    for path in ("/api/3/action/package_search", "/api/action/package_search"):
        try:
            data = json.loads(http_get(f"{catalog.rstrip('/')}{path}?{urlencode(params)}", ua, pause=1))
            return data.get("result", {}).get("results", [])
        except HTTPError as exc:
            last = exc
    raise last  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=60, help="files to save")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--catalog", default="https://open.canada.ca/data", help="base URL of a CKAN portal")
    ap.add_argument("--label", default="other", choices=["other", "report", "letter"])
    ap.add_argument("--per-phrase", type=int, default=5, help="max files per search phrase (keeps the class varied)")
    ap.add_argument("--licences", default=",".join(sorted(OPEN_LICENCES)), help="allowed CKAN license_id values")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="list candidates, download nothing")
    args = ap.parse_args()

    ua = f"fileorg-thesis-dataset/0.1 (mailto:{args.contact})"
    rng = random.Random(args.seed)
    allowed = {x.strip().lower() for x in args.licences.split(",") if x.strip()}

    by_phrase: dict[str, list[tuple[str, str, str]]] = {}
    licences_seen: Counter = Counter()
    for phrase in PHRASES:
        try:
            packages = search(args.catalog, phrase, 50, ua)
        except (HTTPError, RuntimeError, ValueError) as exc:
            print(f"search failed for {phrase!r}: {exc}")
            continue
        found = []
        for pkg in packages:
            licences_seen[pkg.get("license_id") or "none"] += 1
            found += resources(pkg, allowed)
        rng.shuffle(found)
        by_phrase[phrase] = found
        print(f"{phrase!r}: {len(found)} candidate files")
    print("dataset licences seen:", dict(licences_seen.most_common(8)))

    # round-robin over the phrases so no phrase dominates
    queue, taken = [], Counter()
    seen_urls: set[str] = set()
    for i in range(args.per_phrase):
        for phrase, found in by_phrase.items():
            if i < len(found) and found[i][0] not in seen_urls:
                seen_urls.add(found[i][0])
                queue.append((phrase, *found[i]))
    if not queue:
        print("Nothing matched. Check the allowed licences above (--licences) or try another --catalog.")
        return
    if args.dry_run:
        for phrase, url, fmt, lic in queue[:args.n]:
            print(f"[{phrase}] {fmt:<4} {lic[:30]:<30} {url}")
        return

    store, saved = Store(), 0
    for phrase, url, fmt, lic in queue:
        if saved >= args.n:
            break
        try:
            data = http_get(url, ua, pause=1.5, timeout=120)
        except (HTTPError, RuntimeError) as exc:
            Store.exclude(url, f"download failed: {exc}")
            continue
        ext = detect_ext(url, data)
        if ext is None or len(data) > MAX_BYTES:
            Store.exclude(url, "not a real PDF/DOCX/TXT or larger than 25 MB")
            continue
        stem = (Path(unquote(urlparse(url).path)).stem or "download")[:60]
        name = f"{stem}_{hashlib.sha1(url.encode()).hexdigest()[:6]}{ext}"
        ok = store.save(args.label, name, data, f"{url} (query: {phrase})", lic)
        if ok:
            saved += 1
            print(f"[{saved}/{args.n}] {phrase}: {ok.name}")
    print(f"saved {saved}; {args.label} now has {store.count(args.label)} files")
    print("Spot-check the new files and delete anything ambiguous (log it in data/excluded.csv).")


if __name__ == "__main__":
    main()