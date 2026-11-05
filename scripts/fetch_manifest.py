"""Download a hand-picked list of URLs (reports, manuals, forms, ...).

For ``report`` and ``other`` you pick open-licensed documents yourself
(government, NGO, company annual reports, user manuals, forms, flyers) and list
them in a CSV:

    category,url,licence
    report,https://example.org/annual-report-2023.pdf,CC BY 4.0
    other,https://example.org/user-manual.pdf,CC BY 4.0

    python scripts/fetch_manifest.py scripts/manifest.csv --contact you@uni.hu

Only PDF, DOCX and TXT are kept. Anything else is logged in data/excluded.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
import io
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from pathlib import Path

from common import CATEGORIES, Store, http_get


def detect_ext(url: str, data: bytes) -> str | None:
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                return ".docx" if "word/document.xml" in zf.namelist() else None
        except zipfile.BadZipFile:
            return None
    if url.lower().split("?")[0].endswith(".txt"):
        return ".txt"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--contact", required=True)
    args = ap.parse_args()

    ua = f"fileorg-thesis-dataset/0.1 (mailto:{args.contact})"
    store = Store()
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        cat, url, lic = row["category"].strip(), row["url"].strip(), row.get("licence", "").strip()
        if cat not in CATEGORIES:
            print(f"skip (unknown category {cat!r}): {url}")
            continue
        if not lic:
            Store.exclude(url, "no licence given in manifest")
            continue
        try:
            data = http_get(url, ua, pause=1)
        except (HTTPError, RuntimeError) as exc:
            Store.exclude(url, f"download failed: {exc}")
            continue
        ext = detect_ext(url, data)
        if ext is None:
            Store.exclude(url, "not a PDF, DOCX or TXT file")
            continue
        base = Path(unquote(urlparse(url).path)).stem or "download"
        name = f"{base[:60]}_{hashlib.sha1(url.encode()).hexdigest()[:6]}{ext}"
        ok = store.save(cat, name, data, url, lic)
        print(("saved   " if ok else "skipped ") + f"{cat}/{name}")


if __name__ == "__main__":
    main()