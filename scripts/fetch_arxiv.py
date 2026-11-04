"""Download open-licensed arXiv papers into data/raw/academic_paper/.

Uses the arXiv OAI-PMH interface, which reports each paper's licence, and keeps
only Creative Commons papers. Follows arXiv's access rules: identifies itself,
sleeps 5 s between listing requests and 3 s between PDF downloads.

    python scripts/fetch_arxiv.py --contact you@uni.hu -n 100
    python scripts/fetch_arxiv.py --contact you@uni.hu -n 20 --sets cs,math --from 2024-03-01 --until 2024-03-07
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from urllib.error import HTTPError
from urllib.parse import urlencode

from common import DEFAULT_FORMATS, Store, http_get, pdf_to_text, pick_format

OAI_URL = "https://export.arxiv.org/oai2"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
AX = "{http://arxiv.org/OAI/arXiv/}"


def candidates(sets, start, end, per_set, ua):
    out = []
    for s in sets:
        params = {"verb": "ListRecords", "metadataPrefix": "arXiv",
                  "set": s, "from": start, "until": end}
        got = 0
        while params and got < per_set:
            xml = http_get(f"{OAI_URL}?{urlencode(params)}", ua, pause=5)
            root = ET.fromstring(xml)
            for rec in root.iter(OAI + "record"):
                meta = rec.find(f"{OAI}metadata/{AX}arXiv")
                if meta is None:
                    continue
                lic = meta.findtext(AX + "license") or ""
                if "creativecommons.org" not in lic:
                    continue
                out.append((meta.findtext(AX + "id"), lic, s))
                got += 1
            tok = root.find(f".//{OAI}resumptionToken")
            params = ({"verb": "ListRecords", "resumptionToken": tok.text}
                      if tok is not None and tok.text else None)
        print(f"set {s}: {got} CC-licensed candidates")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=100, help="papers to save")
    ap.add_argument("--contact", required=True, help="your e-mail, sent in the User-Agent")
    ap.add_argument("--sets", default="cs,math,q-bio,stat,econ", help="arXiv sets, comma separated")
    ap.add_argument("--from", dest="start", default="2024-01-01")
    ap.add_argument("--until", default="2024-01-31")
    ap.add_argument("--formats", default=DEFAULT_FORMATS,
                    help="format weights; non-pdf = text extracted from the PDF")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ua = f"fileorg-thesis-dataset/0.1 (mailto:{args.contact})"
    rng = random.Random(args.seed)
    sets = args.sets.split(",")
    pool = candidates(sets, args.start, args.until, max(5, 3 * args.n // len(sets)), ua)
    rng.shuffle(pool)

    store, saved = Store(), 0
    for pid, lic, s in pool:
        if saved >= args.n:
            break
        origin = f"https://arxiv.org/abs/{pid} (arXiv {s})"
        try:
            pdf = http_get(f"https://arxiv.org/pdf/{pid}", ua, pause=3)
        except (HTTPError, RuntimeError) as exc:
            Store.exclude(origin, f"download failed: {exc}")
            continue
        if not pdf.startswith(b"%PDF"):
            Store.exclude(origin, "not a PDF")
            continue
        stem, fmt = pid.replace("/", "_"), pick_format(rng, args.formats)
        if fmt == "pdf":
            ok = store.save("academic_paper", f"{stem}.pdf", pdf, origin, lic)
        else:
            text = pdf_to_text(pdf)
            if len(text) < 3000:
                Store.exclude(origin, "too little extractable text")
                continue
            ok = store.save_text("academic_paper", stem, text, fmt, origin + " (text converted)", lic)
        if ok:
            saved += 1
            print(f"[{saved}/{args.n}] {ok.name}")
    print(f"saved {saved} papers; academic_paper now has {store.count('academic_paper')} files")


if __name__ == "__main__":
    main()