"""Shared helpers for the dataset scripts.

* Every saved file is hashed; exact duplicates (also against files already in
  data/raw) are skipped.
* Every saved file is logged in data/sources.csv (file, origin, licence, date).
* Files that were found but rejected are logged in data/excluded.csv.
* Text can be rendered as TXT, DOCX or PDF so each class gets a format mix.

Set FILEORG_ROOT to use a different project root (default: the repo root).
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import textwrap
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(os.environ.get("FILEORG_ROOT") or Path(__file__).resolve().parents[1])
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "data" / "sources.csv"
EXCLUDED = ROOT / "data" / "excluded.csv"

CATEGORIES = ("invoice", "receipt", "contract", "academic_paper",
              "cv_resume", "letter", "report", "other")
DEFAULT_FORMATS = "pdf:0.4,docx:0.3,txt:0.3"


# ------------------------------------------------------------------ network
def http_get(url: str, user_agent: str, *, retries: int = 4,
             timeout: int = 60, pause: float = 0.0) -> bytes:
    """GET with retry/backoff on 429/5xx. ``pause`` = polite delay afterwards."""
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if pause:
                time.sleep(pause)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                try:
                    wait = int(exc.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0
                time.sleep(min(wait or 5 * (attempt + 1), 120))
                last = exc
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


# ------------------------------------------------------------------ formats
def pick_format(rng: random.Random, spec: str) -> str:
    pairs = [p.split(":") for p in spec.split(",")]
    return rng.choices([a for a, _ in pairs], [float(b) for _, b in pairs])[0]


def render_docx(text: str) -> bytes:
    from docx import Document

    doc = Document()
    doc.core_properties.author = ""      # do not leave "python-docx" as author
    doc.core_properties.title = ""
    doc.core_properties.last_modified_by = ""
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def render_pdf(text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for setter in ("setAuthor", "setCreator", "setProducer", "setTitle"):
        if hasattr(c, setter):
            getattr(c, setter)("")
    _, height = A4
    y = height - 60
    c.setFont("Courier", 9)
    for raw in text[:60_000].splitlines():
        for line in textwrap.wrap(raw, 88, replace_whitespace=False) or [""]:
            if y < 60:
                c.showPage()
                c.setFont("Courier", 9)
                y = height - 60
            c.drawString(60, y, line.encode("latin-1", "replace").decode("latin-1"))
            y -= 12
    c.save()
    return buf.getvalue()


def render(text: str, fmt: str) -> bytes:
    if fmt == "txt":
        return text.encode("utf-8")
    if fmt == "docx":
        return render_docx(text)
    if fmt == "pdf":
        return render_pdf(text)
    raise ValueError(f"unknown format: {fmt}")


def pdf_to_text(data: bytes, max_pages: int = 60) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            parts.append("")
    return "\n".join(parts)


# -------------------------------------------------------------------- store
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _append_row(path: Path, header: tuple[str, ...], row: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(header)
        writer.writerow(row)


class Store:
    """Writes files into data/raw/<category>/ with dedupe and provenance."""

    def __init__(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        self.seen = {_sha(p.read_bytes()) for p in RAW.rglob("*")
                     if p.is_file() and p.name != ".gitkeep"}
        self.texts: set[str] = set()

    def count(self, category: str) -> int:
        folder = RAW / category
        return sum(1 for p in folder.glob("*") if p.is_file()) if folder.exists() else 0

    def save(self, category: str, name: str, data: bytes, origin: str, licence: str) -> Path | None:
        path = RAW / category / name
        if path.exists():
            return None
        digest = _sha(data)
        if digest in self.seen:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.seen.add(digest)
        _append_row(SOURCES, ("file", "origin", "licence", "date"),
                    (f"{category}/{name}", origin, licence, date.today().isoformat()))
        return path

    def save_text(self, category: str, stem: str, text: str, fmt: str,
                  origin: str, licence: str) -> Path | None:
        # near-duplicate guard: same text (ignoring whitespace/case) is stored once
        key = _sha(" ".join(text.split()).lower().encode())
        if key in self.texts:
            return None
        path = self.save(category, f"{stem}.{fmt}", render(text, fmt), origin, licence)
        if path:
            self.texts.add(key)
        return path

    @staticmethod
    def exclude(what: str, reason: str) -> None:
        _append_row(EXCLUDED, ("file", "reason"), (what, reason))