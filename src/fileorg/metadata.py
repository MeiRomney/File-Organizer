"""File-system level metadata (name, size, timestamps, MIME type).

Metadata embedded *inside* a document (PDF author, DOCX title, ...) is
extracted together with the text in ``extraction.py``.
"""
from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def filesystem_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    mime_type, _ = mimetypes.guess_type(path.name)
    return {
        "size_bytes": stat.st_size,
        "modified": _iso(stat.st_mtime),
        "ctime": _iso(stat.st_ctime),
        "mime_type": mime_type,
    }
