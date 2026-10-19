"""Scan -> extract -> FileRecord, plus JSONL persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from fileorg.extraction import DEFAULT_MAX_CHARS, extract
from fileorg.metadata import filesystem_metadata
from fileorg.models import FileRecord
from fileorg.scanner import SUPPORTED_EXTENSIONS, scan


def label_from_folder(path: Path, root: Path) -> Optional[str]:
    """Dataset convention: root/<category>/.../file -> label = <category>."""
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else None


def build_record(
    path: Path,
    root: Path,
    labelled: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: Optional[int] = None,
) -> FileRecord:
    fs = filesystem_metadata(path)
    start = time.perf_counter()
    result = extract(path, max_chars=max_chars, max_pages=max_pages)
    elapsed = time.perf_counter() - start

    return FileRecord(
        path=str(path.resolve()),
        rel_path=path.relative_to(root).as_posix(),
        filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=fs["size_bytes"],
        modified=fs["modified"],
        ctime=fs["ctime"],
        mime_type=fs["mime_type"],
        metadata=result.metadata,
        text=result.text,
        extraction_status=result.status,
        extraction_error=result.error,
        extraction_seconds=round(elapsed, 4),
        label=label_from_folder(path, root) if labelled else None,
    )


def process_directory(
    root: str | Path,
    *,
    labelled: bool = False,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[FileRecord]:
    root = Path(root)
    paths = scan(root, extensions=extensions, recursive=recursive)
    records: list[FileRecord] = []
    for index, path in enumerate(paths, start=1):
        records.append(build_record(path, root, labelled, max_chars, max_pages))
        if on_progress:
            on_progress(index, len(paths))
    return records


def write_jsonl(records: Iterable[FileRecord], out_path: str | Path) -> None:
    with Path(out_path).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[FileRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [FileRecord.from_dict(json.loads(line)) for line in fh if line.strip()]
