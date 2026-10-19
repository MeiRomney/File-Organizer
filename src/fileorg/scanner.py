"""Find candidate files on disk."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


def scan(
    root: str | Path,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
    include_hidden: bool = False,
) -> list[Path]:
    """Return a sorted list of files under ``root`` with a supported extension.

    Symlinks to directories are not followed (avoids loops). Hidden files and
    Office lock files (``~$name.docx``) are skipped unless ``include_hidden``.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if not recursive:
            dirnames[:] = []
        for name in filenames:
            if not include_hidden and (name.startswith(".") or name.startswith("~$")):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() in wanted and path.is_file():
                found.append(path)

    found.sort()
    return found
