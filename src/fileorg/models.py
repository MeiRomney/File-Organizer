"""Data model shared by every part of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class FileRecord:
    """Everything we know about one file after scanning and extraction.

    All classifiers (rule-based, ML, embeddings) consume this same record,
    which keeps the comparison between them fair.
    """

    path: str                       # absolute path
    rel_path: str                   # path relative to the scanned root
    filename: str
    extension: str                  # lower-case, with dot, e.g. ".pdf"
    size_bytes: int
    modified: str                   # ISO 8601, UTC
    ctime: str                      # creation time on Windows, metadata-change time on Linux/macOS
    mime_type: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)  # metadata embedded in the file
    text: str = ""                  # extracted text (possibly truncated)
    extraction_status: str = "ok"   # ok | empty | error | unsupported
    extraction_error: Optional[str] = None
    extraction_seconds: float = 0.0
    label: Optional[str] = None     # ground truth, only for the labelled dataset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileRecord":
        return cls(**data)
