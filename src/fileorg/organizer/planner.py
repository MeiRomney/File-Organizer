"""Move planner: decide where every file goes, without touching the disk.

The plan is a plain list of ``PlannedMove`` objects, so it can be shown to the
user (preview) and confirmed before anything moves. Safety rules built into
the planner:

* a file is never planned onto an existing file: a clash gets `` (2)``, `` (3)``
  ... before the extension, also when two planned files share a name (and
  also on case-insensitive file systems such as Windows and macOS);
* a file that is already in the right folder is left alone;
* unreadable files (scanned PDF, encrypted, corrupt) and files the classifier
  is unsure about are left where they are by default, so the user decides;
* the size and modification time seen now are stored, so the executor can
  notice a file that changed after the preview.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord

UNREADABLE_STATUSES = ("error", "empty", "unsupported")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: Optional[float] = None  # None: the classifier gives no probability (rules)


@dataclass(frozen=True)
class PlannedMove:
    src: Path
    dst: Path
    category: str
    confidence: Optional[float]
    size_bytes: int
    mtime_ns: int
    renamed: bool = False  # True when dst got a " (2)" suffix to avoid a clash


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str  # already_in_place | low_confidence | unreadable | unavailable | blocked
    detail: str = ""


@dataclass
class MovePlan:
    root: Path
    dest_root: Path
    moves: list[PlannedMove] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    def category_counts(self) -> Counter:
        return Counter(m.category for m in self.moves)

    def skipped_counts(self) -> Counter:
        return Counter(s.reason for s in self.skipped)


def predict_with_confidence(classifier: Classifier, records: Sequence[FileRecord]) -> list[Prediction]:
    """Label and (when the model provides one) probability for every record."""
    if not records:
        return []
    proba_fn = getattr(classifier, "predict_proba", None)
    if proba_fn is not None:
        try:
            probabilities = proba_fn(records)
        except NotImplementedError:  # e.g. a linear SVM
            probabilities = None
        if probabilities is not None:
            out = []
            for p in probabilities:
                label = max(p, key=p.get)
                out.append(Prediction(label, p[label]))
            return out
    return [Prediction(label) for label in classifier.predict_many(records)]


def unique_destination(dst: Path, taken: Optional[set[str]] = None) -> tuple[Path, bool]:
    """``dst`` or, if that name is in use, ``name (2).ext``, ``name (3).ext`` ... Returns (path, renamed)."""
    taken = taken if taken is not None else set()
    candidate, n = dst, 1
    while str(candidate).lower() in taken or os.path.lexists(candidate):
        n += 1
        candidate = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
    return candidate, candidate != dst


def _folder_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .") or "other"


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def build_plan(
    records: Sequence[FileRecord],
    predictions: Sequence[Prediction],
    root: str | Path,
    dest_root: str | Path | None = None,
    *,
    min_confidence: float = 0.5,
    unsure: str = "skip",
    unreadable: str = "skip",
    folder_names: Optional[Mapping[str, str]] = None,
) -> MovePlan:
    """Turn predictions into a list of moves into ``dest_root/<category>/``.

    ``unsure``: what to do when the confidence is below ``min_confidence``:
    ``skip`` leaves the file where it is, ``other`` files it under ``other``.
    ``unreadable``: ``skip`` leaves files without extractable text alone,
    ``classify`` classifies them anyway (file name and metadata only).
    """
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")
    if unsure not in ("skip", "other") or unreadable not in ("skip", "classify"):
        raise ValueError("unsure must be 'skip' or 'other'; unreadable must be 'skip' or 'classify'")
    root = Path(root).resolve()
    dest_root = Path(dest_root).resolve() if dest_root else root
    names = dict(folder_names or {})
    plan = MovePlan(root, dest_root)
    taken: set[str] = set()

    for record, prediction in sorted(zip(records, predictions), key=lambda pair: pair[0].path):
        src = Path(record.path)
        if unreadable == "skip" and record.extraction_status in UNREADABLE_STATUSES:
            plan.skipped.append(SkippedFile(src, "unreadable", record.extraction_status))
            continue
        label = prediction.label
        if prediction.confidence is not None and prediction.confidence < min_confidence:
            if unsure == "skip":
                plan.skipped.append(SkippedFile(src, "low_confidence", f"{label} {prediction.confidence:.2f}"))
                continue
            label = "other"
        dest_dir = dest_root / _folder_name(names.get(label, label))
        if _same_path(src.parent, dest_dir):
            plan.skipped.append(SkippedFile(src, "already_in_place", label))
            continue
        try:
            stat = src.stat()
        except OSError as exc:
            plan.skipped.append(SkippedFile(src, "unavailable", str(exc)))
            continue
        if os.path.lexists(dest_dir) and not dest_dir.is_dir():
            plan.skipped.append(SkippedFile(src, "blocked", f"{dest_dir} exists and is not a folder"))
            continue
        dst, renamed = unique_destination(dest_dir / src.name, taken)
        taken.add(str(dst).lower())
        plan.moves.append(PlannedMove(src, dst, label, prediction.confidence, stat.st_size, stat.st_mtime_ns, renamed))
    return plan


def format_plan(plan: MovePlan, examples: int = 5) -> str:
    """Text preview for the terminal (ASCII only, so it prints on any console)."""
    total = len(plan.moves) + len(plan.skipped)
    lines = [f"Plan for {plan.root}: {total} file(s) looked at",
             f"  {len(plan.moves)} to move" + (f" into {plan.dest_root}" if plan.dest_root != plan.root else ""),
             f"  {len(plan.skipped)} left where they are"]
    if plan.moves:
        lines.append("")
        by_category: dict[str, list[PlannedMove]] = {}
        for move in plan.moves:
            by_category.setdefault(move.category, []).append(move)
        for category, moves in sorted(by_category.items()):
            lines.append(f"  {category}/  ({len(moves)})")
            for move in moves[:examples]:
                conf = f"  conf {move.confidence:.2f}" if move.confidence is not None else ""
                note = f"  (renamed to {move.dst.name})" if move.renamed else ""
                lines.append(f"      {move.src.name}{conf}{note}")
            if len(moves) > examples:
                lines.append(f"      ... and {len(moves) - examples} more")
    if plan.skipped:
        lines.append("")
        explanation = {"already_in_place": "already in the right folder", "low_confidence": "classifier not sure enough",
                       "unreadable": "no readable text", "unavailable": "could not be read",
                       "blocked": "target folder name is taken by a file"}
        for reason, count in sorted(plan.skipped_counts().items()):
            lines.append(f"  left in place: {count} {explanation.get(reason, reason)}")
        unsure = [s for s in plan.skipped if s.reason == "low_confidence"][:examples]
        for s in unsure:
            lines.append(f"      {s.path.name}  (best guess: {s.detail})")
    return "\n".join(lines)