"""Logic behind the desktop interface (no Tkinter in here, so it is easy to test).

A ``ReviewSession`` holds one scanned folder: the records, the classifier's
predictions, and the user's edits (exclude a file, change its category). The
move plan is always *rebuilt* from the current rows with the normal planner, so
the preview on screen and what the executor does can never disagree, and all
safety rules (no overwriting, unreadable/unsure files left alone, change
detection, log + undo) are the same as in the command line.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from fileorg.classifiers import get_classifier
from fileorg.classifiers.base import Classifier
from fileorg.classifiers.rule_based import CATEGORIES
from fileorg.models import FileRecord
from fileorg.organizer import (
    ExecutionResult, MoveLog, MovePlan, Prediction, build_plan, execute_plan, predict_with_confidence,
)
from fileorg.pipeline import process_directory

CLASSIFIERS = ("rules", "ml", "embeddings")
UNSURE_CHOICES = {"Leave in place": "skip", "File under 'other'": "other"}
UNREADABLE_CHOICES = {"Leave in place": "skip", "Classify by name and metadata": "classify"}
SKIP_TEXT = {
    "already_in_place": "already in the right folder",
    "low_confidence": "classifier not sure enough",
    "unreadable": "no readable text",
    "unavailable": "could not be read",
    "blocked": "target folder name is taken by a file",
}


def log_path(root: str | Path) -> Path:
    """Same location as the command line: <folder>/.fileorg/moves.sqlite."""
    return Path(root).resolve() / ".fileorg" / "moves.sqlite"


def load_classifier(name: str, model_file: Optional[str] = None) -> Classifier:
    if name == "rules":
        return get_classifier("rules")
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier '{name}'")
    if not model_file:
        raise ValueError(f"The '{name}' classifier needs a trained model file. Create one with:\n"
                         f"fileorg train data/records.jsonl --classifier {name} --out models/{name}.joblib")
    if not Path(model_file).is_file():
        raise ValueError(f"Model file not found: {model_file}")
    if name == "ml":
        from fileorg.classifiers.ml import MLClassifier

        return MLClassifier.load(model_file)
    from fileorg.classifiers.embeddings import EmbeddingClassifier

    return EmbeddingClassifier.load(model_file)


@dataclass
class Row:
    record: FileRecord
    label: str                       # current category (changes when the user edits it)
    confidence: Optional[float]      # the classifier's own confidence (None: rules)
    include: bool = True
    overridden: bool = False         # the user chose this category by hand
    original_label: str = ""


@dataclass(frozen=True)
class Outcome:
    action: str       # move | skip | excluded
    text: str         # what to show in the "Result" column
    category: str     # category after the planner's rules (e.g. unsure -> other)
    destination: str  # path relative to the destination root, "" when nothing moves
    reason: str = ""  # planner reason for a skip: low_confidence | unreadable | already_in_place | ...


class ReviewSession:
    def __init__(self, root: Path, dest_root: Optional[Path], rows: list[Row], classifier_name: str,
                 min_confidence: float = 0.5, unsure: str = "skip", unreadable: str = "skip") -> None:
        self.root, self.dest_root, self.rows = Path(root).resolve(), dest_root, rows
        self.classifier_name = classifier_name
        self.min_confidence, self.unsure, self.unreadable = min_confidence, unsure, unreadable

    # ----------------------------------------------------------- editing
    def set_include(self, indices: Iterable[int], include: bool) -> None:
        for i in indices:
            self.rows[i].include = include

    def set_label(self, indices: Iterable[int], label: str) -> None:
        if not label.strip():
            raise ValueError("category name is empty")
        for i in indices:
            row = self.rows[i]
            row.label, row.overridden = label.strip(), True

    def reset(self, indices: Iterable[int]) -> None:
        for i in indices:
            row = self.rows[i]
            row.label, row.overridden, row.include = row.original_label, False, True

    def categories(self) -> list[str]:
        extra = sorted({r.original_label for r in self.rows} - set(CATEGORIES))
        return [*CATEGORIES, *extra]

    # -------------------------------------------------------------- plan
    def plan(self) -> MovePlan:
        records, predictions = [], []
        for row in self.rows:
            if not row.include:
                continue
            record = row.record
            if row.overridden and record.extraction_status != "ok":
                # the user decided: move it even though no text could be read
                record = dataclasses.replace(record, extraction_status="ok")
            records.append(record)
            predictions.append(Prediction(row.label, None if row.overridden else row.confidence))
        return build_plan(records, predictions, self.root, self.dest_root, min_confidence=self.min_confidence,
                          unsure=self.unsure, unreadable=self.unreadable)

    def outcomes(self, plan: Optional[MovePlan] = None) -> list[Outcome]:
        """One Outcome per row, in row order."""
        plan = self.plan() if plan is None else plan
        moves = {m.src: m for m in plan.moves}
        skipped = {s.path: s for s in plan.skipped}
        result = []
        for row in self.rows:
            if not row.include:
                result.append(Outcome("excluded", "excluded by you", row.label, ""))
                continue
            key = Path(row.record.path)
            if key in moves:
                move = moves[key]
                try:
                    dest = move.dst.relative_to(plan.dest_root).as_posix()
                except ValueError:
                    dest = str(move.dst)
                result.append(Outcome("move", "will move" + (" (renamed)" if move.renamed else ""), move.category, dest))
            else:
                reason = skipped[key].reason if key in skipped else "skipped"
                result.append(Outcome("skip", SKIP_TEXT.get(reason, reason), row.label, "", reason))
        return result

    def summary(self, plan: Optional[MovePlan] = None) -> str:
        plan = self.plan() if plan is None else plan
        parts = [f"{len(plan.moves)} to move"]
        parts += [f"{n} {SKIP_TEXT.get(reason, reason)}" for reason, n in sorted(plan.skipped_counts().items())]
        excluded = sum(1 for r in self.rows if not r.include)
        if excluded:
            parts.append(f"{excluded} excluded by you")
        return f"{len(self.rows)} files: " + "; ".join(parts)


# --------------------------------------------------------------- actions
def analyse(
    folder: str | Path,
    *,
    dest: Optional[str | Path] = None,
    classifier_name: str = "rules",
    model_file: Optional[str] = None,
    recursive: bool = False,
    min_confidence: float = 0.5,
    unsure: str = "skip",
    unreadable: str = "skip",
    progress: Optional[Callable[[int, int], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ReviewSession:
    """Scan, extract and classify a folder. Nothing is moved."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    stage = on_stage or (lambda message: None)
    stage("Reading files...")
    records = process_directory(root, recursive=recursive, on_progress=progress)
    stage("Loading the classifier and classifying...")
    classifier = load_classifier(classifier_name, model_file)
    predictions = predict_with_confidence(classifier, records)
    rows = [Row(r, p.label, p.confidence, original_label=p.label) for r, p in zip(records, predictions)]
    return ReviewSession(root, Path(dest).resolve() if dest else None, rows, classifier.name,
                         min_confidence, unsure, unreadable)


def apply_plan(session: ReviewSession, progress: Optional[Callable[[int, int], None]] = None) -> ExecutionResult:
    """Move the files of the current plan; every move is written to the log first."""
    plan = session.plan()
    with MoveLog(log_path(session.root)) as log:
        return execute_plan(plan, log, session.classifier_name, progress=progress)