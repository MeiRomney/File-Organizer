"""Reproducible, stratified train/test split and cross-validation folds.

Rules from ``docs/dataset_plan.md`` implemented here:

* stratified 80% development / 20% test, fixed seed;
* the split is stored in a small CSV (``rel_path,label,subset,fold``) that is
  committed to git, so every approach uses exactly the same files;
* optional *groups* (e.g. the template a synthetic document was generated
  from): all files of one group always land on the same side of the split and
  in the same CV fold, so the test set cannot leak from the training set;
* the 5 CV folds are built on the development part only. The test part is
  never touched until the final evaluation.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from fileorg.models import FileRecord

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_FOLDS = 5


@dataclass
class Split:
    dev: list[FileRecord]
    test: list[FileRecord]
    folds: list[int] = field(default_factory=list)  # fold index per dev record

    @property
    def n_folds(self) -> int:
        return max(self.folds) + 1 if self.folds else 0

    def cv_splits(self) -> Iterator[tuple[list[FileRecord], list[FileRecord]]]:
        """Yield (train, validation) record lists for each CV fold."""
        for k in range(self.n_folds):
            train = [r for r, f in zip(self.dev, self.folds) if f != k]
            val = [r for r, f in zip(self.dev, self.folds) if f == k]
            yield train, val


# ------------------------------------------------------------------ helpers
def load_groups(
    path: str | Path, file_col: str = "file", group_col: str = "template_id"
) -> dict[str, str]:
    """Read ``rel_path -> group`` from ``data/sources.csv`` (empty cells skipped)."""
    groups: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            file, group = (row.get(file_col) or "").strip(), (row.get(group_col) or "").strip()
            if file and group:
                groups[file.replace("\\", "/")] = group
    return groups


def _units_by_label(
    records: Sequence[FileRecord], groups: Optional[Mapping[str, str]]
) -> dict[str, list[list[int]]]:
    """label -> list of units (lists of record indices that must stay together)."""
    by_label: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, rec in enumerate(records):
        if rec.label is None:
            raise ValueError(f"{rec.rel_path} has no label; scan with --labelled")
        group = groups.get(rec.rel_path) if groups else None
        key = ("group", group) if group else ("file", rec.rel_path)
        by_label[rec.label][key].append(i)
    return {
        label: [units[key] for key in sorted(units)]
        for label, units in sorted(by_label.items())
    }


def _pick_test_units(units: list[list[int]], test_size: float, rng: random.Random) -> list[list[int]]:
    units = list(units)
    rng.shuffle(units)
    total = sum(len(u) for u in units)
    target = round(total * test_size)
    chosen: list[list[int]] = []
    count = 0
    for unit in units:
        if count >= target:
            break
        if abs(count + len(unit) - target) <= abs(count - target):  # moves us closer
            chosen.append(unit)
            count += len(unit)
    if not chosen:
        chosen = [units[0]]
    if len(chosen) == len(units):
        chosen = chosen[:-1]
    return chosen


def assign_folds(
    records: Sequence[FileRecord],
    k: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    groups: Optional[Mapping[str, str]] = None,
) -> list[int]:
    """Stratified (and group-aware) fold index for every record."""
    rng = random.Random(seed)
    folds = [-1] * len(records)
    total = [0] * k
    for label, units in _units_by_label(records, groups).items():
        if len(units) < k:
            raise ValueError(f"class '{label}' has only {len(units)} independent units; need >= {k} folds")
        units = list(units)
        rng.shuffle(units)
        units.sort(key=len, reverse=True)  # stable: big units first keeps folds even
        per_label = [0] * k
        for unit in units:
            f = min(range(k), key=lambda j: (per_label[j], total[j], j))
            for i in unit:
                folds[i] = f
            per_label[f] += len(unit)
            total[f] += len(unit)
    return folds


# -------------------------------------------------------------- main entry
def make_split(
    records: Sequence[FileRecord],
    test_size: float = DEFAULT_TEST_SIZE,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    groups: Optional[Mapping[str, str]] = None,
) -> Split:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    records = sorted(records, key=lambda r: r.rel_path)  # input order must not matter
    rng = random.Random(seed)

    test_idx: set[int] = set()
    for label, units in _units_by_label(records, groups).items():
        if len(units) < 2:
            raise ValueError(f"class '{label}' has fewer than 2 independent units; cannot split")
        for unit in _pick_test_units(units, test_size, rng):
            test_idx.update(unit)

    dev = [r for i, r in enumerate(records) if i not in test_idx]
    test = [r for i, r in enumerate(records) if i in test_idx]
    fold_ids = assign_folds(dev, folds, seed, groups) if folds else []
    return Split(dev=dev, test=test, folds=fold_ids)


# ------------------------------------------------------------- persistence
def write_split(split: Split, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rel_path", "label", "subset", "fold"])
        for rec, fold in zip(split.dev, split.folds or [""] * len(split.dev)):
            writer.writerow([rec.rel_path, rec.label, "dev", fold])
        for rec in split.test:
            writer.writerow([rec.rel_path, rec.label, "test", ""])


def apply_split(records: Sequence[FileRecord], path: str | Path) -> Split:
    """Rebuild a stored split for ``records`` (matched by ``rel_path``)."""
    assignment: dict[str, tuple[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assignment[row["rel_path"]] = (row["subset"], row["fold"])
    by_path = {r.rel_path: r for r in records}
    missing = sorted(set(assignment) - set(by_path))
    if missing:
        raise KeyError(f"{len(missing)} files in the split are not in the records, e.g. {missing[:3]}")
    unassigned = sorted(set(by_path) - set(assignment))
    if unassigned:
        raise KeyError(f"{len(unassigned)} records are not in the split, e.g. {unassigned[:3]}")
    dev, fold_ids, test = [], [], []
    for rel in sorted(assignment):
        subset, fold = assignment[rel]
        if subset == "test":
            test.append(by_path[rel])
        else:
            dev.append(by_path[rel])
            fold_ids.append(int(fold) if fold != "" else 0)
    return Split(dev=dev, test=test, folds=fold_ids)