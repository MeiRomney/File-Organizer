"""Hyper-parameter search with the group-aware CV folds of the development set.

Only the development part of the split is used. The test set is never touched
here, so choosing parameters with this module does not leak into the final
evaluation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from fileorg.classifiers.base import Classifier
from fileorg.evaluation.runner import cross_validate
from fileorg.evaluation.split import Split


@dataclass
class TuneResult:
    params: dict[str, Any]
    mean_macro_f1: float
    mean_accuracy: float
    std_accuracy: float
    fit_seconds: float  # total training time over all folds


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def grid_search(
    make: Callable[..., Classifier], grid: Mapping[str, Sequence[Any]], split: Split
) -> list[TuneResult]:
    """Cross-validate every parameter combination; best (macro-F1, then accuracy) first."""
    results = []
    for params in expand_grid(grid):
        cv = cross_validate(lambda p=params: make(**p), split)
        results.append(TuneResult(params, cv.mean_macro_f1, cv.mean_accuracy, cv.std_accuracy,
                                  sum(f.fit_seconds for f in cv.folds)))
    results.sort(key=lambda r: (-r.mean_macro_f1, -r.mean_accuracy, r.fit_seconds))
    return results