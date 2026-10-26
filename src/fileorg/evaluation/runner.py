"""Run any Classifier on a Split the same way (train, predict, score, time)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from fileorg.classifiers.base import Classifier
from fileorg.evaluation.metrics import Report, classification_report
from fileorg.evaluation.split import Split
from fileorg.models import FileRecord


@dataclass
class EvalResult:
    report: Report
    fit_seconds: float
    predict_seconds: float
    predictions: list[str]


@dataclass
class CVResult:
    folds: list[EvalResult]
    pooled: Report  # all validation predictions combined
    records: list[FileRecord]  # validation records, in the same order as ``predictions``
    predictions: list[str]

    def errors(self) -> list[tuple[FileRecord, str]]:
        """(record, predicted label) for every misclassified validation file."""
        return [(r, p) for r, p in zip(self.records, self.predictions) if r.label != p]

    @property
    def std_accuracy(self) -> float:
        m = self.mean_accuracy
        return (sum((f.report.accuracy - m) ** 2 for f in self.folds) / len(self.folds)) ** 0.5

    @property
    def mean_accuracy(self) -> float:
        return sum(f.report.accuracy for f in self.folds) / len(self.folds)

    @property
    def mean_macro_f1(self) -> float:
        return sum(f.report.macro.f1 for f in self.folds) / len(self.folds)


def evaluate(
    classifier: Classifier,
    train: Sequence[FileRecord],
    test: Sequence[FileRecord],
    labels: Sequence[str] | None = None,
) -> EvalResult:
    t0 = time.perf_counter()
    classifier.fit(train)
    t1 = time.perf_counter()
    predictions = classifier.predict_many(test)
    t2 = time.perf_counter()
    report = classification_report([r.label for r in test], predictions, labels)
    return EvalResult(report, t1 - t0, t2 - t1, predictions)


def cross_validate(
    make_classifier: Callable[[], Classifier], split: Split, labels: Sequence[str] | None = None
) -> CVResult:
    """k-fold CV on the development part only (a fresh classifier per fold)."""
    results: list[EvalResult] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    seen: list[FileRecord] = []
    for train, val in split.cv_splits():
        result = evaluate(make_classifier(), train, val, labels)
        results.append(result)
        y_true += [r.label for r in val]
        y_pred += result.predictions
        seen += list(val)
    return CVResult(results, classification_report(y_true, y_pred, labels), seen, y_pred)


def final_test(
    make_classifier: Callable[[], Classifier], split: Split, labels: Sequence[str] | None = None
) -> EvalResult:
    """Train on all development data, score on the test set. Run this once."""
    return evaluate(make_classifier(), split.dev, split.test, labels)