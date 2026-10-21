"""Accuracy, precision, recall, F1, confusion matrix and label agreement.

Pure Python (no scikit-learn needed), so the evaluation code stays small and
easy to check by hand. Macro averages treat every class equally, which is the
right headline number for a balanced dataset and the safe one if a class ends
up smaller; weighted averages are reported next to it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # number of true examples of this class


@dataclass
class Report:
    labels: list[str]
    accuracy: float
    per_class: dict[str, ClassMetrics]
    macro: ClassMetrics
    weighted: ClassMetrics
    confusion: list[list[int]]  # confusion[i][j]: true labels[i] predicted as labels[j]
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro": vars(self.macro),
            "weighted": vars(self.weighted),
            "per_class": {k: vars(v) for k, v in self.per_class.items()},
            "labels": self.labels,
            "confusion": self.confusion,
        }

    def format(self) -> str:
        return format_report(self)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return _div(2 * p * r, p + r)


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> Report:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot evaluate on an empty list")
    label_list = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(label_list)}
    size = len(label_list)
    confusion = [[0] * size for _ in range(size)]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:  # predictions outside the label set are misses
            confusion[index[t]][index[p]] += 1

    per_class: dict[str, ClassMetrics] = {}
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    for label, i in index.items():
        tp = confusion[i][i]
        precision = _div(tp, pred_counts[label])
        recall = _div(tp, true_counts[label])
        per_class[label] = ClassMetrics(precision, recall, _f1(precision, recall), true_counts[label])

    total = sum(m.support for m in per_class.values())
    n_classes = len(per_class)
    macro = ClassMetrics(
        sum(m.precision for m in per_class.values()) / n_classes,
        sum(m.recall for m in per_class.values()) / n_classes,
        sum(m.f1 for m in per_class.values()) / n_classes,
        total,
    )
    weighted = ClassMetrics(
        _div(sum(m.precision * m.support for m in per_class.values()), total),
        _div(sum(m.recall * m.support for m in per_class.values()), total),
        _div(sum(m.f1 * m.support for m in per_class.values()), total),
        total,
    )
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return Report(label_list, correct / len(y_true), per_class, macro, weighted, confusion, len(y_true))


def format_report(report: Report) -> str:
    width = max([len(l) for l in report.labels] + [12])
    lines = [f"{'':<{width}}  precision  recall      f1  support"]
    for label in report.labels:
        m = report.per_class[label]
        lines.append(f"{label:<{width}}  {m.precision:9.3f}  {m.recall:6.3f}  {m.f1:6.3f}  {m.support:7d}")
    lines.append("")
    for name, m in (("macro avg", report.macro), ("weighted avg", report.weighted)):
        lines.append(f"{name:<{width}}  {m.precision:9.3f}  {m.recall:6.3f}  {m.f1:6.3f}  {m.support:7d}")
    lines.append(f"{'accuracy':<{width}}  {'':9}  {'':6}  {report.accuracy:6.3f}  {report.n:7d}")
    return "\n".join(lines)


def format_confusion(report: Report) -> str:
    """Plain-text confusion matrix (rows = true, columns = predicted)."""
    names = report.labels
    width = max(max(len(n) for n in names), 4)
    header = " " * width + "  " + "  ".join(f"{n[:7]:>7}" for n in names)
    rows = [f"{name:<{width}}  " + "  ".join(f"{v:7d}" for v in row)
            for name, row in zip(names, report.confusion)]
    return "\n".join([header, *rows])


# ----------------------------------------------------- label quality check
def agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and of equal length")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two labellers (for the 10% second-labeller check)."""
    observed = agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)