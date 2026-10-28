"""Is one approach really better than another? Confidence intervals and tests.

* ``bootstrap_ci``      95 % interval for accuracy or macro-F1 of one classifier.
* ``paired_bootstrap``  interval and p-value for the *difference* between two
                        classifiers evaluated on the same files.
* ``mcnemar``           exact McNemar test on the files where exactly one of
                        the two classifiers is right.

Files made from the same template are not independent. Pass ``groups`` (the
template id per file) and the bootstrap resamples whole groups instead of
single files, which gives honest, wider intervals. McNemar assumes independent
files, so with grouped data treat its p-value as optimistic and rely on the
group bootstrap. With several comparisons, divide alpha by their number
(Bonferroni); ``compare_classifiers`` reports that number.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

METRICS = ("accuracy", "macro_f1")


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class McNemarResult:
    both_correct: int
    only_a: int
    only_b: int
    both_wrong: int
    p_value: float


@dataclass(frozen=True)
class PairedResult:
    diff: float  # metric(a) - metric(b)
    low: float
    high: float
    p_value: float


@dataclass
class PairComparison:
    a: str
    b: str
    mcnemar: McNemarResult
    accuracy: PairedResult
    macro_f1: PairedResult


@dataclass
class Comparison:
    n: int
    n_groups: Optional[int]
    intervals: dict[str, dict[str, Interval]]  # classifier -> metric -> interval
    pairs: list[PairComparison] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def adjusted_alpha(self) -> float:
        return self.alpha / len(self.pairs) if self.pairs else self.alpha


# ------------------------------------------------------------------ metrics
def _score(metric: str, y_true: Sequence[str], y_pred: Sequence[str], idx: Sequence[int]) -> float:
    if metric == "accuracy":
        return sum(y_true[i] == y_pred[i] for i in idx) / len(idx)
    if metric != "macro_f1":
        raise ValueError(f"metric must be one of {METRICS}")
    tp, fp, fn = Counter(), Counter(), Counter()
    for i in idx:
        t, p = y_true[i], y_pred[i]
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in {y_true[i] for i in idx}:  # classes present in this sample
        precision = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def _resamples(n: int, groups: Optional[Sequence], n_boot: int, rng: random.Random):
    if groups is None:
        for _ in range(n_boot):
            yield [rng.randrange(n) for _ in range(n)]
        return
    members: dict = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    keys = sorted(members, key=str)
    for _ in range(n_boot):
        sample: list[int] = []
        for _ in keys:
            sample.extend(members[rng.choice(keys)])
        yield sample


def _quantile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))]


def _check(y_true, *predictions) -> None:
    if not y_true:
        raise ValueError("no files to compare")
    if any(len(p) != len(y_true) for p in predictions):
        raise ValueError("predictions and labels must have the same length")


# -------------------------------------------------------------------- tests
def bootstrap_ci(
    y_true: Sequence[str], y_pred: Sequence[str], metric: str = "accuracy", n_boot: int = 2000,
    seed: int = 42, groups: Optional[Sequence] = None, alpha: float = 0.05,
) -> Interval:
    _check(y_true, y_pred)
    rng = random.Random(seed)
    values = sorted(_score(metric, y_true, y_pred, s) for s in _resamples(len(y_true), groups, n_boot, rng))
    return Interval(_score(metric, y_true, y_pred, range(len(y_true))),
                    _quantile(values, alpha / 2), _quantile(values, 1 - alpha / 2))


def paired_bootstrap(
    y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str], metric: str = "accuracy",
    n_boot: int = 2000, seed: int = 42, groups: Optional[Sequence] = None, alpha: float = 0.05,
) -> PairedResult:
    _check(y_true, pred_a, pred_b)
    rng = random.Random(seed)
    diffs = sorted(_score(metric, y_true, pred_a, s) - _score(metric, y_true, pred_b, s)
                   for s in _resamples(len(y_true), groups, n_boot, rng))
    everything = range(len(y_true))
    point = _score(metric, y_true, pred_a, everything) - _score(metric, y_true, pred_b, everything)
    below = sum(d <= 0 for d in diffs) / len(diffs)
    above = sum(d >= 0 for d in diffs) / len(diffs)
    return PairedResult(point, _quantile(diffs, alpha / 2), _quantile(diffs, 1 - alpha / 2),
                        min(1.0, 2 * min(below, above)))


def mcnemar(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> McNemarResult:
    """Exact (binomial) two-sided McNemar test on paired correctness."""
    _check(y_true, pred_a, pred_b)
    both = only_a = only_b = neither = 0
    for t, a, b in zip(y_true, pred_a, pred_b):
        if a == t and b == t:
            both += 1
        elif a == t:
            only_a += 1
        elif b == t:
            only_b += 1
        else:
            neither += 1
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return McNemarResult(both, only_a, only_b, neither, p)


# --------------------------------------------------------------- comparison
def compare_classifiers(
    y_true: Sequence[str], predictions: dict[str, Sequence[str]], groups: Optional[Sequence] = None,
    n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
) -> Comparison:
    _check(y_true, *predictions.values())
    intervals = {
        name: {m: bootstrap_ci(y_true, pred, m, n_boot, seed, groups, alpha) for m in METRICS}
        for name, pred in predictions.items()
    }
    pairs = [
        PairComparison(a, b, mcnemar(y_true, predictions[a], predictions[b]),
                       paired_bootstrap(y_true, predictions[a], predictions[b], "accuracy", n_boot, seed, groups, alpha),
                       paired_bootstrap(y_true, predictions[a], predictions[b], "macro_f1", n_boot, seed, groups, alpha))
        for a, b in itertools.combinations(predictions, 2)
    ]
    n_groups = len(set(groups)) if groups is not None else None
    return Comparison(len(y_true), n_groups, intervals, pairs, alpha)


def format_comparison(c: Comparison) -> str:
    resampling = f"{c.n_groups} template groups" if c.n_groups else f"{c.n} files"
    lines = [f"{c.n} files; bootstrap resamples {resampling}; {int((1 - c.alpha) * 100)}% intervals", "",
             f"{'classifier':<14}  {'accuracy [CI]':<26}  macro-F1 [CI]"]
    for name, m in c.intervals.items():
        a, f = m["accuracy"], m["macro_f1"]
        lines.append(f"{name:<14}  {a.point:.3f} [{a.low:.3f}, {a.high:.3f}]      {f.point:.3f} [{f.low:.3f}, {f.high:.3f}]")
    if c.pairs:
        lines += ["", f"Pairwise (A vs B). Bonferroni-adjusted alpha for {len(c.pairs)} comparison(s): {c.adjusted_alpha:.4f}",
                  f"{'A vs B':<24}  {'only A':>6}  {'only B':>6}  {'McNemar p':>9}  {'acc diff [CI]':<26}  {'bootstrap p':>11}"]
        for p in c.pairs:
            d = p.accuracy
            lines.append(f"{p.a + ' vs ' + p.b:<24}  {p.mcnemar.only_a:6d}  {p.mcnemar.only_b:6d}  {p.mcnemar.p_value:9.4f}  "
                         f"{d.diff:+.3f} [{d.low:+.3f}, {d.high:+.3f}]  {d.p_value:11.4f}")
        lines.append("")
        lines.append("A difference is only convincing if its interval excludes 0 and the p-value is below the adjusted alpha.")
    return "\n".join(lines)