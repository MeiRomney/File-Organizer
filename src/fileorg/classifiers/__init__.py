"""Classifiers under comparison: rule-based baseline, ML (TF-IDF), embeddings."""
from __future__ import annotations

from typing import Callable

from fileorg.classifiers.base import Classifier

CLASSIFIER_NAMES = ["rules", "ml"]  # embeddings are added here later


def get_classifier(name: str, **options) -> Classifier:
    """Create a fresh classifier by name (imports are lazy: heavy ML libraries load only when asked)."""
    if name == "rules":
        from fileorg.classifiers.rule_based import RuleBasedClassifier

        return RuleBasedClassifier(**options)
    if name == "ml":
        from fileorg.classifiers.ml import MLClassifier

        return MLClassifier(**options)
    raise ValueError(f"unknown classifier '{name}', choose from {CLASSIFIER_NAMES}")


def factory(name: str, **options) -> Callable[[], Classifier]:
    return lambda: get_classifier(name, **options)