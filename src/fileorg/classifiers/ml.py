"""Machine-learning classifier: TF-IDF features + a linear model.

Features (feature groups, for the ablation experiments):

* ``text``     - word 1-2-grams of the extracted text (sub-linear TF-IDF),
* ``filename`` - words in the file name (letters only, so ``scan_012`` -> ``scan``),
* ``metadata`` - words in the embedded title / subject / keywords / category.
  Author, creator and producer are left out on purpose: they identify the
  tool or the person, not the kind of document.

Each group is vectorised separately, scaled (text 1.0, file name and metadata
0.5 so a file name cannot drown out the content) and concatenated.

Models: ``logreg`` (default; gives class probabilities, useful as a confidence
for the organizer) or ``svm`` (LinearSVC). Both use balanced class weights, so
a class that ends up smaller is not under-served.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Iterable, Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord

FEATURE_GROUPS = ("text", "filename", "metadata")
GROUP_WEIGHT = {"text": 1.0, "filename": 0.5, "metadata": 0.5}
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"  # words of 2+ letters; numbers carry little class information
META_KEYS = ("title", "subject", "keywords", "category")


def _clip(text: str, max_chars: int) -> str:
    """Keep the start and the end of long texts (references and signatures are at the end)."""
    if len(text) <= max_chars:
        return text
    cut = max_chars * 3 // 4
    return text[:cut] + "\n" + text[-(max_chars - cut):]


class MLClassifier(Classifier):
    name = "ml"

    def __init__(
        self,
        features: Iterable[str] = FEATURE_GROUPS,
        model: str = "logreg",
        C: float = 1.0,
        ngram_max: int = 2,
        min_df: int = 2,
        max_features: int = 50_000,
        max_chars: int = 20_000,
        min_confidence: float = 0.0,
        random_state: int = 42,
    ) -> None:
        features = tuple(f for f in FEATURE_GROUPS if f in set(features))  # fixed order
        if not features:
            raise ValueError(f"choose at least one feature group from {FEATURE_GROUPS}")
        if model not in ("logreg", "svm"):
            raise ValueError("model must be 'logreg' or 'svm'")
        if min_confidence and model != "logreg":
            raise ValueError("min_confidence needs model='logreg' (probabilities)")
        self.features, self.model, self.C = features, model, C
        self.ngram_max, self.min_df, self.max_features = ngram_max, min_df, max_features
        self.max_chars, self.min_confidence, self.random_state = max_chars, min_confidence, random_state
        self._vectorizers: dict[str, TfidfVectorizer | None] = {}
        self._clf = None

    # ----------------------------------------------------------- features
    def _doc(self, record: FileRecord, group: str) -> str:
        if group == "text":
            return _clip(record.text or "", self.max_chars)
        if group == "filename":
            return re.sub(r"[^a-z]+", " ", PurePath(record.filename).stem.lower())
        meta = record.metadata or {}
        return " ".join(str(meta.get(k, "")) for k in META_KEYS)

    def _new_vectorizer(self, group: str, min_df: int) -> TfidfVectorizer:
        text = group == "text"
        return TfidfVectorizer(
            lowercase=True, strip_accents="unicode", token_pattern=TOKEN_PATTERN, sublinear_tf=True,
            ngram_range=(1, self.ngram_max) if text else (1, 1),
            min_df=min_df if text else 1, max_features=self.max_features if text else None,
        )

    def _fit_group(self, group: str, docs: list[str]):
        for min_df in dict.fromkeys((self.min_df, 1)):  # retry without min_df on tiny corpora
            vec = self._new_vectorizer(group, min_df)
            try:
                return vec, vec.fit_transform(docs)
            except ValueError:  # empty vocabulary (e.g. no metadata at all)
                continue
        return None, None

    def _matrix(self, records: Sequence[FileRecord]) -> csr_matrix:
        blocks = []
        for group in self.features:
            vec = self._vectorizers.get(group)
            if vec is not None:
                blocks.append(vec.transform([self._doc(r, group) for r in records]) * GROUP_WEIGHT[group])
        return hstack(blocks, format="csr")

    # ------------------------------------------------------------ training
    def fit(self, records: Sequence[FileRecord]) -> None:
        if not records:
            raise ValueError("cannot train on an empty list")
        labels = [r.label for r in records]
        if None in labels:
            raise ValueError("all training records need a label")
        if len(set(labels)) < 2:
            raise ValueError("need at least two classes to train")

        blocks = []
        self._vectorizers = {}
        for group in self.features:
            vec, matrix = self._fit_group(group, [self._doc(r, group) for r in records])
            self._vectorizers[group] = vec
            if vec is not None:
                blocks.append(matrix * GROUP_WEIGHT[group])
        if not blocks:
            raise ValueError("no usable features: all selected feature groups are empty")
        X = hstack(blocks, format="csr")

        if self.model == "logreg":
            self._clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced",
                                           random_state=self.random_state)
        else:
            self._clf = LinearSVC(C=self.C, class_weight="balanced", dual="auto", random_state=self.random_state)
        self._clf.fit(X, labels)

    # ---------------------------------------------------------- prediction
    def _check_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("call fit() before predict()")

    def predict_many(self, records: Sequence[FileRecord]) -> list[str]:
        self._check_fitted()
        if not records:
            return []
        X = self._matrix(records)
        if self.min_confidence:
            proba = self._clf.predict_proba(X)
            labels = self._clf.classes_[proba.argmax(axis=1)]
            return ["other" if p.max() < self.min_confidence else str(lab) for p, lab in zip(proba, labels)]
        return [str(lab) for lab in self._clf.predict(X)]

    def predict(self, record: FileRecord) -> str:
        return self.predict_many([record])[0]

    def predict_proba(self, records: Sequence[FileRecord]) -> list[dict[str, float]]:
        """Class probabilities per record (logreg only); the maximum is the confidence."""
        self._check_fitted()
        if self.model != "logreg":
            raise NotImplementedError("probabilities need model='logreg'")
        proba = self._clf.predict_proba(self._matrix(records))
        return [dict(zip(map(str, self._clf.classes_), map(float, row))) for row in proba]

    # ------------------------------------------------------------ analysis
    def top_features(self, label: str, n: int = 15) -> list[tuple[str, float]]:
        """The n terms that push a document towards ``label`` the most (for error analysis)."""
        self._check_fitted()
        classes = list(self._clf.classes_)
        if label not in classes:
            raise ValueError(f"unknown label '{label}'; trained on {classes}")
        coef = self._clf.coef_
        if len(classes) > 2:
            row = coef[classes.index(label)]
        else:  # binary: one weight vector, for the second class
            row = coef[0] if classes.index(label) == 1 else -coef[0]
        names = [f"{g}:{t}" for g in self.features if self._vectorizers.get(g) is not None
                 for t in self._vectorizers[g].get_feature_names_out()]
        order = np.argsort(row)[::-1][:n]
        return [(names[i], float(row[i])) for i in order]

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MLClassifier":
        """Load a saved model. Only load files you created yourself (pickle format)."""
        obj = joblib.load(path)
        if not isinstance(obj, MLClassifier):
            raise TypeError(f"{path} does not contain an MLClassifier")
        return obj