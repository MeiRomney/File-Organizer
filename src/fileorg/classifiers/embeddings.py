"""Embedding classifier: sentence embeddings + a linear model on top.

A pre-trained sentence-embedding model turns the beginning of a document into
a dense vector; a logistic regression (or linear SVM) learns the categories
from those vectors. The embedding model itself is *not* fine-tuned.

* ``text``     - up to ``max_chunks`` chunks of ``chunk_chars`` characters from
                 the start of the text, embedded separately and averaged.
                 A sentence-embedding model reads only a limited number of
                 tokens (256 for all-MiniLM-L6-v2) and silently truncates the
                 rest, so this is a deliberate, documented limit.
* ``filename`` / ``metadata`` - embedded as short separate strings, scaled by
                 0.5 and concatenated (same idea as the TF-IDF classifier).

Embeddings are cached per string (in memory, optionally on disk), so
cross-validation and tuning encode every document only once. For the timing
experiment use a fresh, uncached encoder.

The ``sentence-transformers`` package is imported lazily; the model is
downloaded on first use (about 90 MB for the default model).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePath
from typing import Iterable, Protocol, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FEATURE_GROUPS = ("text", "filename", "metadata")
GROUP_WEIGHT = {"text": 1.0, "filename": 0.5, "metadata": 0.5}
META_KEYS = ("title", "subject", "keywords", "category")


# ------------------------------------------------------------------ encoders
class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalised embeddings."""


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, device: str | None = None) -> None:
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "the embedding classifier needs sentence-transformers: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._load().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class CachedEncoder:
    """Remember the embedding of every string; optionally persist the cache to disk."""

    def __init__(self, encoder: Encoder, path: str | Path | None = None, signature: str = "") -> None:
        self.encoder, self.signature = encoder, signature
        self.path = Path(path) if path else None
        self._cache: dict[str, np.ndarray] = {}
        self.encoded = 0  # number of strings that really went through the model
        if self.path and self.path.exists():
            stored = joblib.load(self.path)
            if stored.get("signature") == signature:  # never mix embeddings of different models
                self._cache = stored["vectors"]

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if missing:
            vectors = self.encoder.encode(list(missing.values()))
            for key, vec in zip(missing, vectors):
                self._cache[key] = vec
            self.encoded += len(missing)
        return np.vstack([self._cache[k] for k in keys])

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"signature": self.signature, "vectors": self._cache}, self.path)

    def clear(self) -> None:
        self._cache.clear()


_SHARED: dict[tuple, CachedEncoder] = {}


def shared_encoder(model_name: str = DEFAULT_MODEL, cache_path: str | Path | None = None) -> CachedEncoder:
    """One encoder (and one loaded model) per process, so folds and tuning runs reuse it."""
    key = (model_name, str(cache_path))
    if key not in _SHARED:
        _SHARED[key] = CachedEncoder(SentenceTransformerEncoder(model_name), cache_path, signature=model_name)
    return _SHARED[key]


# ---------------------------------------------------------------- classifier
class EmbeddingClassifier(Classifier):
    name = "embeddings"

    def __init__(
        self,
        features: Iterable[str] = FEATURE_GROUPS,
        model: str = "logreg",
        C: float = 10.0,
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        cache_path: str | Path | None = None,
        max_chars: int = 3000,
        chunk_chars: int = 1000,
        max_chunks: int = 3,
        min_confidence: float = 0.0,
        random_state: int = 42,
    ) -> None:
        features = tuple(f for f in FEATURE_GROUPS if f in set(features))
        if not features:
            raise ValueError(f"choose at least one feature group from {FEATURE_GROUPS}")
        if model not in ("logreg", "svm"):
            raise ValueError("model must be 'logreg' or 'svm'")
        if min_confidence and model != "logreg":
            raise ValueError("min_confidence needs model='logreg' (probabilities)")
        self.features, self.model, self.C = features, model, C
        self.model_name, self.cache_path = model_name, cache_path
        self.max_chars, self.chunk_chars, self.max_chunks = max_chars, chunk_chars, max_chunks
        self.min_confidence, self.random_state = min_confidence, random_state
        self._encoder = encoder
        self._clf = None
        self._groups: tuple[str, ...] = ()
        self._dims: dict[str, int] = {}

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = shared_encoder(self.model_name, self.cache_path)
        return self._encoder

    # ----------------------------------------------------------- features
    def _chunks(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()[: self.max_chars]
        pieces = [text[i:i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)]
        return pieces[: self.max_chunks]

    def _side_text(self, record: FileRecord, group: str) -> str:
        if group == "filename":
            return re.sub(r"[^a-z]+", " ", PurePath(record.filename).stem.lower()).strip()
        meta = record.metadata or {}
        return " ".join(str(meta.get(k, "")) for k in META_KEYS).strip()

    def _group_vectors(self, records: Sequence[FileRecord], group: str) -> np.ndarray | None:
        if group == "text":
            pieces = [self._chunks(r.text or "") for r in records]
        else:
            pieces = [[s] if s else [] for s in (self._side_text(r, group) for r in records)]
        flat = [p for ps in pieces for p in ps]
        if not flat:  # nothing to embed in this group
            dim = self._dims.get(group)
            return None if dim is None else np.zeros((len(records), dim), dtype=np.float32)
        embedded = self.encoder.encode(flat)
        self._dims[group] = embedded.shape[1]
        out = np.zeros((len(records), embedded.shape[1]), dtype=np.float32)
        pos = 0
        for i, ps in enumerate(pieces):
            if ps:
                mean = embedded[pos:pos + len(ps)].mean(axis=0)
                norm = np.linalg.norm(mean)
                out[i] = mean / norm if norm else mean
                pos += len(ps)
        return out

    def _matrix(self, records: Sequence[FileRecord]) -> np.ndarray:
        blocks = []
        for group in self._groups:
            vectors = self._group_vectors(records, group)
            if vectors is None:  # cannot happen after fit(); keeps the dimensions consistent
                vectors = np.zeros((len(records), self._dims[group]), dtype=np.float32)
            blocks.append(vectors * GROUP_WEIGHT[group])
        return np.hstack(blocks)

    # ------------------------------------------------------------ training
    def fit(self, records: Sequence[FileRecord]) -> None:
        if not records:
            raise ValueError("cannot train on an empty list")
        labels = [r.label for r in records]
        if None in labels:
            raise ValueError("all training records need a label")
        if len(set(labels)) < 2:
            raise ValueError("need at least two classes to train")

        self._dims, active, blocks = {}, [], []
        for group in self.features:
            vectors = self._group_vectors(records, group)
            if vectors is not None:
                active.append(group)
                blocks.append(vectors * GROUP_WEIGHT[group])
        if not blocks:
            raise ValueError("no usable features: all selected feature groups are empty")
        self._groups = tuple(active)
        X = np.hstack(blocks)

        if self.model == "logreg":
            self._clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced",
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
        self._check_fitted()
        if self.model != "logreg":
            raise NotImplementedError("probabilities need model='logreg'")
        proba = self._clf.predict_proba(self._matrix(records))
        return [dict(zip(map(str, self._clf.classes_), map(float, row))) for row in proba]

    # --------------------------------------------------------- persistence
    def _params(self) -> dict:
        return dict(features=self.features, model=self.model, C=self.C, model_name=self.model_name,
                    cache_path=self.cache_path, max_chars=self.max_chars, chunk_chars=self.chunk_chars,
                    max_chunks=self.max_chunks, min_confidence=self.min_confidence, random_state=self.random_state)

    def save(self, path: str | Path) -> None:
        """Save the trained head only (the embedding model is re-loaded by name)."""
        self._check_fitted()
        joblib.dump({"params": self._params(), "clf": self._clf, "groups": self._groups, "dims": self._dims}, path)

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder | None = None) -> "EmbeddingClassifier":
        """Load a saved model. Only load files you created yourself (pickle format)."""
        data = joblib.load(path)
        obj = cls(**data["params"], encoder=encoder)
        obj._clf, obj._groups, obj._dims = data["clf"], data["groups"], data["dims"]
        return obj