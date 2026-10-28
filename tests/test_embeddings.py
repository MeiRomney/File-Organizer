import random
import re

import numpy as np
import pytest

pytest.importorskip("sklearn")

from fileorg.classifiers import get_classifier
from fileorg.classifiers.embeddings import CachedEncoder, EmbeddingClassifier, SentenceTransformerEncoder
from fileorg.evaluation import cross_validate, make_split
from fileorg.models import FileRecord

VOCAB = {
    "invoice": "invoice total due vat bill payment terms subtotal iban amount customer".split(),
    "letter": "dear sincerely regards writing enclosed look forward faithfully recipient address".split(),
    "report": "executive summary findings recommendations objectives stakeholders quarter chart scope".split(),
}
COMMON = "the of and to in for with on this that is are was".split()


class FakeEncoder:
    """Hashed bag-of-words embeddings: deterministic, no model download."""

    def __init__(self, dim=96):
        self.dim, self.calls, self.seen = dim, 0, 0

    def encode(self, texts):
        self.calls += 1
        self.seen += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, sum(map(ord, word)) % self.dim] += 1
            norm = np.linalg.norm(out[i])
            out[i] = out[i] / norm if norm else out[i]
        return out


def make_records(per_class=15, seed=0, with_meta=True):
    rng = random.Random(seed)
    out = []
    for label, words in VOCAB.items():
        for i in range(per_class):
            text = " ".join(rng.choices(words, k=60) + rng.choices(COMMON, k=60))
            out.append(FileRecord(path=f"/{label}/{i}.txt", rel_path=f"{label}/{i:02d}.txt", filename=f"{label}_{i}.txt",
                                  extension=".txt", size_bytes=1, modified="", ctime="", mime_type=None, text=text,
                                  metadata={"title": label} if with_meta and i % 2 else {}, label=label))
    return out


def test_learns_with_both_models_and_is_deterministic():
    train, test = make_records(seed=1), make_records(seed=2)
    for model in ("logreg", "svm"):
        a = EmbeddingClassifier(model=model, encoder=FakeEncoder())
        a.fit(train)
        preds = a.predict_many(test)
        assert sum(p == r.label for p, r in zip(preds, test)) / len(test) > 0.9
        b = EmbeddingClassifier(model=model, encoder=FakeEncoder())
        b.fit(train)
        assert b.predict_many(test) == preds


def test_chunking_limits_text_that_reaches_the_encoder():
    clf = EmbeddingClassifier(encoder=FakeEncoder(), max_chars=3000, chunk_chars=1000, max_chunks=2)
    assert clf._chunks("") == []
    assert len(clf._chunks("word " * 5000)) == 2
    assert all(len(c) <= 1000 for c in clf._chunks("word " * 5000))
    assert len(clf._chunks("short text")) == 1


def test_cached_encoder_encodes_each_string_once(tmp_path):
    inner = FakeEncoder()
    cache = CachedEncoder(inner, tmp_path / "cache.joblib", signature="fake-1")
    first = cache.encode(["alpha beta", "gamma", "alpha beta"])
    assert inner.seen == 2 and first.shape == (3, 96)
    cache.encode(["gamma", "alpha beta"])
    assert inner.seen == 2  # nothing new
    cache.save()
    reloaded = CachedEncoder(FakeEncoder(), tmp_path / "cache.joblib", signature="fake-1")
    other_model = CachedEncoder(FakeEncoder(), tmp_path / "cache.joblib", signature="fake-2")
    reloaded.encode(["alpha beta"])
    other_model.encode(["alpha beta"])
    assert reloaded.encoded == 0 and other_model.encoded == 1  # a different model never reuses the cache


def test_cross_validation_encodes_every_document_once():
    records = make_records(per_class=20)
    split = make_split(records)
    inner = FakeEncoder()
    shared = CachedEncoder(inner)
    cv = cross_validate(lambda: EmbeddingClassifier(encoder=shared), split)
    assert cv.mean_accuracy > 0.9
    assert inner.seen <= len(records) * 3  # text chunks + file names + titles, not 5x for 5 folds


def test_feature_groups_and_missing_metadata():
    train = make_records(with_meta=False)
    EmbeddingClassifier(features=["text", "metadata"], encoder=FakeEncoder()).fit(train)  # empty group is dropped
    clf = EmbeddingClassifier(features=["filename"], encoder=FakeEncoder())
    clf.fit(train)
    odd = FileRecord(path="/x", rel_path="x", filename="report_final.txt", extension=".txt", size_bytes=1, modified="",
                     ctime="", mime_type=None, text="invoice total due", label=None)
    assert clf.predict(odd) == "report"
    with pytest.raises(ValueError):
        EmbeddingClassifier(features=["bogus"])
    with pytest.raises(ValueError):
        EmbeddingClassifier(model="svm", min_confidence=0.5)


def test_probabilities_min_confidence_errors_and_persistence(tmp_path):
    train, test = make_records(), make_records(seed=3)
    clf = EmbeddingClassifier(encoder=FakeEncoder())
    clf.fit(train)
    proba = clf.predict_proba(test[:3])
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in proba)
    strict = EmbeddingClassifier(encoder=FakeEncoder(), min_confidence=0.999)
    strict.fit(train)
    assert set(strict.predict_many(test)) == {"other"}
    clf.save(tmp_path / "head.joblib")
    loaded = EmbeddingClassifier.load(tmp_path / "head.joblib", encoder=FakeEncoder())
    assert loaded.predict_many(test) == clf.predict_many(test)
    with pytest.raises(RuntimeError):
        EmbeddingClassifier(encoder=FakeEncoder()).predict(test[0])
    with pytest.raises(ValueError):
        EmbeddingClassifier(encoder=FakeEncoder()).fit([r for r in train if r.label == "invoice"])


def test_registry_and_helpful_import_error():
    assert get_classifier("embeddings", encoder=FakeEncoder(), C=3.0).C == 3.0
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install sentence-transformers"):
            SentenceTransformerEncoder().encode(["x"])