import random

import pytest

pytest.importorskip("sklearn")

from fileorg.classifiers import get_classifier
from fileorg.classifiers.ml import MLClassifier
from fileorg.evaluation import cross_validate, expand_grid, grid_search, make_split
from fileorg.models import FileRecord

VOCAB = {
    "invoice": "invoice total due vat bill payment terms subtotal iban amount customer".split(),
    "letter": "dear sincerely regards writing enclosed look forward faithfully recipient address".split(),
    "report": "executive summary findings recommendations objectives stakeholders quarter chart scope".split(),
}
COMMON = "the of and to in for with on this that is are was".split()


def make_records(per_class=15, seed=0):
    rng = random.Random(seed)
    out = []
    for label, words in VOCAB.items():
        for i in range(per_class):
            text = " ".join(rng.choices(words, k=12) + rng.choices(COMMON, k=20))
            out.append(FileRecord(path=f"/{label}/{i}.txt", rel_path=f"{label}/{i:02d}.txt", filename=f"{label}_{i}.txt",
                                  extension=".txt", size_bytes=1, modified="", ctime="", mime_type=None,
                                  text=text, metadata={"title": label} if i % 2 else {}, label=label))
    return out


@pytest.mark.parametrize("model", ["logreg", "svm"])
def test_learns_and_predicts_both_models(model):
    train, test = make_records(seed=1), make_records(seed=2)
    clf = MLClassifier(model=model)
    clf.fit(train)
    preds = clf.predict_many(test)
    assert sum(p == r.label for p, r in zip(preds, test)) / len(test) > 0.95
    assert clf.predict(test[0]) == preds[0]
    assert clf.predict_many([]) == []


def test_deterministic():
    train, test = make_records(), make_records(seed=3)
    a, b = MLClassifier(), MLClassifier()
    a.fit(train), b.fit(train)
    assert a.predict_many(test) == b.predict_many(test)


def test_probabilities_and_min_confidence():
    train, test = make_records(), make_records(seed=4)
    clf = MLClassifier()
    clf.fit(train)
    proba = clf.predict_proba(test[:3])
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in proba)
    assert max(proba[0], key=proba[0].get) == clf.predict(test[0])
    strict = MLClassifier(min_confidence=0.999)
    strict.fit(train)
    assert set(strict.predict_many(test)) == {"other"}
    with pytest.raises(ValueError):
        MLClassifier(model="svm", min_confidence=0.5)
    svm = MLClassifier(model="svm")
    svm.fit(train)
    with pytest.raises(NotImplementedError):
        svm.predict_proba(test[:1])


def test_feature_groups():
    train = make_records()
    clf = MLClassifier(features=["filename"])
    clf.fit(train)
    # the file name alone is enough here, whatever the text says
    odd = FileRecord(path="/x", rel_path="x", filename="report_final.txt", extension=".txt", size_bytes=1, modified="",
                     ctime="", mime_type=None, text="invoice total due vat", label=None)
    assert clf.predict(odd) == "report"
    with pytest.raises(ValueError):
        MLClassifier(features=["bogus"])
    # metadata that is empty everywhere must not crash training
    bare = [FileRecord(**{**r.to_dict(), "metadata": {}}) for r in train]
    MLClassifier(features=["text", "metadata"]).fit(bare)


def test_top_features_and_errors():
    clf = MLClassifier()
    clf.fit(make_records())
    terms = [t for t, _ in clf.top_features("invoice", 10)]
    assert any(t.startswith("text:") and t.split(":")[1] in VOCAB["invoice"] for t in terms)
    with pytest.raises(ValueError):
        clf.top_features("nope")
    with pytest.raises(RuntimeError):
        MLClassifier().predict(make_records()[0])
    with pytest.raises(ValueError):
        MLClassifier().fit([])
    unlabeled = [FileRecord(**{**r.to_dict(), "label": None}) for r in make_records()]
    with pytest.raises(ValueError):
        MLClassifier().fit(unlabeled)
    with pytest.raises(ValueError):
        MLClassifier().fit([r for r in make_records() if r.label == "invoice"])


def test_save_and_load_roundtrip(tmp_path):
    train, test = make_records(), make_records(seed=5)
    clf = MLClassifier()
    clf.fit(train)
    clf.save(tmp_path / "model.joblib")
    assert MLClassifier.load(tmp_path / "model.joblib").predict_many(test) == clf.predict_many(test)


def test_registry_cross_validation_and_grid_search():
    assert get_classifier("ml", C=2.0).C == 2.0
    split = make_split(make_records(per_class=20))
    cv = cross_validate(lambda: get_classifier("ml"), split)
    assert cv.mean_accuracy > 0.9 and len(cv.errors()) <= len(cv.records)

    assert expand_grid({"a": [1, 2], "b": ["x", "y", "z"]}).__len__() == 6
    results = grid_search(lambda **p: get_classifier("ml", **p), {"C": [0.01, 1.0], "model": ["logreg", "svm"]}, split)
    assert len(results) == 4
    scores = [r.mean_macro_f1 for r in results]
    assert scores == sorted(scores, reverse=True)