import pytest

from fileorg.classifiers.base import Classifier
from fileorg.evaluation import (
    apply_split, classification_report, cohen_kappa, cross_validate, final_test,
    format_confusion, load_groups, make_split, write_split,
)
from fileorg.models import FileRecord

CATS = ["invoice", "letter", "report"]


def rec(rel, label, text=""):
    return FileRecord(path="/" + rel, rel_path=rel, filename=rel.split("/")[-1], extension=".txt",
                      size_bytes=1, modified="", ctime="", mime_type=None, text=text, label=label)


def dataset(per_class=20):
    return [rec(f"{c}/{i:03d}.txt", c, text=c) for c in CATS for i in range(per_class)]


def counts(records):
    out = {}
    for r in records:
        out[r.label] = out.get(r.label, 0) + 1
    return out


# ------------------------------------------------------------------ split
def test_split_is_stratified_disjoint_and_complete():
    records = dataset()
    s = make_split(records)
    assert counts(s.test) == {c: 4 for c in CATS}
    assert counts(s.dev) == {c: 16 for c in CATS}
    assert not {r.rel_path for r in s.dev} & {r.rel_path for r in s.test}
    assert len(s.dev) + len(s.test) == len(records)


def test_split_is_deterministic_and_order_independent():
    records = dataset()
    a = make_split(records, seed=7)
    b = make_split(list(reversed(records)), seed=7)
    c = make_split(records, seed=8)
    assert [r.rel_path for r in a.test] == [r.rel_path for r in b.test]
    assert a.folds == b.folds
    assert [r.rel_path for r in a.test] != [r.rel_path for r in c.test]


def test_folds_are_balanced_per_class_and_cover_dev_once():
    s = make_split(dataset())
    assert s.n_folds == 5
    for c in CATS:
        per_fold = [sum(1 for r, f in zip(s.dev, s.folds) if r.label == c and f == k) for k in range(5)]
        assert max(per_fold) - min(per_fold) <= 1
    seen = []
    for train, val in s.cv_splits():
        assert not {r.rel_path for r in train} & {r.rel_path for r in val}
        seen += [r.rel_path for r in val]
    assert sorted(seen) == sorted(r.rel_path for r in s.dev)


def test_groups_never_cross_split_or_folds():
    records = dataset(30)
    groups = {r.rel_path: f"{r.label}-t{int(r.rel_path[-7:-4]) % 6}" for r in records}  # 6 templates/class
    s = make_split(records, groups=groups, folds=3)
    test_groups = {groups[r.rel_path] for r in s.test}
    dev_groups = {groups[r.rel_path] for r in s.dev}
    assert not test_groups & dev_groups
    fold_of = {}
    for r, f in zip(s.dev, s.folds):
        assert fold_of.setdefault(groups[r.rel_path], f) == f


def test_split_errors():
    with pytest.raises(ValueError):
        make_split([rec("a.txt", None), rec("b.txt", None)])
    with pytest.raises(ValueError):
        make_split(dataset(3), folds=5)  # too few files for 5 folds
    with pytest.raises(ValueError):
        make_split([rec("invoice/1.txt", "invoice")], folds=0)


def test_split_file_roundtrip_and_mismatch(tmp_path):
    records = dataset()
    s = make_split(records)
    path = tmp_path / "split.csv"
    write_split(s, path)
    s2 = apply_split(records, path)
    assert [r.rel_path for r in s2.test] == [r.rel_path for r in s.test]
    assert [r.rel_path for r in s2.dev] == [r.rel_path for r in s.dev]
    assert s2.folds == s.folds
    with pytest.raises(KeyError):
        apply_split(records[:-1], path)
    with pytest.raises(KeyError):
        apply_split(records + [rec("new/x.txt", "report")], path)


def test_load_groups(tmp_path):
    f = tmp_path / "sources.csv"
    f.write_text("file,source,template_id\ninvoice\\a.txt,x,T1\nletter/b.txt,y,\n", encoding="utf-8")
    assert load_groups(f) == {"invoice/a.txt": "T1"}


# ---------------------------------------------------------------- metrics
def test_metrics_hand_computed():
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "c", "c"]
    r = classification_report(y_true, y_pred)
    assert r.accuracy == pytest.approx(4 / 6)
    assert r.per_class["a"].precision == pytest.approx(1.0)
    assert r.per_class["a"].recall == pytest.approx(2 / 3)
    assert r.per_class["b"].precision == pytest.approx(1 / 2)
    assert r.per_class["b"].recall == pytest.approx(1 / 2)
    assert r.per_class["c"].precision == pytest.approx(1 / 2)
    assert r.per_class["c"].recall == pytest.approx(1.0)
    assert r.macro.f1 == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert r.weighted.recall == pytest.approx(r.accuracy)  # weighted recall == accuracy
    assert r.confusion == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert "macro avg" in r.format() and "invoice" not in r.format()
    assert format_confusion(r).count("\n") == 3


def test_metrics_zero_division_and_unknown_prediction():
    r = classification_report(["a", "a"], ["unknown", "unknown"], labels=["a", "other"])
    assert r.accuracy == 0.0
    assert r.per_class["a"].precision == 0.0 and r.per_class["a"].f1 == 0.0
    assert r.per_class["other"].support == 0


def test_metrics_validation():
    with pytest.raises(ValueError):
        classification_report([], [])
    with pytest.raises(ValueError):
        classification_report(["a"], ["a", "b"])


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    a = ["x"] * 45 + ["y"] * 5 + ["x"] * 25 + ["y"] * 25
    b = ["x"] * 45 + ["x"] * 5 + ["y"] * 25 + ["y"] * 25  # classic example, kappa = 0.4
    assert cohen_kappa(a, b) == pytest.approx(0.0 + 0.4, abs=1e-9)


# ----------------------------------------------------------------- runner
class KeywordClassifier(Classifier):
    name = "keyword"

    def fit(self, records):
        self.seen = len(records)

    def predict(self, record):
        return record.text


def test_runner_cv_and_final_test():
    s = make_split(dataset())
    cv = cross_validate(KeywordClassifier, s)
    assert len(cv.folds) == 5 and cv.pooled.n == len(s.dev)
    assert cv.mean_accuracy == 1.0 and cv.mean_macro_f1 == 1.0
    final = final_test(KeywordClassifier, s)
    assert final.report.n == len(s.test) and final.report.accuracy == 1.0
    assert final.fit_seconds >= 0 and final.predict_seconds >= 0