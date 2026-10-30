import json

import pytest

from fileorg.classifiers.base import Classifier
from fileorg.classifiers.rule_based import RuleBasedClassifier
from fileorg.evaluation import build_sample_folder, format_benchmark, run_benchmark
from fileorg.models import FileRecord
from fileorg.scanner import scan


def make_source(tmp_path, n=6):
    src = tmp_path / "raw"
    for label in ("invoice", "letter"):
        (src / label).mkdir(parents=True)
        for i in range(n // 2):
            (src / label / f"{label}_{i}.txt").write_text(f"{label.upper()}\nInvoice number {i}\nDear Sir", encoding="utf-8")
    return src


def train_records():
    return [FileRecord(path="/a", rel_path="a", filename="a.txt", extension=".txt", size_bytes=1, modified="", ctime="",
                       mime_type=None, text="invoice", label="invoice")]


class Counting(Classifier):
    name = "counting"

    def __init__(self):
        self.fitted, self.predicted = 0, 0

    def fit(self, records):
        self.fitted = len(records)

    def predict(self, record):
        self.predicted += 1
        return "other"


def test_sample_folder_is_repeatable_with_repetition(tmp_path):
    src = scan(make_source(tmp_path))
    a = build_sample_folder(src, tmp_path / "a", 20, seed=1)
    b = build_sample_folder(src, tmp_path / "b", 20, seed=1)
    c = build_sample_folder(src, tmp_path / "c", 20, seed=2)
    assert a == b and a != c
    assert len(a) == 20 and len(set(a)) <= len(src)  # more files than sources: repetition
    assert len(list((tmp_path / "a").iterdir())) == 20  # copies get unique names
    assert len({p.name for p in scan(tmp_path / "a")}) == 20
    with pytest.raises(ValueError):
        build_sample_folder([], tmp_path / "d", 5, seed=1)


def test_benchmark_structure_cleanup_and_json(tmp_path):
    src, work = make_source(tmp_path), tmp_path / "work"
    messages = []
    result = run_benchmark(src, [5, 12], {"rules": RuleBasedClassifier, "counting": Counting}, train_records(), work,
                           repeats=2, seed=7, log=messages.append)
    assert [s.n for s in result.sizes] == [5, 12]
    assert set(result.fit_seconds) == {"rules", "counting"} and set(result.warmup_seconds) == {"rules", "counting"}
    for s in result.sizes:
        assert s.statuses == {"ok": s.n}
        assert len(s.scan) == len(s.extraction) == len(s.other) == 2
        assert all(len(v) == 2 for v in s.classify.values())
        assert s.total("rules") >= s.median("classify", "rules") > 0
    assert not list(work.glob("n*"))  # sampled folders removed
    assert any("building" in m for m in messages)
    d = result.to_dict()
    json.dumps(d)  # serialisable
    assert d["sizes"][1]["n"] == 12 and "total" in d["sizes"][0]["median"]
    text = format_benchmark(result)
    assert "rules" in text and "ms/file" in text and "Environment" in text


def test_keep_flag_trains_once_and_warms_up_once(tmp_path):
    src, work = make_source(tmp_path), tmp_path / "work"
    counting = Counting()
    run_benchmark(src, [4], {"counting": lambda: counting}, train_records(), work, repeats=3, keep=True)
    assert (work / "n4").is_dir()
    assert counting.fitted == 1
    assert counting.predicted == 1 + 3 * 4  # one warm-up record + 3 repeats of 4 files


def test_benchmark_validation(tmp_path):
    with pytest.raises(ValueError):
        run_benchmark(tmp_path, [5], {}, train_records(), tmp_path / "w")  # nothing to sample
    with pytest.raises(ValueError):
        run_benchmark(make_source(tmp_path), [5], {}, train_records(), tmp_path / "w", repeats=0)