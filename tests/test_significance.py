import pytest

from fileorg.evaluation import bootstrap_ci, compare_classifiers, format_comparison, mcnemar, paired_bootstrap


def test_mcnemar_exact_values():
    truth = ["a"] * 8
    always_right = ["a"] * 8
    wrong_on_5 = ["a"] * 3 + ["b"] * 5
    r = mcnemar(truth, always_right, wrong_on_5)
    assert (r.only_a, r.only_b, r.both_correct, r.both_wrong) == (5, 0, 3, 0)
    assert r.p_value == pytest.approx(2 / 32)  # 2 * P(X <= 0) with n = 5
    tie = mcnemar(["a"] * 6, ["a", "a", "a", "b", "b", "b"], ["b", "b", "b", "a", "a", "a"])
    assert tie.p_value == 1.0
    assert mcnemar(["a", "b"], ["a", "b"], ["a", "b"]).p_value == 1.0  # no discordant pairs


def test_bootstrap_ci_properties():
    truth = ["a", "b"] * 30
    perfect = list(truth)
    ci = bootstrap_ci(perfect, perfect)
    assert (ci.point, ci.low, ci.high) == (1.0, 1.0, 1.0)
    noisy = [t if i % 5 else "z" for i, t in enumerate(truth)]
    ci = bootstrap_ci(truth, noisy, "accuracy")
    assert ci.low <= ci.point <= ci.high and ci.point == pytest.approx(0.8)
    assert bootstrap_ci(truth, noisy, "accuracy") == ci  # same seed, same answer
    f1 = bootstrap_ci(truth, noisy, "macro_f1")
    assert 0 <= f1.low <= f1.point <= f1.high <= 1
    with pytest.raises(ValueError):
        bootstrap_ci(truth, noisy, "bogus")
    with pytest.raises(ValueError):
        bootstrap_ci([], [])


def test_group_bootstrap_is_wider_for_correlated_files():
    # 10 groups of 10 identical files; one whole group is always wrong
    truth, pred, groups = [], [], []
    for g in range(10):
        for _ in range(10):
            truth.append("a")
            pred.append("b" if g == 0 else "a")
            groups.append(f"t{g}")
    plain = bootstrap_ci(truth, pred, "accuracy")
    grouped = bootstrap_ci(truth, pred, "accuracy", groups=groups)
    assert (grouped.high - grouped.low) > (plain.high - plain.low)


def test_paired_bootstrap_detects_real_and_absent_differences():
    truth = ["a", "b", "c"] * 40
    good = list(truth)
    bad = [t if i % 3 else "x" for i, t in enumerate(truth)]
    diff = paired_bootstrap(truth, good, bad)
    assert diff.diff == pytest.approx(1 / 3) and diff.low > 0 and diff.p_value < 0.01
    same = paired_bootstrap(truth, bad, list(bad))
    assert (same.diff, same.low, same.high, same.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_compare_classifiers_and_report_text():
    truth = ["a", "b", "c"] * 20
    preds = {"rules": list(truth), "ml": [t if i % 4 else "x" for i, t in enumerate(truth)],
             "emb": [t if i % 6 else "x" for i, t in enumerate(truth)]}
    c = compare_classifiers(truth, preds, n_boot=300)
    assert len(c.pairs) == 3 and c.adjusted_alpha == pytest.approx(0.05 / 3)
    assert c.intervals["rules"]["accuracy"].point == 1.0
    text = format_comparison(c)
    assert "rules vs ml" in text and "Bonferroni" in text and "template groups" not in text
    grouped = compare_classifiers(truth, preds, groups=[i % 6 for i in range(60)], n_boot=100)
    assert grouped.n_groups == 6 and "6 template groups" in format_comparison(grouped)
    with pytest.raises(ValueError):
        compare_classifiers(truth, {"a": truth[:-1]})