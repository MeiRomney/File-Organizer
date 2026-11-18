import pytest

from fileorg.desktop.session import analyse, apply_plan, load_classifier, log_path
from fileorg.organizer import MoveLog, undo_batch

INVOICE = "INVOICE\nInvoice number: 17\nDue date: 3 May\nBill to\nAcme\nSubtotal: 10\nTotal due: 12\nPayment terms: net 30 days"
LETTER = "Anna Smith\n5 Oak Road\nDear Sir or Madam,\nI am writing to complain about my order.\nYours faithfully,\nAnna Smith"
RECIPE = "Chocolate cake\nIngredients: flour, cocoa\nPreheat the oven. Servings: 8"


@pytest.fixture
def inbox(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    (root / "a.txt").write_text(INVOICE)
    (root / "b.txt").write_text(LETTER)
    (root / "c.txt").write_text(RECIPE)
    (root / "empty.txt").write_text("   ")
    return root


def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".fileorg" not in p.parts}


def test_preview_edit_apply_and_undo(inbox):
    before = files(inbox)
    s = analyse(inbox)
    assert [r.record.filename for r in s.rows] == ["a.txt", "b.txt", "c.txt", "empty.txt"]
    assert [o.category for o in s.outcomes()][:3] == ["invoice", "letter", "other"]
    assert s.outcomes()[3].action == "skip"            # unreadable stays put by default
    assert files(inbox) == before                      # analysing never touches the disk

    s.set_include([0], False)                          # keep a.txt where it is
    s.set_label([2], "report")                         # user decides c.txt is a report
    s.set_label([3], "other")                          # and moves the unreadable file anyway
    out = s.outcomes()
    assert out[0].action == "excluded" and out[2].category == "report" and out[3].action == "move"
    assert "3 to move" in s.summary() and "1 excluded by you" in s.summary()

    result = apply_plan(s)
    assert (result.done, result.failed) == (3, 0)
    assert files(inbox) == {"a.txt", "letter/b.txt", "report/c.txt", "other/empty.txt"}

    with MoveLog(log_path(inbox)) as log:
        assert undo_batch(log).restored == 3
    assert files(inbox) == before


def test_reset_restores_the_classifiers_choice(inbox):
    s = analyse(inbox)
    s.set_label([1], "contract")
    s.set_include([1], False)
    s.reset([1])
    assert s.rows[1].label == "letter" and s.rows[1].include and not s.rows[1].overridden


def test_destination_folder_and_errors(inbox, tmp_path):
    s = analyse(inbox, dest=tmp_path / "sorted")
    apply_plan(s)
    assert (tmp_path / "sorted" / "invoice" / "a.txt").exists()
    with pytest.raises(ValueError):
        analyse(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_classifier("ml")                          # needs a model file
    with pytest.raises(ValueError):
        s.set_label([0], "  ")


def test_window_builds_if_a_display_exists():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    try:
        from fileorg.desktop.app import App

        App(root).grid()
        root.update()
    finally:
        root.destroy()