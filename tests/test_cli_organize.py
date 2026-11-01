import builtins
from pathlib import Path

import pytest

from fileorg.cli import main

INVOICE = "INVOICE\nInvoice number: 17\nDue date: 3 May\nBill to\nAcme\nSubtotal: 10\nTotal due: 12\nPayment terms: net 30 days"
LETTER = "Anna Smith\n5 Oak Road\nDear Sir or Madam,\nI am writing to complain about my order.\nYours faithfully,\nAnna Smith"
RECIPE = "Chocolate cake\nIngredients: flour, cocoa\nPreheat the oven. Servings: 8"


def snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*"))
            if p.is_file() and ".fileorg" not in p.parts}


@pytest.fixture
def inbox(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    (root / "scan_001.txt").write_text(INVOICE)
    (root / "scan_002.txt").write_text(LETTER)
    (root / "notes.txt").write_text(RECIPE)
    (root / "empty.txt").write_text("   ")
    return root


def test_preview_is_the_default_and_changes_nothing(inbox, capsys):
    before = snapshot(inbox)
    assert main(["organize", str(inbox)]) == 0
    out = capsys.readouterr().out
    assert "Preview only" in out and "invoice/" in out and "letter/" in out
    assert snapshot(inbox) == before and not (inbox / ".fileorg").exists()


def test_apply_history_and_undo_roundtrip(inbox, capsys):
    before = snapshot(inbox)
    assert main(["organize", str(inbox), "--apply", "--yes"]) == 0
    assert "Moved 3 file(s) (batch 1)" in capsys.readouterr().out
    after = snapshot(inbox)
    assert set(after) == {"invoice/scan_001.txt", "letter/scan_002.txt", "other/notes.txt", "empty.txt"}

    assert main(["history", str(inbox)]) == 0
    assert "active" in capsys.readouterr().out
    assert main(["history", str(inbox), "--batch", "1"]) == 0
    assert "scan_001.txt -> invoice/" in capsys.readouterr().out

    assert main(["undo", str(inbox), "--yes"]) == 0
    assert "Restored 3 file(s)" in capsys.readouterr().out
    assert snapshot(inbox) == before
    assert not (inbox / "invoice").exists()
    assert main(["undo", str(inbox), "--yes"]) == 0
    assert "Nothing to undo" in capsys.readouterr().out
    main(["history", str(inbox)])
    assert "undone" in capsys.readouterr().out


def test_declining_the_confirmation_moves_nothing(inbox, monkeypatch, capsys):
    before = snapshot(inbox)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    assert main(["organize", str(inbox), "--apply"]) == 1
    assert "Aborted" in capsys.readouterr().out and snapshot(inbox) == before
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    assert main(["organize", str(inbox), "--apply"]) == 0
    assert (inbox / "invoice" / "scan_001.txt").exists()


def test_second_run_leaves_organized_files_alone(inbox, capsys):
    main(["organize", str(inbox), "--apply", "--yes"])
    capsys.readouterr()
    assert main(["organize", str(inbox), "--recursive", "--apply", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "already in the right folder" in out and "Nothing to move" in out


def test_destination_folder_option(inbox, tmp_path, capsys):
    dest = tmp_path / "sorted"
    assert main(["organize", str(inbox), "--dest", str(dest), "--apply", "--yes"]) == 0
    assert (dest / "invoice" / "scan_001.txt").exists() and not (inbox / "scan_001.txt").exists()
    assert main(["undo", str(inbox), "--yes"]) == 0
    assert (inbox / "scan_001.txt").exists()


def test_error_messages(tmp_path, inbox, capsys):
    assert main(["organize", str(tmp_path / "missing")]) == 1
    assert "not a folder" in capsys.readouterr().err
    assert main(["organize", str(inbox), "--classifier", "ml"]) == 1
    assert "fileorg train" in capsys.readouterr().err
    assert main(["undo", str(inbox), "--yes"]) == 1
    assert "No move log" in capsys.readouterr().err


def test_train_save_and_organize_with_a_model(tmp_path, capsys):
    pytest.importorskip("sklearn")
    raw = tmp_path / "raw"
    texts = {"invoice": INVOICE, "letter": LETTER, "other": RECIPE}
    for label, text in texts.items():
        (raw / label).mkdir(parents=True)
        for i in range(12):
            (raw / label / f"{label}_{i}.txt").write_text(f"{text}\nreference {i} {label} extra words {i * 7}")
    records, split, model = tmp_path / "r.jsonl", tmp_path / "split.csv", tmp_path / "models" / "ml.joblib"
    assert main(["scan", str(raw), "--labelled", "-o", str(records)]) == 0
    assert main(["split", str(records), "-o", str(split)]) == 0
    assert main(["train", str(records), "--split", str(split), "--classifier", "ml", "--out", str(model)]) == 0
    assert model.exists()
    capsys.readouterr()

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "x1.txt").write_text(INVOICE + "\nreference 99 invoice")
    (inbox / "x2.txt").write_text(LETTER + "\nreference 98 letter")
    assert main(["organize", str(inbox), "--classifier", "ml", "--model-file", str(model), "--apply", "--yes"]) == 0
    out = capsys.readouterr().out
    assert (inbox / "invoice" / "x1.txt").exists() and (inbox / "letter" / "x2.txt").exists()
    assert "conf" in out or "Moved 2" in out
    assert main(["undo", str(inbox), "--yes"]) == 0
    assert (inbox / "x1.txt").exists()
    assert main(["train", str(records), "--classifier", "rules"]) == 1  # rules need no training