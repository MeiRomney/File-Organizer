import os
import shutil
from pathlib import Path

import pytest

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord
from fileorg.organizer import (
    MoveLog, Prediction, build_plan, execute_plan, format_plan, predict_with_confidence, undo_batch,
    unique_destination,
)


def rec(path: Path, status="ok") -> FileRecord:
    return FileRecord(path=str(path), rel_path=path.name, filename=path.name, extension=path.suffix, size_bytes=1,
                      modified="", ctime="", mime_type=None, text="x", extraction_status=status)


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under root (except the .fileorg folder) with its content."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".fileorg" not in p.parts}


def folders(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir() and ".fileorg" not in p.parts}


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    for name, text in [("a.txt", "invoice a"), ("b.txt", "letter b"), ("c.txt", "invoice c"), ("d.txt", "misc d")]:
        (root / name).write_text(text)
    return root


def plan_for(root, labels, **kw):
    records = [rec(root / name) for name in labels]
    return build_plan(records, [labels[name] if isinstance(labels[name], Prediction) else Prediction(labels[name])
                                for name in labels], root, **kw)


# ------------------------------------------------------------------ planner
def test_plan_groups_files_into_category_folders(folder):
    plan = plan_for(folder, {"a.txt": "invoice", "b.txt": "letter", "c.txt": "invoice", "d.txt": "other"})
    assert {m.src.name: m.dst.relative_to(folder).as_posix() for m in plan.moves} == {
        "a.txt": "invoice/a.txt", "b.txt": "letter/b.txt", "c.txt": "invoice/c.txt", "d.txt": "other/d.txt"}
    assert plan.category_counts() == {"invoice": 2, "letter": 1, "other": 1}
    assert not plan.skipped
    assert not (folder / "invoice").exists()  # planning never touches the disk


def test_plan_is_deterministic_and_validates(folder):
    labels = {"a.txt": "invoice", "b.txt": "letter"}
    assert plan_for(folder, labels).moves == plan_for(folder, labels).moves
    with pytest.raises(ValueError):
        build_plan([rec(folder / "a.txt")], [], folder)
    with pytest.raises(ValueError):
        plan_for(folder, labels, unsure="bogus")


def test_name_clashes_never_overwrite(folder):
    (folder / "invoice").mkdir()
    (folder / "invoice" / "a.txt").write_text("existing")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("same name, other folder")
    records = [rec(folder / "a.txt"), rec(sub / "a.txt")]
    plan = build_plan(records, [Prediction("invoice")] * 2, folder)
    names = sorted(m.dst.name for m in plan.moves)
    assert names == ["a (2).txt", "a (3).txt"] and all(m.renamed for m in plan.moves)
    assert len({str(m.dst).lower() for m in plan.moves}) == 2


def test_case_insensitive_clash_between_planned_files(folder):
    (folder / "B.TXT").write_text("upper")
    plan = build_plan([rec(folder / "b.txt"), rec(folder / "B.TXT")], [Prediction("letter")] * 2, folder)
    assert len({str(m.dst).lower() for m in plan.moves}) == 2


def test_already_in_place_unreadable_low_confidence_and_blocked(folder):
    (folder / "invoice").mkdir()
    (folder / "invoice" / "x.txt").write_text("x")
    (folder / "letter").write_text("I am a file named like a category folder")
    records = [rec(folder / "invoice" / "x.txt"), rec(folder / "a.txt", status="empty"), rec(folder / "b.txt"),
               rec(folder / "c.txt"), rec(folder / "d.txt")]
    preds = [Prediction("invoice", 0.99), Prediction("invoice", 0.99), Prediction("invoice", 0.30),
             Prediction("letter", 0.99), Prediction("invoice", 0.9)]
    plan = build_plan(records, preds, folder, min_confidence=0.5)
    reasons = {s.path.name: s.reason for s in plan.skipped}
    assert reasons == {"x.txt": "already_in_place", "a.txt": "unreadable", "b.txt": "low_confidence",
                       "c.txt": "blocked"}
    assert [m.src.name for m in plan.moves] == ["d.txt"]
    # the other policies
    plan = build_plan(records, preds, folder, unsure="other", unreadable="classify")
    assert {m.src.name: m.category for m in plan.moves} == {"a.txt": "invoice", "b.txt": "other", "d.txt": "invoice"}


def test_missing_source_is_reported_not_fatal(folder):
    ghost = folder / "ghost.txt"
    plan = build_plan([rec(ghost), rec(folder / "a.txt")], [Prediction("letter")] * 2, folder)
    assert [s.reason for s in plan.skipped] == ["unavailable"] and len(plan.moves) == 1


def test_dest_root_and_custom_folder_names(folder, tmp_path):
    dest = tmp_path / "sorted"
    plan = plan_for(folder, {"a.txt": "cv_resume"}, dest_root=dest, folder_names={"cv_resume": "CVs: 2026?"})
    assert plan.moves[0].dst == dest.resolve() / "CVs_ 2026_" / "a.txt"  # ":" and "?" are not allowed in folder names


def test_unique_destination_helper(tmp_path):
    (tmp_path / "f.txt").write_text("1")
    (tmp_path / "f (2).txt").write_text("2")
    path, renamed = unique_destination(tmp_path / "f.txt")
    assert path.name == "f (3).txt" and renamed
    assert unique_destination(tmp_path / "new.txt") == (tmp_path / "new.txt", False)


class FixedProba(Classifier):
    def predict(self, record):
        return "a"

    def predict_proba(self, records):
        return [{"a": 0.8, "b": 0.2} for _ in records]


class NoProba(Classifier):
    def predict(self, record):
        return "z"


def test_predict_with_confidence(folder):
    records = [rec(folder / "a.txt")]
    assert predict_with_confidence(FixedProba(), records) == [Prediction("a", 0.8)]
    assert predict_with_confidence(NoProba(), records) == [Prediction("z", None)]
    assert predict_with_confidence(NoProba(), []) == []


def test_format_plan_preview(folder):
    plan = plan_for(folder, {"a.txt": Prediction("invoice", 0.91), "b.txt": Prediction("letter", 0.2),
                             "c.txt": "invoice", "d.txt": "other"})
    text = format_plan(plan, examples=1)
    assert "3 to move" in text and "invoice/  (2)" in text and "... and 1 more" in text
    assert "not sure enough" in text and "best guess: letter 0.20" in text
    text.encode("ascii")  # prints on any console


# -------------------------------------------------------- execute and undo
def run(folder, labels, db=None, **kw):
    plan = plan_for(folder, labels, **kw)
    log = MoveLog(db or folder / ".fileorg" / "moves.sqlite")
    return plan, log, execute_plan(plan, log, "test", chunk_size=2)


def test_execute_then_undo_restores_everything(folder):
    before, dirs_before = snapshot(folder), folders(folder)
    plan, log, result = run(folder, {"a.txt": "invoice", "b.txt": "letter", "c.txt": "invoice", "d.txt": "other"})
    assert (result.done, result.failed, result.skipped) == (4, 0, 0)
    assert sorted(snapshot(folder)) == ["invoice/a.txt", "invoice/c.txt", "letter/b.txt", "other/d.txt"]
    info = log.batch(result.batch_id)
    assert (info.planned, info.done, info.finished is not None, info.undone) == (4, 4, True, None)

    undone = undo_batch(log)
    assert (undone.restored, undone.failed, undone.removed_dirs) == (4, 0, 3)
    assert snapshot(folder) == before and folders(folder) == dirs_before  # created folders are gone again
    assert log.batch(result.batch_id).undone is not None
    assert log.latest_undoable() is None
    assert undo_batch(log).batch_id is None  # nothing left to undo


def test_changed_file_is_skipped_and_missing_file_fails(folder):
    plan = plan_for(folder, {"a.txt": "invoice", "b.txt": "letter", "c.txt": "invoice"})
    (folder / "a.txt").write_text("edited after the preview, different size")
    (folder / "b.txt").unlink()
    with MoveLog(folder / ".fileorg" / "moves.sqlite") as log:
        result = execute_plan(plan, log)
    assert (result.done, result.failed, result.skipped) == (1, 1, 1)
    reasons = {p.name: msg for p, msg in result.problems}
    assert "changed" in reasons["a.txt"] and "missing" in reasons["b.txt"]
    assert (folder / "a.txt").exists() and (folder / "invoice" / "c.txt").exists()


def test_destination_appearing_after_preview_gets_a_suffix(folder):
    plan = plan_for(folder, {"a.txt": "invoice"})
    (folder / "invoice").mkdir()
    (folder / "invoice" / "a.txt").write_text("someone else's file")
    with MoveLog(folder / ".fileorg" / "moves.sqlite") as log:
        execute_plan(plan, log)
        assert (folder / "invoice" / "a.txt").read_text() == "someone else's file"
        assert (folder / "invoice" / "a (2).txt").read_text() == "invoice a"
        assert undo_batch(log).restored == 1
    assert (folder / "a.txt").read_text() == "invoice a"
    assert (folder / "invoice" / "a.txt").exists()  # the other file stays; the folder is not removed


def test_undo_refuses_to_overwrite_and_reports_missing_or_modified(folder):
    plan, log, result = run(folder, {"a.txt": "invoice", "b.txt": "letter", "c.txt": "invoice", "d.txt": "other"})
    (folder / "a.txt").write_text("a new file took the old place")      # original location occupied
    (folder / "letter" / "b.txt").unlink()                              # deleted by the user
    (folder / "invoice" / "c.txt").write_text("edited and longer now")  # modified
    undone = undo_batch(log)
    assert (undone.restored, undone.failed) == (1, 3)
    messages = {p.name: m for p, m in undone.problems}
    assert "occupied" in messages["a.txt"] and "not found" in messages["b.txt"] and "changed" in messages["c.txt"]
    assert (folder / "a.txt").read_text() == "a new file took the old place"  # never overwritten
    assert log.batch(result.batch_id).undone is None                          # batch stays open for a retry
    # fix the problems; --force restores the edited file
    (folder / "a.txt").unlink()
    forced = undo_batch(log, force=True)
    assert forced.restored == 2 and forced.failed == 1  # b.txt is gone for good
    assert (folder / "c.txt").read_text() == "edited and longer now"


def test_interrupted_run_is_recoverable(folder):
    """A crash after the pending rows were written: some files moved, some did not."""
    plan = plan_for(folder, {"a.txt": "invoice", "b.txt": "letter"})
    log = MoveLog(folder / ".fileorg" / "moves.sqlite")
    batch = log.start_batch(plan.root, plan.dest_root, "test", 2)
    log.add_pending(batch, plan.moves)
    (folder / "invoice").mkdir()
    shutil.move(str(folder / "a.txt"), str(folder / "invoice" / "a.txt"))  # a moved, b did not; log still 'pending'
    undone = undo_batch(log, batch)
    assert (undone.restored, undone.skipped, undone.failed) == (1, 1, 0)
    assert (folder / "a.txt").exists() and (folder / "b.txt").exists()
    assert log.batch(batch).undone is not None


def test_undo_specific_batch_history_and_unknown_batch(folder):
    log = MoveLog(folder / ".fileorg" / "moves.sqlite")
    first = execute_plan(plan_for(folder, {"a.txt": "invoice"}), log, "rules")
    second = execute_plan(plan_for(folder, {"b.txt": "letter"}), log, "ml")
    assert [b.classifier for b in log.batches()] == ["rules", "ml"]
    assert log.latest_undoable() == second.batch_id
    assert undo_batch(log, first.batch_id).restored == 1
    assert (folder / "a.txt").exists() and (folder / "letter" / "b.txt").exists()
    with pytest.raises(KeyError):
        undo_batch(log, 99)
    assert execute_plan(plan_for(folder, {}), log).batch_id is None  # empty plan: no batch


def test_failed_move_is_logged_and_run_continues(folder, monkeypatch):
    real_move = shutil.move

    def flaky(src, dst, *a, **k):
        if Path(src).name == "b.txt":
            raise PermissionError("file is in use")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr("fileorg.organizer.executor.shutil.move", flaky)
    plan, log, result = run(folder, {"a.txt": "invoice", "b.txt": "letter", "c.txt": "invoice"})
    assert (result.done, result.failed) == (2, 1)
    assert "PermissionError" in result.problems[0][1]
    assert (folder / "b.txt").exists()  # the file that could not move is untouched
    monkeypatch.undo()
    assert undo_batch(log).restored == 2  # only what really moved is restored
    assert snapshot(folder).keys() == {"a.txt", "b.txt", "c.txt", "d.txt"}


def test_log_survives_reopening(folder):
    plan, log, result = run(folder, {"a.txt": "invoice", "b.txt": "letter"})
    log.close()
    with MoveLog(folder / ".fileorg" / "moves.sqlite") as reopened:
        assert reopened.latest_undoable() == result.batch_id
        assert undo_batch(reopened).restored == 2