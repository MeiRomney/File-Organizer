"""Apply a plan to the disk, and undo it again.

Order of work for every chunk of files (default 25):

1. write the chunk to the log as ``pending`` and commit,
2. move the files,
3. update the log with the outcome and commit.

So whatever happens (crash, power loss, Ctrl+C), the log knows which files
*might* have moved, and ``undo_batch`` checks the disk to find out. Files are
never overwritten: a destination that appeared after the preview gets a
`` (2)`` suffix, and undo refuses to restore onto an occupied original path.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from fileorg.organizer.log import LoggedMove, MoveLog
from fileorg.organizer.planner import MovePlan, PlannedMove, unique_destination

MTIME_TOLERANCE_NS = 2_000_000_000  # FAT file systems store times with 2 s resolution


@dataclass
class ExecutionResult:
    batch_id: Optional[int]
    done: int = 0
    failed: int = 0
    skipped: int = 0
    problems: list[tuple[Path, str]] = field(default_factory=list)


@dataclass
class UndoResult:
    batch_id: Optional[int]
    restored: int = 0
    failed: int = 0
    skipped: int = 0
    removed_dirs: int = 0
    problems: list[tuple[Path, str]] = field(default_factory=list)


def _make_dirs(directory: Path) -> list[Path]:
    """Create ``directory`` and any missing parents; return the folders that were new."""
    missing: list[Path] = []
    current = directory
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        folder.mkdir(exist_ok=True)
    return missing


def _move_one(move: PlannedMove) -> tuple[str, Optional[str], Path, list[Path]]:
    """(status, error, final destination, folders created)."""
    try:
        stat = move.src.stat()
    except OSError:
        return "failed", "source file is missing", move.dst, []
    if stat.st_size != move.size_bytes or stat.st_mtime_ns != move.mtime_ns:
        return "skipped", "file changed since the preview", move.dst, []
    dst, _ = unique_destination(move.dst)  # something may have appeared since the preview
    created: list[Path] = []
    try:
        created = _make_dirs(dst.parent)
        shutil.move(str(move.src), str(dst))
    except OSError as exc:  # permission denied, file in use, disk full ...
        return "failed", f"{type(exc).__name__}: {exc}", dst, created
    if not os.path.lexists(dst):
        return "failed", "destination missing after the move", dst, created
    return "done", None, dst, created


def execute_plan(
    plan: MovePlan,
    log: MoveLog,
    classifier_name: str = "",
    chunk_size: int = 25,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ExecutionResult:
    if not plan.moves:
        return ExecutionResult(batch_id=None)
    batch_id = log.start_batch(plan.root, plan.dest_root, classifier_name, len(plan.moves))
    result = ExecutionResult(batch_id)
    finished = 0
    try:
        for start in range(0, len(plan.moves), chunk_size):
            chunk = plan.moves[start:start + chunk_size]
            ids = log.add_pending(batch_id, chunk)  # durable before any file moves
            updates, new_dirs = [], []
            for move_id, move in zip(ids, chunk):
                status, error, dst, created = _move_one(move)
                updates.append((move_id, status, error, dst))
                new_dirs += created
                if status == "done":
                    result.done += 1
                elif status == "failed":
                    result.failed += 1
                    result.problems.append((move.src, error or ""))
                else:
                    result.skipped += 1
                    result.problems.append((move.src, error or ""))
                finished += 1
                if progress:
                    progress(finished, len(plan.moves))
            log.update_moves(updates)
            log.add_dirs(batch_id, new_dirs)
    finally:  # also on Ctrl+C: the log must show how far we got
        log.finish_batch(batch_id, result.done, result.failed, result.skipped)
    return result


def _modified(path: Path, move: LoggedMove) -> bool:
    stat = path.stat()
    return stat.st_size != move.size_bytes or abs(stat.st_mtime_ns - move.mtime_ns) > MTIME_TOLERANCE_NS


def undo_batch(log: MoveLog, batch_id: Optional[int] = None, force: bool = False) -> UndoResult:
    """Move the files of one batch back (the latest undoable batch by default)."""
    if batch_id is None:
        batch_id = log.latest_undoable()
        if batch_id is None:
            return UndoResult(batch_id=None)
    elif log.batch(batch_id) is None:
        raise KeyError(f"no batch {batch_id} in {log.path}")
    result = UndoResult(batch_id)

    for move in reversed(log.moves(batch_id, statuses=("done", "pending", "undo_failed"))):
        src_there, dst_there = os.path.lexists(move.src), os.path.lexists(move.dst)
        if move.status == "pending" and src_there and not dst_there:
            log.set_move_status(move.id, "skipped", "was never moved (run was interrupted)")
            result.skipped += 1
            continue
        problem = None
        if not dst_there:
            problem = "file not found at the destination (moved or deleted since?)"
        elif src_there:
            problem = "the original location is occupied by another file"
        elif not force and _modified(move.dst, move):
            problem = "file at the destination was changed or replaced (use --force to restore it anyway)"
        if problem:
            log.set_move_status(move.id, "undo_failed", problem)
            result.failed += 1
            result.problems.append((move.dst, problem))
            continue
        try:
            move.src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.dst), str(move.src))
        except OSError as exc:
            problem = f"{type(exc).__name__}: {exc}"
            log.set_move_status(move.id, "undo_failed", problem)
            result.failed += 1
            result.problems.append((move.dst, problem))
            continue
        log.set_move_status(move.id, "undone")
        result.restored += 1

    # remove the folders this batch created, but only when they are empty again
    for folder in sorted(log.created_dirs(batch_id), key=lambda p: len(p.parts), reverse=True):
        try:
            folder.rmdir()
            result.removed_dirs += 1
        except OSError:
            pass
    if not log.moves(batch_id, statuses=("done", "pending", "undo_failed")):
        log.mark_batch_undone(batch_id)
    return result