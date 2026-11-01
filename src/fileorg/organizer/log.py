"""SQLite move log: the record that makes undo and crash recovery possible.

Three tables: ``batches`` (one row per organise run), ``moves`` (one row per
file, with its status) and ``created_dirs`` (folders a run had to create, so
undo can remove them again when they are empty).

Move statuses:  pending -> done | failed | skipped   (organise)
                done -> undone | undo_failed         (undo)
A ``pending`` row is written *before* the file is moved. If the program dies
in between, the row stays ``pending`` and undo looks at the disk to find out
what really happened.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from fileorg.organizer.planner import PlannedMove

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL, root TEXT NOT NULL, dest_root TEXT NOT NULL, classifier TEXT NOT NULL,
    planned INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0, finished TEXT, undone TEXT);
CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    src TEXT NOT NULL, dst TEXT NOT NULL, category TEXT NOT NULL, confidence REAL,
    size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
    status TEXT NOT NULL, error TEXT, moved TEXT, undone TEXT);
CREATE TABLE IF NOT EXISTS created_dirs (batch_id INTEGER NOT NULL REFERENCES batches(id), path TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_moves_batch ON moves(batch_id);
"""
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class BatchInfo:
    id: int
    created: str
    root: str
    dest_root: str
    classifier: str
    planned: int
    done: int
    failed: int
    skipped: int
    finished: Optional[str]
    undone: Optional[str]


@dataclass(frozen=True)
class LoggedMove:
    id: int
    batch_id: int
    src: Path
    dst: Path
    category: str
    confidence: Optional[float]
    size_bytes: int
    mtime_ns: int
    status: str
    error: Optional[str]


class MoveLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} was written by a newer version of fileorg")
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def __enter__(self) -> "MoveLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- writing
    def start_batch(self, root: Path, dest_root: Path, classifier: str, planned: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO batches (created, root, dest_root, classifier, planned) VALUES (?, ?, ?, ?, ?)",
            (_now(), str(root), str(dest_root), classifier, planned))
        self.conn.commit()
        return int(cur.lastrowid)

    def add_pending(self, batch_id: int, moves: Sequence[PlannedMove]) -> list[int]:
        """Record the intention to move these files (committed before any file is touched)."""
        ids = []
        for m in moves:
            cur = self.conn.execute(
                "INSERT INTO moves (batch_id, src, dst, category, confidence, size_bytes, mtime_ns, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                (batch_id, str(m.src), str(m.dst), m.category, m.confidence, m.size_bytes, m.mtime_ns))
            ids.append(int(cur.lastrowid))
        self.conn.commit()
        return ids

    def update_moves(self, updates: Iterable[tuple[int, str, Optional[str], Path]]) -> None:
        """``(move id, status, error, final destination)`` for finished moves."""
        now = _now()
        self.conn.executemany(
            "UPDATE moves SET status = ?, error = ?, dst = ?, moved = ? WHERE id = ?",
            [(status, error, str(dst), now if status == "done" else None, mid) for mid, status, error, dst in updates])
        self.conn.commit()

    def add_dirs(self, batch_id: int, paths: Iterable[Path]) -> None:
        self.conn.executemany("INSERT INTO created_dirs (batch_id, path) VALUES (?, ?)",
                              [(batch_id, str(p)) for p in paths])
        self.conn.commit()

    def finish_batch(self, batch_id: int, done: int, failed: int, skipped: int) -> None:
        self.conn.execute("UPDATE batches SET done = ?, failed = ?, skipped = ?, finished = ? WHERE id = ?",
                          (done, failed, skipped, _now(), batch_id))
        self.conn.commit()

    def set_move_status(self, move_id: int, status: str, error: Optional[str] = None) -> None:
        self.conn.execute("UPDATE moves SET status = ?, error = ?, undone = ? WHERE id = ?",
                          (status, error, _now() if status == "undone" else None, move_id))
        self.conn.commit()

    def mark_batch_undone(self, batch_id: int) -> None:
        self.conn.execute("UPDATE batches SET undone = ? WHERE id = ?", (_now(), batch_id))
        self.conn.commit()

    # ------------------------------------------------------------- reading
    @staticmethod
    def _batch(row: sqlite3.Row) -> BatchInfo:
        return BatchInfo(**{k: row[k] for k in BatchInfo.__dataclass_fields__})

    def batches(self) -> list[BatchInfo]:
        return [self._batch(r) for r in self.conn.execute("SELECT * FROM batches ORDER BY id")]

    def batch(self, batch_id: int) -> Optional[BatchInfo]:
        row = self.conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        return self._batch(row) if row else None

    def moves(self, batch_id: int, statuses: Optional[Sequence[str]] = None) -> list[LoggedMove]:
        rows = self.conn.execute("SELECT * FROM moves WHERE batch_id = ? ORDER BY id", (batch_id,)).fetchall()
        return [LoggedMove(r["id"], r["batch_id"], Path(r["src"]), Path(r["dst"]), r["category"], r["confidence"],
                           r["size_bytes"], r["mtime_ns"], r["status"], r["error"])
                for r in rows if statuses is None or r["status"] in statuses]

    def created_dirs(self, batch_id: int) -> list[Path]:
        return [Path(r["path"]) for r in
                self.conn.execute("SELECT path FROM created_dirs WHERE batch_id = ?", (batch_id,))]

    def latest_undoable(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT b.id FROM batches b WHERE b.undone IS NULL AND EXISTS ("
            "SELECT 1 FROM moves m WHERE m.batch_id = b.id AND m.status IN ('done', 'pending', 'undo_failed')) "
            "ORDER BY b.id DESC LIMIT 1").fetchone()
        return int(row[0]) if row else None