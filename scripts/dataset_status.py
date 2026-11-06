"""Show files per category and format, and flag problems against dataset_plan.md.

    python scripts/dataset_status.py
"""
from __future__ import annotations

from collections import Counter

from common import CATEGORIES, RAW

MIN_FILES = 60


def main() -> None:
    counts: dict[str, Counter] = {}
    for cat in CATEGORIES:
        folder = RAW / cat
        files = [p for p in folder.glob("*") if p.is_file() and p.name != ".gitkeep"] if folder.exists() else []
        counts[cat] = Counter(p.suffix.lower() for p in files)

    print(f"{'category':<16}{'total':>6}{'pdf':>6}{'docx':>6}{'txt':>6}  notes")
    totals = {}
    for cat, c in counts.items():
        total = sum(c.values())
        totals[cat] = total
        notes = []
        if total < MIN_FILES:
            notes.append(f"below {MIN_FILES}")
        if total and sum(1 for ext in (".pdf", ".docx", ".txt") if c[ext]) < 2:
            notes.append("single format (extension would be a shortcut)")
        print(f"{cat:<16}{total:>6}{c['.pdf']:>6}{c['.docx']:>6}{c['.txt']:>6}  {', '.join(notes)}")

    nonzero = [t for t in totals.values() if t]
    if nonzero:
        print(f"\nTotal {sum(nonzero)} files; smallest class {min(nonzero)}, largest {max(nonzero)}.")
        if max(nonzero) != min(nonzero):
            print("Classes are not identical in size: report macro-averaged metrics or trim to the smallest.")


if __name__ == "__main__":
    main()