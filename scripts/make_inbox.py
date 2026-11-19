"""Flat, shuffled copy of the dataset to try `fileorg organize` on.
    python scripts/make_inbox.py [out_dir] [test|dev|all]
"""
import csv, random, shutil, sys
from pathlib import Path

RAW = Path("data/raw")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/benchmark/inbox")
subset = sys.argv[2] if len(sys.argv) > 2 else "test"

rows = [r for r in csv.DictReader(open("data/split.csv", encoding="utf-8"))
        if subset == "all" or r["subset"] == subset]
random.Random(42).shuffle(rows)

shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
with open(out.parent / "inbox_truth.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "label"])
    for i, r in enumerate(rows):
        name = f"{i:04d}_{Path(r['rel_path']).name}"
        shutil.copy2(RAW / r["rel_path"], out / name)
        w.writerow([name, r["label"]])
print(f"{len(rows)} files in {out}")