import csv, sys
from collections import Counter
from pathlib import Path

inbox = Path(sys.argv[1] if len(sys.argv) > 1 else "data/benchmark/inbox")
truth = {r["file"]: r["label"] for r in csv.DictReader(open(inbox.parent / "inbox_truth.csv", encoding="utf-8"))}

ok = wrong = left = 0
errors = Counter()
for p in inbox.rglob("*"):
    if not p.is_file() or ".fileorg" in p.parts:
        continue
    rel = p.relative_to(inbox)
    if len(rel.parts) == 1:
        left += 1                      # skipped: low confidence / unreadable
    elif rel.parts[0] == truth[p.name]:
        ok += 1
    else:
        wrong += 1
        errors[(truth[p.name], rel.parts[0])] += 1

moved = ok + wrong
print(f"moved {moved}, correct {ok} ({ok / max(moved, 1):.1%}), left in place {left}")
for (t, pred), n in errors.most_common(10):
    print(f"  {t} -> {pred}: {n}")