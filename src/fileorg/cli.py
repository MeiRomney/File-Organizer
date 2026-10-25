"""Command line interface: ``fileorg scan``, ``fileorg split``."""
from __future__ import annotations
from pathlib import Path

import argparse
import sys
from collections import Counter
from typing import Optional, Sequence

from fileorg.evaluation.split import (
    DEFAULT_FOLDS, DEFAULT_SEED, DEFAULT_TEST_SIZE, load_groups, make_split, write_split,
)
from fileorg.extraction import DEFAULT_MAX_CHARS
from fileorg.pipeline import process_directory, read_jsonl, write_jsonl


def _cmd_scan(args: argparse.Namespace) -> int:
    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\r{done}/{total} files", end="", file=sys.stderr, flush=True)

    records = process_directory(
        args.root,
        labelled=args.labelled,
        recursive=not args.no_recursive,
        max_chars=args.max_chars,
        max_pages=args.max_pages,
        on_progress=progress,
    )
    print(file=sys.stderr)
    write_jsonl(records, args.out)

    print(f"Wrote {len(records)} records to {args.out}")
    print("By extension:", dict(Counter(r.extension for r in records)))
    print("By status:   ", dict(Counter(r.extraction_status for r in records)))
    if args.labelled:
        print("By label:    ", dict(sorted(Counter(r.label for r in records).items(), key=lambda kv: str(kv[0]))))
    total = sum(r.extraction_seconds for r in records)
    print(f"Extraction time: {total:.2f}s total")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    if not records:
        print("No records found. Put files in data/raw/<category>/ and run `fileorg scan` first.", file=sys.stderr)
        return 1
    groups = None
    if args.groups:
        if not Path(args.groups).is_file():
            print(f"Groups file not found: {args.groups} (leave out --groups to split by file)", file=sys.stderr)
            return 1
        groups = load_groups(args.groups)
    split = make_split(records, test_size=args.test_size, folds=args.folds, seed=args.seed, groups=groups)

    print(f"Wrote {args.out}: {len(split.dev)} development ({split.n_folds} folds), {len(split.test)} test")
    print("Dev by label: ", dict(sorted(Counter(r.label for r in split.dev).items())))
    print("Test by label:", dict(sorted(Counter(r.label for r in split.test).items())))
    if groups:
        print(f"Grouped by template: {len(set(groups.values()))} groups")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fileorg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a folder and extract text + metadata")
    scan.add_argument("root", help="folder to scan")
    scan.add_argument("-o", "--out", default="records.jsonl", help="output JSONL file")
    scan.add_argument("--labelled", action="store_true",
                      help="use the first-level sub-folder name as the ground-truth label")
    scan.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    scan.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="max characters of text per file")
    scan.add_argument("--max-pages", type=int, default=None, help="max PDF pages to read per file")
    scan.set_defaults(func=_cmd_scan)

    split = sub.add_parser("split", help="create the stratified dev/test split and CV folds")
    split.add_argument("records", help="labelled records JSONL from `fileorg scan --labelled`")
    split.add_argument("-o", "--out", default="data/split.csv", help="split file to write (commit this one)")
    split.add_argument("--seed", type=int, default=DEFAULT_SEED)
    split.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    split.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    split.add_argument("--groups", default=None,
                       help="sources.csv; keeps files with the same template_id together")
    split.set_defaults(func=_cmd_split)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())