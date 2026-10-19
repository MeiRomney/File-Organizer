"""Command line interface: ``fileorg scan <folder> -o records.jsonl``."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Optional, Sequence

from fileorg.extraction import DEFAULT_MAX_CHARS
from fileorg.pipeline import process_directory, write_jsonl


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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
