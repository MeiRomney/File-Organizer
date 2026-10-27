"""Command line interface: ``fileorg scan | split | evaluate | tune``."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Optional, Sequence

from fileorg.classifiers import CLASSIFIER_NAMES, factory
from fileorg.evaluation.metrics import format_confusion
from fileorg.evaluation.runner import cross_validate, final_test
from fileorg.evaluation.tuning import grid_search
from fileorg.evaluation.split import (
    DEFAULT_FOLDS, DEFAULT_SEED, DEFAULT_TEST_SIZE, apply_split, load_groups, make_split, write_split,
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
    groups = load_groups(args.groups) if args.groups else None
    split = make_split(records, test_size=args.test_size, folds=args.folds, seed=args.seed, groups=groups)
    write_split(split, args.out)

    print(f"Wrote {args.out}: {len(split.dev)} development ({split.n_folds} folds), {len(split.test)} test")
    print("Dev by label: ", dict(sorted(Counter(r.label for r in split.dev).items())))
    print("Test by label:", dict(sorted(Counter(r.label for r in split.test).items())))
    if groups:
        print(f"Grouped by template: {len(set(groups.values()))} groups")
    return 0


def _classifier_options(args: argparse.Namespace) -> dict:
    """Constructor options for the chosen classifier, taken from the command line."""
    options: dict = {}
    if args.classifier in ("rules", "ml"):
        options["features"] = [f.strip() for f in args.features.split(",") if f.strip()]
    if args.classifier == "rules":
        options["min_score"] = args.min_score
        options["margin"] = args.margin
    if args.classifier == "ml":
        options.update(model=args.model, C=args.C, ngram_max=args.ngram_max)
    return options


def _parse_grid(text: str) -> dict:
    """'C=0.1,1,10;model=logreg,svm' -> {'C': [0.1, 1, 10], 'model': ['logreg', 'svm']}"""
    def convert(value: str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value

    grid = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, _, values = part.partition("=")
        grid[key.strip()] = [convert(v.strip()) for v in values.split(",") if v.strip()]
    return grid


DEFAULT_GRIDS = {
    "rules": "min_score=2,4,6;margin=0,1,2",
    "ml": "model=logreg,svm;C=0.1,1,10;ngram_max=1,2",
}


def _cmd_tune(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)
    fixed = _classifier_options(args)
    grid = _parse_grid(args.grid or DEFAULT_GRIDS[args.classifier])
    fixed = {k: v for k, v in fixed.items() if k not in grid}  # the grid wins over the fixed options
    results = grid_search(lambda **p: factory(args.classifier, **fixed, **p)(), grid, split)

    print(f"{args.classifier}: {len(results)} settings, {split.n_folds}-fold CV on {len(split.dev)} development files "
          f"(test set untouched)")
    print(f"{'macro-F1':>9}  {'accuracy':>9}  {'std':>6}  {'fit s':>6}  parameters")
    for r in results:
        print(f"{r.mean_macro_f1:9.3f}  {r.mean_accuracy:9.3f}  {r.std_accuracy:6.3f}  {r.fit_seconds:6.2f}  {r.params}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([vars(r) for r in results], fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)
    options = _classifier_options(args)
    make = factory(args.classifier, **options)
    description = f"{args.classifier} {options}" if options else args.classifier

    if args.final:
        print("=" * 70)
        print("FINAL TEST EVALUATION - run this once, after all tuning is finished.")
        print("=" * 70)
        result = final_test(make, split)
        report = result.report
        print(f"{description}: trained on {len(split.dev)} dev files, tested on {len(split.test)} test files")
        print(f"fit {result.fit_seconds:.2f}s, predict {result.predict_seconds:.2f}s")
        errors = [(r, p) for r, p in zip(split.test, result.predictions) if r.label != p]
    else:
        cv = cross_validate(make, split)
        report = cv.pooled
        print(f"{description}: {split.n_folds}-fold cross-validation on {len(split.dev)} development files")
        print(f"accuracy per fold: mean {cv.mean_accuracy:.3f}, std {cv.std_accuracy:.3f}; macro-F1 mean {cv.mean_macro_f1:.3f}")
        errors = cv.errors()

    print()
    print(report.format())
    print()
    print("Confusion matrix (rows = true, columns = predicted):")
    print(format_confusion(report))
    if args.errors and errors:
        print(f"\nFirst {min(args.errors, len(errors))} of {len(errors)} errors (true -> predicted):")
        for rec, pred in errors[: args.errors]:
            print(f"  {rec.rel_path}: {rec.label} -> {pred}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"classifier": description, "mode": "final" if args.final else "cv", **report.to_dict()}, fh, indent=2)
        print(f"\nWrote {args.json}")
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

    def add_classifier_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("records", help="labelled records JSONL")
        p.add_argument("--split", default="data/split.csv", help="split file made by `fileorg split`")
        p.add_argument("--classifier", default="rules", choices=CLASSIFIER_NAMES)
        p.add_argument("--features", default="text,filename,metadata",
                       help="rules/ml: comma-separated feature groups (text, filename, metadata)")
        p.add_argument("--min-score", type=float, default=4.0,
                       help="rules: best score below this means 'other' (lower it for metadata-only runs)")
        p.add_argument("--margin", type=float, default=0.0,
                       help="rules: required lead over the runner-up, otherwise 'other'")
        p.add_argument("--model", default="logreg", choices=["logreg", "svm"], help="ml: linear model")
        p.add_argument("--C", type=float, default=1.0, help="ml: regularisation strength (higher = weaker)")
        p.add_argument("--ngram-max", type=int, default=2, help="ml: longest word n-gram (1 = single words)")

    tune = sub.add_parser("tune", help="grid search on the development folds (never touches the test set)")
    add_classifier_args(tune)
    tune.add_argument("--grid", default=None,
                      help="e.g. 'C=0.1,1,10;model=logreg,svm' (a sensible default exists per classifier)")
    tune.add_argument("--json", default=None, help="also write the results as JSON")
    tune.set_defaults(func=_cmd_tune)

    ev = sub.add_parser("evaluate", help="cross-validate a classifier on the dev set (or run the final test)")
    add_classifier_args(ev)
    ev.add_argument("--final", action="store_true", help="evaluate on the TEST set (do this once, at the very end)")
    ev.add_argument("--errors", type=int, default=15, help="how many misclassified files to list")
    ev.add_argument("--json", default=None, help="also write the report as JSON")
    ev.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())