"""Command line interface: ``fileorg scan | split | evaluate | tune | compare | benchmark | train | organize | undo | history``."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from fileorg.classifiers import CLASSIFIER_NAMES, factory, get_classifier
from fileorg.evaluation.metrics import format_confusion
from fileorg.evaluation.runner import cross_validate, final_test
from fileorg.evaluation.significance import compare_classifiers, format_comparison
from fileorg.evaluation.split import (
    DEFAULT_FOLDS, DEFAULT_SEED, DEFAULT_TEST_SIZE, apply_split, load_groups, make_split, write_split,
)
from fileorg.evaluation.timing import DEFAULT_SIZES, format_benchmark, run_benchmark
from fileorg.evaluation.tuning import grid_search
from fileorg.extraction import DEFAULT_MAX_CHARS
# from fileorg.organizer import MoveLog, build_plan, execute_plan, format_plan, predict_with_confidence, undo_batch
from fileorg.pipeline import process_directory, read_jsonl, write_jsonl


# ------------------------------------------------------------------ scan
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


# ----------------------------------------------------------------- split
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
    write_split(split, args.out)

    print(f"Wrote {args.out}: {len(split.dev)} development ({split.n_folds} folds), {len(split.test)} test")
    print("Dev by label: ", dict(sorted(Counter(r.label for r in split.dev).items())))
    print("Test by label:", dict(sorted(Counter(r.label for r in split.test).items())))
    if groups:
        print(f"Grouped by template: {len(set(groups.values()))} groups")
    return 0


# ------------------------------------------------- classifier options
def _classifier_options(args: argparse.Namespace, name: str) -> dict:
    """Constructor options for classifier ``name``, taken from the command line."""
    options: dict = {}
    if name in ("rules", "ml", "embeddings"):
        options["features"] = [f.strip() for f in args.features.split(",") if f.strip()]
    if name == "rules":
        options["min_score"] = args.min_score
        options["margin"] = args.margin
    if name in ("ml", "embeddings"):
        options["model"] = args.model
        if args.C is not None:
            options["C"] = args.C
    if name == "ml":
        options["ngram_max"] = args.ngram_max
    if name == "embeddings":
        if args.embed_model:
            options["model_name"] = args.embed_model
        if args.embed_cache:
            options["cache_path"] = args.embed_cache
    return options


def _save_embedding_cache(args: argparse.Namespace) -> None:
    if getattr(args, "embed_cache", None):
        from fileorg.classifiers.embeddings import DEFAULT_MODEL, shared_encoder

        shared_encoder(args.embed_model or DEFAULT_MODEL, args.embed_cache).save()


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
    "embeddings": "model=logreg,svm;C=1,10,100",
}


# ------------------------------------------------------------------ tune
def _cmd_tune(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)
    grid = _parse_grid(args.grid or DEFAULT_GRIDS[args.classifier])
    fixed = {k: v for k, v in _classifier_options(args, args.classifier).items() if k not in grid}
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
    _save_embedding_cache(args)
    return 0


# -------------------------------------------------------------- evaluate
def _cmd_evaluate(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)
    options = _classifier_options(args, args.classifier)
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
    _save_embedding_cache(args)
    return 0


# --------------------------------------------------------------- compare
def _cmd_compare(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)
    groups_map = load_groups(args.groups) if args.groups else None

    predictions: dict[str, list[str]] = {}
    evaluated = None
    if args.final:
        print("=" * 70)
        print("FINAL TEST COMPARISON - run this once, after all tuning is finished.")
        print("=" * 70)
        evaluated = split.test
    for name in args.classifiers:
        make = factory(name, **_classifier_options(args, name))
        if args.final:
            predictions[name] = final_test(make, split).predictions
        else:
            cv = cross_validate(make, split)
            evaluated = cv.records
            predictions[name] = cv.predictions
    y_true = [r.label for r in evaluated]
    groups = [groups_map.get(r.rel_path, r.rel_path) for r in evaluated] if groups_map else None

    print("Compared on:", "the TEST set" if args.final else f"{split.n_folds}-fold CV predictions on the development set")
    comparison = compare_classifiers(y_true, predictions, groups=groups, n_boot=args.bootstrap, seed=args.seed)
    print(format_comparison(comparison))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "mode": "final" if args.final else "cv", "n": comparison.n, "n_groups": comparison.n_groups,
                "intervals": {n: {m: vars(i) for m, i in ms.items()} for n, ms in comparison.intervals.items()},
                "pairs": [{"a": p.a, "b": p.b, "mcnemar": vars(p.mcnemar), "accuracy": vars(p.accuracy),
                           "macro_f1": vars(p.macro_f1)} for p in comparison.pairs],
            }, fh, indent=2)
        print(f"\nWrote {args.json}")
    _save_embedding_cache(args)
    return 0


# ------------------------------------------------------------- benchmark
def _timing_options(args: argparse.Namespace, name: str) -> dict:
    """Like the evaluation options, but embeddings get a fresh encoder with no cache."""
    options = _classifier_options(args, name)
    if name == "embeddings":
        from fileorg.classifiers.embeddings import DEFAULT_MODEL, SentenceTransformerEncoder

        options.pop("cache_path", None)
        options["encoder"] = SentenceTransformerEncoder(options.get("model_name", DEFAULT_MODEL))
    return options


def _cmd_benchmark(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    split = apply_split(records, args.split)  # classifiers are trained on the development part only
    makers = {name: factory(name, **_timing_options(args, name)) for name in args.classifiers}
    result = run_benchmark(
        args.source, args.sizes, makers, split.dev, args.work_dir, repeats=args.repeats, seed=args.seed,
        keep=args.keep, log=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(format_benchmark(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0


# ------------------------------------------------- train / organize / undo
def _cmd_train(args: argparse.Namespace) -> int:
    if args.classifier == "rules":
        print("The rule-based classifier needs no training: use `fileorg organize --classifier rules`.", file=sys.stderr)
        return 1
    records = read_jsonl(args.records)
    if args.all_data:
        train = [r for r in records if r.label]
        print("Training on ALL labelled files, including the test set. Do not evaluate this model on the test set.")
    else:
        train = apply_split(records, args.split).dev
    model = factory(args.classifier, **_classifier_options(args, args.classifier))()
    model.fit(train)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Trained {args.classifier} on {len(train)} files; saved to {args.out}")
    _save_embedding_cache(args)
    return 0


def _load_classifier(args: argparse.Namespace):
    if args.classifier == "rules":
        return get_classifier("rules", **_classifier_options(args, "rules"))
    if not args.model_file:
        raise FileNotFoundError(f"--model-file is required for '{args.classifier}'; create one with "
                                f"`fileorg train data/records.jsonl --classifier {args.classifier} --out models/{args.classifier}.joblib`")
    if args.classifier == "ml":
        from fileorg.classifiers.ml import MLClassifier

        return MLClassifier.load(args.model_file)
    from fileorg.classifiers.embeddings import EmbeddingClassifier

    return EmbeddingClassifier.load(args.model_file)


def _log_path(args: argparse.Namespace, root: Path) -> Path:
    return Path(args.log) if args.log else root / ".fileorg" / "moves.sqlite"


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _print_problems(problems, limit: int = 10) -> None:
    for path, message in problems[:limit]:
        print(f"  {path.name}: {message}")
    if len(problems) > limit:
        print(f"  ... and {len(problems) - limit} more")


def _cmd_organize(args: argparse.Namespace) -> int:
    from fileorg.organizer import MoveLog, build_plan, execute_plan, format_plan, predict_with_confidence

    root = Path(args.folder).resolve()
    if not root.is_dir():
        print(f"Error: not a folder: {root}", file=sys.stderr)
        return 1

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\rReading {done}/{total} files", end="", file=sys.stderr, flush=True)

    records = process_directory(root, recursive=args.recursive, on_progress=progress)
    print(file=sys.stderr)
    if not records:
        print("No PDF, DOCX or TXT files found.")
        return 0
    classifier = _load_classifier(args)
    predictions = predict_with_confidence(classifier, records)
    plan = build_plan(records, predictions, root, args.dest, min_confidence=args.min_confidence,
                      unsure=args.unsure, unreadable=args.unreadable)
    print(format_plan(plan, examples=args.examples))

    if not args.apply:
        print("\nPreview only: nothing was moved. Add --apply to carry out these moves.")
        return 0
    if not plan.moves:
        print("\nNothing to move.")
        return 0
    if not args.yes and not _confirm(f"\nMove {len(plan.moves)} file(s) now?"):
        print("Aborted: nothing was moved.")
        return 1

    def move_progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\rMoving {done}/{total} files", end="", file=sys.stderr, flush=True)

    with MoveLog(_log_path(args, root)) as log:
        result = execute_plan(plan, log, classifier.name, progress=move_progress)
    print(file=sys.stderr)
    print(f"\nMoved {result.done} file(s) (batch {result.batch_id}); {result.failed} failed, {result.skipped} skipped.")
    _print_problems(result.problems)
    print(f"To undo: fileorg undo \"{root}\"")
    return 1 if result.failed else 0


def _cmd_undo(args: argparse.Namespace) -> int:
    from fileorg.organizer import MoveLog, undo_batch

    root = Path(args.folder).resolve()
    db = _log_path(args, root)
    if not db.exists():
        print(f"No move log found at {db}.", file=sys.stderr)
        return 1
    with MoveLog(db) as log:
        batch_id = args.batch or log.latest_undoable()
        if batch_id is None:
            print("Nothing to undo.")
            return 0
        info = log.batch(batch_id)
        if info is None:
            print(f"Error: no batch {batch_id} in {db}", file=sys.stderr)
            return 1
        pending = len(log.moves(batch_id, statuses=("done", "pending", "undo_failed")))
        print(f"Batch {batch_id} ({info.created}, {info.classifier}): {pending} file(s) will be moved back.")
        if not args.yes and not _confirm("Undo this batch now?"):
            print("Aborted: nothing was moved.")
            return 1
        result = undo_batch(log, batch_id, force=args.force)
    print(f"Restored {result.restored} file(s); {result.failed} could not be restored, {result.skipped} were never moved; "
          f"{result.removed_dirs} empty folder(s) removed.")
    _print_problems(result.problems)
    return 1 if result.failed else 0


def _cmd_history(args: argparse.Namespace) -> int:
    from fileorg.organizer import MoveLog

    root = Path(args.folder).resolve()
    db = _log_path(args, root)
    if not db.exists():
        print(f"No move log found at {db}.")
        return 0
    with MoveLog(db) as log:
        if args.batch:
            info = log.batch(args.batch)
            if info is None:
                print(f"Error: no batch {args.batch} in {db}", file=sys.stderr)
                return 1
            print(f"Batch {info.id}: {info.created}, {info.classifier}, {info.done} moved, {info.failed} failed")
            for m in log.moves(args.batch):
                conf = f" conf {m.confidence:.2f}" if m.confidence is not None else ""
                error = f"  [{m.error}]" if m.error else ""
                print(f"  {m.status:<11} {m.src.name} -> {m.category}/{conf}{error}")
            return 0
        print(f"{'batch':>5}  {'created (UTC)':<25}  {'classifier':<11}  {'moved':>5}  {'failed':>6}  {'skipped':>7}  status")
        for b in log.batches():
            status = "undone" if b.undone else ("active" if b.finished else "unfinished")
            print(f"{b.id:5d}  {b.created:<25}  {b.classifier:<11}  {b.done:5d}  {b.failed:6d}  {b.skipped:7d}  {status}")
    return 0


# ---------------------------------------------------------------- parser
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

    def add_common(p: argparse.ArgumentParser, with_data: bool = True) -> None:
        if with_data:
            p.add_argument("records", help="labelled records JSONL")
            p.add_argument("--split", default="data/split.csv", help="split file made by `fileorg split`")
        p.add_argument("--features", default="text,filename,metadata",
                       help="rules/ml/embeddings: comma-separated feature groups (text, filename, metadata)")
        p.add_argument("--min-score", type=float, default=4.0,
                       help="rules: best score below this means 'other' (lower it for metadata-only runs)")
        p.add_argument("--margin", type=float, default=0.0,
                       help="rules: required lead over the runner-up, otherwise 'other'")
        p.add_argument("--model", default="logreg", choices=["logreg", "svm"], help="ml/embeddings: linear model")
        p.add_argument("--C", type=float, default=None,
                       help="ml/embeddings: regularisation strength (default: 1 for ml, 10 for embeddings)")
        p.add_argument("--ngram-max", type=int, default=2, help="ml: longest word n-gram (1 = single words)")
        p.add_argument("--embed-model", default=None,
                       help="embeddings: sentence-transformers model name (default all-MiniLM-L6-v2)")
        p.add_argument("--embed-cache", default=None,
                       help="embeddings: file to keep computed embeddings in (e.g. data/embeddings.joblib; not committed)")

    tune = sub.add_parser("tune", help="grid search on the development folds (never touches the test set)")
    add_common(tune)
    tune.add_argument("--classifier", default="ml", choices=CLASSIFIER_NAMES)
    tune.add_argument("--grid", default=None,
                      help="e.g. 'C=0.1,1,10;model=logreg,svm' (a sensible default exists per classifier)")
    tune.add_argument("--json", default=None, help="also write the results as JSON")
    tune.set_defaults(func=_cmd_tune)

    ev = sub.add_parser("evaluate", help="cross-validate a classifier on the dev set (or run the final test)")
    add_common(ev)
    ev.add_argument("--classifier", default="rules", choices=CLASSIFIER_NAMES)
    ev.add_argument("--final", action="store_true", help="evaluate on the TEST set (do this once, at the very end)")
    ev.add_argument("--errors", type=int, default=15, help="how many misclassified files to list")
    ev.add_argument("--json", default=None, help="also write the report as JSON")
    ev.set_defaults(func=_cmd_evaluate)

    cmp_ = sub.add_parser("compare", help="compare classifiers with confidence intervals and significance tests")
    add_common(cmp_)
    cmp_.add_argument("--classifiers", nargs="+", default=["rules", "ml"], choices=CLASSIFIER_NAMES)
    cmp_.add_argument("--groups", default=None,
                      help="sources.csv; the bootstrap then resamples whole template groups (recommended)")
    cmp_.add_argument("--bootstrap", type=int, default=2000, help="number of bootstrap resamples")
    cmp_.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cmp_.add_argument("--final", action="store_true", help="compare on the TEST set (do this once, at the very end)")
    cmp_.add_argument("--json", default=None, help="also write the results as JSON")
    cmp_.set_defaults(func=_cmd_compare)

    bench = sub.add_parser("benchmark", help="time scan, extraction and classification on 100/1,000/10,000 files")
    add_common(bench)
    bench.add_argument("--source", default="data/raw", help="dataset folder to sample the files from")
    bench.add_argument("--classifiers", nargs="+", default=["rules", "ml"], choices=CLASSIFIER_NAMES)
    bench.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES), help="folder sizes to build")
    bench.add_argument("--repeats", type=int, default=3, help="repeat every measurement this many times")
    bench.add_argument("--seed", type=int, default=DEFAULT_SEED)
    bench.add_argument("--work-dir", default="data/benchmark", help="where the sampled folders are built (not committed)")
    bench.add_argument("--keep", action="store_true", help="keep the sampled folders afterwards")
    bench.add_argument("--json", default=None, help="also write all runs and medians as JSON")
    bench.set_defaults(func=_cmd_benchmark)

    train = sub.add_parser("train", help="train a model and save it for `organize`")
    add_common(train)
    train.add_argument("--classifier", default="ml", choices=CLASSIFIER_NAMES)
    train.add_argument("--out", default="models/model.joblib", help="where to save the trained model (not committed)")
    train.add_argument("--all-data", action="store_true",
                       help="train on every labelled file (final model for use; it can no longer be tested fairly)")
    train.set_defaults(func=_cmd_train)

    org = sub.add_parser("organize", help="sort the files of a folder into category folders (preview by default)")
    org.add_argument("folder", help="folder to organize")
    add_common(org, with_data=False)
    org.add_argument("--classifier", default="rules", choices=CLASSIFIER_NAMES)
    org.add_argument("--model-file", default=None, help="saved model from `fileorg train` (ml / embeddings)")
    org.add_argument("--dest", default=None, help="put the category folders here instead of inside the folder")
    org.add_argument("--recursive", action="store_true", help="also organize files in sub-folders")
    org.add_argument("--min-confidence", type=float, default=0.5,
                     help="models with probabilities: below this the file stays where it is")
    org.add_argument("--unsure", default="skip", choices=["skip", "other"],
                     help="low-confidence files: leave them (skip) or put them in 'other'")
    org.add_argument("--unreadable", default="skip", choices=["skip", "classify"],
                     help="files without readable text: leave them (skip) or classify by name/metadata")
    org.add_argument("--examples", type=int, default=5, help="example files shown per category in the preview")
    org.add_argument("--apply", action="store_true", help="really move the files (default: preview only)")
    org.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    org.add_argument("--log", default=None, help="move log database (default: <folder>/.fileorg/moves.sqlite)")
    org.set_defaults(func=_cmd_organize)

    und = sub.add_parser("undo", help="move the files of an organize run back")
    und.add_argument("folder", help="the folder that was organized")
    und.add_argument("--batch", type=int, default=None, help="batch number (default: the latest)")
    und.add_argument("--force", action="store_true", help="restore files even if they were changed since")
    und.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    und.add_argument("--log", default=None)
    und.set_defaults(func=_cmd_undo)

    hist = sub.add_parser("history", help="list organize runs, or the moves of one run")
    hist.add_argument("folder", help="the folder that was organized")
    hist.add_argument("--batch", type=int, default=None, help="show the moves of this batch")
    hist.add_argument("--log", default=None)
    hist.set_defaults(func=_cmd_history)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: file not found: {exc.filename or exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"Error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())