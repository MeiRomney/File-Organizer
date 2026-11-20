# fileorg

Automated file classification and organization based on document content and metadata (Thesis I).

fileorg reads PDF, DOCX and TXT files, classifies each one into one of 8 categories (invoice, receipt, contract, academic paper, CV / resume, letter, report, other), and can move the files into category folders. It always shows a preview first, logs every move, and can undo a run.

Three classification approaches are compared on the same split and metrics:

| Name         | Approach                                                                   |
| ------------ | -------------------------------------------------------------------------- |
| `rules`      | weighted keyword / regex scoring (no training)                             |
| `ml`         | TF-IDF features + logistic regression or linear SVM                        |
| `embeddings` | sentence embeddings (all-MiniLM-L6-v2) + logistic regression or linear SVM |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + pytest, reportlab, faker, psutil
pip install -e ".[embeddings]"     # optional: sentence-transformers (model is ~90 MB, downloaded on first use)
pytest
```

Python 3.10 or newer. The desktop interface uses Tkinter, which ships with Python.

## Quick start

```bash
# sort a folder (preview only; nothing moves)
fileorg organize path/to/folder

# really move the files, then undo if needed
fileorg organize path/to/folder --apply
fileorg undo path/to/folder

# desktop interface
fileorg gui [folder]
```

## Commands

| Command                    | Purpose                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `fileorg scan`             | extract text + metadata from a folder into JSONL (`--labelled` uses sub-folder names as labels)             |
| `fileorg split`            | stratified 80/20 dev/test split with 5 CV folds; `--groups data/sources.csv` keeps template groups together |
| `fileorg tune`             | grid search on the development folds (never touches the test set)                                           |
| `fileorg evaluate`         | cross-validate on the dev set, or `--final` for the one-time test evaluation                                |
| `fileorg compare`          | bootstrap confidence intervals, paired bootstrap, McNemar, Bonferroni-adjusted alpha                        |
| `fileorg benchmark`        | time scan / extraction / classification on 100, 1,000 and 10,000 files                                      |
| `fileorg train`            | train `ml` or `embeddings` and save a model for `organize`                                                  |
| `fileorg organize`         | plan, preview and (with `--apply`) execute moves                                                            |
| `fileorg undo` / `history` | restore a batch / list runs or the moves of one run                                                         |
| `fileorg gui`              | desktop window (review table, edit categories, history and undo)                                            |

## Typical experiment workflow

```bash
fileorg scan data/raw --labelled -o data/records.jsonl
fileorg split data/records.jsonl --groups data/sources.csv -o data/split.csv   # commit split.csv

fileorg tune data/records.jsonl --classifier ml
fileorg evaluate data/records.jsonl --classifier rules --json experiments/rules_cv.json
fileorg compare data/records.jsonl --classifiers rules ml embeddings --groups data/sources.csv

# once, after all tuning is finished
fileorg compare data/records.jsonl --classifiers rules ml embeddings --groups data/sources.csv --final

fileorg benchmark data/records.jsonl --classifiers rules ml embeddings --json experiments/timing.json
```

Use `--embed-cache data/embeddings.joblib` so embeddings are computed once across folds and tuning runs. The benchmark deliberately uses an uncached encoder.

Train a model for real use, then organize with it:

```bash
fileorg train data/records.jsonl --classifier ml --all-data --out models/ml.joblib
fileorg organize path/to/folder --classifier ml --model-file models/ml.joblib
```

Models are pickle files: only load files you created yourself.

## Safety rules of the organizer

- Preview is the default. `--apply` asks for confirmation (`--yes` skips it).
- Files are never overwritten: clashes get ` (2)`, ` (3)` ... (also on case-insensitive file systems).
- Unreadable files (scanned PDF, encrypted, corrupt) and low-confidence predictions stay where they are by default (`--unreadable`, `--unsure`, `--min-confidence`).
- Files changed after the preview are skipped.
- Every move is written to a SQLite log (`<folder>/.fileorg/moves.sqlite`) before it happens, so undo also works after a crash. Undo refuses to overwrite and reports edited or missing files; `--force` restores edited ones.

## Dataset

8 balanced categories, English only, PDF/DOCX/TXT, stored in `data/raw/<category>/` (not committed). Definitions and tie-break rules: `docs/categories.md`. Collection plan, sources and split rules: `docs/dataset_plan.md`.

| Category                            | Source                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| academic_paper                      | arXiv (CC-licensed only, via OAI-PMH)                                                                               |
| contract                            | SEC EDGAR exhibits                                                                                                  |
| invoice, receipt, cv_resume, letter | synthetic (Faker), 5 templates x 3 house styles per category                                                        |
| report                              | GAO and World Bank reports                                                                                          |
| other                               | Wikibooks recipes, IRS forms, Open Government Licence - Canada documents (manuals, forms, minutes, fact sheets ...) |

Provenance and licences: `data/sources.csv`. Rejected files: `data/excluded.csv`. The stored split: `data/split.csv`.

Dataset scripts (run from the repo root; `--contact` is your e-mail, sent in the User-Agent):

```
tools/generate_synthetic.py   synthetic invoice/receipt/cv/letter with template groups (template_id) for group-aware splits
scripts/fetch_arxiv.py        open-licensed arXiv papers
scripts/fetch_edgar_contracts.py   contracts from SEC EDGAR
scripts/fetch_worldbank.py    World Bank reports (check each document's terms before sharing)
scripts/fetch_wikibooks.py    Wikibooks recipes for "other"
scripts/fetch_ckan.py         varied documents from a CKAN portal (default open.canada.ca) for "other"
scripts/fetch_manifest.py     hand-picked URLs from scripts/manifest.csv (GAO reports, IRS forms)
scripts/make_synthetic.py     older simple synthetic generator (superseded by tools/generate_synthetic.py)
scripts/dataset_status.py     files per category and format, flags problems
scripts/make_inbox.py         flat shuffled copy of the test set to try `organize` on
scripts/score_inbox.py        score an organized inbox against the ground truth
```

Do not commit dataset files, models or embeddings (see `.gitignore`).

## Layout

```
src/fileorg/
  scanner.py        find supported files
  metadata.py       file-system metadata (size, dates, MIME type)
  extraction.py     text + embedded metadata from PDF/DOCX/TXT (never raises)
  pipeline.py       scan -> extract -> FileRecord, JSONL read/write
  models.py         FileRecord shared by all classifiers
  classifiers/      base.py, rule_based.py, ml.py, embeddings.py, registry in __init__.py
  evaluation/       split, metrics, runner (CV / final test), tuning, significance, timing
  organizer/        planner (preview), executor (move + undo), log (SQLite)
  desktop/          session.py (logic, no Tk) and app.py (Tkinter window)
  cli.py            all `fileorg` commands
scripts/ tools/     dataset collection and helpers
tests/              pytest tests for every module
docs/               categories.md, dataset_plan.md
data/               split.csv, sources.csv, excluded.csv (raw/ not committed)
experiments/        result files (JSON)
```

## Results so far

Result files live in `experiments/`.

- `timing.json`: processing time on Windows 11, 8 logical cores, 15 GB RAM, no GPU. For 10,000 files the median total (scan + extraction + other + classification) was about 192 s for `rules`, 171 s for `ml` and 806 s for `embeddings`. Extraction dominates for `rules` and `ml`; embedding inference dominates for `embeddings`. One-time training took 0.2 s for `ml` and 73 s for `embeddings`.
- `rules_cv.json`: an early rule-based CV run on 4 classes (319 files). Re-run it on the full 8-class dataset before quoting numbers.

## Roadmap

1. Scanner and extraction (done)
2. Dataset collection and labelling (collected; second-labeller 10% check still open)
3. Rule-based baseline + evaluation (done)
4. ML (TF-IDF) and embedding classifiers, same split and metrics (done)
5. Preview/confirmation, logging, undo (done)
6. Desktop interface (done: `fileorg gui`)
7. Final test evaluation and significance comparison, write-up (open)

Out of scope in version 1: OCR for scanned documents, languages other than English.

## License

MIT. Dataset files keep their own licences (see `data/sources.csv`).
