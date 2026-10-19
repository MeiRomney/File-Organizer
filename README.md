# fileorg

Automated file classification and organization based on document content and metadata (Thesis I).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Usage
Put dataset files into `data/raw/<category>/` (see `docs/categories.md`), then:
```bash
fileorg scan data/raw --labelled -o data/records.jsonl
```
The command extracts text and metadata from every PDF, DOCX and TXT file and prints counts by extension, extraction status and label.

## Layout
```
src/fileorg/
  scanner.py        find supported files
  metadata.py       file-system metadata (size, dates, MIME type)
  extraction.py     text + embedded metadata from PDF/DOCX/TXT (never raises)
  pipeline.py       scan -> extract -> FileRecord, JSONL read/write
  models.py         FileRecord shared by all classifiers
  cli.py            `fileorg scan`
  classifiers/      base interface; rule_based.py, ml.py, embeddings.py come next
  organizer/        planned: move planner, preview/confirm, log, undo
  evaluation/       planned: metrics, confusion matrix, timing
tests/              pytest tests
docs/               categories.md, dataset_plan.md
data/raw/           the labelled dataset (not committed)
experiments/        notebooks and result tables
```

## Roadmap
1. Scanner and extraction (done)
2. Dataset collection and labelling
3. Rule-based baseline + evaluation
4. ML (TF-IDF) and embedding classifiers, same split and metrics
5. Preview/confirmation, logging, undo
6. Interface (desktop)
