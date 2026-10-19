# Dataset plan

## Targets
| Item | Target |
|------|--------|
| Categories | 8 (see `categories.md`) |
| Files per category | **minimum 60, ideal 100**, identical (or nearly identical) for every class |
| Total | 480 to 800 files |
| Formats | PDF, DOCX, TXT |
| Language | English only in version 1 (multilingual support = future work) |

**Balance:** the same number of files per category, including `other`. Report the final counts in the thesis. If a class ends up smaller, say so and also report macro-averaged metrics.

## Avoiding shortcuts the classifier should not learn
- **Format mix per class:** every class has at least two formats, so the file extension is not a giveaway. Convert a share of documents between formats (for example DOCX -> PDF with LibreOffice headless).
- **Neutral filenames in the test set:** the filename-only features are evaluated as a separate experiment. Keep real-world filenames, but record which files were renamed or anonymised.
- **No duplicates:** remove exact duplicates (hash) and near-duplicates (same template, same text).
- **Template diversity:** if you generate synthetic documents, use at least 5 different templates per category and split by template, never by file, otherwise the test set leaks from the training set.

## Sources (record every file in `data/sources.csv`: file, source URL or origin, licence, date)
| Category | Suggested sources |
|----------|-------------------|
| academic_paper | open-access papers (e.g. arXiv, open-access journals) |
| contract | public filings and published template agreements (e.g. SEC EDGAR exhibits), open template libraries |
| invoice, receipt | synthetic documents generated from several templates (Python + Faker + python-docx/reportlab), plus public sample datasets where the licence allows |
| cv_resume | synthetic CVs, public resume datasets, volunteers' CVs **only if fully anonymised and with consent** |
| letter | public sample letters, generated letters, anonymised own letters |
| report | government, NGO and company reports with open licences |
| other | manuals, forms, meeting notes, flyers, recipes: deliberately varied |

**Privacy:** never include personal or sensitive documents. Replace real names, addresses and IDs. Check licences before using or publishing any file, and do not commit dataset files to git.

## Folder layout
```
data/raw/
  invoice/        receipt/        contract/       academic_paper/
  cv_resume/      letter/         report/         other/
data/sources.csv       # where every file came from + licence
data/excluded.csv      # ambiguous files and why they were left out
```
The folder name is the ground-truth label: `fileorg scan data/raw --labelled -o data/records.jsonl`.

## Train / test split
- **Stratified** split so each class keeps its proportion: 80% development (train + cross-validation) / 20% **test**.
- Fix the random seed and store the list of test files in the repository.
- **The test set is untouched until the final evaluation.** All rule writing, tuning and model selection uses only the development part (5-fold cross-validation).
- All three approaches (rules, ML, embeddings) use the **same** split and the same metrics.

## Evaluation data for performance
Accuracy is measured on the labelled dataset above. For the processing-time experiment, build separate folders of **100, 1,000 and 10,000** files (sample with repetition from the dataset) and measure extraction and classification time per stage.

## Checklist
- [ ] Decide language scope (English only, or add Hungarian)
- [ ] Freeze the category definitions
- [ ] Collect 20 files per category (pilot) and check the extraction results
- [ ] Reach 60+ files per category, balanced
- [ ] Fill in `sources.csv` and `excluded.csv`
- [ ] Second labeller checks 10%
- [ ] Create and commit the stratified split (seed and file lists)
