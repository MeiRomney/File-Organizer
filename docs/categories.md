# Document categories (version 1)

Scope: English-language documents in **PDF, DOCX and TXT** format. Scanned documents without a text layer are out of scope (no OCR in version 1).

**Labelling rule:** each file gets exactly **one** label, chosen by the document's *primary purpose*. If, after applying the rules below, a document is still ambiguous, **exclude it from the dataset** instead of guessing.

| # | Category | Folder name |
|---|----------|-------------|
| 1 | Invoice | `invoice` |
| 2 | Receipt | `receipt` |
| 3 | Contract | `contract` |
| 4 | Academic paper | `academic_paper` |
| 5 | CV / resume | `cv_resume` |
| 6 | Letter | `letter` |
| 7 | Report | `report` |
| 8 | Other | `other` |

## 1. invoice
**Definition:** A document issued by a seller asking a buyer to pay for goods or services.
**Typical signals:** invoice number, issue date, due date, bill-to / ship-to, line items, subtotal, VAT/tax, "total due", payment terms, bank details.
**Not this:** a document that confirms payment was already made (receipt); a price quote or offer.

## 2. receipt
**Definition:** Proof that a payment was completed.
**Typical signals:** "paid", "thank you for your purchase", payment method (card/cash), transaction or receipt number, change, short itemised list, POS-style layout.
**Not this:** a request for payment (invoice). A document titled "Invoice" but stamped "PAID" is still an **invoice**, because the title decides (see tie-break rules).

## 3. contract
**Definition:** A legally binding agreement between two or more named parties.
**Includes:** service agreements, NDAs, lease agreements, employment contracts, licence agreements, terms-of-service style agreements that name parties.
**Typical signals:** named parties, numbered clauses, term/termination, liability, governing law, signature block, "hereinafter".
**Not this:** a policy or manual without parties and signatures.

## 4. academic_paper
**Definition:** A scholarly article, conference paper or thesis that presents research.
**Typical signals:** abstract, keywords, introduction, related work, methodology, results, references/bibliography, author affiliations, in-text citations.
**Not this:** an organisational report written for decision makers (report).

## 5. cv_resume
**Definition:** A personal summary of a person's education, work history and skills, used for job applications.
**Typical signals:** contact details, "work experience", "education", "skills", "languages", date ranges, job titles.
**Not this:** a cover letter (letter).

## 6. letter
**Definition:** Formal or business correspondence addressed to a specific recipient.
**Includes:** business letters, cover letters, complaint letters, recommendation letters, official notices.
**Typical signals:** sender and recipient address blocks, date, salutation ("Dear ..."), closing ("Sincerely", "Yours faithfully"), signature.
**Not this:** e-mails, chat messages, newsletters.

## 7. report
**Definition:** A structured document that informs an organisation or the public about findings, status or results, usually with recommendations.
**Includes:** business reports, project/status reports, technical reports, audit reports, annual reports, government and NGO reports.
**Typical signals:** title page, executive summary, table of contents, numbered sections, findings, recommendations, charts and tables.
**Not this:** research papers with an abstract and a reference list (academic_paper).

## 8. other
**Definition:** Any document that clearly fits none of the categories above.
**Includes:** user manuals, forms, meeting notes, flyers, recipes, schedules, policies.
**Purpose:** the "everything else" class. Real folders always contain such files, and the classifier must learn to say "I don't know where this goes".
**Note:** keep this class as varied as possible, and balanced in size with the others.

## Tie-break rules
1. **Title/heading wins** if the document states its own type (for example "INVOICE", "Curriculum Vitae", "Non-Disclosure Agreement").
2. **Purpose beats format:** a contract in table form is still a contract.
3. **Paid invoice** (title says invoice) -> `invoice`; **receipt** (title says receipt or confirms payment only) -> `receipt`.
4. **Thesis or research article** -> `academic_paper`; **company or institutional analysis for practitioners** -> `report`.
5. Still unclear -> **exclude** the file and record it in `data/excluded.csv` with a reason.

## Label quality check
Have a second person (a classmate) independently label a random 10% of the dataset. Report the agreement (percentage and Cohen's kappa) in the thesis, and use disagreements to sharpen the definitions above.
