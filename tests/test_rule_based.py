import pytest

from fileorg.classifiers import get_classifier
from fileorg.classifiers.rule_based import RuleBasedClassifier
from fileorg.models import FileRecord


def rec(text, filename="doc.txt", metadata=None):
    return FileRecord(path="/" + filename, rel_path=filename, filename=filename, extension=".txt", size_bytes=1,
                      modified="", ctime="", mime_type=None, text=text, metadata=metadata or {})


SAMPLES = {
    "invoice": "INVOICE\nAcme Ltd\nInvoice number: INV-2025-0042\nIssue date: 3 May 2025\nDue date: 2 June 2025\n"
               "Bill to\nBeta GmbH\nDescription Qty Unit price Amount\nConsulting 10 $90.00 $900.00\n"
               "Subtotal: $900.00\nVAT (20%): $180.00\nTotal due: $1,080.00\nPayment terms: net 30 days. IBAN GB12 3456",
    "receipt": "Corner Market\n12 High Street\nReceipt #483920\nCashier: Anna\nMilk 1L 1 $1.29\nBread 2 $4.40\n"
               "Subtotal: $5.69\nTax: $0.28\nTOTAL: $5.97\nPaid by: VISA **** 4417\nTHANK YOU FOR YOUR PURCHASE!",
    "contract": "SERVICE AGREEMENT\nThis Agreement is made between Acme Ltd (hereinafter the Supplier) and Beta GmbH.\n"
                "WHEREAS the parties wish to cooperate...\nArticle 1 Term\nArticle 2 Termination\nThe Supplier shall not be "
                "liable for indirect damages. This Agreement is governed by the laws of England.\nSignature: ______",
    "academic_paper": "A Method for Document Classification\nJ. Smith, K. Lee\nAbstract\nIn this paper we propose a new "
                      "approach. Keywords: classification, text\n1. Introduction\nPrior work [1] and Brown et al. showed "
                      "...\n2. Related Work\n3. Methodology\nWe evaluate the baseline on a dataset.\nReferences\n[1] Brown et al. 2020",
    "cv_resume": "Maria Gomez\nmaria@example.com\nProfile\nSoftware engineer.\nWork Experience\nDeveloper, Acme (2016 – 2020)\n"
                 "Education\nBSc Computer Science\nSkills\nPython, SQL\nLanguages\nEnglish, Spanish\nReferences available upon request.",
    "letter": "Anna Smith\n5 Oak Road\n3 May 2025\nAcme Ltd\nDear Sir or Madam,\nI am writing to complain about my order.\n"
              "I look forward to hearing from you.\nYours faithfully,\nAnna Smith",
    "report": "Annual Sustainability Report 2025\nTable of Contents\nExecutive Summary\nKey findings show a 12% drop in "
              "emissions. Recommendations for the board are listed in section 4. See Figure 2 and Table 3. Q3 results...",
    "other": "Chocolate Cake Recipe\nIngredients: 200g flour, 2 tablespoons cocoa\nPreheat the oven to 180C.\nServings: 8",
}


@pytest.mark.parametrize("label,text", SAMPLES.items())
def test_each_category_is_recognised(label, text):
    assert RuleBasedClassifier().predict(rec(text)) == label


def test_title_beats_body_signals():
    # a paid invoice is still an invoice (tie-break rule 3)
    text = "INVOICE\nInvoice number: 7\nStatus: PAID in full\nTotal paid: $50.00\nPaid by: VISA **** 1234"
    assert RuleBasedClassifier().predict(rec(text)) == "invoice"
    # a receipt that mentions the word invoice in the body stays a receipt
    text = "Payment receipt\nThank you for your payment.\nRelates to invoice 77.\nAmount paid: $20"
    assert RuleBasedClassifier().predict(rec(text)) == "receipt"


def test_unknown_and_empty_documents_go_to_other():
    clf = RuleBasedClassifier()
    assert clf.predict(rec("")) == "other"
    assert clf.predict(rec("Some random text without any particular structure.")) == "other"


def test_file_name_and_metadata_features_and_ablation():
    r = rec("", filename="CV_Maria_Gomez.pdf", metadata={"title": "Curriculum Vitae"})
    assert RuleBasedClassifier().predict(r) == "cv_resume"
    assert RuleBasedClassifier(features=["text"]).predict(r) == "other"
    assert RuleBasedClassifier(features=["filename"]).predict(r) == "cv_resume"
    assert RuleBasedClassifier(features=["metadata"], min_score=3).predict(r) == "cv_resume"
    # file-name words must match whole words: "information" is not "form"
    assert RuleBasedClassifier().predict(rec("", filename="information.txt")) == "other"
    with pytest.raises(ValueError):
        RuleBasedClassifier(features=["bogus"])


def test_margin_sends_close_calls_to_other():
    text = SAMPLES["invoice"]
    assert RuleBasedClassifier(margin=1000).predict(rec(text)) == "other"


def test_explain_lists_fired_rules_and_scores_add_up():
    clf = RuleBasedClassifier()
    r = rec(SAMPLES["letter"])
    explained = clf.explain(r)
    assert explained["letter"] and not explained["invoice"]
    assert clf.scores(r)["letter"] == pytest.approx(sum(p for _, p in explained["letter"]))


def test_long_text_keeps_head_and_tail():
    body = "Abstract\n" + "filler " * 10000 + "\nReferences\n[1] Someone et al."
    clf = RuleBasedClassifier(max_chars=2000)
    zones = clf._zones(rec(body))
    assert "References" in zones["text"] and zones["text"].startswith("Abstract")
    assert len(zones["text"]) <= 2001


def test_registry():
    assert get_classifier("rules").name == "rules"
    with pytest.raises(ValueError):
        get_classifier("nope")