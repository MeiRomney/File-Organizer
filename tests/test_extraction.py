import pytest

from fileorg.extraction import extract
from fileorg.pipeline import process_directory, read_jsonl, write_jsonl


def test_txt_utf8(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Invoice No. 123\n\n\n\nTotal due: 50 EUR", encoding="utf-8")
    r = extract(f)
    assert r.status == "ok"
    assert "Invoice No. 123" in r.text
    assert "\n\n\n" not in r.text  # blank lines are collapsed


def test_txt_legacy_hungarian_encoding(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes("árvíztűrő tükörfúrógép".encode("cp1250"))
    assert extract(f).text == "árvíztűrő tükörfúrógép"


def test_txt_truncation_does_not_break_multibyte_chars(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("ő" * 100, encoding="utf-8")
    r = extract(f, max_chars=10)
    assert r.status == "ok"
    assert set(r.text) == {"ő"}


def test_empty_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("   \n")
    assert extract(f).status == "empty"


def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    assert extract(f).status == "unsupported"


def test_corrupt_pdf_reports_error_instead_of_raising(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf")
    r = extract(f)
    assert r.status == "error"
    assert r.error


def test_docx_text_tables_and_metadata(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.core_properties.author = "Jane Doe"
    doc.core_properties.title = "Service agreement"
    doc.add_paragraph("This Agreement is made between the parties.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Term"
    table.rows[0].cells[1].text = "12 months"
    f = tmp_path / "a.docx"
    doc.save(f)

    r = extract(f)
    assert r.status == "ok"
    assert "Agreement is made" in r.text
    assert "Term | 12 months" in r.text
    assert r.metadata["author"] == "Jane Doe"
    assert r.metadata["title"] == "Service agreement"


def test_pdf_text_and_page_count(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    f = tmp_path / "a.pdf"
    c = canvas.Canvas(str(f))
    c.drawString(72, 750, "Abstract. We propose a method.")
    c.showPage()
    c.drawString(72, 750, "References")
    c.save()

    r = extract(f)
    assert r.status == "ok"
    assert "We propose a method" in r.text
    assert r.metadata["pages"] == 2
    assert "References" not in extract(f, max_pages=1).text


def test_labelled_pipeline_and_jsonl_roundtrip(tmp_path):
    (tmp_path / "invoice").mkdir()
    (tmp_path / "letter").mkdir()
    (tmp_path / "invoice" / "1.txt").write_text("Invoice total due")
    (tmp_path / "letter" / "2.txt").write_text("Dear Sir or Madam")
    (tmp_path / "loose.txt").write_text("no label here")

    records = process_directory(tmp_path, labelled=True)
    labels = {r.rel_path: r.label for r in records}
    assert labels == {"invoice/1.txt": "invoice", "letter/2.txt": "letter", "loose.txt": None}

    out = tmp_path / "records.jsonl"
    write_jsonl(records, out)
    assert [r.to_dict() for r in read_jsonl(out)] == [r.to_dict() for r in records]
