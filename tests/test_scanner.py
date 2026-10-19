from fileorg.scanner import scan


def test_scan_filters_extensions_and_hidden(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.DOCX").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x")
    (tmp_path / "d.png").write_bytes(b"x")
    (tmp_path / ".hidden.txt").write_text("x")
    (tmp_path / "~$lock.docx").write_bytes(b"x")
    sub = tmp_path / "invoice"
    sub.mkdir()
    (sub / "e.pdf").write_bytes(b"x")

    names = [p.name for p in scan(tmp_path)]
    assert names == ["a.pdf", "b.DOCX", "c.txt", "e.pdf"]


def test_scan_non_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("x")
    assert [p.name for p in scan(tmp_path, recursive=False)] == ["a.txt"]
