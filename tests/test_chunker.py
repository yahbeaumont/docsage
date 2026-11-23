from pathlib import Path

from docsage.corpus.parse import chunk_id, parse_document, split_sections, window


def test_split_respects_fences():
    text = "# A\n\ntexto\n\n```python\n# nao e heading\n```\n\n## B\n\nmais"
    sections = split_sections(text)
    assert [h for h, _ in sections] == [("A",), ("A", "B")]


def test_window_overlaps():
    text = "\n\n".join(f"paragrafo {i} " + "palavra " * 60 for i in range(6))
    pieces = window(text, size=120, overlap=40)
    assert len(pieces) > 1
    assert all(p.strip() for p in pieces)


def test_chunk_id_deterministic():
    assert chunk_id("d", "H", 1) == chunk_id("d", "H", 1)
    assert chunk_id("d", "H", 1) != chunk_id("d", "H", 2)


def test_parse_skips_tiny(tmp_path: Path):
    doc = tmp_path / "a.md"
    doc.write_text("# H\n\ncurto\n")
    assert parse_document(doc, tmp_path) == []


def test_parse_keeps_heading_context(tmp_path: Path):
    doc = tmp_path / "a.md"
    doc.write_text("# Top\n\n## Sub\n\n" + "palavra " * 80)
    chunks = parse_document(doc, tmp_path)
    assert chunks and chunks[0].heading_path == ("Top", "Sub")
    assert chunks[0].text.startswith("Top > Sub")
