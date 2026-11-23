from docsage.generate.cite import extract_markers, strip_invalid, validate


def test_extract():
    assert extract_markers("a [[abc123]] b [[def456]]") == ["abc123", "def456"]


def test_strip_invalid_counts_orphans():
    cleaned, orphan = strip_invalid("ok [[good12]] ruim [[bad999]]", {"good12"})
    assert "bad999" not in cleaned
    assert orphan == 1


def test_validate_reports_coverage():
    _, report = validate("Uma frase [[good12]]. Outra sem citação.", {"good12"})
    assert report.validity == 1.0
    assert report.coverage == 0.5


def test_validate_empty_answer():
    _, report = validate("", set())
    assert report.validity == 1.0
