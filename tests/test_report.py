import json

from docsage.eval import judges
from docsage.eval.report import compare, render


def _payload(label: str, faith: float) -> dict:
    return {
        "label": label,
        "timestamp": "2026-11-25T00:00:00+00:00",
        "metrics": {"n": 2, "faithfulness": faith, "recall_at_k": 0.5},
        "results": [
            {"id": "f001", "category": "factual", "answerable": True, "faithfulness": 0.2,
             "answer": "resposta ruim"},
            {"id": "u001", "category": "unanswerable", "answerable": False, "faithfulness": 1.0},
        ],
    }


def test_render_lists_metrics_and_worst_cases(tmp_path):
    out = tmp_path / "r.md"
    render(_payload("docsage", 0.8), out)
    text = out.read_text()
    assert "| faithfulness | 0.8000 |" in text
    assert "f001" in text
    assert "u001" not in text


def test_compare_shows_delta(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_payload("a", 0.8)))
    b.write_text(json.dumps(_payload("b", 0.9)))
    table = compare(a, b)
    assert "| faithfulness | 0.800 | 0.900 | +0.100 |" in table


async def test_judges_faithfulness_with_stub(monkeypatch):
    async def fake_json(prompt):
        if "Decomponha" in prompt:
            return {"claims": ["c1", "c2"]}
        return {"supported": "Afirmação" in prompt}

    monkeypatch.setattr(judges, "_json", fake_json)
    assert await judges.faithfulness("resp", "contexto claim") == 1.0


async def test_judges_entails_false(monkeypatch):
    async def fake_json(prompt):
        return {"supported": False}

    monkeypatch.setattr(judges, "_json", fake_json)
    assert await judges.entails("ctx", "claim") is False


async def test_context_recall_counts_claims(monkeypatch):
    async def fake_json(prompt):
        return {"supported": True}

    monkeypatch.setattr(judges, "_json", fake_json)
    assert await judges.context_recall("Uma frase. Outra frase.", "ctx") == 1.0
