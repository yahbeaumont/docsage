import pytest

from docsage.generate import answer as answer_module
from docsage.models import Chunk
from docsage.retrieve import pipeline
from docsage.retrieve.hybrid import Candidate
from docsage.retrieve.pipeline import RetrievalResult


def _candidate(score: float = 0.9, cid: str = "a" * 16) -> Candidate:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="d1",
        path="docs/en/tutorial/dependencies/index.md",
        heading_path=("Tutorial", "Dependencies"),
        text="Use Depends para injetar dependências.",
        n_tokens=8,
        ordinal=0,
    )
    return Candidate(chunk, score, "rerank")


@pytest.fixture
def no_rerank(monkeypatch):
    settings = pipeline.get_settings()
    monkeypatch.setattr(settings, "rerank_enabled", False, raising=False)


async def test_retrieve_abstains_when_empty(monkeypatch):
    async def empty(query, mode=None):
        return [], {}

    monkeypatch.setattr(pipeline, "search", empty)
    result = await pipeline.retrieve("pergunta")
    assert result.abstain is True
    assert result.top_score == 0.0
    assert result.candidates == []


async def test_retrieve_uses_threshold_override(monkeypatch, no_rerank):
    async def one(query, mode=None):
        candidate = _candidate(0.5)
        return [candidate], {candidate.chunk.chunk_id: 0.5}

    monkeypatch.setattr(pipeline, "search", one)
    low = await pipeline.retrieve("pergunta", abstain_threshold=0.1)
    high = await pipeline.retrieve("pergunta", abstain_threshold=0.9)
    assert low.abstain is False
    assert high.abstain is True


async def test_retrieve_reranks_when_enabled(monkeypatch):
    async def two(query, mode=None):
        first, second = _candidate(0.1, "a" * 16), _candidate(0.2, "b" * 16)
        return [first, second], {first.chunk.chunk_id: 0.1, second.chunk.chunk_id: 0.2}

    monkeypatch.setattr(pipeline, "search", two)
    monkeypatch.setattr(pipeline, "rerank", lambda q, c, k: list(reversed(c))[:k])
    result = await pipeline.retrieve("pergunta", abstain_threshold=0.0)
    assert result.candidates[0].chunk.chunk_id == "b" * 16


async def test_answer_abstains_without_calling_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("não deveria chamar o modelo")

    monkeypatch.setattr(answer_module, "get_client", boom)
    response = await answer_module.answer("q", RetrievalResult([], True, 0.0))
    assert response.abstained is True
    assert response.citations == []
    assert response.usage.cost_usd == 0.0


async def test_answer_strips_orphan_citations(monkeypatch):
    from docsage.llm import Completion

    async def fake_complete(self, prompt, **kwargs):
        return Completion(
            "Use Depends [[aaaaaaaaaaaaaaaa]]. Inventado [[zzzzzzzzzzzzzzzz]].", 100, 20
        )

    monkeypatch.setattr(answer_module.get_client().__class__, "complete", fake_complete)
    retrieval = RetrievalResult([_candidate(0.9, "a" * 16)], False, 0.9)
    response = await answer_module.answer("Como injetar dependência?", retrieval)
    assert "zzzzzzzzzzzzzzzz" not in response.answer
    assert response.citation_report.orphan_markers == 1
    assert len(response.citations) == 1
    assert response.usage.cost_usd >= 0.0
    assert response.latency_ms >= 0
