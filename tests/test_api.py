import pytest
from fastapi.testclient import TestClient

from docsage.api import main as api
from docsage.models import CitationReport, QueryResponse, Usage
from docsage.retrieve.pipeline import RetrievalResult


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.store, "ensure_collection", lambda: None)
    monkeypatch.setattr(api.store, "count", lambda: 42)
    with TestClient(api.app) as test_client:
        yield test_client


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_points(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["points"] == 42


def test_readyz_fails_on_empty_index(client, monkeypatch):
    monkeypatch.setattr(api.store, "count", lambda: 0)
    response = client.get("/readyz")
    assert response.status_code == 503


def test_readyz_fails_when_store_unavailable(client, monkeypatch):
    def boom():
        raise RuntimeError("sem conexão")

    monkeypatch.setattr(api.store, "count", boom)
    assert client.get("/readyz").status_code == 503


def test_query_returns_answer_and_request_id(client, monkeypatch):
    async def fake_retrieve(question, mode=None, top_k=None, abstain_threshold=None):
        return RetrievalResult([], True, 0.0)

    async def fake_answer(question, retrieval):
        return QueryResponse(
            answer="Não encontrado na documentação.",
            citations=[],
            abstained=True,
            citation_report=CitationReport(
                total_markers=0, valid_markers=0, orphan_markers=0, sentences=0,
                supported_sentences=0,
            ),
            usage=Usage(),
            latency_ms=1,
            request_id="abc123",
        )

    monkeypatch.setattr(api, "retrieve", fake_retrieve)
    monkeypatch.setattr(api, "answer", fake_answer)
    response = client.post("/v1/query", json={"question": "Como usar Depends?"})
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert "x-request-id" in response.headers


def test_query_rejects_short_question(client):
    assert client.post("/v1/query", json={"question": "ab"}).status_code == 422


def test_metrics_endpoint_exposed(client):
    assert client.get("/metrics").status_code == 200
