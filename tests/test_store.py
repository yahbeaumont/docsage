import pytest

from docsage.config import get_settings
from docsage.index import store
from docsage.models import Chunk


@pytest.fixture
def embedded(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("COLLECTION", "test_docsage")
    monkeypatch.setenv("EMBED_DIM", "8")
    get_settings.cache_clear()
    store.reset_client()
    yield
    store.reset_client()
    get_settings.cache_clear()


def _chunk(i: int, dim: int = 8) -> tuple[Chunk, list[float]]:
    chunk = Chunk(
        chunk_id=f"{i:016x}",
        doc_id="d1",
        path=f"docs/x{i}.md",
        heading_path=("H",),
        text=f"texto {i} sobre Depends",
        n_tokens=5,
        ordinal=i,
    )
    vector = [0.0] * dim
    vector[i % dim] = 1.0
    return chunk, vector


def test_point_id_is_deterministic_uuid():
    first = store.point_id("abc123def456abcd")
    assert first == store.point_id("abc123def456abcd")
    assert len(first) == 36


def test_upsert_search_count_reset(embedded):
    store.ensure_collection()
    chunks = [_chunk(i) for i in range(4)]
    store.upsert([c for c, _ in chunks], [v for _, v in chunks])
    assert store.count() == 4

    query = [0.0] * 8
    query[1] = 1.0
    hits = store.search(query, 2)
    assert len(hits) == 2
    assert hits[0][0].chunk_id == chunks[1][0].chunk_id
    assert hits[0][1] >= hits[1][1]

    store.reset()
    store.ensure_collection()
    assert store.count() == 0


def test_ensure_collection_is_idempotent(embedded):
    store.ensure_collection()
    store.ensure_collection()
    assert store.count() == 0
