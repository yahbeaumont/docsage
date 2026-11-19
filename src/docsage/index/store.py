import atexit
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from docsage.config import get_settings
from docsage.models import Chunk

# O Qdrant só aceita ids do tipo uint64 ou UUID; chunk_id é sha1 truncado em hex.
# O ponto usa uuid5 determinístico e o chunk_id original fica no payload.
_NAMESPACE = uuid.UUID("6f1c0f4e-1f2b-5c3d-8a90-3d5f7c1b2e44")

_client: QdrantClient | None = None


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def client() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        if s.qdrant_path:
            _client = QdrantClient(path=s.qdrant_path, timeout=30)
        else:
            _client = QdrantClient(url=s.qdrant_url, timeout=30)
        # o modo embutido segura um lock de arquivo; fechar explicitamente no fim
        # do processo evita o ImportError do __del__ durante o shutdown.
        atexit.register(reset_client)
    return _client


def reset_client() -> None:
    """Fecha e esquece o cliente (usado pelos testes e pelo rebuild do índice)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collection() -> None:
    s = get_settings()
    existing = {c.name for c in client().get_collections().collections}
    if s.collection in existing:
        return
    client().create_collection(
        collection_name=s.collection,
        vectors_config=qm.VectorParams(size=s.embed_dim, distance=qm.Distance.COSINE),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
    )
    client().create_payload_index(
        collection_name=s.collection, field_name="path", field_schema=qm.PayloadSchemaType.KEYWORD
    )


def upsert(chunks: list[Chunk], vectors: list[list[float]], batch: int = 128) -> None:
    s = get_settings()
    for start in range(0, len(chunks), batch):
        slice_chunks = chunks[start : start + batch]
        client().upsert(
            collection_name=s.collection,
            points=qm.Batch(
                ids=[point_id(c.chunk_id) for c in slice_chunks],
                vectors=vectors[start : start + batch],
                payloads=[c.model_dump() for c in slice_chunks],
            ),
            wait=True,
        )


def search(vector: list[float], k: int) -> list[tuple[Chunk, float]]:
    s = get_settings()
    hits = client().query_points(
        collection_name=s.collection, query=vector, limit=k, with_payload=True
    ).points
    return [(Chunk.model_validate(h.payload), float(h.score)) for h in hits]


def count() -> int:
    return client().count(collection_name=get_settings().collection, exact=True).count


def reset() -> None:
    s = get_settings()
    if s.collection in {c.name for c in client().get_collections().collections}:
        client().delete_collection(s.collection)
