import asyncio
from dataclasses import dataclass

from docsage.config import get_settings
from docsage.index import store
from docsage.index.bm25 import BM25Index
from docsage.index.vectors import embed_one
from docsage.logging import get_logger
from docsage.models import Chunk

log = get_logger(__name__)


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    score: float
    source: str


def _rrf(rankings: list[list[Candidate]], k: int) -> list[Candidate]:
    """Reciprocal Rank Fusion: 1/(k + rank), somado entre ranqueadores."""
    acc: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, cand in enumerate(ranking, start=1):
            acc[cand.chunk.chunk_id] = acc.get(cand.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[cand.chunk.chunk_id] = cand.chunk
    ordered = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
    return [Candidate(by_id[cid], score, "fused") for cid, score in ordered]


async def search(
    query: str, mode: str | None = None
) -> tuple[list[Candidate], dict[str, float]]:
    s = get_settings()
    mode = mode or s.retrieval_mode
    rankings: list[list[Candidate]] = []
    dense_scores: dict[str, float] = {}
    if mode in ("hybrid", "sparse_only"):
        idx = BM25Index.load(s.bm25_index_path)
        rankings.append(
            [Candidate(c, sc, "sparse") for c, sc in idx.search(query, s.top_k_sparse)]
        )
    if mode in ("hybrid", "vector_only"):
        vec = await asyncio.to_thread(embed_one, query)
        dense = [Candidate(c, sc, "dense") for c, sc in store.search(vec, s.top_k_dense)]
        dense_scores = {c.chunk.chunk_id: c.score for c in dense}
        rankings.append(dense)
    if len(rankings) == 1:
        # braço único (vector_only / sparse_only): sem fusão, mas com o mesmo teto de candidatos
        return rankings[0][: s.top_k_fused], dense_scores
    fused = _rrf(rankings, s.rrf_k)
    log.debug("retrieve.fused", n=len(fused))
    return fused[: s.top_k_fused], dense_scores
