from functools import lru_cache

from sentence_transformers import CrossEncoder

from docsage.config import get_settings
from docsage.retrieve.hybrid import Candidate


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(get_settings().rerank_model, max_length=get_settings().rerank_max_length)


def rerank(query: str, candidates: list[Candidate], k: int) -> list[Candidate]:
    if not candidates:
        return []
    scores = _model().predict([(query, c.chunk.text) for c in candidates])
    ranked = sorted(
        (Candidate(c.chunk, float(s), "rerank") for c, s in zip(candidates, scores, strict=True)),
        key=lambda c: c.score,
        reverse=True,
    )
    return ranked[:k]
