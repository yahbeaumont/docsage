from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from docsage.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embed_model)


def embed(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(texts, batch_size=get_settings().embed_batch,
                           normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_one(text: str) -> list[float]:
    return embed([text])[0].tolist()
