"""Índice lexical BM25 persistido em disco."""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from docsage.models import Chunk

TOKEN = re.compile(r"[a-zA-Z0-9_]+")
STOP = frozenset(
    [
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did", "of",
    "to", "in", "on", "for", "with", "and", "or", "if", "it", "this", "that", "as", "at",
    "by", "from", "how", "what", "why", "when", "where", "which", "who"
    ]
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text) if t.lower() not in STOP and len(t) > 1]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.chunks = data["chunks"]
        obj._bm25 = data["bm25"]
        return obj
