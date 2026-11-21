"""Golden set com resolução de `gold_chunk_ids` a partir de caminho + substring."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from docsage.logging import get_logger
from docsage.models import Chunk

log = get_logger(__name__)
MIN_RESOLUTION = 0.90


@dataclass
class GoldenItem:
    id: str
    question: str
    answer: str
    category: str
    answerable: bool
    gold_paths: list[str] = field(default_factory=list)
    gold_text_contains: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.gold_chunk_ids) or not self.answerable


def _load_chunks(chunks_path: Path) -> dict[str, Chunk]:
    return {
        c.chunk_id: c
        for c in (
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def resolve(items: list[GoldenItem], chunks_path: Path) -> list[GoldenItem]:
    chunks = _load_chunks(chunks_path)
    for item in items:
        if not item.answerable or not item.gold_paths:
            continue
        needles = [n.lower() for n in item.gold_text_contains]
        # Um chunk canônico por documento: o primeiro (ordinal) que contém a
        # substring de referência. Sem isso, um doc com 10 chunks contendo o termo
        # gera gold arbitrário e recall@k que nunca chega a 1.
        hits: dict[str, Chunk] = {}
        for chunk in chunks.values():
            if chunk.path not in item.gold_paths:
                continue
            if not any(n in chunk.text.lower() for n in needles):
                continue
            current = hits.get(chunk.path)
            if current is None or chunk.ordinal < current.ordinal:
                hits[chunk.path] = chunk
        item.gold_chunk_ids = [hits[path].chunk_id for path in sorted(hits)]
    return items


def load(path: Path, chunks_path: Path, strict: bool = True) -> list[GoldenItem]:
    raw = [
        GoldenItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = resolve(raw, chunks_path)

    answerable = [i for i in items if i.answerable]
    resolved = [i for i in answerable if i.gold_chunk_ids]
    ratio = len(resolved) / len(answerable) if answerable else 1.0

    unresolved = [i.id for i in answerable if not i.gold_chunk_ids]
    log.info("golden.resolution", ratio=round(ratio, 3), unresolved=len(unresolved))

    if ratio < MIN_RESOLUTION:
        message = f"resolução {ratio:.1%} < {MIN_RESOLUTION:.0%}; itens órfãos: {unresolved[:10]}"
        if strict:
            raise ValueError(message)
        log.warning("golden.low_resolution", detail=message)

    return items


def write(path: Path, items: list[GoldenItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            payload = {
                "id": item.id, "category": item.category, "answerable": item.answerable,
                "question": item.question, "answer": item.answer,
                "gold_paths": item.gold_paths, "gold_text_contains": item.gold_text_contains,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
