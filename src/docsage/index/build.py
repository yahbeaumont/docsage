from pathlib import Path

from docsage.config import get_settings
from docsage.corpus.fetch import fetch, list_markdown
from docsage.corpus.parse import parse_document
from docsage.index import store
from docsage.index.bm25 import BM25Index
from docsage.index.vectors import embed
from docsage.logging import get_logger
from docsage.models import Chunk

log = get_logger(__name__)


def build_corpus(repo: str, commit: str, out: Path) -> Path:
    root = fetch(repo, commit, out)
    chunks_path = Path("data/chunks.jsonl")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with chunks_path.open("w", encoding="utf-8") as fh:
        for doc in list_markdown(root):
            for chunk in parse_document(doc, root):
                fh.write(chunk.model_dump_json() + "\n")
                total += 1
    log.info("corpus.chunked", chunks=total, path=str(chunks_path))
    return chunks_path


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def build_index(chunks_path: Path) -> None:
    s = get_settings()
    chunks = load_chunks(chunks_path)
    log.info("index.start", chunks=len(chunks))

    BM25Index(chunks).save(s.bm25_index_path)
    log.info("index.bm25.saved", path=s.bm25_index_path)

    store.reset()
    store.ensure_collection()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 512):
        batch = chunks[start : start + 512]
        vectors.extend(embed([c.text for c in batch]).tolist())
        log.info("index.embed", done=min(start + 512, len(chunks)), total=len(chunks))
    store.upsert(chunks, vectors)
    log.info("index.done", points=store.count())
