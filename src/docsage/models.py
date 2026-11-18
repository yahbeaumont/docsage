from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    path: str
    heading_path: tuple[str, ...]
    text: str
    n_tokens: int
    ordinal: int


class Citation(BaseModel):
    chunk_id: str
    doc: str
    heading: str
    span: tuple[int, int]
    score: float


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["hybrid", "vector_only", "sparse_only"] | None = None


class CitationReport(BaseModel):
    total_markers: int
    valid_markers: int
    orphan_markers: int
    sentences: int
    supported_sentences: int

    @property
    def validity(self) -> float:
        return self.valid_markers / self.total_markers if self.total_markers else 1.0

    @property
    def coverage(self) -> float:
        return self.supported_sentences / self.sentences if self.sentences else 1.0


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    citation_report: CitationReport
    usage: Usage
    latency_ms: int
    request_id: str
