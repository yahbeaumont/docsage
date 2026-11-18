from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: Literal["openai", "ollama", "vllm"] = "openai"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    # modelos com raciocínio gastam o orçamento de saída pensando; "minimal" evita
    # resposta vazia quando max_answer_tokens é pequeno. Vazio = não envia o campo.
    reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = ""

    # --- embeddings / rerank ---
    # multilíngue: as perguntas do golden set são em PT e o corpus em EN;
    # bge-small-en (só inglês) perdia o pareamento cross-lingual.
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384
    embed_batch: int = 64

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 512 tokens em CPU custa ~3x mais por par sem ganho de nDCG no corpus;
    # 192 cobre o chunk típico (média 189 tokens).
    rerank_max_length: int = 192
    rerank_enabled: bool = True
    cohere_api_key: str = ""

    # --- vector store ---
    # qdrant_path vazio usa o servidor em qdrant_url; com caminho preenchido o cliente
    # roda embutido (QdrantClient(path=...)) e dispensa Docker.
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""
    collection: str = "docsage"
    bm25_index_path: str = ".bm25/index.pkl"

    top_k_dense: int = 50
    top_k_sparse: int = 50
    top_k_fused: int = 40
    top_k_rerank: int = 8
    rrf_k: int = 60

    retrieval_mode: Literal["hybrid", "vector_only", "sparse_only"] = "hybrid"
    abstain_threshold: float = 0.28
    max_context_tokens: int = 6000
    max_answer_tokens: int = 700

    request_timeout_s: float = 30.0
    max_retries: int = 3
    eval_concurrency: int = Field(default=6, ge=1, le=32)
    # o proxy devolve 200 com choices nulo quando recebe rajada demais; o semáforo
    # do cliente limita as chamadas simultâneas independentemente do eval_concurrency.
    llm_concurrency: int = Field(default=4, ge=1, le=32)

    @field_validator("embed_dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embed_dim deve ser > 0")
        return v

    @field_validator("abstain_threshold")
    @classmethod
    def _threshold_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("abstain_threshold deve ser finito")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
