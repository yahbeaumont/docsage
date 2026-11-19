import asyncio
import os
from dataclasses import dataclass

from docsage.config import get_settings
from docsage.logging import get_logger
from docsage.retrieve.hybrid import Candidate, search
from docsage.retrieve.rerank import rerank
from docsage.telemetry import get_tracer

log = get_logger(__name__)

# O cross-encoder já usa todas as threads do processo. Rodar N reranks em paralelo
# (um por item do eval) multiplica as threads pelo paralelismo e trava a máquina:
# medido aqui, 6 reranks simultâneos em 6 vCPU renderam 6x menos vazão por item.
# Serializar o rerank mantém a CPU ocupada de forma útil e deixa o paralelismo
# para as chamadas de LLM, que são I/O.
RERANK_CONCURRENCY = int(os.getenv("RERANK_CONCURRENCY", "1"))
_rerank_sem = asyncio.Semaphore(RERANK_CONCURRENCY)


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[Candidate]
    abstain: bool
    top_score: float


async def retrieve(
    query: str,
    mode: str | None = None,
    top_k: int | None = None,
    abstain_threshold: float | None = None,
) -> RetrievalResult:
    """Funde, reranqueia e decide abstenção.

    `abstain_threshold` sobrepõe a configuração: o A/B precisa que cada braço
    use o seu próprio limiar em vez do valor global.
    """
    s = get_settings()
    threshold = s.abstain_threshold if abstain_threshold is None else abstain_threshold
    tracer = get_tracer()
    with tracer.span("retrieve", question=query[:200], mode=mode or s.retrieval_mode) as span:
        fused, dense_scores = await search(query, mode)
        if not fused:
            span.update(top_score=0.0, n=0)
            return RetrievalResult([], True, 0.0)
        if s.rerank_enabled and len(fused) > 1:
            async with _rerank_sem:
                final = await asyncio.to_thread(rerank, query, fused, top_k or s.top_k_rerank)
        else:
            final = fused[: top_k or s.top_k_rerank]
        # O logit do cross-encoder não é calibrado (varia de -10 a +10 conforme o
        # checkpoint). A abstenção usa o cosseno do melhor chunk final, que é
        # limitado e comparável entre modelos; o rerank decide a ordem.
        top = max((dense_scores.get(c.chunk.chunk_id, 0.0) for c in final), default=0.0)
        abstain = top < threshold
        span.update(top_score=round(top, 4), n=len(final))
    log.info("retrieve.done", n=len(final), top=round(top, 4), abstain=abstain)
    return RetrievalResult(final, abstain, top)
