import time
import uuid

from docsage.config import get_settings
from docsage.generate.cite import extract_markers, validate
from docsage.generate.prompts import SYSTEM, USER, build_context
from docsage.llm import get_client
from docsage.logging import get_logger
from docsage.models import Citation, CitationReport, QueryResponse, Usage
from docsage.retrieve.pipeline import RetrievalResult
from docsage.telemetry import get_tracer

log = get_logger(__name__)

# US$ por token (entrada, saída). Modelo fora da tabela reporta custo 0 e avisa:
# preferimos custo ausente a custo inventado.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6),
    "gpt-4.1-mini": (0.40 / 1e6, 1.60 / 1e6),
    "gpt-5-mini": (0.25 / 1e6, 2.00 / 1e6),
    "gpt-5-nano": (0.05 / 1e6, 0.40 / 1e6),
    "gemini-3-flash-preview": (0.50 / 1e6, 3.00 / 1e6),
}
ABSTAIN_MESSAGE = "Não encontrado na documentação."


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICES:
        log.warning("cost.unknown_model", model=model)
        return 0.0
    inp, out = PRICES[model]
    return prompt_tokens * inp + completion_tokens * out


def _empty_report() -> CitationReport:
    return CitationReport(
        total_markers=0, valid_markers=0, orphan_markers=0, sentences=0, supported_sentences=0
    )


def _build_citations(cleaned: str, retrieval: RetrievalResult) -> list[Citation]:
    by_id = {c.chunk.chunk_id: c for c in retrieval.candidates}
    citations: list[Citation] = []
    for cid in dict.fromkeys(m for m in extract_markers(cleaned) if m in by_id):
        cand = by_id[cid]
        citations.append(
            Citation(
                chunk_id=cid,
                doc=cand.chunk.path,
                heading=" > ".join(cand.chunk.heading_path),
                span=(0, len(cand.chunk.text)),
                score=cand.score,
            )
        )
    return citations


async def answer(question: str, retrieval: RetrievalResult) -> QueryResponse:
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:16]
    s = get_settings()
    tracer = get_tracer()

    with tracer.span("generate", question=question[:200], abstained=retrieval.abstain) as span:
        if retrieval.abstain:
            span.update(output=ABSTAIN_MESSAGE, citation_validity=1.0)
            return QueryResponse(
                answer=ABSTAIN_MESSAGE,
                citations=[],
                abstained=True,
                citation_report=_empty_report(),
                usage=Usage(),
                latency_ms=int((time.perf_counter() - started) * 1000),
                request_id=request_id,
            )

        prompt = USER.format(context=build_context(retrieval.candidates), question=question)
        completion = await get_client().complete(
            prompt, system=SYSTEM, max_tokens=s.max_answer_tokens, model=s.chat_model
        )
        allowed = {c.chunk.chunk_id for c in retrieval.candidates}
        cleaned, report = validate(completion.text, allowed)
        citations = _build_citations(cleaned, retrieval)
        span.update(output=cleaned, citation_validity=report.validity)

    tracer.score("citation_validity", report.validity)
    tracer.score("unsupported_ratio", 1.0 - report.coverage)
    log.info(
        "answer.done",
        request_id=request_id,
        validity=round(report.validity, 3),
        coverage=round(report.coverage, 3),
        citations=len(citations),
    )
    return QueryResponse(
        answer=cleaned,
        citations=citations,
        abstained=False,
        citation_report=report,
        usage=Usage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cost_usd=_cost(s.chat_model, completion.prompt_tokens, completion.completion_tokens),
        ),
        latency_ms=int((time.perf_counter() - started) * 1000),
        request_id=request_id,
    )
