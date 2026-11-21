"""Orquestra as métricas de geração em paralelo, com isolamento por item."""

import asyncio

from docsage.eval import judges
from docsage.logging import get_logger

log = get_logger(__name__)
SEM = asyncio.Semaphore(8)


async def _guard(coro, default: float) -> float:
    async with SEM:
        try:
            return await coro
        except Exception as exc:
            log.warning("judge.failed", error=str(exc))
            return default


async def score_item(
    question: str, answer: str, gold_answer: str, context: str
) -> dict[str, float]:
    faith, relev, recall = await asyncio.gather(
        _guard(judges.faithfulness(answer, context), 0.0),
        _guard(judges.answer_relevance(question, answer), 0.0),
        _guard(judges.context_recall(gold_answer, context), 0.0),
    )
    return {"faithfulness": faith, "answer_relevance": relev, "context_recall": recall}
