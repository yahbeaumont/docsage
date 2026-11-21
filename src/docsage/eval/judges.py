"""Juízes LLM. Rodam em modelo separado do gerador para reduzir circularidade."""

import re

from docsage.config import get_settings
from docsage.llm import get_client

DECOMPOSE = """Decomponha a resposta em afirmações atômicas e verificáveis.
Responda só com JSON: {{"claims": ["...", "..."]}}
Resposta:
{answer}"""

ENTAIL = """Uma afirmação é suportada se o contexto a implica diretamente.
Responda só com JSON: {{"supported": true|false, "why": "..."}}
Contexto:
{context}
Afirmação: {claim}"""

RELEVANCE = """Gere 3 perguntas que a resposta abaixo responderia melhor.
Responda só com JSON: {{"questions": ["...", "...", "..."]}}
Resposta:
{answer}"""

_CTX_CLAIM = re.compile(r"[^.!?]+[.!?]")


async def _json(prompt: str) -> dict:
    from docsage.logging import get_logger

    try:
        return await get_client().complete_json(prompt, model=get_settings().judge_model)
    except Exception as exc:
        get_logger(__name__).warning("judge.invalid_json", error=str(exc))
        raise


async def decompose(answer: str) -> list[str]:
    data = await _json(DECOMPOSE.format(answer=answer))
    return [c.strip() for c in data.get("claims", []) if c.strip()]


async def entails(context: str, claim: str) -> bool:
    data = await _json(ENTAIL.format(context=context[:6000], claim=claim))
    return bool(data.get("supported"))


async def faithfulness(answer: str, context: str) -> float:
    claims = await decompose(answer)
    if not claims:
        return 1.0
    supported = [await entails(context, c) for c in claims]
    return sum(supported) / len(supported)


async def generated_questions(answer: str) -> list[str]:
    data = await _json(RELEVANCE.format(answer=answer))
    return data.get("questions", [])[:3]


def cosine(a: str, b: str) -> float:
    from docsage.index.vectors import embed

    va, vb = embed([a, b])
    return float((va @ vb) / ((va @ va) ** 0.5 * (vb @ vb) ** 0.5))


async def answer_relevance(question: str, answer: str) -> float:
    questions = await generated_questions(answer)
    if not questions:
        return 0.0
    return sum(cosine(question, q) for q in questions) / len(questions)


async def context_recall(gold_answer: str, context: str) -> float:
    claims = [c.strip() for c in _CTX_CLAIM.findall(gold_answer) if c.strip()]
    if not claims:
        return 1.0
    hits = [await entails(context, c) for c in claims]
    return sum(hits) / len(hits)
