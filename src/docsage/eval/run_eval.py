"""Executa o golden set e grava métricas por item + macro-média."""

import asyncio
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from docsage.config import get_settings
from docsage.eval import dataset, ragas_metrics, retrieval_metrics
from docsage.eval.report import render
from docsage.generate.answer import answer as generate
from docsage.logging import configure, get_logger
from docsage.retrieve.pipeline import retrieve
from docsage.telemetry import get_tracer

configure()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)


def _settings_snapshot() -> dict:
    s = get_settings()
    return {
        "chat_model": s.chat_model,
        "judge_model": s.judge_model,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model if s.rerank_enabled else None,
        "retrieval_mode": s.retrieval_mode,
        "rerank_enabled": s.rerank_enabled,
        "top_k_dense": s.top_k_dense,
        "top_k_sparse": s.top_k_sparse,
        "top_k_fused": s.top_k_fused,
        "top_k_rerank": s.top_k_rerank,
        "rrf_k": s.rrf_k,
        "abstain_threshold": s.abstain_threshold,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


async def evaluate_item(item: dataset.GoldenItem, k: int, sem: asyncio.Semaphore) -> dict:
    async with sem:
        started = time.perf_counter()
        retrieval = await retrieve(item.question)
        ids = [c.chunk.chunk_id for c in retrieval.candidates]
        context = "\n\n".join(c.chunk.text for c in retrieval.candidates)
        result = {
            "id": item.id,
            "category": item.category,
            "answerable": item.answerable,
            "retrieved": ids,
            "recall_at_k": retrieval_metrics.recall_at_k(ids, item.gold_chunk_ids, k),
            "ndcg_at_k": retrieval_metrics.ndcg_at_k(ids, item.gold_chunk_ids, k),
            "mrr": retrieval_metrics.mrr(ids, item.gold_chunk_ids),
            "abstained": retrieval.abstain,
            "top_score": retrieval.top_score,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        if retrieval.abstain:
            # abster em item não-respondível é acerto; abster em item respondível é
            # resposta faltando e não pode contar como faithfulness 1.0.
            result.update(
                faithfulness=1.0 if not item.answerable else 0.0,
                answer_relevance=0.0,
                context_recall=0.0,
                citation_validity=1.0,
                context_precision=0.0,
                unsupported_ratio=0.0,
            )
            return result

        response = await generate(item.question, retrieval)
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        used = [c.chunk_id for c in response.citations]
        scores = await ragas_metrics.score_item(
            item.question, response.answer, item.answer, context
        )
        result.update(
            scores,
            citation_validity=response.citation_report.validity,
            context_precision=retrieval_metrics.context_precision(ids, used),
            unsupported_ratio=1.0 - response.citation_report.coverage,
            answer=response.answer,
            cost_usd=response.usage.cost_usd,
            latency_ms=response.latency_ms,
        )
        tracer = get_tracer()
        for name, value in scores.items():
            tracer.score(name, value)
        return result


async def evaluate_item_safe(item: dataset.GoldenItem, k: int, sem: asyncio.Semaphore) -> dict:
    """Um item que falha não pode derrubar a medição inteira.

    O item fica marcado com `error` e é excluído das médias; a contagem aparece em
    `n_errors` e no relatório. Falha silenciosa virando zero seria mentira.
    """
    try:
        return await evaluate_item(item, k, sem)
    except Exception as exc:
        log.error("eval.item_error", item=item.id, error=type(exc).__name__, detail=str(exc)[:300])
        return {
            "id": item.id,
            "category": item.category,
            "answerable": item.answerable,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "retrieved": [],
            "abstained": False,
            "latency_ms": 0,
            "top_score": 0.0,
        }


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    latencies = sorted(r.get("latency_ms", 0) for r in answerable)
    costs = [r.get("cost_usd", 0.0) for r in answerable]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, round(q * (len(values) - 1)))
        return float(values[idx])

    return {
        "n": len(ok),
        "n_errors": len(errors),
        "error_ids": [r["id"] for r in errors],
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_k": retrieval_metrics.macro([r["recall_at_k"] for r in answerable]),
        "ndcg_at_k": retrieval_metrics.macro([r["ndcg_at_k"] for r in answerable]),
        "mrr": retrieval_metrics.macro([r["mrr"] for r in answerable]),
        "context_precision": retrieval_metrics.macro([r["context_precision"] for r in answerable]),
        "context_recall": retrieval_metrics.macro([r["context_recall"] for r in answerable]),
        "faithfulness": retrieval_metrics.macro([r["faithfulness"] for r in answerable]),
        "answer_relevance": retrieval_metrics.macro([r["answer_relevance"] for r in answerable]),
        "citation_validity": retrieval_metrics.macro([r["citation_validity"] for r in ok]),
        "abstention_accuracy": retrieval_metrics.macro(
            [float(r["abstained"]) for r in unanswerable]
        ),
        "false_abstention_rate": retrieval_metrics.macro(
            [float(r["abstained"]) for r in answerable]
        ),
        "latency_p50_ms": pct(latencies, 0.50),
        "latency_p95_ms": pct(latencies, 0.95),
        "cost_usd_total": sum(costs),
        "cost_usd_per_item": (sum(costs) / len(answerable)) if answerable else 0.0,
    }


async def run(limit: int | None, k: int, out: Path, label: str) -> None:
    items = dataset.load(Path("eval/golden.jsonl"), Path("data/chunks.jsonl"))
    if limit:
        items = items[:limit]
    settings = _settings_snapshot()
    sem = asyncio.Semaphore(get_settings().eval_concurrency)
    log.info("eval.start", n=len(items), mode=label, concurrency=get_settings().eval_concurrency)
    started = datetime.now(UTC)
    results = await asyncio.gather(*(evaluate_item_safe(i, k, sem) for i in items))
    metrics = summarize(results)
    if metrics["n_errors"]:
        log.warning("eval.errors", n=metrics["n_errors"], ids=metrics["error_ids"])
    payload = {
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1),
        "k": k,
        "settings": settings,
        "metrics": metrics,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    render(payload, out.with_suffix(".md"))
    get_tracer().flush()
    log.info("eval.done", **{m: (round(v, 4) if isinstance(v, float) else v)
                             for m, v in metrics.items()})


@app.command()
def main(
    limit: int | None = typer.Option(None),
    k: int = typer.Option(10),
    out: Path = typer.Option(Path("eval-out/run.json")),
    label: str = typer.Option("docsage"),
    baseline: bool = typer.Option(False),
) -> None:
    if baseline:
        os.environ["RETRIEVAL_MODE"] = "vector_only"
        os.environ["RERANK_ENABLED"] = "false"
        os.environ["ABSTAIN_THRESHOLD"] = "0.0"
        label = "baseline"
        get_settings.cache_clear()
    asyncio.run(run(limit, k, out, label))


if __name__ == "__main__":
    app()
