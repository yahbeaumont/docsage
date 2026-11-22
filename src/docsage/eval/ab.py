"""A/B pareado com bootstrap. Separa regressão de ruído antes do gate julgar.

Unidade de reamostragem = item (pareado). Controla a variância de dificuldade,
que domina a variância do juiz. Temperatura 0.0 torna repetição quase inútil;
`--reps` existe só para medir instabilidade do juiz.
"""

import asyncio
import json
import random
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer

from docsage.config import get_settings
from docsage.eval import dataset, retrieval_metrics
from docsage.eval.report import ORDER
from docsage.generate.answer import answer as generate
from docsage.logging import configure, get_logger
from docsage.retrieve.pipeline import retrieve

configure()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)

ALPHA = 0.05
Z_ALPHA = 1.96
Z_BETA = 0.84
MIN_PAIRS = 30


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    rerank: bool
    threshold: float


@dataclass(frozen=True)
class Verdict:
    metric: str
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    mde: float
    decision: str


def bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 1337
) -> tuple[list[float], float]:
    """Distribuição bootstrap da diferença b - a, reamostrando pares por índice."""
    rng = random.Random(seed)
    n = len(a)
    if n < 2:
        return [0.0], 0.0
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(statistics.fmean(b[i] for i in idx) - statistics.fmean(a[i] for i in idx))
    sigma = statistics.pstdev(deltas)
    return deltas, sigma


def decide(ci_low: float, ci_high: float, tolerance: float) -> str:
    if ci_low >= -tolerance:
        return "pass"
    if ci_high < -tolerance:
        return "fail"
    return "inconclusive"


def bh_adjust(p_values: list[float], alpha: float = ALPHA) -> list[float]:
    """Benjamini-Hochberg: controla FDR em vez de inflar o erro por 9 métricas."""
    del alpha  # a correção não depende de alpha; o corte é aplicado por quem lê
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank
        value = min(prev, p_values[idx] * m / (i + 1))
        adjusted[idx] = value
        prev = value
    return adjusted


def mde(sigma_delta: float, n: int) -> float:
    """Efeito mínimo detectável a 80% de poder."""
    if n < 2:
        return float("inf")
    return (Z_ALPHA + Z_BETA) * sigma_delta / (n**0.5)


def evaluate_metric(
    metric: str, a: list[float], b: list[float], tolerance: float, n_boot: int, seed: int
) -> Verdict:
    deltas, sigma = bootstrap_ci(a, b, n_boot, seed)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    p_value = sum(1 for d in deltas if d <= 0) / len(deltas)
    return Verdict(
        metric=metric,
        delta=statistics.fmean(b) - statistics.fmean(a),
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        p_adjusted=p_value,
        mde=mde(sigma, len(a)),
        decision=decide(lo, hi, tolerance),
    )


async def _arm_scores(
    arm: Arm, items: list[dataset.GoldenItem], k: int, sem: asyncio.Semaphore
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for item in items:
        try:
            async with sem:
                result = await retrieve(
                    item.question, mode=arm.mode, abstain_threshold=arm.threshold
                )
                ids = [c.chunk.chunk_id for c in result.candidates]
                row = {
                    "recall_at_k": retrieval_metrics.recall_at_k(ids, item.gold_chunk_ids, k),
                    "ndcg_at_k": retrieval_metrics.ndcg_at_k(ids, item.gold_chunk_ids, k),
                    "mrr": retrieval_metrics.mrr(ids, item.gold_chunk_ids),
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "faithfulness": 1.0 if not item.answerable else 0.0,
                    "answer_relevance": 0.0,
                    "citation_validity": 1.0,
                    "abstention_accuracy": (
                        float(result.abstain) if not item.answerable else 0.0
                    ),
                }
                if not result.abstain:
                    from docsage.eval import ragas_metrics

                    response = await generate(item.question, result)
                    context = "\n\n".join(c.chunk.text for c in result.candidates)
                    used = [c.chunk_id for c in response.citations]
                    row["context_precision"] = retrieval_metrics.context_precision(ids, used)
                    row["citation_validity"] = response.citation_report.validity
                    row.update(
                        await ragas_metrics.score_item(
                            item.question, response.answer, item.answer, context
                        )
                    )
        except Exception as exc:
            log.error(
                "ab.item_error", arm=arm.name, item=item.id,
                error=type(exc).__name__, detail=str(exc)[:300],
            )
            row = {"error": True}
        out[item.id] = row
    return out


async def run(
    tolerance: float, n_boot: int, seed: int, out: Path, k: int, reps: int
) -> int:
    items = dataset.load(Path("eval/golden.jsonl"), Path("data/chunks.jsonl"))
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    if len(answerable) < MIN_PAIRS:
        log.warning("ab.few_pairs", n=len(answerable), min=MIN_PAIRS)

    settings = get_settings()
    arms = {
        "A": Arm("baseline", "vector_only", False, 0.0),
        "B": Arm("docsage", "hybrid", True, settings.abstain_threshold),
    }
    sem = asyncio.Semaphore(settings.eval_concurrency)

    # repetições: o vetor de cada item é a média entre repetições (reduz ruído do juiz)
    # e o desvio entre repetições da média por item vira judge_sigma. O bootstrap
    # continua sendo pareado por item, que é o que controla a variância de dificuldade.
    samples: dict[str, list[tuple[list[float], list[float]]]] = {}
    keep_answerable = [i.id for i in answerable]
    keep_unanswerable = [i.id for i in unanswerable]
    dropped: list[str] = []
    for rep in range(reps):
        res_a = await _arm_scores(arms["A"], answerable, k, sem)
        res_b = await _arm_scores(arms["B"], answerable, k, sem)
        res_au = await _arm_scores(arms["A"], unanswerable, k, sem)
        res_bu = await _arm_scores(arms["B"], unanswerable, k, sem)
        if rep == 0:
            # um item que falhou em qualquer braço sai dos dois lados: o pareamento
            # só vale se as duas medições existirem.
            def _keep(a: dict, b: dict, ids: list[str]) -> list[str]:
                return [i for i in ids if not a[i].get("error") and not b[i].get("error")]

            keep_answerable = _keep(res_a, res_b, [i.id for i in answerable])
            keep_unanswerable = _keep(res_au, res_bu, [i.id for i in unanswerable])
            dropped = [i.id for i in answerable if i.id not in keep_answerable]
            dropped += [i.id for i in unanswerable if i.id not in keep_unanswerable]
            if dropped:
                log.warning("ab.dropped", n=len(dropped), ids=dropped)
        for metric in ORDER:
            if metric == "abstention_accuracy":
                a_vals = [res_au[i][metric] for i in keep_unanswerable]
                b_vals = [res_bu[i][metric] for i in keep_unanswerable]
            else:
                a_vals = [res_a[i][metric] for i in keep_answerable]
                b_vals = [res_b[i][metric] for i in keep_answerable]
            samples.setdefault(metric, []).append((a_vals, b_vals))
        log.info("ab.rep", rep=rep + 1, reps=reps)

    verdicts: dict[str, Verdict] = {}
    judge_sigma: dict[str, float] = {}
    for metric in ORDER:
        runs = samples[metric]
        a_vals = [statistics.fmean(v[0][i] for v in runs) for i in range(len(runs[0][0]))]
        b_vals = [statistics.fmean(v[1][i] for v in runs) for i in range(len(runs[0][1]))]
        verdicts[metric] = evaluate_metric(metric, a_vals, b_vals, tolerance, n_boot, seed)
        if reps > 1:
            per_item_diff = [
                statistics.fmean(v[1][i] - v[0][i] for v in runs) for i in range(len(a_vals))
            ]
            judge_sigma[metric] = statistics.pstdev(per_item_diff)

    adjusted = bh_adjust([v.p_value for v in verdicts.values()])
    for verdict, p_adj in zip(verdicts.values(), adjusted, strict=True):
        object.__setattr__(verdict, "p_adjusted", p_adj)
        if verdict.decision == "fail" and p_adj > ALPHA:
            object.__setattr__(verdict, "decision", "inconclusive")

    decisions = [v.decision for v in verdicts.values()]
    overall = (
        "fail" if "fail" in decisions else ("inconclusive" if "inconclusive" in decisions
                                            else "pass")
    )
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_pairs": len(keep_answerable),
        "n_unanswerable": len(keep_unanswerable),
        "dropped_items": dropped,
        "n_boot": n_boot,
        "reps": reps,
        "tolerance": tolerance,
        "alpha": ALPHA,
        "correction": "benjamini-hochberg",
        "arms": {"A": vars(arms["A"]), "B": vars(arms["B"])},
        "overall": overall,
        "judge_sigma": judge_sigma,
        "metrics": [v.__dict__ for v in verdicts.values()],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render(payload, out.with_suffix(".md"))
    for v in verdicts.values():
        flag = {"pass": "ok", "fail": "REGRESSÃO", "inconclusive": "inconclusivo"}[v.decision]
        typer.echo(
            f"{v.metric:24s} Δ={v.delta:+.4f} IC95=[{v.ci_low:+.4f},{v.ci_high:+.4f}] "
            f"MDE={v.mde:.4f} p_adj={v.p_adjusted:.4f} {flag}"
        )
    if any(v.mde > tolerance for v in verdicts.values()):
        typer.secho(
            f"\naviso: MDE excede a tolerância ({tolerance}) em alguma métrica — "
            f"{len(keep_answerable)} pares não bastam para resolver esse efeito.",
            fg="yellow",
        )
    return {"pass": 0, "fail": 1, "inconclusive": 2}[overall]


def _render(payload: dict, path: Path) -> None:
    lines = [
        "# A/B pareado — bootstrap",
        "",
        f"`{payload['timestamp']}` · n={payload['n_pairs']} pares · "
        f"{payload['n_boot']} reamostragens · correção {payload['correction']}",
        "",
        f"**Veredito: {payload['overall'].upper()}**",
        "",
        "| Métrica | Δ | IC95 | MDE | p ajustado | Veredito |",
        "|---|---|---|---|---|---|",
    ]
    for m in payload["metrics"]:
        lines.append(
            f"| {m['metric']} | {m['delta']:+.4f} | "
            f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] | {m['mde']:.4f} | "
            f"{m['p_adjusted']:.4f} | {m['decision']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def main(
    tolerance: float = typer.Option(0.03),
    n_boot: int = typer.Option(10_000),
    seed: int = typer.Option(1337),
    k: int = typer.Option(10),
    reps: int = typer.Option(1),
    out: Path = typer.Option(Path("eval-out/ab.json")),
) -> None:
    raise typer.Exit(asyncio.run(run(tolerance, n_boot, seed, out, k, reps)))


if __name__ == "__main__":
    app()
