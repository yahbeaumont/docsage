"""Calibra ABSTAIN_THRESHOLD por curva precision/recall com validação cruzada k-fold.

A abstenção é tratada como classificação binária: positivo = abster.
TP = absteve em item não-respondível. FP = absteve em item respondível.
Precisão = TP/(TP+FP) — das abstenções, quantas eram corretas.
Recall   = TP/(TP+FN) — dos não-respondíveis, quantos foram pegos.
"""

import asyncio
import random
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer

from docsage.eval import dataset
from docsage.logging import configure, get_logger
from docsage.retrieve.pipeline import retrieve

configure()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)

MIN_UNANSWERABLE = 8
GRID_POINTS = 60


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    answerable_kept: float
    tp: int
    fp: int
    fn: int


def evaluate_point(
    scores: list[float], labels: list[bool], threshold: float
) -> OperatingPoint:
    """labels[i] = True quando o item é não-respondível (deve abster)."""
    tp = fp = fn = tn = 0
    for score, unanswerable in zip(scores, labels, strict=True):
        abstained = score < threshold
        if unanswerable and abstained:
            tp += 1
        elif unanswerable and not abstained:
            fn += 1
        elif not unanswerable and abstained:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    answerable_kept = tn / (tn + fp) if tn + fp else 1.0
    return OperatingPoint(threshold, precision, recall, f1, answerable_kept, tp, fp, fn)


def curve(scores: list[float], labels: list[bool]) -> list[OperatingPoint]:
    unique = sorted(set(scores))
    if len(unique) <= 1:
        return [evaluate_point(scores, labels, unique[0] if unique else 0.0)]
    step = max(1, len(unique) // GRID_POINTS)
    grid = unique[::step]
    if grid[-1] != unique[-1]:
        grid.append(unique[-1])
    return [evaluate_point(scores, labels, t) for t in grid]


def choose_threshold(
    scores: list[float], labels: list[bool], min_precision: float = 0.60
) -> OperatingPoint:
    """Maximiza F1 sujeito a precisão mínima; relaxa a restrição se nada satisfizer."""
    points = curve(scores, labels)
    feasible = [p for p in points if p.precision >= min_precision and p.tp > 0]
    pool = feasible or [p for p in points if p.tp > 0] or points
    best = max(pool, key=lambda p: (p.f1, p.precision))
    log.info(
        "calibrate.choose",
        threshold=round(best.threshold, 4),
        f1=round(best.f1, 3),
        constrained=bool(feasible),
    )
    return best


async def _score_all(items: list[dataset.GoldenItem]) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    labels: list[bool] = []
    for item in items:
        result = await retrieve(item.question)
        scores.append(result.top_score)
        labels.append(not item.answerable)
    return scores, labels


def _kfold_indices(n: int, folds: int, seed: int) -> list[list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::folds] for i in range(folds)]


async def calibrate(folds: int, min_precision: float, seed: int, out: Path) -> None:
    items = dataset.load(Path("eval/golden.jsonl"), Path("data/chunks.jsonl"))
    unanswerable = sum(1 for i in items if not i.answerable)
    if unanswerable < MIN_UNANSWERABLE:
        log.warning("calibrate.few_unanswerable", n=unanswerable, min=MIN_UNANSWERABLE)

    scores, labels = await _score_all(items)
    n = len(items)

    held_out: list[OperatingPoint] = []
    fold_thresholds: list[float] = []
    for k, test_idx in enumerate(_kfold_indices(n, folds, seed)):
        train = [i for i in range(n) if i not in set(test_idx)]
        point = choose_threshold([scores[i] for i in train], [labels[i] for i in train],
                                 min_precision)
        fold_thresholds.append(point.threshold)
        oof = evaluate_point([scores[i] for i in test_idx], [labels[i] for i in test_idx],
                             point.threshold)
        held_out.append(oof)
        log.info("calibrate.fold", fold=k, threshold=round(point.threshold, 4),
                 oof_precision=round(oof.precision, 3), oof_recall=round(oof.recall, 3))

    chosen = float(statistics.median(fold_thresholds))
    spread = float(np.std(fold_thresholds))
    overall = evaluate_point(scores, labels, chosen)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "chosen_threshold": round(chosen, 4),
        "fold_spread_std": round(spread, 4),
        "fold_thresholds": [round(t, 4) for t in fold_thresholds],
        "min_precision_constraint": min_precision,
        "unanswerable_items": unanswerable,
        "total_items": n,
        "in_sample": overall.__dict__,
        "oof_mean_precision": round(statistics.mean(p.precision for p in held_out), 4),
        "oof_mean_recall": round(statistics.mean(p.recall for p in held_out), 4),
        "curve": [p.__dict__ for p in curve(scores, labels)],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env = [
        "# Gerado por python -m docsage.calibrate — não editar à mão",
        f"# {payload['timestamp']} | folds={folds} seed={seed}",
        f"# OOF precisão={payload['oof_mean_precision']:.3f} "
        f"recall={payload['oof_mean_recall']:.3f} | dispersão σ={spread:.4f}",
        "# σ alto (> 0.05) significa limiar instável: colete mais itens não-respondíveis.",
        f"ABSTAIN_THRESHOLD={chosen:.4f}",
    ]
    out.write_text("\n".join(env) + "\n", encoding="utf-8")
    log.info("calibrate.done", threshold=round(chosen, 4), oof_recall=payload["oof_mean_recall"])

    if spread > 0.05:
        log.warning("calibrate.unstable", spread=round(spread, 4))


@app.command()
def main(
    folds: int = typer.Option(5, min=2),
    min_precision: float = typer.Option(0.60, min=0.1, max=1.0),
    seed: int = typer.Option(1337),
    out: Path = typer.Option(Path("configs/calibrated.env")),
) -> None:
    asyncio.run(calibrate(folds, min_precision, seed, out))


if __name__ == "__main__":
    app()
