import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def context_precision(context: list[str], used: list[str]) -> float:
    if not context:
        return 0.0
    return len(set(context) & set(used)) / len(context)


def macro(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
