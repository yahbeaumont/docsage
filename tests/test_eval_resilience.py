"""Item com erro não pode derrubar a medição nem inflar as médias."""

from docsage.eval.run_eval import summarize


def _item(id_: str, answerable: bool, **extra) -> dict:
    base = {
        "id": id_,
        "category": "factual" if answerable else "unanswerable",
        "answerable": answerable,
        "retrieved": [],
        "recall_at_k": 1.0,
        "ndcg_at_k": 1.0,
        "mrr": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0,
        "faithfulness": 1.0,
        "answer_relevance": 1.0,
        "citation_validity": 1.0,
        "abstained": not answerable,
        "latency_ms": 100,
        "cost_usd": 0.001,
    }
    base.update(extra)
    return base


def test_error_item_is_counted_and_excluded_from_means():
    results = [
        _item("a1", True),
        _item("a2", True, faithfulness=0.0, recall_at_k=0.0),
        {"id": "a3", "category": "factual", "answerable": True, "error": "LLMError: x"},
        _item("u1", False),
    ]
    m = summarize(results)
    assert m["n"] == 3
    assert m["n_errors"] == 1
    assert m["error_ids"] == ["a3"]
    # a3 não entra: a média de recall é (1.0 + 0.0) / 2, não (1.0 + 0.0 + 0) / 3
    assert m["recall_at_k"] == 0.5
    assert m["faithfulness"] == 0.5
    assert m["abstention_accuracy"] == 1.0
    assert m["cost_usd_per_item"] == 0.001


def test_all_items_failing_does_not_divide_by_zero():
    m = summarize([{"id": "a1", "category": "factual", "answerable": True, "error": "boom"}])
    assert m["n"] == 0
    assert m["n_errors"] == 1
    assert m["cost_usd_per_item"] == 0.0
