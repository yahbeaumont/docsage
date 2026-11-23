from docsage.eval.retrieval_metrics import context_precision, mrr, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert recall_at_k(["a"], ["b"], 3) == 0.0


def test_mrr():
    assert mrr(["a", "b"], ["b"]) == 0.5
    assert mrr(["a"], ["z"]) == 0.0


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_ndcg_penalizes_position():
    assert ndcg_at_k(["x", "a"], ["a"], 2) < 1.0


def test_context_precision():
    assert context_precision(["a", "b"], ["a"]) == 0.5
