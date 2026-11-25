from docsage.eval.gate import allowed_drop, check


def test_no_regression():
    base = {"metrics": {"faithfulness": 0.90}, "noise_std": {}}
    cur = {"metrics": {"faithfulness": 0.89}}
    assert check(base, cur, 0.03) == []


def test_regression_detected():
    base = {"metrics": {"faithfulness": 0.90}, "noise_std": {}}
    cur = {"metrics": {"faithfulness": 0.80}}
    failures = check(base, cur, 0.03)
    assert failures and failures[0]["metric"] == "faithfulness"


def test_noise_widens_tolerance():
    assert allowed_drop("faithfulness", 0.03, {"faithfulness": 0.02}) == 0.04
    assert allowed_drop("mrr", 0.03, {}) == 0.03


def test_missing_metric_fails():
    failures = check({"metrics": {"a": 1.0}, "noise_std": {}}, {"metrics": {}}, 0.03)
    assert failures[0]["reason"] == "ausente na execução atual"


def test_non_numeric_metrics_are_skipped():
    """error_ids é lista: o gate não pode tentar subtrair e estourar."""
    base = {"metrics": {"recall_at_k": 0.5, "error_ids": []}, "noise_std": {}}
    cur = {"metrics": {"recall_at_k": 0.5, "error_ids": ["q01"]}}
    assert check(base, cur, 0.03) == []


def test_non_numeric_metric_does_not_hide_regression():
    base = {"metrics": {"recall_at_k": 0.5, "error_ids": []}, "noise_std": {}}
    cur = {"metrics": {"recall_at_k": 0.4, "error_ids": []}}
    failures = check(base, cur, 0.03)
    assert [f["metric"] for f in failures] == ["recall_at_k"]
