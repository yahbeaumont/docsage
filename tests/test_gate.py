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
