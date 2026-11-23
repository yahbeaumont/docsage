import pytest

from docsage.eval.ab import bh_adjust, bootstrap_ci, decide, evaluate_metric, mde


def test_clear_improvement_passes():
    a = [0.4] * 100
    b = [0.9] * 100
    v = evaluate_metric("m", a, b, 0.03, 500, 1)
    assert v.decision == "pass" and v.delta == pytest.approx(0.5)


def test_clear_regression_fails():
    a = [0.9] * 100
    b = [0.4] * 100
    assert evaluate_metric("m", a, b, 0.03, 500, 1).decision == "fail"


def test_identical_arms_inconclusive_or_pass():
    a = [0.7, 0.8, 0.6, 0.75] * 25
    v = evaluate_metric("m", a, list(a), 0.03, 500, 1)
    assert v.decision in ("pass", "inconclusive")


def test_decide_boundaries():
    assert decide(0.0, 0.1, 0.03) == "pass"
    assert decide(-0.2, -0.1, 0.03) == "fail"
    assert decide(-0.1, 0.1, 0.03) == "inconclusive"


def test_bootstrap_ci_symmetric_on_noise():
    deltas, sigma = bootstrap_ci([0.5] * 50, [0.5] * 50, 200, 1)
    assert sigma == 0.0 and all(abs(d) < 1e-9 for d in deltas)


def test_bh_monotone():
    adjusted = bh_adjust([0.001, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert adjusted[0] >= 0.001


def test_mde_grows_as_n_shrinks():
    assert mde(0.1, 20) > mde(0.1, 200)


def test_mde_infinite_for_tiny_n():
    assert mde(0.1, 1) == float("inf")
