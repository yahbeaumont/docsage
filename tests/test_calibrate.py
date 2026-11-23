from docsage.calibrate import choose_threshold, curve, evaluate_point


def test_point_counts():
    p = evaluate_point([0.1, 0.5, 0.9], [True, False, True], 0.3)
    assert (p.tp, p.fp, p.fn) == (1, 0, 1)
    assert p.precision == 1.0 and p.recall == 0.5


def test_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [True, True, False, False]
    p = choose_threshold(scores, labels)
    assert p.precision == 1.0 and p.recall == 1.0


def test_precision_constraint_respected():
    scores = [0.1, 0.4, 0.45, 0.9]
    labels = [True, False, False, True]
    p = choose_threshold(scores, labels, min_precision=1.0)
    assert p.precision == 1.0


def test_curve_monotonic_recall():
    # A curva é varrida com threshold crescente. Quanto maior o limiar, mais itens
    # abstêm, então recall de não-respondíveis só pode subir (e precisão cair).
    # A asserção original do scaffold pedia a ordem inversa, o que contradiz
    # `abstained = score < threshold`; o teste é que estava errado.
    scores = [0.1, 0.3, 0.6, 0.9]
    labels = [True, True, False, False]
    recalls = [p.recall for p in curve(scores, labels)]
    assert recalls == sorted(recalls)


def test_single_score_no_crash():
    assert len(curve([0.5], [True])) == 1
