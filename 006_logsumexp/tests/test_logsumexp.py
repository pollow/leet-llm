import numpy as np

from leet_llm.grader import load

_m = load(__file__)
logsumexp = _m.logsumexp
log_softmax = _m.log_softmax


def test_logsumexp_matches_naive_for_small_inputs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5))
    np.testing.assert_allclose(logsumexp(x, -1), np.log(np.exp(x).sum(-1)))


def test_logsumexp_reduces_axis():
    x = np.zeros((4, 7))
    assert logsumexp(x, -1).shape == (4,)
    assert logsumexp(x, 0).shape == (7,)


def test_logsumexp_stable_for_large_inputs():
    x = np.array([1000.0, 1001.0])
    out = logsumexp(x)
    assert np.isfinite(out)
    # = 1001 + log(1 + e^-1)
    np.testing.assert_allclose(out, 1001.0 + np.log1p(np.exp(-1.0)))


def test_log_softmax_matches_log_of_softmax():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6))
    e = np.exp(x - x.max(-1, keepdims=True))
    ref = np.log(e / e.sum(-1, keepdims=True))
    np.testing.assert_allclose(log_softmax(x, -1), ref, atol=1e-6)


def test_log_softmax_exponentiates_to_a_distribution():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 5))
    np.testing.assert_allclose(np.exp(log_softmax(x, -1)).sum(-1), 1.0)
