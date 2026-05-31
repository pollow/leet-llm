import numpy as np

from leet_llm.grader import load

softmax = load(__file__).softmax


def _reference(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def test_sums_to_one_and_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    out = softmax(x, axis=-1)
    np.testing.assert_allclose(out.sum(-1), 1.0)
    np.testing.assert_allclose(out, _reference(x))


def test_stable_for_large_inputs():
    x = np.array([1000.0, 1001.0, 1002.0])
    out = softmax(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 1.0)
    np.testing.assert_allclose(out, _reference(x))


def test_shift_invariance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5))
    np.testing.assert_allclose(softmax(x), softmax(x + 100.0))


def test_respects_axis():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 4))
    out = softmax(x, axis=0)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out.sum(0), 1.0)
    np.testing.assert_allclose(out, _reference(x, axis=0))
