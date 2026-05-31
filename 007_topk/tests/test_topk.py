import numpy as np

from leet_llm.grader import load

_m = load(__file__)
argmax = _m.argmax
top_k = _m.top_k


def test_argmax_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4))
    np.testing.assert_array_equal(argmax(x, -1), np.argmax(x, -1))
    np.testing.assert_array_equal(argmax(x, 0), np.argmax(x, 0))


def test_top_k_known_example():
    x = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]])
    values, indices = top_k(x, 2)
    np.testing.assert_array_equal(indices, [[3, 1]])
    np.testing.assert_allclose(values, [[8.0, 5.0]])


def test_top_k_is_descending_and_consistent():
    rng = np.random.default_rng(1)
    x = rng.permutation(2 * 20).astype(float).reshape(2, 20)  # no ties
    values, indices = top_k(x, 5)
    assert values.shape == indices.shape == (2, 5)
    # descending
    assert np.all(np.diff(values, axis=-1) <= 0)
    # indices actually point at those values
    np.testing.assert_allclose(values, np.take_along_axis(x, indices, axis=-1))
    # same multiset as the true 5 largest
    np.testing.assert_allclose(np.sort(values, -1), np.sort(x, -1)[:, -5:])


def test_top_k_full_width():
    rng = np.random.default_rng(2)
    x = rng.permutation(8).astype(float).reshape(1, 8)
    values, _ = top_k(x, 8)
    np.testing.assert_allclose(values, np.sort(x, -1)[:, ::-1])
