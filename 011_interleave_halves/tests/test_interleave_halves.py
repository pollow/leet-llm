import numpy as np

from leet_llm.grader import load

_m = load(__file__)
interleave = _m.interleave
deinterleave = _m.deinterleave
split_halves = _m.split_halves
join_halves = _m.join_halves


def test_interleave_explicit():
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    np.testing.assert_array_equal(interleave(a, b), [1, 10, 2, 20, 3, 30])


def test_deinterleave_explicit():
    x = np.array([1, 10, 2, 20, 3, 30])
    evens, odds = deinterleave(x)
    np.testing.assert_array_equal(evens, [1, 2, 3])
    np.testing.assert_array_equal(odds, [10, 20, 30])


def test_split_halves_explicit():
    x = np.array([1, 2, 3, 4, 5, 6])
    front, back = split_halves(x)
    np.testing.assert_array_equal(front, [1, 2, 3])
    np.testing.assert_array_equal(back, [4, 5, 6])


def test_join_halves_explicit():
    np.testing.assert_array_equal(join_halves([1, 2, 3], [4, 5, 6]), [1, 2, 3, 4, 5, 6])


def test_interleave_round_trip_batched():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal((2, 3, 4))
    x = interleave(a, b)
    assert x.shape == (2, 3, 8)
    e, o = deinterleave(x)
    np.testing.assert_allclose(e, a)
    np.testing.assert_allclose(o, b)


def test_halves_round_trip_batched():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 5))
    b = rng.standard_normal((2, 5))
    x = join_halves(a, b)
    assert x.shape == (2, 10)
    f, s = split_halves(x)
    np.testing.assert_allclose(f, a)
    np.testing.assert_allclose(s, b)


def test_interleave_differs_from_join():
    # the two layouts are genuinely different orderings
    a = np.array([1, 2])
    b = np.array([3, 4])
    np.testing.assert_array_equal(interleave(a, b), [1, 3, 2, 4])
    np.testing.assert_array_equal(join_halves(a, b), [1, 2, 3, 4])
