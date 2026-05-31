import numpy as np

from leet_llm.grader import load

_m = load(__file__)
masked_fill = _m.masked_fill
triangular_mask = _m.triangular_mask


def test_masked_fill_replaces_marked_positions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4))
    mask = x > 0
    out = masked_fill(x, mask, -1.0)
    assert np.all(out[mask] == -1.0)
    np.testing.assert_array_equal(out[~mask], x[~mask])


def test_masked_fill_does_not_mutate_input():
    x = np.arange(6.0).reshape(2, 3)
    original = x.copy()
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = True
    masked_fill(x, mask, 99.0)
    np.testing.assert_array_equal(x, original)


def test_masked_fill_broadcasts_mask():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 3))
    mask = triangular_mask(3)  # (3, 3) broadcasts over the leading axis
    out = masked_fill(x, mask, 0.0)
    assert out.shape == (2, 3, 3)
    assert np.all(out[:, mask] == 0.0)


def test_triangular_mask_matches_triu():
    m = triangular_mask(4)
    assert m.dtype == bool
    np.testing.assert_array_equal(m, np.triu(np.ones((4, 4), dtype=bool), k=1))


def test_triangular_mask_diagonal_is_false():
    m = triangular_mask(5)
    assert not m.diagonal().any()
    assert not np.tril(m).any()  # nothing on or below the diagonal
