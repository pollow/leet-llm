import numpy as np

from leet_llm.grader import load

_m = load(__file__)
embedding = _m.embedding


def test_explicit_lookup():
    # Hand-written goldens: output rows are exactly the indexed table rows.
    table = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    ids = np.array([[0, 2], [1, 3]])
    expected = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
        ]
    )
    np.testing.assert_allclose(embedding(ids, table), expected)


def test_shape_and_dtype():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((10, 4))
    ids = rng.integers(0, 10, size=(3, 5))
    out = embedding(ids, table)
    assert out.shape == (3, 5, 4)
    assert out.dtype == table.dtype


def test_1d_ids():
    table = np.arange(15, dtype=float).reshape(5, 3)
    ids = np.array([4, 0, 2])
    out = embedding(ids, table)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, table[[4, 0, 2]])
