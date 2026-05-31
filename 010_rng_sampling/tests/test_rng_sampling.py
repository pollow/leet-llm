import numpy as np

from leet_llm.grader import load

sample_categorical = load(__file__).sample_categorical


def test_output_shape_and_dtype():
    probs = np.full((5, 3), 1 / 3)
    out = sample_categorical(probs, np.random.default_rng(0))
    assert out.shape == (5,)
    assert np.issubdtype(out.dtype, np.integer)


def test_degenerate_distributions_are_deterministic():
    # one-hot rows: the sure index must be returned regardless of rng
    probs = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    out = sample_categorical(probs, np.random.default_rng(123))
    np.testing.assert_array_equal(out, [1, 3])


def test_reproducible_with_same_seed():
    rng_probs = np.random.default_rng(7).dirichlet(np.ones(5), size=8)
    a = sample_categorical(rng_probs, np.random.default_rng(42))
    b = sample_categorical(rng_probs, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_empirical_frequencies_match_probs():
    probs = np.array([0.1, 0.6, 0.3])
    draws = sample_categorical(
        np.tile(probs, (40000, 1)), np.random.default_rng(0)
    )
    freq = np.bincount(draws, minlength=3) / draws.size
    np.testing.assert_allclose(freq, probs, atol=0.02)


def test_indices_in_range():
    probs = np.random.default_rng(1).dirichlet(np.ones(6), size=(3, 4))
    out = sample_categorical(probs, np.random.default_rng(2))
    assert out.shape == (3, 4)
    assert out.min() >= 0 and out.max() < 6
