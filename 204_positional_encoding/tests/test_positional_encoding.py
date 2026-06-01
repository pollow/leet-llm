import numpy as np

from leet_llm.grader import load

_m = load(__file__)
sinusoidal_pe = _m.sinusoidal_pe


def test_shape_and_range():
    pe = sinusoidal_pe(10, 16)
    assert pe.shape == (10, 16)
    assert pe.max() <= 1.0 and pe.min() >= -1.0


def test_first_position_is_zeros_and_ones():
    # pos=0: sin(0)=0 on even indices, cos(0)=1 on odd indices
    pe = sinusoidal_pe(5, 8)
    np.testing.assert_allclose(pe[0], [0, 1, 0, 1, 0, 1, 0, 1], atol=1e-12)


def test_first_freq_columns_are_sin_cos_of_position():
    # i=0 -> 10000**0 = 1 -> column 0 = sin(pos), column 1 = cos(pos)
    L = 7
    pe = sinusoidal_pe(L, 4)
    pos = np.arange(L)
    np.testing.assert_allclose(pe[:, 0], np.sin(pos), atol=1e-12)
    np.testing.assert_allclose(pe[:, 1], np.cos(pos), atol=1e-12)


def test_second_freq_columns():
    # d=4, i=1 -> 10000**(2/4) = 100 -> column 2 = sin(pos/100), column 3 = cos(pos/100)
    L, d = 7, 4
    pe = sinusoidal_pe(L, d)
    pos = np.arange(L)
    np.testing.assert_allclose(pe[:, 2], np.sin(pos / 100.0), atol=1e-12)
    np.testing.assert_allclose(pe[:, 3], np.cos(pos / 100.0), atol=1e-12)
