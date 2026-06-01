import numpy as np

from leet_llm.grader import load

_m = load(__file__)
position_ids = _m.position_ids


def test_positions_count_up_with_zero_padding():
    np.testing.assert_array_equal(position_ids([[5, 6, 7], [8]]), [[0, 1, 2], [0, 0, 0]])


def test_truncation():
    np.testing.assert_array_equal(position_ids([[1, 2, 3, 4]], max_len=2), [[0, 1]])


def test_dtype_is_int():
    assert position_ids([[1, 2]]).dtype == np.int64
