import numpy as np

from leet_llm.grader import load

_m = load(__file__)
pad_batch = _m.pad_batch
padding_mask = _m.padding_mask


def test_pad_shape_and_values():
    out = pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, [[1, 2, 3], [4, 0, 0]])


def test_pad_custom_id_and_truncation():
    out = pad_batch([[1, 2, 3, 4], [5]], pad_id=9, max_len=2)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, [[1, 2], [5, 9]])


def test_mask_marks_real_positions():
    np.testing.assert_array_equal(padding_mask([[1, 2, 3], [4]]), [[1, 1, 1], [1, 0, 0]])


def test_mask_with_maxlen():
    np.testing.assert_array_equal(padding_mask([[1, 2, 3, 4], [5]], max_len=2), [[1, 1], [1, 0]])


def test_dtype_is_int():
    assert pad_batch([[1]]).dtype == np.int64
    assert padding_mask([[1]]).dtype == np.int64
