import numpy as np

from leet_llm.grader import load

_m = load(__file__)
build_batch = _m.build_batch


def test_keys_and_shared_shape():
    b = build_batch([[1, 2, 3], [4]], pad_id=0)
    assert set(b) == {"input_ids", "pad_mask", "position_ids"}
    for v in b.values():
        assert v.shape == (2, 3)


def test_contents_line_up():
    b = build_batch([[1, 2, 3], [4]])
    np.testing.assert_array_equal(b["input_ids"], [[1, 2, 3], [4, 0, 0]])
    np.testing.assert_array_equal(b["pad_mask"], [[1, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(b["position_ids"], [[0, 1, 2], [0, 0, 0]])


def test_maxlen_and_pad_id():
    b = build_batch([[1, 2, 3, 4], [5]], pad_id=9, max_len=2)
    np.testing.assert_array_equal(b["input_ids"], [[1, 2], [5, 9]])
    np.testing.assert_array_equal(b["pad_mask"], [[1, 1], [1, 0]])
    np.testing.assert_array_equal(b["position_ids"], [[0, 1], [0, 0]])
