import numpy as np

from leet_llm.grader import load

_m = load(__file__)
group_last_axis = _m.group_last_axis
ungroup_last_axis = _m.ungroup_last_axis


def test_group_shape():
    x = np.random.default_rng(0).standard_normal((2, 5, 12))
    assert group_last_axis(x, 4).shape == (2, 4, 5, 3)


def test_group_places_contiguous_blocks():
    # group g must be the slice x[..., g*f:(g+1)*f] of the original last axis.
    x = np.random.default_rng(1).standard_normal((2, 3, 8))
    out = group_last_axis(x, 2)  # f = 4
    np.testing.assert_array_equal(out[:, 0], x[..., 0:4])
    np.testing.assert_array_equal(out[:, 1], x[..., 4:8])


def test_ungroup_shape():
    x = np.zeros((2, 4, 5, 3))
    assert ungroup_last_axis(x).shape == (2, 5, 12)


def test_round_trip_is_identity():
    x = np.random.default_rng(2).standard_normal((3, 7, 16))
    np.testing.assert_array_equal(ungroup_last_axis(group_last_axis(x, 8)), x)


def test_single_group_is_just_a_new_axis():
    x = np.random.default_rng(3).standard_normal((2, 4, 6))
    np.testing.assert_array_equal(group_last_axis(x, 1)[:, 0], x)
