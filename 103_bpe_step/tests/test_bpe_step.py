from leet_llm.grader import load

_m = load(__file__)
count_pairs = _m.count_pairs
apply_merge = _m.apply_merge


def test_count_pairs_basic():
    # "banana"-like stream b a n a n a -> ids 1 0 2 0 2 0
    assert count_pairs([1, 0, 2, 0, 2, 0]) == {(1, 0): 1, (0, 2): 2, (2, 0): 2}


def test_count_pairs_empty_and_single():
    assert count_pairs([]) == {}
    assert count_pairs([5]) == {}


def test_apply_merge_replaces_all():
    assert apply_merge([0, 2, 0, 2, 0], (0, 2), 9) == [9, 9, 0]


def test_apply_merge_non_overlapping():
    # the first (0,0) is consumed, leaving a lone 0
    assert apply_merge([0, 0, 0], (0, 0), 7) == [7, 0]


def test_apply_merge_absent_pair_is_identity():
    assert apply_merge([1, 2, 3], (4, 5), 9) == [1, 2, 3]
