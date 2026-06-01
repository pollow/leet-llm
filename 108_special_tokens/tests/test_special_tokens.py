from leet_llm.grader import load

_m = load(__file__)
add_special_tokens = _m.add_special_tokens
strip_special_tokens = _m.strip_special_tokens


def test_add_bos_and_eos():
    assert add_special_tokens([5, 6], bos_id=1, eos_id=2) == [1, 5, 6, 2]


def test_add_only_bos():
    assert add_special_tokens([5, 6], bos_id=1) == [1, 5, 6]


def test_add_nothing():
    assert add_special_tokens([5, 6]) == [5, 6]


def test_strip_edges_and_interior():
    assert strip_special_tokens([1, 5, 0, 6, 2], [0, 1, 2]) == [5, 6]


def test_round_trip():
    ids = [5, 6, 7]
    wrapped = add_special_tokens(ids, bos_id=1, eos_id=2)
    assert strip_special_tokens(wrapped, {1, 2}) == ids
