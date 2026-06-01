from leet_llm.grader import load

_m = load(__file__)
bpe_decode = _m.bpe_decode

TOKENS = [" ", "a", "b", "ab", " ab"]


def test_decode_concatenates_pieces():
    assert bpe_decode([3, 4], TOKENS) == "ab ab"


def test_pieces_carry_their_spacing():
    assert bpe_decode([1, 4], TOKENS) == "a ab"


def test_empty():
    assert bpe_decode([], TOKENS) == ""


def test_single_token():
    assert bpe_decode([3], TOKENS) == "ab"
