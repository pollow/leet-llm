from leet_llm.grader import load

_m = load(__file__)
tiktoken_encode = _m.tiktoken_encode
tiktoken_decode = _m.tiktoken_decode

# Tiny rank table: every single byte we use, plus two merges.
# rank value is both the merge priority (lower = earlier) and the token id.
RANKS = {b"a": 0, b"b": 1, b" ": 2, b"ab": 3, b" a": 4}


def test_chunk_present_as_whole_piece():
    assert tiktoken_encode("ab", RANKS) == [3]


def test_byte_pair_merge_within_chunk():
    # "abb": a,b,b -> merge (a,b)=rank3 -> [ab, b] -> [3, 1]
    assert tiktoken_encode("abb", RANKS) == [3, 1]


def test_no_applicable_merge():
    assert tiktoken_encode("ba", RANKS) == [1, 0]


def test_regex_split_then_lowest_rank_merge():
    # "ab ab" -> ["ab", " ab"]; in " ab", (a,b)=rank3 beats ( ,a)=rank4
    assert tiktoken_encode("ab ab", RANKS) == [3, 2, 3]


def test_round_trip():
    for s in ["ab", "abb", "ba", "ab ab"]:
        assert tiktoken_decode(tiktoken_encode(s, RANKS), RANKS) == s
