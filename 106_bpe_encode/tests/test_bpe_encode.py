from leet_llm.grader import load

_m = load(__file__)
bpe_encode = _m.bpe_encode

# A tiny tokenizer hand-built from the "ab ab" worked example in the design doc.
TOKENS = [" ", "a", "b", "ab", " ab"]
SCORES = [0.0, 0.0, 0.0, -1.0, -2.0]


def test_greedy_merge_matches_reference():
    # a b _ a b -> merge "ab"(-1) twice, then " ab"(-2) -> [ab, " ab"]
    assert bpe_encode("ab ab", TOKENS, SCORES) == [3, 4]


def test_highest_score_wins_first():
    # "ab" (score -1) must be merged before " ab" (score -2)
    assert bpe_encode("a ab", TOKENS, SCORES) == [1, 4]


def test_unknown_chars_dropped():
    # 'z' is not in the vocab; the rest still encodes
    assert bpe_encode("abz", TOKENS, SCORES) == [3]


def test_no_applicable_merge():
    assert bpe_encode("ba", TOKENS, SCORES) == [2, 1]


def test_empty_text():
    assert bpe_encode("", TOKENS, SCORES) == []
