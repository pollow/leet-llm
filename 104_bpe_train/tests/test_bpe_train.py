from leet_llm.grader import load

_m = load(__file__)
bpe_train = _m.bpe_train


def test_first_merges_are_deterministic():
    tokens, scores = bpe_train("ab ab ab cab", vocab_size=6)
    # base = sorted unique chars; space sorts before letters
    assert tokens[:4] == [" ", "a", "b", "c"]
    # (a,b) is most frequent -> "ab"; then ("ab"," ") -> "ab "
    assert tokens[4] == "ab"
    assert tokens[5] == "ab "
    assert scores[4] == -1.0
    assert scores[5] == -2.0


def test_base_then_descending_scores():
    text = "the cat sat on the mat the cat ran on the mat"
    base = sorted(set(text))
    tokens, scores = bpe_train(text, vocab_size=30)
    assert tokens[: len(base)] == base
    assert len(tokens) == 30
    assert scores[: len(base)] == [0.0] * len(base)
    assert scores[len(base) :] == [float(-(i + 1)) for i in range(30 - len(base))]


def test_leading_space_pieces_emerge():
    tokens, _ = bpe_train("the the the", vocab_size=12)
    assert any(t.startswith(" ") and len(t) > 1 for t in tokens)


def test_no_merges_below_base_size():
    tokens, scores = bpe_train("abc", vocab_size=2)
    assert tokens == ["a", "b", "c"]  # already 3 base chars, no room to merge
    assert scores == [0.0, 0.0, 0.0]
