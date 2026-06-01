from leet_llm.grader import load

_m = load(__file__)
build_char_vocab = _m.build_char_vocab
char_encode = _m.char_encode
char_decode = _m.char_decode


def test_vocab_is_sorted_unique():
    stoi, itos = build_char_vocab("banana")
    assert itos == ["a", "b", "n"]
    assert stoi == {"a": 0, "b": 1, "n": 2}


def test_encode_maps_each_char():
    stoi, _ = build_char_vocab("banana")
    assert char_encode("nab", stoi) == [2, 0, 1]


def test_round_trip():
    text = "hello, world!"
    stoi, itos = build_char_vocab(text)
    assert char_decode(char_encode(text, stoi), itos) == text


def test_decode_inverts_indices():
    _, itos = build_char_vocab("abc")
    assert char_decode([2, 1, 0], itos) == "cba"
