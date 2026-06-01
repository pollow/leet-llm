from leet_llm.grader import load

_m = load(__file__)
regex_split = _m.regex_split


def test_word_keeps_leading_space():
    assert regex_split("Hello world") == ["Hello", " world"]


def test_punctuation_is_separate():
    assert regex_split("Hello world!") == ["Hello", " world", "!"]


def test_contraction_splits():
    assert regex_split("don't") == ["don", "'t"]


def test_letters_then_numbers():
    assert regex_split("abc123") == ["abc", "123"]


def test_leading_space_chunk():
    assert regex_split(" hi") == [" hi"]


def test_join_reconstructs_input():
    s = "The cat. don't be 42 sad!"
    assert "".join(regex_split(s)) == s
