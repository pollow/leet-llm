from leet_llm.grader import load

_m = load(__file__)
text_to_byte_ids = _m.text_to_byte_ids
byte_ids_to_text = _m.byte_ids_to_text


def test_ascii_is_one_byte_each():
    assert text_to_byte_ids("abc") == [97, 98, 99]


def test_all_ids_in_byte_range():
    ids = text_to_byte_ids("héllo wörld — 日本語")
    assert all(0 <= b < 256 for b in ids)


def test_multibyte_char_expands():
    assert len(text_to_byte_ids("é")) == 2   # two UTF-8 bytes
    assert len(text_to_byte_ids("日")) == 3   # three UTF-8 bytes


def test_round_trip_unicode():
    text = "héllo wörld — 日本語 🚀"
    assert byte_ids_to_text(text_to_byte_ids(text)) == text


def test_empty():
    assert text_to_byte_ids("") == []
    assert byte_ids_to_text([]) == ""
