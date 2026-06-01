import json

from leet_llm.grader import load

_m = load(__file__)
save_tokenizer = _m.save_tokenizer
load_tokenizer = _m.load_tokenizer


def test_round_trip(tmp_path):
    tokens = [" ", "a", "b", "ab", " ab"]
    scores = [0.0, 0.0, 0.0, -1.0, -2.0]
    p = tmp_path / "tok.json"
    save_tokenizer(tokens, scores, str(p))
    t2, s2 = load_tokenizer(str(p))
    assert t2 == tokens
    assert s2 == scores


def test_file_is_the_expected_json_shape(tmp_path):
    p = tmp_path / "tok.json"
    save_tokenizer(["a", "b"], [0.0, -1.0], str(p))
    with open(p, encoding="utf-8") as f:
        model = json.load(f)
    assert set(model.keys()) == {"tokens", "scores"}
    assert model["tokens"] == ["a", "b"]


def test_preserves_unicode_pieces(tmp_path):
    p = tmp_path / "tok.json"
    save_tokenizer(["é", "日", " 🚀"], [0.0, -1.0, -2.0], str(p))
    tokens, _ = load_tokenizer(str(p))
    assert tokens == ["é", "日", " 🚀"]
