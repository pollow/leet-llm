import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
translate = _m.translate

# build params/cfg via task 301 through the facade
from leet_llm import TransformerConfig, load_marian

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_greedy.npz")


def _cfg():
    return TransformerConfig(
        d_model=int(_D["d_model"]), n_heads=int(_D["n_heads"]),
        n_enc_layers=int(_D["n_enc_layers"]), n_dec_layers=int(_D["n_dec_layers"]),
        d_ff=int(_D["d_ff"]), vocab_size=int(_D["vocab_size"]), max_pos=int(_D["max_pos"]),
        scale_embedding=bool(_D["scale_embedding"]), pad_id=int(_D["pad_id"]),
        eos_id=int(_D["eos_id"]), decoder_start_id=int(_D["decoder_start_id"]))


def _params():
    return load_marian({k: _D[k] for k in _D.files}, _cfg())


def test_greedy_matches_hf_generate():
    # forced_eos=None ⇒ HF greedy is pure argmax; the tiny model never emits eos within
    # the budget, so translate with a matching budget reproduces the full sequence.
    cfg = _cfg()
    expected = _D["expected_ids"][0].tolist()       # [decoder_start, ...11 greedy tokens]
    out = translate(_D["src_ids"], _params(), cfg, max_new_tokens=len(expected) - 1)
    assert out == expected


def test_starts_with_decoder_start():
    cfg = _cfg()
    out = translate(_D["src_ids"], _params(), cfg, max_new_tokens=4)
    assert out[0] == cfg.decoder_start_id


def test_stops_at_eos():
    # Deterministically exercise the stop: set eos to the first generated token, so
    # translate must halt right after emitting it. Uses the same proven logits.
    import dataclasses
    expected = _D["expected_ids"][0].tolist()
    cfg = dataclasses.replace(_cfg(), eos_id=int(expected[1]))
    out = translate(_D["src_ids"], _params(), cfg, max_new_tokens=12)
    assert out == [cfg.decoder_start_id, int(expected[1])]
    assert out[-1] == cfg.eos_id


_REAL_W = pathlib.Path(__file__).parent.parent / "opus_mt_en_zh.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(not (_REAL_W.exists() and _REAL_REF.exists()),
                    reason="run 302_translate/download.sh to fetch real opus-mt-en-zh weights")
def test_real_en_zh_matches_hf_greedy():
    R = np.load(_REAL_REF)
    cfg = TransformerConfig(
        d_model=int(R["d_model"]), n_heads=int(R["n_heads"]),
        n_enc_layers=int(R["n_enc_layers"]), n_dec_layers=int(R["n_dec_layers"]),
        d_ff=int(R["d_ff"]), vocab_size=int(R["vocab_size"]), max_pos=int(R["max_pos"]),
        scale_embedding=bool(R["scale_embedding"]), pad_id=int(R["pad_id"]),
        eos_id=int(R["eos_id"]), decoder_start_id=int(R["decoder_start_id"]))
    _W = np.load(_REAL_W)
    params = load_marian({k: _W[k] for k in _W.files}, cfg)
    out = translate(R["src_ids"], params, cfg, max_new_tokens=64)
    expected = R["expected_ids"][0].tolist()
    assert out[: len(expected)] == expected
