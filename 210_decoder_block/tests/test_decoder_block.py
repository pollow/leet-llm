import pathlib

import numpy as np
import pytest

from leet_llm import AttnParams, FFNParams
from leet_llm.grader import load

_m = load(__file__)
decoder_block = _m.decoder_block
DecoderBlockParams = _m.DecoderBlockParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


def _causal(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _params(d):
    return DecoderBlockParams(
        self_attn=AttnParams(Wq=d["sWq"], Wk=d["sWk"], Wv=d["sWv"], Wo=d["sWo"]),
        cross_attn=AttnParams(Wq=d["cWq"], Wk=d["cWk"], Wv=d["cWv"], Wo=d["cWo"]),
        ffn=FFNParams(W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"]),
        norm1_gamma=d["n1g"], norm1_beta=d["n1b"],
        norm2_gamma=d["n2g"], norm2_beta=d["n2b"],
        norm3_gamma=d["n3g"], norm3_beta=d["n3b"],
    )


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen golden from a float64 torch seq2seq decoder block (masked self + cross + FFN)."""
    d = np.load(path)
    L = d["x"].shape[-2]
    out = decoder_block(d["x"], d["enc_out"], _params(d), int(d["n_heads"]), self_mask=_causal(L))
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_shape_preserved():
    d = np.load(_FIXTURES[0])
    L = d["x"].shape[-2]
    out = decoder_block(d["x"], d["enc_out"], _params(d), int(d["n_heads"]), self_mask=_causal(L))
    assert out.shape == d["x"].shape
