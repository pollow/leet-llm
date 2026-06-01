import pathlib

import numpy as np
import pytest

from leet_llm import AttnParams, FFNParams
from leet_llm.grader import load

_m = load(__file__)
gpt_block = _m.gpt_block
GPTBlockParams = _m.GPTBlockParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


def _causal(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _params(d):
    return GPTBlockParams(
        attn=AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"]),
        ffn=FFNParams(W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"]),
        norm1_gamma=d["n1g"], norm1_beta=d["n1b"],
        norm2_gamma=d["n2g"], norm2_beta=d["n2b"],
    )


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen golden from a float64 torch GPT block (pre-norm, causal self-attn)."""
    d = np.load(path)
    L = d["x"].shape[-2]
    out = gpt_block(d["x"], _params(d), int(d["n_heads"]), mask=_causal(L))
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_mask_defaults_to_causal():
    d = np.load(_FIXTURES[0])
    L = d["x"].shape[-2]
    explicit = gpt_block(d["x"], _params(d), int(d["n_heads"]), mask=_causal(L))
    default = gpt_block(d["x"], _params(d), int(d["n_heads"]))
    np.testing.assert_allclose(default, explicit, atol=1e-12)
