import pathlib

import numpy as np
import pytest

from leet_llm import AttnParams, SwiGLUParams
from leet_llm.grader import load

_m = load(__file__)
llama_decoder_block = _m.llama_decoder_block
LlamaBlockParams = _m.LlamaBlockParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


def _causal(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _params(d):
    return LlamaBlockParams(
        attn=AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"]),
        ffn=SwiGLUParams(W1=d["gate"], W3=d["up"], W2=d["down"]),
        attn_norm=d["attn_norm"], ffn_norm=d["ffn_norm"],
    )


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen golden from a float64 torch Llama block (RMSNorm + RoPE-GQA + SwiGLU)."""
    d = np.load(path)
    L = d["x"].shape[-2]
    eps = float(d["eps"]) if "eps" in d.files else 1e-5
    out = llama_decoder_block(
        d["x"], _params(d), int(d["n_heads"]), int(d["n_kv_heads"]), d["positions"],
        mask=_causal(L), eps=eps,
    )
    np.testing.assert_allclose(out, d["out"], rtol=1e-9, atol=1e-9)


def test_mask_defaults_to_causal():
    d = np.load(_FIXTURES[0])
    L = d["x"].shape[-2]
    args = (_params(d), int(d["n_heads"]), int(d["n_kv_heads"]), d["positions"])
    explicit = llama_decoder_block(d["x"], *args, mask=_causal(L))
    default = llama_decoder_block(d["x"], *args)
    np.testing.assert_allclose(default, explicit, atol=1e-12)
