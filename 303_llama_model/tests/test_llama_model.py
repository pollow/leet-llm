import pathlib

import numpy as np

from leet_llm.grader import load

_m = load(__file__)
LlamaConfig = _m.LlamaConfig
load_llama = _m.load_llama
llama_forward = _m.llama_forward

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_llama.npz")


def _cfg():
    return LlamaConfig(dim=int(_D["dim"]), n_layers=int(_D["n_layers"]),
                       n_heads=int(_D["n_heads"]), n_kv_heads=int(_D["n_kv_heads"]),
                       vocab_size=int(_D["vocab_size"]), max_seq_len=int(_D["max_seq_len"]),
                       norm_eps=float(_D["norm_eps"]), rope_base=float(_D["rope_base"]))


def _params():
    return load_llama({k: _D[k] for k in _D.files}, _cfg())


def test_logits_match_oracle():
    out = llama_forward(_D["input_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["logits"], rtol=1e-9, atol=1e-9)


def test_logits_shape():
    out = llama_forward(_D["input_ids"], _params(), _cfg())
    assert out.shape == (1, _D["input_ids"].shape[1], int(_D["vocab_size"]))


def test_causal_ignores_future():
    p, cfg = _params(), _cfg()
    base = llama_forward(_D["input_ids"], p, cfg)
    ids2 = _D["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_D["vocab_size"])
    pert = llama_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


import sys
import pytest

_SIB = pathlib.Path(__file__).resolve().parents[3] / "llama3.np"
_LOCAL = _SIB / "stories15M.model.npz"
_LINK = pathlib.Path(__file__).parent.parent / "stories15M.model.npz"
_WEIGHTS = _LOCAL if _LOCAL.exists() else _LINK


@pytest.mark.skipif(not _WEIGHTS.exists(),
                    reason="run 303_llama_model/download.sh to fetch stories15M")
def test_real_stories15m_matches_llama3np():
    sys.path.insert(0, str(_SIB))
    from config import ModelArgs
    from llama3 import Llama
    args = ModelArgs()
    ref = Llama(str(_WEIGHTS), args)
    ids = np.array([[1, 306, 505, 263]])
    ref_logits = ref(ids, start_pos=0)[:, -1, :]
    cfg = LlamaConfig(dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
                      n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads,
                      vocab_size=args.vocab_size, max_seq_len=args.max_seq_len,
                      norm_eps=args.norm_eps)
    W = dict(np.load(str(_WEIGHTS)))
    out = llama_forward(ids, load_llama(W, cfg), cfg)
    np.testing.assert_allclose(out[:, -1, :], ref_logits, rtol=1e-6, atol=1e-5)
