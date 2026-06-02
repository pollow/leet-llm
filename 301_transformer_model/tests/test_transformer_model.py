import pathlib

import numpy as np

from leet_llm.grader import load

_m = load(__file__)
TransformerConfig = _m.TransformerConfig
load_marian = _m.load_marian
encoder = _m.encoder
decoder = _m.decoder
transformer_logits = _m.transformer_logits

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_marian.npz")


def _cfg():
    return TransformerConfig(
        d_model=int(_D["d_model"]), n_heads=int(_D["n_heads"]),
        n_enc_layers=int(_D["n_enc_layers"]), n_dec_layers=int(_D["n_dec_layers"]),
        d_ff=int(_D["d_ff"]), vocab_size=int(_D["vocab_size"]), max_pos=int(_D["max_pos"]),
        scale_embedding=bool(_D["scale_embedding"]), pad_id=int(_D["pad_id"]),
        eos_id=int(_D["eos_id"]), decoder_start_id=int(_D["decoder_start_id"]),
    )


def _params():
    weights = {k: _D[k] for k in _D.files}
    return load_marian(weights, _cfg())


def test_encoder_matches_hf():
    out = encoder(_D["src_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["enc_out"], rtol=1e-9, atol=1e-9)


def test_decoder_matches_hf():
    mem = _D["enc_out"]
    out = decoder(_D["tgt_ids"], mem, _params(), _cfg())
    np.testing.assert_allclose(out, _D["dec_out"], rtol=1e-9, atol=1e-9)


def test_logits_match_hf():
    out = transformer_logits(_D["src_ids"], _D["tgt_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["logits"], rtol=1e-9, atol=1e-9)


def test_logits_shape():
    out = transformer_logits(_D["src_ids"], _D["tgt_ids"], _params(), _cfg())
    assert out.shape == (1, _D["tgt_ids"].shape[1], int(_D["vocab_size"]))


def test_causal_decoder_ignores_future():
    # Perturbing a later target token must not change an earlier position's hidden state.
    p, cfg = _params(), _cfg()
    mem = _D["enc_out"]
    base = decoder(_D["tgt_ids"], mem, p, cfg)
    tgt2 = _D["tgt_ids"].copy()
    tgt2[0, -1] = (tgt2[0, -1] + 1) % int(_D["vocab_size"])
    pert = decoder(tgt2, mem, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)
