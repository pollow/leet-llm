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
    act = str(_D["activation"].item()) if "activation" in _D else "gelu"
    return TransformerConfig(
        d_model=int(_D["d_model"]),
        n_heads=int(_D["n_heads"]),
        n_enc_layers=int(_D["n_enc_layers"]),
        n_dec_layers=int(_D["n_dec_layers"]),
        d_ff=int(_D["d_ff"]),
        vocab_size=int(_D["vocab_size"]),
        max_pos=int(_D["max_pos"]),
        scale_embedding=bool(_D["scale_embedding"]),
        pad_id=int(_D["pad_id"]),
        eos_id=int(_D["eos_id"]),
        decoder_start_id=int(_D["decoder_start_id"]),
        activation=act,
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


def test_load_marian_shapes():
    """load_marian should map HF keys to MarianParams with correct shapes, no transpose."""
    p = _params()
    cfg = _cfg()
    V, d = cfg.vocab_size, cfg.d_model
    P = cfg.max_pos
    assert p.enc_embed.shape == (V, d)
    assert p.dec_embed.shape == (V, d)
    assert p.enc_pos.shape == (P, d)
    assert p.dec_pos.shape == (P, d)
    assert p.lm_head.shape == (V, d)
    assert p.final_logits_bias.shape == (V,)


def test_load_marian_values_no_transpose():
    """Values must map directly without transpose; HF stores (out,in)."""
    p = _params()
    np.testing.assert_array_equal(p.enc_embed, _D["model.encoder.embed_tokens.weight"])
    np.testing.assert_array_equal(p.dec_embed, _D["model.decoder.embed_tokens.weight"])
    np.testing.assert_array_equal(p.enc_pos, _D["model.encoder.embed_positions.weight"])
    np.testing.assert_array_equal(p.dec_pos, _D["model.decoder.embed_positions.weight"])
    np.testing.assert_array_equal(p.lm_head, _D["lm_head.weight"])
    e0 = p.enc_layers[0]
    np.testing.assert_array_equal(
        e0.attn.Wq, _D["model.encoder.layers.0.self_attn.q_proj.weight"]
    )
    np.testing.assert_array_equal(
        e0.attn.Wk, _D["model.encoder.layers.0.self_attn.k_proj.weight"]
    )
    np.testing.assert_array_equal(
        e0.attn.Wv, _D["model.encoder.layers.0.self_attn.v_proj.weight"]
    )
    np.testing.assert_array_equal(
        e0.attn.Wo, _D["model.encoder.layers.0.self_attn.out_proj.weight"]
    )
    d0 = p.dec_layers[0]
    np.testing.assert_array_equal(
        d0.cross_attn.Wq, _D["model.decoder.layers.0.encoder_attn.q_proj.weight"]
    )
    np.testing.assert_array_equal(
        d0.cross_attn.Wk, _D["model.decoder.layers.0.encoder_attn.k_proj.weight"]
    )


def test_load_marian_final_bias_reshape():
    """final_logits_bias in HF is (1,V), in MarianParams must be (V,) reshaped."""
    p = _params()
    raw = _D["final_logits_bias"]
    assert p.final_logits_bias.ndim == 1
    assert p.final_logits_bias.shape[0] == int(_D["vocab_size"])
    np.testing.assert_array_equal(p.final_logits_bias, raw.reshape(-1))


def test_load_marian_tied_embeddings():
    """Marian ties shared embedding to lm_head; fixture stores separate keys but values equal."""
    p = _params()
    np.testing.assert_array_equal(
        _D["model.shared.weight"], _D["model.encoder.embed_tokens.weight"]
    )
    np.testing.assert_array_equal(_D["model.shared.weight"], _D["lm_head.weight"])
    np.testing.assert_array_equal(p.enc_embed, _D["model.shared.weight"])
    np.testing.assert_array_equal(p.lm_head, _D["model.shared.weight"])


def test_load_marian_layer_counts_and_norms():
    """Enc/dec layer lists length must match cfg, and norm gamma/beta map to correct HF keys."""
    p = _params()
    cfg = _cfg()
    assert len(p.enc_layers) == cfg.n_enc_layers
    assert len(p.dec_layers) == cfg.n_dec_layers
    e0 = p.enc_layers[0]
    np.testing.assert_array_equal(
        e0.norm1_gamma, _D["model.encoder.layers.0.self_attn_layer_norm.weight"]
    )
    np.testing.assert_array_equal(
        e0.norm1_beta, _D["model.encoder.layers.0.self_attn_layer_norm.bias"]
    )
    np.testing.assert_array_equal(
        e0.norm2_gamma, _D["model.encoder.layers.0.final_layer_norm.weight"]
    )
    np.testing.assert_array_equal(
        e0.norm2_beta, _D["model.encoder.layers.0.final_layer_norm.bias"]
    )
    d0 = p.dec_layers[0]
    np.testing.assert_array_equal(
        d0.norm1_gamma, _D["model.decoder.layers.0.self_attn_layer_norm.weight"]
    )
    np.testing.assert_array_equal(
        d0.norm2_gamma, _D["model.decoder.layers.0.encoder_attn_layer_norm.weight"]
    )
    np.testing.assert_array_equal(
        d0.norm3_gamma, _D["model.decoder.layers.0.final_layer_norm.weight"]
    )


def test_load_marian_attn_bias_present():
    """All four linear projections in every attn must have bias per README post-norm spec."""
    p = _params()
    for el in p.enc_layers:
        assert el.attn.bq is not None and el.attn.bk is not None
        assert el.attn.bv is not None and el.attn.bo is not None
    for dl in p.dec_layers:
        for attn in (dl.self_attn, dl.cross_attn):
            assert attn.bq is not None and attn.bk is not None
            assert attn.bv is not None and attn.bo is not None


def test_load_marian_activation_field():
    """TransformerConfig should carry activation to choose gelu vs silu; real Marian uses swish."""
    cfg = _cfg()
    # tiny fixture uses gelu per gen script; real_ref uses swish – test both paths exist
    assert cfg.activation in ("gelu", "swish", "silu")
    # ensure load_marian does not depend on activation (weights same), only ffn behavior changes
    p = _params()
    assert hasattr(p, "enc_embed")
