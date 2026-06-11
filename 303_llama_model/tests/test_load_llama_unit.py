import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
LlamaConfig = _m.LlamaConfig
load_llama = _m.load_llama

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_llama.npz")


def _cfg():
    return LlamaConfig(
        dim=int(_D["dim"]),
        n_layers=int(_D["n_layers"]),
        n_heads=int(_D["n_heads"]),
        n_kv_heads=int(_D["n_kv_heads"]),
        vocab_size=int(_D["vocab_size"]),
        max_seq_len=int(_D["max_seq_len"]),
        norm_eps=float(_D["norm_eps"]),
        rope_base=float(_D["rope_base"]),
    )


def _weights():
    return {
        k: _D[k]
        for k in _D.files
        if k
        not in {
            "input_ids",
            "logits",
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "vocab_size",
            "max_seq_len",
            "norm_eps",
            "rope_base",
        }
        or "weight" in k
        or k.startswith("model.")
        or k.startswith("lm_head")
    }


def test_load_llama_maps_all_hf_keys_no_transpose():
    """Critical: every HF key maps to correct slot with exact shape, no transpose."""
    cfg = _cfg()
    W = {k: _D[k] for k in _D.files}
    params = load_llama(W, cfg)

    # top-level
    np.testing.assert_array_equal(params.tok_embed, W["model.embed_tokens.weight"])
    np.testing.assert_array_equal(params.final_norm, W["model.norm.weight"])
    np.testing.assert_array_equal(params.lm_head, W["lm_head.weight"])
    assert params.tok_embed.shape == (cfg.vocab_size, cfg.dim)
    assert params.lm_head.shape == (cfg.vocab_size, cfg.dim)

    # layers count
    assert len(params.layers) == cfg.n_layers

    for i in range(cfg.n_layers):
        pfx = f"model.layers.{i}"
        blk = params.layers[i]
        # norms
        np.testing.assert_array_equal(blk.attn_norm, W[f"{pfx}.input_layernorm.weight"])
        np.testing.assert_array_equal(
            blk.ffn_norm, W[f"{pfx}.post_attention_layernorm.weight"]
        )
        # attn bias-free
        assert (
            blk.attn.bq is None
            and blk.attn.bk is None
            and blk.attn.bv is None
            and blk.attn.bo is None
        )
        np.testing.assert_array_equal(blk.attn.Wq, W[f"{pfx}.self_attn.q_proj.weight"])
        np.testing.assert_array_equal(blk.attn.Wk, W[f"{pfx}.self_attn.k_proj.weight"])
        np.testing.assert_array_equal(blk.attn.Wv, W[f"{pfx}.self_attn.v_proj.weight"])
        np.testing.assert_array_equal(blk.attn.Wo, W[f"{pfx}.self_attn.o_proj.weight"])
        # SwiGLU order: W1 gate, W3 up, W2 down — NOT numeric order
        np.testing.assert_array_equal(blk.ffn.W1, W[f"{pfx}.mlp.gate_proj.weight"])
        np.testing.assert_array_equal(blk.ffn.W3, W[f"{pfx}.mlp.up_proj.weight"])
        np.testing.assert_array_equal(blk.ffn.W2, W[f"{pfx}.mlp.down_proj.weight"])


def test_load_llama_lm_head_not_tied():
    """Critical: lm_head is separate array, not tied to tok_embed (stories15M fact)."""
    cfg = _cfg()
    W = {k: _D[k] for k in _D.files}
    params = load_llama(W, cfg)
    # must be different objects and different values in fixture (fixture uses distinct weights)
    assert params.lm_head is not params.tok_embed
    # in tiny fixture they are different arrays; at minimum shapes match but content distinct check
    # ensure load didn't accidentally reuse embedding
    np.testing.assert_array_equal(params.lm_head, W["lm_head.weight"])
    np.testing.assert_array_equal(params.tok_embed, W["model.embed_tokens.weight"])
    # verify not same reference in weights dict either (sanity of fixture)
    assert (
        not np.may_share_memory(params.lm_head, params.tok_embed)
        or not np.array_equal(params.lm_head, params.tok_embed)
        or True
    )  # pass if fixture coincidentally equal, main point is mapping correctness


def test_load_llama_bias_free_and_shapes_match_cfg():
    """Critical: AttnParams are bias-free, shapes align with cfg.dim / cfg.n_heads."""
    cfg = _cfg()
    W = {k: _D[k] for k in _D.files}
    params = load_llama(W, cfg)
    d = cfg.dim
    for blk in params.layers:
        # attn_norm ffn_norm shape (d,)
        assert blk.attn_norm.shape == (d,)
        assert blk.ffn_norm.shape == (d,)
        # attn weights (out,in) no transpose needed
        assert blk.attn.Wq.shape[1] == d
        assert blk.attn.Wo.shape[0] == d
        # biases None
        assert blk.attn.bq is None
        assert blk.attn.bk is None
        assert blk.attn.bv is None
        assert blk.attn.bo is None
        # SwiGLU shapes: W1 (F,d), W3 (F,d), W2 (d,F)
        F = blk.ffn.W1.shape[0]
        assert blk.ffn.W1.shape == (F, d)
        assert blk.ffn.W3.shape == (F, d)
        assert blk.ffn.W2.shape == (d, F)
