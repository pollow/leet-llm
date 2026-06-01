"""210 — generate frozen golden fixtures for the seq2seq decoder block (post-norm).

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 210_decoder_block/tests/gen_fixtures.py

Oracle: float64 torch
  a = LN1(x + MaskedSelfAttn(x));  b = LN2(a + CrossAttn(a, enc_out));  y = LN3(b + FFN(b)).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"


def _split(t, n):
    *lead, L, d = t.shape
    return t.reshape(*lead, L, n, d // n).transpose(-3, -2)


def _merge(t):
    *lead, h, L, dk = t.shape
    return t.transpose(-3, -2).reshape(*lead, L, h * dk)


def _mha(x_q, W, h, x_kv=None, mask=None):
    Wq, Wk, Wv, Wo = W
    xkv = x_q if x_kv is None else x_kv
    q = _split(F.linear(x_q, Wq), h)
    k = _split(F.linear(xkv, Wk), h)
    v = _split(F.linear(xkv, Wv), h)
    am = None if mask is None else torch.from_numpy(np.where(mask, -np.inf, 0.0))
    return F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)), Wo)


def _ffn(x, W):
    W1, b1, W2, b2 = W
    return F.linear(F.gelu(F.linear(x, W1, b1)), W2, b2)


def _ln(x, g, b):
    return F.layer_norm(x, (x.shape[-1],), g, b, eps=1e-5)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    B, L, Lenc, d, d_ff, n_heads = 2, 4, 6, 16, 32, 4

    def W():
        return rng.standard_normal((d, d))

    arr = {
        "x": rng.standard_normal((B, L, d)),
        "enc_out": rng.standard_normal((B, Lenc, d)),
        "sWq": W(), "sWk": W(), "sWv": W(), "sWo": W(),
        "cWq": W(), "cWk": W(), "cWv": W(), "cWo": W(),
        "W1": rng.standard_normal((d_ff, d)), "b1": rng.standard_normal(d_ff),
        "W2": rng.standard_normal((d, d_ff)), "b2": rng.standard_normal(d),
        "n1g": rng.standard_normal(d), "n1b": rng.standard_normal(d),
        "n2g": rng.standard_normal(d), "n2b": rng.standard_normal(d),
        "n3g": rng.standard_normal(d), "n3b": rng.standard_normal(d),
        "n_heads": np.array(n_heads),
    }
    causal = np.triu(np.ones((L, L), dtype=bool), k=1)
    t = {k: torch.from_numpy(v) for k, v in arr.items() if v.ndim > 0}
    sW = (t["sWq"], t["sWk"], t["sWv"], t["sWo"])
    cW = (t["cWq"], t["cWk"], t["cWv"], t["cWo"])
    fW = (t["W1"], t["b1"], t["W2"], t["b2"])
    a = _ln(t["x"] + _mha(t["x"], sW, n_heads, mask=causal), t["n1g"], t["n1b"])
    b = _ln(a + _mha(a, cW, n_heads, x_kv=t["enc_out"]), t["n2g"], t["n2b"])
    y = _ln(b + _ffn(b, fW), t["n3g"], t["n3b"])
    arr["out"] = y.numpy()
    np.savez(FIX / "basic.npz", **arr)
    print(f"  wrote basic.npz  x{arr['x'].shape} enc{arr['enc_out'].shape} -> out{arr['out'].shape}")


if __name__ == "__main__":
    main()
