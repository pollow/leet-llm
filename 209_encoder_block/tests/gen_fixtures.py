"""209 — generate frozen golden fixtures for the encoder block (post-norm, bidirectional).

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 209_encoder_block/tests/gen_fixtures.py

Oracle: float64 torch composition  a = LN1(x + SelfAttn(x)); y = LN2(a + FFN(a)).
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


def _mha(x, W, h, mask=None):
    Wq, Wk, Wv, Wo = W
    q, k, v = (_split(F.linear(x, w), h) for w in (Wq, Wk, Wv))
    am = None if mask is None else torch.from_numpy(np.where(mask, -np.inf, 0.0))
    return F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)), Wo)


def _ffn(x, W, act="gelu"):
    W1, b1, W2, b2 = W
    fn = {"gelu": F.gelu, "silu": F.silu, "swish": F.silu, "relu": F.relu}[act]
    return F.linear(fn(F.linear(x, W1, b1)), W2, b2)


def _ln(x, g, b):
    return F.layer_norm(x, (x.shape[-1],), g, b, eps=1e-5)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    B, L, d, d_ff, n_heads = 2, 6, 16, 32, 4
    base = {
        "Wq": rng.standard_normal((d, d)),
        "Wk": rng.standard_normal((d, d)),
        "Wv": rng.standard_normal((d, d)),
        "Wo": rng.standard_normal((d, d)),
        "W1": rng.standard_normal((d_ff, d)),
        "b1": rng.standard_normal(d_ff),
        "W2": rng.standard_normal((d, d_ff)),
        "b2": rng.standard_normal(d),
        "n1g": rng.standard_normal(d),
        "n1b": rng.standard_normal(d),
        "n2g": rng.standard_normal(d),
        "n2b": rng.standard_normal(d),
        "n_heads": np.array(n_heads),
    }
    for act in ("gelu", "silu"):
        arr = base.copy()
        arr["x"] = rng.standard_normal((B, L, d))
        arr["activation"] = np.array(act)
        t = {
            k: torch.from_numpy(v)
            for k, v in arr.items()
            if isinstance(v, np.ndarray) and v.ndim > 0
        }
        attn_W = (t["Wq"], t["Wk"], t["Wv"], t["Wo"])
        ffn_W = (t["W1"], t["b1"], t["W2"], t["b2"])
        a = _ln(t["x"] + _mha(t["x"], attn_W, n_heads), t["n1g"], t["n1b"])
        y = _ln(a + _ffn(a, ffn_W, act), t["n2g"], t["n2b"])
        arr["out"] = y.numpy()
        name = "basic" if act == "gelu" else "silu_basic"
        np.savez(FIX / f"{name}.npz", **arr)
        print(
            f"  wrote {name}.npz  x{arr['x'].shape} -> out{arr['out'].shape} act={act}"
        )


if __name__ == "__main__":
    main()
