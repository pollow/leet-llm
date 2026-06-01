"""215 — generate frozen golden fixtures for grouped-query attention.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 215_gqa/tests/gen_fixtures.py

Oracle: float64 torch. Q has n_heads heads; K/V have n_kv_heads heads, each repeated
n_heads // n_kv_heads times (repeat_interleave) so kv head i serves query heads
[i*g : (i+1)*g]. Head split matches L0 001 group_last_axis.
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


def _causal(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _gqa_ref(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, mask):
    xt = torch.from_numpy(x)
    Wq, Wk, Wv, Wo = (torch.from_numpy(w) for w in (Wq, Wk, Wv, Wo))
    q = _split(F.linear(xt, Wq), n_heads)
    k = _split(F.linear(xt, Wk), n_kv_heads)
    v = _split(F.linear(xt, Wv), n_kv_heads)
    g = n_heads // n_kv_heads
    k = k.repeat_interleave(g, dim=-3)
    v = v.repeat_interleave(g, dim=-3)
    attn_mask = None if mask is None else torch.from_numpy(np.where(mask, -np.inf, 0.0))
    oh = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return F.linear(_merge(oh), Wo).numpy()


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    d, n_heads = 8, 4
    dk = d // n_heads

    def weights(n_kv):
        return (
            rng.standard_normal((d, d)),          # Wq
            rng.standard_normal((n_kv * dk, d)),  # Wk
            rng.standard_normal((n_kv * dk, d)),  # Wv
            rng.standard_normal((d, d)),          # Wo
        )

    cases = [
        ("gqa_4h_2kv", rng.standard_normal((2, 5, d)), 2, _causal(5)),
        ("mqa_4h_1kv", rng.standard_normal((2, 3, d)), 1, None),
        ("mha_equiv_4h_4kv", rng.standard_normal((2, 4, d)), 4, None),
    ]
    for name, x, n_kv, mask in cases:
        Wq, Wk, Wv, Wo = weights(n_kv)
        out = _gqa_ref(x, Wq, Wk, Wv, Wo, n_heads, n_kv, mask)
        arrays = {"x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                  "n_heads": np.array(n_heads), "n_kv_heads": np.array(n_kv), "out": out}
        if mask is not None:
            arrays["mask"] = mask
        np.savez(FIX / f"{name}.npz", **arrays)
        print(f"  wrote {name}.npz  x{x.shape} heads={n_heads} kv={n_kv} -> out{out.shape}")


if __name__ == "__main__":
    main()
