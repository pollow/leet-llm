"""206 — generate frozen golden fixtures for multi-head attention.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 206_multi_head_attention/tests/gen_fixtures.py

Oracle: explicit q/k/v projections (F.linear, x@W.T) + per-head F.scaled_dot_product_attention
+ output projection, in float64. Head split matches L0 001 group_last_axis: the last axis d
is reshaped to (n_heads, d_k) with the head index outer (contiguous blocks), then moved ahead
of the length axis.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"


def _split(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    *lead, L, d = t.shape
    return t.reshape(*lead, L, n_heads, d // n_heads).transpose(-3, -2)  # (..., H, L, dk)


def _merge(t: torch.Tensor) -> torch.Tensor:
    *lead, H, L, dk = t.shape
    return t.transpose(-3, -2).reshape(*lead, L, H * dk)  # (..., L, d)


def _mha_ref(x_q, Wq, Wk, Wv, Wo, n_heads, x_kv=None, mask=None,
             bq=None, bk=None, bv=None, bo=None):
    xq = torch.from_numpy(x_q)
    xkv = torch.from_numpy(x_q if x_kv is None else x_kv)
    Wq, Wk, Wv, Wo = (torch.from_numpy(w) for w in (Wq, Wk, Wv, Wo))
    tb = lambda b: None if b is None else torch.from_numpy(b)
    qh = _split(F.linear(xq, Wq, tb(bq)), n_heads)
    kh = _split(F.linear(xkv, Wk, tb(bk)), n_heads)
    vh = _split(F.linear(xkv, Wv, tb(bv)), n_heads)
    attn_mask = None if mask is None else torch.from_numpy(np.where(mask, -np.inf, 0.0))
    oh = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
    return F.linear(_merge(oh), Wo, tb(bo)).numpy()


def _causal(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    d = 8

    def W():
        return rng.standard_normal((d, d))

    def Wset():
        return W(), W(), W(), W()

    # name, x_q, n_heads, x_kv, mask
    cases = [
        ("self_nomask", rng.standard_normal((2, 4, d)), 2, None, None),
        ("self_causal", rng.standard_normal((2, 6, d)), 4, None, _causal(6)),
        ("cross", rng.standard_normal((2, 3, d)), 2, rng.standard_normal((2, 5, d)), None),
    ]
    for name, x_q, n_heads, x_kv, mask in cases:
        Wq, Wk, Wv, Wo = Wset()
        out = _mha_ref(x_q, Wq, Wk, Wv, Wo, n_heads, x_kv, mask)
        arrays = {"x_q": x_q, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                  "n_heads": np.array(n_heads), "out": out}
        if x_kv is not None:
            arrays["x_kv"] = x_kv
        if mask is not None:
            arrays["mask"] = mask
        np.savez(FIX / f"{name}.npz", **arrays)
        print(f"  wrote {name}.npz  x_q{x_q.shape} heads={n_heads} -> out{out.shape}")

    # biased self-attention (classic Transformer / opus-mt path)
    Wq, Wk, Wv, Wo = Wset()
    bq, bk, bv, bo = (rng.standard_normal(d) for _ in range(4))
    x_b = rng.standard_normal((2, 4, d))
    out_b = _mha_ref(x_b, Wq, Wk, Wv, Wo, 2, bq=bq, bk=bk, bv=bv, bo=bo)
    np.savez(FIX / "self_biased.npz", x_q=x_b, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
             bq=bq, bk=bk, bv=bv, bo=bo, n_heads=np.array(2), out=out_b)
    print("  wrote self_biased.npz")


if __name__ == "__main__":
    main()
