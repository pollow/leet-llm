"""216 — generate frozen golden fixtures for the Llama decoder block.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 216_llama_decoder_block/tests/gen_fixtures.py

Oracle: float64 torch, pre-norm RMSNorm:
  a = RMSNorm(x, w_attn); attn = Wo @ GQA(RoPE_interleaved(q,k of a), causal);
  h = x + attn; y = h + SwiGLU(RMSNorm(h, w_ffn)).
RoPE uses the interleaved (Meta) convention via official torch complex rotation.
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


def _rope_i(x, positions, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = torch.outer(positions.to(torch.float64), inv)  # (L, d/2)
    cis = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis).reshape(x.shape)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    B, L, d, d_ff, n_heads, n_kv_heads = 2, 5, 16, 32, 4, 2
    dk = d // n_heads
    arr = {
        "x": rng.standard_normal((B, L, d)),
        "Wq": rng.standard_normal((d, d)),
        "Wk": rng.standard_normal((n_kv_heads * dk, d)),
        "Wv": rng.standard_normal((n_kv_heads * dk, d)),
        "Wo": rng.standard_normal((d, d)),
        "gate": rng.standard_normal((d_ff, d)),
        "up": rng.standard_normal((d_ff, d)),
        "down": rng.standard_normal((d, d_ff)),
        "attn_norm": rng.standard_normal(d),
        "ffn_norm": rng.standard_normal(d),
        "n_heads": np.array(n_heads), "n_kv_heads": np.array(n_kv_heads),
        "positions": np.arange(L),
    }
    causal = np.triu(np.ones((L, L), dtype=bool), k=1)
    t = {k: torch.from_numpy(v) for k, v in arr.items() if v.ndim > 0}
    pos = t["positions"]
    a = F.rms_norm(t["x"], (d,), weight=t["attn_norm"], eps=1e-5)
    q = _rope_i(_split(F.linear(a, t["Wq"]), n_heads), pos)
    k = _rope_i(_split(F.linear(a, t["Wk"]), n_kv_heads), pos)
    v = _split(F.linear(a, t["Wv"]), n_kv_heads)
    g = n_heads // n_kv_heads
    k = k.repeat_interleave(g, dim=-3)
    v = v.repeat_interleave(g, dim=-3)
    am = torch.from_numpy(np.where(causal, -np.inf, 0.0))
    o = F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)), t["Wo"])
    h = t["x"] + o
    f = F.rms_norm(h, (d,), weight=t["ffn_norm"], eps=1e-5)
    swiglu = F.linear(F.silu(F.linear(f, t["gate"])) * F.linear(f, t["up"]), t["down"])
    arr["out"] = (h + swiglu).numpy()
    np.savez(FIX / "basic.npz", **arr)
    print(f"  wrote basic.npz  x{arr['x'].shape} heads={n_heads} kv={n_kv_heads} -> out{arr['out'].shape}")


if __name__ == "__main__":
    main()
