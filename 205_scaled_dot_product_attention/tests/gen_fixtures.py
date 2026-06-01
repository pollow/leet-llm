"""205 — generate frozen golden fixtures for scaled_dot_product_attention.

AUTHORING ONLY. Requires the ``gen`` dependency group (torch is the trusted oracle):

    uv run --group gen python 205_scaled_dot_product_attention/tests/gen_fixtures.py

Torch runs in float64 so it matches NumPy's default precision to ~1e-12. The committed
``fixtures/*.npz`` are loaded by ``test_*.py`` with NO torch dependency at grade time.

Mask convention (matches L0 009): boolean, ``True`` ⇒ hide (set score to −∞). We translate
that to torch's additive float ``attn_mask`` (−inf where hidden, 0 elsewhere).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"


def _causal(n: int) -> np.ndarray:
    # our convention: True strictly above the diagonal = future positions to hide
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _reference(q, k, v, mask):
    qt, kt, vt = (torch.from_numpy(a) for a in (q, k, v))
    attn_mask = None
    if mask is not None:
        attn_mask = torch.from_numpy(np.where(mask, -np.inf, 0.0))
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask)
    return out.numpy()


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    def R(*shape):
        return rng.standard_normal(shape)

    key_pad = np.zeros((2, 1, 4), dtype=bool)
    key_pad[..., -1] = True  # hide the last key for every query (padding-style)

    cases = [
        # name,           q,            k,            v,             mask
        ("basic_nomask", R(2, 4, 8), R(2, 4, 8), R(2, 4, 8), None),
        ("causal", R(2, 5, 8), R(2, 5, 8), R(2, 5, 8), _causal(5)),
        ("cross_diff_dv", R(2, 3, 8), R(2, 6, 8), R(2, 6, 10), None),
        ("heads_causal_4d", R(2, 3, 4, 8), R(2, 3, 4, 8), R(2, 3, 4, 8), _causal(4)),
        ("key_padding", R(2, 4, 8), R(2, 4, 8), R(2, 4, 8), key_pad),
    ]

    for name, q, k, v, mask in cases:
        out = _reference(q, k, v, mask)
        arrays = {"q": q, "k": k, "v": v, "out": out}
        if mask is not None:
            arrays["mask"] = mask
        np.savez(FIX / f"{name}.npz", **arrays)
        print(f"  wrote {name}.npz  q{q.shape} k{k.shape} v{v.shape} -> out{out.shape}")


if __name__ == "__main__":
    main()
