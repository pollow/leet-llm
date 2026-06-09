"""213 — generate frozen golden fixtures for both RoPE conventions.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 213_rope/tests/gen_fixtures.py

Oracles (both float64):
- interleaved: official torch complex rotation (torch.view_as_complex / torch.polar) — the
  Meta / llama3.np formulation.
- rotate-half: HuggingFace ``transformers`` official ``rotate_half`` + standard cos/sin
  (identical to ``apply_rotary_pos_emb``).

Fixtures are prefixed by convention: ``interleaved_*`` and ``half_*``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers.models.llama.modeling_llama import rotate_half

FIX = pathlib.Path(__file__).parent / "fixtures"


def _interleaved_ref(x, positions, base=10000.0):
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2) / d))  # (d/2,)
    # positions broadcasts against x's L axis: (L,) -> (L, d/2), (B, L) -> (B, L, d/2).
    ang = positions[..., None].astype(np.float64) * inv_freq  # (..., L, d/2)
    xt = torch.from_numpy(x)
    cis = torch.polar(torch.ones_like(torch.from_numpy(ang)), torch.from_numpy(ang))
    xc = torch.view_as_complex(xt.reshape(*xt.shape[:-1], -1, 2))  # (..., L, d/2)
    return torch.view_as_real(xc * cis).reshape(x.shape).numpy()


def _half_ref(x, positions, base=10000.0):
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    pos = torch.as_tensor(positions, dtype=torch.float64)
    ang = pos[..., None] * inv_freq  # (..., L, d/2)
    emb = torch.cat([ang, ang], dim=-1)  # (..., L, d)
    cos, sin = emb.cos(), emb.sin()
    xt = torch.from_numpy(x)
    return (xt * cos + rotate_half(xt) * sin).numpy()


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    rng = np.random.default_rng(0)
    cases = [
        ("seq3d", rng.standard_normal((2, 4, 8)), np.arange(4)),
        ("heads4d", rng.standard_normal((2, 3, 5, 8)), np.arange(5)),
        ("offset_positions", rng.standard_normal((1, 1, 4, 8)), np.arange(10, 14)),
        # per-sample positions: each sequence in the batch sits at its own offset
        # (real decode: differing KV-cache lengths). positions is (B, L), not shared.
        ("batched_positions", rng.standard_normal((3, 4, 8)),
         np.array([[0, 1, 2, 3], [10, 11, 12, 13], [5, 6, 7, 8]])),
    ]
    for name, x, positions in cases:
        np.savez(FIX / f"interleaved_{name}.npz", x=x, positions=positions,
                 out=_interleaved_ref(x, positions))
        np.savez(FIX / f"half_{name}.npz", x=x, positions=positions,
                 out=_half_ref(x, positions))
        print(f"  wrote interleaved_{name}.npz / half_{name}.npz  x{x.shape}")


if __name__ == "__main__":
    main()
