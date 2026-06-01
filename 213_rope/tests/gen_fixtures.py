"""213 — generate frozen golden fixtures for RoPE (rotate-half convention).

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 213_rope/tests/gen_fixtures.py

Torch has no canonical RoPE op, so the oracle here is the *pinned* rotate-half reference
(HuggingFace / Llama convention). The fixture locks that specific layout; the test's
invariants (norm preservation, relative-position dot product) verify the math independently.
"""

from __future__ import annotations

import pathlib

import numpy as np

FIX = pathlib.Path(__file__).parent / "fixtures"


def rope_ref(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    d = x.shape[-1]
    half = d // 2
    inv_freq = base ** (-2.0 * np.arange(half) / d)  # (half,)
    ang = positions[:, None] * inv_freq[None, :]  # (L, half)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)  # (L, d)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    cases = [
        ("seq3d", rng.standard_normal((2, 4, 8)), np.arange(4)),
        ("heads4d", rng.standard_normal((2, 3, 5, 8)), np.arange(5)),
        ("offset_positions", rng.standard_normal((1, 1, 4, 8)), np.arange(10, 14)),
    ]
    for name, x, positions in cases:
        out = rope_ref(x, positions)
        np.savez(FIX / f"{name}.npz", x=x, positions=positions, out=out)
        print(f"  wrote {name}.npz  x{x.shape} positions{positions.shape}")


if __name__ == "__main__":
    main()
