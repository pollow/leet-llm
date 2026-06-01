"""214 — generate frozen golden fixtures for SwiGLU FFN.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 214_swiglu/tests/gen_fixtures.py

Oracle: torch float64 composition (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T, bias-free,
built from F.linear / F.silu.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    specs = {"small": (2, 2, 3), "medium": (3, 4, 8)}  # (batch, d, d_ff)
    for name, (b, d, d_ff) in specs.items():
        x = rng.standard_normal((b, d))
        W1 = rng.standard_normal((d_ff, d))
        W3 = rng.standard_normal((d_ff, d))
        W2 = rng.standard_normal((d, d_ff))
        xt, W1t, W3t, W2t = (torch.from_numpy(a) for a in (x, W1, W3, W2))
        gate = F.silu(F.linear(xt, W1t))
        out = F.linear(gate * F.linear(xt, W3t), W2t).numpy()
        np.savez(FIX / f"{name}.npz", x=x, W1=W1, W3=W3, W2=W2, out=out)
        print(f"  wrote {name}.npz  x{x.shape} d={d} d_ff={d_ff}")


if __name__ == "__main__":
    main()
