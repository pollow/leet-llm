"""202 — generate frozen golden fixtures for GELU / SiLU.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 202_activations/tests/gen_fixtures.py

Oracle: torch float64 ``F.gelu`` (exact erf form, approximate='none') and ``F.silu``.
(``sigmoid`` is deterministic from formula and tested analytically.)
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
    cases = {
        "vec": rng.standard_normal(64) * 3.0,
        "mat": rng.standard_normal((4, 8)) * 2.0,
        "sweep": np.linspace(-6.0, 6.0, 50),
    }
    for name, x in cases.items():
        xt = torch.from_numpy(x)
        np.savez(
            FIX / f"{name}.npz",
            x=x,
            gelu=F.gelu(xt).numpy(),
            silu=F.silu(xt).numpy(),
            sigmoid=torch.sigmoid(xt).numpy(),
        )
        print(f"  wrote {name}.npz  x{x.shape}")


if __name__ == "__main__":
    main()
