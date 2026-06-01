"""212 — generate frozen golden fixtures for RMSNorm.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 212_rms_norm/tests/gen_fixtures.py

Oracle: torch float64 ``F.rms_norm`` with eps=1e-5 (matches our default).
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
    specs = {"basic": (4, 8), "threeD": (2, 3, 6)}
    for name, shape in specs.items():
        d = shape[-1]
        x = rng.standard_normal(shape) * 2.0
        weight = rng.standard_normal(d)
        out = F.rms_norm(
            torch.from_numpy(x), (d,), weight=torch.from_numpy(weight), eps=1e-5
        ).numpy()
        np.savez(FIX / f"{name}.npz", x=x, weight=weight, out=out)
        print(f"  wrote {name}.npz  x{x.shape}")


if __name__ == "__main__":
    main()
