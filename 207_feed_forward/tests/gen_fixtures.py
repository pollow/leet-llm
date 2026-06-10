"""207 — generate frozen golden fixtures for the classic FFN.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 207_feed_forward/tests/gen_fixtures.py

Oracle: torch float64 ``F.linear`` + exact ``F.gelu``: gelu(x @ W1.T + b1) @ W2.T + b2.
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
    specs = {
        "basic": (2, 4, 16, 32, "gelu"),
        "twoD": (3, None, 8, 16, "gelu"),
        "silu_basic": (2, 4, 16, 32, "silu"),
        "silu_twoD": (3, None, 8, 16, "silu"),
    }
    act_map = {"gelu": F.gelu, "silu": F.silu, "swish": F.silu, "relu": F.relu}
    for name, (b, length, d, d_ff, act) in specs.items():
        shape = (b, length, d) if length else (b, d)
        x = rng.standard_normal(shape)
        W1 = rng.standard_normal((d_ff, d))
        b1 = rng.standard_normal(d_ff)
        W2 = rng.standard_normal((d, d_ff))
        b2 = rng.standard_normal(d)
        xt, W1t, b1t, W2t, b2t = (torch.from_numpy(a) for a in (x, W1, b1, W2, b2))
        fn = act_map[act]
        out = F.linear(fn(F.linear(xt, W1t, b1t)), W2t, b2t).numpy()
        np.savez(
            FIX / f"{name}.npz",
            x=x,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            out=out,
            activation=np.array(act),
        )
        print(f"  wrote {name}.npz  x{x.shape} d={d} d_ff={d_ff} act={act}")


if __name__ == "__main__":
    main()
