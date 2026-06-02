"""304 — sampling-transform goldens (HF logits warpers) + a greedy token sequence.

AUTHORING ONLY (gen group):
    uv run --group gen python 304_generate/tests/gen_fixtures.py
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
from transformers import TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((1, 50)).astype(np.float64)
    lt = torch.from_numpy(logits)
    ids = torch.zeros((1, 1), dtype=torch.long)  # warpers ignore ids here
    temp = TemperatureLogitsWarper(0.7)(ids, lt.clone()).numpy()
    tk = TopKLogitsWarper(top_k=5)(ids, lt.clone()).numpy()
    tp = TopPLogitsWarper(top_p=0.9)(ids, lt.clone()).numpy()
    np.savez(FIX / "warpers.npz", logits=logits, temp_0p7=temp,
             topk_5=tk, topp_0p9=tp)
    print("  wrote warpers.npz")


if __name__ == "__main__":
    main()
