"""207 — Feed-Forward Network (classic MLP).

Implement ``ffn`` (and the ``FFNParams`` container). See README.md for details.
Run `uv run grade 207` to check your work.

Hint: reuse ``from leet_llm import affine, gelu, silu`` (003, 202).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import affine, gelu, silu


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


@dataclass(frozen=True)
class FFNParams:
    """Two-layer MLP weights, applied as ``x @ W.T + b``."""

    W1: np.ndarray  # (d_ff, d)
    b1: np.ndarray  # (d_ff,)
    W2: np.ndarray  # (d, d_ff)
    b2: np.ndarray  # (d,)


def _act(x: np.ndarray, name: str) -> np.ndarray:
    if name in ("gelu", "gelu_new", "gelu_pytorch_tanh"):
        return gelu(x)
    if name in ("relu",):
        return relu(x)
    if name in ("silu", "swish"):
        return silu(x)
    raise ValueError(f"unsupported activation {name}")


def ffn(x: np.ndarray, params: FFNParams, activation: str = "gelu") -> np.ndarray:
    """Classic FFN with configurable activation."""
    x = affine(x, params.W1, params.b1)
    x = _act(x, activation)
    x = affine(x, params.W2, params.b2)
    return x
