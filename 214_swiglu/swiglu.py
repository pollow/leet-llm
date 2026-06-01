"""214 — SwiGLU Feed-Forward.

Implement ``swiglu_ffn`` (and the ``SwiGLUParams`` container). See README.md.
Run `uv run grade 214` to check your work.

Hint: reuse ``from leet_llm import silu, affine`` (202, 003); the linears are bias-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwiGLUParams:
    """Bias-free SwiGLU weights, applied as ``x @ W.T``."""

    W1: np.ndarray  # (d_ff, d) — gate
    W3: np.ndarray  # (d_ff, d) — up
    W2: np.ndarray  # (d, d_ff) — down


def swiglu_ffn(x: np.ndarray, params: SwiGLUParams) -> np.ndarray:
    """SwiGLU FFN: ``(silu(x @ W1.T) * (x @ W3.T)) @ W2.T``."""
    raise NotImplementedError("Implement swiglu_ffn — see 214_swiglu/README.md")
