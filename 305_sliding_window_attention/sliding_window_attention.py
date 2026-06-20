"""305 — Sliding-Window (Banded Causal) Mask — Mistral delta.

Implement ``sliding_window_mask``. See README.md for the full explanation.
Run `uv run grade 305` to check your work.

Hint: reuse ``from leet_llm import triangular_mask`` (009). The band is
``(i - W, i]`` — query position ``i`` attends to key positions ``j`` with
``i - W < j <= i``.
"""

from __future__ import annotations

import numpy as np


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Return an additive ``(seq_len, seq_len)`` causal sliding-window mask.

    The mask is ``0.0`` where query ``i`` may attend to key ``j`` (i.e. ``j``
    is within the causal window ``(i − window, i]``) and ``-inf`` elsewhere.

    Parameters
    ----------
    seq_len:
        Sequence length ``L``.
    window:
        Number of past tokens each query can attend to (Mistral
        ``sliding_window`` config field). When ``window >= seq_len`` the mask
        reduces to the standard causal (lower-triangular) mask.

    Returns
    -------
    np.ndarray, shape ``(L, L)``, dtype float64
        Additive pre-softmax mask.  Add it to the raw attention scores
        ``Q K^T / sqrt(d_k)`` before the softmax.
    """
    raise NotImplementedError("Implement sliding_window_mask — see 305_sliding_window_attention/README.md")
