"""213 — RoPE (Rotary Position Embedding), conventions + reusable long-context helpers.

Implement ``rope_interleaved``, ``rope_half``, ``rope_qk_dot``, and the reusable helpers
``rope_scaled_freqs`` / ``rope_attention_scale`` / ``rope_from_freqs``. See README.md.
Run `uv run grade 213` to check your work.

Hint: you may reuse ``from leet_llm import interleave, deinterleave, split_halves,
join_halves`` (011). The interleaved form is the one L3 / 216 use.
"""

from __future__ import annotations

import numpy as np

from leet_llm import deinterleave, interleave, split_halves


def calc_angle(dim_head: int, positions: np.ndarray, base: float = 10000.0, inv_freq: np.ndarray = None):
    """Helper class for calculating angle tensor for RoPE."""
    if inv_freq is None:
        idx = np.arange(0, dim_head, 2)  # [0, 2, 4, ..., dim_head]
        inv_freq = np.pow(base, -idx / dim_head)  # [dim_head/2, ]
    return positions[..., None] * inv_freq  # [batch, seq_len, dim_head / 2]


def rope_interleaved(x: np.ndarray, positions: np.ndarray, base: float = 10000.0, inv_freq: np.ndarray = None) -> np.ndarray:
    """RoPE, interleaved (Meta) convention: rotate adjacent pairs (x_2i, x_2i+1)."""
    a, b = deinterleave(x)  # [batch, seq_len, dim_head / 2]

    dim_head = x.shape[-1]
    angle = calc_angle(dim_head, positions, base, inv_freq)

    out_a = a * np.cos(angle) - b * np.sin(angle)
    out_b = a * np.sin(angle) + b * np.cos(angle)

    return interleave(out_a, out_b)


def rope_half(x: np.ndarray, positions: np.ndarray, base: float = 10000.0, inv_freq: np.ndarray = None) -> np.ndarray:
    """RoPE, rotate-half (HF) convention: out = x*cos + [-x2, x1]*sin."""
    a, b = split_halves(x)
    rotate_half = np.concatenate([-b, a], axis=-1)

    dim_head = x.shape[-1]
    angle = calc_angle(dim_head, positions, base, inv_freq)
    # [batch, seq_len, dim_head]
    angle = np.concatenate([angle, angle], axis=-1)

    return x * np.cos(angle) + rotate_half * np.sin(angle)


def rope_qk_dot(q: np.ndarray, k: np.ndarray, m: int, n: int, base: float = 10000.0) -> np.ndarray:
    """Return <RoPE(q, m), RoPE(k, n)> over the last axis (interleaved convention).

    Used to verify RoPE's defining property: this depends only on the relative position
    (n - m), and equals <q, k> when m == n.
    """
    rope_q_m = rope_interleaved(q, np.array(m))
    rope_k_n = rope_interleaved(k, np.array(n))
    return np.sum(rope_q_m * rope_k_n, axis=-1)


def rope_scaled_freqs(
    head_dim: int,
    base: float,
    scaling: dict | None = None,
) -> np.ndarray:
    """Compute the RoPE inverse frequencies under a ``rope_scaling`` schedule.

    The baseline (``default``) schedule is::

        inv_freq = 1.0 / base ** (arange(0, head_dim, 2) / head_dim)   # (head_dim/2,)

    ``scaling`` selects the schedule via ``scaling["rope_type"]``:

    - ``"default"`` / ``None`` — the baseline above (213's frequencies).
    - ``"llama3"`` — keep high frequencies, divide low frequencies by ``factor``,
      smoothly interpolate the medium band (Llama-3.1's schedule).
    Parameters
    ----------
    head_dim:
        Per-head dimension ``d`` (RoPE rotates ``d/2`` pairs).
    base:
        RoPE base frequency (``rope_theta``).
    scaling:
        ``rope_scaling`` dict.  For ``llama3`` reads:
        ``factor``, ``low_freq_factor``, ``high_freq_factor``,
        ``original_max_position_embeddings``.

    Returns
    -------
    np.ndarray, shape ``(head_dim / 2,)``
        The (possibly rescaled) inverse frequencies, float64.
    """
    idx = np.arange(0, head_dim, 2)
    inv_freqs = np.pow(base, -idx / head_dim)

    if scaling is not None and scaling.get("rope_type", "default") == "llama3":
        wavelen = np.pi * 2 / inv_freqs
        O = scaling.get("original_max_position_embeddings")
        low_freq_factor = scaling.get("low_freq_factor")
        high_freq_factor = scaling.get("high_freq_factor")
        scaling_factor = scaling.get("factor")

        low_freq_wavelen = O / low_freq_factor
        high_freq_wavelen = O / high_freq_factor

        low_freq_idx = wavelen > low_freq_wavelen
        inv_freqs[low_freq_idx] /= scaling_factor
        high_freq_idx = wavelen < high_freq_wavelen
        medium_freq_idx = ~(low_freq_idx | high_freq_idx)

        s = (O / wavelen - low_freq_factor) / \
            (high_freq_factor - low_freq_factor)
        inv_freqs[medium_freq_idx] = (1 - s)[medium_freq_idx] * inv_freqs[medium_freq_idx] / \
            scaling_factor + s[medium_freq_idx] * inv_freqs[medium_freq_idx]

    return inv_freqs


def rope_attention_scale(scaling: dict | None = None) -> float:
    """Return schedule-specific RoPE attention scale."""
    raise NotImplementedError("Implement rope_attention_scale — see 213_rope/README.md")


def rope_from_freqs(
    x: np.ndarray,
    positions: np.ndarray,
    inv_freq: np.ndarray,
    pair_type: str = "interleaved"
) -> np.ndarray:
    """RoPE applied with a precomputed ``inv_freq``.
    Parameters
    ----------
    x:
        Activations whose last axis is ``head_dim``, e.g. ``(B, H, L, head_dim)``.
    positions:
        Position indices, shape ``(L,)``.
    inv_freq:
        Inverse frequencies, shape ``(head_dim / 2,)``.
    pair_type:
        "interleaved" or "half"

    Returns
    -------
    np.ndarray
        ``x`` rotated by RoPE, same shape as ``x``.
    """
    if pair_type == "interleaved":
        return rope_interleaved(x, positions, inv_freq=inv_freq)
    elif pair_type == "half":
        return rope_half(x, positions, inv_freq=inv_freq)
    
    raise ValueError(f"Unsupported pair_type ({pair_type})", )
