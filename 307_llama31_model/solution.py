"""307 — Long-context RoPE scaling: the Llama-3.1 whole-model forward.

Llama-3.1 is 303's Llama with one localized delta: the RoPE inverse
frequencies are **rescaled** by a ``rope_scaling`` schedule so the same
pretrained weights generalise to far longer contexts.  Everything else —
GQA, SwiGLU, RMSNorm, rotate-half RoPE — is unchanged from the baseline.

Two delta operators plus the full Llama-3.1 decoder assembly:

1. ``rope_scaled_freqs(head_dim, base, scaling)`` — compute the per-pair
   inverse frequencies ``inv_freq`` for ``default`` (or ``None``) and
   ``llama3``.  This task focuses on Llama-3.1's native schedule.
2. ``rope_from_freqs(x, positions, inv_freq)`` — rotate-half RoPE applied with
   a **precomputed** ``inv_freq`` (213's ``rope_half`` derives the frequencies
   from ``base`` internally; here the scaled frequencies are passed in).
3. ``Llama31Config`` / ``Llama31Params`` / ``load_llama31`` / ``llama31_forward``
   — the Llama-3.1 decoder-only model wired through 303's forward.

See README.md. Run ``uv run grade 307`` to check your work.

Hints:
- ``rope_scaled_freqs``: start from the default schedule
  ``inv_freq = 1 / base**(arange(0, head_dim, 2) / head_dim)``.  Handle
  ``scaling is None``/``rope_type=default`` as baseline and implement the
  native Llama-3.1 ``rope_type=llama3`` frequency bend.
- ``rope_from_freqs``: reuse ``from leet_llm import split_halves`` (011).  Build
  ``angle = positions[..., None] * inv_freq``, duplicate it
  (``concat([angle, angle], -1)``), and return ``x*cos + rotate_half(x)*sin``
  where ``rotate_half(x) = concat([-x2, x1], -1)``.  Identical to 213's
  ``rope_half`` except the frequencies are supplied, not derived from ``base``.
- ``llama31_forward``: call ``llama_forward`` (303) directly first so you can observe
  the default behavior; then refactor the reused path to inject the 307 RoPE schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from leet_llm import LlamaConfig, LlamaParams, llama_forward, load_llama, split_halves


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



def rope_from_freqs(
    x: np.ndarray,
    positions: np.ndarray,
    inv_freq: np.ndarray,
) -> np.ndarray:
    """Rotate-half RoPE applied with a precomputed ``inv_freq``.

    Identical rotation to 213's ``rope_half`` except the inverse frequencies are
    supplied (already scaled by ``rope_scaled_freqs``) rather than derived from a
    ``base``::

        angle = positions[..., None] * inv_freq          # (L, d/2)
        angle = concat([angle, angle], axis=-1)          # (L, d)
        out   = x * cos(angle) + rotate_half(x) * sin(angle)

    with ``rotate_half(x) = concat([-x2, x1], axis=-1)`` over the two halves of
    the last axis.

    Parameters
    ----------
    x:
        Activations whose last axis is ``head_dim``, e.g. ``(B, H, L, head_dim)``.
    positions:
        Position indices, shape ``(L,)``.
    inv_freq:
        Inverse frequencies, shape ``(head_dim / 2,)``.

    Returns
    -------
    np.ndarray
        ``x`` rotated by RoPE, same shape as ``x``.
    """
    a, b = split_halves(x)
    rotate_half = np.concatenate([-b, a], axis = -1)

    angle = positions[..., None] * inv_freq
    angle = np.concatenate([angle, angle], axis=-1)

    return x * np.cos(angle) + rotate_half * np.sin(angle)


# ---------------------------------------------------------------------------
# Llama-3.1 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Llama31Config:
    """Configuration for a Llama-3.1 decoder-only model.

    Attributes
    ----------
    dim:
        Hidden size (``hidden_size``).
    n_layers:
        Number of decoder layers.
    n_heads:
        Number of query attention heads.
    n_kv_heads:
        Number of key/value heads (GQA).
    vocab_size:
        Vocabulary size.
    max_seq_len:
        Maximum sequence length (position indices).
    norm_eps:
        RMSNorm epsilon.
    rope_base:
        RoPE base frequency (``rope_theta``).
    rope_scaling:
        The ``rope_scaling`` schedule passed to ``rope_scaled_freqs`` (``None``
        → default RoPE).  Llama-3.1 uses ``rope_type="llama3"``.
    """

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int = 131072
    norm_eps: float = 1e-5
    rope_base: float = 500000.0
    rope_scaling: dict | None = None


Llama31Params = LlamaParams


def _cast_float_arrays_to_float64(obj):
    """Recursively cast floating NumPy arrays inside nested dataclasses/lists."""
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating):
            return obj.astype(np.float64, copy=False)
        return obj
    if isinstance(obj, list):
        return [_cast_float_arrays_to_float64(v) for v in obj]
    if is_dataclass(obj):
        values = {
            f.name: _cast_float_arrays_to_float64(getattr(obj, f.name))
            for f in fields(obj)
        }
        return type(obj)(**values)
    return obj


def load_llama31(weights: dict, cfg: Llama31Config) -> Llama31Params:
    """Reuse 303's Llama weight mapping; 307 changes only RoPE frequencies."""
    cfg303 = LlamaConfig(
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        norm_eps=cfg.norm_eps,
        rope_base=cfg.rope_base,
    )
    params = load_llama(weights, cfg303)
    return _cast_float_arrays_to_float64(params)


def llama31_forward(
    input_ids: np.ndarray,
    params: Llama31Params,
    cfg: Llama31Config,
    start_pos: int = 0,
) -> np.ndarray:
    """Call 303's forward directly as the baseline.

    This intentionally shows the default 303 RoPE path first. The 307 task then asks for
    refactoring the reused block/forward path so scaled rotate-half RoPE can be injected.
    Returns logits of shape ``(B, L, V)``.
    """
    cfg303 = LlamaConfig(
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        norm_eps=cfg.norm_eps,
        rope_base=cfg.rope_base,
    )
    # TODO(307): inject scaled rotate-half RoPE for llama3 parity.
    return llama_forward(input_ids, params, cfg303, start_pos=start_pos)
