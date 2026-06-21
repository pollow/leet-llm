"""307 — Long-context RoPE scaling: the Llama-3.1 whole-model forward.

Llama-3.1 is 303's Llama with one localized delta: the RoPE inverse
frequencies are **rescaled** by a ``rope_scaling`` schedule so the same
pretrained weights generalise to far longer contexts.  Everything else —
GQA, SwiGLU, RMSNorm, rotate-half RoPE — is unchanged from the baseline.

Three delta operators plus the full Llama-3.1 decoder assembly:

1. ``rope_scaled_freqs(head_dim, base, scaling)`` — compute the per-pair
   inverse frequencies ``inv_freq`` for a given ``rope_type`` (one of
   ``default`` / ``linear`` / ``dynamic`` / ``llama3`` / ``yarn``).  With
   ``scaling=None`` (or ``rope_type="default"``) this is the plain RoPE
   schedule ``1 / base**(arange(0, d, 2) / d)``; the other types bend those
   frequencies so high positions interpolate instead of extrapolate.
2. ``rope_from_freqs(x, positions, inv_freq)`` — rotate-half RoPE applied with
   a **precomputed** ``inv_freq`` (213's ``rope_half`` derives the frequencies
   from ``base`` internally; here the scaled frequencies are passed in).
3. ``rope_attention_scale(scaling)`` — the *other* half of RoPE scaling: a scalar
   attention "temperature" multiplied into the rotated q & k.  It is ``1.0`` for
   ``default`` / ``linear`` / ``dynamic`` / ``llama3`` (so Llama-3.1 is a no-op),
   and the YaRN ``mscale`` for ``yarn`` — reused by GPT-OSS (309), whose real RoPE
   is YaRN.
4. ``Llama31Config`` / ``Llama31Params`` / ``load_llama31`` / ``llama31_forward``
   — the Llama-3.1 decoder-only model composing L2 primitives.

See README.md. Run ``uv run grade 307`` to check your work.

Hints:
- ``rope_scaled_freqs``: start from the default schedule
  ``inv_freq = 1 / base**(arange(0, head_dim, 2) / head_dim)``.  Then per
  ``scaling["rope_type"]``: ``linear`` divides ``inv_freq`` by ``factor``;
  ``dynamic`` recomputes ``base`` from ``factor``/``seq_len``/``max_position_embeddings``;
  ``llama3`` keeps high frequencies, divides low frequencies by ``factor``, and
  smoothly interpolates the band between ``high_freq_factor`` and
  ``low_freq_factor``; ``yarn`` blends extrapolation/interpolation with a linear
  ramp over the correction range.  The README has every closed form.
- ``rope_from_freqs``: reuse ``from leet_llm import split_halves`` (011).  Build
  ``angle = positions[..., None] * inv_freq``, duplicate it
  (``concat([angle, angle], -1)``), and return ``x*cos + rotate_half(x)*sin``
  where ``rotate_half(x) = concat([-x2, x1], -1)``.  Identical to 213's
  ``rope_half`` except the frequencies are supplied, not derived from ``base``.
- ``rope_attention_scale``: return ``1.0`` for every schedule except ``yarn``; for
  ``yarn`` return the ``mscale`` (``0.1 * log(factor) + 1`` when ``factor > 1``,
  else ``1.0``) unless an explicit ``attention_factor`` is given.  Used to scale the
  rotated q & k (equivalently, the cos/sin).  See README.
- ``llama31_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` (212)
  → q/k/v ``affine`` (003, NO bias) + ``group_last_axis`` (001) → ``rope_from_freqs``
  on q & k with the scaled ``inv_freq`` (then ``* rope_attention_scale``) → repeat-kv
  → ``sdpa`` (205) with a ``triangular_mask`` (009) → ``ungroup_last_axis`` (001) +
  o_proj → ``add_residual`` (208) → ``rms_norm`` → ``swiglu_ffn`` (214) → residual] →
  final ``rms_norm`` → ``@ lm_head.T``.  Use rotate-half RoPE, NOT
  ``llama_decoder_block`` (216, interleaved-only).  Compute ``inv_freq`` and the
  attention scale once.  (For Llama-3.1's ``llama3`` schedule the scale is ``1.0``.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rope_scaled_freqs(
    head_dim: int,
    base: float,
    scaling: dict | None = None,
) -> np.ndarray:
    """Compute the RoPE inverse frequencies under a ``rope_scaling`` schedule.

    The baseline (``default``) schedule is::

        inv_freq = 1.0 / base ** (arange(0, head_dim, 2) / head_dim)   # (head_dim/2,)

    ``scaling`` selects a long-context variant via ``scaling["rope_type"]``:

    - ``"default"`` / ``None`` — the baseline above (213's frequencies).
    - ``"linear"`` — ``inv_freq /= factor`` (positions linearly interpolated).
    - ``"dynamic"`` — NTK-by-parts: recompute ``base`` from ``factor``,
      ``seq_len`` and ``max_position_embeddings``, then the baseline formula.
    - ``"llama3"`` — keep high frequencies, divide low frequencies by ``factor``,
      smoothly interpolate the medium band (Llama-3.1's schedule).
    - ``"yarn"`` — blend extrapolation/interpolation with a linear ramp over the
      correction range derived from ``beta_fast``/``beta_slow``.

    Parameters
    ----------
    head_dim:
        Per-head dimension ``d`` (RoPE rotates ``d/2`` pairs).
    base:
        RoPE base frequency (``rope_theta``).
    scaling:
        ``rope_scaling`` dict.  Recognised keys depend on ``rope_type``
        (see README): ``factor``, ``low_freq_factor``, ``high_freq_factor``,
        ``original_max_position_embeddings``, ``max_position_embeddings``,
        ``seq_len``, ``beta_fast``, ``beta_slow``, ``truncate``.

    Returns
    -------
    np.ndarray, shape ``(head_dim / 2,)``
        The (possibly rescaled) inverse frequencies, float64.
    """
    raise NotImplementedError("Implement rope_scaled_freqs — see 307_llama31_model/README.md")


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
    raise NotImplementedError("Implement rope_from_freqs — see 307_llama31_model/README.md")


def rope_attention_scale(scaling: dict | None = None) -> float:
    """Attention "temperature" scale for a ``rope_scaling`` schedule.

    The companion to ``rope_scaled_freqs``: YaRN (and longrope) rescale not only the
    frequencies but also the attention logits, by multiplying the cos/sin — and hence
    the rotated q & k — by a scalar ``attention_factor``.  Every other schedule leaves
    it at ``1.0``.

    For ``rope_type="yarn"`` with no explicit ``attention_factor`` and no
    ``mscale``/``mscale_all_dim``::

        scale = 0.1 * log(factor) + 1.0        if factor > 1
              = 1.0                              otherwise

    Parameters
    ----------
    scaling:
        The ``rope_scaling`` dict (``None`` → ``1.0``).  Reads ``rope_type``,
        ``factor``, and optionally ``attention_factor`` / ``mscale`` /
        ``mscale_all_dim``.

    Returns
    -------
    float
        The attention scale (``1.0`` for ``default`` / ``linear`` / ``dynamic`` /
        ``llama3``).
    """
    raise NotImplementedError("Implement rope_attention_scale — see 307_llama31_model/README.md")


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


@dataclass(frozen=True)
class Llama31Params:
    """Packed weights for a Llama-3.1 model.

    Attributes
    ----------
    tok_embed:
        Token embedding table, shape ``(V, d)``.
    layers:
        List of per-layer dicts (see ``load_llama31`` for key names).
    final_norm:
        Final RMSNorm weight, shape ``(d,)``.
    lm_head:
        Output projection, shape ``(V, d)``.
    """

    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts
    final_norm: np.ndarray  # (d,)
    lm_head: np.ndarray     # (V, d)


def load_llama31(weights: dict, cfg: Llama31Config) -> Llama31Params:
    """Map HF-named weight arrays into ``Llama31Params``.

    Llama-3.1 uses rotate-half RoPE, so weights map **as-is** (no un-permute).
    The attention projections are **bias-free** (like Llama, unlike GPT-OSS)::

        model.embed_tokens.weight                            (V, d)
        model.norm.weight                                    (d,)
        lm_head.weight                                       (V, d)  [absent → tie to embed]

    Per layer ``model.layers.{i}``::

        .input_layernorm.weight                  (d,)
        .post_attention_layernorm.weight         (d,)
        .self_attn.q_proj.weight                 (n_heads    * head_dim, d)
        .self_attn.k_proj.weight                 (n_kv_heads * head_dim, d)
        .self_attn.v_proj.weight                 (n_kv_heads * head_dim, d)
        .self_attn.o_proj.weight                 (d, n_heads * head_dim)
        .mlp.gate_proj.weight                    (intermediate, d)
        .mlp.up_proj.weight                      (intermediate, d)
        .mlp.down_proj.weight                    (d, intermediate)
    """
    raise NotImplementedError("Implement load_llama31 — see 307_llama31_model/README.md")


def llama31_forward(
    input_ids: np.ndarray,
    params: Llama31Params,
    cfg: Llama31Config,
    start_pos: int = 0,
) -> np.ndarray:
    """Token embed → N Llama blocks → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    Llama-3.1 = 303's Llama with the RoPE frequencies rescaled by
    ``rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)`` (computed once).
    Compose from granular L2 primitives (NOT ``llama_decoder_block``):

      ``embedding`` → per layer [``rms_norm`` → q/k/v ``affine`` (no bias) +
      head-split → ``rope_from_freqs`` on q & k (then ``* rope_attention_scale``) →
      repeat-kv → ``sdpa`` with a causal ``triangular_mask`` → merge + o_proj →
      ``add_residual`` → ``rms_norm`` → ``swiglu_ffn`` → residual] → final
      ``rms_norm`` → ``@ lm_head.T``.

    RoPE is rotate-half (``rope_from_freqs``); the attention scale is ``1.0`` for
    Llama-3.1's ``llama3`` schedule.  Long-context decode (the KV-cache side of the
    scaled schedule) is deferred to L4.
    """
    raise NotImplementedError("Implement llama31_forward — see 307_llama31_model/README.md")
