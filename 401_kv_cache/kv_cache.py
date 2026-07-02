"""401 — Stateful prefill / decode over a KV cache (Qwen3, GQA).

Turns the *stateless* Qwen3 forward (306) into a *serving engine*: run the prompt
once (``prefill``), then generate one token at a time (``decode_step``) reusing the
key/value tensors already computed — never recomputing the prefix.

Registered surfaces (see README.md):

- ``KVCache``    — per-layer preallocated K/V store, GQA-shaped
                   ``(n_kv_heads, max_seq_len, head_dim)``. The *seam* 402/403 reuse.
- ``prefill``    — full-prompt forward, fills the cache, returns last-position logits.
- ``decode_step``— single-token forward at offset ``cache.length`` with a ``(1×kv_len)``
                   causal mask, appends its K/V, returns logits.
- ``kv_generate``— greedy driver: ``prefill`` then ``n_new`` × ``decode_step``.

Run ``uv run grade 401`` to check your work.

Reuse (do NOT re-inline): ``embedding`` (201), ``rms_norm`` (212), ``qk_norm`` (306),
``rope_half`` (213), ``sdpa`` (205), ``affine`` (003), ``group_last_axis`` /
``ungroup_last_axis`` (001), ``swiglu_ffn`` (214), ``add_residual`` (208),
``triangular_mask`` (009), and ``Qwen3Config`` / ``Qwen3Params`` / ``load_qwen3`` (306).
Do NOT call ``qwen3_forward`` / ``qwen3_decoder_block`` — they are stateless and
discard K/V. Re-author the per-layer decode loop around the K/V seam.
"""

from __future__ import annotations

import numpy as np

from leet_llm import (
    Qwen3Config,
    Qwen3Params,
    add_residual,
    affine,
    embedding,
    group_last_axis,
    load_qwen3,
    qk_norm,
    rms_norm,
    rope_half,
    sdpa,
    swiglu_ffn,
    triangular_mask,
    ungroup_last_axis,
)

__all__ = ["KVCache", "prefill", "decode_step", "kv_generate"]


class KVCache:
    """Per-layer preallocated key/value store for one sequence (GQA-shaped).

    Each layer owns a contiguous store of shape ``(n_kv_heads, max_seq_len, head_dim)``
    — the post-RoPE keys and raw values already computed for the tokens seen so far.
    This is the physical thing HF calls a ``StaticCache`` and vLLM later pages.

    Interface (the contract 402 / 403 build on):

    - ``append(layer, k, v)`` — write this step's keys/values for one layer at the
      current write position. ``k`` / ``v`` have shape ``(n_kv_heads, t, head_dim)``.
    - ``get(layer) -> (K, V)`` — the contiguous cached K/V for that layer, each of
      length ``self.length``.
    - ``length`` — number of tokens cached (advances by exactly 1 per decode step).

    **Write-then-read within one forward call.** Each call to ``prefill`` or
    ``decode_step`` loops over layers; for each layer it calls ``append`` first (writing
    that layer's K/V), then calls ``get`` for attention. ``get(layer)`` MUST return the
    tokens appended earlier in the *same call* — not only tokens from prior calls.
    ``length`` reflects *committed* tokens; it advances by exactly 1 after the *entire*
    forward (not per-layer). A naive implementation that advances ``length`` on the first
    layer will write subsequent layers at the wrong offset.

    GQA-specific by construction (one store per KV head); no MLA generalization.
    """

    def __init__(self, cfg: Qwen3Config) -> None:
        raise NotImplementedError

    @property
    def length(self) -> int:
        """Tokens cached (defined by layer 0's timeline, like HF ``seen_tokens``)."""
        raise NotImplementedError

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def prefill(
    prompt_ids: np.ndarray,
    params: Qwen3Params,
    cfg: Qwen3Config,
    cache: KVCache,
) -> np.ndarray:
    """Full-prompt Qwen3 forward at ``positions = arange(len)``; fills ``cache`` for
    every layer; returns **last-position** logits ``(1, V)``.

    This is the *compute-bound* phase — one ``L×L`` causal attention over the prompt.
    """
    raise NotImplementedError


def decode_step(
    token_id: int,
    params: Qwen3Params,
    cfg: Qwen3Config,
    cache: KVCache,
) -> np.ndarray:
    """Single-token forward at ``positions = [cache.length]`` with a ``(1×kv_len)``
    causal mask (the new query attends every cached key); appends its per-layer K/V;
    returns logits ``(1, V)``.

    This is the *memory-bandwidth-bound* phase — one query row over ``kv_len`` keys.
    """
    raise NotImplementedError


def kv_generate(
    prompt_ids: np.ndarray,
    params: Qwen3Params,
    cfg: Qwen3Config,
    n_new: int,
) -> list[int]:
    """Greedy driver: ``prefill`` the prompt, then ``n_new`` × (argmax → ``decode_step``).

    Returns ``prompt + generated`` token ids. The Tier-C real-weights demo entry point.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Skippable real-weights demo: watch your own KVCache generate Qwen3-0.6B text.
    # Grade-time tests are hermetic (frozen fixture); this needs `./download.sh` first.
    from utils import run_kv_generate_cli

    run_kv_generate_cli(
        module_name="401_kv_cache/kv_cache.py",
        load_fn=load_qwen3,
        kv_generate_fn=kv_generate,
        config_cls=Qwen3Config,
    )
