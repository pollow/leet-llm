"""401 — Stateful prefill / decode over a KV cache (Qwen3, GQA) — REFERENCE SOLUTION.

Turns the *stateless* Qwen3 forward (306) into a *serving engine*: run the prompt
once (``prefill``), then generate one token at a time (``decode_step``) reusing the
key/value tensors already computed — never recomputing the prefix.

Re-authors 306's block math around a ``KVCache`` seam: the per-layer attention appends
this step's post-RoPE keys / raw values to the cache and reads back the *whole* cached
prefix, so decode attends a single query row over ``kv_len`` keys instead of rebuilding
the square block. The module-global ``sdpa`` name is called directly (never aliased) so
a spy monkeypatched onto this module intercepts every attention call.

Reuse (do NOT re-inline): ``embedding`` (201), ``rms_norm`` (212), ``qk_norm`` (306),
``rope_half`` (213), ``sdpa`` (205), ``affine`` (003), ``group_last_axis`` /
``ungroup_last_axis`` (001), ``swiglu_ffn`` (214), ``add_residual`` (208),
``triangular_mask`` (009), and ``Qwen3Config`` / ``Qwen3Params`` / ``load_qwen3`` (306).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("LEET_LLM_TARGET", "solution")

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
    — the post-RoPE keys and raw values for the tokens seen so far. This is the
    physical thing HF calls a ``StaticCache`` and vLLM later pages.

    - ``append(layer, k, v)`` — write this step's keys/values (shape
      ``(n_kv_heads, t, head_dim)``) at the layer's current write position.
    - ``get(layer) -> (K, V)`` — the contiguous cached K/V of length ``self.length``.
    - ``length`` — tokens cached (defined by layer 0's timeline, like HF
      ``seen_tokens``); advances by exactly 1 per decode step.
    """

    def __init__(self, cfg: Qwen3Config) -> None:
        self.cfg = cfg
        shape = (cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self._k = [np.zeros(shape, dtype=np.float64) for _ in range(cfg.n_layers)]
        self._v = [np.zeros(shape, dtype=np.float64) for _ in range(cfg.n_layers)]
        self._sizes = [0 for _ in range(cfg.n_layers)]

    @property
    def length(self) -> int:
        """Tokens cached (defined by layer 0's timeline, like HF ``seen_tokens``)."""
        return self._sizes[0]

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        t = k.shape[1]
        off = self._sizes[layer]
        self._k[layer][:, off : off + t] = k
        self._v[layer][:, off : off + t] = v
        self._sizes[layer] = off + t

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        n = self._sizes[layer]
        return self._k[layer][:, :n], self._v[layer][:, :n]


def _attn_with_cache(
    a: np.ndarray,
    params,
    cfg: Qwen3Config,
    layer: int,
    positions: np.ndarray,
    mask: np.ndarray,
    cache: KVCache,
) -> np.ndarray:
    """306's GQA + qk-norm + rotate-half attention, threaded through the cache seam.

    ``a`` is the pre-attention-normed hidden state ``(1, t, dim)``. Projects q/k/v,
    normalises + rotates, *appends this step's post-RoPE K and raw V to the cache*,
    reads back the whole cached prefix, and attends a ``(t, kv_len)`` block.
    """
    n_heads, n_kv_heads = cfg.n_heads, cfg.n_kv_heads
    n_g = n_heads // n_kv_heads

    Q = affine(a, params.Wq, params.bq)
    K = affine(a, params.Wk, params.bk)
    V = affine(a, params.Wv, params.bv)

    Q = group_last_axis(Q, n_heads)          # (1, n_heads, t, hd)
    K = group_last_axis(K, n_kv_heads)       # (1, n_kv_heads, t, hd)
    V = group_last_axis(V, n_kv_heads)

    q_shape = Q.shape                         # (1, n_heads, t, hd)
    grouped_shape = [q_shape[0], n_kv_heads, n_g] + list(q_shape[2:])
    Q = Q.reshape(grouped_shape)             # (1, n_kv_heads, n_g, t, hd)
    K = K[:, :, None, ...]                    # (1, n_kv_heads, 1, t, hd)
    V = V[:, :, None, ...]

    Q, K = qk_norm(Q, K, params.q_norm, params.k_norm, cfg.qk_norm_eps)

    Q = rope_half(Q, positions, base=cfg.rope_base)
    K = rope_half(K, positions, base=cfg.rope_base)

    # --- the KV-cache seam: store this step, read back the whole prefix ---
    cache.append(layer, K[0, :, 0], V[0, :, 0])   # each (n_kv_heads, t, hd)
    K_cached, V_cached = cache.get(layer)         # each (n_kv_heads, kv_len, hd)
    K_full = K_cached[None, :, None, ...]         # (1, n_kv_heads, 1, kv_len, hd)
    V_full = V_cached[None, :, None, ...]

    gqa = sdpa(Q, K_full, V_full, mask)          # score row (…, t, kv_len)
    gqa = gqa.reshape(q_shape)
    gqa = ungroup_last_axis(gqa)                  # (1, t, n_heads*hd)
    return affine(gqa, params.Wo, params.bo)


def _decoder_block_with_cache(
    x: np.ndarray,
    params,
    cfg: Qwen3Config,
    layer: int,
    positions: np.ndarray,
    mask: np.ndarray,
    cache: KVCache,
) -> np.ndarray:
    a = rms_norm(x, params.attn_norm, eps=cfg.norm_eps)
    attn = _attn_with_cache(a, params.attn, cfg, layer, positions, mask, cache)
    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=cfg.norm_eps)
    ffn = swiglu_ffn(f, params.ffn)
    return add_residual(h, ffn)


def _forward_with_cache(
    input_ids: np.ndarray,
    params: Qwen3Params,
    cfg: Qwen3Config,
    positions: np.ndarray,
    mask: np.ndarray,
    cache: KVCache,
) -> np.ndarray:
    """One forward over ``input_ids`` (1, t); fills every layer's cache; returns
    last-position logits ``(1, V)``."""
    h = embedding(input_ids, params.tok_embed)
    for layer, block in enumerate(params.layers):
        h = _decoder_block_with_cache(h, block, cfg, layer, positions, mask, cache)
    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T            # (1, t, V)
    return logits[:, -1, :]                   # (1, V)


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
    prompt_ids = np.asarray(prompt_ids)
    L = prompt_ids.shape[-1]
    positions = np.arange(0, L)
    mask = triangular_mask(L)                 # (L, L) lower-triangular causal
    return _forward_with_cache(prompt_ids, params, cfg, positions, mask, cache)


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
    p = cache.length                          # this token's absolute position
    positions = np.array([p])
    kv_len = p + 1
    mask = np.zeros((1, kv_len), dtype=bool)  # all-visible: query sees every cached key
    input_ids = np.array([[int(token_id)]])
    return _forward_with_cache(input_ids, params, cfg, positions, mask, cache)


def kv_generate(
    prompt_ids: np.ndarray,
    params: Qwen3Params,
    cfg: Qwen3Config,
    n_new: int,
) -> list[int]:
    """Greedy driver: ``prefill`` the prompt, then ``n_new`` × (argmax → ``decode_step``).

    Returns ``prompt + generated`` token ids. The Tier-C real-weights demo entry point.
    """
    prompt_ids = np.asarray(prompt_ids)
    cache = KVCache(cfg)
    logits = prefill(prompt_ids, params, cfg, cache)   # (1, V)

    out = [int(t) for t in prompt_ids.reshape(-1)]
    nxt = int(np.argmax(logits[0]))
    out.append(nxt)
    for _ in range(n_new - 1):
        logits = decode_step(nxt, params, cfg, cache)
        nxt = int(np.argmax(logits[0]))
        out.append(nxt)
    return out


if __name__ == "__main__":
    from utils import run_kv_generate_cli

    run_kv_generate_cli(
        module_name="401_kv_cache/solution.py",
        load_fn=load_qwen3,
        kv_generate_fn=kv_generate,
        config_cls=Qwen3Config,
    )
