"""309 — Attention sinks + GPT-OSS MoE: GPT-OSS whole-model forward.

Two new delta operators plus the full GPT-OSS decoder assembly:

1. ``attention_with_sinks(scores, sink_logits)`` — softmax with one
   extra learned per-head "sink" logit in the denominator, then drop the sink
   column so the returned attention rows sum to ``< 1``.
2. ``gptoss_moe_ffn(...)`` — GPT-OSS's sparse mixture-of-experts FFN.  It is NOT
   Mixtral's ``moe_ffn`` (308): the router has a bias, the gate softmax is taken
   over the **selected top-k** logits (not all experts), each expert carries
   biases, the gate/up halves are **interleaved** (``::2`` / ``1::2``), and the
   activation is a **clamped** GLU ``(up + 1) * gate * sigmoid(alpha * gate)``.
3. ``GptOssConfig`` / ``GptOssParams`` / ``load_gptoss`` / ``gptoss_forward`` —
   the GPT-OSS decoder-only model composing L2 primitives.

See README.md. Run ``uv run grade 309`` to check your work.

Hints:
- ``attention_with_sinks``: reuse ``from leet_llm import softmax`` (005). Expect
  ``scores`` to be pre-masked, append per-head ``sink_logits`` as one extra
  column, softmax over the widened last axis, then return everything except the
  sink column.
- ``gptoss_moe_ffn``: reuse ``top_k`` (007) and ``softmax`` (005). Route each token to
  its top-k experts, softmax the selected logits, and accumulate the weighted clamped-GLU
  expert outputs.  GPT-OSS stores expert weights as ``x @ gate_up_proj[e]`` and
  ``gated @ down_proj[e]`` (no transpose), with gate = even cols, up = odd cols.
- ``gptoss_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` (212) →
  q/k/v ``affine`` (003, WITH bias) + ``group_last_axis`` (001) → YaRN RoPE on q & k →
  repeat-kv + scores ``* head_dim**-0.5`` + layer mask → ``attention_with_sinks``
  → ``@ v`` → merge + o_proj (WITH bias) → ``add_residual``
  (208) → ``rms_norm`` → ``gptoss_moe_ffn`` → residual] → final ``rms_norm`` →
  ``@ lm_head.T``.  Even layers (0, 2, …) use ``sliding_window_mask`` (305); odd layers
  use full causal.
- **YaRN RoPE:** use the formulas in this task's README Step C, implement reusable
  primitives in `213` (`rope_scaled_freqs`, `rope_attention_scale`), then consume
  them here. Compute ``inv_freq = rope_scaled_freqs(head_dim, rope_base,
  cfg.rope_scaling)`` and ``af = rope_attention_scale(cfg.rope_scaling)``, then
  apply ``rope_from_freqs(.., positions, inv_freq) * af`` to q and k (rotate-half,
  NOT ``llama_decoder_block`` 216). ``cfg.rope_scaling=None`` recovers plain
  rotate-half RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    AttnParams,
    RopeParams,
    add_residual,
    deinterleave,
    embedding,
    gqa,
    rms_norm,
    rope_attention_scale,
    sigmoid,
    sliding_window_mask,
    softmax,
    top_k,
    triangular_mask,
)


def attention_with_sinks(
    scores: np.ndarray,
    sink_logits: np.ndarray,
) -> np.ndarray:
    """Softmax-with-sinks attention weights (GPT-OSS delta).

    GPT-OSS adds one learned **sink** logit per head to the softmax denominator.
    Concatenate ``sink_logits`` as an extra key column, softmax over the widened
    axis, then drop the sink column.  Because probability mass leaks into the
    (discarded) sink, each returned attention row sums to ``< 1``.

    Parameters
    ----------
    scores:
        Pre-softmax attention logits, shape ``(B, H, L, L)`` with any
        causal/sliding mask already applied.
    sink_logits:
        Per-head sink logits, shape ``(H,)``. 
    Returns
    -------
    np.ndarray
        Attention weights of shape ``(B, H, L, L)`` (the sink column removed).
        Rows sum to ``1 - sink_mass`` where ``sink_mass`` is the softmax mass on
        the sink column.
    """
    shape = list(scores.shape)
    shape[-1] = 1
    s4 = sink_logits[None, :, None, None]
    sink = np.broadcast_to(s4, shape)
    scores = np.concatenate([scores, sink], axis=-1) # (B, H, L, L+1)
    y = softmax(scores)
    y = y[:, :, :, :-1]
    return y


def gptoss_moe_ffn(
    x: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray,
    gate_up_proj: np.ndarray,
    gate_up_bias: np.ndarray,
    down_proj: np.ndarray,
    down_bias: np.ndarray,
    num_active_experts: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:
    """GPT-OSS sparse mixture-of-experts FFN (the GPT-OSS MoE delta).

    Parameters
    ----------
    x:
        Input activations, shape ``(B, L, d)``.
        Internally flattened to token-major ``(T, d)`` for routing, so passing
        pre-flattened ``(T, d)`` also works and is returned in the same shape.
    router_weight:
        Router weight, shape ``(num_experts, d)``.
    router_bias:
        Router bias, shape ``(num_experts,)``.
    gate_up_proj:
        Stacked gate/up projection, shape ``(num_experts, d, 2 * intermediate)``.
        Applied as ``x @ gate_up_proj[e]`` (input dim first — NO transpose).
    gate_up_bias:
        Gate/up bias, shape ``(num_experts, 2 * intermediate)``.
    down_proj:
        Down projection, shape ``(num_experts, intermediate, d)``.
        Applied as ``gated @ down_proj[e]``.
    down_bias:
        Down bias, shape ``(num_experts, d)``.
    num_active_experts:
        Number of experts each token routes to.
    alpha:
        GLU sigmoid gain (GPT-OSS uses ``1.702``).
    limit:
        Clamp limit for the gate/up pre-activations (GPT-OSS uses ``7.0``).

    Returns
    -------
    np.ndarray
        Same shape as ``x``.
    """
    x_shape = x.shape
    d_model = x_shape[-1]
    x = x.reshape((-1, d_model)) # [T, d]
    router_logits = x @ router_weight.T + router_bias # [T, k]
    weights, idx = top_k(router_logits, num_active_experts) # [T, k]
    weights = softmax(weights)

    out = np.zeros_like(x)
    T = x.shape[0]

    for t in range(T):
        for j, k in enumerate(idx[t]):
            gate_up = x[t] @ gate_up_proj[k] + gate_up_bias[k]
            gate, up = deinterleave(gate_up)
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            glu = gate * sigmoid(alpha * gate)
            gated = (up + 1) * glu
            out[t] += (gated @ down_proj[k] + down_bias[k]) * weights[t, j]

    return out.reshape(x_shape)



# ---------------------------------------------------------------------------
# GPT-OSS whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GptOssConfig:
    """Configuration for a GPT-OSS decoder-only model.

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
    head_dim:
        Per-head dimension (GPT-OSS sets this explicitly; ``n_heads*head_dim``
        need not equal ``dim``).
    vocab_size:
        Vocabulary size.
    intermediate_size:
        Per-expert FFN intermediate width.
    num_local_experts:
        Number of experts per MoE layer.
    num_experts_per_tok:
        Top-k experts each token routes to.
    sliding_window:
        Window size for the sliding-attention (even-indexed) layers.
    norm_eps:
        RMSNorm epsilon.
    rope_base:
        RoPE base frequency (``rope_theta``).
    max_seq_len:
        Maximum sequence length (position indices).
    rope_scaling:
        GPT-OSS's RoPE schedule — a ``rope_scaling`` dict passed to
        ``rope_scaled_freqs`` / ``rope_attention_scale`` (213).  GPT-OSS ships
        ``rope_type="yarn"``; ``None`` falls back to default rotate-half RoPE.
    """

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    sliding_window: int = 128
    norm_eps: float = 1e-5
    rope_base: float = 150000.0
    max_seq_len: int = 4096
    rope_scaling: dict | None = None


@dataclass(frozen=True)
class GptOssParams:
    """Packed weights for a GPT-OSS model.

    Attributes
    ----------
    tok_embed:
        Token embedding table, shape ``(V, d)``.
    layers:
        List of per-layer dicts (see ``load_gptoss`` for key names).
    final_norm:
        Final RMSNorm weight, shape ``(d,)``.
    lm_head:
        Output projection, shape ``(V, d)``.
    """

    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts
    final_norm: np.ndarray  # (d,)
    lm_head: np.ndarray     # (V, d)


def load_gptoss(weights: dict, cfg: GptOssConfig) -> GptOssParams:
    """Map HF-named weight arrays into ``GptOssParams``.

    HF weight names (no un-permute — rotate-half layout as-is)::

        model.embed_tokens.weight                              (V, d)
        model.norm.weight                                      (d,)
        lm_head.weight                                         (V, d)  [absent → tie to embed]

    Per layer ``model.layers.{i}`` — note attention carries q/k/v/o **biases**::

        .input_layernorm.weight                  (d,)
        .post_attention_layernorm.weight         (d,)
        .self_attn.q_proj.weight                 (n_heads    * head_dim, d)
        .self_attn.q_proj.bias                   (n_heads    * head_dim,)
        .self_attn.k_proj.weight                 (n_kv_heads * head_dim, d)
        .self_attn.k_proj.bias                   (n_kv_heads * head_dim,)
        .self_attn.v_proj.weight                 (n_kv_heads * head_dim, d)
        .self_attn.v_proj.bias                   (n_kv_heads * head_dim,)
        .self_attn.o_proj.weight                 (d, n_heads * head_dim)
        .self_attn.o_proj.bias                   (d,)
        .self_attn.sinks                         (n_heads,)
        .mlp.router.weight                       (num_experts, d)
        .mlp.router.bias                         (num_experts,)
        .mlp.experts.gate_up_proj                (num_experts, d, 2 * intermediate)
        .mlp.experts.gate_up_proj_bias           (num_experts, 2 * intermediate)
        .mlp.experts.down_proj                   (num_experts, intermediate, d)
        .mlp.experts.down_proj_bias              (num_experts, d)
    """
    def _f(name: str) -> np.ndarray:
        arr = weights[name]
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float64, copy=False)
        return arr

    tok_embed = _f("model.embed_tokens.weight")
    final_norm = _f("model.norm.weight")
    lm_head = _f("lm_head.weight") if "lm_head.weight" in weights else tok_embed

    layers = []
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}"
        layers.append(
            {
                "attn_norm": _f(f"{prefix}.input_layernorm.weight"),
                "ffn_norm": _f(f"{prefix}.post_attention_layernorm.weight"),
                "Wq": _f(f"{prefix}.self_attn.q_proj.weight"),
                "bq": _f(f"{prefix}.self_attn.q_proj.bias"),
                "Wk": _f(f"{prefix}.self_attn.k_proj.weight"),
                "bk": _f(f"{prefix}.self_attn.k_proj.bias"),
                "Wv": _f(f"{prefix}.self_attn.v_proj.weight"),
                "bv": _f(f"{prefix}.self_attn.v_proj.bias"),
                "Wo": _f(f"{prefix}.self_attn.o_proj.weight"),
                "bo": _f(f"{prefix}.self_attn.o_proj.bias"),
                "sinks": _f(f"{prefix}.self_attn.sinks"),
                "router_weight": _f(f"{prefix}.mlp.router.weight"),
                "router_bias": _f(f"{prefix}.mlp.router.bias"),
                "gate_up_proj": _f(f"{prefix}.mlp.experts.gate_up_proj"),
                "gate_up_bias": _f(f"{prefix}.mlp.experts.gate_up_proj_bias"),
                "down_proj": _f(f"{prefix}.mlp.experts.down_proj"),
                "down_bias": _f(f"{prefix}.mlp.experts.down_proj_bias"),
            }
        )

    return GptOssParams(
        tok_embed=tok_embed,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )

def gptoss_decoder_block(
    x: np.ndarray,
    layer: dict,
    cfg: GptOssConfig,
    positions: np.ndarray,
    mask: np.ndarray,
    af: float,
) -> np.ndarray:
    """One GPT-OSS block: RoPE-GQA(+sinks) + GPT-OSS MoE with pre-norm residuals."""
    a = rms_norm(x, layer["attn_norm"], eps=cfg.norm_eps)
    attn_params = AttnParams(
        Wq=layer["Wq"],
        Wk=layer["Wk"],
        Wv=layer["Wv"],
        Wo=layer["Wo"],
        bq=layer["bq"],
        bk=layer["bk"],
        bv=layer["bv"],
        bo=layer["bo"],
    )
    attn = gqa(
        a,
        attn_params,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        mask=mask,
        positions=positions,
        rope_params=RopeParams(
            base=cfg.rope_base,
            pair_type="half",
            scaling=cfg.rope_scaling,
        ),
        af=af,
        sink_logits=layer["sinks"],
    )
    h = add_residual(x, attn)

    f = rms_norm(h, layer["ffn_norm"], eps=cfg.norm_eps)
    moe = gptoss_moe_ffn(
        f,
        layer["router_weight"],
        layer["router_bias"],
        layer["gate_up_proj"],
        layer["gate_up_bias"],
        layer["down_proj"],
        layer["down_bias"],
        cfg.num_experts_per_tok,
    )
    return add_residual(h, moe)


def gptoss_forward(
    input_ids: np.ndarray,
    params: GptOssParams,
    cfg: GptOssConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Token embed → N GPT-OSS blocks → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    GPT-OSS = rotate-half Llama with **attention sinks** and the **GPT-OSS MoE**.
    Compose from granular L2 primitives (NOT ``llama_decoder_block``):

      ``embedding`` → per layer [``rms_norm`` → q/k/v ``affine`` (WITH bias) +
      head-split → YaRN RoPE on q & k → repeat-kv + scores ``* head_dim**-0.5``
      → ``attention_with_sinks`` with the layer mask → ``@ v`` → merge + o_proj
      (WITH bias) → ``add_residual`` → ``rms_norm`` → ``gptoss_moe_ffn`` → residual]
      → final ``rms_norm`` → ``@ lm_head.T``.

    Even-indexed layers (0, 2, …) use ``sliding_window_mask`` (305); odd-indexed
    layers use full causal.  RoPE is GPT-OSS's real **YaRN** schedule, implemented in
    the RoPE/GQA tasks (213/215): ``inv_freq = rope_scaled_freqs(head_dim, rope_base,
    cfg.rope_scaling)`` and
    ``af = rope_attention_scale(cfg.rope_scaling)``, applied as
    ``rope_from_freqs(.., positions, inv_freq) * af`` on q & k (rotate-half).  The
    long-context KV-cache (streaming sink eviction) is deferred to L4.
    """
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]
    positions = np.arange(start_pos, start_pos + L)
    full_mask = triangular_mask(L)
    sliding_mask = sliding_window_mask(L, cfg.sliding_window)
    af = rope_attention_scale(cfg.rope_scaling)

    for i, layer in enumerate(params.layers):
        mask = sliding_mask if i % 2 == 0 else full_mask
        h = gptoss_decoder_block(h, layer, cfg, positions, mask, af)

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T
    return logits
