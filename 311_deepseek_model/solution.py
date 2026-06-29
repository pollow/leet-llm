"""Task 311: DeepSeek-V3 forward contracts.

This module defines the student-facing APIs for:
- ``_project_q_low_rank``: internal helper for DeepSeek's low-rank Q projection.
- ``mla_project``: DeepSeek MLA attention operator.
- ``deepseek_moe_ffn``: DeepSeek's sigmoid/group-limited MoE operator.
- ``load_deepseek`` + ``deepseek_forward``: whole-model forward wiring.

The tutorial rationale and step-by-step implementation guidance live in
``311_deepseek_model/README.md``. Docstrings below focus on shape contracts,
wiring invariants, and high-risk gotchas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    SwiGLUParams,
    add_residual,
    embedding,
    masked_fill,
    rms_norm,
    rope_half,
    sdpa,
    sigmoid,
    swiglu_ffn,
    top_k,
    triangular_mask,
)



def _project_q_low_rank(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
) -> np.ndarray:
    """Build Q with DeepSeek's low-rank query projection.

    Contract
    --------
    Input ``x`` has shape ``(B, L, d)``. Return shape is
    ``(B, H, L, qk_head_dim)``, where
    ``qk_head_dim = qk_nope_head_dim + qk_rope_head_dim``.

    Required layer keys: ``q_a_proj``, ``q_a_layernorm``, ``q_b_proj``.
    """
    B, L, _ = x.shape
    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

    qa = x @ layer["q_a_proj"].T
    qa_norm = rms_norm(qa, layer["q_a_layernorm"], cfg.norm_eps)
    qb = qa_norm @ layer["q_b_proj"].T
    return qb.reshape((B, L, cfg.n_heads, qk_head_dim)).transpose((0, 2, 1, 3))


def mla_project(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
    positions: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute one DeepSeek MLA attention block output.

    Parameters
    ----------
    x:
        Input activations, shape ``(B, L, d)``.
    layer:
        Per-layer weight dict produced by ``load_deepseek``.
    cfg:
        MLA dimensions and RoPE configuration.
    positions:
        Integer position indices, shape ``(L,)``.

    Returns
    -------
    np.ndarray
        Output shape ``(B, L, d)``.

    Required invariants
    -------------------
    - Use decoupled RoPE: apply ``rope_half`` only to rope slices.
    - Build full ``q``/``k`` by concatenating nope and rope slices.
    - Use causal scaled-dot-product attention before ``o_proj``.
    """
    q = _project_q_low_rank(x, layer, cfg) # [B, H, L, qk_head_dim]
    B, H, L, _ = q.shape
    q_nope = q[..., :cfg.qk_nope_head_dim]
    q_rope = q[..., cfg.qk_nope_head_dim:]

    compressed = x @ layer["kv_a_proj"].T # [B, L, kv_lora_rank + qk_rope_head_dim]
    assert compressed.shape[-1] == cfg.kv_lora_rank + cfg.qk_rope_head_dim
    c_kv = compressed[..., :cfg.kv_lora_rank]
    k_rope = compressed[..., cfg.kv_lora_rank:]

    c_kv = rms_norm(c_kv, layer["kv_a_layernorm"], cfg.norm_eps)
    kv = c_kv @ layer["kv_b_proj"].T # [B, L, H * (k_nope + v)]
    kv = kv.reshape((B, L, H, -1)).transpose(0, 2, 1, 3)
    k_nope = kv[..., :cfg.qk_nope_head_dim]
    v = kv[..., cfg.qk_nope_head_dim:]

    k_rope = rope_half(k_rope, positions, base=cfg.rope_base)
    q_rope = rope_half(q_rope, positions, base=cfg.rope_base)

    k_rope = k_rope[:, None, ...]
    k_rope = np.broadcast_to(k_rope, (B, H, L, cfg.qk_rope_head_dim))

    q_full = np.concatenate([q_nope, q_rope], axis=-1)
    k_full = np.concatenate([k_nope, k_rope], axis=-1)

    out = sdpa(q_full, k_full, v, mask) # [B, H, L, v_dim]
    out = out.transpose((0, 2, 1, 3)).reshape((B, L, -1)) # [B, L, H * v_dim]
    return out @ layer["o_proj"].T


def _deepseek_group_limited_topk(
    scores_biased: np.ndarray,
    cfg: "DeepseekConfig",
) -> np.ndarray:
    """Select routed experts using DeepSeek's group-limited top-k policy."""
    T = scores_biased.shape[0]
    experts_per_group = cfg.n_routed_experts // cfg.n_group

    group_view = scores_biased.reshape(T, cfg.n_group, experts_per_group)
    group_top = min(2, experts_per_group)
    group_vals, _ = top_k(group_view, group_top)
    group_scores = group_vals.sum(axis=-1)
    _, selected_groups = top_k(group_scores, cfg.topk_group)

    group_mask = np.zeros((T, cfg.n_group), dtype=bool)
    np.put_along_axis(group_mask, selected_groups, True, axis=1)
    expert_mask = np.broadcast_to(
        group_mask[:, :, None],
        (T, cfg.n_group, experts_per_group),
    ).reshape(T, cfg.n_routed_experts)

    masked_scores = masked_fill(scores_biased, ~expert_mask, -np.inf)
    _, topk_indices = top_k(masked_scores, cfg.num_experts_per_tok)
    return topk_indices


def _dispatch_deepseek_experts(
    x_flat: np.ndarray,
    topk_indices: np.ndarray,
    topk_weights: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
) -> np.ndarray:
    """Run selected routed experts and accumulate their weighted outputs."""
    routed = np.zeros_like(x_flat)
    gate_up = layer["experts_gate_up_proj"]
    down = layer["experts_down_proj"]
    ffn_dim = cfg.moe_intermediate_size

    for k in range(cfg.num_experts_per_tok):
        expert_idx = topk_indices[:, k]
        weights = topk_weights[:, k]
        for e in range(cfg.n_routed_experts):
            token_mask = expert_idx == e
            if not token_mask.any():
                continue
            expert_out = swiglu_ffn(
                x_flat[token_mask],
                SwiGLUParams(
                    W1=gate_up[e, :ffn_dim],
                    W3=gate_up[e, ffn_dim:],
                    W2=down[e],
                ),
            )
            routed[token_mask] += expert_out * weights[token_mask, None]

    return routed


def deepseek_moe_ffn(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
) -> np.ndarray:
    """Compute DeepSeek's MoE FFN block.

    DeepSeek routing differs from Mixtral: it uses sigmoid scores, adds a bias
    only for expert selection, restricts selection by group, and adds shared
    always-on experts.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape((-1, d))
    T = x_flat.shape[0]

    scores = sigmoid(x_flat @ layer["gate"].T)
    topk_indices = _deepseek_group_limited_topk(
        scores + layer["e_score_correction_bias"],
        cfg,
    )
    topk_weights = scores[np.arange(T)[:, None], topk_indices]
    topk_weights = topk_weights / (topk_weights.sum(axis=-1, keepdims=True) + 1e-20)
    topk_weights = topk_weights * cfg.routed_scaling_factor

    routed = _dispatch_deepseek_experts(x_flat, topk_indices, topk_weights, layer, cfg)
    shared = swiglu_ffn(
        x,
        SwiGLUParams(
            W1=layer["shared_gate_proj"],
            W3=layer["shared_up_proj"],
            W2=layer["shared_down_proj"],
        ),
    )
    return routed.reshape(orig_shape) + shared



# ---------------------------------------------------------------------------
# DeepSeek-V3 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepseekConfig:
    """Runtime hyperparameters for task-311 forward."""

    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    moe_intermediate_size: int
    q_lora_rank: int = 1536
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    rope_type: str = "default"
    rope_factor: float = 1.0
    mscale: float = 0.0
    mscale_all_dim: float = 0.0
    intermediate_size: int = 0  # dense MLP size (layers < first_k_dense_replace)
    tie_word_embeddings: bool = False


@dataclass(frozen=True)
class DeepseekParams:
    """Packed tensors consumed by ``deepseek_forward``."""

    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts (see load_deepseek)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray     # (V, d)


def load_deepseek(weights: dict, cfg: DeepseekConfig) -> DeepseekParams:
    """Map HF-style weight names into ``DeepseekParams``.

    Contract
    --------
    - Consume embedding/final norm/lm-head tensors.
    - Build one layer dict per decoder block with:
      - MLA weights
      - dense FFN weights for early layers
      - MoE + shared expert weights for later layers

    Notes
    -----
    - Keep rotate-half RoPE layout as-is (no extra un-permute).
    - Exact expected keys and shapes are documented in README.
    """
    def _f(name: str) -> np.ndarray:
        arr = weights[name]
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float64, copy=False)
        return arr

    tok_embed = _f("model.embed_tokens.weight")
    final_norm = _f("model.norm.weight")

    if cfg.tie_word_embeddings or "lm_head.weight" not in weights:
        lm_head = tok_embed
    else:
        lm_head = _f("lm_head.weight")

    layers: list[dict[str, np.ndarray]] = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}"
        layer: dict[str, np.ndarray] = {
            "input_layernorm": _f(f"{p}.input_layernorm.weight"),
            "post_attention_layernorm": _f(f"{p}.post_attention_layernorm.weight"),
            "kv_a_proj": _f(f"{p}.self_attn.kv_a_proj_with_mqa.weight"),
            "kv_a_layernorm": _f(f"{p}.self_attn.kv_a_layernorm.weight"),
            "kv_b_proj": _f(f"{p}.self_attn.kv_b_proj.weight"),
            "o_proj": _f(f"{p}.self_attn.o_proj.weight"),
        }

        layer["q_a_proj"] = _f(f"{p}.self_attn.q_a_proj.weight")
        layer["q_a_layernorm"] = _f(f"{p}.self_attn.q_a_layernorm.weight")
        layer["q_b_proj"] = _f(f"{p}.self_attn.q_b_proj.weight")

        if i < cfg.first_k_dense_replace:
            layer["gate_proj"] = _f(f"{p}.mlp.gate_proj.weight")
            layer["up_proj"] = _f(f"{p}.mlp.up_proj.weight")
            layer["down_proj"] = _f(f"{p}.mlp.down_proj.weight")
        else:
            layer["gate"] = _f(f"{p}.mlp.gate.weight")
            layer["e_score_correction_bias"] = _f(
                f"{p}.mlp.gate.e_score_correction_bias"
            )
            layer["experts_gate_up_proj"] = _f(f"{p}.mlp.experts.gate_up_proj")
            layer["experts_down_proj"] = _f(f"{p}.mlp.experts.down_proj")
            layer["shared_gate_proj"] = _f(f"{p}.mlp.shared_experts.gate_proj.weight")
            layer["shared_up_proj"] = _f(f"{p}.mlp.shared_experts.up_proj.weight")
            layer["shared_down_proj"] = _f(f"{p}.mlp.shared_experts.down_proj.weight")

        layers.append(layer)

    return DeepseekParams(
        tok_embed=tok_embed,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def deepseek_forward(
    input_ids: np.ndarray,
    params: DeepseekParams,
    cfg: DeepseekConfig,
) -> np.ndarray:
    """Run DeepSeek-V3 causal forward and return logits ``(B, L, vocab_size)``.

    Required wiring order per layer:
    1. pre-attn RMSNorm
    2. ``mla_project``
    3. residual add
    4. post-attn RMSNorm
    5. dense SwiGLU (``i < first_k_dense_replace``) or DeepSeek MoE
    6. residual add

    Then apply final RMSNorm and project with ``lm_head.T``.
    """
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]
    positions = np.arange(0, L)

    full_mask = triangular_mask(L)

    for i, layer in enumerate(params.layers):
        h_attn_in = rms_norm(h, layer["input_layernorm"], cfg.norm_eps)
        attn = mla_project(h_attn_in, layer, cfg, positions, full_mask)
        h = add_residual(h, attn)

        f = rms_norm(h, layer["post_attention_layernorm"], cfg.norm_eps)
        if i < cfg.first_k_dense_replace:
            ffn = swiglu_ffn(
                f,
                SwiGLUParams(
                    W1=layer["gate_proj"],
                    W3=layer["up_proj"],
                    W2=layer["down_proj"],
                ),
            )
        else:
            ffn = deepseek_moe_ffn(f, layer, cfg)
        h = add_residual(h, ffn)

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    return h @ params.lm_head.T
