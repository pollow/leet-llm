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
    raise NotImplementedError(
        "Implement _project_q_low_rank — see 311_deepseek_model/README.md"
    )


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
    raise NotImplementedError("Implement mla_project — see 311_deepseek_model/README.md")


def deepseek_moe_ffn(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
) -> np.ndarray:
    """Compute DeepSeek's MoE FFN block.

    Required invariants
    -------------------
    - Use sigmoid router scores, not softmax.
    - Add ``e_score_correction_bias`` only for expert selection.
    - Restrict routing with group-limited top-k.
    - Weight selected experts with the unbiased sigmoid scores.
    - Add the always-on shared experts branch.
    """
    raise NotImplementedError("Implement deepseek_moe_ffn — see 311_deepseek_model/README.md")


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
    raise NotImplementedError("Implement load_deepseek — see 311_deepseek_model/README.md")


def deepseek_forward(
    input_ids: np.ndarray,
    params: DeepseekParams,
    cfg: DeepseekConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Run DeepSeek-V3 causal forward and return logits ``(B, L, vocab_size)``.

    Required wiring order per layer:
    1. pre-attn RMSNorm
    2. ``mla_project``
    3. residual add
    4. post-attn RMSNorm
    5. dense SwiGLU (``i < first_k_dense_replace``) or ``deepseek_moe_ffn``
    6. residual add

    Then apply final RMSNorm and project with ``lm_head.T``.
    """
    raise NotImplementedError("Implement deepseek_forward — see 311_deepseek_model/README.md")
