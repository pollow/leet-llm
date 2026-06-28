"""Task 311: DeepSeek-V3 forward contracts.

This module defines the student-facing APIs for:
- ``mla_project``: DeepSeek MLA attention operator.
- ``load_deepseek`` + ``deepseek_forward``: whole-model forward wiring.

The tutorial rationale and step-by-step implementation guidance live in
``311_deepseek_model/README.md``. Docstrings below focus on shape contracts,
wiring invariants, and high-risk gotchas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def mla_project(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
    positions: np.ndarray,
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
    q_lora_rank: Optional[int] = None
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
    5. dense SwiGLU (``i < first_k_dense_replace``) or DeepSeek MoE
    6. residual add

    Then apply final RMSNorm and project with ``lm_head.T``.
    """
    raise NotImplementedError("Implement deepseek_forward — see 311_deepseek_model/README.md")
