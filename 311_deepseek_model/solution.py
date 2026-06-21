"""311 — Multi-head Latent Attention (MLA) + DeepSeek-V3 whole-model forward.

Two tasks in one module (the loader finds exactly one stub .py per folder):

1. ``mla_project`` — the DeepSeek delta operator (low-rank KV with decoupled RoPE).
2. ``DeepseekConfig`` / ``DeepseekParams`` / ``load_deepseek`` / ``deepseek_forward``
   — the full DeepSeek-V3 decoder-only model, composing L2 primitives.

See README.md. Run ``uv run grade 311`` to check your work.

Hints:
- ``mla_project``: reuse ``from leet_llm import rms_norm`` (212), ``rope_half`` (213),
  ``sdpa`` (205). Key steps: (1) KV down-proj → split latent c_kv + shared k_rope;
  (2) c_kv → rms_norm → kv_b_proj → per-head [k_nope, v]; (3) Q from q_proj (or
  q_a_proj→rms_norm→q_b_proj) → per-head [q_nope, q_rope]; (4) apply rope_half ONLY
  to q_rope and the shared k_rope (decoupled slice); (5) broadcast k_rope to n_heads;
  (6) concat [q_nope, q_rope] and [k_nope, k_rope] for full key/query; then sdpa +
  o_proj. Scaling = qk_head_dim**(-0.5) (adjust for mscale if yarn RoPE, see README).
- ``deepseek_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` → MLA →
  ``add_residual`` (208) → ``rms_norm`` → dense SwiGLU (layers < first_k_dense_replace)
  or DeepSeek MoE (layers >= first_k_dense_replace) → residual] → final ``rms_norm``
  → ``@ lm_head.T``. DeepSeek MoE: sigmoid gating + e_score_correction_bias + group
  top-k selection + routed_scaling_factor (reuse ``moe_ffn`` (308) for the expert
  dispatch core) + always-on shared experts added to routed output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def mla_project(
    x: np.ndarray,
    layer: dict,
    cfg: "DeepseekConfig",
    positions: np.ndarray,
) -> np.ndarray:
    """Multi-head Latent Attention (MLA) projection + attention.

    Performs the full MLA attention computation for one layer of DeepSeek-V3:
    low-rank KV compression with decoupled RoPE, then scaled-dot-product attention.

    Parameters
    ----------
    x:
        Input activations, shape ``(B, L, d)``.
    layer:
        Per-layer parameter dict with keys for MLA weights (see ``load_deepseek``).
    cfg:
        ``DeepseekConfig`` holding MLA dimensions and RoPE parameters.
    positions:
        Integer position indices, shape ``(L,)``.

    Returns
    -------
    np.ndarray
        Output shape ``(B, L, d)``.
    """
    raise NotImplementedError("Implement mla_project — see 311_deepseek_model/README.md")


# ---------------------------------------------------------------------------
# DeepSeek-V3 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepseekConfig:
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
    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts (see load_deepseek)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray     # (V, d)


def load_deepseek(weights: dict, cfg: DeepseekConfig) -> DeepseekParams:
    """Map HF-named arrays into DeepseekParams.

    HF weight names (no un-permute — rotate-half layout as-is):
      model.embed_tokens.weight                                          (V, d)
      model.norm.weight                                                  (d,)
      lm_head.weight                                                     (V, d) [absent → embed]

    Per layer (MLA attention):
      model.layers.{i}.input_layernorm.weight                            (d,)
      model.layers.{i}.post_attention_layernorm.weight                   (d,)
      model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight              (kv_lora_rank+qk_rope_head_dim, d)
      model.layers.{i}.self_attn.kv_a_layernorm.weight                  (kv_lora_rank,)
      model.layers.{i}.self_attn.kv_b_proj.weight                       (n_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank)
      model.layers.{i}.self_attn.o_proj.weight                          (d, n_heads*v_head_dim)
      if q_lora_rank is None:
        model.layers.{i}.self_attn.q_proj.weight                        (n_heads*qk_head_dim, d)
      else:
        model.layers.{i}.self_attn.q_a_proj.weight                      (q_lora_rank, d)
        model.layers.{i}.self_attn.q_a_layernorm.weight                 (q_lora_rank,)
        model.layers.{i}.self_attn.q_b_proj.weight                      (n_heads*qk_head_dim, q_lora_rank)

    Dense layers (layer index < first_k_dense_replace):
      model.layers.{i}.mlp.gate_proj.weight                             (intermediate_size, d)
      model.layers.{i}.mlp.up_proj.weight                               (intermediate_size, d)
      model.layers.{i}.mlp.down_proj.weight                             (d, intermediate_size)

    MoE layers (layer index >= first_k_dense_replace):
      model.layers.{i}.mlp.gate.weight                                  (n_routed_experts, d)
      model.layers.{i}.mlp.gate.e_score_correction_bias                 (n_routed_experts,)
      model.layers.{i}.mlp.experts.gate_up_proj                        (n_routed_experts, 2*moe_intermediate_size, d)
      model.layers.{i}.mlp.experts.down_proj                            (n_routed_experts, d, moe_intermediate_size)
      model.layers.{i}.mlp.shared_experts.gate_proj.weight              (n_shared_experts*moe_intermediate_size, d)
      model.layers.{i}.mlp.shared_experts.up_proj.weight                (n_shared_experts*moe_intermediate_size, d)
      model.layers.{i}.mlp.shared_experts.down_proj.weight              (d, n_shared_experts*moe_intermediate_size)
    """
    raise NotImplementedError("Implement load_deepseek — see 311_deepseek_model/README.md")


def deepseek_forward(
    input_ids: np.ndarray,
    params: DeepseekParams,
    cfg: DeepseekConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Token embed → N DeepSeek-V3 blocks (causal) → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    DeepSeek-V3 = Llama assembly with MLA replacing GQA and a mixed dense/MoE FFN:
      ``embedding`` → per layer [``rms_norm`` → **``mla_project``** → ``add_residual`` →
      ``rms_norm`` → (dense SwiGLU if layer < first_k_dense_replace, else DeepSeek MoE)
      → residual] → final ``rms_norm`` → ``@ lm_head.T``.

    DeepSeek MoE = sigmoid gating + additive e_score_correction_bias + group top-k +
    ``moe_ffn`` (308) for expert dispatch + always-on shared experts (``swiglu_ffn``).
    """
    raise NotImplementedError("Implement deepseek_forward — see 311_deepseek_model/README.md")
