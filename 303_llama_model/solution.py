"""303 — Whole decoder-only Llama (rebuild of llama3.np on stories15M).

Stack the L2 Llama blocks into a full model and emit vocab logits matching llama3.np.
See README.md. Run ``uv run grade 303`` to check your work.

Reuse: ``from leet_llm import llama_decoder_block, rms_norm, triangular_mask, AttnParams,
SwiGLUParams, LlamaBlockParams``. HF weight-layout facts are GIVEN in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    AttnParams,
    LlamaBlockParams,
    SwiGLUParams,
    llama_decoder_block,
    rms_norm,
    triangular_mask,
    embedding,
)


@dataclass(frozen=True)
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_base: float = 10000.0


@dataclass(frozen=True)
class LlamaParams:
    tok_embed: np.ndarray  # (V, d)
    layers: list  # list[LlamaBlockParams] (from leet_llm import LlamaBlockParams)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray  # (V, d)


def load_llama(weights: dict, cfg: LlamaConfig) -> LlamaParams:
    """Map a dict of HF-named arrays (see README table) into LlamaParams."""
    tok_embed = weights["model.embed_tokens.weight"]
    final_norm = weights["model.norm.weight"]
    lm_head = weights["lm_head.weight"]

    layers: list[LlamaBlockParams] = []
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}"
        attn_norm = weights[f"{prefix}.input_layernorm.weight"]
        ffn_norm = weights[f"{prefix}.post_attention_layernorm.weight"]

        Wq = weights[f"{prefix}.self_attn.q_proj.weight"]
        Wk = weights[f"{prefix}.self_attn.k_proj.weight"]
        Wv = weights[f"{prefix}.self_attn.v_proj.weight"]
        Wo = weights[f"{prefix}.self_attn.o_proj.weight"]
        attn = AttnParams(
            Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, bq=None, bk=None, bv=None, bo=None
        )

        W1 = weights[f"{prefix}.mlp.gate_proj.weight"]  # gate
        W3 = weights[f"{prefix}.mlp.up_proj.weight"]  # up
        W2 = weights[f"{prefix}.mlp.down_proj.weight"]  # down
        ffn = SwiGLUParams(W1=W1, W3=W3, W2=W2)

        layers.append(
            LlamaBlockParams(attn=attn, ffn=ffn, attn_norm=attn_norm, ffn_norm=ffn_norm)
        )

    return LlamaParams(
        tok_embed=tok_embed, layers=layers, final_norm=final_norm, lm_head=lm_head
    )


def llama_forward(
    input_ids: np.ndarray, params: LlamaParams, cfg: LlamaConfig, start_pos: int = 0
) -> np.ndarray:
    """Token embed → N Llama blocks (causal, positions start_pos..) → final RMSNorm → lm_head.
    Returns logits (B, L, V)."""
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]
    # start_pos: ignore for now — only used by L4 KV-cache decoding
    positions = np.arange(start_pos, start_pos + L)
    mask = triangular_mask(L)

    for blockParam in params.layers:
        h = llama_decoder_block(h, blockParam, cfg.n_heads, cfg.n_kv_heads,
            positions=positions, mask=mask, eps=cfg.norm_eps)

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T

    return logits
