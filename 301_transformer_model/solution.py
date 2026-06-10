"""301 — Whole Encoder-Decoder Transformer (classic, Vaswani 2017 / Marian).

Assemble the L2 operators into a full encoder-decoder model and produce vocab logits,
matching Hugging Face's ``MarianMTModel``. See README.md.
Run ``uv run grade 301`` to check your work.

Reuse via the facade: ``from leet_llm import (encoder_block, decoder_block, AttnParams,
FFNParams, triangular_mask)``. HuggingFace config/weight facts are GIVEN in the README —
they are framework plumbing, not the puzzle. Your job is the assembly/wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import EncoderBlockParams, DecoderBlockParams, AttnParams, FFNParams
from leet_llm import triangular_mask, embedding, encoder_block, decoder_block


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_heads: int
    n_enc_layers: int
    n_dec_layers: int
    d_ff: int
    vocab_size: int
    max_pos: int
    scale_embedding: bool = False
    eps: float = 1e-5
    pad_id: int = 0
    eos_id: int = 0
    decoder_start_id: int = 0
    activation: str = "gelu"


@dataclass(frozen=True)
class MarianParams:
    enc_embed: np.ndarray  # (V, d)
    dec_embed: np.ndarray  # (V, d)
    enc_pos: np.ndarray  # (P, d) fixed sinusoidal table
    dec_pos: np.ndarray  # (P, d)
    enc_layers: list[EncoderBlockParams]
    dec_layers: list[DecoderBlockParams]
    lm_head: np.ndarray  # (V, d), tied to shared embedding
    final_logits_bias: np.ndarray  # (V,)


def load_marian(weights: dict, cfg: TransformerConfig) -> MarianParams:
    """Map a dict of HF-named arrays (see README table) into MarianParams."""
    marian_params = MarianParams(
        enc_embed=weights["model.encoder.embed_tokens.weight"],
        dec_embed=weights["model.decoder.embed_tokens.weight"],
        enc_pos=weights["model.encoder.embed_positions.weight"],
        dec_pos=weights["model.decoder.embed_positions.weight"],
        enc_layers=[
            EncoderBlockParams(
                attn=AttnParams(
                    Wq=weights[f"model.encoder.layers.{i}.self_attn.q_proj.weight"],
                    bq=weights[f"model.encoder.layers.{i}.self_attn.q_proj.bias"],
                    Wk=weights[f"model.encoder.layers.{i}.self_attn.k_proj.weight"],
                    bk=weights[f"model.encoder.layers.{i}.self_attn.k_proj.bias"],
                    Wv=weights[f"model.encoder.layers.{i}.self_attn.v_proj.weight"],
                    bv=weights[f"model.encoder.layers.{i}.self_attn.v_proj.bias"],
                    Wo=weights[f"model.encoder.layers.{i}.self_attn.out_proj.weight"],
                    bo=weights[f"model.encoder.layers.{i}.self_attn.out_proj.bias"],
                ),
                norm1_gamma=weights[
                    f"model.encoder.layers.{i}.self_attn_layer_norm.weight"
                ],
                norm1_beta=weights[
                    f"model.encoder.layers.{i}.self_attn_layer_norm.bias"
                ],
                ffn=FFNParams(
                    W1=weights[f"model.encoder.layers.{i}.fc1.weight"],
                    b1=weights[f"model.encoder.layers.{i}.fc1.bias"],
                    W2=weights[f"model.encoder.layers.{i}.fc2.weight"],
                    b2=weights[f"model.encoder.layers.{i}.fc2.bias"],
                ),
                norm2_gamma=weights[
                    f"model.encoder.layers.{i}.final_layer_norm.weight"
                ],
                norm2_beta=weights[f"model.encoder.layers.{i}.final_layer_norm.bias"],
            )
            for i in range(cfg.n_enc_layers)
        ],
        dec_layers=[
            DecoderBlockParams(
                self_attn=AttnParams(
                    Wq=weights[f"model.decoder.layers.{i}.self_attn.q_proj.weight"],
                    bq=weights[f"model.decoder.layers.{i}.self_attn.q_proj.bias"],
                    Wk=weights[f"model.decoder.layers.{i}.self_attn.k_proj.weight"],
                    bk=weights[f"model.decoder.layers.{i}.self_attn.k_proj.bias"],
                    Wv=weights[f"model.decoder.layers.{i}.self_attn.v_proj.weight"],
                    bv=weights[f"model.decoder.layers.{i}.self_attn.v_proj.bias"],
                    Wo=weights[f"model.decoder.layers.{i}.self_attn.out_proj.weight"],
                    bo=weights[f"model.decoder.layers.{i}.self_attn.out_proj.bias"],
                ),
                cross_attn=AttnParams(
                    Wq=weights[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"],
                    bq=weights[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"],
                    Wk=weights[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"],
                    bk=weights[f"model.decoder.layers.{i}.encoder_attn.k_proj.bias"],
                    Wv=weights[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"],
                    bv=weights[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"],
                    Wo=weights[
                        f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"
                    ],
                    bo=weights[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"],
                ),
                ffn=FFNParams(
                    W1=weights[f"model.decoder.layers.{i}.fc1.weight"],
                    b1=weights[f"model.decoder.layers.{i}.fc1.bias"],
                    W2=weights[f"model.decoder.layers.{i}.fc2.weight"],
                    b2=weights[f"model.decoder.layers.{i}.fc2.bias"],
                ),
                norm1_gamma=weights[
                    f"model.decoder.layers.{i}.self_attn_layer_norm.weight"
                ],
                norm1_beta=weights[
                    f"model.decoder.layers.{i}.self_attn_layer_norm.bias"
                ],
                norm2_gamma=weights[
                    f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
                ],
                norm2_beta=weights[
                    f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
                ],
                norm3_gamma=weights[
                    f"model.decoder.layers.{i}.final_layer_norm.weight"
                ],
                norm3_beta=weights[f"model.decoder.layers.{i}.final_layer_norm.bias"],
            )
            for i in range(cfg.n_dec_layers)
        ],
        lm_head=weights["lm_head.weight"],
        final_logits_bias=weights["final_logits_bias"].reshape(cfg.vocab_size),
    )
    return marian_params


def encoder(
    src_ids: np.ndarray, params: MarianParams, cfg: TransformerConfig
) -> np.ndarray:
    """Token+positional embed → N post-norm encoder blocks → memory (B, S, d)."""
    emb = embedding(src_ids, params.enc_embed)
    if cfg.scale_embedding:
        emb *= np.sqrt(cfg.d_model)
    L = src_ids.shape[-1]
    h = emb + params.enc_pos[np.arange(L)]

    act = getattr(cfg, "activation", "gelu")
    for i in range(cfg.n_enc_layers):
        h = encoder_block(h, params.enc_layers[i], cfg.n_heads, activation=act)

    return h


def decoder(
    tgt_ids: np.ndarray,
    memory: np.ndarray,
    params: MarianParams,
    cfg: TransformerConfig,
) -> np.ndarray:
    """Causal-masked self-attn + cross-attn over memory → hidden (B, T, d)."""
    emb = embedding(tgt_ids, params.dec_embed)
    if cfg.scale_embedding:
        emb *= np.sqrt(cfg.d_model)
    L = tgt_ids.shape[-1]
    h = emb + params.dec_pos[np.arange(L)]

    causal_mask = triangular_mask(L)
    act = getattr(cfg, "activation", "gelu")
    for i in range(cfg.n_dec_layers):
        h = decoder_block(
            h, memory, params.dec_layers[i], cfg.n_heads, causal_mask, activation=act
        )

    return h


def transformer_logits(
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    params: MarianParams,
    cfg: TransformerConfig,
) -> np.ndarray:
    """Full forward → logits (B, T, V) = decoder(...) @ lm_head.T + final_logits_bias."""
    memory = encoder(src_ids, params, cfg)
    dec_out = decoder(tgt_ids, memory, params, cfg)
    logits = dec_out @ params.lm_head.T + params.final_logits_bias
    return logits
