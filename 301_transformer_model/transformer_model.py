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
    enc_layers: (
        list  # list[EncoderBlockParams] (from leet_llm import EncoderBlockParams)
    )
    dec_layers: list  # list[DecoderBlockParams]
    lm_head: np.ndarray  # (V, d), tied to shared embedding
    final_logits_bias: np.ndarray  # (V,)


def load_marian(weights: dict, cfg: TransformerConfig) -> MarianParams:
    """Map a dict of HF-named arrays (see README table) into MarianParams."""
    raise NotImplementedError(
        "Implement load_marian — see 301_transformer_model/README.md"
    )


def encoder(
    src_ids: np.ndarray, params: MarianParams, cfg: TransformerConfig
) -> np.ndarray:
    """Token+positional embed → N post-norm encoder blocks → memory (B, S, d)."""
    raise NotImplementedError("Implement encoder — see 301_transformer_model/README.md")


def decoder(
    tgt_ids: np.ndarray,
    memory: np.ndarray,
    params: MarianParams,
    cfg: TransformerConfig,
) -> np.ndarray:
    """Causal-masked self-attn + cross-attn over memory → hidden (B, T, d)."""
    raise NotImplementedError("Implement decoder — see 301_transformer_model/README.md")


def transformer_logits(
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    params: MarianParams,
    cfg: TransformerConfig,
) -> np.ndarray:
    """Full forward → logits (B, T, V) = decoder(...) @ lm_head.T + final_logits_bias."""
    raise NotImplementedError(
        "Implement transformer_logits — see 301_transformer_model/README.md"
    )
