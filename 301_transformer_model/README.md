# 301 — Whole Encoder-Decoder Transformer (opus-mt assembly)

**Level 3 · Whole-Model & Inference**

## Description

Assemble the classic encoder-decoder Transformer (Vaswani et al. 2017) into a single
end-to-end forward that emits vocabulary logits. The L2 building blocks — `encoder_block`
(209), `decoder_block` (210), `mha` (206), `ffn` (207), `layer_norm` (203),
`triangular_mask` (009) — all exist; your job is the wiring. The fixture oracle and the
real capstone target are both `Helsinki-NLP/opus-mt-en-zh` (a `MarianMTModel`), so the
weight-loading table below is the ground truth.

## The architecture (problem spec)

**Encoder** — embed the source tokens, then stack `n_enc_layers` post-norm encoder blocks:

```
embed = embed_tokens[src_ids] * embed_scale + embed_positions[0:S]
for each encoder block:
    h = LayerNorm₁( h + MHA(h, h) )   # bidirectional self-attention, no mask
    h = LayerNorm₂( h + FFN(h) )
memory = h                              # shape (B, S, d)
```

**Decoder** — embed the target tokens, apply a causal mask, then stack `n_dec_layers`
post-norm decoder blocks (causal self-attention, then cross-attention over the encoder
memory):

```
h = embed_tokens[tgt_ids] * embed_scale + embed_positions[0:T]
causal_mask = triangular_mask(T)
for each decoder block:
    h = LayerNorm₁( h + MHA(h, h, causal_mask) )   # masked self-attention
    h = LayerNorm₂( h + MHA(h, memory) )           # cross-attention: q=h, kv=memory
    h = LayerNorm₃( h + FFN(h) )
dec_out = h                                          # shape (B, T, d)
```

**Logits** — project the decoder output through the tied LM head and add the bias:

```
logits = dec_out @ lm_head.T + final_logits_bias    # shape (B, T, V)
```

This is **post-norm** (LayerNorm after residual addition). Biases are present on all four
linear projections in every attention sublayer (`q/k/v/out`). There is **no** final norm
applied to the encoder or decoder output before the next stage.

## HF config & weight layout (GIVEN — framework facts)

The HF `MarianMTModel` stores weights as a flat dict with the keys below. Map each key
directly into the `MarianParams` slot shown — no transposition is needed, because HF
`nn.Linear` stores weights as `(out, in)` and all L2 operators use `x @ W.T`.

| HF state-dict key | `MarianParams` slot |
|---|---|
| `model.shared.weight` / `model.encoder.embed_tokens.weight` `(V, d)` | `enc_embed` |
| `model.decoder.embed_tokens.weight` `(V, d)` | `dec_embed` |
| `model.encoder.embed_positions.weight` `(P, d)` | `enc_pos` |
| `model.decoder.embed_positions.weight` `(P, d)` | `dec_pos` |
| `model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}` | `enc_layers[i].attn` → `AttnParams(Wq=q.weight, bq=q.bias, …, Wo=out.weight, bo=out.bias)` |
| `model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}` | `enc_layers[i].norm1_gamma` / `norm1_beta` |
| `model.encoder.layers.{i}.fc1.{weight,bias}` `(F, d)` / `(F,)` | `enc_layers[i].ffn.W1` / `b1` |
| `model.encoder.layers.{i}.fc2.{weight,bias}` `(d, F)` / `(d,)` | `enc_layers[i].ffn.W2` / `b2` |
| `model.encoder.layers.{i}.final_layer_norm.{weight,bias}` | `enc_layers[i].norm2_gamma` / `norm2_beta` |
| `model.decoder.layers.{i}.self_attn.*` | `dec_layers[i].self_attn` (+ `norm1` = `self_attn_layer_norm`) |
| `model.decoder.layers.{i}.encoder_attn.*` | `dec_layers[i].cross_attn` (+ `norm2` = `encoder_attn_layer_norm`) |
| `model.decoder.layers.{i}.fc1/fc2` (+ `norm3` = `final_layer_norm`) | `dec_layers[i].ffn` |
| `lm_head.weight` `(V, d)` | `lm_head` |
| `final_logits_bias` `(1, V)` | `final_logits_bias` — reshape to `(V,)` |

Additional GIVEN facts:

- **No transpose**: all weights map directly (`W` not `W.T`) because HF uses `(out, in)` and
  L2 operators do `x @ W.T`.
- **Embedding scale**: `embed_scale = sqrt(d_model) if cfg.scale_embedding else 1.0`.
  Marian sets `scale_embedding=True`.
- **Positional embeddings**: a fixed sinusoidal table stored in the checkpoint; you simply
  gather rows by index — `pos_table[arange(L)]`. This differs from task 204
  (`sinusoidal_pe`), which computes the table on the fly.
- **Tied embeddings**: `model.shared.weight` is the same tensor as `model.encoder.embed_tokens.weight`;
  `lm_head.weight` is tied to the shared embedding. The fixture stores them as separate
  keys.
- **`final_logits_bias`**: a learned `(1, V)` bias added after the LM-head projection;
  reshape it to `(V,)` before storing in `MarianParams`.
- **Special-token ids** (`TransformerConfig`): `pad_id=63`, `eos_id=0`,
  `decoder_start_id=63` for the tiny fixture (match the real checkpoint's HF config for the
  live capstone).

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017: https://arxiv.org/abs/1706.03762
- HF `MarianMTModel` documentation: https://huggingface.co/docs/transformers/model_doc/marian
- OPUS-MT model card (`Helsinki-NLP/opus-mt-en-zh`): https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
- Reuse `from leet_llm import (encoder_block, decoder_block, AttnParams, FFNParams,
  triangular_mask, EncoderBlockParams, DecoderBlockParams)`.

## Function Signature

```python
@dataclass(frozen=True)
class TransformerConfig:
    d_model: int; n_heads: int; n_enc_layers: int; n_dec_layers: int
    d_ff: int; vocab_size: int; max_pos: int
    scale_embedding: bool = False; eps: float = 1e-5
    pad_id: int = 0; eos_id: int = 0; decoder_start_id: int = 0
    activation: str = "gelu"   # "gelu" for Vaswani original, "swish"/"silu" for MarianMT

@dataclass(frozen=True)
class MarianParams:
    enc_embed: np.ndarray       # (V, d)
    dec_embed: np.ndarray       # (V, d)
    enc_pos: np.ndarray         # (P, d) fixed sinusoidal table
    dec_pos: np.ndarray         # (P, d)
    enc_layers: list            # list[EncoderBlockParams]
    dec_layers: list            # list[DecoderBlockParams]
    lm_head: np.ndarray         # (V, d), tied to shared embedding
    final_logits_bias: np.ndarray  # (V,)

def load_marian(weights: dict, cfg: TransformerConfig) -> MarianParams: ...
#   Map HF-named arrays (see table above) into MarianParams.

def encoder(src_ids: np.ndarray, params: MarianParams, cfg: TransformerConfig) -> np.ndarray: ...
#   src_ids: (B, S)   ->   memory: (B, S, d)

def decoder(tgt_ids: np.ndarray, memory: np.ndarray, params: MarianParams,
            cfg: TransformerConfig) -> np.ndarray: ...
#   tgt_ids: (B, T)   memory: (B, S, d)   ->   hidden: (B, T, d)

def transformer_logits(src_ids: np.ndarray, tgt_ids: np.ndarray, params: MarianParams,
                       cfg: TransformerConfig) -> np.ndarray: ...
#   ->   logits: (B, T, V)
```

## How to Test

```bash
uv run grade 301
```
