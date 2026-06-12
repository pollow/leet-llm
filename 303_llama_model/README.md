# 303 — Whole Decoder-Only Llama (stories15M / llama3.np rebuild)

**Level 3 · Whole-Model & Inference**

## Description

Rebuild `llama3.np`'s decoder-only Llama into a single end-to-end forward that emits
vocabulary logits. The L2 building blocks — `llama_decoder_block` (216), `rms_norm` (212),
`triangular_mask` (009), and the `AttnParams` / `SwiGLUParams` / `LlamaBlockParams`
dataclasses (206 / 214 / 216) — all exist; your job is the wiring plus mapping a flat dict
of HF-named weight arrays into those dataclasses. The fixture oracle is a tiny composed
interleaved-RoPE Llama, and the real capstone target is Karpathy's **stories15M**
checkpoint (the same one `llama3.np` runs), so the weight-loading table below is the
ground truth.

## The architecture (problem spec)

A decoder-only stack: token embedding → `n_layers` pre-norm Llama blocks → final RMSNorm →
LM head.

```
h = tok_embed[input_ids]                              # (B, L, d)
positions = arange(start_pos, start_pos + L)
mask = triangular_mask(L)                             # causal
for block in layers:
    h = llama_decoder_block(h, block, cfg.n_heads, cfg.n_kv_heads,
                            positions=positions, mask=mask, eps=cfg.norm_eps)
h = rms_norm(h, final_norm, cfg.norm_eps)
logits = h @ lm_head.T                                # (B, L, V)
```

Key facts:

- **Interleaved RoPE** — stories15M / llama3.np use the INTERLEAVED (Meta) RoPE convention
  (the one task 213 / 216 implement), *not* HF's rotate-half. Because `llama_decoder_block`
  already applies interleaved RoPE internally, you just pass `positions`.
- **eps = 1e-6** — Llama's RMSNorm epsilon (note: smaller than the L2 default of `1e-5`,
  so always pass `cfg.norm_eps` explicitly).
- **lm_head NOT tied** — stories15M ships a separate `lm_head.weight`; do not reuse the
  token embedding for the output projection.
- **`n_kv_heads == n_heads` for stories15M** — GQA degenerates to plain MHA here, but keep
  both config fields so the block's GQA path stays correct for models where they differ.
- **No biases** — every projection in a Llama block is bias-free (`AttnParams` biases stay
  `None`).

## HF config & weight layout (GIVEN — framework facts)

The checkpoint stores weights as a flat dict with the HF keys below. Map each key directly
into the corresponding slot — **no transposition** is needed, because HF `nn.Linear` stores
weights as `(out, in)` and all L2 operators apply `x @ W.T`.

| HF key | slot |
|---|---|
| `model.embed_tokens.weight` `(V, d)` | `tok_embed` |
| `model.layers.{i}.input_layernorm.weight` `(d,)` | `layers[i].attn_norm` |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `layers[i].attn` → `AttnParams(Wq=q, Wk=k, Wv=v, Wo=o)` (bias-free) |
| `model.layers.{i}.post_attention_layernorm.weight` `(d,)` | `layers[i].ffn_norm` |
| `model.layers.{i}.mlp.gate_proj.weight` `(F, d)` | `layers[i].ffn.W1` (gate) |
| `model.layers.{i}.mlp.up_proj.weight` `(F, d)` | `layers[i].ffn.W3` (up) |
| `model.layers.{i}.mlp.down_proj.weight` `(d, F)` | `layers[i].ffn.W2` (down) |
| `model.norm.weight` `(d,)` | `final_norm` |
| `lm_head.weight` `(V, d)` | `lm_head` (NOT tied in stories15M) |

Dataclass field reminders (verify against the L2 stubs you reuse):

- `AttnParams(Wq, Wk, Wv, Wo, bq=None, bk=None, bv=None, bo=None)` — build it bias-free.
- `SwiGLUParams(W1, W3, W2)` — `W1` is the **gate** proj, `W3` the **up** proj, `W2` the
  **down** proj (NOT in numeric order).
- `LlamaBlockParams(attn, ffn, attn_norm, ffn_norm)` — `attn` is an `AttnParams`, `ffn` is a
  `SwiGLUParams`, the two norms are bare `(d,)` RMSNorm weight vectors.

## Read More

- *LLaMA: Open and Efficient Foundation Language Models*, Touvron et al. 2023:
  https://arxiv.org/abs/2302.13971
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE), Su et al. 2021:
  https://arxiv.org/abs/2104.09864
- `llama3.np` — minimal NumPy Llama 3 inference: https://github.com/likejazz/llama3.np
- *GLU Variants Improve Transformer* (SwiGLU), Shazeer 2020: https://arxiv.org/abs/2002.05202
- Reuse `from leet_llm import (llama_decoder_block, rms_norm, triangular_mask, AttnParams,
  SwiGLUParams, LlamaBlockParams)`.

## Function Signature

```python
@dataclass(frozen=True)
class LlamaConfig:
    dim: int; n_layers: int; n_heads: int; n_kv_heads: int; vocab_size: int
    max_seq_len: int = 2048; norm_eps: float = 1e-6; rope_base: float = 10000.0

@dataclass(frozen=True)
class LlamaParams:
    tok_embed: np.ndarray       # (V, d)
    layers: list                # list[LlamaBlockParams]
    final_norm: np.ndarray      # (d,) RMSNorm weight
    lm_head: np.ndarray         # (V, d)

def load_llama(weights: dict, cfg: LlamaConfig) -> LlamaParams: ...
#   Map HF-named arrays (see table above) into LlamaParams.

def llama_forward(input_ids: np.ndarray, params: LlamaParams, cfg: LlamaConfig,
                  start_pos: int = 0) -> np.ndarray: ...
#   input_ids: (B, L)   ->   logits: (B, L, V)
#   start_pos: ignore for now — only used by L4 KV-cache decoding
```

## How to Test

```bash
uv run grade 303
```

The real-weight parity test (`test_real_stories15m_matches_llama3np`) compares your
`llama_forward` against the `llama3.np` reference on the actual stories15M checkpoint. It
runs automatically if the `../../llama3.np` sibling repo is present (it provides both the
weights and the reference `Llama` class). The weights themselves are fetched by the **304**
capstone's downloader (shared across both Track-B tasks):

```bash
bash 304_generate/download.sh
```
