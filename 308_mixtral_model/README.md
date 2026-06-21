# 308 — Sparse MoE FFN + Mixtral Whole-Model Forward

## Description

Mixtral is a Llama-style decoder-only transformer where each layer's feed-forward
network is replaced by a **sparse mixture-of-experts (MoE)** block.  Instead of a
single SwiGLU FFN, each layer holds `num_local_experts` independent SwiGLU expert
networks plus a small linear *router*.  For each token, the router selects the
top-`k` experts, renormalises their gate scores, and returns a weighted sum of those
experts' outputs.  The attention mechanism (GQA, rotate-half RoPE, no QKV bias) is
identical to Llama/Mistral.

This task introduces one new operator:

- **`moe_ffn`** — sparse top-k mixture-of-experts FFN (the Mixtral delta).

Everything else composes L2 primitives you have already built.

## The Math

### Router and top-k gating

Let `x ∈ R^(T, d)` be the flattened token activations (B×L tokens, each of dim d),
and `W_r ∈ R^(E, d)` be the router weight matrix (`E` = `num_local_experts`).

```
router_logits   = x @ W_r.T              # (T, E)
routing_weights = softmax(router_logits) # softmax over ALL E experts
weights, idx    = top_k(routing_weights, k=num_experts_per_tok)
                                         # weights: (T, k),  idx: (T, k)
weights         = weights / Σ_k weights  # renormalise selected k weights to sum=1
```

### Expert dispatch and accumulation

Each expert `e` is a SwiGLU FFN parameterised by `SwiGLUParams(W1, W3, W2)`:

```
expert_out_e(x_t) = swiglu_ffn(x_t, experts[e])
                  = (SiLU(x_t @ W1.T) * (x_t @ W3.T)) @ W2.T
```

The MoE output for token `t`:

```
moe_ffn(x_t) = Σ_{k=0}^{K-1}  weights[t, k] · expert_out_{idx[t,k]}(x_t)
```

### HF weight layout

The HF checkpoint stores expert weights in compact 3-D tensors:

| HF weight name | shape | meaning |
|---|---|---|
| `model.layers.{i}.mlp.gate.weight` | `(E, d)` | router W_r |
| `model.layers.{i}.mlp.experts.gate_up_proj` | `(E, 2·Fd, d)` | `[W1; W3]` stacked for all experts |
| `model.layers.{i}.mlp.experts.down_proj` | `(E, d, Fd)` | `W2` for all experts |

To recover per-expert SwiGLU weights from `gate_up_proj[e]` of shape `(2·Fd, d)`:
`W1 = gate_up_proj[e, :Fd, :]` (gate) and `W3 = gate_up_proj[e, Fd:, :]` (up).

### Full Mixtral block

```
h ← embedding(input_ids)
for each layer i:
    a   = rms_norm(h, attn_norm)
    q,k,v = affine(a, Wq), affine(a, Wk), affine(a, Wv)   # no bias
    q,k = split into heads, apply rope_half (rotate-half RoPE)
    k,v = repeat_kv (GQA)
    o   = sdpa(q, k, v, causal_mask)
    h   = add_residual(h, merge_heads(o) @ Wo.T)
    f   = rms_norm(h, ffn_norm)
    h   = h + moe_ffn(f, router_weight, experts, num_experts_per_tok)
h = rms_norm(h, final_norm)
logits = h @ lm_head.T
```

## Function Signature

```python
def moe_ffn(
    x: np.ndarray,              # (T, d) tokens × dim
    router_weight: np.ndarray,  # (num_experts, d)
    experts: list,              # list[SwiGLUParams], length num_experts
    top_k: int,
) -> np.ndarray:                # (T, d) same shape as x
```

```python
@dataclass(frozen=True)
class MixtralConfig:
    dim: int; n_layers: int; n_heads: int; n_kv_heads: int
    vocab_size: int; num_local_experts: int; num_experts_per_tok: int
    max_seq_len: int = 4096; norm_eps: float = 1e-5; rope_base: float = 10000.0

def load_mixtral(weights: dict, cfg: MixtralConfig) -> MixtralParams: ...
def mixtral_forward(input_ids, params, cfg, start_pos=0) -> np.ndarray: ...
```

## Read More

- Mixtral paper: <https://arxiv.org/abs/2401.04088>
- HF Mixtral source: `transformers/models/mixtral/modeling_mixtral.py`
- Task 214 (`swiglu_ffn`, `SwiGLUParams`) — the expert building block
- Task 215 (`gqa`) — grouped-query attention used in attention layers
- Task 213 (`rope_half`) — rotate-half RoPE (no un-permute)
- Task 005 (`softmax`), 007 (`top_k`) — used in the router

## How to Test

```bash
uv run grade 308
```

To download the real tiny-random checkpoint and run the extended parity test:

```bash
308_mixtral_model/download.sh
uv run grade 308
```
