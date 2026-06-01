# 213 — RoPE (Rotary Position Embedding)

**Level 2 · Operators & Layers**

## Description

The Llama upgrade to additive positional encoding (204). Instead of *adding* a position
signal, **RoPE** *rotates* each query and key vector by an angle proportional to its
position. Because a dot product between two rotated vectors depends only on the *difference*
of their angles, attention scores become naturally **relative**.

There are **two conventions** for *which* coordinates get paired into each 2-D rotation —
they implement the same idea but lay the numbers out differently, so they are not
interchangeable on the same weights:

- **Interleaved** (Meta / original Llama / `llama3.np`): rotate **adjacent pairs**
  `(x₀,x₁), (x₂,x₃), …`
- **Rotate-half** (HuggingFace / most of the ecosystem): rotate a coordinate against its
  partner **half a vector away** — pair `(xᵢ, x_{i+d/2})`.

You'll implement **both**, plus a checker for RoPE's defining property. (The capstone L3
model uses the **interleaved** form; the two are related by permuting the q/k weights.)

## The Math

For head dim `d` (even), base `θ=10000`, frequency index `i ∈ {0,…,d/2−1}`:

```
inv_freqᵢ = θ^(−2i/d)
angle(p, i) = p · inv_freqᵢ           # position p
```

**Interleaved** — pair `(x_{2i}, x_{2i+1})` rotated by `angle(p,i)`:
```
out_{2i}   = x_{2i}·cos − x_{2i+1}·sin
out_{2i+1} = x_{2i}·sin + x_{2i+1}·cos
```

**Rotate-half** — build `cos,sin` of width `d` by tiling `[angle, angle]`, split
`x = [x₁ | x₂]` into halves, and:
```
rotate_half(x) = [ −x₂, x₁ ]
out = x ⊙ cos + rotate_half(x) ⊙ sin
```

**The defining property (the checker).** RoPE is an orthogonal rotation `Rₚ`, and rotations
compose as `Rₘᵀ Rₙ = R₍ₙ₋ₘ₎`. Therefore
```
⟨RoPE(q, m), RoPE(k, n)⟩ = ⟨q, RoPE(k, n−m)⟩
```
i.e. the rotated dot product depends only on `q`, `k`, and the **relative** position
`(n−m)` — it is unchanged if you shift both positions by the same amount, and at `m=n` it
recovers the plain `⟨q, k⟩`. (Note it is *not* a function of `⟨q,k⟩` alone.) `rope_qk_dot`
computes `⟨RoPE(q,m), RoPE(k,n)⟩` (interleaved) so the tests can check this.

## Function Signature

```python
def rope_interleaved(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray: ...
def rope_half(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray: ...
#   x: (..., L, d) with d even (the L axis is -2)   positions: (L,) int   -> (..., L, d)

def rope_qk_dot(q: np.ndarray, k: np.ndarray, m: int, n: int, base: float = 10000.0) -> np.ndarray: ...
#   q, k: (..., d)   ->   <RoPE(q,m), RoPE(k,n)>  over the last axis  (interleaved)
```

## Read More

- *RoFormer*, Su et al. 2021: https://arxiv.org/abs/2104.09864
- Interleaved layout: Meta's `llama` / `llama3.np` (`apply_rotary_emb`, complex form).
- Rotate-half layout: HuggingFace `transformers` (`rotate_half`, `apply_rotary_pos_emb`).
- You may reuse `from leet_llm import interleave, deinterleave, split_halves, join_halves` (011).

## How to Test

```bash
uv run grade 213
```
