# 204 — Sinusoidal Positional Encoding

**Level 2 · Operators & Layers**

## Description

Attention is **permutation-invariant** — on its own it can't tell position 0 from position
5. The original transformer fixes this by *adding* a fixed sinusoidal signal to the token
embeddings, giving each position a unique, smoothly-varying fingerprint. (A learned
positional embedding — task 201's lookup, indexed by position instead of token — is the
GPT-2 alternative; see Read More.)

## The Math

For position `pos ∈ {0,…,L−1}` and embedding dimension index `i ∈ {0,…,d/2−1}`:

```
PE[pos, 2i]   = sin( pos / 10000^{2i/d} )
PE[pos, 2i+1] = cos( pos / 10000^{2i/d} )
```

The result is a fixed `(L, d)` matrix added to the embeddings — no learned parameters.

## Function Signature

```python
def sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray: ...
#   ->   (seq_len, dim)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.5 (positional encoding): https://arxiv.org/abs/1706.03762
- Learned positional embeddings (GPT-2): radford et al. 2019, *Language Models are Unsupervised Multitask Learners*.

## How to Test

```bash
uv run grade 204
```
