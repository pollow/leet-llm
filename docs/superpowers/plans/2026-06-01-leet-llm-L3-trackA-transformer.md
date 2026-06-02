# L3 Track A — Classic Encoder-Decoder Transformer (`opus-mt-en-zh`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author the L3 Track A task scaffolds — a prerequisite L2 attention-bias extension, then task **301 `transformer_model`** (build a whole encoder-decoder forward) and task **302 `translate`** (greedy decode + the real en→zh capstone) — so a learner can rebuild the classic Transformer and translate with `Helsinki-NLP/opus-mt-en-zh`.

**Architecture:** Pure-functional assembly over the L2 operators, imported by name via the `leet_llm` facade. The graded oracle is a **tiny genuine `MarianMTModel`** (random init, float64) whose per-stage activations are frozen into committed `.npz` fixtures — fully hermetic, no weight download. A separate `@skipif`-gated test runs the learner's code on the **real** downloaded checkpoint and matches HF greedy token ids. Decoding is stateless recompute (KV-cache is L4); greedy only.

**Tech Stack:** Python 3.11+, NumPy 2.x (runtime), `uv`. Authoring-only `gen` group: `torch` (CPU) + `transformers` — used by `gen_fixtures.py`/`convert.py`, never at grade time.

**Authoring conventions (match L2 exactly):**
- Each task folder `3NN_slug/` holds: `README.md`, the learner stub `slug.py`, an empty `solution.py` (`NotImplementedError`, identical to the stub — **no reference shipped**), and `tests/` (`test_slug.py`, `gen_fixtures.py`, `fixtures/*.npz`).
- Tests start with `from leet_llm.grader import load` / `_m = load(__file__)`.
- Fixtures regenerate via `uv run --group gen python 3NN_slug/tests/gen_fixtures.py`; oracle runs in float64 (`.double()`), tolerances `rtol=1e-9, atol=1e-9`.
- Reusable public names get an entry in `leet_llm/_registry.py`.
- **Validation-before-ship (every task):** to prove the fixtures + tests are correct without shipping a reference, temporarily paste a correct implementation into `solution.py`, run `LEET_LLM_TARGET=solution uv run grade 3NN` until green, then **restore `solution.py` to the `NotImplementedError` stub** and confirm `uv run grade 3NN` fails cleanly. The reference is never committed.

---

## Scope

This plan is **Track A only** (opus-mt encoder-decoder), a self-contained working deliverable. **Track B** (Llama / `stories15M`, tasks 303–304) and **Track C** (OSS zoo) get follow-on plans. Spec: `docs/superpowers/specs/2026-06-01-leet-llm-L3-whole-model-inference-design.md`.

## File structure

```
206_multi_head_attention/                 # MODIFIED (Task 0)
├── multi_head_attention.py               #   AttnParams gains optional bq/bk/bv/bo
├── solution.py                           #   kept identical (NotImplementedError)
├── README.md                             #   note optional biases
└── tests/{gen_fixtures.py,test_multi_head_attention.py}  # biased case + bias-aware _params
215_gqa/                                  # MODIFIED (Task 0)
├── gqa.py / README.md                    #   note biases flow via AttnParams
└── tests/{gen_fixtures.py,test_gqa.py}   #   bias-aware _params (reuses 206 AttnParams)
301_transformer_model/                    # CREATE (Task 1)
├── README.md
├── transformer_model.py                  #   TransformerConfig, MarianParams, load_marian, encoder, decoder, transformer_logits
├── solution.py                           #   NotImplementedError stub (== learner stub)
└── tests/{gen_fixtures.py, test_transformer_model.py, fixtures/*.npz}
302_translate/                            # CREATE (Task 2)
├── README.md
├── translate.py                          #   translate() greedy loop
├── solution.py
├── download.sh                           #   fetch Helsinki-NLP/opus-mt-en-zh
├── convert.py                            #   HF checkpoint -> opus_mt_en_zh.npz (gen group)
└── tests/{gen_fixtures.py, test_translate.py, fixtures/*.npz}
leet_llm/_registry.py                     # MODIFIED: add 301/302 public names
README.md                                 # MODIFIED: L3 progress row
```

---

## Task 0: Extend L2 attention with optional biases

**Files:**
- Modify: `206_multi_head_attention/multi_head_attention.py`
- Modify: `206_multi_head_attention/solution.py`
- Modify: `206_multi_head_attention/tests/gen_fixtures.py`
- Modify: `206_multi_head_attention/tests/test_multi_head_attention.py`
- Modify: `206_multi_head_attention/README.md`
- Modify: `215_gqa/tests/gen_fixtures.py`, `215_gqa/tests/test_gqa.py`, `215_gqa/README.md`

Rationale: the classic Transformer (and `opus-mt`) use biases on q/k/v/out projections; L2's `AttnParams` was bias-free. Add optional `bq/bk/bv/bo` (default `None` ⇒ unchanged Llama behavior).

- [ ] **Step 1: Add optional bias fields to `AttnParams`** (in BOTH `multi_head_attention.py` and `solution.py`, keep them identical)

Replace the `AttnParams` dataclass with:

```python
@dataclass(frozen=True)
class AttnParams:
    """Projection weights for attention.

    ``Wq/Wk/Wv/Wo`` are ``(out, in)`` matrices applied as ``x @ W.T`` (reuse L0 003 affine).
    The four biases are optional: ``None`` ⇒ bias-free (Llama / GQA). The classic
    Transformer and GPT-2-style models pass real biases (L3 opus-mt capstone).
    """

    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    bq: np.ndarray | None = None
    bk: np.ndarray | None = None
    bv: np.ndarray | None = None
    bo: np.ndarray | None = None
```

Update the `mha` docstring to add: *"Apply the optional `params.bq/bk/bv/bo` to the q/k/v/out projections when present (treat `None` as zero)."* Do NOT change the `mha`/`gqa` function signatures — biases travel inside `params`. (`solution.py` stays `NotImplementedError`.)

- [ ] **Step 2: Add a biased oracle case to 206 `gen_fixtures.py`**

Change `_mha_ref` to accept biases and a helper, then add a biased case:

```python
def _mha_ref(x_q, Wq, Wk, Wv, Wo, n_heads, x_kv=None, mask=None,
             bq=None, bk=None, bv=None, bo=None):
    xq = torch.from_numpy(x_q)
    xkv = torch.from_numpy(x_q if x_kv is None else x_kv)
    Wq, Wk, Wv, Wo = (torch.from_numpy(w) for w in (Wq, Wk, Wv, Wo))
    tb = lambda b: None if b is None else torch.from_numpy(b)
    qh = _split(F.linear(xq, Wq, tb(bq)), n_heads)
    kh = _split(F.linear(xkv, Wk, tb(bk)), n_heads)
    vh = _split(F.linear(xkv, Wv, tb(bv)), n_heads)
    attn_mask = None if mask is None else torch.from_numpy(np.where(mask, -np.inf, 0.0))
    oh = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
    return F.linear(_merge(oh), Wo, tb(bo)).numpy()
```

In `main()`, after the existing three cases, add a biased self-attention case:

```python
    # biased self-attention (classic Transformer / opus-mt path)
    Wq, Wk, Wv, Wo = Wset()
    bq, bk, bv, bo = (rng.standard_normal(d) for _ in range(4))
    x_b = rng.standard_normal((2, 4, d))
    out_b = _mha_ref(x_b, Wq, Wk, Wv, Wo, 2, bq=bq, bk=bk, bv=bv, bo=bo)
    np.savez(FIX / "self_biased.npz", x_q=x_b, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
             bq=bq, bk=bk, bv=bv, bo=bo, n_heads=np.array(2), out=out_b)
    print("  wrote self_biased.npz")
```

- [ ] **Step 3: Make the 206 test pass biases through `_params`**

Replace `_params` in `test_multi_head_attention.py`:

```python
def _params(d):
    opt = {k: d[k] for k in ("bq", "bk", "bv", "bo") if k in d.files}
    return AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"], **opt)
```

The existing parametrized `test_matches_torch_fixture` now also covers `self_biased.npz` automatically (glob).

- [ ] **Step 4: Mirror bias-awareness in 215 gqa test** (GQA itself stays bias-free in practice, but `_params` must not break if a biased fixture is added later)

In `215_gqa/tests/test_gqa.py`, replace the inline `AttnParams(...)` in `test_matches_torch_fixture` with the same optional-bias construction:

```python
    opt = {k: d[k] for k in ("bq", "bk", "bv", "bo") if k in d.files}
    p = AttnParams(Wq=d["Wq"], Wk=d["Wk"], Wv=d["Wv"], Wo=d["Wo"], **opt)
```

(No new gqa fixture needed; this only future-proofs the loader.)

- [ ] **Step 5: Regenerate 206 fixtures**

Run: `uv run --group gen python 206_multi_head_attention/tests/gen_fixtures.py`
Expected: prints `wrote self_nomask.npz … wrote self_biased.npz`; `tests/fixtures/self_biased.npz` exists.

- [ ] **Step 6: Validate the extension (temporary reference)**

Paste a correct `mha` (and `AttnParams`) into `206_multi_head_attention/solution.py` — split heads (L0 001), per-head `sdpa` (205), `affine` with bias (`x @ W.T + b`, `b=0` when `None`), merge, output proj. Then:

Run: `LEET_LLM_TARGET=solution uv run grade 206`
Expected: PASS (all fixtures incl. `self_biased`).

Then **restore `solution.py`** to the exact `NotImplementedError` stub (identical to `multi_head_attention.py`).

Run: `uv run grade 206`
Expected: FAIL with `NotImplementedError` (clean unsolved state).

- [ ] **Step 7: Update READMEs**

In `206_.../README.md` and `215_.../README.md`, add one sentence under the signature: *"`AttnParams` carries optional `bq/bk/bv/bo` biases (default `None` = bias-free, as Llama uses); the classic Transformer passes real biases."* No method recipe.

- [ ] **Step 8: Commit**

```bash
git add 206_multi_head_attention 215_gqa
git commit -m "L2: AttnParams gains optional q/k/v/out biases (classic-transformer path)"
```

---

## Task 1: `301_transformer_model` — build the encoder-decoder forward

**Files:**
- Create: `301_transformer_model/transformer_model.py`
- Create: `301_transformer_model/solution.py`
- Create: `301_transformer_model/README.md`
- Create: `301_transformer_model/tests/gen_fixtures.py`
- Create: `301_transformer_model/tests/test_transformer_model.py`
- Modify: `leet_llm/_registry.py`

**Marian → our params map (ground truth from a tiny `MarianMTModel`):**

| HF state-dict key | Our slot |
|---|---|
| `model.shared.weight` / `model.encoder.embed_tokens.weight` `(V,d)` | `enc_embed` |
| `model.decoder.embed_tokens.weight` `(V,d)` | `dec_embed` |
| `model.encoder.embed_positions.weight` `(P,d)` (fixed sinusoidal table) | `enc_pos` |
| `model.decoder.embed_positions.weight` `(P,d)` | `dec_pos` |
| `…encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}` | `enc_layers[i].attn` `AttnParams(Wq=q.weight,…,bq=q.bias,…,Wo=out.weight,bo=out.bias)` |
| `…encoder.layers.{i}.self_attn_layer_norm.{weight,bias}` | `enc_layers[i].norm1_gamma/beta` |
| `…encoder.layers.{i}.fc1.{weight,bias}` `(F,d)/(F,)` | `enc_layers[i].ffn.W1/b1` |
| `…encoder.layers.{i}.fc2.{weight,bias}` `(d,F)/(d,)` | `enc_layers[i].ffn.W2/b2` |
| `…encoder.layers.{i}.final_layer_norm.{weight,bias}` | `enc_layers[i].norm2_gamma/beta` |
| `…decoder.layers.{i}.self_attn.*` | `dec_layers[i].self_attn` (+ `norm1`=`self_attn_layer_norm`) |
| `…decoder.layers.{i}.encoder_attn.*` | `dec_layers[i].cross_attn` (+ `norm2`=`encoder_attn_layer_norm`) |
| `…decoder.layers.{i}.fc1/fc2` (+ `norm3`=`final_layer_norm`) | `dec_layers[i].ffn` |
| `lm_head.weight` `(V,d)` | `lm_head` |
| `final_logits_bias` `(1,V)` | `final_logits_bias` (reshape to `(V,)`) |

No transpose anywhere: HF stores Linear weight as `(out,in)`, our `affine`/`mha`/`ffn` all do `x @ W.T`, so weights map **directly**. Forward: `embed = embed_tokens[ids]*embed_scale + embed_positions[arange(L)]`, `embed_scale = sqrt(d) if cfg.scale_embedding else 1.0`. Encoder = post-norm `encoder_block` ×N, no final norm. Decoder = causal `triangular_mask`, `decoder_block` ×N (cross-attn over encoder output). Logits = `dec_out @ lm_head.T + final_logits_bias`.

- [ ] **Step 1: Write the stub `transformer_model.py`**

```python
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


@dataclass(frozen=True)
class MarianParams:
    enc_embed: np.ndarray            # (V, d)
    dec_embed: np.ndarray            # (V, d)
    enc_pos: np.ndarray              # (P, d) fixed sinusoidal table
    dec_pos: np.ndarray              # (P, d)
    enc_layers: list                 # list[EncoderBlockParams] (from leet_llm import EncoderBlockParams)
    dec_layers: list                 # list[DecoderBlockParams]
    lm_head: np.ndarray              # (V, d), tied to shared embedding
    final_logits_bias: np.ndarray    # (V,)


def load_marian(weights: dict, cfg: TransformerConfig) -> MarianParams:
    """Map a dict of HF-named arrays (see README table) into MarianParams."""
    raise NotImplementedError("Implement load_marian — see 301_transformer_model/README.md")


def encoder(src_ids: np.ndarray, params: MarianParams, cfg: TransformerConfig) -> np.ndarray:
    """Token+positional embed → N post-norm encoder blocks → memory (B, S, d)."""
    raise NotImplementedError("Implement encoder — see 301_transformer_model/README.md")


def decoder(tgt_ids: np.ndarray, memory: np.ndarray, params: MarianParams,
            cfg: TransformerConfig) -> np.ndarray:
    """Causal-masked self-attn + cross-attn over memory → hidden (B, T, d)."""
    raise NotImplementedError("Implement decoder — see 301_transformer_model/README.md")


def transformer_logits(src_ids: np.ndarray, tgt_ids: np.ndarray, params: MarianParams,
                       cfg: TransformerConfig) -> np.ndarray:
    """Full forward → logits (B, T, V) = decoder(...) @ lm_head.T + final_logits_bias."""
    raise NotImplementedError("Implement transformer_logits — see 301_transformer_model/README.md")
```

Copy this file verbatim to `301_transformer_model/solution.py`.

- [ ] **Step 2: Write `tests/gen_fixtures.py`** (tiny genuine MarianMTModel oracle)

```python
"""301 — frozen goldens from a tiny genuine HuggingFace MarianMTModel.

AUTHORING ONLY (needs the ``gen`` group):
    uv run --group gen python 301_transformer_model/tests/gen_fixtures.py

A tiny MarianMTModel (random init, float64) is the oracle. We dump its full state_dict
(HF names), the input ids, and three goldens: encoder output, decoder output, final logits.
``scale_embedding=True`` and ``decoder_attention_heads=4`` exercise the embed-scale and
multi-head wrinkles; ``encoder_layers=2`` localizes block-stacking bugs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MarianConfig, MarianMTModel

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    torch.manual_seed(0)
    cfg = MarianConfig(
        vocab_size=64, decoder_vocab_size=64, d_model=16,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=32, activation_function="gelu",
        scale_embedding=True, share_encoder_decoder_embeddings=True,
        pad_token_id=63, eos_token_id=0, bos_token_id=63,
        decoder_start_token_id=63, forced_eos_token_id=0,
    )
    model = MarianMTModel(cfg).double().eval()

    src = np.array([[5, 6, 7, 8, 0]])                 # ends with eos
    tgt = np.array([[63, 9, 10, 11]])                 # starts with decoder_start
    with torch.no_grad():
        out = model(input_ids=torch.tensor(src), decoder_input_ids=torch.tensor(tgt),
                    output_hidden_states=True)

    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(
        src_ids=src, tgt_ids=tgt,
        enc_out=out.encoder_hidden_states[-1].numpy(),
        dec_out=out.decoder_hidden_states[-1].numpy(),
        logits=out.logits.numpy(),
        # config scalars for the test to rebuild TransformerConfig
        d_model=np.array(16), n_heads=np.array(4),
        n_enc_layers=np.array(2), n_dec_layers=np.array(2),
        d_ff=np.array(32), vocab_size=np.array(64), max_pos=np.array(32),
        scale_embedding=np.array(True), pad_id=np.array(63), eos_id=np.array(0),
        decoder_start_id=np.array(63),
    )
    np.savez(FIX / "tiny_marian.npz", **arrays)
    print(f"  wrote tiny_marian.npz  ({len(arrays)} arrays)  logits{out.logits.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the generator**

Run: `uv run --group gen python 301_transformer_model/tests/gen_fixtures.py`
Expected: `wrote tiny_marian.npz (… arrays) logits torch.Size([1, 4, 64])`; file exists.

- [ ] **Step 4: Write `tests/test_transformer_model.py`**

```python
import pathlib

import numpy as np

from leet_llm.grader import load

_m = load(__file__)
TransformerConfig = _m.TransformerConfig
load_marian = _m.load_marian
encoder = _m.encoder
decoder = _m.decoder
transformer_logits = _m.transformer_logits

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_marian.npz")


def _cfg():
    return TransformerConfig(
        d_model=int(_D["d_model"]), n_heads=int(_D["n_heads"]),
        n_enc_layers=int(_D["n_enc_layers"]), n_dec_layers=int(_D["n_dec_layers"]),
        d_ff=int(_D["d_ff"]), vocab_size=int(_D["vocab_size"]), max_pos=int(_D["max_pos"]),
        scale_embedding=bool(_D["scale_embedding"]), pad_id=int(_D["pad_id"]),
        eos_id=int(_D["eos_id"]), decoder_start_id=int(_D["decoder_start_id"]),
    )


def _params():
    weights = {k: _D[k] for k in _D.files}
    return load_marian(weights, _cfg())


def test_encoder_matches_hf():
    out = encoder(_D["src_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["enc_out"], rtol=1e-9, atol=1e-9)


def test_decoder_matches_hf():
    mem = _D["enc_out"]
    out = decoder(_D["tgt_ids"], mem, _params(), _cfg())
    np.testing.assert_allclose(out, _D["dec_out"], rtol=1e-9, atol=1e-9)


def test_logits_match_hf():
    out = transformer_logits(_D["src_ids"], _D["tgt_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["logits"], rtol=1e-9, atol=1e-9)


def test_logits_shape():
    out = transformer_logits(_D["src_ids"], _D["tgt_ids"], _params(), _cfg())
    assert out.shape == (1, _D["tgt_ids"].shape[1], int(_D["vocab_size"]))


def test_causal_decoder_ignores_future():
    # Perturbing a later target token must not change an earlier position's hidden state.
    p, cfg = _params(), _cfg()
    mem = _D["enc_out"]
    base = decoder(_D["tgt_ids"], mem, p, cfg)
    tgt2 = _D["tgt_ids"].copy()
    tgt2[0, -1] = (tgt2[0, -1] + 1) % int(_D["vocab_size"])
    pert = decoder(tgt2, mem, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)
```

- [ ] **Step 5: Run tests — verify they FAIL cleanly**

Run: `uv run grade 301`
Expected: FAIL — `NotImplementedError` from `load_marian`.

- [ ] **Step 6: Validate with a temporary reference**

Paste a correct implementation into `301_transformer_model/solution.py` (replace the four `NotImplementedError` bodies; keep the dataclasses). Reference logic:

```python
from leet_llm import encoder_block, decoder_block, AttnParams, FFNParams, triangular_mask
from leet_llm import EncoderBlockParams, DecoderBlockParams

def _attn(w, p):  # w=weights dict, p=prefix e.g. "...self_attn"
    return AttnParams(Wq=w[f"{p}.q_proj.weight"], Wk=w[f"{p}.k_proj.weight"],
                      Wv=w[f"{p}.v_proj.weight"], Wo=w[f"{p}.out_proj.weight"],
                      bq=w[f"{p}.q_proj.bias"], bk=w[f"{p}.k_proj.bias"],
                      bv=w[f"{p}.v_proj.bias"], bo=w[f"{p}.out_proj.bias"])

def _ffn(w, p):
    return FFNParams(W1=w[f"{p}.fc1.weight"], b1=w[f"{p}.fc1.bias"],
                     W2=w[f"{p}.fc2.weight"], b2=w[f"{p}.fc2.bias"])

def load_marian(weights, cfg):
    w = weights
    enc, dec = [], []
    for i in range(cfg.n_enc_layers):
        p = f"model.encoder.layers.{i}"
        enc.append(EncoderBlockParams(
            attn=_attn(w, f"{p}.self_attn"), ffn=_ffn(w, p),
            norm1_gamma=w[f"{p}.self_attn_layer_norm.weight"], norm1_beta=w[f"{p}.self_attn_layer_norm.bias"],
            norm2_gamma=w[f"{p}.final_layer_norm.weight"], norm2_beta=w[f"{p}.final_layer_norm.bias"]))
    for i in range(cfg.n_dec_layers):
        p = f"model.decoder.layers.{i}"
        dec.append(DecoderBlockParams(
            self_attn=_attn(w, f"{p}.self_attn"), cross_attn=_attn(w, f"{p}.encoder_attn"), ffn=_ffn(w, p),
            norm1_gamma=w[f"{p}.self_attn_layer_norm.weight"], norm1_beta=w[f"{p}.self_attn_layer_norm.bias"],
            norm2_gamma=w[f"{p}.encoder_attn_layer_norm.weight"], norm2_beta=w[f"{p}.encoder_attn_layer_norm.bias"],
            norm3_gamma=w[f"{p}.final_layer_norm.weight"], norm3_beta=w[f"{p}.final_layer_norm.bias"]))
    enc_emb = w.get("model.encoder.embed_tokens.weight", w["model.shared.weight"])
    dec_emb = w.get("model.decoder.embed_tokens.weight", w["model.shared.weight"])
    return MarianParams(enc_embed=enc_emb, dec_embed=dec_emb,
                        enc_pos=w["model.encoder.embed_positions.weight"],
                        dec_pos=w["model.decoder.embed_positions.weight"],
                        enc_layers=enc, dec_layers=dec, lm_head=w["lm_head.weight"],
                        final_logits_bias=w["final_logits_bias"].reshape(-1))

def _embed(ids, table, pos, cfg):
    import numpy as np
    scale = np.sqrt(cfg.d_model) if cfg.scale_embedding else 1.0
    L = ids.shape[1]
    return table[ids] * scale + pos[np.arange(L)]

def encoder(src_ids, params, cfg):
    h = _embed(src_ids, params.enc_embed, params.enc_pos, cfg)
    for blk in params.enc_layers:
        h = encoder_block(h, blk, cfg.n_heads, mask=None)
    return h

def decoder(tgt_ids, memory, params, cfg):
    h = _embed(tgt_ids, params.dec_embed, params.dec_pos, cfg)
    self_mask = triangular_mask(tgt_ids.shape[1])
    for blk in params.dec_layers:
        h = decoder_block(h, memory, blk, cfg.n_heads, self_mask=self_mask, cross_mask=None)
    return h

def transformer_logits(src_ids, tgt_ids, params, cfg):
    mem = encoder(src_ids, params, cfg)
    h = decoder(tgt_ids, mem, params, cfg)
    return h @ params.lm_head.T + params.final_logits_bias
```

This requires the L2 operators (209/210/206/207/203) to be **solved**. Run with the reference target so the facade serves L2 references too:

Run: `LEET_LLM_TARGET=solution uv run grade 301`
Expected: PASS (all five tests). If 209/210/etc. are unsolved in `solution.py`, first paste their references too (they are needed transitively) — note this in the commit message.

Then **restore `301_transformer_model/solution.py`** to the verbatim `NotImplementedError` stub from Step 1.

Run: `uv run grade 301`
Expected: FAIL with `NotImplementedError` (clean unsolved state).

- [ ] **Step 7: Register public names** in `leet_llm/_registry.py` (add under an `# L3 — Whole-Model & Inference` comment):

```python
    "TransformerConfig": ("301_transformer_model", "TransformerConfig"),
    "MarianParams": ("301_transformer_model", "MarianParams"),
    "load_marian": ("301_transformer_model", "load_marian"),
    "encoder": ("301_transformer_model", "encoder"),
    "decoder": ("301_transformer_model", "decoder"),
    "transformer_logits": ("301_transformer_model", "transformer_logits"),
```

- [ ] **Step 8: Write `README.md`** with these sections (no step-by-step array recipe):
  1. **Description** — assemble the classic encoder-decoder Transformer (Attention Is All You Need) into a full forward that emits vocab logits; the real target is `Helsinki-NLP/opus-mt-en-zh`.
  2. **The architecture (problem spec)** — the data flow `embed×scale + sinusoidal-pos → N× encoder_block → memory`; `embed → N× decoder_block(causal self-attn, cross-attn over memory) → dec_out`; `logits = dec_out @ lm_head.T + final_logits_bias`. State that it is **post-norm**, biases on all linears, no final encoder/decoder norm.
  3. **HF config & weight layout (GIVEN — framework facts)** — the full key→slot table above; the `(out,in)` Linear convention (no transpose for `x@W.T`); `embed_scale = √d_model if scale_embedding`; positions are a **fixed sinusoidal table** you gather by index (contrast L2 204); embeddings tied (`shared`↔`lm_head`); `final_logits_bias` added to logits; special-token ids.
  4. **Read More** — Attention Is All You Need; HF `MarianMTModel` docs; the OPUS-MT model card.
  5. **Function Signature** — the five public callables.
  6. **How to Test** — `uv run grade 301`.

- [ ] **Step 9: Commit**

```bash
git add 301_transformer_model leet_llm/_registry.py
git commit -m "L3 301: whole encoder-decoder transformer (opus-mt assembly) scaffold"
```

---

## Task 2: `302_translate` — greedy decode loop + real en→zh capstone

**Files:**
- Create: `302_translate/translate.py`, `302_translate/solution.py`, `302_translate/README.md`
- Create: `302_translate/download.sh`, `302_translate/convert.py`
- Create: `302_translate/tests/gen_fixtures.py`, `302_translate/tests/test_translate.py`, `302_translate/tests/fixtures/*.npz`
- Modify: `leet_llm/_registry.py`

- [ ] **Step 1: Write the stub `translate.py`**

```python
"""302 — Greedy translation with the encoder-decoder Transformer.

Encode the source once, then autoregressively greedy-decode target tokens until EOS.
Stateless recompute (no KV-cache — that is L4). See README.md.
Run ``uv run grade 302`` to check your work.

Reuse: ``from leet_llm import transformer_logits, TransformerConfig, load_marian``.
"""

from __future__ import annotations

import numpy as np


def translate(src_ids: np.ndarray, params, cfg, max_new_tokens: int = 64) -> list[int]:
    """Greedy-decode a single source sequence (shape (1, S)) → list[int] of target ids
    (including the leading decoder_start_id and the trailing eos_id if produced)."""
    raise NotImplementedError("Implement translate — see 302_translate/README.md")
```

Copy verbatim to `solution.py`.

- [ ] **Step 2: Write `tests/gen_fixtures.py`** — reuse the tiny Marian, capture HF greedy ids

```python
"""302 — frozen greedy-decode goldens from the tiny genuine MarianMTModel.

AUTHORING ONLY (gen group):
    uv run --group gen python 302_translate/tests/gen_fixtures.py

Same tiny config + seed as task 301 so the committed weights match. We capture HF's
greedy ``generate`` (num_beams=1, do_sample=False) output ids as the token-sequence oracle.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MarianConfig, MarianMTModel

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    torch.manual_seed(0)
    cfg = MarianConfig(
        vocab_size=64, decoder_vocab_size=64, d_model=16,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=32, activation_function="gelu",
        scale_embedding=True, share_encoder_decoder_embeddings=True,
        pad_token_id=63, eos_token_id=0, bos_token_id=63,
        decoder_start_token_id=63, forced_eos_token_id=0,
    )
    model = MarianMTModel(cfg).double().eval()
    src = np.array([[5, 6, 7, 8, 0]])
    with torch.no_grad():
        gen = model.generate(torch.tensor(src), max_length=12, num_beams=1, do_sample=False)
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(src_ids=src, expected_ids=gen.numpy(),
                  d_model=np.array(16), n_heads=np.array(4),
                  n_enc_layers=np.array(2), n_dec_layers=np.array(2),
                  d_ff=np.array(32), vocab_size=np.array(64), max_pos=np.array(32),
                  scale_embedding=np.array(True), pad_id=np.array(63), eos_id=np.array(0),
                  decoder_start_id=np.array(63))
    np.savez(FIX / "tiny_greedy.npz", **arrays)
    print("  wrote tiny_greedy.npz  expected_ids", gen.numpy().tolist())


if __name__ == "__main__":
    main()
```

Run: `uv run --group gen python 302_translate/tests/gen_fixtures.py`
Expected: prints `expected_ids [[63, …, 0]]`; file exists.

- [ ] **Step 3: Write `tests/test_translate.py`**

```python
import pathlib

import numpy as np

from leet_llm.grader import load

_m = load(__file__)
translate = _m.translate

# build params/cfg via task 301 through the facade
from leet_llm import TransformerConfig, load_marian

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_greedy.npz")


def _cfg():
    return TransformerConfig(
        d_model=int(_D["d_model"]), n_heads=int(_D["n_heads"]),
        n_enc_layers=int(_D["n_enc_layers"]), n_dec_layers=int(_D["n_dec_layers"]),
        d_ff=int(_D["d_ff"]), vocab_size=int(_D["vocab_size"]), max_pos=int(_D["max_pos"]),
        scale_embedding=bool(_D["scale_embedding"]), pad_id=int(_D["pad_id"]),
        eos_id=int(_D["eos_id"]), decoder_start_id=int(_D["decoder_start_id"]))


def _params():
    return load_marian({k: _D[k] for k in _D.files}, _cfg())


def test_greedy_matches_hf_generate():
    cfg = _cfg()
    out = translate(_D["src_ids"], _params(), cfg, max_new_tokens=12)
    expected = [t for t in _D["expected_ids"][0].tolist()]
    # HF prepends decoder_start and stops at eos; compare up to and including first eos
    assert out[: len(expected)] == expected


def test_starts_with_decoder_start_and_stops_at_eos():
    cfg = _cfg()
    out = translate(_D["src_ids"], _params(), cfg, max_new_tokens=12)
    assert out[0] == cfg.decoder_start_id
    assert cfg.eos_id in out
```

Run: `uv run grade 302`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 4: Validate with a temporary reference**

Paste into `302_translate/solution.py`:

```python
import numpy as np
from leet_llm import transformer_logits

def translate(src_ids, params, cfg, max_new_tokens=64):
    ids = [cfg.decoder_start_id]
    for _ in range(max_new_tokens):
        tgt = np.array([ids])
        logits = transformer_logits(src_ids, tgt, params, cfg)  # (1, t, V)
        nxt = int(np.argmax(logits[0, -1]))
        ids.append(nxt)
        if nxt == cfg.eos_id:
            break
    return ids
```

Run: `LEET_LLM_TARGET=solution uv run grade 302` → Expected: PASS.
Restore `solution.py` to the stub; `uv run grade 302` → Expected: FAIL cleanly.

- [ ] **Step 5: Write `download.sh`**

```bash
#!/usr/bin/env bash
# Fetch the real opus-mt-en-zh checkpoint, then convert it to our .npz layout.
# CC-BY-4.0 (Helsinki-NLP / OPUS-MT). Weights are NOT committed; this is opt-in.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
uv run --group gen python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Helsinki-NLP/opus-mt-en-zh")
print("downloaded to", p)
PY
uv run --group gen python "$HERE/convert.py"
echo "Done -> $HERE/opus_mt_en_zh.npz"
```

Make executable: `chmod +x 302_translate/download.sh`.

- [ ] **Step 6: Write `convert.py`**

```python
"""302 — convert the real opus-mt-en-zh checkpoint to our .npz (AUTHORING/DEMO, gen group).

    uv run --group gen python 302_translate/convert.py

Writes opus_mt_en_zh.npz (full HF state_dict, float64) next to this file, plus a small
committed reference fixture tests/fixtures/real_ref.npz holding the HF greedy ids for a
fixed English prompt. The big .npz is git-ignored; only real_ref.npz is committed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = pathlib.Path(__file__).parent
NAME = "Helsinki-NLP/opus-mt-en-zh"
PROMPT = "I have a dream that one day this nation will rise up."


def main() -> None:
    tok = MarianTokenizer.from_pretrained(NAME)
    model = MarianMTModel.from_pretrained(NAME).double().eval()
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(HERE / "opus_mt_en_zh.npz", **arrays)

    enc = tok([PROMPT], return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**enc, num_beams=1, do_sample=False, max_length=64)
    cfg = model.config
    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    np.savez(HERE / "tests" / "fixtures" / "real_ref.npz",
             src_ids=enc["input_ids"].numpy(), expected_ids=gen.numpy(),
             d_model=np.array(cfg.d_model), n_heads=np.array(cfg.decoder_attention_heads),
             n_enc_layers=np.array(cfg.encoder_layers), n_dec_layers=np.array(cfg.decoder_layers),
             d_ff=np.array(cfg.decoder_ffn_dim), vocab_size=np.array(cfg.vocab_size),
             max_pos=np.array(cfg.max_position_embeddings),
             scale_embedding=np.array(bool(cfg.scale_embedding)),
             pad_id=np.array(cfg.pad_token_id), eos_id=np.array(cfg.eos_token_id),
             decoder_start_id=np.array(cfg.decoder_start_token_id),
             activation=np.array(cfg.activation_function))
    print("translation:", tok.decode(gen[0], skip_special_tokens=True))
    print("wrote opus_mt_en_zh.npz (gitignored) + tests/fixtures/real_ref.npz")


if __name__ == "__main__":
    main()
```

Add `302_translate/opus_mt_en_zh.npz` to `.gitignore`.

- [ ] **Step 7: Add the gated real-weight capstone test** (append to `tests/test_translate.py`)

```python
import pytest

_REAL_W = pathlib.Path(__file__).parent.parent / "opus_mt_en_zh.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(not (_REAL_W.exists() and _REAL_REF.exists()),
                    reason="run 302_translate/download.sh to fetch real opus-mt-en-zh weights")
def test_real_en_zh_matches_hf_greedy():
    R = np.load(_REAL_REF)
    cfg = TransformerConfig(
        d_model=int(R["d_model"]), n_heads=int(R["n_heads"]),
        n_enc_layers=int(R["n_enc_layers"]), n_dec_layers=int(R["n_dec_layers"]),
        d_ff=int(R["d_ff"]), vocab_size=int(R["vocab_size"]), max_pos=int(R["max_pos"]),
        scale_embedding=bool(R["scale_embedding"]), pad_id=int(R["pad_id"]),
        eos_id=int(R["eos_id"]), decoder_start_id=int(R["decoder_start_id"]))
    params = load_marian({k: np.load(_REAL_W)[k] for k in np.load(_REAL_W).files}, cfg)
    out = translate(R["src_ids"], params, cfg, max_new_tokens=64)
    expected = R["expected_ids"][0].tolist()
    assert out[: len(expected)] == expected
```

Note: if `convert.py` reports `activation` ≠ `"gelu"`, the FFN activation in L2 207 must be parameterized — record this as a follow-up (see spec §8) and set the tiny-config `activation_function` to match before relying on hermetic parity.

- [ ] **Step 8: Register + README + commit**

Add to `leet_llm/_registry.py`: `"translate": ("302_translate", "translate"),`.

`README.md` sections: Description (greedy translation, encode-once/decode-many, stateless); The algorithm/problem spec (start at `decoder_start_id`, argmax the last-position logits, append, stop at `eos_id`); HF facts (special-token ids; MarianTokenizer used only in the opt-in demo); **How to run for real** (`bash 302_translate/download.sh` then the demo), with the CC-BY-4.0 attribution note; How to Test (`uv run grade 302`).

```bash
git add 302_translate leet_llm/_registry.py .gitignore
git commit -m "L3 302: greedy translate loop + gated real opus-mt-en-zh capstone scaffold"
```

- [ ] **Step 9: Update root `README.md` progress row**

Change the L3 line to `✅ scaffolded (Track A: 301–302)` (Track B + zoo still planned).

```bash
git add README.md && git commit -m "docs: mark L3 Track A scaffolded"
```

---

## Self-Review

**Spec coverage (Track A slice of the spec):**
- §2.1 pure-functional assembly → 301 (`encoder`/`decoder`/`transformer_logits`), 302 (`translate`). ✓
- §2.2 stateless recompute → 302 recomputes the full prefix each step; no cache. ✓
- §2.3 greedy → 302 argmax; matches HF `num_beams=1`. ✓
- §2.4 two tasks per capstone (build/run) → 301 build, 302 run. ✓
- §2.5 no shipped weights → `download.sh`+`convert.py`, big `.npz` gitignored, only `real_ref.npz` committed. ✓
- §3 tiny genuine-HF fixtures (Tier 1) + gated real-weight capstone (Tier 2) → Steps 2/3 (301) and 2/3/7 (302). ✓
- §6 HF facts given in README → 301 Step 8 §3, 302 Step 8. ✓
- Attention-bias gap (user-approved) → Task 0. ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code; the only deferred item (FFN activation if real config ≠ gelu) is an explicit, spec-tracked follow-up with a concrete trigger, not a silent gap.

**Type consistency:** `TransformerConfig`, `MarianParams`, `load_marian(weights, cfg)`, `encoder(src_ids, params, cfg)`, `decoder(tgt_ids, memory, params, cfg)`, `transformer_logits(src_ids, tgt_ids, params, cfg)`, `translate(src_ids, params, cfg, max_new_tokens)` are used identically in stubs, references, tests, and registry. `AttnParams(... bq/bk/bv/bo=None)` matches Task 0. Block param dataclasses match the verified L2 field names (`EncoderBlockParams.attn/ffn/norm1_*/norm2_*`; `DecoderBlockParams.self_attn/cross_attn/ffn/norm1_*/norm2_*/norm3_*`; `FFNParams.W1/b1/W2/b2`).

**Dependency note:** validating 301/302 references requires the transitive L2 operators (206/207/203/209/210, and 213/214/215/216 are *not* on Track A) to be solved under `LEET_LLM_TARGET=solution`. If L2 `solution.py` files are still `NotImplementedError`, paste their references during Step 6 validation only, then restore. This does not affect shipped scaffolds.
