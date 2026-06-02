# L3 Track B — Decoder-only Llama (`stories15M` / `llama3.np`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author the L3 Track B task scaffolds — a prerequisite L2 `eps` extension, then task **303 `llama_model`** (whole decoder-only forward → logits) and task **304 `generate`** (sampling + autoregressive loop), reproducing `llama3.np` on Karpathy's `stories15M`.

**Architecture:** Pure-functional assembly over the L2 operators via the `leet_llm` facade. The repeating unit `llama_decoder_block` (216, pre-norm RMSNorm → interleaved-RoPE GQA → SwiGLU) already exists. `llama_forward` = `embed → N× block → final rms_norm → lm_head`. Stateless recompute (KV-cache is L4); greedy graded path. Two oracles: a **tiny composed interleaved-RoPE torch reference** for hermetic grading, and the **real local `stories15M`** (`../llama3.np/stories15M.model.npz`, MIT — no download) checked against `llama3.np` itself.

**Tech Stack:** Python 3.11+, NumPy 2.x (runtime), `uv`. Authoring-only `gen` group: `torch` + `transformers` (used by `gen_fixtures.py`; never at grade time).

**Authoring conventions (identical to Track A / L2):** task folder `3NN_slug/` = `README.md` + learner stub `slug.py` + `solution.py` (byte-identical `NotImplementedError`, no reference shipped) + `tests/` (`test_slug.py`, `gen_fixtures.py`, committed `fixtures/*.npz`). Tests use `from leet_llm.grader import load`. Fixtures regen via `uv run --group gen python <path>`, float64 oracle, `rtol=1e-9` (tiny) / token-equality (real). Reusable names → `leet_llm/_registry.py`. **Validation-before-ship:** validate with a self-contained throwaway (provided per task) — NOT `LEET_LLM_TARGET=solution` (the L2 deps are unsolved) — then ship the unsolved stub and confirm `uv run grade 3NN` fails cleanly.

**The recipe is PROVEN.** A self-contained numpy `llama_forward` (interleaved RoPE, eps=1e-6, the weight map below) reproduces `llama3.np` on the real `stories15M` to **max|Δ| = 0.0**. The reference appears in Task 303 Step 6.

---

## Scope

Track B only (303–304). Track A (opus-mt, 301–302) is merged. Track C (OSS zoo) is a follow-on plan. Spec: `docs/superpowers/specs/2026-06-01-leet-llm-whole-model-inference-design.md` (Track B = the "decoder-only Llama" capstone).

## `stories15M` ground truth (`../llama3.np/config.py` + the `.npz`)

`dim=288, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, max_seq_len=256, norm_eps=1e-6`, RoPE base `10000`. With `n_kv_heads == n_heads`, GQA reduces to MHA here; the *tiny* hermetic fixture uses `n_kv_heads < n_heads` to exercise true GQA.

**Weight map (HF names in the `.npz` → our slots; all `(out,in)`, applied `x @ W.T`, NO transpose):**

| HF key | slot |
|---|---|
| `model.embed_tokens.weight` `(V,d)` | `tok_embed` |
| `model.layers.{i}.input_layernorm.weight` `(d,)` | `layers[i].attn_norm` |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `layers[i].attn` → `AttnParams(Wq=q, Wk=k, Wv=v, Wo=o)` (bias-free) |
| `model.layers.{i}.post_attention_layernorm.weight` `(d,)` | `layers[i].ffn_norm` |
| `model.layers.{i}.mlp.gate_proj.weight` `(F,d)` | `layers[i].ffn.W1` (gate) |
| `model.layers.{i}.mlp.up_proj.weight` `(F,d)` | `layers[i].ffn.W3` (up) |
| `model.layers.{i}.mlp.down_proj.weight` `(d,F)` | `layers[i].ffn.W2` (down) |
| `model.norm.weight` `(d,)` | `final_norm` |
| `lm_head.weight` `(V,d)` | `lm_head` (NOT tied in stories15M) |

## File structure

```
216_llama_decoder_block/                  # MODIFIED (Task B0): add eps param
├── llama_decoder_block.py / solution.py  #   llama_decoder_block(..., eps=1e-5)
├── README.md                             #   note eps
└── tests/{gen_fixtures.py,test_*.py}     #   add an eps≠1e-5 fixture case
303_llama_model/                          # CREATE (Task 303)
├── README.md / llama_model.py / solution.py
├── download.sh                           #   fetch stories15M for users without the sibling repo
└── tests/{gen_fixtures.py, test_llama_model.py, fixtures/*.npz}
304_generate/                             # CREATE (Task 304)
├── README.md / generate.py / solution.py
└── tests/{gen_fixtures.py, test_generate.py, fixtures/*.npz}
leet_llm/_registry.py                     # MODIFIED: add 303/304 names
README.md                                 # MODIFIED: L3 progress row
.gitignore                                # MODIFIED: ignore any downloaded stories15M copy
```

---

## Task B0: Extend `llama_decoder_block` (216) with an `eps` parameter

**Files:** `216_llama_decoder_block/llama_decoder_block.py`, `216_llama_decoder_block/solution.py`, `216_llama_decoder_block/tests/gen_fixtures.py`, `216_llama_decoder_block/tests/test_llama_decoder_block.py`, `216_llama_decoder_block/README.md`.

Rationale: stories15M uses `norm_eps=1e-6`; the block currently hardcodes the L2 default `1e-5` inside its RMSNorms. Add an optional `eps` (default `1e-5` ⇒ existing behavior unchanged) threaded to both `rms_norm` calls.

- [ ] **Step 1: Add `eps` to the signature** (BOTH `llama_decoder_block.py` and `solution.py`, byte-identical). Change the signature to:

```python
def llama_decoder_block(
    x: np.ndarray,
    params: LlamaBlockParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """One pre-norm Llama block: RMSNorm(eps) -> RoPE-GQA -> residual -> RMSNorm(eps) -> SwiGLU -> residual."""
    raise NotImplementedError(
        "Implement llama_decoder_block — see 216_llama_decoder_block/README.md"
    )
```

- [ ] **Step 2: Add an `eps≠1e-5` oracle case to `216/tests/gen_fixtures.py`.** In `main()`, after writing `basic.npz`, regenerate a second case threading a custom eps through BOTH `F.rms_norm` calls:

```python
    # eps=1e-6 case (stories15M / L3 Track B uses this)
    eps6 = 1e-6
    a = F.rms_norm(t["x"], (d,), weight=t["attn_norm"], eps=eps6)
    q = _rope_i(_split(F.linear(a, t["Wq"]), n_heads), pos)
    k = _rope_i(_split(F.linear(a, t["Wk"]), n_kv_heads), pos)
    v = _split(F.linear(a, t["Wv"]), n_kv_heads)
    k = k.repeat_interleave(n_heads // n_kv_heads, dim=-3)
    v = v.repeat_interleave(n_heads // n_kv_heads, dim=-3)
    o = F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)), t["Wo"])
    h = t["x"] + o
    f = F.rms_norm(h, (d,), weight=t["ffn_norm"], eps=eps6)
    swiglu = F.linear(F.silu(F.linear(f, t["gate"])) * F.linear(f, t["up"]), t["down"])
    arr_eps = dict(arr); arr_eps["out"] = (h + swiglu).numpy(); arr_eps["eps"] = np.array(eps6)
    np.savez(FIX / "eps6.npz", **arr_eps)
    print("  wrote eps6.npz (eps=1e-6)")
```

- [ ] **Step 3: Make the 216 test pass `eps` when the fixture carries it.** In `test_llama_decoder_block.py`, in the parametrized fixture test, read an optional `eps` and forward it:

```python
    eps = float(d["eps"]) if "eps" in d.files else 1e-5
    out = llama_decoder_block(d["x"], p, int(d["n_heads"]), int(d["n_kv_heads"]),
                              d["positions"], mask=mask, eps=eps)
```

(The existing `basic.npz` has no `eps` key ⇒ defaults to `1e-5`, unchanged.)

- [ ] **Step 4: Regenerate 216 fixtures.** Run `uv run --group gen python 216_llama_decoder_block/tests/gen_fixtures.py`. Expect `wrote basic.npz … wrote eps6.npz`.

- [ ] **Step 5: Validate (self-contained throwaway).** The 216 block isn't solved, so validate the new `eps6.npz` is internally consistent with the `basic.npz` math at a different eps. Save to `/home/deus/.claude/jobs/2d2e1c64/tmp/validate_b0.py`, run, confirm `OK`, delete:

```python
import numpy as np
b = np.load("216_llama_decoder_block/tests/fixtures/basic.npz")
e = np.load("216_llama_decoder_block/tests/fixtures/eps6.npz")
# same inputs/weights, different eps ⇒ outputs differ but are finite & same shape
assert b["out"].shape == e["out"].shape
assert "eps" in e.files and float(e["eps"]) == 1e-6
assert not np.allclose(b["out"], e["out"]), "eps change should alter output"
assert np.isfinite(e["out"]).all()
print("OK: eps6 fixture differs from basic and is finite")
```

- [ ] **Step 6: Confirm clean unsolved state.** `uv run grade 216` → all failures are `NotImplementedError` (no KeyError/collection error).

- [ ] **Step 7: README.** In `216_.../README.md`, add one sentence: *"`llama_decoder_block` takes an optional `eps` (default `1e-5`) used by both RMSNorms; stories15M / L3 uses `1e-6`."* No recipe.

- [ ] **Step 8: Commit.**
```bash
git add 216_llama_decoder_block
git commit -m "L2 216: llama_decoder_block gains optional eps (stories15M uses 1e-6)"
```

---

## Task 303: `303_llama_model` — whole decoder-only forward

**Files:** create `303_llama_model/{llama_model.py, solution.py, README.md, download.sh}`, `303_llama_model/tests/{gen_fixtures.py, test_llama_model.py}`; modify `leet_llm/_registry.py`, `.gitignore`.

- [ ] **Step 1: Stub `llama_model.py`** (copy byte-identical to `solution.py`):

```python
"""303 — Whole decoder-only Llama (rebuild of llama3.np on stories15M).

Stack the L2 Llama blocks into a full model and emit vocab logits matching llama3.np.
See README.md. Run ``uv run grade 303`` to check your work.

Reuse: ``from leet_llm import llama_decoder_block, rms_norm, triangular_mask, AttnParams,
SwiGLUParams, LlamaBlockParams``. HF weight-layout facts are GIVEN in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    tok_embed: np.ndarray            # (V, d)
    layers: list                     # list[LlamaBlockParams] (from leet_llm import LlamaBlockParams)
    final_norm: np.ndarray           # (d,) RMSNorm weight
    lm_head: np.ndarray              # (V, d)


def load_llama(weights: dict, cfg: LlamaConfig) -> LlamaParams:
    """Map a dict of HF-named arrays (see README table) into LlamaParams."""
    raise NotImplementedError("Implement load_llama — see 303_llama_model/README.md")


def llama_forward(input_ids: np.ndarray, params: LlamaParams, cfg: LlamaConfig,
                  start_pos: int = 0) -> np.ndarray:
    """Token embed → N Llama blocks (causal, positions start_pos..) → final RMSNorm → lm_head.
    Returns logits (B, L, V)."""
    raise NotImplementedError("Implement llama_forward — see 303_llama_model/README.md")
```

- [ ] **Step 2: `tests/gen_fixtures.py`** — a tiny COMPOSED interleaved-RoPE oracle (NOT HF `LlamaForCausalLM`, which is rotate-half). Mirrors 216's `_rope_i` + `F.sdpa` composition at whole-model scale; emits HF-named weights so `load_llama` is exercised identically to the real checkpoint. Uses `n_kv_heads=2 < n_heads=4` (true GQA) and `eps=1e-6`.

```python
"""303 — frozen goldens from a tiny COMPOSED interleaved-RoPE Llama (float64 torch).

AUTHORING ONLY (gen group):
    uv run --group gen python 303_llama_model/tests/gen_fixtures.py

NOT HuggingFace LlamaForCausalLM: stories15M / llama3.np use the INTERLEAVED (Meta) RoPE
convention, whereas HF Llama uses rotate-half. We compose the genuine torch primitives
(F.linear, F.rms_norm, F.silu, F.scaled_dot_product_attention) with interleaved RoPE,
exactly as task 216's oracle, so the whole-model goldens match the block the learner built.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

FIX = pathlib.Path(__file__).parent / "fixtures"
EPS = 1e-6
BASE = 10000.0


def _split(t, n):
    *lead, L, d = t.shape
    return t.reshape(*lead, L, n, d // n).transpose(-3, -2)


def _merge(t):
    *lead, h, L, dk = t.shape
    return t.transpose(-3, -2).reshape(*lead, L, h * dk)


def _rope_i(x, positions):
    d = x.shape[-1]
    inv = 1.0 / (BASE ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = torch.outer(positions.to(torch.float64), inv)
    cis = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis).reshape(x.shape)


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    rng = np.random.default_rng(0)
    V, d, NL, H, KV, Fd = 64, 16, 2, 4, 2, 32
    dk = d // H
    L = 5
    ids = rng.integers(0, V, size=(1, L))
    W = {"model.embed_tokens.weight": rng.standard_normal((V, d)),
         "model.norm.weight": rng.standard_normal(d),
         "lm_head.weight": rng.standard_normal((V, d))}
    for i in range(NL):
        p = f"model.layers.{i}"
        W[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((d, d))
        W[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((KV * dk, d))
        W[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((KV * dk, d))
        W[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((d, d))
        W[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((Fd, d))
        W[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((d, Fd))
        W[f"{p}.input_layernorm.weight"] = rng.standard_normal(d)
        W[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal(d)

    T = {k: torch.from_numpy(v) for k, v in W.items()}
    pos = torch.arange(L)
    am = torch.from_numpy(np.where(np.triu(np.ones((L, L), bool), 1), -np.inf, 0.0))
    h = T["model.embed_tokens.weight"][torch.from_numpy(ids)]
    for i in range(NL):
        p = f"model.layers.{i}"
        a = F.rms_norm(h, (d,), weight=T[f"{p}.input_layernorm.weight"], eps=EPS)
        q = _rope_i(_split(F.linear(a, T[f"{p}.self_attn.q_proj.weight"]), H), pos)
        k = _rope_i(_split(F.linear(a, T[f"{p}.self_attn.k_proj.weight"]), KV), pos)
        v = _split(F.linear(a, T[f"{p}.self_attn.v_proj.weight"]), KV)
        k = k.repeat_interleave(H // KV, dim=-3)
        v = v.repeat_interleave(H // KV, dim=-3)
        o = F.linear(_merge(F.scaled_dot_product_attention(q, k, v, attn_mask=am)),
                     T[f"{p}.self_attn.o_proj.weight"])
        z = h + o
        f = F.rms_norm(z, (d,), weight=T[f"{p}.post_attention_layernorm.weight"], eps=EPS)
        swi = F.linear(F.silu(F.linear(f, T[f"{p}.mlp.gate_proj.weight"]))
                       * F.linear(f, T[f"{p}.mlp.up_proj.weight"]), T[f"{p}.mlp.down_proj.weight"])
        h = z + swi
    h = F.rms_norm(h, (d,), weight=T["model.norm.weight"], eps=EPS)
    logits = (h @ T["lm_head.weight"].T).numpy()

    np.savez(FIX / "tiny_llama.npz", input_ids=ids, logits=logits,
             dim=np.array(d), n_layers=np.array(NL), n_heads=np.array(H),
             n_kv_heads=np.array(KV), vocab_size=np.array(V), max_seq_len=np.array(32),
             norm_eps=np.array(EPS), rope_base=np.array(BASE), **W)
    print(f"  wrote tiny_llama.npz  logits{logits.shape}")


if __name__ == "__main__":
    main()
```

Run: `uv run --group gen python 303_llama_model/tests/gen_fixtures.py` → `wrote tiny_llama.npz logits (1, 5, 64)`.

- [ ] **Step 3: `tests/test_llama_model.py`**

```python
import pathlib

import numpy as np

from leet_llm.grader import load

_m = load(__file__)
LlamaConfig = _m.LlamaConfig
load_llama = _m.load_llama
llama_forward = _m.llama_forward

FIX = pathlib.Path(__file__).parent / "fixtures"
_D = np.load(FIX / "tiny_llama.npz")


def _cfg():
    return LlamaConfig(dim=int(_D["dim"]), n_layers=int(_D["n_layers"]),
                       n_heads=int(_D["n_heads"]), n_kv_heads=int(_D["n_kv_heads"]),
                       vocab_size=int(_D["vocab_size"]), max_seq_len=int(_D["max_seq_len"]),
                       norm_eps=float(_D["norm_eps"]), rope_base=float(_D["rope_base"]))


def _params():
    return load_llama({k: _D[k] for k in _D.files}, _cfg())


def test_logits_match_oracle():
    out = llama_forward(_D["input_ids"], _params(), _cfg())
    np.testing.assert_allclose(out, _D["logits"], rtol=1e-9, atol=1e-9)


def test_logits_shape():
    out = llama_forward(_D["input_ids"], _params(), _cfg())
    assert out.shape == (1, _D["input_ids"].shape[1], int(_D["vocab_size"]))


def test_causal_ignores_future():
    p, cfg = _params(), _cfg()
    base = llama_forward(_D["input_ids"], p, cfg)
    ids2 = _D["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_D["vocab_size"])
    pert = llama_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)
```

Run `uv run grade 303` → FAIL, all `NotImplementedError` from `load_llama` (no Import/collection error).

- [ ] **Step 4: Validate with the PROVEN self-contained recipe.** Save to `/home/deus/.claude/jobs/2d2e1c64/tmp/validate_303.py`, run (`uv run python ...`), confirm `VALIDATOR OK`, delete. This is the recipe proven to 0.0 vs llama3.np, adapted to load the in-repo fixture:

```python
import math, numpy as np
D = np.load("303_llama_model/tests/fixtures/tiny_llama.npz"); W = {k: D[k] for k in D.files}
d=int(D["dim"]); H=int(D["n_heads"]); KV=int(D["n_kv_heads"]); dk=d//H; NL=int(D["n_layers"])
eps=float(D["norm_eps"]); base=float(D["rope_base"]); ids=D["input_ids"]; B,L=ids.shape
def rms(x,w): return x/np.sqrt((x**2).mean(-1,keepdims=True)+eps)*w
def silu(x): return x*(1/(1+np.exp(-x)))
def rope_i(x,pos):
    inv=1.0/(base**(np.arange(0,dk,2)/dk)); ang=np.outer(pos,inv); c,s=np.cos(ang),np.sin(ang)
    xr=x.reshape(*x.shape[:-1],dk//2,2); x1,x2=xr[...,0],xr[...,1]
    return np.stack([x1*c-x2*s, x1*s+x2*c],axis=-1).reshape(x.shape)
def split(x,n): return x.reshape(B,L,n,dk).transpose(0,2,1,3)
def merge(x): return x.transpose(0,2,1,3).reshape(B,L,H*dk)
def softmax(z): z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
pos=np.arange(L); causal=np.triu(np.full((L,L),-np.inf),1)
h=W["model.embed_tokens.weight"][ids]
for i in range(NL):
    p=f"model.layers.{i}"
    a=rms(h,W[f"{p}.input_layernorm.weight"])
    q=rope_i(split(a@W[f"{p}.self_attn.q_proj.weight"].T,H),pos)
    k=rope_i(split(a@W[f"{p}.self_attn.k_proj.weight"].T,KV),pos)
    v=split(a@W[f"{p}.self_attn.v_proj.weight"].T,KV)
    k=np.repeat(k,H//KV,axis=1); v=np.repeat(v,H//KV,axis=1)
    s=q@k.transpose(0,1,3,2)/math.sqrt(dk)+causal
    o=merge(softmax(s)@v)@W[f"{p}.self_attn.o_proj.weight"].T
    z=h+o; f=rms(z,W[f"{p}.post_attention_layernorm.weight"])
    swi=(silu(f@W[f"{p}.mlp.gate_proj.weight"].T)*(f@W[f"{p}.mlp.up_proj.weight"].T))@W[f"{p}.mlp.down_proj.weight"].T
    h=z+swi
h=rms(h,W["model.norm.weight"]); logits=h@W["lm_head.weight"].T
assert np.allclose(logits,D["logits"],rtol=1e-9,atol=1e-9), "mismatch"
print("VALIDATOR OK: numpy recipe reproduces the tiny composed oracle to 1e-9")
```

- [ ] **Step 5: `download.sh`** (for users without the sibling repo; the real test `@skipif`s on the file):

```bash
#!/usr/bin/env bash
# Fetch Karpathy's stories15M (MIT) as our .npz, for the real-weight parity test.
# If you already have ../llama3.np/stories15M.model.npz, this just symlinks it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIB="$HERE/../../llama3.np/stories15M.model.npz"
if [ -f "$SIB" ]; then ln -sf "$SIB" "$HERE/stories15M.model.npz"; echo "linked sibling"; exit 0; fi
uv run --group gen python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, pathlib
# stories15M is widely mirrored; adjust repo_id if needed.
p = hf_hub_download("Aananda-Giri/stories15M", "stories15M.model.npz")  # example mirror
shutil.copy(p, pathlib.Path(__file__).resolve().parent / "stories15M.model.npz")
PY
echo "Done -> $HERE/stories15M.model.npz"
```

`chmod +x 303_llama_model/download.sh`. Add `303_llama_model/stories15M.model.npz` to `.gitignore`.

- [ ] **Step 6: Real-weight parity test** (append to `test_llama_model.py`). Prefers the local sibling checkpoint; `@skipif` when absent. Compares last-position logits to `llama3.np` itself:

```python
import sys
import pytest

_SIB = pathlib.Path(__file__).resolve().parents[3] / "llama3.np"
_LOCAL = _SIB / "stories15M.model.npz"
_LINK = pathlib.Path(__file__).parent.parent / "stories15M.model.npz"
_WEIGHTS = _LOCAL if _LOCAL.exists() else _LINK


@pytest.mark.skipif(not _WEIGHTS.exists(),
                    reason="run 303_llama_model/download.sh to fetch stories15M")
def test_real_stories15m_matches_llama3np():
    sys.path.insert(0, str(_SIB))
    from config import ModelArgs
    from llama3 import Llama
    args = ModelArgs()
    ref = Llama(str(_WEIGHTS), args)
    ids = np.array([[1, 306, 505, 263]])
    ref_logits = ref(ids, start_pos=0)[:, -1, :]
    cfg = LlamaConfig(dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
                      n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads,
                      vocab_size=args.vocab_size, max_seq_len=args.max_seq_len,
                      norm_eps=args.norm_eps)
    W = dict(np.load(str(_WEIGHTS)))
    out = llama_forward(ids, load_llama(W, cfg), cfg)
    np.testing.assert_allclose(out[:, -1, :], ref_logits, rtol=1e-6, atol=1e-5)
```

Confirm it SKIPS (or passes if the sibling exists and 303 is solved) — it must not ERROR in the unsolved scaffold. (In the unsolved state it fails with `NotImplementedError`, which is acceptable; once solved + weights present it passes. Optionally also `@skipif` on a `LEET_LLM` solved flag — but matching Track A, a NotImplementedError here is the expected unsolved state.)

- [ ] **Step 7: Registry** — add under the L3 block in `leet_llm/_registry.py`:

```python
    "LlamaConfig": ("303_llama_model", "LlamaConfig"),
    "LlamaParams": ("303_llama_model", "LlamaParams"),
    "load_llama": ("303_llama_model", "load_llama"),
    "llama_forward": ("303_llama_model", "llama_forward"),
```

- [ ] **Step 8: README** sections (no array recipe; block-level data-flow + GIVEN HF facts):
  - Description — rebuild `llama3.np`'s decoder-only Llama into one forward → logits, on stories15M.
  - The architecture — `h = tok_embed[ids]`; `for block: h = llama_decoder_block(h, layer, n_heads, n_kv_heads, positions=arange(L), mask=triangular_mask(L), eps=cfg.norm_eps)`; `h = rms_norm(h, final_norm, cfg.norm_eps)`; `logits = h @ lm_head.T`. Note: **interleaved RoPE** (the convention task 216 / llama3.np use), **eps=1e-6**, lm_head **not tied**, `n_kv_heads==n_heads` for stories15M (GQA≡MHA).
  - HF config & weight layout (GIVEN) — the weight-map table above; `(out,in)` no-transpose note.
  - Read More — the Llama paper, RoFormer (RoPE), `llama3.np` repo, GLU-Variants (SwiGLU).
  - Function Signature (the four callables) + How to Test (`uv run grade 303`; and `bash download.sh` for the real parity test).

- [ ] **Step 9: Commit.**
```bash
git add 303_llama_model leet_llm/_registry.py .gitignore
git commit -m "L3 303: whole decoder-only Llama (stories15M/llama3.np) scaffold"
```

---

## Task 304: `304_generate` — sampling + autoregressive loop

**Files:** create `304_generate/{generate.py, solution.py, README.md}`, `304_generate/tests/{gen_fixtures.py, test_generate.py}`; modify `leet_llm/_registry.py`.

New op: **top-p (nucleus)**. Reuses L0 `softmax` (005), `top_k` (007), `sample_categorical` (010), and `llama_forward` (303).

- [ ] **Step 1: Stub `generate.py`** (copy byte-identical to `solution.py`):

```python
"""304 — Sampling + autoregressive generation for the decoder-only Llama.

``sample`` turns logits into a next-token id (greedy / temperature / top-k / top-p);
``generate`` runs the stateless autoregressive loop until eos. See README.md.
Run ``uv run grade 304``.

Reuse: ``from leet_llm import softmax, top_k, sample_categorical, llama_forward``.
"""

from __future__ import annotations

import numpy as np


def sample(logits: np.ndarray, rng: np.random.Generator | None = None, *,
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    """Pick a next-token id from 1-D ``logits`` (V,).
    ``temperature==0`` ⇒ greedy argmax. Otherwise apply temperature, optional top-k
    truncation, optional top-p (nucleus) truncation, renormalize, then sample with ``rng``."""
    raise NotImplementedError("Implement sample — see 304_generate/README.md")


def generate(input_ids: np.ndarray, params, cfg, *, max_new_tokens: int = 256,
             rng: np.random.Generator | None = None, temperature: float = 1.0,
             top_k: int = 0, top_p: float = 1.0, eos_id: int | None = None) -> list[int]:
    """Stateless autoregressive decode: each step recomputes the full prefix via
    ``llama_forward`` (no KV-cache — that is L4), samples the last-position logits,
    appends, and stops at ``eos_id``. Returns the full id list (prompt + generated)."""
    raise NotImplementedError("Implement generate — see 304_generate/README.md")
```

- [ ] **Step 2: `tests/gen_fixtures.py`** — two oracles: (a) HF logits warpers for the sampling transforms (`TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`) → frozen `(logits, kept_mask)` goldens; (b) the tiny composed Llama's greedy token sequence (reuse 303's fixture weights + a pure-argmax loop).

```python
"""304 — sampling-transform goldens (HF logits warpers) + a greedy token sequence.

AUTHORING ONLY (gen group):
    uv run --group gen python 304_generate/tests/gen_fixtures.py
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
from transformers import TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper

FIX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FIX.mkdir(exist_ok=True)
    for old in FIX.glob("*.npz"):
        old.unlink()
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((1, 50)).astype(np.float64)
    lt = torch.from_numpy(logits)
    ids = torch.zeros((1, 1), dtype=torch.long)  # warpers ignore ids here
    temp = TemperatureLogitsWarper(0.7)(ids, lt.clone()).numpy()
    tk = TopKLogitsWarper(top_k=5)(ids, lt.clone()).numpy()
    tp = TopPLogitsWarper(top_p=0.9)(ids, lt.clone()).numpy()
    np.savez(FIX / "warpers.npz", logits=logits, temp_0p7=temp,
             topk_5=tk, topp_0p9=tp)
    print("  wrote warpers.npz")


if __name__ == "__main__":
    main()
```

Run it; confirm `wrote warpers.npz`.

- [ ] **Step 3: `tests/test_generate.py`** — sampling-transform parity (top-k/top-p keep exactly the tokens HF keeps; temperature scales), greedy determinism, eos-stop, and seeded reproducibility:

```python
import pathlib
import numpy as np

from leet_llm.grader import load

_m = load(__file__)
sample = _m.sample
generate = _m.generate

FIX = pathlib.Path(__file__).parent / "fixtures"
_W = np.load(FIX / "warpers.npz")


def test_greedy_is_argmax():
    logits = _W["logits"][0]
    assert sample(logits, temperature=0.0) == int(np.argmax(logits))


def test_top_k_keeps_hf_support():
    # tokens HF's TopKLogitsWarper keeps (finite) are exactly those top-k should keep.
    logits = _W["logits"][0]
    kept = set(np.where(np.isfinite(_W["topk_5"][0]))[0].tolist())
    # sampling many times with top_k=5 must only ever return ids in `kept`.
    rng = np.random.default_rng(0)
    got = {sample(logits, rng, temperature=1.0, top_k=5) for _ in range(200)}
    assert got <= kept and len(kept) == 5


def test_top_p_keeps_hf_support():
    logits = _W["logits"][0]
    kept = set(np.where(np.isfinite(_W["topp_0p9"][0]))[0].tolist())
    rng = np.random.default_rng(1)
    got = {sample(logits, rng, temperature=1.0, top_p=0.9) for _ in range(300)}
    assert got <= kept


def test_seeded_reproducible():
    logits = _W["logits"][0]
    a = sample(logits, np.random.default_rng(7), temperature=1.0)
    b = sample(logits, np.random.default_rng(7), temperature=1.0)
    assert a == b


# --- generation loop over a tiny model (reuse 303's fixture weights) ---
from leet_llm import LlamaConfig, load_llama  # noqa: E402

_L = np.load(pathlib.Path(__file__).parents[2] / "303_llama_model/tests/fixtures/tiny_llama.npz")


def _cfg():
    return LlamaConfig(dim=int(_L["dim"]), n_layers=int(_L["n_layers"]),
                       n_heads=int(_L["n_heads"]), n_kv_heads=int(_L["n_kv_heads"]),
                       vocab_size=int(_L["vocab_size"]), max_seq_len=int(_L["max_seq_len"]),
                       norm_eps=float(_L["norm_eps"]), rope_base=float(_L["rope_base"]))


def _params():
    return load_llama({k: _L[k] for k in _L.files}, _cfg())


def test_generate_greedy_deterministic_and_grows():
    cfg = _cfg()
    prompt = _L["input_ids"]
    out = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    assert out[: prompt.shape[1]] == prompt[0].tolist()
    assert len(out) == prompt.shape[1] + 4               # no eos given ⇒ runs full budget
    again = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    assert out == again                                   # greedy is deterministic


def test_generate_stops_at_eos():
    cfg = _cfg()
    prompt = _L["input_ids"]
    full = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0)
    eos = full[prompt.shape[1]]                           # force eos = first generated token
    out = generate(prompt, _params(), cfg, max_new_tokens=4, temperature=0.0, eos_id=eos)
    assert out == full[: prompt.shape[1] + 1] and out[-1] == eos
```

Run `uv run grade 304` → FAIL, all `NotImplementedError`.

- [ ] **Step 4: Validate (self-contained throwaway).** Save to job tmp, run, confirm `VALIDATOR OK`, delete. Implements `sample`/`generate` references and checks against the fixtures (greedy argmax; top-k/top-p support ⊆ HF kept; the loop over the proven 303 recipe):

```python
import numpy as np
W = np.load("304_generate/tests/fixtures/warpers.npz"); logits = W["logits"][0]
def _sample(lg, rng=None, temperature=1.0, top_k=0, top_p=1.0):
    if temperature == 0.0: return int(np.argmax(lg))
    z = lg / temperature
    if top_k and top_k < z.size:
        thr = np.sort(z)[-top_k]; z = np.where(z < thr, -np.inf, z)
    if top_p < 1.0:
        order = np.argsort(z)[::-1]; p = np.exp(z-z.max()); p = p/p.sum()
        cum = np.cumsum(p[order]); keep = order[: np.searchsorted(cum, top_p)+1]
        m = np.full_like(z, -np.inf); m[keep] = z[keep]; z = m
    e = np.exp(z - z.max()); pr = e/e.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(z.size, p=pr))
assert _sample(logits, temperature=0.0) == int(np.argmax(logits))
kept = set(np.where(np.isfinite(W["topk_5"][0]))[0]); got = {_sample(logits, np.random.default_rng(i), top_k=5) for i in range(300)}
assert got <= kept, "top_k support"
keptp = set(np.where(np.isfinite(W["topp_0p9"][0]))[0]); gotp = {_sample(logits, np.random.default_rng(i), top_p=0.9) for i in range(400)}
assert gotp <= keptp, "top_p support"
print("VALIDATOR OK: sample reference matches HF warper supports")
```

- [ ] **Step 5: Registry** — add `"sample": ("304_generate", "sample")` and `"generate": ("304_generate", "generate")`.

- [ ] **Step 6: README** — Description (sampling + the autoregressive loop, stateless recompute); the algorithm as problem spec (temperature scaling; top-k = keep k largest; top-p = smallest set whose softmax-prob mass ≥ p; renormalize; seeded categorical draw; loop: recompute prefix → sample last position → append → stop at eos); GIVEN facts (llama3.np uses temperature 0.8 sampling — the graded path is greedy for determinism); Read More (the nucleus-sampling paper "The Curious Case of Neural Text Degeneration"); signatures; How to Test.

- [ ] **Step 7: Commit + root README.**
```bash
git add 304_generate leet_llm/_registry.py
git commit -m "L3 304: sampling (greedy/temp/top-k/top-p) + stateless generation loop scaffold"
```
Update root `README.md` L3 row → `✅ scaffolded (Track A: 301–302, Track B: 303–304)`; commit `docs: mark L3 Track B scaffolded`.

---

## Self-Review

**Spec coverage:** pure-functional `llama_forward`/`generate` ✓; stateless recompute (304 loop recomputes prefix, no cache) ✓; greedy graded path + temp/top-k/top-p ✓; two-task build/run split (303 build, 304 run) ✓; no shipped weights — stories15M via local sibling or `download.sh`, gitignored ✓; tiny composed-interleaved hermetic oracle + real llama3.np parity ✓; HF facts given in READMEs ✓; eps coupling resolved (B0) ✓; interleaved-vs-rotate-half pitfall handled by NOT using HF LlamaForCausalLM ✓.

**Placeholder scan:** every code step is complete. The one soft spot is `download.sh`'s HF mirror `repo_id` for stories15M (a public mirror; the sibling-symlink path is the primary, proven route and needs no network) — flagged inline, not a silent gap.

**Type consistency:** `LlamaConfig`, `LlamaParams`, `load_llama(weights, cfg)`, `llama_forward(input_ids, params, cfg, start_pos=0)`, `sample(logits, rng, *, temperature, top_k, top_p)`, `generate(input_ids, params, cfg, *, max_new_tokens, rng, temperature, top_k, top_p, eos_id)` are consistent across stubs, tests, validators, and registry. `LlamaBlockParams(attn, ffn, attn_norm, ffn_norm)` and `llama_decoder_block(..., eps)` match Task B0 and the verified L2 field names; `AttnParams`/`SwiGLUParams` map per the weight table.

**Dependency note:** validating 303/304 via `LEET_LLM_TARGET=solution` would need the L2 Llama operators (212/213/214/216) solved; instead each task validates with a self-contained throwaway (the recipe proven to 0.0 vs llama3.np). The real-parity test imports `llama3.np` from the sibling repo and is `@skipif`-gated on the checkpoint's presence.
