# 304 — Sampling & Autoregressive Generation (decoder-only Llama)

**Level 3 · Whole-Model & Inference**

## Description

Implement the two pieces that turn the Llama forward pass from task 303 into a text
generator: `sample` (logits → next-token id) and `generate` (the autoregressive loop).
`sample` supports greedy, temperature, top-k, and top-p (nucleus) decoding. `generate`
drives `llama_forward` in **stateless recompute** mode — the full prefix is reprocessed
at every step (no KV-cache; caching is L4) — sampling one token at a time until `eos_id`
or the budget runs out.

## The algorithm (problem spec)

`sample(logits, rng, ...)` operates on a 1-D logit vector `(V,)`:

```
if temperature == 0:            return argmax(logits)        # greedy, ignores rng/top-k/top-p
z = logits / temperature                                     # temperature scaling
if top_k > 0 and top_k < V:     keep the k largest entries of z, mask the rest to -inf
if top_p < 1.0:                 keep the smallest set of tokens whose softmax-prob
                                mass is >= p (nucleus), mask the rest to -inf
p = softmax(z)                  over the surviving (unmasked) entries  → renormalized
return categorical_draw(p, rng) # seeded draw with the given Generator
```

- **temperature** divides logits before softmax: lower ⇒ sharper, higher ⇒ flatter.
- **top-k** keeps the `k` largest logits and discards the rest.
- **top-p (nucleus)** sorts tokens by probability (descending) and keeps the smallest
  prefix whose cumulative probability reaches `p`; that prefix always includes at least
  the single most-likely token. After truncation the kept probabilities are renormalized.
- The draw uses the passed `rng` (a `np.random.Generator`) so results are reproducible
  for a fixed seed.

`generate(input_ids, params, cfg, ...)` runs the loop:

```
ids = list(input_ids[0])                         # input_ids has shape (1, S)

repeat up to max_new_tokens times:
    logits = llama_forward(array([ids]), params, cfg)   # (1, t, V)
    next_id = sample(logits[0, -1], rng, temperature, top_k, top_p)
    ids.append(next_id)
    if next_id == eos_id: break

return ids   # full sequence: prompt + generated (includes eos_id if produced)
```

The returned list starts with the prompt tokens. It ends with `eos_id` when the model
emits it within the budget; otherwise it ends after `max_new_tokens` appended tokens.

## GIVEN facts

- The reference `llama3.np` samples with **temperature 0.8**. The **graded path here is
  greedy** (`temperature=0.0`) so the generation-loop tests are deterministic; the
  sampling transforms (temperature / top-k / top-p) are checked separately against the
  HF logits-warper goldens in `tests/fixtures/warpers.npz`.
- `params` and `cfg` come from task 303; import them via
  `from leet_llm import LlamaConfig, load_llama, llama_forward`.

## Read More

- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) — the
  nucleus-sampling (top-p) paper: https://arxiv.org/abs/1904.09751

## Function Signature

```python
def sample(logits: np.ndarray, rng: np.random.Generator | None = None, *,
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    ...
#   logits: (V,)            1-D logit vector for one position
#   rng: np.random.Generator (for the seeded categorical draw)
#   returns: int            the chosen next-token id


def generate(input_ids: np.ndarray, params, cfg, *, max_new_tokens: int = 256,
             rng: np.random.Generator | None = None, temperature: float = 1.0,
             top_k: int = 0, top_p: float = 1.0, eos_id: int | None = None) -> list[int]:
    ...
#   input_ids: (1, S)       the prompt
#   params: LlamaParams, cfg: LlamaConfig  (from leet_llm import load_llama, LlamaConfig)
#   returns: list[int]      the full sequence (prompt + generated)
```

Reuse `softmax` (005), `top_k` (007), `sample_categorical` (010), and `llama_forward`
(303) via `from leet_llm import softmax, top_k, sample_categorical, llama_forward`.

## How to Test

```bash
uv run grade 304
```

The hermetic tests (sampling warpers + the tiny-model generation loop) need no weights. The
real-weight capstone test (`test_real_stories15m_greedy_story`) runs only once you've fetched
the checkpoint; until then it is skipped.

## Run it for real (watch your NumPy tell a story)

Fetch + convert `stories15M` — Karpathy's llama2.c TinyStories model, the same checkpoint
`llama3.np` runs (Llama-2 == Llama-3 architecture; the only deltas are the 32k SentencePiece
tokenizer and `rope_theta=10000`). Weights are **not** committed; this downloads only the
three files we need (`config.json` + `model.safetensors` + `tokenizer.model`, ~60 MB) and
un-permutes q/k from HF's rotate-half layout back to our interleaved RoPE:

```bash
uv sync --group gen
bash 304_generate/download.sh
# -> writes 304_generate/stories15M.model.npz + tokenizer.model (gitignored)
#    and tests/fixtures/real_ref.npz (committed greedy-story reference)
```

Then generate a story with **your own** code:

```python
import numpy as np
import sentencepiece as spm
from leet_llm import LlamaConfig, load_llama, generate

sp = spm.SentencePieceProcessor(model_file="304_generate/tokenizer.model")
R = np.load("304_generate/tests/fixtures/real_ref.npz")
cfg = LlamaConfig(dim=int(R["dim"]), n_layers=int(R["n_layers"]), n_heads=int(R["n_heads"]),
                  n_kv_heads=int(R["n_kv_heads"]), vocab_size=int(R["vocab_size"]),
                  max_seq_len=int(R["max_seq_len"]), norm_eps=float(R["norm_eps"]),
                  rope_base=float(R["rope_base"]))
W = np.load("304_generate/stories15M.model.npz")
params = load_llama({k: W[k] for k in W.files}, cfg)

ids = np.array([[sp.bos_id()] + sp.encode("Once upon a time")])
out = generate(ids, params, cfg, max_new_tokens=64, temperature=0.0)
print(sp.decode(out))
# -> Once upon a time, there was a little girl named Lily. She loved to play outside ...
```

## Benchmark (the L4 baseline)

`generate` here is **stateless recompute** — every step reprocesses the whole prefix, so cost
is quadratic in length. That is exactly the inefficiency L4's KV-cache removes; this benchmark
captures the "before" number:

```bash
LEET_LLM_TARGET=solution uv run --group gen python 304_generate/tools/benchmark.py --limit 20
# writes benchmark_baseline.json (latency percentiles, tokens/sec) + prints sample stories
```
