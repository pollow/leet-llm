# 302 — Greedy Translation (opus-mt-en-zh)

**Level 3 · Whole-Model & Inference**

## Description

Implement greedy decoding for the encoder-decoder Transformer from task 301.
Starting from `decoder_start_id`, repeatedly pick the highest-probability next token
until `eos_id` or a step budget is reached. This is **stateless recompute**: no KV-cache;
the full target prefix is reprocessed each step (KV-caching is L4).

## The algorithm (problem spec)

Initialize the output with the start token, then loop:

1. Form the current target prefix as a batch of shape `(1, t)`.
2. Run the full 301 forward to get vocabulary logits.
3. Take argmax over the last position to choose the next token id.
4. Append it; stop on `eos_id` or after `max_new_tokens` steps.
5. Return the id list including start token and eos if produced.

The returned list always starts with `decoder_start_id`. It ends with `eos_id` when the
model produces it within budget; otherwise it ends at the budget limit.

## HF facts (GIVEN — framework plumbing)

- **Special-token ids** (tiny fixture): `decoder_start_id=63`, `eos_id=0`.
- **`MarianTokenizer`** is used only in the opt-in real-weight demo below; you do not need
  it to implement `translate`.
- The `params` and `cfg` objects come from task 301 (`load_marian`, `TransformerConfig`);
  import them via `from leet_llm import transformer_logits, TransformerConfig, load_marian`.

## Run it for real

First fetch the real `Helsinki-NLP/opus-mt-en-zh` weights (CC-BY-4.0, Helsinki-NLP /
OPUS-MT; weights are **not** committed — this is opt-in). This downloads only the
PyTorch weights, tokenizer and config (~315 MB), not the TF / Flax / Rust copies:

```bash
uv sync --group gen
uv run --group gen python 302_translate/convert.py
# -> writes 302_translate/opus_mt_en_zh.npz  (~300 MB)  and tests/fixtures/real_ref.npz
```

If you see `MarianTokenizer requires SentencePiece`, run `uv add --group gen sentencepiece`
or `uv pip install sentencepiece` once — it is now listed in pyproject.toml gen group.

Then translate a sentence:

```python
import numpy as np
from transformers import MarianTokenizer
from leet_llm import TransformerConfig, load_marian, translate

NAME = "Helsinki-NLP/opus-mt-en-zh"
tok = MarianTokenizer.from_pretrained(NAME)
W = np.load("302_translate/opus_mt_en_zh.npz")
R = np.load("302_translate/tests/fixtures/real_ref.npz")
cfg = TransformerConfig(
    d_model=int(R["d_model"]), n_heads=int(R["n_heads"]),
    n_enc_layers=int(R["n_enc_layers"]), n_dec_layers=int(R["n_dec_layers"]),
    d_ff=int(R["d_ff"]), vocab_size=int(R["vocab_size"]), max_pos=int(R["max_pos"]),
    scale_embedding=bool(R["scale_embedding"]), pad_id=int(R["pad_id"]),
    eos_id=int(R["eos_id"]), decoder_start_id=int(R["decoder_start_id"]))
params = load_marian({k: W[k] for k in W.files}, cfg)

prompt = "I have a dream that one day this nation will rise up."
enc = tok([prompt], return_tensors="np")
ids = translate(enc["input_ids"], params, cfg, max_new_tokens=64)
print(tok.decode(ids, skip_special_tokens=True))
```

Model weights are licensed CC-BY-4.0 by Helsinki-NLP / OPUS-MT
(https://huggingface.co/Helsinki-NLP/opus-mt-en-zh).

## Function Signature

```python
def translate(src_ids: np.ndarray, params, cfg, max_new_tokens: int = 64) -> list[int]:
    ...
#   src_ids: (1, S)
#   params: MarianParams (from leet_llm import load_marian)
#   cfg: TransformerConfig (from leet_llm import TransformerConfig)
#   returns: list of int token ids, starting with cfg.decoder_start_id
```

## Hints

* `transformer_logits` from task 301 already wraps encoder + decoder + LM head, so the loop only needs src_ids, current tgt prefix, params, cfg. No separate memory argument is exposed.
* Conceptually the encoder output depends only on src and could be computed once, but the 301 API recomputes it each step as part of stateless recompute; this is intentional for L3 simplicity and does not affect correctness. KV-cache in L4 refers to decoder self-attention history, not encoder memory.

## How to Test

```bash
uv run grade 302
```
