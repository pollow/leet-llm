# 302 — Greedy Translation (opus-mt-en-zh)

**Level 3 · Whole-Model & Inference**

## Description

Implement the greedy decoding loop that drives the encoder-decoder Transformer you built
in task 301. Encode the source sequence once, then autoregressively sample target tokens
by taking the argmax of the last-position logits — stopping when the model emits `eos_id`
or the budget is exhausted. This is **stateless recompute**: no KV-cache; the full prefix
is reprocessed at each step (KV-caching is L4).

## The algorithm (problem spec)

```
ids = [decoder_start_id]
encode the source once → memory

repeat up to max_new_tokens times:
    tgt = array([ids])                          # shape (1, current_len)
    logits = transformer_logits(src_ids, tgt, params, cfg)   # (1, t, V)
    next_id = argmax(logits[0, -1])             # scalar int
    ids.append(next_id)
    if next_id == eos_id: break

return ids   # includes decoder_start_id; includes eos_id if produced
```

The returned list always starts with `decoder_start_id`. It ends with `eos_id` when the
model produces it within the budget; otherwise it ends with the last token before the
budget ran out.

## HF facts (GIVEN — framework plumbing)

- **Special-token ids** (tiny fixture): `decoder_start_id=63`, `eos_id=0`.
- **`MarianTokenizer`** is used only in the opt-in real-weight demo below; you do not need
  it to implement `translate`.
- The `params` and `cfg` objects come from task 301 (`load_marian`, `TransformerConfig`);
  import them via `from leet_llm import transformer_logits, TransformerConfig, load_marian`.

## Run it for real

First fetch the real `Helsinki-NLP/opus-mt-en-zh` weights (CC-BY-4.0, Helsinki-NLP /
OPUS-MT; weights are **not** committed — this is opt-in):

```bash
bash 302_translate/download.sh
```

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

## How to Test

```bash
uv run grade 302
```
