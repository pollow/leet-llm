# 108 — Special Tokens

**Level 1 · Tokenization & Batching**

## Description

Models need markers that aren't text: a beginning-of-sequence token (`<s>`, BOS) and an
end-of-sequence token (`</s>`, EOS). Encoders add them around the real ids; decoders strip
them back out. These live at fixed ids in the vocabulary (BOS=1, EOS=2 in the `stories15M`
tokenizer). This task is the generic plumbing, independent of which tokenizer produced the
ids.

## The Math

- `add_special_tokens(ids, bos_id, eos_id)` returns `[bos_id] + ids + [eos_id]`, including
  only the ones that are not `None`.
- `strip_special_tokens(ids, special_ids)` removes every id in `special_ids`, wherever it
  appears.

So `strip(add(ids, bos, eos), {bos, eos}) == ids`.

## Function Signature

```python
def add_special_tokens(ids: list[int], bos_id: int | None = None,
                       eos_id: int | None = None) -> list[int]: ...
def strip_special_tokens(ids: list[int], special_ids: Iterable[int]) -> list[int]: ...
```

## Read More

- Llama special tokens / chat format: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py

## How to Test

```bash
uv run grade 108
```
