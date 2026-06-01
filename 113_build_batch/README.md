# 113 — Build Batch

**Level 1 · Tokenization & Batching**

## Description

The finish line for Level 1: assemble the exact bundle of tensors a transformer consumes.
Given a list of encoded sequences, produce a dict with `input_ids`, `pad_mask`, and
`position_ids` — all `(B, L)` and mutually consistent. This composes your work from 111 and
112; later levels will feed this dict straight into the model.

## The Math

```
build_batch(seqs) = {
    "input_ids":    pad_batch(seqs, pad_id, max_len),     # (B, L)
    "pad_mask":     padding_mask(seqs, max_len),          # (B, L), 0/1
    "position_ids": position_ids(seqs, max_len),          # (B, L)
}
```

All three share the same `(B, L)` shape, so a row's id, mask bit, and position line up.

## Function Signature

```python
def build_batch(seqs: list[list[int]], pad_id: int = 0,
                max_len: int | None = None) -> dict[str, np.ndarray]: ...
```

## Read More

- Reuse your own pieces: `from leet_llm import pad_batch, padding_mask, position_ids`.

## How to Test

```bash
uv run grade 113
```
