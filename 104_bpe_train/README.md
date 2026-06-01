# 104 — BPE Training Loop

**Level 1 · Tokenization & Batching**

## Description

Now drive the primitives from 103 in a loop to *learn* a vocabulary from a corpus. Start
with one symbol per character, then repeatedly merge the most frequent adjacent pair into
a new piece, recording the order in which merges happened. That order is the tokenizer's
"score". This is exactly how the real `stories15M` tokenizer's vocabulary was built — just
on a much larger corpus.

## The Math

- **Base vocabulary:** the sorted unique characters of the corpus, each with score `0`.
- **Each round:** count adjacent pairs (103); take the **highest count**. On a tie, pick
  the **lexicographically smallest pair** — compare the left piece string, then the right.
  Merge it everywhere (103), append the new piece (the two strings concatenated), and give
  it the next **descending** score: `-1`, `-2`, `-3`, …
- **Stop** when the vocabulary reaches `vocab_size` (or no pairs remain).

The score column *is* the merge order — higher score = merged earlier = higher priority.
Spaces are ordinary symbols, so pieces like `" ab"` (leading space) arise naturally.

## Function Signature

```python
def bpe_train(text: str, vocab_size: int) -> tuple[list[str], list[float]]:
    ...  # returns (tokens, scores), parallel lists indexed by token id
```

## Read More

- Sennrich et al. 2016 (BPE for NLP): https://arxiv.org/abs/1508.07909
- Karpathy, *Let's build the GPT Tokenizer*: https://www.youtube.com/watch?v=zduSFxRajkE

## How to Test

```bash
uv run grade 104
```
