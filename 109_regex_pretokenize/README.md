# 109 — Regex Pre-tokenizer (tiktoken-style)

**Level 1 · Tokenization & Batching**

## Description

The `stories15M` BPE merges across the whole stream. The **tiktoken** family (GPT-2/4,
Llama-3, Qwen) does something different first: it splits text into chunks with a regular
expression, and BPE merges are **never allowed to cross chunk boundaries**. This keeps
words, numbers, and punctuation from fusing together. This task builds that splitter — the
first half of loading a real tiktoken tokenizer (the encoder is 110).

> This is **load-only** machinery for future models — we don't train a tiktoken tokenizer.

## The Math

Apply the GPT-2 pattern, which matches, in priority order: common English contractions
(`'s`, `'t`, `'re`, …); an optional leading space then letters (`\p{L}+`); optional space
then digits (`\p{N}+`); optional space then a run of punctuation/symbols; and runs of
whitespace. The crucial detail: a **single leading space stays attached** to the following
word (`"Hello world"` → `["Hello", " world"]`), which is how byte-level tokenizers encode
word boundaries. Concatenating the chunks reproduces the input exactly.

## Function Signature

```python
def regex_split(text: str) -> list[str]: ...
```

Uses the third-party `regex` module (Unicode `\p{L}`/`\p{N}` classes), already a project
dependency.

## Read More

- GPT-2 encoder regex: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- tiktoken: https://github.com/openai/tiktoken

## How to Test

```bash
uv run grade 109
```
